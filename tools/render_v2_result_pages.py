"""Format archived v2 data as GitHub tables; never analyze or alter raw data."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).absolute().parents[1]
STAGE = '0601方案/v2'
BATCH = STAGE + '/results/seed3351-10epoch-20261015'
RAW = BATCH + '/raw_export'
RECEIPT = 'provenance/updates/20261015_v2_result_pages.json'
NAMING = BATCH + '/实验命名对照.json'
SETTINGS = [('val', 'cw', 'VAL CW'), ('val', 'ow', 'VAL OW'),
            ('test', 'cw', 'TEST CW'), ('test', 'ow', 'TEST OW')]


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def digest(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def write_json(name, value):
    # Match the repository's existing metadata writers and native newline style.
    # Export members themselves are read-only and never pass through this writer.
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |',
            '|---|' + '---:|' * (len(headers)-1)] + [
                '| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]


def source_data():
    summary = read(RAW+'/summary/summary.json')
    protocol = read(RAW+'/protocol.json')
    jobs = [j for j in protocol['jobs'] if j['seed'] == 3351]
    rows = {(r['variant'], r['phase'], r['setting']): r for r in summary['rows']}
    if len(jobs) != 32 or len(rows) != len(summary['rows']) or len(rows) != 128:
        raise ValueError('Expected 32 runs and 128 unique evaluations')
    for job in jobs:
        epochs = set()
        for phase, setting, _ in SETTINGS:
            row = rows[job['variant'], phase, setting]
            metric = read(RAW+'/runs/'+job['variant']+'/eval_'+phase+'_'+setting+'/metrics.json')
            for key, raw_key in [('AUC_pct', 'AUC'), ('best_hm_pct', 'best_hm'),
                                 ('best_seen_pct', 'best_seen'), ('best_unseen_pct', 'best_unseen')]:
                if abs(row[key]-100*metric['metrics_raw'][raw_key]) > 1e-8:
                    raise ValueError('Summary differs from original metric: '+job['variant']+'/'+key)
            if row['checkpoint_sha256'] != metric['checkpoint_sha256'] or row['best_epoch'] != metric['best_epoch']:
                raise ValueError('Checkpoint differs')
            epochs.add(row['best_epoch'])
        if len(epochs) != 1:
            raise ValueError('Best epoch differs between settings')
        curve = summary['val_curves'][job['variant']]
        if [v['epoch'] for v in curve] != list(range(1, 11)):
            raise ValueError('Expected ten original epoch records')
        log = (ROOT/RAW/'runs'/job['variant']/'train.log').read_text(encoding='utf-8')
        readings = [(int(epoch), 100*float(value)) for epoch, value in
                    re.findall(r'Val/Test Epoch (\d+): AUC=([0-9.eE+-]+)', log)]
        if len(readings) != 10 or any(epoch != row['epoch'] or abs(value-row['VAL_CW_AUC_pct']) > 1e-8
                                      for (epoch, value), row in zip(readings, curve)):
            raise ValueError('Epoch readings differ from original training log')
    return jobs, rows, summary['val_curves']


def read_labels(required_names=None):
    naming = read(NAMING)
    entries = naming['variants']
    labels = {entry['base_variant']: entry for entry in entries}
    if len(entries) != len(labels) or len(labels) != 32:
        raise ValueError('Expected one naming record for each of the 32 variants')
    if required_names is not None and set(labels) != set(required_names):
        raise ValueError('Display names do not cover exactly the completed matrix')
    options = read(naming['variant_options'])
    source = naming['original_design']
    original = subprocess.check_output(['git', 'show', source['git_ref']+':'+source['git_path']], cwd=ROOT)
    design_lines = original.decode('utf-8-sig').splitlines()
    original_roles = set()
    for variant, entry in labels.items():
        label = entry['comparison_name']
        if any(c in label for c in ('|', '\n', '[', ']', '（', '）')):
            raise ValueError('Unsafe table comparison name: '+variant)
        for key, value in entry['expected_options'].items():
            if options[variant][key] != value:
                raise ValueError('Comparison name contradicts actual option: '+variant+'/'+key)
        letter = entry.get('original_design_variant')
        if letter:
            matches = [line for line in design_lines if line.startswith('| '+letter+'. ')]
            if len(matches) != 1:
                raise ValueError('Original core role is not unique: '+letter)
            fields = [field.strip() for field in matches[0].strip('|').split('|')]
            if label != fields[0]:
                raise ValueError('Core comparison name is not the original design spelling: '+variant)
            flags = tuple(field == '✅' for field in fields[1:5])
            if flags != tuple(options[variant][key] for key in ('text', 'prior', 'feature', 'logic')):
                raise ValueError('Module roles contradict original design: '+variant)
            original_roles.add(letter)
    if original_roles != set('ABCDEFGH'):
        raise ValueError('Original eight core design roles must all be covered')
    return labels, hashlib.sha256(original).hexdigest()


def group_link(job, labels, target=None):
    name = job['base_variant']+'（'+labels[job['base_variant']]['comparison_name']+'）'
    return '['+name+']('+(target or 'raw_export/runs/'+job['variant']+'/')+')'


def render():
    jobs, rows, curves = source_data()
    labels, _ = read_labels(j['base_variant'] for j in jobs)
    page = [
        '# 统一修订版 v2：seed 3351 结果', '',
        '记录日期：2026-10-15。UT-Zappos，32组各10 epoch；按首次严格最大VAL CW AUC选模，',
        '同一checkpoint评价VAL/TEST CW/OW，共128份评价。', '',
        '本页沿用02公式对齐版的“组、最佳轮次、VAL CW、VAL OW、TEST CW、TEST OW”表头。只展示原始记录，不做差分、排名、跨seed统计或结果分析。', '',
        '## 训练设置', '',
        '- seed：3351；ViT B16、GloVe 300d；batch=64；每组实际10 epoch。',
        '- 版本：20261014修正版；与旧02公式对齐版r2的30/10/10 epoch结果分别归档。',
        '- 组别和顺序沿用原protocol的32组，不按成绩排序；点击组名可查看原始训练与评价文件。',
        '- 本次仅包含seed3351；seed3352、3353未纳入此快照，不以缺失记录填0。', '',
        '命名格式为“实际组名（原设计对照名）”；括号只对照研究角色或项目，不表示修订组与旧公式模型等同。原设计没有的独立对照注明“新增”；[32组命名对照及实现边界](实验命名对照.md)单独列出，原始run_id不变。', '',
        '## 选中 checkpoint 的指标', '',
        'AUC单位为百分比，显示4位小数；每行的四项指标来自同一选中checkpoint。最佳轮次直接读取归档记录，不重新选模。', '',
    ]
    auc = []
    for job in jobs:
        values = [rows[job['variant'], phase, setting] for phase, setting, _ in SETTINGS]
        auc.append([group_link(job, labels), values[0]['best_epoch']] + [f"{v['AUC_pct']:.4f}" for v in values])
    page += table(['组', '最佳轮次', 'VAL CW', 'VAL OW', 'TEST CW', 'TEST OW'], auc)
    page += ['', '## 训练记录与完整指标', '',
             '- [逐轮训练记录](训练记录.md)：32组各10轮VAL CW AUC原始日志读数，不计算增益或趋势。',
             '- [完整评价指标](完整评价指标.md)：128份评价的AUC、best HM、best seen、best unseen及原metrics入口。',
             '- [原始指标汇总](raw_export/summary/summary.json)：保留未舍入精度及全部原始字段。', '',
             '## 校验与文件', '',
             '1766个导出成员完整归档；32组训练均记录1–10轮完成；128份评价的checkpoint身份一致性已核验。原始score、标签、日志与配置字节保持不变。', '',
             '- [训练与评价文件](raw_export/runs/)：train.log、smoke、训练完成记录、checkpoint身份、四类评价和原始分数。',
             '- [实际协议](raw_export/protocol.json)与[实际配置](../../configs/seed3351-10epoch-20261015/configs/)。',
             '- [导出文件映射](export_member_map.json)、[原始文件SHA256](raw_export/export_manifest.json)与[已完成的核验记录](audit.json)。',
             '- [源码映射](source_links.json)关联20261014修正版；知识与缓存通过[inputs.json](../../inputs.json)索引。',
             '- [归档与校验说明](归档与校验.md)：完整文件分区、原ZIP身份和恢复命令。', '',
             '本次仅进行网页展示整理，未从score重算指标，也未执行后分析。模型checkpoint权重和数据集图像原本不在下载包中。', '']

    detail = ['# 统一修订版 v2：seed 3351 完整评价指标', '', '[返回结果首页](README.md) · [逐轮训练记录](训练记录.md) · [命名对照](实验命名对照.md)', '',
              '32组 × VAL/TEST × CW/OW = 128份评价。数值直接读取原始汇总，单位均为百分比，显示4位小数；点击组名查看对应metrics.json。', '',
              '主名称后的括号为原设计角色或项目对照，不是旧模型的等价别名；“新增”表示原设计无独立同名组。', '',
              '四类评价沿用同一个VAL CW选中checkpoint，不分别选模。best seen、best unseen为原导出的同名字段，不能当作best HM对应同一bias下的两项准确率。', '']
    for phase, setting, label in SETTINGS:
        detail += ['## '+label, '']
        data = []
        for job in jobs:
            row = rows[job['variant'], phase, setting]
            link = group_link(job, labels, 'raw_export/runs/'+job['variant']+'/eval_'+phase+'_'+setting+'/metrics.json')
            data.append([link, row['best_epoch']] + [f'{row[key]:.4f}' for key in ['AUC_pct', 'best_hm_pct', 'best_seen_pct', 'best_unseen_pct']])
        detail += table(['组', '最佳轮次', 'AUC', 'best HM', 'best seen', 'best unseen'], data) + ['']

    training = ['# 统一修订版 v2：seed 3351 逐轮训练记录', '', '[返回结果首页](README.md) · [完整评价指标](完整评价指标.md) · [命名对照](实验命名对照.md)', '',
                '名称沿用结果首页的“实际组名（原设计对照名）”；括号只对照设计角色，原始run_id和组序不变。', '',
                '仅列出32组各10轮VAL CW AUC原始日志读数，单位为百分比，显示2位小数。原日志按0–1单位打印4位小数，因此精度不同于选中checkpoint的原metrics.json。', '',
                '数据来自原summary.json的val_curves，组序沿用protocol；不重新选模、不计算轮次增益或趋势。点击组名查看train.log。为便于网页阅读，将10轮拆成两张表。', '']
    for start, stop in [(1, 5), (6, 10)]:
        training += [f'## 第{start}–{stop}轮', '']
        values = []
        for job in jobs:
            link = group_link(job, labels, 'raw_export/runs/'+job['variant']+'/train.log')
            by_epoch = {r['epoch']: r for r in curves[job['variant']]}
            values.append([link]+[f"{by_epoch[epoch]['VAL_CW_AUC_pct']:.2f}" for epoch in range(start, stop+1)])
        training += table(['组']+[f'Epoch {epoch}' for epoch in range(start, stop+1)], values) + ['']
    naming = read(NAMING)
    references = ['# 统一修订版 v2：实验命名对照', '', '[返回结果首页](README.md) · [完整评价指标](完整评价指标.md) · [逐轮训练记录](训练记录.md)', '',
                  '依据[原实验设计]('+naming['original_design']['url']+')和[v2设计说明](../../复现公式与代码v2.md)。', '',
                  naming['policy'], '',
                  'A–H名称按原设计§10.1原文列出，模块角色与修正版variant_options逐项核对；不是依据成绩或代码拼写猜测。其他项目按原设计相应章节对应；括号含“新增”的组来自v2补充对照。', '',
                  'Rfull还承担原设计的Template-only、每类3条描述、树深3层和mean聚合共同基准；不另外虚构重复训练组。No-pair、知识noise等名称按实际移除范围注明，不能据名称推断已删除所有组合知识或注入了错误事实。', '',
                  '| 组（原设计对照名） | 原设计位置 | 实际实现与边界 |', '|---|---|---|']
    references += ['| '+group_link(job, labels)+' | '+labels[job['base_variant']]['source_section']+' | '+labels[job['base_variant']]['implementation_note']+' |' for job in jobs]
    references += ['', '机器可读映射与各项配置核对条件见[实验命名对照.json](实验命名对照.json)。本页只整理实验定义，不分析数值、差分或优劣。', '']
    return {BATCH+'/README.md': '\n'.join(page), BATCH+'/完整评价指标.md': '\n'.join(detail),
            BATCH+'/训练记录.md': '\n'.join(training), BATCH+'/实验命名对照.md': '\n'.join(references)}


def verify_pages(pages):
    for name, content in pages.items():
        if (ROOT/name).read_bytes() != content.encode('utf-8'):
            raise ValueError('Formatted page is stale: '+name)
        for target in re.findall(r'\]\(([^)\n]+)\)', content):
            if re.match(r'^[a-z]+://', target):
                continue
            if not ((ROOT/name).parent/target.split('#')[0]).exists():
                raise ValueError('Broken result page link: '+target)
    print(json.dumps({'passed': True, 'pages': len(pages), 'auc_rows': 32,
                      'detailed_metric_rows': 128, 'original_epoch_readings': 320,
                      'design_name_mappings': 32, 'original_core_roles_checked': 8,
                      'raw_metrics_consistent': True, 'raw_bytes_modified': False,
                      'analysis_performed': False}, ensure_ascii=False))


def register(pages):
    inputs = read(STAGE+'/inputs.json')
    names = sorted(set(pages) | {BATCH+'/归档与校验.md'})
    inputs['data_files'] = sorted(set(inputs['data_files']) | set(names) | {NAMING})
    inputs['result_batches'][0]['web_pages'] = {'summary': BATCH+'/README.md',
             'all_metrics': BATCH+'/完整评价指标.md', 'epoch_records': BATCH+'/训练记录.md',
             'archive_notes': BATCH+'/归档与校验.md', 'design_names': BATCH+'/实验命名对照.md'}
    write_json(STAGE+'/inputs.json', inputs)
    stages = read('provenance/stages.json')
    for stage in stages:
        if stage['id'] == '19':
            stage['data_files'] = len(inputs['data_files'])
    write_json('provenance/stages.json', stages)
    source_hashes = {name: digest(name) for name in [RAW+'/summary/summary.json', RAW+'/protocol.json']}
    _, original_design_hash = read_labels()
    write_json(RECEIPT, {'date': '2026-10-15', 'stage': 'v2', 'archive_stage': '19',
              'operation': 'Format archived records as GitHub Markdown tables matching 02-公式对齐版',
              'source_files_sha256': source_hashes, 'completed_seeds': [3351],
              'displayed_runs': 32, 'displayed_evaluations': 128, 'displayed_epoch_readings': 320,
              'group_order': 'Original protocol order; not sorted by metrics',
              'naming': {'mapping': NAMING, 'mapping_sha256': digest(NAMING),
                   'original_design': read(NAMING)['original_design'], 'original_design_sha256': original_design_hash,
                   'variant_options_sha256': digest(read(NAMING)['variant_options']),
                   'display_format': 'base_variant（original design comparison name）',
                   'design_roles_not_implementation_aliases': True, 'all_32_names_checked_against_actual_options': True,
                   'original_core_role_names_verified': 8},
              'analysis_performed': False, 'metric_recomputation_performed': False,
              'raw_export_bytes_modified': False, 'frozen_source_or_configs_modified': False,
              'presentation_checks': {'auc_table_rows': 32, 'detailed_metric_rows': 128,
                   'epoch_readings_checked_against_train_logs': 320,
                   'metrics_checked_against_raw_json': True, 'relative_links_valid': True,
                   'number_format': 'metrics: percent with 4 decimals; log epoch readings: percent with 2 decimals'},
              'server_training_started': False, 'reference_format': '0601方案/02-公式对齐版/results/seed3351-30epoch-20261012/README.md',
              'pages': names})
    selected = set(names) | {NAMING, STAGE+'/README.md', STAGE+'/results/README.md', STAGE+'/inputs.json',
               'provenance/stages.json', RECEIPT, 'tools/render_v2_result_pages.py'}
    manifest = {r['path']: r for r in read('provenance/current_manifest.json')}
    for name in sorted(selected):
        manifest[name] = {'path': name, 'sha256': digest(name), 'bytes': (ROOT/name).stat().st_size}
    write_json('provenance/current_manifest.json', [manifest[n] for n in sorted(manifest)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read-only check that tables and links match original records')
    parser.add_argument('--register', action='store_true', help='Refresh presentation metadata after formatting')
    args = parser.parse_args()
    pages = render()
    if not args.check:
        for name, content in pages.items():
            (ROOT/name).write_bytes(content.encode('utf-8'))
    verify_pages(pages)
    if args.register:
        register(pages)
