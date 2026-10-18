"""Lossless import, verification and restoration of VG-CPOD v1 server results."""
import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'VG-CPOD方案/v1'
BATCH = STAGE + '/results/ut-zap50k-3seeds-20261018'
MAP = BATCH + '/export_member_map.json'
RECEIPT = 'provenance/updates/20261018_vgcpod_v1_server_results.json'
ARCHIVE = 'VG-CPOD_v1_results_20261018.tar.gz'
EXPECTED_SHA = 'd448d5c0d7b67204ba52ddb11d81dde424348f9ad0e77881438137cb59bd14d1'
REPORT = 'VG-CPOD_v1_实验结果分析_20261018'
SEEDS = (3351, 3352, 3353)
VARIANTS = ('V0', 'V1', 'V1L', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_bytes() if path.exists() else b''
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if b'\r\n' in old:
        payload = payload.replace(b'\n', b'\r\n')
    path.write_bytes(payload)


def text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding='utf-8')


def safe(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('Unsafe relative path: ' + repr(name))
    return p


def relative(target, parent):
    return os.path.relpath(ROOT / target, ROOT / parent).replace('\\', '/')


def github_math(value):
    opened = False
    lines = []
    for line in value.splitlines():
        if line == '$$':
            lines.append('```' if opened else '```math')
            opened = not opened
        else:
            lines.append(line)
    if opened:
        raise ValueError('Unclosed display math')
    value = '\n'.join(lines) + '\n'
    return re.sub(r'\\operatorname\{([^}]+)\}', lambda m: r'\mathop{\mathrm{' + m[1] + r'}}\nolimits ', value)


def import_data(source):
    source = Path(source).resolve()
    if (ROOT / MAP).exists():
        raise FileExistsError('Batch already exists; use --check')
    data = source / 'server_results_20261018'
    archive = data / ARCHIVE
    assert sha(archive) == EXPECTED_SHA
    assert (data / (ARCHIVE + '.sha256')).read_text().split()[0] == EXPECTED_SHA
    release = read(ROOT / STAGE / 'results/release_member_map.json')['members']
    source_files = [p for p in data.rglob('*') if p.is_file() and p.suffix != '.pyc'
                    and p.name not in (ARCHIVE, ARCHIVE + '.sha256')]
    by_hash = {}
    for p in source_files:
        by_hash.setdefault(sha(p), []).append(p)
    records = []
    created = set()

    def store(src, dest, runtime, kind, metadata=None):
        safe(dest)
        safe(runtime)
        target = ROOT / dest
        digest = sha(src)
        if target.exists():
            if sha(target) != digest:
                raise ValueError('Preserve existing file with different bytes: ' + dest)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            created.add(dest)
        assert sha(target) == digest
        row = dict(original_path=runtime, runtime_path=runtime, current_path=dest,
                   sha256=digest, bytes=target.stat().st_size, stage='20', kind=kind)
        if metadata:
            row.update(metadata)
        records.append(row)

    names = set()
    aliases = []
    with tarfile.open(archive, encoding='utf-8', errors='surrogateescape') as t:
        for member in t:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError('Expected regular evidence file: ' + repr(member.name))
            raw_name = member.name.encode('utf-8', 'surrogateescape')
            try:
                name = raw_name.decode('utf-8')
            except UnicodeDecodeError:
                name = raw_name.decode('gb18030')
            safe(name)
            assert name not in names
            names.add(name)
            h = hashlib.sha256()
            with t.extractfile(member) as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    h.update(block)
            digest = h.hexdigest()
            src = data / name
            if not src.is_file() or sha(src) != digest:
                candidates = by_hash.get(digest, [])
                if len(candidates) != 1:
                    raise ValueError('Cannot identify extracted member: ' + name)
                src = candidates[0]
                aliases.append(dict(runtime_name=name, extracted_name=src.relative_to(data).as_posix()))
            assert src.stat().st_size == member.size
            if name.startswith('vgcpod_project/vgcpod_v1/'):
                rel = name[len('vgcpod_project/vgcpod_v1/'):]
                if rel in release and sha(ROOT / release[rel]) == digest:
                    dest = release[rel]
                elif PurePosixPath(rel).suffix in ('.py', '.sh'):
                    dest = STAGE + '/code/server-20261018/vgcpod_v1/' + rel
                elif rel.startswith(('assets/', 'configs/')):
                    dest = STAGE + '/configs/server-20261018/vgcpod_v1/' + rel
                elif rel.endswith('.md'):
                    dest = BATCH + '/package_support/server/' + rel + '.txt'
                else:
                    dest = BATCH + '/raw_export/' + name
            elif name.startswith('vgcpod_v1_runs/'):
                dest = BATCH + '/raw_export/' + name
            else:
                raise ValueError('Unexpected archive root: ' + name)
            store(src, dest, 'server_results_20261018/' + name, 'vgcpod_v1_server_export_member',
                  dict(archive_name_bytes_base64=base64.b64encode(raw_name).decode('ascii'),
                       archive_name_encoding='utf-8' if raw_name.decode('utf-8', 'replace') == name else 'gb18030'))
    assert len(names) == len(source_files) == 579
    for name in (ARCHIVE, ARCHIVE + '.sha256'):
        store(data / name, BATCH + '/archives/' + name, 'server_results_20261018/' + name,
              'vgcpod_v1_original_archive')
    for src in sorted((source / 'analysis_20261018').rglob('*')):
        if not src.is_file() or any(p in ('__pycache__', 'mpl_cache') for p in src.parts):
            continue
        rel = src.relative_to(source / 'analysis_20261018').as_posix()
        if src.suffix in ('.py', '.js'):
            dest = STAGE + '/code/analysis_20261018/' + rel
        elif rel.endswith('.md'):
            dest = BATCH + '/package_support/analysis/' + rel + '.txt'
        else:
            dest = BATCH + '/analysis/' + rel
        store(src, dest, 'analysis_20261018/' + rel, 'vgcpod_v1_analysis_original')
    for name in (REPORT + '.md', REPORT + '.html', 'VG-CPOD公式及代码v1.md'):
        if name == 'VG-CPOD公式及代码v1.md':
            dest = STAGE + '/results/package_support/desktop/' + name + '.txt'
        else:
            dest = BATCH + '/package_support/analysis/' + name + ('.txt' if name.endswith('.md') else '')
        store(source / name, dest, name, 'vgcpod_v1_analysis_original')
    write(ROOT / MAP, dict(date='2026-10-18', archive_sha256=EXPECTED_SHA,
          archive_members=579, extracted_filename_aliases=aliases, records=records,
          member_bytes_modified=False, filename_recovery='Invalid UTF-8 Chinese archive names decoded as GB18030; original bytes retained in base64.'))
    layout = read(ROOT / 'provenance/layout.json')
    layout['files'].extend(records)
    write(ROOT / 'provenance/layout.json', layout)
    publish_report()
    generate_pages()
    inventory = []
    runroot = ROOT / BATCH / 'raw_export/vgcpod_v1_runs'
    for p in sorted(runroot.rglob('protocol.json')):
        if not (p.parent / 'metrics.json').exists():
            continue
        protocol = read(p)
        metrics = read(p.parent / 'metrics.json')
        inventory.append(dict(run=p.parent.relative_to(runroot).as_posix(),
            protocol_path=p.relative_to(ROOT).as_posix(), protocol_sha256=sha(p),
            seed=protocol['seed'], config=protocol['config'],
            selected_epoch=metrics.get('selected_epoch'), source_sha256=protocol['source_sha256']))
    assert len(inventory) == 35
    write(ROOT / STAGE / 'configs/server-20261018/experiment_inventory.json', inventory)
    write(ROOT / BATCH / 'upload_scope.json', dict(date='2026-10-18', server_export_members=579,
        original_members_including_analysis=len(records), main_runs=27, sensitivity_runs=6,
        baseline_runs=2, evaluations=140, checkpoints=66, main_figures=7, analysis_qa_images=5,
        server_diagnostic_images=3, original_archive_included=True, bytecode_and_matplotlib_cache_included=False,
        omitted_image_feature_arrays=read(ROOT / BATCH / 'analysis/audit.json')['omitted_feature_arrays'],
        original_bytes_preserved=True, raw_results_not_reselected=True,
        original_name_bytes_index=MAP, large_files='Git LFS via existing .gitattributes'))
    write(ROOT / RECEIPT, dict(date='2026-10-18', line='VG-CPOD方案', stage='v1', archive_stage='20',
        source_archive_sha256=EXPECTED_SHA, main_runs=27, sensitivity_runs=6, total_trained_runs=33,
        evaluations=140, completed_seeds=list(SEEDS), original_archive_members=579,
        records=len(records), new_payload_files=len(created), checkpoints_included=True,
        figures_and_analysis_included=True, training_started_by_import=False,
        mapping=MAP, result_batch=BATCH, byte_identity_preserved=True))
    inputs = read(ROOT / STAGE / 'inputs.json')
    inputs.update(status='main_3seeds_and_sensitivity_complete', formal_training_results_included=True,
                  completed_runs=27, planned_runs=27, sensitivity_runs=6, total_trained_runs=33,
                  baselines=2, evaluations=140, completed_seeds=list(SEEDS), all_planned_complete=True,
                  result_batches=[dict(path=BATCH, main_runs=27, sensitivity_runs=6, baseline_runs=2,
                                       epochs=10, evaluations=140, source_archive_sha256=EXPECTED_SHA)])
    write(ROOT / STAGE / 'inputs.json', inputs)
    print(json.dumps(dict(imported=True, archive_members=579, original_records=len(records), new_files=len(created)), ensure_ascii=False))


def publish_report():
    original = ROOT / BATCH / 'package_support/analysis'
    report = (original / (REPORT + '.md.txt')).read_text(encoding='utf-8').replace('analysis_20261018/', '')
    html = (original / (REPORT + '.html')).read_text(encoding='utf-8').replace('analysis_20261018/', '')
    for name in ('analyze_results.py', 'detail_breakdown.py', 'build_report.py'):
        link = relative(STAGE + '/code/analysis_20261018/' + name, BATCH + '/analysis')
        report = report.replace('](' + name + ')', '](' + link + ')')
        html = html.replace('href="' + name + '"', 'href="' + link + '"')
    text(ROOT / BATCH / 'analysis/分析报告.md', github_math(report))
    text(ROOT / BATCH / 'analysis/分析报告.html', html)


def generate_pages():
    base = ROOT / BATCH
    metrics = read(base / 'analysis/analysis_data.json')
    runs = base / 'raw_export/vgcpod_v1_runs'
    header = '''# UT-Zappos · VG-CPOD v1 · 三 seed 正式结果

2026-10-18：主矩阵 27 组、敏感性 6 组均完成 10 epochs；另有两组冻结基线，共 140 份评价与 66 份 best/last 权重。

**[完整结果分析与七张主图](analysis/分析报告.md)** · [HTML阅读版](analysis/分析报告.html) · [分析复现](analysis/README.md)

本页列出原始选中权重的读数，分析见上方独立报告。选模沿用首次严格最大 VAL CW AUC_native；同一权重评价 VAL/TEST CW/OW。HM 使用对应 VAL 的偏置。

## 三 seed 汇总

| 模型 | VAL CW AUC_native | TEST CW AUC_native | TEST OW AUC_native | TEST CW HM | TEST OW HM |
|---|---:|---:|---:|---:|---:|
'''
    def ms(v, setting, key):
        r = next(r for r in metrics['main_aggregate'] if r['variant'] == v and r['setting'] == setting)
        return f"{r[key + '_mean']:.4f} ± {r[key + '_sd']:.4f}"
    for v in VARIANTS:
        values = [ms(v, s, k) for s, k in [('val_cw','AUC_native'),('test_cw','AUC_native'),
                   ('test_ow','AUC_native'),('test_cw','HM_at_val_bias'),('test_ow','HM_at_val_bias')]]
        header += '| ' + v + ' | ' + ' | '.join(values) + ' |\n'
    header += '''
均值 ± 三 seed 样本标准差；AUC 单位为面积 × 100。完整精度见 [CSV](analysis/tables/main_mean_sd.csv)。

## 逐运行原始读数

| 运行 | seed | 最优 epoch | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---:|---:|---:|---:|---:|---:|
'''
    folders = [runs / 'main/ut-zap50k' / v / ('seed'+str(s)) for v in VARIANTS for s in SEEDS]
    folders += sorted((runs / 'sensitivity').glob('*/ut-zap50k/*/seed*'))
    folders += [runs / 'main/ut-zap50k' / v for v in ('name_clip','chain_zeroshot')]
    for folder in folders:
        m, p = read(folder / 'metrics.json'), read(folder / 'protocol.json')
        rel = folder.relative_to(runs).as_posix()
        vals = [f"{m['metrics'][s]['AUC_native']:.4f}" for s in ('val_cw','val_ow','test_cw','test_ow')]
        header += '| ['+rel+'](raw_export/vgcpod_v1_runs/'+rel+'/) | '+str(p['seed'])+' | '+str(m.get('selected_epoch','—'))+' | '+' | '.join(vals)+' |\n'
    header += '''
## 文件布局与恢复

- [raw_export/vgcpod_v1_runs/](raw_export/vgcpod_v1_runs/)：全部原始运行数据、best/last 权重、预测、曲线、缓存元数据、诊断图与服务器日志。
- [原始 tar.gz](archives/VG-CPOD_v1_results_20261018.tar.gz)及[SHA256](archives/VG-CPOD_v1_results_20261018.tar.gz.sha256)：原下载包完整保留，使用 Git LFS。
- [分析目录](analysis/)：七张主图、五张排版检查图、12 张 CSV 表、JSON 诊断与报告。
- [完整成员映射](export_member_map.json)：579 个导出成员与原始分析文件逐项映射到仓库路径，并记录 SHA256、字节数和恢复路径。
- [原始分析文档](package_support/analysis/)：保留本地 Markdown/HTML 原字节；发布报告只调整仓库相对链接及等价的 GitHub 数学表示。
- [上传范围](upload_scope.json)。

服务器源代码映射到版本的 `code/`，配置映射到 `configs/`；与已有冻结包相同的成员直接引用既有文件。新增[三卡 worker 脚本](../../code/server-20261018/vgcpod_v1/scripts/)单独保留。

在仓库根目录执行：

```bash
git lfs pull
python tools/import_vgcpod_results_20261018.py --check
python tools/import_vgcpod_results_20261018.py --restore /srv/vgcpod-results-20261018
```

恢复命令逐项复制并校验原始成员，在新目录中还原 `server_results_20261018/`、`analysis_20261018/` 和原报告。原 tar 中三个中文文件名使用 GB18030 字节；成员内容保持不变，恢复时采用正确的中文文件名，原文件名字节也记录在映射表中。

备份没有包含九个图像特征数组及数据集原图，服务器已有缓存可用于新的前向诊断。现有预测和权重已完整归档，可以复核本轮结果。
'''
    text(base / 'README.md', header)
    analysis_readme = '''# 分析文件与复现

**[完整分析报告](分析报告.md)** · [HTML阅读版](分析报告.html) · [原始结果](../README.md)

主图位于 `figures/`，逐 seed 指标和类别细分位于 `tables/`，排版检查位于 `qa/`。原始科学数据与分析表均保持字节身份；发布报告已改为仓库相对链接和 GitHub 数学块。

## 从仓库复现

先拉取 LFS 并恢复到新的外部目录：

```bash
git lfs pull
python tools/import_vgcpod_results_20261018.py --check
python tools/import_vgcpod_results_20261018.py --restore /srv/vgcpod-results-20261018
cd /srv/vgcpod-results-20261018
python analysis_20261018/analyze_results.py
python analysis_20261018/detail_breakdown.py
python analysis_20261018/build_report.py
```

分析环境：Python 3.12.14、NumPy 2.3.5、PyTorch 2.5.1 CPU、Matplotlib 3.10.7。服务器训练环境仍以原始 protocol.json 为准。分析脚本只读取已有预测与权重，不进行训练；恢复后的原始文档还包含当次机器路径，复算命令使用当前环境的 `python` 即可。

三个 Python 脚本与页面检查脚本分别保存在版本 `code/analysis_20261018/`，恢复命令会把它们放回原来的相对位置。页面检查脚本记录本机的 Playwright 安装路径，原始检查截图已一并保存。

配对统计先在同 seed 内相减，随后取均值和样本标准差；敏感性仅 seed3351。`AUC_native` 为原有限扫描，`AUC_exact` 为完整临界偏置扫描；固定偏置 HM 使用 VAL 偏置。`partner_effective_count` 为 exp(平均熵)，`gamma_variance_selected_epoch` 为日志中的池化方差。
'''
    text(base / 'analysis/README.md', analysis_readme)


def check():
    mapping = read(ROOT / MAP)
    records = mapping['records']
    assert mapping['archive_members'] == 579
    assert len([r for r in records if r['kind'] == 'vgcpod_v1_server_export_member']) == 579
    assert len({r['runtime_path'] for r in records}) == len(records)
    for r in records:
        assert sha(ROOT / r['current_path']) == r['sha256'], r['current_path']
        assert (ROOT / r['current_path']).stat().st_size == r['bytes']
    archive = ROOT / BATCH / 'archives' / ARCHIVE
    assert sha(archive) == EXPECTED_SHA
    runs = ROOT / BATCH / 'raw_export/vgcpod_v1_runs'
    models = list((runs / 'main/ut-zap50k').glob('V*/seed*'))
    models += list((runs / 'sensitivity').glob('*/ut-zap50k/*/seed*'))
    assert len(models) == 33
    for model in models:
        h, m = read(model / 'history.json'), read(model / 'metrics.json')
        assert [v['epoch'] for v in h] == list(range(1,11)) and h[-1]['step'] == 3600
        best = max(range(len(h)), key=lambda i: h[i]['val_cw']['AUC_native']) + 1
        assert m['selected_epoch'] == best
        assert sha(model / 'checkpoints/best.pt') == m['checkpoint_sha256']
    assert len(list(runs.rglob('predictions/*.npz'))) == 140
    assert len(list(runs.rglob('checkpoints/*.pt'))) == 66
    assert len(list((ROOT / BATCH / 'analysis/figures').glob('*.png'))) == 7
    broken, syntax = [], []
    for p in (ROOT / BATCH).rglob('*.md'):
        content = p.read_text(encoding='utf-8')
        assert content.count('```') % 2 == 0, p
        for link in re.findall(r'\]\(([^)\n]+)\)', content):
            if not re.match(r'^[a-z]+://', link) and not (p.parent / link.split('#')[0]).exists():
                broken.append((p.relative_to(ROOT).as_posix(), link))
    for p in (ROOT / STAGE / 'code/analysis_20261018').glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'))
    assert not broken, broken
    return dict(passed=True, original_file_hashes_verified=len(records), archive_members=579,
                main_runs=27, sensitivity_runs=6, evaluations=140, checkpoints=66,
                main_figures=7, broken_links=broken, python_syntax_errors=syntax,
                weights_deserialized_by_import=False, training_started_by_import=False)


def restore(destination):
    check()
    destination = Path(destination).expanduser().resolve()
    if destination.exists() or not destination.is_absolute() or destination == ROOT or ROOT in destination.parents:
        raise FileExistsError('Choose a new directory outside the repository')
    records = read(ROOT / MAP)['records']
    for r in records:
        p = destination / safe(r['runtime_path'])
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / r['current_path'], p)
        assert sha(p) == r['sha256']
    write(destination / 'restoration_receipt.json', dict(passed=True, files=len(records), source_archive_sha256=EXPECTED_SHA))
    return dict(passed=True, restored_files=len(records), destination=str(destination))


def finalize():
    publish_report()
    generate_pages()
    owned = set()
    inputs = read(ROOT / STAGE / 'inputs.json')
    for r in read(ROOT / MAP)['records']:
        owned.add(r['current_path'])
    owned.update(p.relative_to(ROOT).as_posix() for p in (ROOT / BATCH).rglob('*') if p.is_file())
    owned.add(STAGE + '/configs/server-20261018/experiment_inventory.json')
    for kind, marker in [('code_files','/code/'), ('configs','/configs/'), ('data_files','/results/')]:
        inputs[kind] = sorted(set(inputs[kind]) | {p for p in owned if marker in p})
    inputs['documents'] = sorted(set(inputs['documents']) | {p for p in owned if p.endswith('.md')})
    write(ROOT / STAGE / 'inputs.json', inputs)
    stages = read(ROOT / 'provenance/stages.json')
    for stage in stages:
        if stage['id'] == '20':
            stage.update(status='三seed主矩阵27组与敏感性6组完成；140份评价、权重、预测与分析已归档',
                completed_runs=27, sensitivity_runs=6, total_trained_runs=33, evaluations=140,
                completed_seeds=list(SEEDS), formal_training_results_included=True,
                configs=len(inputs['configs']), data_files=len(inputs['data_files']), code_files=len(inputs['code_files']))
    write(ROOT / 'provenance/stages.json', stages)
    document = BATCH + '/analysis/分析报告.md'
    source = BATCH + '/package_support/analysis/' + REPORT + '.md.txt'
    document_record = 'provenance/documents/vgcpod_v1_results_analysis_20261018.json'
    write(ROOT / document_record, dict(import_date='2026-10-18', version='v1', line='VG-CPOD方案',
          document=document, source_document=source, source_sha256=sha(ROOT/source),
          published_sha256=sha(ROOT/document), display_formulas=3, formula_semantics_unchanged=True,
          changes='Repository-relative links; GitHub math fences; equivalent mathop/roman/nolimits operators.'))
    owned.update(('README.md', 'VG-CPOD方案/README.md', STAGE+'/README.md', STAGE+'/results/README.md',
                  STAGE+'/inputs.json', STAGE+'/runbook.json', 'tools/register_vgcpod_v1.py',
                  'tools/import_vgcpod_results_20261018.py', 'tools/check_vgcpod_results_math.cjs', 'provenance/layout.json',
                  'provenance/stages.json', RECEIPT, document_record))
    current = {r['path']:r for r in read(ROOT / 'provenance/current_manifest.json')}
    for rel in sorted(owned):
        p = ROOT / rel
        current[rel] = dict(path=rel, sha256=sha(p), bytes=p.stat().st_size)
    write(ROOT / 'provenance/current_manifest.json', [current[p] for p in sorted(current)])
    return check()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--import-data', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--restore', type=Path)
    args = parser.parse_args()
    if args.check:
        result = check()
    elif args.finalize:
        result = finalize()
    elif args.restore:
        result = restore(args.restore)
    elif args.import_data and args.source:
        import_data(args.source)
        result = check()
    else:
        parser.error('Use --source PATH --import-data, --check, --finalize or --restore PATH')
    print(json.dumps(result, ensure_ascii=False))
