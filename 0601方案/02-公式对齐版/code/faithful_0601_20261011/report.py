"""Per-seed and paired summaries of actual, completed runs only."""
import argparse
import csv
import math
import re
from pathlib import Path
import numpy as np
from common import read, sha, write


def stats(values):
    return {'n': len(values), 'mean': float(np.mean(values)),
            'sample_std': float(np.std(values, ddof=1)) if len(values) > 1 else None}


def summarize(work):
    work = Path(work)
    p = read(work / 'protocol.json')
    states, rows, curves, warnings = [], [], {}, []
    for j in p['jobs']:
        run = Path(j['run'])
        state = read(run / 'status.json') if (run / 'status.json').exists() else {'state': 'not_started'}
        states.append({'variant': j['variant'], 'base_variant': j['base_variant'], 'seed': j['seed'], 'tier': j['tier'], **state})
        if state['state'] != 'complete':
            continue
        if state.get('epochs') != p['epochs']:
            raise ValueError('Completed job budget differs from protocol')
        curve = []
        text = (run / 'train.log').read_text(encoding='utf-8', errors='replace')
        for epoch, auc, hm in re.findall(r'Val/Test Epoch (\d+): AUC=([\d.eE+-]+) \| best_hm=([\d.eE+-]+)', text):
            curve.append({'epoch': int(epoch), 'val_cw_auc_pct': 100 * float(auc), 'val_cw_hm_pct': 100 * float(hm)})
        curves[j['variant']] = curve
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                m = read(run / ('eval_' + phase + '_' + setting) / 'metrics.json')
                values = {key + '_pct': 100 * float(m['metrics_raw'][key])
                          for key in ('AUC', 'best_hm', 'best_seen', 'best_unseen')}
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError('Nonfinite metric')
                rows.append({'variant': j['variant'], 'base_variant': j['base_variant'], 'seed': j['seed'],
                             'phase': phase, 'setting': setting, 'best_epoch': m['best_epoch'],
                             'checkpoint_sha256': m['checkpoint_sha256'], **values})
                if phase == 'val' and setting == 'cw' and m['best_epoch'] >= p['epochs'] - 2:
                    warnings.append(j['variant'] + ': best VAL checkpoint in final three epochs; budget adequacy needs review, not proof of convergence')
    core = all(s['state'] == 'complete' for s in states if s['tier'] == 0)
    done = all(s['state'] == 'complete' for s in states)
    aggregates = []
    for base in sorted({j['base_variant'] for j in p['jobs']}):
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                group = [r for r in rows if r['base_variant'] == base and r['phase'] == phase and r['setting'] == setting]
                if group:
                    aggregates.append({'variant': base, 'phase': phase, 'setting': setting,
                                       'seeds': [r['seed'] for r in group],
                                       'AUC_pct': stats([r['AUC_pct'] for r in group]),
                                       'HM_pct': stats([r['best_hm_pct'] for r in group])})
    lookup = {(r['base_variant'], r['seed'], r['phase'], r['setting']): r for r in rows}
    pairs = [('H', 'A'), ('H', 'B'), ('H', 'F'), ('G', 'E'), ('H', 'G'), ('B', 'A'),
             ('E', 'A'), ('F', 'E'), ('D', 'J0'), ('E', 'Jtext'), ('J0', 'A'), ('Jtext', 'C'),
             ('H', 'H_no_pair'), ('H', 'H_shuffle'), ('H', 'H_random_prior'), ('H', 'H_uniform_prior')]
    differences = []
    for left, right in pairs:
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                shared, auc, hm = [], [], []
                for seed in p['seeds']:
                    x, y = lookup.get((left, seed, phase, setting)), lookup.get((right, seed, phase, setting))
                    if x and y:
                        shared.append(seed)
                        auc.append(x['AUC_pct'] - y['AUC_pct'])
                        hm.append(x['best_hm_pct'] - y['best_hm_pct'])
                if shared:
                    differences.append({'comparison': left + '-' + right, 'phase': phase, 'setting': setting,
                                        'paired_seeds': shared, 'AUC_pp': stats(auc), 'HM_pp': stats(hm),
                                        'AUC_positive_seeds': sum(x > 0 for x in auc)})
    dest = work / 'summary'
    dest.mkdir(exist_ok=True)
    write(dest / 'summary.json', {'core_complete': core, 'all_planned_complete': done, 'states': states,
                                  'rows': rows, 'aggregates': aggregates, 'paired_differences': differences,
                                  'val_curves': curves, 'budget_review_flags': warnings,
                                  'protocol_sha256': sha(work / 'protocol.json'), 'not_covered': p['not_covered']})
    fields = ['variant', 'base_variant', 'seed', 'phase', 'setting', 'best_epoch', 'AUC_pct',
              'best_hm_pct', 'best_seen_pct', 'best_unseen_pct', 'checkpoint_sha256']
    with (dest / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 0601 原评分公式版：实际训练与汇总', '',
             'UT-Zappos；每组 {} 轮；冻结 seeds={}。所有结果来自新训练，不拼旧残差版。'.format(p['epochs'], p['seeds']), '',
             '核心及接口对照完整：{}；全计划完整：{}。完成不要求涨点；一 seed 完成不冒充三 seed 完成。'.format(core, done), '',
             '## 实际状态', '', '| 组 | seed | 状态 |', '|---|---:|---|']
    lines += ['| {} | {} | {} |'.format(s['base_variant'], s['seed'], s['state']) for s in states]
    lines += ['', '## 单 seed 主表，百分数', '', '| 组 | seed | 划分 | 设置 | 最佳轮 | AUC | HM |', '|---|---:|---|---|---:|---:|---:|']
    lines += ['| {} | {} | {} | {} | {} | {:.4f} | {:.4f} |'.format(r['base_variant'], r['seed'], r['phase'], r['setting'], r['best_epoch'], r['AUC_pct'], r['best_hm_pct']) for r in rows]
    def display(s):
        return '{:.4f} ± {:.4f}'.format(s['mean'], s['sample_std']) if s['sample_std'] is not None else '{:.4f}（单 seed，无标准差）'.format(s['mean'])
    lines += ['', '## 已完成 seed 的均值与样本标准差', '', '| 组 | 划分 | 设置 | n | AUC | HM |', '|---|---|---|---:|---|---|']
    lines += ['| {} | {} | {} | {} | {} | {} |'.format(r['variant'], r['phase'], r['setting'], len(r['seeds']), display(r['AUC_pct']), display(r['HM_pct'])) for r in aggregates]
    lines += ['', '## 配对差分，百分点', '', '| 比较 | 划分 | 设置 | 配对 n | ΔAUC | ΔHM |', '|---|---|---|---:|---|---|']
    lines += ['| {} | {} | {} | {} | {} | {} |'.format(r['comparison'], r['phase'], r['setting'], len(r['paired_seeds']), display(r['AUC_pp']), display(r['HM_pp'])) for r in differences]
    lines += ['', '## 公式与接口边界', '']
    for key in ('formula', 'temperature', 'mvp_interface', 'native_baseline', 'logic_adapter', 'description_interface',
                'logic_operators', 'head', 'knowledge', 'controls', 'training_budget', 'augmentation', 'test_policy'):
        lines.append('- {}：{}'.format(key, p[key]))
    lines += ['', '原文 §9 评分与 §8.2 loss 接法已落实，但接口选择不能冒充原文唯一规定；本表不是原作者知识资产/论文成绩复现。', '',
              '## 训练预算复核', ''] + (['- ' + w for w in warnings] if warnings else ['- 暂无末尾最佳 checkpoint 标记；这不自动证明收敛。查看 summary.json 中 VAL 曲线。'])
    lines += ['', '若需调整训练预算或参数，先形成新版本并统一比较组，不按 TEST 选择，不将不同预算拼表。', '',
              '## 尚未覆盖', ''] + ['- ' + x for x in p['not_covered']]
    (dest / '0601_原公式版训练汇总.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return core, done


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    summarize(p.parse_args().work)
