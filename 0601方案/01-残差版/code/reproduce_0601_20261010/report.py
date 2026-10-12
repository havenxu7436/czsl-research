"""Summarize real completed files; incomplete jobs never get invented metrics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common import read, sha, write


def error_analysis(folder):
    labels = np.load(folder / 'labels.npz')
    candidates = read(folder / 'candidates.json')
    selected = labels['top5_indices_at_bias0'][:, 0]
    pairs = [candidates[int(i)] for i in selected]
    # Native evaluator candidate dictionaries use original string labels.
    primitive_order = read(folder.parent / 'meta/labels.json')
    attrs, objs = primitive_order['attrs'], primitive_order['objs']
    truth_a = labels['attr']
    truth_o = labels['obj']
    pred_a = np.array([attrs.index(p[0]) for p in pairs])
    pred_o = np.array([objs.index(p[1]) for p in pairs])
    object_error = pred_o != truth_o
    attr_only = (pred_o == truth_o) & (pred_a != truth_a)
    both_correct = (pred_o == truth_o) & (pred_a == truth_a)
    return {'n': len(selected), 'object_wrong_count': int(object_error.sum()),
            'object_correct_attribute_wrong_count': int(attr_only.sum()),
            'pair_correct_count': int(both_correct.sum()),
            'definition': 'bias0 argmax candidate error partition, not best-HM prediction or causal diagnosis'}


def summarize(work):
    work = Path(work)
    protocol = read(work / 'protocol.json')
    rows, states, errors = [], [], {}
    for job in protocol['jobs']:
        run = Path(job['run'])
        state_file = run / 'status.json'
        state = read(state_file) if state_file.exists() else {'state': 'not_started'}
        states.append({'variant': job['variant'], 'tier': job['tier'], **state})
        if state.get('state') != 'complete':
            continue
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                folder = run / ('eval_' + phase + '_' + setting)
                meta = read(folder / 'metrics.json')
                metrics = meta['metrics_raw']
                required = ('AUC', 'best_hm', 'best_seen', 'best_unseen')
                values = [float(metrics[k]) for k in required]
                if not all(np.isfinite(v) for v in values):
                    raise ValueError('Nonfinite stored metrics ' + job['variant'])
                rows.append({'variant': job['variant'], 'tier': job['tier'], 'phase': phase, 'setting': setting,
                             'best_epoch': meta['best_epoch'], 'checkpoint_sha256': meta['checkpoint_sha256'],
                             **{k + '_pct': 100 * float(metrics[k]) for k in required}})
                errors[job['variant'] + '_' + phase + '_' + setting] = error_analysis(folder)
    core_done = all(s['state'] == 'complete' for s in states if s['tier'] == 0)
    all_done = all(s['state'] == 'complete' for s in states)
    dest = work / 'summary'
    dest.mkdir(exist_ok=True)
    write(dest / 'summary.json', {'core_complete': core_done, 'all_planned_complete': all_done,
                                  'states': states, 'rows': rows, 'error_partition_bias0': errors,
                                  'protocol_sha256': sha(work / 'protocol.json'), 'not_covered': protocol['not_covered']})
    fields = ['variant', 'tier', 'phase', 'setting', 'best_epoch', 'AUC_pct', 'best_hm_pct', 'best_seen_pct', 'best_unseen_pct', 'checkpoint_sha256']
    with (dest / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    lines = ['# 0601 复现：实际训练与结果汇总', '',
             '范围：UT-Zappos，seed 3351，10 轮原生 CPF 联合训练。所有 A–H 重训；按 VAL CW AUC 选模，同一权重评价 VAL/TEST CW/OW。', '',
             '核心 A–H 完成：`{}`；全部排定任务完成：`{}`。负增益不影响完成判定。'.format(core_done, all_done), '',
             '## 实际执行状态', '', '| 组 | 优先级 | 状态 |', '|---|---:|---|']
    for state in states:
        lines.append('| {} | {} | {} |'.format(state['variant'], state['tier'], state['state']))
    lines += ['', '## 主表（百分数）', '', '| 组 | 划分 | 设置 | 最佳轮 | AUC | HM | best_seen | best_unseen |', '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
            row['variant'], row['phase'], row['setting'], row['best_epoch'], *[row[k + '_pct'] for k in ('AUC', 'best_hm', 'best_seen', 'best_unseen')]))
    lines += ['', '## 原方案关键差分（百分点）', '', '| 比较 | 划分 | 设置 | ΔAUC | ΔHM |', '|---|---|---|---:|---:|']
    lookup = {(r['variant'], r['phase'], r['setting']): r for r in rows}
    for left, right in [('H', 'A'), ('H', 'B'), ('H', 'F'), ('G', 'E'), ('H', 'G'), ('B', 'A'), ('F', 'E'),
                        ('H', 'H_no_pair'), ('H', 'H_shuffle'), ('H', 'H_random_prior'), ('H', 'H_uniform_prior')]:
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                x, y = lookup.get((left, phase, setting)), lookup.get((right, phase, setting))
                if x and y:
                    lines.append('| {}−{} | {} | {} | {:+.4f} | {:+.4f} |'.format(left, right, phase, setting,
                                 x['AUC_pct'] - y['AUC_pct'], x['best_hm_pct'] - y['best_hm_pct']))
    lines += ['', '## 实现与解读边界', '',
              '- G/H：§6.1 四个 300→512 投影＋pair MLP；使用 CPF 共享对象/属性视觉向量，不复制 B×N×M×D。保留原 composition 分数，再加 factorized 知识分数。',
              '- G/H 使用 GT-object 条件属性 CE，H 属性逻辑输入来自实际 TRAIN 候选组合分数边缘化；B/F 保留已审计的原语属性适配。因此 H−F 包含 head/条件 CE/适配变化，不能称唯一变量仅为一个 MLP。',
              '- loss=composition CE＋0.4 object CE＋0.6 attribute CE＋β logic；β=0.01 只乘一次。ViT 与 CPF 原有可训练模块共同更新，不是冻结缓存小头。',
              '- 保留 legacy 的训练温度/推理余弦尺度差异；prior λ=0.1 的相对强度也因此不同。不宣称分数是已严格归一化的联合概率。',
              '- 主 G/H 没有 pair-level LLM 文本，因而是 template-only 实验；没有无中生有的 LLM/template 对比。',
              '- 常数 prior centered 后全零；从头训练对照与固定 checkpoint 关闭 prior 的检查是两种不同证据。随机 prior centered 后匹配零均值/单位总体标准差。',
              '- Shuffled descriptions 保留原语名称、OI、pair 模板和 prior，打乱 Text K 与 OE/AE 逻辑描述的归属。',
              '- 噪声扩展仅删除 Text/Feature primitive 描述；没有修改 OI、逻辑描述、prior 或模板。',
              '- 上述 error partition 是 bias0 的错误分类，不证明先验或知识造成错误；best_seen/best_unseen 不一定属于同一 bias 工作点。',
              '- 描述、逻辑 bank 都是重建资产，不是原作者遗失数据；这是设计指南工程复现，不是论文正式结果复现。', '',
              '## 未覆盖的项目', ''] + ['- ' + x for x in protocol['not_covered']]
    uniform_check = work / 'checks/uniform_readout.json'
    if uniform_check.exists():
        lines += ['', '## 固定权重 prior 检查', '', '```json', json.dumps(read(uniform_check), ensure_ascii=False, indent=2), '```']
    lines += ['', '各 run 保留实际配置、训练日志、smoke、输入/源码/权重哈希与评价目录。报告只计完整执行的结果；中断/失败需检查对应日志。', '']
    (dest / '0601_实际训练与汇总.md').write_text('\n'.join(lines), encoding='utf-8')
    return core_done, all_done


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    summarize(p.parse_args().work)
