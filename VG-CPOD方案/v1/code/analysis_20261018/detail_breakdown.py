"""Break down conditional attribute transfer and the largest material loss."""
import csv
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_results import RUNS, OUT, TABLES, FIGURES, SEEDS, VARIANTS, read, csv_write


def main():
    manifest = read(RUNS / 'cache_ut_clip_l14/features_manifest.json')
    labels = manifest['labels']
    train = set(map(tuple, manifest['pairs']['train']))
    rows, confusion, arrays = [], [], {}
    for v in VARIANTS:
        for seed in SEEDS:
            with np.load(RUNS / f'main/ut-zap50k/{v}/seed{seed}/predictions/test_ow.npz', allow_pickle=False) as z:
                a, o = z['attribute_truth'], z['object_truth']
                pred = z['conditional_logits_gt_object'].argmax(1)
            seen = np.array([(labels['attrs'][i], labels['objs'][j]) in train for i, j in zip(a, o)])
            arrays[v, seed] = pred
            for attr in sorted(set(a.tolist())):
                for group, group_mask in [('all', np.ones(len(a), dtype=bool)), ('seen', seen), ('unseen', ~seen)]:
                    mask = (a == attr) & group_mask
                    if not mask.any():
                        continue
                    rows.append(dict(variant=v, seed=seed, attribute=labels['attrs'][attr], group=group,
                                     images=int(mask.sum()), correct=int((pred[mask] == a[mask]).sum()),
                                     accuracy=float((pred[mask] == a[mask]).mean() * 100)))
                    if labels['attrs'][attr] == 'Suede':
                        for k, n in sorted(Counter(pred[mask].tolist()).items()):
                            confusion.append(dict(variant=v, seed=seed, true_attribute='Suede', group=group,
                                                  predicted_attribute=labels['attrs'][k], images=n,
                                                  percentage=n / int(mask.sum()) * 100))
    csv_write('attribute_seen_unseen.csv', rows)
    csv_write('suede_confusion.csv', confusion)
    pairs = list(csv.DictReader((TABLES / 'per_pair_metrics.csv').open(encoding='utf-8-sig')))
    changes = []
    for (attr, obj) in sorted({(r['attribute'], r['object']) for r in pairs if r['setting'] == 'test_ow'}):
        subset = [r for r in pairs if r['setting'] == 'test_ow' and r['attribute'] == attr and r['object'] == obj]
        row = dict(attribute=attr, object=obj, seen=subset[0]['seen'], images=int(subset[0]['images']))
        for v in ('V0', 'V1', 'V1L', 'V2', 'V5'):
            vr = [r for r in subset if r['variant'] == v]
            row[v + '_gtobj_attr_acc'] = float(np.mean([float(r['gt_object_attr_acc']) for r in vr]))
            row[v + '_cw_pair_acc_calibrated'] = float(np.mean([float(r['pair_acc_calibrated']) for r in pairs
                if r['setting'] == 'test_cw' and r['attribute'] == attr and r['object'] == obj and r['variant'] == v]))
        row['V5_minus_V0_cond'] = row['V5_gtobj_attr_acc'] - row['V0_gtobj_attr_acc']
        row['contribution_total_cond_points'] = row['images'] / len(a) * row['V5_minus_V0_cond']
        changes.append(row)
    changes.sort(key=lambda r: r['contribution_total_cond_points'])
    csv_write('test_pair_contributions.csv', changes)
    contributions = []
    for attr in sorted({r['attribute'] for r in rows}):
        base = [r for r in rows if r['attribute'] == attr and r['variant'] == 'V0' and r['group'] == 'all']
        full = [r for r in rows if r['attribute'] == attr and r['variant'] == 'V5' and r['group'] == 'all']
        delta = float(np.mean([r['accuracy'] for r in full]) - np.mean([r['accuracy'] for r in base]))
        contributions.append(dict(attribute=attr, images=base[0]['images'], delta_accuracy=delta,
                                  contribution_total_cond_points=base[0]['images'] / len(a) * delta))
    contributions.sort(key=lambda r: r['contribution_total_cond_points'])
    csv_write('attribute_contributions.csv', contributions)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 140})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    diag = list(csv.DictReader((TABLES / 'prediction_diagnostics.csv').open(encoding='utf-8-sig')))
    variants = ['V0', 'V1', 'V1L', 'V2', 'V5']
    for offset, group, color in [(-.18, 'seen', '#236b8e'), (.18, 'unseen', '#b94a57')]:
        grid = [[float(r['gt_object_attr_acc_' + group]) for r in diag
                 if r['group'] == 'main' and r['setting'] == 'test_ow' and r['variant'] == v] for v in variants]
        axs[0].bar(np.arange(len(variants)) + offset, [np.mean(g) for g in grid], width=.34,
                   yerr=[np.std(g, ddof=1) for g in grid], capsize=3, color=color, label=group.capitalize())
    axs[0].set_xticks(range(len(variants)), variants)
    axs[0].set_ylabel('GT-object attribute accuracy (%)')
    axs[0].set_title('Conditional attribute transfer')
    axs[0].legend()
    axs[1].barh([r['attribute'] for r in contributions], [r['contribution_total_cond_points'] for r in contributions],
                color=['#b94a57' if r['contribution_total_cond_points'] < 0 else '#236b8e' for r in contributions])
    axs[1].axvline(0, color='#333', lw=1)
    axs[1].set_xlabel('Contribution to total accuracy change (points)')
    axs[1].set_title('V5 - V0: weighted material contributions')
    fig.tight_layout()
    fig.savefig(FIGURES / '07_attribute_generalization.png')
    plt.close(fig)
    summary = {'attribute_contributions': contributions, 'pair_contributions': changes,
               'suede_confusion': []}
    for v in ('V0', 'V1', 'V1L', 'V2', 'V5'):
        for group in ('all', 'seen', 'unseen'):
            support = next(r['images'] for r in rows if r['variant'] == v and r['attribute'] == 'Suede' and r['group'] == group)
            for name in labels['attrs']:
                values = [sum(r['images'] for r in confusion if r['variant'] == v and r['seed'] == seed
                              and r['group'] == group and r['predicted_attribute'] == name) for seed in SEEDS]
                if any(values):
                    summary['suede_confusion'].append(dict(variant=v, group=group, predicted_attribute=name,
                        support=support, mean_images=float(np.mean(values)), percentage=float(np.mean(values) / support * 100)))
    (OUT / 'detail_data.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # Rebuild the OW figure with a concise y-axis label that fits the canvas.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for setting, label, color in [('test_cw', 'CW', '#236b8e'), ('test_ow', 'OW', '#b94a57')]:
        values = [np.mean([float(r['max_unseen_accuracy']) for r in diag
                          if r['group'] == 'main' and r['variant'] == v and r['setting'] == setting]) for v in VARIANTS]
        axs[0].plot(VARIANTS, values, marker='o', label=label, color=color)
    axs[0].set_ylabel('Unseen-only ranking accuracy (%)')
    axs[0].set_title('Positive-bias limit: only unseen candidates compete')
    axs[0].legend()
    values = [np.mean([float(r['unseen_limit_extra_candidate_pct']) for r in diag
                      if r['group'] == 'main' and r['variant'] == v and r['setting'] == 'test_ow']) for v in VARIANTS]
    axs[1].bar(VARIANTS, values, color='#7d78aa')
    axs[1].set_ylabel('Extra-candidate win rate (%)')
    axs[1].set_title('Where the extra 91 OW candidates win')
    for ax in axs:
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES / '06_open_world_competition.png')
    plt.close(fig)
    print('DETAIL_READY: 5 CSVs, 2 figures, detail_data.json')


if __name__ == '__main__':
    main()
