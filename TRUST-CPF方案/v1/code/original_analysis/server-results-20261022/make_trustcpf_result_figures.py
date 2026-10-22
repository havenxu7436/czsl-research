"""Create the publication tables and figures from archived derived metrics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text(encoding='utf-8'))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    variants = ['r0', 'capacity', 'dense', 'drop', 'free', 'mean', 'robust']
    colors = ['#a4aab0', '#265f9d', '#7098c0', '#419582', '#78b7a8', '#cb9344', '#b54b49']
    exported = []

    def save(fig, stem):
        for ext in ['png', 'svg']:
            p = figures / (stem + '.' + ext)
            fig.savefig(p, dpi=240, bbox_inches='tight')
            exported.append(p.name)
        plt.close(fig)

    def write_csv(name, rows):
        with (out / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    rows = []
    for v in variants:
        for i, seed in enumerate([3351, 3352, 3353]):
            for world in ['cw', 'ow']:
                rows.append({'variant': v, 'seed': seed, 'world': world,
                             'test_auc_native': data['main'][v][world]['AUC_native']['values'][i],
                             'test_auc_exact': data['main'][v][world]['AUC_exact']['values'][i],
                             'val_auc_native': data['errors'][v]['val'][world]['val_auc']['values'][i]})
    write_csv('main_3seed.csv', rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout='constrained')
    for ax, world in zip(axes, ['cw', 'ow']):
        stats = [data['main'][v][world]['AUC_native'] for v in variants]
        x = np.arange(len(variants))
        ax.bar(x, [s['mean'] for s in stats], yerr=[s['sd'] for s in stats],
               color=colors, capsize=4, alpha=.9)
        for j, s in enumerate(stats):
            ax.scatter(j + np.array([-.12, 0, .12]), s['values'], color='#263441', s=14, zorder=3)
        ax.set(xticks=x, xticklabels=variants, ylabel='TEST AUC (points)', title=world.upper(), ylim=(0, 17))
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('Main experiment: three training seeds; error bars = sample SD')
    save(fig, '01_main_auc')

    rows = []
    for v, worlds in data['paired'].items():
        for world, stats in worlds.items():
            for seed, value in zip([3351, 3352, 3353], stats['values']):
                rows.append({'comparison': 'robust-' + v, 'world': world, 'seed': seed, 'auc_delta': value})
    write_csv('paired_deltas.csv', rows)

    rows = []
    for seed, audit in data['audit'].items():
        for model in ['all', 'type_constant', 'edge_lookup']:
            rows.append({'seed': seed, 'predictor': model, **audit[model]})
    write_csv('utility_audit.csv', rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout='constrained')
    xs = np.arange(3)
    for j, (model, label, color) in enumerate([('all', 'Utility model', '#b54b49'),
                                            ('type_constant', 'Type constant', '#265f9d')]):
        axes[0].bar(xs + (j - .5) * .34,
                    [data['audit'][s][model]['rmse'] for s in ['3351', '3352', '3353']],
                    .32, color=color, label=label)
        axes[1].bar(xs + (j - .5) * .34,
                    [100 * data['audit'][s][model]['direction_balanced_accuracy'] for s in ['3351', '3352', '3353']],
                    .32, color=color, label=label)
    axes[0].set(ylabel='Q5 RMSE (utility units)', ylim=(0, .02))
    axes[1].set(ylabel='Balanced direction accuracy (%)', ylim=(0, 65))
    axes[1].axhline(50, color='#666', linestyle='--', linewidth=1)
    for ax in axes:
        ax.set(xticks=xs, xticklabels=['3351', '3352', '3353'], xlabel='Training seed')
        ax.grid(axis='y', alpha=.18)
    axes[0].legend(frameon=False)
    fig.suptitle('Independent utility audit: 3,072 Q5 observations per seed')
    save(fig, '02_utility_audit')

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout='constrained')
    for ax, kind, title in zip(axes, ['source_mismatch', 'deletion'], ['Source mismatch', 'Relation deletion']):
        for v, color in zip(['dense', 'drop', 'mean', 'robust'], ['#7098c0', '#419582', '#cb9344', '#b54b49']):
            item = data['recovered_ow_rauc'][v + '/' + kind]
            curve = item['curve']
            rates = [0., .1, .2, .3, .5]
            ax.plot(rates, [curve[str(r)]['mean'] for r in rates], '-o', color=color, label=v, markersize=4)
            rows.append({'variant': v, 'kind': kind, 'training_seed': 3351,
                         'curve_count': 15, 'rauc': item['all']['mean'],
                         'clean_auc': curve['0.0']['mean'], 'rate_0_5_auc': curve['0.5']['mean']})
        capacity = data['main']['capacity']['ow']['AUC_native']['values'][0]
        ax.axhline(capacity, color='#265f9d', linestyle='--', linewidth=1.3,
                   label='capacity: code-implied reference')
        ax.set(title=title, xlabel='Nominal rate within one relation type', ylabel='OW AUC (points)',
               xlim=(-.015, .515), ylim=(7.3, 12.4))
        ax.grid(alpha=.18)
    axes[1].legend(frameon=False, fontsize=8, loc='center right')
    fig.suptitle('600 OW noise evaluations; training seed3351; means over 5 perturbation seeds x 3 types')
    save(fig, '03_noise_ow_curves')
    write_csv('noise_rauc.csv', rows)

    fig, ax = plt.subplots(figsize=(9.7, 4), layout='constrained')
    vv = ['capacity', 'dense', 'drop', 'mean', 'robust']
    x = np.arange(len(vv))
    ax.bar(x - .17, [data['errors'][v]['test']['cw']['object_accuracy']['mean'] for v in vv],
           .32, color='#265f9d', label='Object component of joint top-1')
    ax.bar(x + .17, [data['errors'][v]['test']['cw']['gt_object_attr_accuracy']['mean'] for v in vv],
           .32, color='#cb9344', label='Attribute accuracy given GT object')
    ax.set(xticks=x, xticklabels=vv, ylabel='Accuracy (%)', ylim=(0, 52),
           title='TEST error decomposition: means over three training seeds')
    ax.grid(axis='y', alpha=.18)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    save(fig, '04_error_decomposition')

    rows = []
    for key, value in data['same_q5_subset_comparison'].items():
        rows.append({'predictor': key, **value['held_subset']})
    write_csv('same_q5_heldout_subset.csv', rows)
    (out / 'figure_metadata.json').write_text(json.dumps({
        'source': 'analysis.json', 'training_seeds': [3351, 3352, 3353],
        'noise_training_seed': 3351, 'noise_recovered_ow_records': 600,
        'capacity_noise_line': 'code-implied reference; not additional noise evaluations',
        'files': exported}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'figures': len(exported), 'output': str(out)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
