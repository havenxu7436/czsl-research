"""Plot audited pilot trajectories and record cross-version checkpoint precision."""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / 'conditional_attribute_pilot_20261009'))
import numpy as np
import torch
from common import read_json, tensor_load, write_json
from head import VARIANTS
from runtime import fresh_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    args = parser.parse_args()
    folder = args.results.resolve()
    pilot = folder / 'extracted/pilot'
    audit = read_json(folder / 'independent_audit.json')
    if not audit['passed'] or audit['independent_metric_checks'] != 660:
        raise ValueError('Need the independent complete pilot audit')
    ref = read_json(folder / 'extracted/release/reference.json')
    bank = torch.load(args.bank, map_location='cpu', weights_only=True)
    archived_initial = tensor_load(pilot / 'runs/P_K/best_auc.pt')['head_state']
    initial = fresh_head(bank, 'P_K', ref['control_protocol'], 'cpu').state_dict()
    precision = dict(local_torch=str(torch.__version__), server_torch=read_json(pilot / 'environment.json')['torch'],
                     local_regenerated_initial_max_abs_difference={k: float((v-archived_initial[k]).abs().max()) for k, v in initial.items()},
                     archived_four_common_initial_states_bitwise_equal=all(
                         torch.equal(v, tensor_load(pilot / 'runs' / variant / 'best_auc.pt')['head_state'][k])
                         for variant in ('P_name', 'P_perm1', 'P_perm2') for k, v in archived_initial.items()),
                     independent_score_metric_replay_uses_saved_server_scores=True,
                     cross_version_generated_initial_sha_is_not_used_as_a_success_criterion=True)
    write_json(folder / 'checkpoint_precision_audit.json', precision)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(P_name='#777777', P_K='#c44e52', P_generic='#0072b2', P_perm1='#009e73', P_perm2='#cc79a7')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    histories = {v: read_json(pilot / 'runs' / v / 'history.json')['ledger'] for v in VARIANTS}
    for variant, history in histories.items():
        epoch = [x['epoch'] for x in history]
        color = colors[variant]
        axes[0, 0].plot(epoch, [100*x['metrics_raw']['cw']['AUC'] for x in history], color=color, marker='o', markersize=3, label=variant)
        axes[0, 1].plot(epoch, [100*x['metrics_raw']['ow']['AUC'] for x in history], color=color, marker='o', markersize=3)
        axes[1, 0].plot(epoch[1:], [x['train_epoch']['mean_conditional_nll'] for x in history[1:]], color=color, marker='o', markersize=3)
        axes[1, 1].plot(epoch, [100*x['conditional_diagnostics']['by_seen_pair']['unseen_pair']['attr_accuracy'] for x in history],
                        color=color, marker='o', markersize=3)
    for ax, field, title in ((axes[0, 0], 'cw', 'VAL closed-world AUC'), (axes[0, 1], 'ow', 'VAL open-world AUC')):
        baseline = 100*histories['P_K'][0]['metrics_raw'][field]['AUC']
        ax.axhline(baseline, color='black', linestyle='--', linewidth=1, alpha=.55)
        ax.set_title(title)
        ax.set_ylabel('AUC (%)')
    axes[1, 0].set_title('Online TRAIN conditional NLL (epoch averages)')
    axes[1, 0].set_ylabel('NLL')
    axes[1, 1].set_title('VAL unseen-pair attribute accuracy, given GT object')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].axhline(100*histories['P_K'][0]['conditional_diagnostics']['by_seen_pair']['unseen_pair']['attr_accuracy'],
                      color='black', linestyle='--', linewidth=1, alpha=.55)
    for ax in axes.flat:
        ax.set_xlabel('Epoch (0 = frozen baseline)')
        ax.set_xticks(np.arange(11))
        ax.grid(alpha=.2)
    axes[0, 0].legend(loc='lower left', fontsize=9)
    axes[0, 0].scatter([7], [100*histories['P_generic'][7]['metrics_raw']['cw']['AUC']], marker='*', s=145,
                       color=colors['P_generic'], edgecolors='black', zorder=5)
    fig.suptitle('UT-Zappos seed 0: complete fixed-budget conditional-head pilot', fontsize=14)
    fig.savefig(folder / 'pilot_trajectories.png', dpi=180)
    fig.savefig(folder / 'pilot_trajectories.svg')
    plt.close(fig)
    # Compact, unrounded machine-readable tables for the review.
    baseline = histories['P_K'][0]
    rows = []
    for variant, history in histories.items():
        selected = max(history, key=lambda x: x['metrics_raw']['cw']['AUC'])
        trained = max(history[1:], key=lambda x: x['metrics_raw']['cw']['AUC'])
        rows.append(dict(variant=variant, selected_epoch=selected['epoch'], selected_metrics=selected['metrics_raw'],
                         selected_cw_delta_pp=100*(selected['metrics_raw']['cw']['AUC']-baseline['metrics_raw']['cw']['AUC']),
                         selected_ow_delta_pp=100*(selected['metrics_raw']['ow']['AUC']-baseline['metrics_raw']['ow']['AUC']),
                         best_trained_cw_epoch=trained['epoch'], best_trained_cw=100*trained['metrics_raw']['cw']['AUC'],
                         same_epoch_ow=100*trained['metrics_raw']['ow']['AUC'], final_epoch_metrics=history[-1]['metrics_raw'],
                         epoch_1_diagnostics=history[1]['conditional_diagnostics'], epoch_10_diagnostics=history[-1]['conditional_diagnostics']))
    write_json(folder / 'review_tables.json', dict(rows=rows, baseline=baseline['metrics_raw'],
                                                  selected_epoch_uses_cw_only=True, original_budget_gate=read_json(pilot / 'summary.json')['decision']))
    print('PILOT_REVIEW_FIGURE_AND_TABLES_SAVED', folder)


if __name__ == '__main__':
    main()
