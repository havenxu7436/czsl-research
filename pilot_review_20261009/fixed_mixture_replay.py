"""Development-only replay of all archived epochs using one fixed mixture readout.

No fitting, alpha search, TEST inference, or modification of the original pilot.
The readout was chosen after inspecting pilot VAL; this is not preregistration.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from correction_diagnostics import ALPHA, arithmetic_mix, evaluate
from common import read_json, sha, write_json
from head import TEMPERATURE, VARIANTS
from runtime import FIELDS, make_dataset
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol_path = args.out.with_name('fixed_mixture_replay_protocol.json')
    if args.out.exists() or protocol_path.exists():
        raise ValueError('Preserve existing replay outputs')
    folder = args.results / 'extracted'
    ref = read_json(folder / 'release/reference.json')
    if not read_json(args.results / 'independent_audit.json')['passed']:
        raise ValueError('Original audit required')
    if sha(args.evaluator) != ref['c_files'][ref['c_source'] + '/evaluator_ge.py']:
        raise ValueError('Native evaluator identity changed')
    split = {}
    for phase in ('train', 'val', 'test'):
        path = args.split / (phase + '_pairs.txt')
        key = '/data/xhw/czsl_repro/data/ut-zap50k/compositional-split-natural/' + path.name
        if sha(path) != ref['c_files'][key]:
            raise ValueError('Split identity changed')
        split[phase] = [tuple(line.split()) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    datasets = {setting: make_dataset(ref, setting, split) for setting in ('cw', 'ow')}
    spec = importlib.util.spec_from_file_location('fixed_mixture_native_evaluator', args.evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base_path = folder / 'pilot/runs/P_K/epochs/epoch_0000/attempt_0000/grid_scores.npy'
    base = torch.from_numpy(np.load(base_path, allow_pickle=False).copy())
    with np.load(folder / 'pilot/val_truth.npz', allow_pickle=False) as truth:
        attrs = torch.from_numpy(truth['attrs'].copy())
        objects = torch.from_numpy(truth['objects'].copy())
    baseline = read_json(folder / 'pilot/baseline.json')['metrics_raw']
    replay_base = evaluate(base, attrs, objects, datasets, module.Evaluator)
    if any(replay_base[s][k] != baseline[s][k] for s in datasets for k in FIELDS):
        raise ValueError('Baseline native replay differs')
    old_summary = read_json(folder / 'pilot/summary.json')
    protocol = dict(schema='fixed_mixture_replay_development_protocol_v1',
                    alpha=ALPHA, alpha_search=False, temperature=TEMPERATURE,
                    mixture='Arithmetic conditional-probability mixture; frozen baseline object marginal',
                    source_protocol_sha256=old_summary['protocol_sha256'],
                    prior_diagnostic_sha256=sha(args.results / 'correction_diagnostics_fixed_quarter.json'),
                    script_sha256=sha(Path(__file__)),
                    source_score_evidence='All five original head identities, epochs 0..10',
                    selection='First strict maximum VAL CW AUC over epochs 0..10; report OW at same epoch',
                    development_only=True, not_preregistered=True,
                    chosen_after_inspecting_original_VAL_and_partial_readout_diagnostics=True,
                    new_optimizer_steps=0, original_gates_unchanged=True,
                    test_pair_file_used_only_for_original_candidate_vocabulary=True,
                    test_images_labels_scores_used=False,
                    budget_thresholds=dict(cw_baseline=0.20, cw_name=0.10, cw_mean_perm=0.10,
                                           cw_hm_floor=-0.5, ow_auc_floor=-0.5, ow_unseen_floor=-2.0))
    write_json(protocol_path, protocol)
    pb = torch.softmax(base.flatten(1) / TEMPERATURE, dim=1).reshape_as(base)
    rows = torch.arange(len(attrs))
    baseline_correct = base[rows, objects].argmax(1).eq(attrs)
    labels = ref['labels']
    mask = torch.zeros((12, 16), dtype=torch.bool)
    for a, o in split['train']:
        mask[labels['objs'].index(o), labels['attrs'].index(a)] = True
    unseen = ~mask[objects, attrs]
    records, selected = {}, {}
    for variant in VARIANTS:
        records[variant] = []
        history = read_json(folder / 'pilot/runs' / variant / 'history.json')['ledger']
        if [item['epoch'] for item in history] != list(range(11)):
            raise ValueError('Incomplete epoch coverage')
        for item in history:
            path = folder / 'pilot/runs' / variant / item['path'] / 'grid_scores.npy'
            if sha(path) != item['files_sha256']['grid_scores.npy']:
                raise ValueError('Archived grid hash changed')
            trained = torch.from_numpy(np.load(path, allow_pickle=False).copy())
            if trained.shape != base.shape or not torch.isfinite(trained).all():
                raise ValueError('Invalid archived grid')
            if item['epoch'] == 0:
                if not torch.equal(trained, base):
                    raise ValueError('Initial scores not exact baseline')
                grid, metrics = base, replay_base
            else:
                grid = arithmetic_mix(base, trained)
                metrics = evaluate(grid, attrs, objects, datasets, module.Evaluator)
            probabilities = torch.softmax(grid.flatten(1) / TEMPERATURE, dim=1).reshape_as(base)
            error = float((probabilities.sum(-1) - pb.sum(-1)).abs().max())
            tv = float(.5 * (probabilities - pb).abs().sum((1, 2)).max())
            if error > 1e-5 or tv > ALPHA + 1e-5:
                raise ValueError('Mixture probability identity failed')
            correct = grid[rows, objects].argmax(1).eq(attrs)
            current = dict(epoch=item['epoch'], metrics_raw=metrics,
                           deltas_pp=dict(cw_auc=100*(metrics['cw']['AUC']-baseline['cw']['AUC']),
                                          ow_auc=100*(metrics['ow']['AUC']-baseline['ow']['AUC']),
                                          cw_hm=100*(metrics['cw']['best_hm']-baseline['cw']['best_hm']),
                                          ow_best_unseen=100*(metrics['ow']['best_unseen']-baseline['ow']['best_unseen'])),
                           max_object_marginal_error=error, joint_total_variation_max=tv,
                           gt_object_attr_accuracy=float(correct.float().mean()),
                           unseen_gt_object_attr_accuracy=float(correct[unseen].float().mean()),
                           corrected=int((correct & ~baseline_correct).sum()),
                           damaged=int((~correct & baseline_correct).sum()))
            records[variant].append(current)
            if variant not in selected or metrics['cw']['AUC'] > selected[variant]['metrics_raw']['cw']['AUC']:
                selected[variant] = current
            print(variant, item['epoch'], 'CW', round(100*metrics['cw']['AUC'], 6),
                  'OW', round(100*metrics['ow']['AUC'], 6), flush=True)
    def cw_auc(variant):
        return selected[variant]['metrics_raw']['cw']['AUC']
    pk = selected['P_K']
    deltas = dict(pk_minus_baseline_cw_auc_pp=pk['deltas_pp']['cw_auc'],
                  pk_minus_name_cw_auc_pp=100*(cw_auc('P_K')-cw_auc('P_name')),
                  pk_minus_perm1_cw_auc_pp=100*(cw_auc('P_K')-cw_auc('P_perm1')),
                  pk_minus_perm2_cw_auc_pp=100*(cw_auc('P_K')-cw_auc('P_perm2')),
                  pk_minus_mean_perm_cw_auc_pp=100*(cw_auc('P_K')-.5*(cw_auc('P_perm1')+cw_auc('P_perm2'))))
    conditions = dict(cw_baseline=deltas['pk_minus_baseline_cw_auc_pp'] >= .20,
                      cw_name=deltas['pk_minus_name_cw_auc_pp'] >= .10,
                      cw_mean_perm=deltas['pk_minus_mean_perm_cw_auc_pp'] >= .10,
                      cw_hm=pk['deltas_pp']['cw_hm'] >= -.5,
                      ow_auc=pk['deltas_pp']['ow_auc'] >= -.5,
                      ow_best_unseen=pk['deltas_pp']['ow_best_unseen'] >= -2)
    write_json(args.out, dict(schema='fixed_mixture_full_epoch_development_replay_v1',
                              protocol_sha256=sha(protocol_path), protocol=protocol,
                              baseline_metrics_raw=baseline, records=records, selected=selected,
                              pk_development_gate=dict(passed=all(conditions.values()), conditions=conditions, deltas_pp=deltas),
                              original_pilot_gate_passed=old_summary['decision']['pilot_budget_gate_passed'],
                              no_claim_of_statistical_significance_or_knowledge_attribution=True,
                              generic_still_missing_same_architecture_content_controls=True))
    print('FIXED_MIXTURE_FULL_EPOCH_REPLAY_COMPLETE', 'DEVELOPMENT_K_GATE', all(conditions.values()), flush=True)


if __name__ == '__main__':
    main()
