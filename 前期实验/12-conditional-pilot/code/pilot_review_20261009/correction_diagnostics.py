"""Post-hoc fixed-alpha score diagnostics; no training and no change to old gates."""
import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / 'conditional_attribute_pilot_20261009'))
import numpy as np
import torch
from common import read_json, sha, write_json
from head import TEMPERATURE, VARIANTS, corrected_scores
from runtime import make_dataset, native_metrics

ALPHA = 0.25  # Fixed before this diagnostic; no alpha search/selection.


def evaluate(grid, attrs, objects, datasets, evaluator):
    with contextlib.redirect_stdout(io.StringIO()):
        return {setting: native_metrics(grid, attrs, objects, ds, evaluator) for setting, ds in datasets.items()}


def arithmetic_mix(base, trained):
    # Mixture of conditionals, with the same frozen object marginal.
    qb = torch.softmax(base / TEMPERATURE, dim=-1)
    qt = torch.softmax(trained / TEMPERATURE, dim=-1)
    mixture = (1-ALPHA) * qb + ALPHA * qt
    if not (mixture > 0).all():
        raise ValueError('Cannot take log of a zero mixture probability')
    object_log_mass = TEMPERATURE * torch.logsumexp(base / TEMPERATURE, dim=-1, keepdim=True)
    return object_log_mass + TEMPERATURE * mixture.log()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    folder = args.results / 'extracted'
    ref = read_json(folder / 'release/reference.json')
    if not read_json(args.results / 'independent_audit.json')['passed']:
        raise ValueError('Audit first')
    if sha(args.evaluator) != ref['c_files'][ref['c_source'] + '/evaluator_ge.py']:
        raise ValueError('Native evaluator changed')
    if sha(args.metadata) != ref['c_files']['/data/xhw/czsl_repro/data/ut-zap50k/metadata_compositional-split-natural.t7']:
        raise ValueError('Metadata changed')
    split = {phase: [tuple(line.split()) for line in (args.split / (phase + '_pairs.txt')).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
             for phase in ('train', 'val', 'test')}
    datasets = {s: make_dataset(ref, s, split) for s in ('cw', 'ow')}
    spec = importlib.util.spec_from_file_location('correction_verified_evaluator', args.evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base_path = folder / 'pilot/runs/P_K/epochs/epoch_0000/attempt_0000/grid_scores.npy'
    base = torch.from_numpy(np.load(base_path, allow_pickle=False).copy())
    with np.load(folder / 'pilot/val_truth.npz', allow_pickle=False) as truth:
        attrs, objects = torch.from_numpy(truth['attrs'].copy()), torch.from_numpy(truth['objects'].copy())
    original = read_json(folder / 'pilot/baseline.json')['metrics_raw']
    labels = ref['labels']
    train_mask = torch.zeros((12, 16), dtype=torch.bool)
    eligible_mask = torch.zeros((12, 16), dtype=torch.bool)
    for a, o in split['train']:
        train_mask[labels['objs'].index(o), labels['attrs'].index(a)] = True
    for a, o in split['train'] + split['val']:
        eligible_mask[labels['objs'].index(o), labels['attrs'].index(a)] = True
    seen_gt = train_mask[objects, attrs]
    counts = torch.zeros((12, 16), dtype=torch.float32)
    for entry in torch.load(args.metadata, weights_only=True):
        if entry['set'] == 'train' and entry['attr'] in labels['attrs'] and entry['obj'] in labels['objs']:
            counts[labels['objs'].index(entry['obj']), labels['attrs'].index(entry['attr'])] += 1
    if int(counts.sum()) != 22998:
        raise ValueError('Unexpected TRAIN pair counts')
    frequency = (counts + 1).log()
    frequency = frequency - frequency.mean(-1, keepdim=True)
    base_prob = torch.softmax(base.flatten(1) / TEMPERATURE, dim=1).reshape_as(base)
    base_marginal = base_prob.sum(-1)
    baseline_attr_correct = base[torch.arange(len(attrs)), objects].argmax(1).eq(attrs)
    records = {}
    for variant in VARIANTS:
        history = read_json(folder / 'pilot/runs' / variant / 'history.json')['ledger']
        # Probe the best actual trained checkpoint, plus the final checkpoint.
        best_trained = max(history[1:], key=lambda item: item['metrics_raw']['cw']['AUC'])
        epochs = sorted({best_trained['epoch'], 10})
        records[variant] = []
        for epoch in epochs:
            item = history[epoch]
            trained = torch.from_numpy(np.load(folder / 'pilot/runs' / variant / item['path'] / 'grid_scores.npy', allow_pickle=False).copy())
            # Attribute-centering removes the per-object normalization constant.
            residual = trained - base
            contrast = residual - residual.mean(-1, keepdim=True)
            mean = contrast.mean(0)
            total_energy = float(contrast.square().mean())
            static_energy = float(mean.square().mean())
            pair_component = mean - mean.mean(0, keepdim=True)
            correlation = float(np.corrcoef(mean.flatten().numpy(), frequency.flatten().numpy())[0, 1])
            energies = dict(static_mean_fraction_of_contrast_energy=static_energy / total_energy,
                            static_pair_interaction_fraction_of_total_energy=float(pair_component.square().mean()) / total_energy,
                            correlation_with_smoothed_TRAIN_log_frequency=correlation,
                            mean_seen_minus_unseen_contrast=float(mean[train_mask].mean() - mean[~train_mask].mean()),
                            uses_VAL_inputs_for_descriptive_statistics_only=True,
                            mean_is_not_used_to_center_predictions_or_train_a_model=True)
            probes = {'original_trained': trained,
                      'quarter_logit_residual': corrected_scores(base, ALPHA * (trained-base)),
                      'quarter_conditional_mixture': arithmetic_mix(base, trained)}
            cases = {}
            for name, grid in probes.items():
                metrics = item['metrics_raw'] if name == 'original_trained' else evaluate(grid, attrs, objects, datasets, module.Evaluator)
                probabilities = torch.softmax(grid.flatten(1) / TEMPERATURE, dim=1).reshape_as(base)
                error = float((probabilities.sum(-1)-base_marginal).abs().max())
                if error > 1e-5:
                    raise ValueError('Object margin identity failed')
                prediction = grid[torch.arange(len(attrs)), objects].argmax(1)
                correct = prediction.eq(attrs)
                predicted_train_pair = train_mask[objects, prediction]
                # Full-grid score argmax outside the CW VAL candidate mask, at bias 0.
                argmax = grid.flatten(1).argmax(1)
                outside = ~eligible_mask.flatten()[argmax]
                deltas = {s + '_auc_pp': 100*(metrics[s]['AUC']-original[s]['AUC']) for s in datasets}
                deltas.update(cw_hm_pp=100*(metrics['cw']['best_hm']-original['cw']['best_hm']),
                              ow_best_unseen_pp=100*(metrics['ow']['best_unseen']-original['ow']['best_unseen']))
                cases[name] = dict(metrics_raw=metrics, deltas_pp=deltas,
                                   max_object_marginal_error=error,
                                   unseen_gt_object_attr_accuracy=float(correct[~seen_gt].float().mean()),
                                   unseen_predicted_TRAIN_pair_fraction=float(predicted_train_pair[~seen_gt].float().mean()),
                                   conditional_corrected=int((correct & ~baseline_attr_correct).sum()),
                                   conditional_damaged=int((~correct & baseline_attr_correct).sum()),
                                   full_grid_bias0_predicted_outside_CW_VAL_candidates_fraction=float(outside.float().mean()),
                                   joint_total_variation_max=float(.5*(probabilities-base_prob).abs().sum((1, 2)).max()))
                print(variant, epoch, name, 'CW_DELTA', round(deltas['cw_auc_pp'], 6), 'OW_DELTA', round(deltas['ow_auc_pp'], 6), flush=True)
            records[variant].append(dict(epoch=epoch, static_contrast=energies, cases=cases))
    write_json(args.out, dict(schema='pilot_fixed_quarter_correction_diagnostics_v1',
                              alpha=ALPHA, alpha_search=False, training_steps=0,
                              diagnostic_is_post_hoc=True, original_gates_unchanged=True,
                              not_a_preregistered_result_or_statistical_confirmation=True,
                              selection='Original best trained CW checkpoint (epochs 1..10), plus epoch 10; no new epoch selection',
                              records=records))
    print('FIXED_QUARTER_DIAGNOSTICS_COMPLETE')


if __name__ == '__main__':
    main()
