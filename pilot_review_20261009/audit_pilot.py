"""Independently audit the downloaded fixed-budget pilot; no model training."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / 'conditional_attribute_pilot_20261009'))
import numpy as np
import torch
from common import read_json, sha, state_sha, tensor_load, verify_files, write_json
from head import ConditionalHead, TEMPERATURE, VARIANTS
from pilot import decision, owned_child
from runtime import FIELDS, conditional_by_seen, make_dataset, native_metrics, paired_predictions, train_fold


def must(condition, message):
    if not condition:
        raise ValueError(message)


def vectors(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--stage-a', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.input.resolve()
    pilot, release = root / 'pilot', root / 'release'
    manifest = read_json(root / 'manifest.json')
    must(manifest['execution_passed'], 'Pilot execution failed')
    verify_files({str(owned_child(root, name)): spec['sha256'] for name, spec in manifest['files'].items()})
    release_manifest = read_json(release / 'release_manifest.json')
    verify_files({str(release / name): digest for name, digest in release_manifest.items()})
    for name in release_manifest:
        must((release / name).read_bytes() == (WORKSPACE / 'conditional_attribute_pilot_20261009' / name).read_bytes(),
             'Executed source differs from verified local r2: ' + name)
    protocol, ref, summary = read_json(pilot / 'protocol.json'), read_json(release / 'reference.json'), read_json(pilot / 'summary.json')
    protocol_sha = sha(pilot / 'protocol.json')
    must(protocol['release_manifest_sha256'] == sha(release / 'release_manifest.json'), 'Release identity differs')
    must(protocol['settings'] == ref['pilot_settings'] and protocol['controls'] == ref['control_protocol'], 'Protocol changed')
    must(protocol['budget_gate'] == ref['budget_gate'], 'Budget gate changed')
    must(summary['protocol_sha256'] == protocol_sha, 'Summary protocol differs')
    must(summary['formal_optimizer_steps'] == 4500 and summary['backbone_optimizer_steps'] == 0
         and summary['image_decodes'] == 0 and not summary['test_evaluation'], 'Budget/scope differs')
    must(read_json(pilot / 'execution_status.json') == dict(state='complete', protocol_sha256=protocol_sha), 'Completion differs')
    selfcheck = read_json(pilot / 'resume_selfchecks/attempt_0000/selfcheck.json')
    must(selfcheck['passed'] and selfcheck['next_step_parameter_state_exact'] and selfcheck['protocol_sha256'] == protocol_sha,
         'CUDA resume selfcheck failed')
    must(not (pilot / 'failures').exists(), 'Unexpected failure attempts in completed r2')
    must(sha(args.evaluator) == ref['c_files'][ref['c_source'] + '/evaluator_ge.py'], 'Native evaluator hash differs')
    must(sha(args.bank) == ref['c_reference']['bank_sha256'], 'Knowledge bank hash differs')
    must(sha(args.metadata) == ref['c_files']['/data/xhw/czsl_repro/data/ut-zap50k/metadata_compositional-split-natural.t7'],
         'Metadata hash differs')
    split = {}
    for phase in ('train', 'val', 'test'):
        path = args.split / (phase + '_pairs.txt')
        must(sha(path) == ref['c_files']['/data/xhw/czsl_repro/data/ut-zap50k/compositional-split-natural/' + path.name],
             'Split hash differs')
        split[phase] = [tuple(line.split()) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    datasets = {setting: make_dataset(ref, setting, split) for setting in ('cw', 'ow')}
    labels = read_json(pilot / 'labels.json')
    must(labels == ref['labels'], 'Primitive order differs')
    for setting, ds in datasets.items():
        must([list(p) for p in ds.pairs] == read_json(pilot / ('candidates_' + setting + '.json')), 'Candidate order differs')
    spec = importlib.util.spec_from_file_location('verified_pilot_audit_evaluator', args.evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(pilot / 'val_truth.npz', allow_pickle=False) as truth:
        attrs, objects = torch.from_numpy(truth['attrs'].copy()), torch.from_numpy(truth['objects'].copy())
    must(len(attrs) == 3214, 'VAL size differs')
    base_path = pilot / 'runs/P_K/epochs/epoch_0000/attempt_0000/grid_scores.npy'
    base = torch.from_numpy(np.load(base_path, allow_pickle=False).copy())
    must(base.shape == (3214, 12, 16) and base.dtype == torch.float32, 'Grid shape/type differs')
    seen = np.asarray([(labels['attrs'][int(a)], labels['objs'][int(o)]) in set(split['train']) for a, o in zip(attrs, objects)])
    baseline_prediction = base[torch.arange(3214), objects].argmax(1).numpy()
    baseline_vectors = None
    with contextlib.redirect_stdout(io.StringIO()):
        baseline_metrics = {s: native_metrics(base, attrs, objects, ds, module.Evaluator) for s, ds in datasets.items()}
    must(all(abs(baseline_metrics[s][k] - ref['metrics'][s][k]) <= 1e-10 for s in datasets for k in FIELDS), 'Baseline replay differs')
    probability_base = torch.softmax(base.flatten(1) / TEMPERATURE, dim=1).reshape_as(base)
    macro_classes = sorted(set(attrs.tolist()))
    bank = torch.load(args.bank, map_location='cpu', weights_only=True)
    histories, reports, selected_vectors, source_diffs = {}, {}, {}, {}
    metric_checks, largest_metric_error, largest_nll_error, largest_margin_error = 0, 0.0, 0.0, 0.0
    for variant in VARIANTS:
        run = pilot / 'runs' / variant
        history = read_json(run / 'history.json')
        completed = read_json(run / 'complete.json')
        must(completed == summary['results'][variant], 'Complete and summary differ')
        must(history['completed_epoch'] == 10 and history['cumulative_optimizer_steps'] == 900, 'Incomplete group')
        must(history['protocol_sha256'] == protocol_sha and [x['epoch'] for x in history['ledger']] == list(range(11)),
             'Epoch ledger differs')
        last, best = tensor_load(run / 'last.pt'), tensor_load(run / 'best_auc.pt')
        must(last['ledger'] == history['ledger'] and last['best'] == history['best'], 'Checkpoint ledger differs')
        must(last['completed_epoch'] == 10 and last['cumulative_optimizer_steps'] == 900, 'Checkpoint budget differs')
        must(best['best'] == history['best'] == completed['selected'], 'Selected record differs')
        must(all(torch.equal(v, best['head_state'][k]) for k, v in last['best_state'].items()), 'Selected tensor state differs')
        must(sha(run / 'last.pt') == completed['last_checkpoint_sha256']
             and sha(run / 'best_auc.pt') == completed['best_checkpoint_sha256'], 'Checkpoint identity differs')
        must(all(torch.isfinite(v).all() for v in last['head_state'].values()), 'Nonfinite checkpoint')
        torch.manual_seed(0)
        head = ConditionalHead(bank, variant)
        must(head.permutations == protocol['controls']['permutations'][variant], 'Knowledge permutation differs')
        must(sum(p.numel() for p in head.parameters()) == protocol['initializations'][variant]['parameter_count'], 'Parameter count differs')
        initial_cpu_sha = state_sha(head)
        source_diffs[variant] = dict(local_cpu_initial_matches_server=initial_cpu_sha == completed['initial_state_sha256'],
                                    best_output_weight_norm=float(best['head_state']['output.weight'].norm()),
                                    last_output_weight_norm=float(last['head_state']['output.weight'].norm()))
        if completed['best_epoch'] == 0:
            must(state_digest(best['head_state']) == completed['initial_state_sha256'], 'Epoch 0 selected weights are not initial')
        independently_selected = max(history['ledger'], key=lambda x: x['metrics_raw']['cw']['AUC'])
        must(history['best'] == independently_selected and completed['best_epoch'] == independently_selected['epoch'], 'Selection differs')
        epoch_reports = []
        for item in history['ledger']:
            folder = owned_child(run, item['path'])
            verify_files({str(owned_child(folder, name)): digest for name, digest in item['files_sha256'].items()})
            saved_eval = read_json(folder / 'evaluation.json')
            must(saved_eval['metrics_raw'] == item['metrics_raw'] and saved_eval['conditional_diagnostics'] == item['conditional_diagnostics'],
                 'Evaluation ledger differs')
            epoch = item['epoch']
            grid = torch.from_numpy(np.load(folder / 'grid_scores.npy', allow_pickle=False).copy())
            must(grid.shape == base.shape and torch.isfinite(grid).all(), 'Invalid score grid')
            if epoch == 0:
                must(torch.equal(base, grid), 'Epoch 0 group differs')
            else:
                train = item['train_epoch']
                must(train['images'] == 22998 and train['steps'] == 90, 'Epoch training budget differs')
                must(set(train['nonzero_gradient_parameters']) == set(dict(head.named_parameters())), 'Inactive trained parameters')
            with contextlib.redirect_stdout(io.StringIO()):
                independent = {s: native_metrics(grid, attrs, objects, ds, module.Evaluator) for s, ds in datasets.items()}
            for setting in datasets:
                for field in FIELDS:
                    error = abs(independent[setting][field] - item['metrics_raw'][setting][field])
                    largest_metric_error = max(largest_metric_error, error)
                    must(error <= 1e-7, 'Independent metric mismatch: %s/%d/%s/%s %.12g' % (variant, epoch, setting, field, error))
                    metric_checks += 1
            prediction = vectors(folder / 'conditional_predictions.npz')
            must(np.array_equal(prediction['indices'], np.arange(3214)) and np.array_equal(prediction['attrs'], attrs.numpy())
                 and np.array_equal(prediction['objects'], objects.numpy()), 'Prediction identities differ')
            logits = grid[torch.arange(3214), objects] / TEMPERATURE
            nll = torch.nn.functional.cross_entropy(logits, attrs, reduction='none').numpy()
            predicted = logits.argmax(1).numpy()
            correct = predicted == attrs.numpy()
            base_correct = baseline_prediction == attrs.numpy()
            must(np.array_equal(prediction['conditional_prediction'], predicted) and np.array_equal(prediction['correct'], correct)
                 and np.array_equal(prediction['baseline_correct'], base_correct), 'Conditional argmax differs')
            nll_error = float(np.max(np.abs(prediction['conditional_nll'] - nll)))
            largest_nll_error = max(largest_nll_error, nll_error)
            must(nll_error <= 2e-5, 'Per-image NLL differs')
            diagnostic = item['conditional_diagnostics']
            must(abs(correct.mean() - diagnostic['gt_object_attr_accuracy']) <= 1e-7, 'Attribute accuracy differs')
            must(abs(np.mean([correct[attrs.numpy() == a].mean() for a in macro_classes]) - diagnostic['macro_attr_accuracy']) <= 1e-7,
                 'Macro attribute accuracy differs')
            must(int((correct & ~base_correct).sum()) == diagnostic['corrected_from_baseline']
                 and int((~correct & base_correct).sum()) == diagnostic['damaged_from_baseline'], 'Correction/damage differs')
            probability = torch.softmax(grid.flatten(1) / TEMPERATURE, dim=1).reshape_as(grid)
            margin_error = float((probability.sum(-1) - probability_base.sum(-1)).abs().max())
            largest_margin_error = max(largest_margin_error, margin_error)
            must(margin_error < 1e-5, 'Object margin changed')
            independent_seen = conditional_by_seen(prediction, datasets)
            must(independent_seen == diagnostic['by_seen_pair'], 'Seen/unseen conditional diagnostic differs')
            # Where does a GT-object classifier place unseen-pair images?
            predicted_seen_pair = np.asarray([(labels['attrs'][int(a)], labels['objs'][int(o)]) in set(split['train'])
                                              for a, o in zip(predicted, objects)])
            subset_data = {}
            for name, mask in (('seen', seen), ('unseen', ~seen)):
                subset_data[name] = dict(images=int(mask.sum()), accuracy=float(correct[mask].mean()),
                                         corrected=int((correct & ~base_correct & mask).sum()),
                                         damaged=int((~correct & base_correct & mask).sum()),
                                         predicted_TRAIN_pair_fraction=float(predicted_seen_pair[mask].mean()))
            seen_mask_grid = torch.zeros((12, 16), dtype=torch.bool)
            for a, o in split['train']:
                seen_mask_grid[labels['objs'].index(o), labels['attrs'].index(a)] = True
            conditional_prob = torch.softmax(grid / TEMPERATURE, dim=-1)[torch.arange(3214), objects]
            seen_mass = (conditional_prob * seen_mask_grid[objects]).sum(1).numpy()
            for name, mask in (('seen', seen), ('unseen', ~seen)):
                subset_data[name]['mean_conditional_TRAIN_pair_mass'] = float(seen_mass[mask].mean())
            report = dict(epoch=epoch, independent_metrics_raw=independent, subsets=subset_data,
                          max_per_image_nll_error=nll_error, max_object_margin_error=margin_error,
                          train_nll=item['train_epoch']['mean_conditional_nll'] if epoch else None,
                          val_nll=diagnostic['conditional_nll'], macro_accuracy=diagnostic['macro_attr_accuracy'])
            epoch_reports.append(report)
            if epoch == 0 and baseline_vectors is None:
                baseline_vectors = prediction
            if epoch == completed['best_epoch']:
                selected_vectors[variant] = prediction
        histories[variant] = history
        reports[variant] = epoch_reports
        print('INDEPENDENT_GROUP_AUDIT_PASSED', variant, 'best_epoch', completed['best_epoch'], flush=True)
    for epoch in range(1, 11):
        must(len({histories[v]['ledger'][epoch]['train_epoch']['epoch_order_sha256'] for v in VARIANTS}) == 1, 'Unmatched group order')
    must(decision(summary['results'], baseline_metrics, ref['budget_gate']) == summary['decision'], 'Decision reconstruction differs')
    must({v: paired_predictions(selected_vectors['P_K'], selected_vectors[v]) for v in ('P_name', 'P_perm1', 'P_perm2', 'P_generic')}
         == summary['pk_paired_val_conditional'], 'Selected pairing differs')
    # Verify TRAIN mechanics against real metadata and Stage A image order.
    train_ids = read_json(args.stage_a / 'stageA/train_image_ids.json')
    must(sha(args.stage_a / 'stageA/train_image_ids.json') == ref['stageA_files'][ref['stageA_base'] + '/train_image_ids.json'], 'TRAIN image order differs')
    metadata = torch.load(args.metadata, map_location='cpu', weights_only=True)
    by_image = {entry['image']: entry for entry in metadata if entry['set'] == 'train'}
    train_labels = dict(attrs=torch.tensor([labels['attrs'].index(by_image[name]['attr']) for name in train_ids]),
                       objects=torch.tensor([labels['objs'].index(by_image[name]['obj']) for name in train_ids]))
    fit, hold, counts = train_fold(train_labels, train_ids)
    mechanic = read_json(pilot / 'mechanics_complete.json')
    verify_files({str(owned_child(pilot, name)): digest for name, digest in mechanic['files_sha256'].items()})
    folder = owned_child(pilot, mechanic['path'])
    saved_fold = read_json(folder / 'fold.json')
    must(fit.tolist() == saved_fold['fit_indices'] and hold.tolist() == saved_fold['hold_indices']
         and counts == saved_fold['pair_counts'], 'Mechanics fold differs from real image hash split')
    mechanics_summary = read_json(folder / 'summary.json')
    hold_vectors = {}
    for variant in VARIANTS:
        diagnostic = read_json(folder / (variant + '.json'))
        prediction = vectors(folder / (variant + '_hold_predictions.npz'))
        must(np.array_equal(prediction['indices'], hold.numpy())
             and np.array_equal(prediction['attrs'], train_labels['attrs'][hold].numpy())
             and np.array_equal(prediction['objects'], train_labels['objects'][hold].numpy()), 'Mechanics image/label identity differs')
        must(np.array_equal(prediction['correct'], prediction['conditional_prediction'] == prediction['attrs']), 'Mechanics correctness differs')
        must(abs(float(prediction['correct'].mean()) - diagnostic['hold_after']['gt_object_attr_accuracy']) <= 1e-7, 'Mechanics accuracy differs')
        must(diagnostic['train_epoch']['steps'] == 72 and diagnostic['train_epoch']['images'] == 18426, 'Mechanics budget differs')
        hold_vectors[variant] = prediction
    must({v: paired_predictions(hold_vectors['P_K'], hold_vectors[v]) for v in ('P_name', 'P_perm1', 'P_perm2', 'P_generic')}
         == mechanics_summary['pk_paired_holdout'], 'Mechanics pairing differs')
    total_seconds = sum(item.get('epoch_seconds', 0) for history in histories.values() for item in history['ledger'])
    record = dict(passed=True, protocol_sha256=protocol_sha, executed_release_matches_local_r2=True,
                  independent_metric_checks=metric_checks, max_metric_difference_raw=largest_metric_error,
                  max_per_image_nll_difference=largest_nll_error, max_object_margin_error=largest_margin_error,
                  checkpoints_safe_loaded_and_identity_checked=True, checkpoint_states=source_diffs,
                  selected_checkpoint_forward_independently_replayed=False,
                  checkpoint_forward_limitation='Frozen visual feature caches remain on server; archived CUDA replay is provenance-verified, not independently re-executed locally.',
                  fit_images=len(fit), hold_images=len(hold), matched_epoch_orders=True,
                  formal_steps=4500, mechanics_steps=360, discarded_resume_check_steps=3,
                  formal_epoch_seconds_sum=total_seconds, reports=reports,
                  selected_metrics={v: histories[v]['best']['metrics_raw'] for v in VARIANTS},
                  mechanics_reports=mechanics_summary['reports'], gate=summary['decision'])
    write_json(args.out, record)
    print('PILOT_INDEPENDENT_AUDIT_PASSED', metric_checks, 'metrics', 'largest_difference', largest_metric_error, flush=True)


if __name__ == '__main__':
    main()
