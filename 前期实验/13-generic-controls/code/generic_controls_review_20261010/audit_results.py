"""Independent audit of the downloaded generic mixture-control result grids."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

FIELDS = ('AUC', 'best_hm', 'best_seen', 'best_unseen', 'hm_seen', 'hm_unseen')
VARIANTS = ('G_K', 'G_name', 'G_perm1', 'G_perm2')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def dataset(labels, split, setting):
    attrs, objs = labels['attrs'], labels['objs']
    pairs = sorted(set(split['train'] + split['val'] + split['test'])) if setting == 'cw' else [(a, o) for a in attrs for o in objs]
    return SimpleNamespace(attrs=attrs, objs=objs, pairs=pairs, train_pairs=split['train'],
                           val_pairs=split['val'], test_pairs=split['test'],
                           attr2idx={a: i for i, a in enumerate(attrs)}, obj2idx={o: i for i, o in enumerate(objs)},
                           pair2idx={p: i for i, p in enumerate(pairs)}, phase='val', open_world=setting == 'ow')


def metrics(grid, attrs, objects, ds, evaluator_class):
    ai = torch.tensor([ds.attr2idx[a] for a, _ in ds.pairs])
    oi = torch.tensor([ds.obj2idx[o] for _, o in ds.pairs])
    matrix = grid[:, oi, ai].contiguous()
    prediction = {pair: matrix[:, i] for i, pair in enumerate(ds.pairs)}
    gt_pair = torch.tensor([ds.pair2idx[(ds.attrs[int(a)], ds.objs[int(o)])] for a, o in zip(attrs, objects)])
    evaluator = evaluator_class(ds, None)
    with contextlib.redirect_stdout(io.StringIO()):
        scored = evaluator.score_model(prediction, objects, bias=1e3, topk=1)
        stats = evaluator.evaluate_predictions(scored, attrs, objects, gt_pair, prediction, topk=1)
    return {k: float(v) for k, v in stats.items()}


def history_path(controls, variant):
    if variant == 'G_K':
        return controls / 'runs/G_K/reference_replays/attempt_0000/history.json'
    return controls / ('runs/' + variant + '/history.json')


def run_path(controls, variant):
    return controls / ('runs/' + variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extracted', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--old-pilot', type=Path, required=True)
    parser.add_argument('--old-replay', type=Path, required=True)
    parser.add_argument('--local-release', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    controls, release = args.extracted / 'controls', args.extracted / 'release'
    manifest = read(args.extracted / 'manifest.json')
    summary, protocol = read(controls / 'summary.json'), read(controls / 'protocol.json')
    errors, checks = [], 0

    def require(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    require(manifest['execution_passed'] is True, 'Export did not pass')
    require(manifest['new_formal_optimizer_steps'] == 2700, 'Manifest formal budget differs')
    require(summary['passed'] is True, 'Summary failed')
    require(summary['new_formal_optimizer_steps'] == 2700, 'Summary formal budget differs')
    require(summary['backbone_optimizer_steps'] == 0 and summary['image_decodes'] == 0 and summary['test_evaluation'] is False,
            'Unexpected backbone/image/TEST work')
    require(digest(controls / 'protocol.json') == summary['protocol_sha256'], 'Protocol hash differs')
    require(protocol['alpha'] == .25 and protocol['expected_new_formal_optimizer_steps'] == 2700,
            'Protocol alpha/budget differs')
    require(protocol['training_loss'] == 'unchanged_unmixed_conditional_CE', 'Training loss changed')
    require(protocol['readout'] == 'fixed_arithmetic_probability_mixture', 'Readout changed')
    require(protocol['formal_variants'] == ['G_name', 'G_perm1', 'G_perm2'], 'Formal identities differ')
    require(summary['source_old_gate'] is False and summary['decision']['original_gates_unchanged'], 'Old gate was rewritten')
    require(summary['reference_reproduction_diagnostic_steps'] == 90 and summary['mechanics_diagnostic_steps'] == 216,
            'Diagnostic step accounting differs')
    require(not list(controls.glob('failures/*/failure.json')), 'Failure attempts exist')

    # Executed release and local sealed release must be identical.
    release_manifest = read(release / 'release_manifest.json')
    local_manifest = read(args.local_release / 'release_manifest.json')
    require(release_manifest == local_manifest, 'Executed/local release manifests differ')
    for name, expected in release_manifest.items():
        require(digest(release / name) == expected, 'Executed release member changed: ' + name)
        require(digest(args.local_release / name) == expected, 'Local release member changed: ' + name)

    reference = read(release / 'experiment.json')
    require(reference['source_protocol_sha256'] == protocol['source_protocol_sha256'], 'Source protocol identity differs')
    require(reference['source_initial_state_sha256'] == protocol['initializations']['G_K']['initial_state_sha256'],
            'Correct-head initialization identity differs')
    init_shas = {v: protocol['initializations'][v]['initial_state_sha256'] for v in VARIANTS}
    require(len(set(init_shas.values())) == 1, 'Generic initializations not all equal')
    require(all(protocol['initializations'][v]['parameter_count'] == 60864 for v in VARIANTS), 'Parameter count mismatch')
    replay_check = read(controls / 'reference_checks_complete.json')
    require(all(replay_check[k] is True for k in ('passed', 'initial_parameters_exact', 'last_checkpoint_actual_forward_exact',
                                                   'first_epoch_training_ledger_exact', 'first_epoch_VAL_scores_exact',
                                                   'all_diagnostic_weights_discarded')), 'Source correct-run reuse check failed')

    if digest(args.evaluator) != '24b48c4d48e3475baec0a656fca194cdbc6d4b4e32e87c0f22cda3daf586d978':
        raise ValueError('Native evaluator identity differs')
    spec = importlib.util.spec_from_file_location('generic_control_audit_evaluator', args.evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = read(controls / 'labels.json')
    split = {phase: [tuple(x.split()) for x in (args.split / (phase + '_pairs.txt')).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
             for phase in ('train', 'val', 'test')}
    datasets = {setting: dataset(labels, split, setting) for setting in ('cw', 'ow')}
    require(list(map(len, (split['train'], split['val'], split['test']))) == [83, 30, 36], 'Split sizes differ')
    for setting, ds in datasets.items():
        require(read(controls / ('candidates_' + setting + '.json')) == [list(x) for x in ds.pairs], 'Candidate order differs: ' + setting)
    with np.load(controls / 'val_truth.npz', allow_pickle=False) as truth:
        attrs = torch.from_numpy(truth['attrs'].copy())
        objects = torch.from_numpy(truth['objects'].copy())
    require(len(attrs) == 3214 and attrs.shape == objects.shape, 'VAL truth shape differs')
    baseline = read(controls / 'baseline.json')['metrics_raw']
    base = torch.from_numpy(np.load(args.old_pilot / 'runs/P_generic/epochs/epoch_0000/attempt_0000/grid_scores.npy', allow_pickle=False).copy())
    require(torch.equal(base, torch.from_numpy(np.load(controls / 'runs/G_K/reference_replays/attempt_0000/epoch_0000/grid_scores.npy', allow_pickle=False).copy())),
            'G_K zero readout is not exact baseline')
    base_probability = torch.softmax(base.flatten(1) / .05, dim=1).reshape_as(base)
    base_object = base_probability.sum(-1)
    rows = torch.arange(len(attrs))
    base_correct = base[rows, objects].argmax(1).eq(attrs)
    train_mask = torch.zeros((12, 16), dtype=torch.bool)
    for a, o in split['train']:
        train_mask[labels['objs'].index(o), labels['attrs'].index(a)] = True
    seen = train_mask[objects, attrs]

    max_metric_diff = 0.0
    max_mix_error = 0.0
    max_object_error = 0.0
    max_tv = 0.0
    metric_count = 0
    records = {}
    source_history = read(args.old_pilot / 'runs/P_generic/history.json')['ledger']
    source_orders = {item['epoch']: item['train_epoch']['epoch_order_sha256'] for item in source_history[1:]}
    old_replay = read(args.old_replay)
    for variant in VARIANTS:
        history = read(history_path(controls, variant))
        ledger = history['ledger']
        require([x['epoch'] for x in ledger] == list(range(11)), variant + ' incomplete epochs')
        selected = max(ledger, key=lambda x: x['metrics_raw']['cw']['AUC'])
        expected_selected = summary['results'][variant]['selected']
        require(selected['epoch'] == summary['results'][variant]['best_epoch'], variant + ' best epoch differs')
        require(selected['metrics_raw'] == expected_selected['metrics_raw'], variant + ' selected metrics differ')
        records[variant] = []
        for item in ledger:
            folder = run_path(controls, variant) / item['path']
            for name, expected in item['files_sha256'].items():
                require(digest(folder / name) == expected, variant + ' artifact hash differs: ' + name)
            grid = torch.from_numpy(np.load(folder / 'grid_scores.npy', allow_pickle=False).copy())
            unmixed = torch.from_numpy(np.load(folder / 'unmixed_grid_scores.npy', allow_pickle=False).copy())
            require(grid.shape == (3214, 12, 16) and unmixed.shape == grid.shape, variant + ' grid shape differs')
            if item['epoch'] == 0:
                require(torch.equal(grid, base) and torch.equal(unmixed, base), variant + ' epoch zero not exact baseline')
            joint = torch.softmax(grid.flatten(1) / .05, dim=1).reshape_as(base)
            q_unmixed = torch.softmax(unmixed / .05, dim=-1)
            expected_joint = .75 * base_probability + .25 * base_object.unsqueeze(-1) * q_unmixed
            mix_error = float((joint - expected_joint).abs().max())
            object_error = float((joint.sum(-1) - base_object).abs().max())
            tv = float(.5 * (joint - base_probability).abs().sum((1, 2)).max())
            max_mix_error, max_object_error, max_tv = max(max_mix_error, mix_error), max(max_object_error, object_error), max(max_tv, tv)
            require(mix_error < 1e-5 and object_error < 1e-5 and tv <= .25001, variant + ' mixture identity failed')
            recomputed = {s: metrics(grid, attrs, objects, ds, module.Evaluator) for s, ds in datasets.items()}
            for setting in datasets:
                for field in FIELDS:
                    diff = abs(recomputed[setting][field] - item['metrics_raw'][setting][field])
                    max_metric_diff = max(max_metric_diff, diff)
                    metric_count += 1
                    require(diff < 1e-12, variant + ' native metric differs')
            prediction = grid[rows, objects].argmax(1)
            correct = prediction.eq(attrs)
            log_q = torch.log_softmax(grid[rows, objects] / .05, dim=-1)
            nll = -log_q[rows, attrs]
            diag = item['conditional_diagnostics']
            require(abs(float(correct.float().mean()) - diag['gt_object_attr_accuracy']) < 1e-8, variant + ' attr accuracy differs')
            require(abs(float(nll.mean()) - diag['conditional_nll']) < 1e-6, variant + ' conditional NLL differs')
            require(int((correct & ~base_correct).sum()) == diag['corrected_from_baseline'], variant + ' correction count differs')
            require(int((~correct & base_correct).sum()) == diag['damaged_from_baseline'], variant + ' damage count differs')
            for key, mask in (('seen_pair', seen), ('unseen_pair', ~seen)):
                by = diag['by_seen_pair'][key]
                require(by['images'] == int(mask.sum()), variant + ' seen-group size differs')
                # The archived statistic used NumPy bool means (float64), while
                # this independent path reduces float32 Torch values.
                require(abs(by['attr_accuracy'] - float(correct[mask].float().mean())) < 5e-8, variant + ' seen-group accuracy differs')
            if item['epoch'] > 0:
                require(item['train_epoch']['steps'] == 90, variant + ' per-epoch formal steps differ')
                require(item['train_epoch']['epoch_order_sha256'] == source_orders[item['epoch']], variant + ' data order differs')
            if variant == 'G_K':
                source_item = source_history[item['epoch']]
                require(digest(folder / 'unmixed_grid_scores.npy') == source_item['files_sha256']['grid_scores.npy'], 'G_K source score not exact')
                replay_item = old_replay['records']['P_generic'][item['epoch']]
                for setting in datasets:
                    for field in FIELDS:
                        require(item['metrics_raw'][setting][field] == replay_item['metrics_raw'][setting][field], 'G_K prior replay differs')
            records[variant].append(dict(epoch=item['epoch'], metrics_raw=recomputed,
                                         gt_object_attr_accuracy=float(correct.float().mean()),
                                         unseen_gt_object_attr_accuracy=float(correct[~seen].float().mean()),
                                         corrected=int((correct & ~base_correct).sum()),
                                         damaged=int((~correct & base_correct).sum()),
                                         joint_total_variation_max=tv))
    require(metric_count == 528, 'Unexpected native metric count')
    require(summary['new_formal_optimizer_steps'] == sum(summary['results'][v]['formal_optimizer_steps'] for v in VARIANTS),
            'Formal step sum differs')
    require(summary['results']['G_K']['formal_optimizer_steps'] == 0 and summary['results']['G_K']['original_source_optimizer_steps'] == 900,
            'G_K source/new budget accounting differs')

    # Independently recompute the reported decision.
    selected = {v: max(records[v], key=lambda x: x['metrics_raw']['cw']['AUC']) for v in VARIANTS}
    b = baseline
    k = selected['G_K']['metrics_raw']
    def diff(right, setting='cw', field='AUC'):
        return 100 * (k[setting][field] - right[setting][field])
    decision = dict(gk_minus_baseline_cw_auc_pp=diff(b),
                    gk_minus_name_cw_auc_pp=diff(selected['G_name']['metrics_raw']),
                    gk_minus_perm1_cw_auc_pp=diff(selected['G_perm1']['metrics_raw']),
                    gk_minus_perm2_cw_auc_pp=diff(selected['G_perm2']['metrics_raw']),
                    gk_minus_mean_permutations_cw_auc_pp=100*(k['cw']['AUC']-.5*(selected['G_perm1']['metrics_raw']['cw']['AUC']+selected['G_perm2']['metrics_raw']['cw']['AUC'])),
                    gk_minus_baseline_cw_hm_pp=diff(b, field='best_hm'),
                    gk_minus_baseline_ow_auc_pp=diff(b, 'ow'),
                    gk_minus_baseline_ow_best_unseen_pp=diff(b, 'ow', 'best_unseen'))
    for key, value in decision.items():
        require(abs(value-summary['decision']['deltas_pp'][key]) < 1e-12, 'Decision delta differs: ' + key)
    thresholds = protocol['budget_gate']
    conditions = dict(cw_baseline=decision['gk_minus_baseline_cw_auc_pp'] >= thresholds['pk_minus_baseline_cw_auc_pp'],
                      cw_name=decision['gk_minus_name_cw_auc_pp'] >= thresholds['pk_minus_name_cw_auc_pp'],
                      cw_mean_perm=decision['gk_minus_mean_permutations_cw_auc_pp'] >= thresholds['pk_minus_mean_permutations_cw_auc_pp'],
                      cw_hm=decision['gk_minus_baseline_cw_hm_pp'] >= thresholds['cw_hm_floor_pp'],
                      ow_auc=decision['gk_minus_baseline_ow_auc_pp'] >= thresholds['ow_auc_floor_pp'],
                      ow_best_unseen=decision['gk_minus_baseline_ow_best_unseen_pp'] >= thresholds['ow_best_unseen_floor_pp'])
    require(conditions == summary['decision']['conditions'], 'Decision conditions differ')
    require(all(conditions.values()) == summary['decision']['generic_budget_gate_passed'], 'Gate differs')

    result = dict(schema='generic_mixture_controls_independent_audit_v1', passed=not errors,
                  checks=checks, errors=errors, native_metrics_recomputed=metric_count,
                  largest_native_metric_difference=max_metric_diff,
                  largest_mixture_probability_error=max_mix_error,
                  largest_object_marginal_error=max_object_error,
                  largest_joint_total_variation=max_tv,
                  selected=selected, independently_recomputed_deltas_pp=decision,
                  independently_recomputed_conditions=conditions,
                  independently_recomputed_gate=all(conditions.values()),
                  old_gate_unchanged=summary['source_old_gate'] is False,
                  formal_optimizer_steps=summary['new_formal_optimizer_steps'],
                  test_evaluation=False, records=records)
    write(args.out, result)
    print('GENERIC_CONTROLS_INDEPENDENT_AUDIT', 'PASS' if result['passed'] else 'FAIL',
          'checks', checks, 'metrics', metric_count, 'maxdiff', max_metric_diff, flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
