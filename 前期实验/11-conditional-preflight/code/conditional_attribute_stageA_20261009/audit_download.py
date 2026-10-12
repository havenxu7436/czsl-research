"""Independent local Stage A evidence and original-evaluator score audit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing audit')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    root = args.input
    def J(path):
        return json.loads(path.read_text(encoding='utf-8'))
    manifest = J(root / 'manifest.json')['files']
    for name, entry in manifest.items():
        if sha(root / name) != entry['sha256'] or (root / name).stat().st_size != entry['size']:
            raise ValueError('Extracted file mismatch: ' + name)
    ref = J(root / 'release/reference.json')
    prepared, probe = J(root / 'stageA/prepared.json'), J(root / 'bf_probe/probe.json')
    driver = J(root / 'logs/driver.json')
    if not prepared['passed'] or not probe['passed'] or any(driver['return_codes'].values()):
        raise ValueError('Runtime check failed')
    for name, digest in J(root / 'release/release_manifest.json').items():
        if sha(root / 'release' / name) != digest:
            raise ValueError('Executed release mismatch: ' + name)
    release_sha = sha(root / 'release/release_manifest.json')
    if prepared['release_manifest_sha256'] != release_sha or probe['release_manifest_sha256'] != release_sha:
        raise ValueError('Run used a different release')
    cache_hashes = {}
    for name, digest in prepared['files_sha256'].items():
        if name in ('train.pt', 'val.pt'):
            cache_hashes[name] = digest
        elif sha(root / 'stageA' / name) != digest:
            raise ValueError('Stage A output fingerprint differs: ' + name)
    if sha(args.evaluator) != ref['c_files'][ref['c_source'] + '/evaluator_ge.py']:
        raise ValueError('Original evaluator identity differs')
    split = {}
    for phase in ('train', 'val', 'test'):
        name = phase + '_pairs.txt'
        path = args.split / name
        server = '/data/xhw/czsl_repro/data/ut-zap50k/compositional-split-natural/' + name
        if sha(path) != ref['c_files'][server]:
            raise ValueError('Split identity differs: ' + phase)
        split[phase] = [tuple(line.split()) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    spec = importlib.util.spec_from_file_location('verified_native_stageA_evaluator', args.evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = J(root / 'stageA/labels.json')
    attrs, objs = labels['attrs'], labels['objs']
    comparisons = {}
    for setting in ('cw', 'ow'):
        pairs = [tuple(p) for p in J(root / ('stageA/candidates_' + setting + '.json'))]
        expected_pairs = sorted(set(split['train'] + split['val'] + split['test'])) if setting == 'cw' else [(a, o) for a in attrs for o in objs]
        if pairs != expected_pairs:
            raise ValueError('Candidate order differs')
        scores = np.load(root / ('stageA/val_scores_' + setting + '.npy'), allow_pickle=False)
        if scores.shape != (3214, len(pairs)) or not np.isfinite(scores).all():
            raise ValueError('Scores invalid')
        with np.load(root / ('stageA/labels_' + setting + '.npz'), allow_pickle=False) as truth:
            a, o, p = [torch.from_numpy(truth[k].copy()) for k in ('attr', 'obj', 'pair')]
            if not np.array_equal(scores.argmax(1), truth['top1_at_bias0']):
                raise ValueError('Saved top1 differs')
        ds = SimpleNamespace(attrs=attrs, objs=objs, pairs=pairs,
                             train_pairs=split['train'], val_pairs=split['val'], test_pairs=split['test'],
                             attr2idx={a: i for i, a in enumerate(attrs)}, obj2idx={o: i for i, o in enumerate(objs)},
                             pair2idx={p: i for i, p in enumerate(pairs)}, phase='val', open_world=setting == 'ow')
        matrix = torch.from_numpy(scores)
        prediction = {pair: matrix[:, i] for i, pair in enumerate(pairs)}
        ev = module.Evaluator(ds, None)
        result = ev.score_model(prediction, o, bias=1e3, topk=1)
        stats = ev.evaluate_predictions(result, a, o, p, prediction, topk=1)
        fields = ('AUC', 'best_hm', 'best_seen', 'best_unseen', 'hm_seen', 'hm_unseen')
        metrics = {k: float(stats[k]) for k in fields}
        difference = {k: metrics[k] - ref['metrics'][setting][k] for k in fields}
        if max(abs(v) for v in difference.values()) > 1e-10:
            raise ValueError('Independent native metric replay differs')
        comparisons[setting] = dict(independent_metrics_raw=metrics, original_difference_raw=difference,
                                     original_bias0_top1_equal=prepared['replay'][setting]['original_bias0_top1_equal'])
    geometry = {}
    for variant, record in probe['variants'].items():
        observations = record['observations']
        geometry[variant] = [dict(batch=x['batch'],
                                 weighted_joint_norm_ratio_pct=100*x['scopes']['joint']['loss_logic_weighted']['right_over_left_norm'],
                                 weighted_joint_cosine=x['scopes']['joint']['loss_logic_weighted']['cosine'],
                                 ae_attribute_cosine=x['scopes']['attribute']['loss_ae']['cosine'],
                                 oi_joint_cosine=x['scopes']['joint']['loss_oi']['cosine'],
                                 cpf_total_cosine=x['scopes']['joint']['cpf_vs_total']['cosine']) for x in observations]
    smoke = J(root / 'stageA/head_smoke.json')
    if not all(x['zero_replay_exact'] and x['force_off_replay_exact'] and max(x['probabilities'].values()) < 3e-6 for x in smoke.values()):
        raise ValueError('Real head interface check failed')
    if (prepared['formal_training_optimizer_steps'] != 0 or prepared['backbone_optimizer_steps'] != 0
            or prepared['test_images_decoded'] != 0 or probe['optimizer_steps'] != 0):
        raise ValueError('Unexpected new training or TEST access')
    report = dict(passed=True, independent_native_metric_checks=12, comparisons=comparisons,
                  source_release_sha256=release_sha, cache_sha256_server_reported=cache_hashes,
                  cache_tensors_not_downloaded_or_locally_loaded=True, gradients=geometry,
                  maximum_real_probability_error=max(v for x in smoke.values() for v in x['probabilities'].values()),
                  stageA_seconds=prepared['seconds'], raw_evidence_hashes_verified=True,
                  interpretation='Engineering pass, not efficacy. Three eval-mode TRAIN batches at selected weights, not epoch means or full Adam update geometry.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('STAGE_A_INDEPENDENT_AUDIT_PASSED', json.dumps(comparisons), flush=True)


if __name__ == '__main__':
    main()
