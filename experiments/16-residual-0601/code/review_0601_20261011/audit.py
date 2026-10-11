"""Read-only audit and native CPU metric replay of downloaded 0601 results."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

sys.dont_write_bytecode = True

VARIANTS = ['G', 'H', 'C', 'D', 'A', 'B', 'E', 'F', 'H_no_pair',
            'H_shuffle', 'H_random_prior', 'H_uniform_prior']
FIELDS = ['AUC', 'best_hm', 'best_seen', 'best_unseen', 'hm_seen', 'hm_unseen']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset(labels, split, phase, setting):
    attrs, objs = labels['attrs'], labels['objs']
    pairs = sorted(set(sum(split.values(), []))) if setting == 'cw' else [(a, o) for a in attrs for o in objs]
    return SimpleNamespace(attrs=attrs, objs=objs, pairs=pairs, train_pairs=split['train'],
                           val_pairs=split['val'], test_pairs=split['test'],
                           attr2idx={a: i for i, a in enumerate(attrs)}, obj2idx={o: i for i, o in enumerate(objs)},
                           pair2idx={p: i for i, p in enumerate(pairs)}, phase=phase, open_world=setting == 'ow')


def evaluate(matrix, truth, ds, evaluator_class):
    matrix = torch.from_numpy(np.array(matrix, dtype=np.float32, copy=True))
    attrs = torch.from_numpy(truth['attr'].copy())
    objs = torch.from_numpy(truth['obj'].copy())
    pairs = torch.tensor([ds.pair2idx[(ds.attrs[a], ds.objs[o])] for a, o in zip(attrs.tolist(), objs.tolist())])
    predictions = {p: matrix[:, i] for i, p in enumerate(ds.pairs)}
    with contextlib.redirect_stdout(io.StringIO()):
        ev = evaluator_class(ds, None)
        output = ev.score_model(predictions, objs, bias=1e3, topk=1)
        stats = ev.evaluate_predictions(output, attrs, objs, pairs, predictions, topk=1)
    return {k: float(stats[k]) for k in FIELDS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.download / 'extracted'
    archive = args.download / 'czsl_0601_20261010.zip'
    checks, errors = 0, []

    def require(value, message):
        nonlocal checks
        checks += 1
        if not value:
            errors.append(message)

    require(sha(archive) == archive.with_suffix('.zip.sha256').read_text(encoding='utf-8-sig').split()[0], 'ZIP sidecar mismatch')
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, 'ZIP CRC failure')
        manifest = json.loads(z.read('export_manifest.json'))
        require(len(z.namelist()) == len(set(z.namelist())), 'Duplicate archive names')
        require(set(z.namelist()) == set(manifest['files_sha256']) | {'export_manifest.json'}, 'Archive member manifest differs')
        for name, fingerprint in manifest['files_sha256'].items():
            require(hashlib.sha256(z.read(name)).hexdigest() == fingerprint, 'ZIP SHA mismatch: ' + name)
            require(sha(root / name) == fingerprint, 'Extracted bytes mismatch: ' + name)
    with zipfile.ZipFile(args.release) as z:
        expected_source = {n.split('/source/', 1)[1]: hashlib.sha256(z.read(n)).hexdigest()
                           for n in z.namelist() if '/source/' in n and not n.endswith('/')}
        actual_source = {str(p.relative_to(root / 'source')).replace('\\', '/'): sha(p)
                         for p in (root / 'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        require(expected_source == actual_source, 'Executed training source differs from delivered r2 source')
    protocol = read(root / 'protocol.json')
    summary = read(root / 'summary/summary.json')
    require([j['variant'] for j in protocol['jobs']] == VARIANTS, 'Unexpected scheduled variants')
    require(summary['core_complete'] and summary['all_planned_complete'], 'Summary completion false')
    require(summary['protocol_sha256'] == sha(root / 'protocol.json'), 'Summary protocol mismatch')
    split = {}
    for phase in ('train', 'val', 'test'):
        path = args.split / (phase + '_pairs.txt')
        server_path = protocol['root'] + '/data/ut-zap50k/compositional-split-natural/' + path.name
        require(sha(path) == protocol['frozen_inputs'][server_path], 'Local real split bytes differ: ' + phase)
        split[phase] = [tuple(line.split()) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require([len(split[p]) for p in ('train', 'val', 'test')] == [83, 30, 36], 'Wrong split sizes')
    spec = importlib.util.spec_from_file_location('replayed_evaluator', root / 'source/evaluator_ge.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.device == 'cpu', 'Local replay must be CPU')
    rows, replay, durations = [], [], {}
    base_labels = read(root / 'runs/A/meta/labels.json')
    base_truth = {}
    for job in protocol['jobs']:
        variant = job['variant']
        folder = root / 'runs' / variant
        config = yaml.safe_load((folder / 'config_requested.yml').read_text(encoding='utf-8'))
        require(sha(root / 'configs' / (variant + '.yml')) == job['cfg_sha256'], 'Requested config SHA differs: ' + variant)
        require((root / 'configs' / (variant + '.yml')).read_bytes() == (folder / 'config_requested.yml').read_bytes(), 'Run config differs: ' + variant)
        identity = variant if len(variant) == 1 else 'H'
        for name, on in [('text', identity in 'CEFGH'), ('calib', identity in 'DEFGH'),
                         ('feature', identity in 'GH'), ('logic', identity in 'BFH')]:
            require(config['MODEL']['kcc_' + name] == on, 'Incorrect method toggle: ' + variant + '/' + name)
        require(config['TRAIN']['seed'] == 3351 and config['TRAIN']['max_epoch'] == 10, 'Budget/seed mismatch: ' + variant)
        status, trained, ckpt = read(folder / 'status.json'), read(folder / 'train_complete.json'), read(folder / 'meta/checkpoint.json')
        log = (folder / 'train.log').read_text(encoding='utf-8', errors='replace')
        require(status['state'] == 'complete', 'Run incomplete: ' + variant)
        require(trained['epochs'] == list(range(1, 11)), 'Train ledger incomplete: ' + variant)
        require({int(x) for x in re.findall(r'FULL_EPOCH_FINISHED epoch=(\d+)', log)} == set(range(1, 11)), 'Log epochs incomplete: ' + variant)
        require(trained['best_sha256'] == ckpt['sha256'] == status['checkpoint_sha256'], 'Checkpoint identity mismatch: ' + variant)
        labels = read(folder / 'meta/labels.json')
        require(labels == base_labels, 'Primitive ordering changed: ' + variant)
        durations[variant] = sum(float(x) for x in re.findall(r'FULL_EPOCH_FINISHED epoch=\d+ seconds=([\d.]+)', log)) / 60
        if config['MODEL']['kcc_feature']:
            smoke = read(folder / 'smoke.json')
            require(all(smoke[k] is True for k in ('passed', 'same_native_initialization', 'cpu_cuda_rng_preserved',
                                                 'gain0_inference_identity', 'gain0_task_loss_identity')), 'Smoke failed: ' + variant)
            require(smoke['optimizer_steps'] == 0 and smoke['test_images_read'] == 0, 'Smoke budget/leakage changed: ' + variant)
            require(all(np.isfinite(v) and v > 0 for v in smoke['gradients'].values()), 'Smoke gradient missing: ' + variant)
            require('FULL_GRADIENT epoch=10' in log, 'Final feature gradients missing: ' + variant)
        if config['MODEL']['kcc_logic']:
            require('BF_LOGIC epoch=10' in log, 'Final logic activity missing: ' + variant)
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                dest = folder / ('eval_' + phase + '_' + setting)
                meta = read(dest / 'metrics.json')
                truth = np.load(dest / 'labels.npz', allow_pickle=False)
                ds = dataset(labels, split, phase, setting)
                require(read(dest / 'candidates.json') == [list(p) for p in ds.pairs], 'Candidate ordering wrong: ' + variant + '/' + phase + setting)
                require(meta['checkpoint_sha256'] == ckpt['sha256'], 'Evaluation weight mismatch: ' + variant + '/' + phase + setting)
                require(meta['config_sha256'] == sha(dest / 'config_eval_effective.yml'), 'Eval config hash mismatch: ' + variant + '/' + phase + setting)
                require(meta['score_shape'] == [3214 if phase == 'val' else 2914, len(ds.pairs)], 'Score shape differs')
                require((dest / 'scores.dat').stat().st_size == int(np.prod(meta['score_shape'])) * 4, 'Score byte size differs')
                key = phase + setting
                actual_truth = {k: truth[k] for k in ('attr', 'obj', 'pair')}
                if key not in base_truth:
                    base_truth[key] = (actual_truth, read(dest / 'image_ids.json'))
                else:
                    require(all(np.array_equal(v, base_truth[key][0][k]) for k, v in actual_truth.items()), 'Truth order differs: ' + variant + '/' + key)
                    require(read(dest / 'image_ids.json') == base_truth[key][1], 'Image order differs: ' + variant + '/' + key)
                matrix = np.memmap(dest / 'scores.dat', mode='r', dtype=np.float32, shape=tuple(meta['score_shape']))
                require(np.isfinite(matrix).all(), 'Nonfinite scores: ' + variant + '/' + key)
                recalculated = evaluate(matrix, truth, ds, module.Evaluator)
                difference = max(abs(recalculated[k] - float(meta['metrics_raw'][k])) for k in FIELDS)
                require(difference < 2e-6, 'Native metric replay mismatch: ' + variant + '/' + key + '/' + str(difference))
                replay.append({'variant': variant, 'phase': phase, 'setting': setting, 'max_abs_difference': difference})
                rows.append({'variant': variant, 'phase': phase, 'setting': setting,
                             'best_epoch': meta['best_epoch'], **{k + '_pct': 100 * recalculated[k] for k in FIELDS}})
                truth.close()
        print('AUDITED', variant, flush=True)
    # Diagnostic readout changes on VAL only. No model retraining, no new TEST tuning.
    prior = torch.load(root / 'cache_shuffle/K_oa_prior_centered.pt', weights_only=True).numpy()
    grid_rows = []
    for variant in ('D', 'E', 'G', 'H'):
        for setting in ('cw', 'ow'):
            dest = root / 'runs' / variant / ('eval_val_' + setting)
            meta = read(dest / 'metrics.json')
            truth = np.load(dest / 'labels.npz', allow_pickle=False)
            ds = dataset(base_labels, split, 'val', setting)
            pi = prior[[ds.obj2idx[o] for a, o in ds.pairs], [ds.attr2idx[a] for a, o in ds.pairs]]
            matrix = np.memmap(dest / 'scores.dat', mode='r', dtype=np.float32, shape=tuple(meta['score_shape']))
            no_prior = np.asarray(matrix) - np.float32(.1) * pi[None, :]
            for strength in (0, .005, .01, .02, .05, .1):
                current = np.asarray(matrix) if strength == .1 else no_prior + np.float32(strength) * pi[None, :]
                result = evaluate(current, truth, ds, module.Evaluator)
                grid_rows.append({'variant': variant, 'setting': setting, 'lambda_inference': strength,
                                  **{k + '_pct': 100 * result[k] for k in FIELDS}})
            truth.close()
        print('VAL_PRIOR_DIAGNOSED', variant, flush=True)
    result = {'passed': not errors, 'checks': checks, 'errors': errors, 'zip_sha256': sha(archive),
              'archive_members_verified': len(manifest['files_sha256']), 'native_metric_replays': replay,
              'replayed_rows': rows, 'training_minutes_per_run': durations, 'val_only_fixed_weight_prior_grid': grid_rows,
              'boundaries': ['No GPU training performed locally', 'No model weights in result ZIP; weight identity metadata checked, not actual weights',
                             'Prior grid changes readout only; cannot replace freshly trained ablations', 'No new TEST lambda grid or TEST hyperparameter selection']}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'checks': checks, 'errors': errors, 'member_count': result['archive_members_verified'],
                      'metric_replays': len(replay), 'max_replay_diff': max(x['max_abs_difference'] for x in replay),
                      'training_minutes': durations}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
