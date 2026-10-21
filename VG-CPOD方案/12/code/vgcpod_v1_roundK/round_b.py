"""B1/B2/B3: frozen CLIP cache, three independent GPU jobs, A5/A6 paired summaries."""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ('B1', 'B2', 'B3')
SETTINGS = ('val_cw', 'val_ow', 'test_cw', 'test_ow')
METRICS = ('AUC_native', 'AUC_exact', 'HM_at_val_bias', 'gt_object_attr_acc',
           'gt_object_attr_acc_unseen', 'support_mass_unseen')
COLUMNS = tuple(setting + '_' + key for setting in SETTINGS for key in METRICS)
DEFAULT_OUT = Path('/data/xhw/vgcpod_v1_runs/roundB_20261019')
DEFAULT_REFERENCE = Path('/data/xhw/vgcpod_v1_runs/roundA_20261019')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(str(temp), str(path))


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_columns(metrics):
    return {setting + '_' + key: metrics[setting][key] for setting in SETTINGS for key in METRICS}


def aggregate(rows, label='case'):
    result = []
    for name in sorted({r[label] for r in rows}):
        group = sorted([r for r in rows if r[label] == name], key=lambda x: x['seed'])
        row = {label: name, 'n_seeds': len(group), 'seeds': [r['seed'] for r in group]}
        for key in COLUMNS:
            values = [r[key] for r in group]
            row[key + '_mean'] = statistics.mean(values)
            row[key + '_sd'] = statistics.stdev(values) if len(values) > 1 else None
        result.append(row)
    return result


def references(reference_out):
    """Scalar comparisons work from the packaged A-group snapshot as well."""
    result = {(r['case'], r['seed']): r for r in read(ROOT / 'references/round_a_confirm.json')['runs']}
    for case in ('A1', 'A5', 'A6'):
        for path in sorted((Path(reference_out) / case).glob('ut-zap50k/V*/seed*/metrics.json')):
            protocol = read(path.parent / 'protocol.json')
            saved = read(path)
            result[(case, protocol['seed'])] = dict(case=case, seed=protocol['seed'],
                config=protocol['config'], feature_manifest_sha256=protocol['feature_manifest_sha256'],
                selected_epoch=saved['selected_epoch'], metrics=saved['metrics'])
    return result


def export_pair_gains(out, runs, reference_out):
    """Compare saved predictions to the corresponding A5/A6 files; no model inference."""
    import numpy as np
    rows = []
    for (case, seed), run in sorted(runs.items()):
        protocol = read(run / 'protocol.json')
        cached_manifest = Path(protocol['cache_path_at_run']) / 'features_manifest.json'
        labels = read(cached_manifest)['labels'] if cached_manifest.is_file() else read(ROOT / 'assets/labels.json')
        for ref_case, variant in [('A5', 'V1'), ('A6', 'V0')]:
            ref = Path(reference_out) / ref_case / 'ut-zap50k' / variant / ('seed%d' % seed)
            if not (ref / 'protocol.json').exists():
                continue
            if read(ref / 'protocol.json')['feature_manifest_sha256'] != protocol['feature_manifest_sha256']:
                continue
            for setting in SETTINGS:
                left, right = run / 'predictions' / (setting + '.npz'), ref / 'predictions' / (setting + '.npz')
                if not left.exists() or not right.exists():
                    continue
                with np.load(left, allow_pickle=False) as l, np.load(right, allow_pickle=False) as r:
                    for key in ('image_ids', 'candidate_pairs', 'attribute_truth', 'object_truth'):
                        if not np.array_equal(l[key], r[key]):
                            raise ValueError('Prediction axes differ: ' + str(left) + ' vs ' + str(right))
                    a, o = l['attribute_truth'], l['object_truth']
                    lc = l['conditional_logits_gt_object'].argmax(1) == a
                    rc = r['conditional_logits_gt_object'].argmax(1) == a
                    lp, rp = l['log_probabilities'].argmax(1), r['log_probabilities'].argmax(1)
                    candidate_pairs = l['candidate_pairs']
                    # Candidate pairs are strings; truths index the cache's primitive axes.
                    for obj, attr in sorted(set(zip(o.tolist(), a.tolist()))):
                        mask = (o == obj) & (a == attr)
                        true_pair = np.array([labels['attrs'][attr], labels['objs'][obj]])
                        lok = np.all(candidate_pairs[lp[mask]] == true_pair, axis=1)
                        rok = np.all(candidate_pairs[rp[mask]] == true_pair, axis=1)
                        rows.append(dict(comparison=case + '-' + ref_case, seed=seed, setting=setting,
                            attribute=labels['attrs'][attr], object=labels['objs'][obj], images=int(mask.sum()),
                            pair_gain_unbiased_pct=float((lok.mean() - rok.mean()) * 100),
                            gt_object_attr_gain_pct=float((lc[mask].mean() - rc[mask].mean()) * 100),
                            conditional_correct_delta=int(lc[mask].sum()) - int(rc[mask].sum())))
    csv_write(Path(out) / 'per_pair_gains.csv', rows)
    return len(rows)


def summarize(out, reference_out=DEFAULT_REFERENCE, pair_gains=True):
    out = Path(out)
    rows, runs = [], {}
    for case in CASES:
        for path in sorted((out / case).glob('ut-zap50k/V*/seed*/metrics.json')):
            saved = read(path)
            protocol = read(path.parent / 'protocol.json')
            cfg = protocol['config']
            history = read(path.parent / 'history.json')
            chosen = history[saved['selected_epoch'] - 1]
            row = dict(case=case, seed=protocol['seed'], variant=cfg['variant'],
                readout_feature=cfg['readout_feature'], readout_alpha=cfg['readout_alpha'],
                gate_input=cfg['gate_input'], gate_eta=cfg['gate_eta'], beta=cfg['task_aux_beta'],
                cp_lr=cfg['lr'] * cfg['cp_lr_scale'], selected_epoch=saved['selected_epoch'],
                completed_epochs=len(history), feature_manifest_sha256=protocol['feature_manifest_sha256'])
            row.update(metric_columns(saved['metrics']))
            row.update(train_delta_rms=chosen['train']['centered_delta_rms'],
                       train_clip_fraction=chosen['train']['clipped_step_fraction'],
                       train_gamma_image_variance=chosen['train']['gamma_image_variance'],
                       train_gamma_abs_deviation=chosen['train']['gamma_abs_deviation'])
            rows.append(row)
            runs[(case, row['seed'])] = path.parent
    if not rows:
        raise FileNotFoundError('No completed B-group runs under ' + str(out))
    ref = references(reference_out)
    deltas = []
    for row in rows:
        for ref_case in ('A5', 'A6'):
            old = ref.get((ref_case, row['seed']))
            if old and old['feature_manifest_sha256'] == row['feature_manifest_sha256']:
                values = metric_columns(old['metrics'])
                deltas.append(dict(comparison=row['case'] + '-' + ref_case, seed=row['seed'],
                                   **{key: row[key] - values[key] for key in COLUMNS}))
    for high, low in [('B3', 'B2'), ('B1', 'B2')]:
        for row in [r for r in rows if r['case'] == high]:
            other = next((r for r in rows if r['case'] == low and r['seed'] == row['seed'] and
                          r['feature_manifest_sha256'] == row['feature_manifest_sha256']), None)
            if other:
                deltas.append(dict(comparison=high + '-' + low, seed=row['seed'],
                                   **{key: row[key] - other[key] for key in COLUMNS}))
    by_case = aggregate(rows)
    by_comparison = aggregate(deltas, 'comparison') if deltas else []
    pair_count = export_pair_gains(out, runs, reference_out) if pair_gains else 0
    csv_write(out / 'round_b_summary.csv', rows)
    csv_write(out / 'round_b_mean_sd.csv', by_case)
    csv_write(out / 'round_b_paired_deltas.csv', deltas)
    csv_write(out / 'round_b_paired_mean_sd.csv', by_comparison)
    ranking = sorted(by_case, key=lambda r: (-r['val_cw_AUC_native_mean'], r['case']))
    dump(out / 'round_b_summary.json', dict(runs=rows, by_case=by_case, paired_deltas=deltas,
        paired_mean_sd=by_comparison, val_cw_ranking=ranking, per_pair_rows=pair_count,
        selection_rule='epoch: first strict maximum VAL CW AUC_native; case ranking: mean VAL CW across available seeds'))
    print('SUMMARY', out / 'round_b_summary.csv', flush=True)
    for row in rows:
        print(row['case'], 'seed', row['seed'], 'VAL_CW', round(row['val_cw_AUC_native'], 4),
              'VAL_OW', round(row['val_ow_AUC_native'], 4), 'TEST_CW', round(row['test_cw_AUC_native'], 4),
              'TEST_OW', round(row['test_ow_AUC_native'], 4),
              'TEST_unseen_attr', round(row['test_ow_gt_object_attr_acc_unseen'], 4), flush=True)
    return rows


def validate_completed(run, cfg, seed, cache):
    from vgcpod.common import sha, source_hashes
    protocol, result = read(run / 'protocol.json'), read(run / 'metrics.json')
    return (protocol['config'] == cfg and protocol['seed'] == seed and
            protocol['source_sha256'] == source_hashes() and
            protocol['feature_manifest_sha256'] == sha(cache / 'features_manifest.json') and
            read(run / 'history.json')[-1]['epoch'] == cfg['epochs'] and
            result['checkpoint_sha256'] == sha(run / 'checkpoints/best.pt'))


def worker(gpu, jobs, args):
    events = []
    for case, seed in jobs:
        config = ROOT / 'configs/round_b' / (case + '.json')
        cfg = read(config)
        run = args.out / case / cfg['dataset'] / cfg['variant'] / ('seed%d' % seed)
        log = args.out / 'logs' / ('%s_seed%d.log' % (case, seed))
        if (run / 'metrics.json').exists():
            if validate_completed(run, cfg, seed, args.cache):
                print('ALREADY_COMPLETE', case, seed, flush=True)
                events.append(dict(case=case, seed=seed, gpu=gpu, status='already_complete'))
                continue
            raise ValueError('Existing run has different configuration/source: ' + str(run))
        command = [sys.executable, '-u', str(ROOT / 'run.py'), 'train', '--cache', str(args.cache),
                   '--out', str(args.out / case), '--config', str(config), '--seed', str(seed), '--device', args.device]
        if (run / 'protocol.json').exists():
            command.append('--resume')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED='0',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        print('START', case, seed, 'GPU', gpu, 'log', log, flush=True)
        with log.open('a', encoding='utf-8') as f:
            f.write('\nROUND_B_START ' + json.dumps(command) + '\n')
            f.flush()
            completed = subprocess.run(command, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        status = 'complete' if completed.returncode == 0 else 'failed'
        events.append(dict(case=case, seed=seed, gpu=gpu, status=status,
                           exit_code=completed.returncode, log=str(log)))
        print(status.upper(), case, seed, 'GPU', gpu, flush=True)
    return events


def launch(args):
    args.cache, args.out = args.cache.resolve(), args.out.resolve()
    if not (args.cache / 'features_manifest.json').is_file():
        raise FileNotFoundError('Feature cache absent: ' + str(args.cache))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'logs').mkdir(exist_ok=True)
    with (args.out / '.round_b.lock').open('a') as lock:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [(case, seed) for seed in args.seeds for case in args.cases]
        queues = [jobs[i::len(args.gpus)] for i in range(len(args.gpus))]
        events = []
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = {pool.submit(worker, gpu, queue, args): gpu
                       for gpu, queue in zip(args.gpus, queues) if queue}
            for future in as_completed(futures):
                try:
                    events += future.result()
                except Exception as e:
                    events.append(dict(gpu=futures[future], status='failed', error=str(e)))
                dump(args.out / 'round_b_status.json', events)
        if any(args.out.glob('B*/ut-zap50k/V*/seed*/metrics.json')):
            summarize(args.out, args.reference_out)
        if any(e['status'] == 'failed' for e in events):
            raise SystemExit('ROUND_B_FAILED: see round_b_status.json and logs')
        print('ROUND_B_COMPLETE', args.out, 'jobs', len(jobs), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('run', 'summarize'):
        p = sub.add_parser(command)
        p.add_argument('--out', type=Path, default=DEFAULT_OUT)
        p.add_argument('--reference-out', type=Path, default=DEFAULT_REFERENCE)
        if command == 'run':
            p.add_argument('--cache', type=Path, default=Path('/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'))
            p.add_argument('--gpus', nargs='+', default=['1', '2', '3'])
            p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
            p.add_argument('--seeds', nargs='+', type=int, default=[3351])
            p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.command == 'run':
        if any(len(set(getattr(args, key))) != len(getattr(args, key)) for key in ('gpus', 'cases', 'seeds')):
            parser.error('GPU IDs, cases, and seeds must each be unique')
        launch(args)
    else:
        summarize(args.out, args.reference_out)


if __name__ == '__main__':
    main()
