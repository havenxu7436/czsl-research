"""A1-A6 independent experiments on three GPUs; cache-only training, no CLIP re-encoding."""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# GPU1: A1 -> A2; GPU2: A3 -> A4; GPU3: A6 -> A5.
ORDER = ('A1', 'A3', 'A6', 'A2', 'A4', 'A5')
CASES = tuple(sorted(ORDER))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(str(temp), str(path))


def summarize(out, references=ROOT / 'references' / 'original_v1.json'):
    rows = []
    for case in CASES:
        for path in sorted((out / case).glob('ut-zap50k/V*/seed*/metrics.json')):
            result = read(path)
            protocol = read(path.parent / 'protocol.json')
            cfg = protocol['config']
            history = read(path.parent / 'history.json')
            chosen = history[result['selected_epoch'] - 1]
            r = dict(case=case, seed=protocol['seed'], variant=cfg['variant'], beta=cfg['task_aux_beta'],
                     cp_lr=cfg['lr'] * cfg['cp_lr_scale'] if cfg['variant'] != 'V0' else None,
                     selected_epoch=result['selected_epoch'], completed_epochs=len(history),
                     feature_manifest_sha256=protocol['feature_manifest_sha256'])
            for setting in ('val_cw', 'val_ow', 'test_cw', 'test_ow'):
                for key in ('AUC_native', 'AUC_exact', 'HM_at_val_bias', 'gt_object_attr_acc',
                            'gt_object_attr_acc_unseen', 'support_mass_unseen'):
                    r[setting + '_' + key] = result['metrics'][setting].get(key)
            r.update(train_delta_rms=chosen['train']['centered_delta_rms'],
                     train_clip_fraction=chosen['train']['clipped_step_fraction'])
            rows.append(r)
    if not rows:
        raise FileNotFoundError('No completed A-group runs under ' + str(out))
    with (out / 'round_a_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ref = read(references) if references and Path(references).is_file() else {'runs': []}
    deltas = []
    for r in rows:
        for variant in ('V1', 'V0'):
            old = next((x for x in ref['runs'] if x['variant'] == variant and x['seed'] == r['seed']
                        and x['feature_manifest_sha256'] == r['feature_manifest_sha256']), None)
            if old:
                dr = dict(case=r['case'], seed=r['seed'], reference='old_' + variant)
                for setting in ('val_cw', 'val_ow', 'test_cw', 'test_ow'):
                    dr[setting + '_delta_auc'] = r[setting + '_AUC_native'] - old['metrics'][setting]['AUC_native']
                deltas.append(dr)
    # Rank completed V1 runs by VAL CW only, never by TEST.
    candidates = sorted([r for r in rows if r['variant'] == 'V1' and r['completed_epochs'] == 10],
                        key=lambda r: (-r['val_cw_AUC_native'], r['case'], r['seed']))
    dump(out / 'round_a_summary.json', {'runs': rows, 'deltas_to_original': deltas,
        'v1_val_cw_ranking': candidates, 'selection_rule': 'best epoch by VAL CW; compare configurations using VAL'})
    print('SUMMARY', out / 'round_a_summary.csv', flush=True)
    for r in sorted(rows, key=lambda r: (r['case'], r['seed'])):
        print(r['case'], 'seed', r['seed'], 'VAL_CW', round(r['val_cw_AUC_native'], 4),
              'VAL_OW', round(r['val_ow_AUC_native'], 4), 'TEST_CW', round(r['test_cw_AUC_native'], 4), flush=True)
    return rows


def validate_completed(run, cfg, seed, cache):
    """A resumed completed job can be skipped only when its identity still matches."""
    from vgcpod.common import sha, source_hashes
    p, result = read(run / 'protocol.json'), read(run / 'metrics.json')
    return (p['config'] == cfg and p['seed'] == seed and p['source_sha256'] == source_hashes()
            and p['feature_manifest_sha256'] == sha(cache / 'features_manifest.json')
            and read(run / 'history.json')[-1]['epoch'] == cfg['epochs']
            and result['checkpoint_sha256'] == sha(run / 'checkpoints/best.pt'))


def worker(gpu, jobs, args):
    events = []
    for case, seed in jobs:
        config = ROOT / 'configs' / 'round_a' / (case + '.json')
        cfg = read(config)
        run = args.out / case / cfg['dataset'] / cfg['variant'] / ('seed%d' % seed)
        log = args.out / 'logs' / ('%s_seed%d.log' % (case, seed))
        if (run / 'metrics.json').exists():
            if validate_completed(run, cfg, seed, args.cache):
                print('ALREADY_COMPLETE', case, seed, flush=True)
                events.append(dict(case=case, seed=seed, gpu=gpu, status='already_complete'))
                continue
            raise ValueError('Completed run differs from current configuration/source: ' + str(run))
        command = [sys.executable, '-u', str(ROOT / 'run.py'), 'train', '--cache', str(args.cache),
                   '--out', str(args.out / case), '--config', str(config), '--variant', cfg['variant'],
                   '--seed', str(seed), '--device', args.device]
        if (run / 'protocol.json').exists():
            command.append('--resume')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED='0',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        print('START', case, seed, 'GPU', gpu, 'log', log, flush=True)
        with log.open('a', encoding='utf-8') as f:
            f.write('\nROUND_A_START ' + json.dumps(command) + '\n')
            f.flush()
            completed = subprocess.run(command, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        status = 'complete' if completed.returncode == 0 else 'failed'
        events.append(dict(case=case, seed=seed, gpu=gpu, status=status, exit_code=completed.returncode, log=str(log)))
        print(status.upper(), case, seed, 'GPU', gpu, flush=True)
    return events


def launch(args):
    args.cache, args.out = args.cache.resolve(), args.out.resolve()
    if not (args.cache / 'features_manifest.json').is_file():
        raise FileNotFoundError('Feature cache absent: ' + str(args.cache))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'logs').mkdir(exist_ok=True)
    # Linux lock releases automatically on exit/crash; blocks duplicate queues for one output root.
    with (args.out / '.round_a.lock').open('a') as lock:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [(case, seed) for seed in args.seeds for case in args.cases]
        queues = [jobs[i::len(args.gpus)] for i in range(len(args.gpus))]
        events = []
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = {pool.submit(worker, gpu, q, args): gpu for gpu, q in zip(args.gpus, queues) if q}
            for future in as_completed(futures):
                try:
                    events += future.result()
                except Exception as e:
                    events.append(dict(gpu=futures[future], status='failed', error=str(e)))
        dump(args.out / 'round_a_status.json', events)
        if any(args.out.glob('A*/ut-zap50k/V*/seed*/metrics.json')):
            summarize(args.out)
        failed = [e for e in events if e['status'] == 'failed']
        if failed:
            raise SystemExit('ROUND_A_FAILED: see round_a_status.json and logs')
        print('ROUND_A_COMPLETE', args.out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--cache', type=Path, default=Path('/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'))
    p.add_argument('--out', type=Path, default=Path('/data/xhw/vgcpod_v1_runs/roundA_20261019'))
    p.add_argument('--gpus', nargs='+', default=['1', '2', '3'])
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(ORDER))
    p.add_argument('--seeds', nargs='+', type=int, default=[3351])
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p = sub.add_parser('summarize')
    p.add_argument('--out', type=Path, default=Path('/data/xhw/vgcpod_v1_runs/roundA_20261019'))
    args = parser.parse_args()
    if args.command == 'run':
        if len(set(args.gpus)) != len(args.gpus) or len(set(args.cases)) != len(args.cases):
            parser.error('GPU IDs and cases must be unique')
        launch(args)
    else:
        summarize(args.out)


if __name__ == '__main__':
    main()
