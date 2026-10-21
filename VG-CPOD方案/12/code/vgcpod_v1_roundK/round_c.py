"""C1/C2 and an original-source B3 repeat on three independent GPUs."""
import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import round_b as shared

ROOT = Path(__file__).resolve().parent
CASES = ('C1', 'C2')
DEFAULT_OUT = Path('/data/xhw/vgcpod_v1_runs/roundC_20261020')
DEFAULT_A_OUT = Path('/data/xhw/vgcpod_v1_runs/roundA_20261019')
DEFAULT_B_OUT = Path('/data/xhw/vgcpod_v1_runs/roundB_20261019')
DEFAULT_B_ROOT = Path('/data/xhw/vgcpod_project/vgcpod_v1_roundB')
PRINT_LOCK = threading.Lock()


def announce(*items):
    with PRINT_LOCK:
        print(' '.join(map(str, items)), flush=True)


def collect(root, cases, source_group):
    rows, records = [], {}
    for case in cases:
        for path in sorted((Path(root) / case).glob('ut-zap50k/V*/seed*/metrics.json')):
            saved = shared.read(path)
            protocol = shared.read(path.parent / 'protocol.json')
            cfg = protocol['config']
            history = shared.read(path.parent / 'history.json')
            selected = history[saved['selected_epoch'] - 1]['train']
            row = dict(case=case, seed=protocol['seed'], source_group=source_group,
                run_dir=str(path.parent), variant=cfg['variant'], rho=cfg['rho'],
                readout_feature=cfg.get('readout_feature', 'global'), gate_input=cfg.get('gate_input', 'none'),
                gate_eta=cfg.get('gate_eta', 0.), gate_center_modes=cfg.get('gate_center_modes', False),
                beta=cfg['task_aux_beta'], cp_lr=cfg['lr']*cfg['cp_lr_scale'],
                selected_epoch=saved['selected_epoch'], completed_epochs=len(history),
                feature_manifest_sha256=protocol['feature_manifest_sha256'])
            row.update(shared.metric_columns(saved['metrics']))
            row.update(train_delta_rms=selected['centered_delta_rms'],
                train_clip_fraction=selected['clipped_step_fraction'], train_gamma_mean=selected['gamma_mean'],
                train_gamma_image_variance=selected.get('gamma_image_variance'),
                train_gamma_abs_deviation=selected.get('gamma_abs_deviation'),
                train_gamma_mode_mean_deviation=selected.get('gamma_mode_mean_deviation'))
            rows.append(row)
            records[(case, row['seed'])] = row
    return rows, records


def reference_records(a_out):
    raw = shared.references(a_out)
    for record in shared.read(ROOT / 'references/round_b_pilot.json')['runs']:
        raw[(record['case'], record['seed'])] = record
    refs = {}
    for key, record in raw.items():
        refs[key] = dict(case=record['case'], seed=record['seed'],
            feature_manifest_sha256=record['feature_manifest_sha256'],
            **shared.metric_columns(record['metrics']))
        variant = record['config']['variant']
        path = Path(a_out) / record['case'] / 'ut-zap50k' / variant / ('seed%d' % record['seed'])
        if record['case'].startswith('A') and (path / 'metrics.json').exists():
            refs[key]['run_dir'] = str(path)
    return refs


def export_pair_gains(out, pairs):
    """Read saved predictions for every scalar comparison that has both exports."""
    import numpy as np
    rows = []
    for left, right in pairs:
        if 'run_dir' not in left or 'run_dir' not in right:
            continue
        run = Path(left['run_dir'])
        manifest = Path(shared.read(run / 'protocol.json')['cache_path_at_run']) / 'features_manifest.json'
        labels = shared.read(manifest)['labels'] if manifest.exists() else shared.read(ROOT / 'assets/labels.json')
        for setting in shared.SETTINGS:
            lp = run / 'predictions' / (setting + '.npz')
            rp = Path(right['run_dir']) / 'predictions' / (setting + '.npz')
            if not lp.exists() or not rp.exists():
                continue
            with np.load(lp, allow_pickle=False) as l, np.load(rp, allow_pickle=False) as r:
                for key in ('image_ids', 'candidate_pairs', 'attribute_truth', 'object_truth'):
                    if not np.array_equal(l[key], r[key]):
                        raise ValueError('Prediction axes differ: ' + str(lp) + ' vs ' + str(rp))
                a, o = l['attribute_truth'], l['object_truth']
                lc = l['conditional_logits_gt_object'].argmax(-1) == a
                rc = r['conditional_logits_gt_object'].argmax(-1) == a
                li, ri = l['log_probabilities'].argmax(-1), r['log_probabilities'].argmax(-1)
                for obj, attr in sorted(set(zip(o.tolist(), a.tolist()))):
                    mask = (o == obj) & (a == attr)
                    pair = np.array([labels['attrs'][attr], labels['objs'][obj]])
                    lok = np.all(l['candidate_pairs'][li[mask]] == pair, axis=-1)
                    rok = np.all(r['candidate_pairs'][ri[mask]] == pair, axis=-1)
                    rows.append(dict(comparison=left['case']+'-'+right['case'], seed=left['seed'],
                        setting=setting, attribute=pair[0], object=pair[1], images=int(mask.sum()),
                        pair_gain_unbiased_pct=float((lok.mean()-rok.mean())*100),
                        gt_object_attr_gain_pct=float((lc[mask].mean()-rc[mask].mean())*100),
                        conditional_correct_delta=int(lc[mask].sum())-int(rc[mask].sum())))
    shared.csv_write(Path(out) / 'per_pair_gains.csv', rows)
    return len(rows)


def summarize(out, a_out=DEFAULT_A_OUT, b_out=DEFAULT_B_OUT):
    out = Path(out)
    rows, records = collect(out, CASES, 'roundC')
    b_rows, b_records = collect(b_out, ('B3',), 'roundB')
    rows += b_rows
    if not rows:
        raise FileNotFoundError('No completed C1/C2/B3 metrics')
    out.mkdir(parents=True, exist_ok=True)
    refs = reference_records(a_out)
    refs.update(b_records)
    refs.update(records)
    deltas, pairs = [], []
    for row in rows:
        reference_names = ('A5', 'A6', 'B3') if row['case'] in CASES else ('A5', 'A6')
        if row['case'] == 'C2':
            reference_names += ('C1',)
        for name in reference_names:
            other = refs.get((name, row['seed']))
            if other and other['feature_manifest_sha256'] == row['feature_manifest_sha256']:
                deltas.append(dict(comparison=row['case']+'-'+name, seed=row['seed'],
                    **{key:row[key]-other[key] for key in shared.COLUMNS}))
                pairs.append((row, other))
    grouped = shared.aggregate(rows)
    paired = shared.aggregate(deltas, 'comparison') if deltas else []
    pair_count = export_pair_gains(out, pairs)
    shared.csv_write(out / 'round_c_summary.csv', rows)
    shared.csv_write(out / 'round_c_mean_sd.csv', grouped)
    shared.csv_write(out / 'round_c_paired_deltas.csv', deltas)
    shared.csv_write(out / 'round_c_paired_mean_sd.csv', paired)
    ranking = sorted([r for r in grouped if r['case'] in CASES], key=lambda r:(-r['val_cw_AUC_native_mean'],r['case']))
    shared.dump(out / 'round_c_summary.json', dict(runs=rows, by_case=grouped, paired_deltas=deltas,
        paired_mean_sd=paired, c_val_cw_ranking=ranking, per_pair_rows=pair_count,
        selection_rule='epoch: first strict maximum VAL CW AUC_native; C cases: mean VAL CW across available seeds; deltas use matching seeds'))
    announce('SUMMARY', out / 'round_c_summary.csv')
    for row in rows:
        announce(row['case'], 'seed', row['seed'], 'VAL_CW', round(row['val_cw_AUC_native'],4),
            'VAL_OW', round(row['val_ow_AUC_native'],4), 'TEST_CW', round(row['test_cw_AUC_native'],4),
            'TEST_OW', round(row['test_ow_AUC_native'],4),
            'TEST_unseen_attr', round(row['test_ow_gt_object_attr_acc_unseen'],4))
    return rows


def environment(gpu=None):
    env = dict(os.environ, PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def worker_c(gpu, jobs, args):
    events = []
    for case, seed in jobs:
        config = ROOT / 'configs/round_c' / (case + '.json')
        cfg = shared.read(config)
        run = args.out / case / cfg['dataset'] / cfg['variant'] / ('seed%d' % seed)
        log = args.out / 'logs' / ('%s_seed%d.log' % (case, seed))
        if (run / 'metrics.json').exists():
            if shared.validate_completed(run, cfg, seed, args.cache):
                announce('ALREADY_COMPLETE', case, seed, 'GPU', gpu)
                events.append(dict(case=case, seed=seed, gpu=gpu, status='already_complete'))
                continue
            raise ValueError('Existing run has different configuration/source: ' + str(run))
        command = [sys.executable, '-u', str(ROOT / 'run.py'), 'train', '--cache', str(args.cache),
            '--out', str(args.out / case), '--config', str(config), '--seed', str(seed), '--device', args.device]
        if (run / 'protocol.json').exists():
            command.append('--resume')
        announce('START', case, seed, 'GPU', gpu, 'log', log)
        with log.open('a', encoding='utf-8') as f:
            f.write('\nROUND_C_START ' + ' '.join(command) + '\n')
            f.flush()
            process = subprocess.run(command, cwd=str(ROOT), env=environment(gpu), stdout=f, stderr=subprocess.STDOUT)
        status = 'complete' if process.returncode == 0 else 'failed'
        events.append(dict(case=case, seed=seed, gpu=gpu, status=status, exit_code=process.returncode, log=str(log)))
        announce(status.upper(), case, seed, 'GPU', gpu)
    return events


def worker_b3(gpu, args):
    """Delegate to the unchanged B launcher, which owns its original output lock."""
    log = args.out / 'logs' / ('B3_seed%d.log' % args.b3_seed)
    command = [sys.executable, '-u', str(args.b_root / 'round_b.py'), 'run', '--cache', str(args.cache),
        '--out', str(args.b_out), '--reference-out', str(args.a_out), '--gpus', str(gpu),
        '--cases', 'B3', '--seeds', str(args.b3_seed), '--device', args.device]
    announce('START', 'B3', args.b3_seed, 'GPU', gpu, 'original_source', args.b_root)
    with log.open('a', encoding='utf-8') as f:
        process = subprocess.run(command, cwd=str(args.b_root), env=environment(), stdout=f, stderr=subprocess.STDOUT)
    status = 'complete' if process.returncode == 0 else 'failed'
    if process.returncode == 0:
        event = next((r for r in shared.read(args.b_out / 'round_b_status.json')
                      if r.get('case')=='B3' and r.get('seed')==args.b3_seed), None)
        if event and event['status'] == 'already_complete':
            status = 'already_complete'
    announce(status.upper(), 'B3', args.b3_seed, 'GPU', gpu)
    return [dict(case='B3', seed=args.b3_seed, gpu=gpu, status=status,
                 exit_code=process.returncode, log=str(log), source_root=str(args.b_root), output_root=str(args.b_out))]


def launch(args):
    for key in ('out','cache','a_out','b_out','b_root'):
        setattr(args, key, getattr(args, key).resolve())
    if not (args.cache / 'features_manifest.json').is_file():
        raise FileNotFoundError('Feature cache absent: ' + str(args.cache))
    if not args.skip_b3 and not (args.b_root / 'round_b.py').is_file():
        raise FileNotFoundError('Original B source absent; set --b-root: ' + str(args.b_root))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'logs').mkdir(exist_ok=True)
    with (args.out / '.round_c.lock').open('a') as lock:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c_gpus = args.gpus if args.skip_b3 else args.gpus[:-1]
        jobs = [(case,seed) for seed in args.seeds for case in args.cases]
        queues = [jobs[i::len(c_gpus)] for i in range(len(c_gpus))]
        plan = [dict(case=case,seed=seed,gpu=gpu,source='roundC')
                for gpu,queue in zip(c_gpus,queues) for case,seed in queue]
        if not args.skip_b3:
            plan.append(dict(case='B3',seed=args.b3_seed,gpu=args.gpus[-1],source=str(args.b_root)))
        shared.dump(args.out / 'round_c_jobs.json', plan)
        events = []
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = {pool.submit(worker_c,gpu,queue,args):gpu for gpu,queue in zip(c_gpus,queues) if queue}
            if not args.skip_b3:
                futures[pool.submit(worker_b3,args.gpus[-1],args)] = args.gpus[-1]
            for future in as_completed(futures):
                try:
                    events += future.result()
                except Exception as e:
                    events.append(dict(gpu=futures[future],status='failed',error=str(e)))
                shared.dump(args.out / 'round_c_status.json', events)
        if any(args.out.glob('C*/ut-zap50k/V*/seed*/metrics.json')) or any(args.b_out.glob('B3/ut-zap50k/V*/seed*/metrics.json')):
            summarize(args.out,args.a_out,args.b_out)
        if any(r['status']=='failed' for r in events):
            raise SystemExit('ROUND_C_FAILED: see round_c_status.json and logs')
        announce('ROUND_C_COMPLETE',args.out,'jobs',len(plan))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('run','summarize'):
        p = sub.add_parser(command)
        p.add_argument('--out',type=Path,default=DEFAULT_OUT)
        p.add_argument('--a-out',type=Path,default=DEFAULT_A_OUT)
        p.add_argument('--b-out',type=Path,default=DEFAULT_B_OUT)
        if command=='run':
            p.add_argument('--cache',type=Path,default=Path('/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'))
            p.add_argument('--b-root',type=Path,default=DEFAULT_B_ROOT)
            p.add_argument('--gpus',nargs='+',default=['1','2','3'])
            p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
            p.add_argument('--seeds',nargs='+',type=int,default=[3351])
            p.add_argument('--b3-seed',type=int,default=3352)
            p.add_argument('--skip-b3',action='store_true')
            p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args = parser.parse_args()
    if args.command=='run':
        if any(len(set(getattr(args,key)))!=len(getattr(args,key)) for key in ('gpus','cases','seeds')):
            parser.error('GPU IDs, cases, and seeds must each be unique')
        if not args.skip_b3 and len(args.gpus)<2:
            parser.error('Use at least two GPU IDs with the B3 job, or --skip-b3 for C-only training')
        launch(args)
    else:
        summarize(args.out,args.a_out,args.b_out)


if __name__=='__main__':
    main()
