"""Safe GPU queue, full native training, explicit resume, and real evidence export."""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import yaml
from common import read, sha, write, verify_files, verify_release
from report import summarize

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc)


def log_command(argv, cwd, env, path, append=False):
    mode = 'a' if append else 'x'
    with Path(path).open(mode, encoding='utf-8') as f:
        f.write('\nCOMMAND ' + json.dumps([str(a) for a in argv]) + '\n')
        f.flush()
        process = subprocess.Popen([str(a) for a in argv], cwd=str(cwd), env=env,
                                   stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if rc:
        raise subprocess.CalledProcessError(rc, argv)


def verify_protocol(work):
    protocol = read(work / 'protocol.json')
    verify_files(protocol['frozen_inputs'])
    for job in protocol['jobs']:
        if sha(job['cfg']) != job['cfg_sha256']:
            raise ValueError('Config changed: ' + job['cfg'])
    return protocol


def worker(work, variant, gpu, resume):
    verify_release(HERE)
    protocol = verify_protocol(work)
    job = next(j for j in protocol['jobs'] if j['variant'] == variant)
    run, cfg_path, source = Path(job['run']), Path(job['cfg']), Path(protocol['source'])
    config = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    if run.exists() and not resume:
        raise FileExistsError('Run exists, use explicit --resume ' + str(run))
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'status.json').exists() and read(run / 'status.json').get('state') == 'complete':
        print('ALREADY_COMPLETE', variant)
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED='3351',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    ckpt_dir = Path(config['TRAIN']['checkpoint_dir']) / (config['config_name'] + '_' + config['TRAIN']['train_name'])
    best, last = ckpt_dir / 'best_auc.pth', ckpt_dir / 'last_resume.pth'
    write(run / 'status.json', {'state': 'running', 'gpu': gpu, 'pid': os.getpid(), 'time': now().isoformat()})
    meta = run / 'meta'
    meta.mkdir(exist_ok=True)
    if not (run / 'config_requested.yml').exists():
        shutil.copyfile(cfg_path, run / 'config_requested.yml')
        shutil.copyfile(work / 'protocol.json', meta / 'protocol.json')
        shutil.copyfile(Path(config['MODEL']['kcc_cache']) / 'cache_manifest.json', meta / 'cache_manifest.json')
        shutil.copyfile(Path(config['MODEL']['kcc_cache']) / 'labels.json', meta / 'labels.json')
        shutil.copyfile(Path(config['MODEL']['bf_logic_bank']) / 'manifest.json', meta / 'logic_bank_manifest.json')
        write(meta / 'command_train.json', {'argv': [sys.executable, '-u', 'train.py', '--cfg', str(cfg_path)],
                                            'cwd': str(source), 'physical_gpu': gpu})
    try:
        if config['MODEL']['kcc_feature'] and not (run / 'smoke.json').exists():
            smoke_log = run / ('smoke_' + now().strftime('%Y%m%dT%H%M%S') + '.log')
            log_command([sys.executable, '-u', 'tests/smoke_full.py', '--cfg', cfg_path, '--out', run / 'smoke.json'], source, env, smoke_log)
        train_done = run / 'train_complete.json'
        if not train_done.exists():
            if (run / 'train.log').exists():
                if not resume or not last.is_file():
                    raise ValueError('Interrupted training has no completed-epoch resume checkpoint; preserve this run, use a new work directory')
                env['FULL0601_RESUME'] = str(last)
            log_command([sys.executable, '-u', 'train.py', '--cfg', cfg_path], source, env, run / 'train.log', append=(run / 'train.log').exists())
            log = (run / 'train.log').read_text(encoding='utf-8', errors='replace')
            epochs = {int(e) for e in re.findall(r'FULL_EPOCH_FINISHED epoch=(\d+)', log)}
            for resumed in re.findall(r'FULL_RESUME completed_epoch=(\d+)', log):
                epochs.update(range(1, int(resumed) + 1))
            if epochs != set(range(1, 11)) or not best.is_file():
                raise ValueError('Training did not finish all ten epochs / best checkpoint missing')
            if config['MODEL']['kcc_feature'] and 'FULL_GRADIENT epoch=10' not in log:
                raise ValueError('No final-epoch feature gradient evidence')
            if config['MODEL']['kcc_logic'] and 'BF_LOGIC epoch=10' not in log:
                raise ValueError('No final-epoch active logic evidence')
            write(train_done, {'epochs': sorted(epochs), 'best_sha256': sha(best), 'time': now().isoformat()})
        if sha(best) != read(train_done)['best_sha256']:
            raise ValueError('Best checkpoint changed since completed training')
        write(meta / 'checkpoint.json', {'path': str(best), 'sha256': sha(best)})
        env.pop('FULL0601_RESUME', None)
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                dest = run / ('eval_' + phase + '_' + setting)
                if (dest / 'metrics.json').exists():
                    if read(dest / 'metrics.json')['checkpoint_sha256'] != sha(best):
                        raise ValueError('Stale evaluation ' + str(dest))
                    continue
                if dest.exists():
                    # Keep failed evaluation evidence; retry writes a fresh directory.
                    parked = dest.with_name(dest.name + '_incomplete_' + now().strftime('%Y%m%dT%H%M%S'))
                    dest.rename(parked)
                log_path = run / ('eval_' + phase + '_' + setting + '_' + now().strftime('%Y%m%dT%H%M%S') + '.log')
                log_command([sys.executable, '-u', 'tools/eval_mvp.py', '--cfg', cfg_path, '--load', best,
                             '--phase', phase, '--setting', setting, '--out', dest, '--save-scores',
                             'EVAL.open_world', 'True' if setting == 'ow' else 'False'], source, env, log_path)
        verify_protocol(work)
        write(run / 'status.json', {'state': 'complete', 'gpu': gpu, 'time': now().isoformat(),
                                    'checkpoint_sha256': sha(best), 'epochs': 10})
        print('JOB_COMPLETE', variant, flush=True)
    except BaseException as error:
        write(run / 'status.json', {'state': 'failed', 'gpu': gpu, 'time': now().isoformat(), 'error': str(error)})
        raise


def available_gpus(pool, reserved):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    candidates = []
    for line in output.splitlines():
        index, used, free, util = [int(x.strip()) for x in line.split(',')]
        if (pool is None or index in pool) and index not in reserved and used < 512 and free >= 16000 and util <= 10:
            candidates.append(index)
    return candidates


def uniform_readout_check(work):
    """Analytic fixed-weight centered constant-prior identity, not a training result."""
    import numpy as np
    import torch
    run = work / 'runs/H'
    folder = run / 'eval_val_ow'
    meta = read(folder / 'metrics.json')
    shape = tuple(meta['score_shape'])
    scores = np.memmap(folder / 'scores.dat', mode='r', dtype=np.float32, shape=shape)
    candidates = read(folder / 'candidates.json')
    original = Path(read(work / 'protocol.json')['root']) / 'knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k'
    labels = read(original / 'labels.json')
    obj = [labels['objs'].index(p[1]) for p in candidates]
    attr = [labels['attrs'].index(p[0]) for p in candidates]
    prior = torch.load(original / 'K_oa_prior_centered.pt', weights_only=True).numpy()[obj, attr]
    no_prior = np.asarray(scores) - 0.1 * prior[None, :]
    constant = np.ones((12, 16), dtype=np.float32)
    centered = (constant - constant.mean()) / max(float(constant.std()), 1e-8)
    replacement = no_prior + 0.1 * centered[obj, attr][None, :]
    result = {'constant_centered_max_abs': float(abs(centered).max()),
              'fixed_weight_uniform_vs_prior_off_max_abs': float(abs(replacement - no_prior).max()),
              'fixed_weight_top1_equal': bool(np.array_equal(replacement.argmax(1), no_prior.argmax(1))),
              'kind': 'arithmetic identity on actual H VAL OW scores; not a second forward/evaluator run',
              'fresh_uniform_training': 'H_uniform_prior is a separate full ten-epoch run'}
    if result['fixed_weight_uniform_vs_prior_off_max_abs'] != 0 or not result['fixed_weight_top1_equal']:
        raise ValueError('Constant prior identity failed')
    write(work / 'checks/uniform_readout.json', result)


def export(work, target):
    export_dir = work / 'exports'
    export_dir.mkdir(exist_ok=True)
    archive = export_dir / ('results_' + now().strftime('%Y%m%dT%H%M%S%f') + '.zip')
    fingerprints = {}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(work.rglob('*')):
            rel = path.relative_to(work)
            if not path.is_file() or any(p in ('exports', 'checkpoints', '__pycache__', 'logs') for p in rel.parts) or path.suffix in ('.pth', '.pyc') or path.name.endswith('.pending') or path.name == '.runner.lock':
                continue
            name = str(rel).replace('\\', '/')
            z.write(path, name)
        for info in z.infolist():
            fingerprints[info.filename] = hashlib.sha256(z.read(info.filename)).hexdigest()
        z.writestr('export_manifest.json', json.dumps({'files_sha256': fingerprints,
                    'checkpoints_included': False, 'checkpoint_identity_files_included': True,
                    'warning': 'Intermediate archives may contain live logs; metric/status files are atomic snapshots.'}, indent=2))
    if target.exists():
        # Only replace an earlier export from exactly this work; versioned copy retained.
        with zipfile.ZipFile(target) as z:
            if read(work / 'protocol.json') != json.loads(z.read('protocol.json')):
                raise FileExistsError('Existing export belongs to different protocol; preserve it ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.pending')
    shutil.copyfile(archive, temporary)
    os.replace(temporary, target)
    target.with_suffix(target.suffix + '.sha256').write_text(sha(target) + '  ' + target.name + '\n', encoding='utf-8')
    print('RESULTS_EXPORTED', target, flush=True)


def run(args):
    import fcntl
    verify_release(HERE)
    work = args.work.resolve()
    protocol = verify_protocol(work)
    with (work / '.runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shutil.disk_usage(work).free < 32 * 1024 ** 3 and not args.resume:
            raise ValueError('Need >=32 GiB free for fresh model+optimizer checkpoints. Do not delete old experiments.')
        pool = None if args.gpus == 'auto' else {int(x) for x in args.gpus.split(',')}
        deadline = datetime.fromisoformat(args.deadline).astimezone(timezone.utc)
        running, attempted = {}, set()
        stopped_launching = False
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(work / 'pip_freeze.txt').open('w'), check=True)
        write(work / 'hardware.json', {'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
                                      'python': sys.version, 'gpu_policy': args.gpus, 'deadline': args.deadline})
        print('QUEUE_STARTED jobs={} gpu_pool={} deadline={} max_workers={}'.format(len(protocol['jobs']), args.gpus, args.deadline, args.max_workers), flush=True)
        try:
            while True:
                for variant, entry in list(running.items()):
                    rc = entry['process'].poll()
                    if rc is not None:
                        entry['handle'].close()
                        print('QUEUE_JOB_EXIT', variant, 'rc', rc, flush=True)
                        del running[variant]
                        if rc and not (work / 'runs' / variant / 'status.json').exists():
                            write(work / 'runs' / variant / 'status.json', {'state': 'failed', 'error': 'worker exited rc=' + str(rc)})
                        summarize(work)
                        export(work, args.export)
                completed, pending = set(), []
                failed = []
                for job in protocol['jobs']:
                    status = Path(job['run']) / 'status.json'
                    state = read(status).get('state') if status.exists() else 'not_started'
                    if state == 'complete':
                        completed.add(job['variant'])
                    elif job['variant'] not in attempted and job['variant'] not in running:
                        if state == 'not_started' or args.resume:
                            pending.append(job)
                        else:
                            failed.append(job['variant'])
                if not running and not pending:
                    break
                if now() >= deadline:
                    stopped_launching = True
                # Reserve an estimated 90 min per new job; no forced kill of an active run.
                if (deadline - now()).total_seconds() < args.reserve_job_minutes * 60:
                    stopped_launching = True
                if stopped_launching:
                    if not running:
                        print('LAUNCH_CUTOFF_REACHED incomplete_jobs=', [j['variant'] for j in pending], flush=True)
                        break
                else:
                    outstanding_tier = min([j['tier'] for j in pending] + [e['tier'] for e in running.values()])
                    if outstanding_tier > 0 and not all(j['variant'] in completed for j in protocol['jobs'] if j['tier'] < outstanding_tier):
                        print('PRIORITY_INCOMPLETE: do not start lower-priority experiments', flush=True)
                        if not running:
                            break
                    else:
                        if shutil.disk_usage(work).free < 3 * 1024 ** 3:
                            print('LOW_DISK: preserve existing files; do not start another job', flush=True)
                            if not running:
                                break
                            free = []
                        else:
                            free = available_gpus(pool, {entry['gpu'] for entry in running.values()})
                        for job in [j for j in pending if j['tier'] == outstanding_tier]:
                            if not free or len(running) >= args.max_workers:
                                break
                            gpu = free.pop(0)
                            variant = job['variant']
                            log_dir = work / 'driver_logs'
                            log_dir.mkdir(exist_ok=True)
                            handle = (log_dir / (variant + '_' + now().strftime('%Y%m%dT%H%M%S') + '.log')).open('x')
                            command = [sys.executable, '-u', str(HERE / 'run_suite.py'), '--work', str(work), '--job', variant, '--gpu', str(gpu)]
                            if args.resume:
                                command.append('--resume')
                            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                            running[variant] = {'process': process, 'gpu': gpu, 'handle': handle, 'tier': job['tier']}
                            attempted.add(variant)
                            print('QUEUE_JOB_START', variant, 'physical_gpu', gpu, 'pid', process.pid, flush=True)
                print('QUEUE_STATUS complete={}/{} running={} queued={} free_GiB={:.1f}'.format(
                    len(completed), len(protocol['jobs']), list(running), len(pending), shutil.disk_usage(work).free / 1024**3), flush=True)
                time.sleep(20)
        finally:
            # On interruption stop only child processes belonging to THIS queue.
            for entry in running.values():
                process = entry['process']
                if process.poll() is None:
                    # Worker is a newly created process-group leader; signal only our group.
                    if os.getpgid(process.pid) != process.pid:
                        raise RuntimeError('Unexpected child process-group identity')
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                entry['handle'].close()
            if (work / 'runs/H/status.json').exists() and read(work / 'runs/H/status.json').get('state') == 'complete':
                uniform_readout_check(work)
            core, all_done = summarize(work)
            export(work, args.export)
            print('SUITE_COMPLETE core_complete={} all_planned_complete={} export={}'.format(core, all_done, args.export), flush=True)
        return 0 if all_done else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--gpus', default='auto')
    p.add_argument('--max-workers', type=int, default=4)
    p.add_argument('--deadline', default='2026-10-11T17:00:00+08:00')
    p.add_argument('--reserve-job-minutes', type=float, default=90)
    p.add_argument('--export', type=Path, default=Path('/var/tmp/czsl_0601_20261010.zip'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--job', help=argparse.SUPPRESS)
    p.add_argument('--gpu', type=int, help=argparse.SUPPRESS)
    p.add_argument('--summarize-only', action='store_true')
    a = p.parse_args()
    if a.job:
        worker(a.work.resolve(), a.job, a.gpu, a.resume)
    elif a.summarize_only:
        summarize(a.work.resolve())
        export(a.work.resolve(), a.export)
    else:
        sys.exit(run(a))
