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
from report_revision import summarize

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
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(config['TRAIN']['seed']),
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
        if (config['MODEL']['kcc_feature'] or config['MODEL']['kcc_logic'] or config['MODEL'].get('formula_joint', False)) and not (run / 'smoke.json').exists():
            smoke_log = run / ('smoke_' + now().strftime('%Y%m%dT%H%M%S') + '.log')
            log_command([sys.executable, '-u', 'tests/smoke_formula.py', '--cfg', cfg_path, '--out', run / 'smoke.json'], source, env, smoke_log)
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
            if epochs != set(range(1, int(config['TRAIN']['max_epoch']) + 1)) or not best.is_file():
                raise ValueError('Training did not finish all protocol epochs / best checkpoint missing')
            if config['MODEL']['kcc_feature'] and 'FULL_GRADIENT epoch={}'.format(config['TRAIN']['max_epoch']) not in log:
                raise ValueError('No final-epoch feature gradient evidence')
            if config['MODEL']['kcc_logic'] and 'BF_LOGIC epoch={}'.format(config['TRAIN']['max_epoch']) not in log:
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
                                    'checkpoint_sha256': sha(best), 'epochs': int(config['TRAIN']['max_epoch'])})
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

