"""Deadline-free, non-overwriting queue for the frozen formula protocol."""
import argparse
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from common import read, write, verify_release
from engine import worker, available_gpus, export, verify_protocol, now
from report_revision import summarize

HERE = Path(__file__).resolve().parent


def reject_live_workers(protocol):
    """A lost tmux/controller must not duplicate surviving training groups."""
    for job in protocol['jobs']:
        path = Path(job['run']) / 'status.json'
        if not path.exists():
            continue
        state = read(path)
        if state.get('state') != 'running' or 'pid' not in state:
            continue
        pid = int(state['pid'])
        try:
            os.killpg(pid, 0)  # read-only process-group existence check
        except ProcessLookupError:
            continue
        except PermissionError:
            raise RuntimeError('Cannot establish that old worker group ended; inspect pid=' + str(pid))
        raise RuntimeError('Old worker/train process group still alive: job={} pgid={}; inspect, do not start another queue'.format(job['variant'], pid))


def run(args):
    import fcntl
    verify_release(HERE)
    work = args.work.resolve()
    protocol = verify_protocol(work)
    seeds = set(protocol['seeds']) if args.seeds == 'all' else {int(s) for s in args.seeds.split(',')}
    if not seeds or not seeds.issubset(set(protocol['seeds'])):
        raise ValueError('Seed filter must be a subset of the frozen protocol')
    jobs = [j for j in protocol['jobs'] if j['seed'] in seeds]
    pool = None if args.gpus == 'auto' else {int(s) for s in args.gpus.split(',')}
    with (work / '.runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reject_live_workers(protocol)
        remaining = sum(not ((Path(j['run']) / 'status.json').exists() and
                             read(Path(j['run']) / 'status.json').get('state') == 'complete') for j in jobs)
        required_gib = 10 + 2 * remaining
        if remaining and shutil.disk_usage(work).free < required_gib * 1024 ** 3:
            raise ValueError('Need >= {} GiB free for {} remaining durable checkpoint jobs; choose another writable work mount, never delete old experiments.'.format(required_gib, remaining))
        with (work / 'pip_freeze.txt').open('w') as handle:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=handle, check=True)
        write(work / 'hardware.json', {'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
                                      'python': sys.version, 'gpu_policy': args.gpus,
                                      'deadline': None, 'selected_seeds': sorted(seeds)})
        running, attempted = {}, set()
        print('FORMULA_QUEUE_STARTED jobs={} epochs={} seeds={} gpus={} no_deadline=True'.format(
            len(jobs), protocol['epochs'], sorted(seeds), args.gpus), flush=True)
        scope_done = False
        try:
            while True:
                for job_id, entry in list(running.items()):
                    rc = entry['process'].poll()
                    if rc is not None:
                        entry['handle'].close()
                        del running[job_id]
                        print('QUEUE_JOB_EXIT', job_id, 'rc', rc, flush=True)
                        if rc and not (work / 'runs' / job_id / 'status.json').exists():
                            write(work / 'runs' / job_id / 'status.json', {'state': 'failed', 'error': 'worker rc=' + str(rc)})
                        summarize(work)
                complete, pending, failed = set(), [], []
                for job in jobs:
                    path = Path(job['run']) / 'status.json'
                    state = read(path).get('state') if path.exists() else 'not_started'
                    if state == 'complete':
                        complete.add(job['variant'])
                    elif job['variant'] not in running:
                        if job['variant'] in attempted:
                            failed.append(job['variant'])
                        elif state == 'not_started' or args.resume:
                            pending.append(job)
                        else:
                            failed.append(job['variant'])
                if failed:
                    # An execution/formula failure is a quality gate, not a
                    # reason to keep launching unrelated expensive experiments.
                    if not running:
                        print('EXECUTION_GATE_FAILED preserve_logs=', failed, flush=True)
                        break
                    pending = []
                if not running and not pending:
                    scope_done = len(complete) == len(jobs)
                    break
                if pending:
                    tier = min([j['tier'] for j in pending] + [e['tier'] for e in running.values()])
                    lower_done = all(j['variant'] in complete for j in jobs if j['tier'] < tier)
                    if tier > 0 and not lower_done:
                        if not running:
                            print('CORE_EXECUTION_INCOMPLETE: controls not launched', flush=True)
                            break
                    elif shutil.disk_usage(work).free < 5 * 1024 ** 3:
                        print('LOW_DISK: preserve files; no new job', flush=True)
                        if not running:
                            break
                    else:
                        free = available_gpus(pool, {e['gpu'] for e in running.values()})
                        for job in [j for j in pending if j['tier'] == tier]:
                            if not free or len(running) >= args.max_workers:
                                break
                            gpu, job_id = free.pop(0), job['variant']
                            log_dir = work / 'driver_logs'
                            log_dir.mkdir(exist_ok=True)
                            handle = (log_dir / (job_id + '_' + now().strftime('%Y%m%dT%H%M%S') + '.log')).open('x')
                            command = [sys.executable, '-u', str(HERE / 'run_suite.py'), '--work', str(work),
                                       '--job', job_id, '--gpu', str(gpu)]
                            if args.resume:
                                command.append('--resume')
                            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                            running[job_id] = {'process': process, 'gpu': gpu, 'handle': handle, 'tier': tier}
                            attempted.add(job_id)
                            print('QUEUE_JOB_START', job_id, 'gpu', gpu, 'pid', process.pid, flush=True)
                print('QUEUE_STATUS selected_complete={}/{} running={} queued={}'.format(
                    len(complete), len(jobs), list(running), len(pending)), flush=True)
                time.sleep(20)
        finally:
            for entry in running.values():
                process = entry['process']
                if process.poll() is None:
                    if os.getpgid(process.pid) != process.pid:
                        raise RuntimeError('Unexpected worker process group')
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                entry['handle'].close()
            core, all_done = summarize(work)
            export(work, args.export)
            print('FORMULA_SUITE_COMPLETE selected_seeds_complete={} core_complete={} all_planned_complete={} export={}'.format(
                scope_done, core, all_done, args.export), flush=True)
        return 0 if scope_done else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--gpus', default='auto')
    p.add_argument('--max-workers', type=int, default=3)
    p.add_argument('--seeds', default='all')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--export', type=Path, default=Path('/var/tmp/czsl_0601_revision_20261013.zip'))
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--job', help=argparse.SUPPRESS)
    p.add_argument('--gpu', type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.job:
        worker(a.work.resolve(), a.job, a.gpu, a.resume)
    elif a.summarize_only:
        verify_protocol(a.work.resolve())
        summarize(a.work.resolve())
        export(a.work.resolve(), a.export)
    else:
        if a.max_workers < 1:
            raise ValueError('max-workers must be positive')
        sys.exit(run(a))
