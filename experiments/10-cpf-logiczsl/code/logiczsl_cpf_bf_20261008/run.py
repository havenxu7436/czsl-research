#!/usr/bin/env python3
"""Run a frozen-protocol native CPF training + VAL CW/OW job, never TEST."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
import yaml
from common import sha, write_json, write_new
from prepare import verify_install


def command_log(command, cwd, env, log):
    with Path(log).open('x', encoding='utf-8') as f:
        process = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()
            rc = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if rc:
        raise subprocess.CalledProcessError(rc, command)


def verify_protocol(work, seed):
    path = work / 'experiments/protocol' / f'seed_{seed}.json'
    protocol = json.loads(path.read_text(encoding='utf-8'))
    source = Path(protocol['source'])
    verify_install(source)
    if sha(source / 'bf_install.json') != protocol['install_sha256']:
        raise ValueError('Install manifest changed')
    for spec in protocol['configs'].values():
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('Frozen config changed: ' + spec['path'])
    for mapping in ('split_sha256', 'metadata_sha256'):
        for name, fingerprint in protocol[mapping].items():
            if sha(name) != fingerprint:
                raise ValueError('Frozen dataset file changed: ' + name)
    sample = yaml.safe_load(Path(protocol['configs']['F']['path']).read_text(encoding='utf-8'))
    bank = Path(sample['MODEL']['bf_logic_bank'])
    cache = Path(sample['MODEL']['kcc_cache'])
    for path, expected in ((bank / 'manifest.json', protocol['logic_bank_manifest_sha256']),
                           (cache / 'cache_manifest.json', protocol['legacy_cache_manifest_sha256']),
                           (cache / 'manifest.json', protocol['legacy_tensor_manifest_sha256'])):
        if sha(path) != expected:
            raise ValueError('Knowledge manifest changed: ' + str(path))
    manifest = json.loads((bank / 'manifest.json').read_text(encoding='utf-8'))
    if sha(bank / 'bank.pt') != manifest['bank_sha256']:
        raise ValueError('Frozen bank changed')
    for variant in ('B', 'F'):
        smoke = work / 'experiments/smoke' / f'{variant}_s{seed}.json'
        evidence = json.loads(smoke.read_text(encoding='utf-8'))
        if (evidence.get('passed') is not True or evidence['config_sha256'] != protocol['configs'][variant]['sha256']
                or evidence['install_sha256'] != protocol['install_sha256']
                or evidence['logic_manifest_sha256'] != protocol['logic_bank_manifest_sha256']):
            raise ValueError('Missing/stale smoke report: ' + str(smoke))
    return path, protocol


def run(work, seed, variant, gpu, python):
    work = Path(work).resolve()
    protocol_path, protocol = verify_protocol(work, seed)
    spec = protocol['configs'][variant]
    c = yaml.safe_load(Path(spec['path']).read_text(encoding='utf-8'))
    run_dir = Path(spec['run'])
    if run_dir.exists():
        raise FileExistsError('Run exists; no silent overwrite/resume: ' + str(run_dir))
    if shutil.disk_usage(work).free < 4 * 1024 ** 3:
        raise ValueError('Less than 4 GiB free; preserve existing results and free space before training')
    if c['TRAIN']['kcc_test_at_end'] is not False or c['TRAIN']['max_epoch'] != 10:
        raise ValueError('Frozen budget/VAL-only protocol violated')
    source = Path(protocol['source'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', PYTHONHASHSEED=str(seed))
    # Verify all legacy cache tensors using the existing, unchanged verifier.
    subprocess.run([python, 'tools/build_knowledge.py', '--output-dir', c['MODEL']['kcc_cache'], '--verify-only'],
                   cwd=source, env=env, check=True)
    run_dir.mkdir(parents=True)
    meta = run_dir / 'meta'
    meta.mkdir()
    write_new(run_dir / 'status.txt', 'running\n')
    shutil.copyfile(spec['path'], run_dir / 'config_requested.yml')
    shutil.copyfile(protocol_path, meta / 'protocol.json')
    shutil.copyfile(Path(c['MODEL']['kcc_cache']) / 'cache_manifest.json', meta / 'cache_manifest.json')
    shutil.copyfile(Path(c['MODEL']['bf_logic_bank']) / 'manifest.json', meta / 'logic_bank_manifest.json')
    write_json(meta / 'dataset_hashes.json', {**protocol['split_sha256'], **protocol['metadata_sha256']})
    files = [p for p in sorted(source.rglob('*')) if p.is_file() and p.suffix in ('.py', '.yml', '.yaml', '.sh')]
    write_new(meta / 'source.sha256', ''.join(sha(p) + '  ' + str(p.relative_to(source)).replace('\\','/') + '\n' for p in files))
    with tarfile.open(meta / 'source_snapshot.tar.gz', 'w:gz') as archive:
        for file in files:
            archive.add(file, arcname=str(file.relative_to(source)))
    write_new(meta / 'pip_freeze.txt', subprocess.check_output([python, '-m', 'pip', 'freeze'], env=env, text=True))
    write_json(meta / 'command_train.json', {'argv': [python, '-u', 'train.py', '--cfg', spec['path']],
                                            'cwd': str(source), 'physical_gpu': str(gpu)})
    write_new(run_dir / 'start_time.txt', datetime.now(timezone.utc).isoformat() + '\n')
    try:
        command_log([python, '-u', 'train.py', '--cfg', spec['path']], source, env, run_dir / 'train.log')
        checkpoint = Path(c['TRAIN']['checkpoint_dir']) / (c['config_name'] + '_' + c['TRAIN']['train_name']) / 'best_auc.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError('Validation-best checkpoint missing')
        if variant in ('B', 'F') and 'BF_LOGIC epoch=' not in (run_dir / 'train.log').read_text(encoding='utf-8'):
            raise ValueError('No evidence of active logic loss in training')
        write_json(meta / 'checkpoint.json', {'path': str(checkpoint), 'sha256': sha(checkpoint)})
        for setting, opened in (('cw', 'False'), ('ow', 'True')):
            command_log([python, 'tools/eval_mvp.py', '--cfg', spec['path'], '--load', checkpoint,
                         '--phase', 'val', '--setting', setting, '--out', run_dir / ('eval_val_' + setting),
                         'EVAL.open_world', opened], source, env, run_dir / ('eval_val_' + setting + '.log'))
        verify_protocol(work, seed)
        (run_dir / 'status.txt').write_text('train_and_val_complete\n', encoding='utf-8')
        write_new(run_dir / 'end_time.txt', datetime.now(timezone.utc).isoformat() + '\n')
        print('BF_TRAIN_VAL_COMPLETE', run_dir)
    except BaseException as error:
        (run_dir / 'status.txt').write_text('failed: ' + str(error) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=Path('/var/tmp/czsl_xhw_bf_20261008'))
    p.add_argument('--seed', type=int, default=3351)
    p.add_argument('--variant', choices=['A_ref', 'B', 'E_ref', 'F'], required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--python', default=sys.executable)
    a = p.parse_args()
    run(a.work, a.seed, a.variant, a.gpu, a.python)
