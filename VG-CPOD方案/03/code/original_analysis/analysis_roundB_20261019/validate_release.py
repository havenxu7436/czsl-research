"""Local CPU validation of legacy compatibility and the production subprocess CLI."""
from pathlib import Path
import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'vgcpod_v1_roundB'
sys.path.insert(0, str(PACKAGE))
import torch
from tests.test_math import fixture, synthetic_cache
from vgcpod import engine
from vgcpod.common import read_json

torch.set_num_threads(1)


def main():
    spec = importlib.util.spec_from_file_location('legacy_a_model', ROOT / 'vgcpod_v1_roundA/vgcpod/model.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    compatibility = []
    for variant in legacy.VARIANTS:
        model, cfg, x, y, mask = fixture(variant)
        cfg = dict(cfg, task_aux_beta=.1, cp_lr_scale=.3)
        model.cfg.update(cfg)
        old = legacy.VGCPOD(model.ea, model.eo, model.ec, cfg, 3351)
        optimizers = [engine.make_optimizer(m, cfg) for m in (old, model)]
        for step in range(4):
            results = []
            for m, opt in zip((old, model), optimizers):
                engine.set_learning_rates(opt, cfg, step, 4)
                opt.zero_grad(set_to_none=True)
                out = m(*x)
                loss = m.objective(out, *y, mask)['total']
                loss.backward()
                engine.clip_total_norm([p for p in m.parameters() if p.requires_grad], cfg['gradient_clip'])
                opt.step()
                results.append((out['joint'].detach(), loss.detach()))
            assert all(torch.equal(a, b) for a, b in zip(*results)), variant
            assert all(torch.equal(value, model.state_dict()[key]) for key, value in old.state_dict().items()), variant
        compatibility.append(dict(variant=variant, optimizer_steps=4, bitwise_equal=True))
    (PACKAGE / 'validation/legacy_compatibility.json').write_text(json.dumps(compatibility, indent=2), encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='vgcpod_b_cli_') as temp:
        temp = Path(temp)
        stage = temp / 'package'
        shutil.copytree(PACKAGE, stage, ignore=shutil.ignore_patterns('__pycache__', 'validation', '*.pyc'))
        small = synthetic_cache(temp / 'cache')
        for case in ('B1', 'B2', 'B3'):
            path = stage / 'configs/round_b' / (case + '.json')
            cfg = read_json(path)
            cfg.update(rank=small['rank'], adapter_rank=small['adapter_rank'], workers=0,
                       batch_size=8, eval_batch_size=8, epochs=2)
            path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        command = [sys.executable, '-u', str(stage / 'round_b.py'), 'run', '--cache', str(temp / 'cache'),
                   '--out', str(temp / 'runs'), '--reference-out', str(temp / 'no_external_references'),
                   '--gpus', '1', '2', '3', '--cases', 'B1', 'B2', 'B3', '--seeds', '3351', '--device', 'cpu']
        logs = []
        for invocation in ('first', 'repeat'):
            process = subprocess.run(command, cwd=stage, env=dict(os.environ), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, encoding='utf-8', errors='replace', timeout=120)
            logs.append(invocation + '\n' + process.stdout)
            assert process.returncode == 0, process.stdout
            status = read_json(temp / 'runs/round_b_status.json')
            expected = 'complete' if invocation == 'first' else 'already_complete'
            assert len(status) == 3 and all(row['status'] == expected for row in status), status
            assert 'ROUND_B_COMPLETE' in process.stdout
            if invocation == 'first':
                assert all(('GPU %s' % gpu) in process.stdout for gpu in (1, 2, 3))
        summary = read_json(temp / 'runs/round_b_summary.json')
        assert len(summary['runs']) == 3 and len(summary['by_case']) == 3
        (PACKAGE / 'validation/cli_output.txt').write_text('\n'.join(logs), encoding='utf-8')
    sys.path.insert(0, str(ROOT / 'tmp/round_a_bash_validation'))
    import bashlex
    scripts = []
    for path in sorted((PACKAGE / 'scripts').glob('*.sh')):
        content = path.read_text(encoding='utf-8')
        bashlex.parse(content)
        assert b'\r\n' not in path.read_bytes()
        scripts.append(path.name)
    files = list((PACKAGE / 'vgcpod').glob('*.py')) + [PACKAGE / 'run.py', PACKAGE / 'round_b.py']
    for path in files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 8))
    result = dict(local_validation=True, fixture='synthetic features, CPU; production launcher subprocesses',
                  torch=torch.__version__, python=sys.version.split()[0],
                  legacy_variants_bitwise_equal=len(compatibility), first_cli_jobs=3, repeat_cli_skipped=3,
                  python38_syntax_files=len(files), bash_syntax_scripts=scripts)
    (PACKAGE / 'validation/result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
