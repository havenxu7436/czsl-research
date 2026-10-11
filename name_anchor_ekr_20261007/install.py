"""Install an independent name-anchor exploration; never modify UT v2."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Audited patch anchor changed: ' + old[:80])
    return text.replace(old, new, 1)


def install(source, destination, bundle=None):
    bundle = Path(bundle or __file__).resolve()
    if bundle.is_file():
        bundle = bundle.parent
    src, dst = Path(source).resolve(), Path(destination).resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError('Use a separate destination outside the existing source')
    ref = json.loads((bundle / 'reference.json').read_text(encoding='utf-8'))
    for name, digest in ref['source_code_manifest'].items():
        if sha(src / name) != digest:
            raise ValueError('UT v2 source differs from the reviewed snapshot: ' + name)
    if sha(src.parent / 'prepared.json') != ref['prepared_json_sha256']:
        raise ValueError('UT v2 prepared.json differs from the reviewed snapshot')
    if dst.exists():
        identity = dst / 'name_install.json'
        if identity.is_file():
            old = json.loads(identity.read_text(encoding='utf-8'))
            if old['source'] == str(src) and old['bundle_sha256'] == sha(bundle / 'SHA256SUMS'):
                for name, digest in old['installed_files_sha256'].items():
                    if sha(dst / name) != digest:
                        raise ValueError('Installed exploration source changed: ' + name)
                print('NAME INSTALL ALREADY VERIFIED', dst)
                return
        raise FileExistsError('Destination exists; preserve it and use a new version')
    # Build replacement text before creating a destination, so unsupported source fails early.
    runtime = (src / 'transition/runtime.py').read_text(encoding='utf-8')
    runtime = replace_once(runtime, 'from .router import Router', 'from .name_router import NameRouter as Router')
    runtime = replace_once(runtime, '        self.base = base_model(ds, cfg, cfg.MODEL.branch_anchor)',
                           '        from .name_plan import validate_config\n'
                           '        validate_config(cfg)\n'
                           '        self.base = base_model(ds, cfg, cfg.MODEL.branch_anchor)')
    runtime = runtime.replace('C anchor', 'A_unit anchor')
    run = (src / 'transition/run.py').read_text(encoding='utf-8')
    run = replace_once(run, 'from .runtime import BranchModel, base_model, load_branch, read_cfg, trainset',
                      'from .name_runtime import BranchModel, base_model, load_branch, read_cfg, trainset')
    start = run.index('    gate_file = getattr(cfg.TRAIN, "semantic_gate_file", "")', run.index('def branch(args):'))
    stop = run.index('    run = Path(cfg.TRAIN.kcc_run_dir)', start)
    run = run[:start] + ('    from .name_plan import validate_config\n'
                        '    validate_config(cfg, require_replay=True)\n') + run[stop:]
    run = run.replace('Frozen C', 'Frozen A_unit').replace('"C0"', '"N0"')
    for name, text in (('name_runtime.py', runtime), ('name_run.py', run)):
        compile(text, name, 'exec')
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', '.git', 'runs', 'experiments', 'outputs', 'output', 'data'))
    for p in (bundle / 'transition').glob('*.py'):
        shutil.copy2(p, dst / 'transition' / p.name)
    for name, text in (('name_runtime.py', runtime), ('name_run.py', run)):
        with (dst / 'transition' / name).open('w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    shutil.copy2(bundle / 'reference.json', dst / 'name_reference.json')
    identity = {'version': 'name_anchor_ekr_exploration_20261007', 'source': str(src),
                'bundle_sha256': sha(bundle / 'SHA256SUMS'),
                'old_research_gate_passed': False,
                'installed_files_sha256': {str(p.relative_to(dst)).replace('\\', '/'): sha(p)
                     for p in sorted(dst.rglob('*.py'))}}
    (dst / 'name_install.json').write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    print('NAME EXPLORATION INSTALLED', dst)
    print('Original code, knowledge, gate and completed experiments were not edited.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--destination', required=True)
    a = p.parse_args()
    install(a.source, a.destination)
