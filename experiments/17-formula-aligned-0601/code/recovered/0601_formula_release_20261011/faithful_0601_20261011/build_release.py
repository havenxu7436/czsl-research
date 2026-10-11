"""Mechanical, checked build from the previously verified execution source."""
import ast
import shutil
import zipfile
from pathlib import Path
from common import read, sha, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'output/0601_priority_results_20261011/extracted/source'
PREVIOUS = ROOT / 'reproduce_0601_20261010'


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Missing/ambiguous build anchor: ' + old[:100])
    return text.replace(old, new, 1)


def transform_model(text):
    text = once(text, 'class DDM(FullFeatureMixin, KCCCacheMixin, BaselineDDM):',
                'class LegacyDDM(FullFeatureMixin, KCCCacheMixin, BaselineDDM):')
    text += '\n\nfrom .formula_model import FormulaMixin\n\nclass DDM(FormulaMixin, LegacyDDM):\n    pass\n'
    return text


def transform_cache(text):
    text = once(text, 'if getattr(cfg.MODEL, "kcc_score_mode", "legacy") != "legacy":',
                'if getattr(cfg.MODEL, "kcc_score_mode", "legacy") not in ("legacy", "0601_formula"):')
    text = once(text, 'expected = torch.zeros_like(k) if k.std(unbiased=False) < 1e-8 else (k - k.mean()) / k.std(unbiased=False)',
                'expected = (k - k.mean()) / (k.std() + 1e-8)')
    return text.replace('population-standardized K', 'sample-std+epsilon standardized K')


def transform_engine(text):
    text = once(text, "PYTHONHASHSEED='3351'", "PYTHONHASHSEED=str(config['TRAIN']['seed'])")
    text = once(text, "if config['MODEL']['kcc_feature'] and not (run / 'smoke.json').exists():",
                "if (config['MODEL']['kcc_feature'] or config['MODEL']['kcc_logic'] or config['MODEL'].get('formula_joint', False)) and not (run / 'smoke.json').exists():")
    text = text.replace('tests/smoke_full.py', 'tests/smoke_formula.py')
    text = once(text, 'epochs != set(range(1, 11))', "epochs != set(range(1, int(config['TRAIN']['max_epoch']) + 1))")
    text = once(text, "'FULL_GRADIENT epoch=10' not in log", "'FULL_GRADIENT epoch={}'.format(config['TRAIN']['max_epoch']) not in log")
    text = once(text, "'BF_LOGIC epoch=10' not in log", "'BF_LOGIC epoch={}'.format(config['TRAIN']['max_epoch']) not in log")
    text = once(text, "'epochs': 10", "'epochs': int(config['TRAIN']['max_epoch'])")
    text = text.replace('all ten epochs', 'all protocol epochs')
    # The deadline queue is deliberately not shipped as executable dead code.
    text = text[:text.index('\ndef run(args):')]
    return text


def build():
    for folder in ('source', 'assets', 'reference_models'):
        if (HERE / folder).exists():
            raise FileExistsError('Preserve existing build: ' + str(HERE / folder))
    # Validate the exact source snapshot consumed in yesterday's server run.
    expected = read(PREVIOUS / 'release_manifest.json')
    for path in BASE.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            rel = path.relative_to(BASE).as_posix()
            if sha(path) != expected['source/' + rel]:
                raise ValueError('Previously executed source differs: ' + rel)
    shutil.copyfile(PREVIOUS / 'common.py', HERE / 'common.py') if not (HERE / 'common.py').exists() else None
    shutil.copyfile(PREVIOUS / 'prepare.py', HERE / 'legacy_prepare.py')
    engine = transform_engine((PREVIOUS / 'run_suite.py').read_text(encoding='utf-8'))
    (HERE / 'engine.py').write_text(engine, encoding='utf-8', newline='\n')
    shutil.copytree(PREVIOUS / 'assets', HERE / 'assets', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(BASE, HERE / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (HERE / 'reference_models').mkdir()
    shutil.copyfile(BASE / 'models/kcc_cpf.py', HERE / 'reference_models/kcc_cpf.py')
    for name, transform in (('models/kcc_cpf.py', transform_model), ('models/kcc_cache.py', transform_cache)):
        path = HERE / 'source' / name
        path.write_text(transform(path.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
    path = HERE / 'source/models/full_feature.py'
    text = once(path.read_text(encoding='utf-8'), 'torch.manual_seed(603351)',
                "torch.manual_seed(600000 + int(cfg.TRAIN.seed))")
    text = text.replace('The native CPF composition score is retained as a residual base, not removed.',
                        'Legacy residual helpers remain for disabled-feature A/C regression only.\nFormulaMixin consumes head.forward directly and does NOT use these residual helpers.')
    path.write_text(text, encoding='utf-8', newline='\n')
    shutil.copyfile(HERE / 'formula_model.py', HERE / 'source/models/formula_model.py')
    shutil.copyfile(HERE / 'smoke_formula.py', HERE / 'source/tests/smoke_formula.py')
    # All modules and imported entrypoints must parse under server Python3.8.
    for path in HERE.rglob('*.py'):
        if '__pycache__' not in path.parts:
            ast.parse(path.read_text(encoding='utf-8'), feature_version=8)
    print('FORMULA_SOURCE_BUILT', HERE, flush=True)


def package():
    checks = read(HERE / 'local_test.json')
    if not checks['passed']:
        raise ValueError('Do not release a failing test suite')
    manifest = {p.relative_to(HERE).as_posix(): sha(p) for p in sorted(HERE.rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and p.name != 'release_manifest.json'}
    write(HERE / 'release_manifest.json', manifest)
    archive = ROOT / 'output/0601_formula_release_20261011.zip'
    if archive.exists():
        raise FileExistsError('Preserve existing ZIP: ' + str(archive))
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in manifest:
            z.write(HERE / name, HERE.name + '/' + name)
        z.write(HERE / 'release_manifest.json', HERE.name + '/release_manifest.json')
    archive.with_suffix(archive.suffix + '.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='utf-8')
    print('FORMULA_RELEASE_BUILT', archive, 'sha256', sha(archive), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--package', action='store_true')
    package() if p.parse_args().package else build()
