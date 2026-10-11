"""Verify archive files and optionally run CPU tests in a temporary copy."""
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

RELEASE_SHA = '1516d14011c015754adfd54b421999ff3e7d05ee0a257f186e13e29a660c8c49'
DESIGN_SHA = 'ebf696dec79c7788eb2ee02c095ca44828429a7e48225db77d56d9b08125ec0c'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(root, cpu_tests=False):
    root = root.resolve()
    bundle = json.loads((root / 'bundle_manifest.json').read_text(encoding='utf-8-sig'))
    files = bundle['files_sha256']
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path: ' + name)
        if digest(root / relative) != expected:
            raise ValueError('Bundle checksum mismatch: ' + name)
    release = root / 'code/faithful_0601_20261011'
    manifest = json.loads((release / 'release_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        if digest(release / name) != expected:
            raise ValueError('Release checksum mismatch: ' + name)
    archive = root / 'release/0601_formula_release_20261011_r2.zip'
    if digest(archive) != RELEASE_SHA:
        raise ValueError('Original r2 ZIP changed')
    if digest(root / 'docs/ZSCL实验设计0601.md') != DESIGN_SHA:
        raise ValueError('Original design changed')
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise ValueError('r2 ZIP CRC mismatch')
        for item in handle.infolist():
            if item.is_dir():
                continue
            if not item.filename.startswith('faithful_0601_20261011/'):
                raise ValueError('Unexpected release member')
            relative = item.filename.split('/', 1)[1]
            expected = hashlib.sha256(handle.read(item.filename)).hexdigest()
            if digest(release / relative) != expected:
                raise ValueError('Extracted release differs: ' + relative)
    cache = json.loads((release / 'assets/legacy_cache_manifest.json').read_text(encoding='utf-8'))
    text_dir = root / 'knowledge/text_inputs/ut-zap50k'
    text_names = ('attributes.txt', 'objects.txt', 'oi_dict.json', 'oe_dict.json',
                  'ae_dict.json', 'manifest.json', 'review_status.csv')
    for name in text_names:
        expected = cache['input_sha256']['/data/xhw/czsl_repro/knowledge/kcc0601_v1/trusted_v1/ut-zap50k/' + name]
        if digest(text_dir / name) != expected:
            raise ValueError('Text input mismatch: ' + name)
    bank = json.loads((release / 'assets/logic_bank/manifest.json').read_text(encoding='utf-8'))
    logic = root / 'knowledge/logic_reconstruction'
    if digest(logic / 'knowledge.json') != bank['knowledge_sha256']:
        raise ValueError('Logic knowledge differs from bank provenance')
    if digest(logic / 'labels.json') != bank['labels_sha256']:
        raise ValueError('Logic labels differ from bank provenance')
    plan = json.loads((root / 'configs/run_plan.json').read_text(encoding='utf-8'))
    matrix = json.loads((root / 'configs/variant_matrix.json').read_text(encoding='utf-8'))
    expected_variants = set('ABCDEFGH') | {'J0', 'Jtext', 'H_no_pair', 'H_shuffle',
                                          'H_random_prior', 'H_uniform_prior'}
    variants = matrix['variants']
    if {v['name'] for v in variants} != expected_variants or len(variants) != 14:
        raise ValueError('Variant specification is incomplete')
    if matrix['seeds'] != plan['seeds'] or len(variants) * len(plan['seeds']) != plan['total_jobs']:
        raise ValueError('Run plan count/seed mismatch')
    if plan['release_sha256'] != RELEASE_SHA or plan['epochs'] != 30:
        raise ValueError('Run plan does not describe frozen r2 protocol')
    for variant in variants:
        name = variant['name']
        identity = name if len(name) == 1 else 'H'
        expected = {'text': identity in 'CEFGH', 'calibration': identity in 'DEFGH',
                    'feature': identity in 'GH', 'logic': identity in 'BFH',
                    'formula_joint': name not in ('A', 'B', 'C')}
        if name in ('J0', 'Jtext'):
            expected.update(text=name == 'Jtext', calibration=False, feature=False, logic=False)
        if any(variant[k] != value for k, value in expected.items()):
            raise ValueError('Variant switch mismatch: ' + name)
    python_files = list(release.rglob('*.py')) + list((root / 'tools').glob('*.py'))
    for path in python_files:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=8)
    result = {'passed': True, 'files_verified': len(files), 'release_files_verified': len(manifest),
              'release_sha256': RELEASE_SHA, 'original_design_sha256': DESIGN_SHA,
              'zip_crc_passed': True, 'text_input_files_verified': len(text_names),
              'logic_knowledge_and_labels_verified': True,
              'planned_jobs': plan['total_jobs'], 'python38_syntax_files': len(python_files),
              'server_cuda_training_performed': False, 'cpu_tests': 'not requested'}
    if cpu_tests:
        with tempfile.TemporaryDirectory(prefix='0601_desktop_verify_') as temporary:
            working = Path(temporary) / release.name
            shutil.copytree(release, working)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            help_checks = []
            for name in ('prepare.py', 'run_suite.py', 'report.py', 'status.py'):
                process = subprocess.run([sys.executable, str(working / name), '--help'],
                                         capture_output=True, env=env)
                if process.returncode:
                    raise RuntimeError(name + ': ' + process.stderr.decode('utf-8', errors='replace'))
                help_checks.append(name)
            process = subprocess.run([sys.executable, str(working / 'test_formula.py')],
                                     capture_output=True, env=env)
            if process.returncode:
                raise RuntimeError((process.stdout + process.stderr).decode('utf-8', errors='replace'))
            result['cpu_tests'] = json.loads((working / 'local_test.json').read_text(encoding='utf-8'))
            result['entrypoint_help_passed'] = help_checks
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--cpu-tests', action='store_true')
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.bundle, arguments.cpu_tests), ensure_ascii=False, indent=2))
