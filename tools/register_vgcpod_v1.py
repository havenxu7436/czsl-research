"""Partition and verify the independent VG-CPOD v1 route without training."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LINE = 'VG-CPOD方案'
STAGE = LINE + '/v1'
PACKAGE = 'vgcpod_v1'
ARCHIVE_STAGE = '20'
RECEIPT = 'provenance/updates/20261018_vgcpod_v1.json'
MAP = STAGE + '/results/release_member_map.json'
CLIP_SHA = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_bytes() if path.exists() else b''
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if b'\r\n' in old:
        data = data.replace(b'\n', b'\r\n')
    path.write_bytes(data)


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive path: ' + name)
    return p


def destination(name):
    p = safe_name(name)
    if p.suffix in ('.py', '.sh') or name in ('requirements.txt', 'vendor/CLIP_LICENSE'):
        return STAGE + '/code/' + PACKAGE + '/' + name, 'code_files'
    if name.startswith(('configs/', 'assets/')):
        return STAGE + '/configs/' + PACKAGE + '/' + name, 'configs'
    if p.suffix == '.md':
        return STAGE + '/results/package_support/' + PACKAGE + '/' + name + '.txt', 'data_files'
    if name.startswith('validation/'):
        return STAGE + '/results/local-validation-20261018/' + p.name, 'data_files'
    return STAGE + '/results/' + PACKAGE + '/' + name, 'data_files'


def register(package, archive):
    package, archive = Path(package), Path(archive)
    if (ROOT / MAP).exists():
        raise FileExistsError('Route already registered; use --check')
    manifest = read(package / 'release_manifest.json')
    expected = manifest['files_sha256']
    names = sorted(expected) + ['release_manifest.json']
    for name, h in expected.items():
        safe_name(name)
        if sha(package / name) != h:
            raise ValueError('Frozen package changed: ' + name)
    with zipfile.ZipFile(archive) as z:
        members = {p.filename: z.read(p) for p in z.infolist() if not p.is_dir()}
    if set(members) != {PACKAGE + '/' + n for n in names}:
        raise ValueError('ZIP member set differs from frozen package')
    for name in names:
        if members[PACKAGE + '/' + name] != (package / name).read_bytes():
            raise ValueError('ZIP member differs: ' + name)
    mapping, records, categories = {}, [], {'knowledge': [], 'configs': [], 'code_files': [], 'data_files': []}
    for name in names:
        target, category = destination(name)
        path = ROOT / target
        if path.exists():
            raise FileExistsError('Canonical target exists: ' + target)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / name, path)
        mapping[name] = target
        categories[category].append(target)
        records.append({'original_path': archive.name + '::' + PACKAGE + '/' + name,
                        'runtime_path': PACKAGE + '/' + name, 'current_path': target,
                        'sha256': sha(path), 'bytes': path.stat().st_size,
                        'stage': ARCHIVE_STAGE, 'kind': 'vgcpod_v1_release_member'})
    deployment = ROOT / STAGE / 'results/deployment' / archive.name
    deployment.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, deployment)
    deployment.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='utf-8')
    write(ROOT / MAP, {'source_zip_sha256': sha(archive), 'original_package': PACKAGE,
                      'runtime_package': PACKAGE, 'members': mapping})
    layout = read(ROOT / 'provenance/layout.json')
    layout['files'].extend(records)
    write(ROOT / 'provenance/layout.json', layout)
    stages = read(ROOT / 'provenance/stages.json')
    if any(s['id'] == ARCHIVE_STAGE or s.get('line') == LINE for s in stages):
        raise ValueError('Stage identity is already used')
    stages.append({'id': ARCHIVE_STAGE, 'title': 'v1 · CLIP视觉引导类别低秩算子',
                   'version': 'v1', 'line': LINE, 'number': 'v1', 'path': STAGE,
                   'primary_package': PACKAGE, 'active_package': PACKAGE,
                   'status': '公式、代码与CPU验收已归档；无正式训练结果',
                   'knowledge_assets': 0, 'configs': len(categories['configs']),
                   'data_files': len(categories['data_files']), 'completed_runs': 0,
                   'evaluations': 0, 'planned_runs': 27, 'local_cpu_tests': 12})
    write(ROOT / 'provenance/stages.json', stages)
    write(ROOT / STAGE / 'inputs.json', dict(categories, stage='v1', line=LINE,
          archive_stage=ARCHIVE_STAGE, primary_package=PACKAGE, status='implementation_and_cpu_validation',
          formal_training_results_included=False, completed_runs=0, planned_runs=27,
          code_and_package_members_preserved=True, release_member_map=MAP))
    write(ROOT / RECEIPT, {'date': '2026-10-18', 'line': LINE, 'stage': 'v1', 'archive_stage': ARCHIVE_STAGE,
          'source_zip_sha256': sha(archive), 'release_members': len(names),
          'original_member_bytes_modified': False, 'local_cpu_tests': 12,
          'formal_training_results_included': False, 'completed_runs': 0, 'planned_runs': 27,
          'training_started_by_import': False, 'mapping': MAP,
          'runtime_restoration': 'this route only; no stages 01-19 are copied'})
    return mapping


def check():
    record = read(ROOT / MAP)
    mapping = record['members']
    manifest = read(ROOT / mapping['release_manifest.json'])['files_sha256']
    if set(mapping) != set(manifest) | {'release_manifest.json'}:
        raise ValueError('Incomplete release mapping')
    records = [r for r in read(ROOT / 'provenance/layout.json')['files'] if r['stage'] == ARCHIVE_STAGE]
    if len(records) != len(mapping):
        raise ValueError('Layout count differs')
    for r in records:
        name = r['runtime_path'].split('/', 1)[1]
        if r['current_path'] != mapping[name] or sha(ROOT / r['current_path']) != r['sha256']:
            raise ValueError('Layout identity differs: ' + name)
    for name, h in manifest.items():
        if sha(ROOT / mapping[name]) != h:
            raise ValueError('Mapped member differs: ' + name)
    deployment = next((ROOT / STAGE / 'results/deployment').glob('*.zip'))
    if sha(deployment) != record['source_zip_sha256']:
        raise ValueError('Deployment ZIP changed')
    with zipfile.ZipFile(deployment) as z:
        if set(z.namelist()) != {PACKAGE + '/' + n for n in mapping}:
            raise ValueError('Deployment member set differs')
        for name, target in mapping.items():
            if z.read(PACKAGE + '/' + name) != (ROOT / target).read_bytes():
                raise ValueError('Deployment member differs: ' + name)
    result = {'passed': True, 'line': LINE, 'stage': 'v1', 'release_members': len(mapping),
              'manifest_hashes_verified': len(manifest), 'byte_identity_preserved': True,
              'formal_training_results_included': False, 'training_started': False}
    return result


def materialize(destination):
    check()
    destination = Path(destination).expanduser()
    if not destination.is_absolute():
        raise ValueError('Choose an absolute destination')
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents or destination.exists():
        raise FileExistsError('Choose a new destination outside the repository: ' + str(destination))
    from reproduce import safe_child
    mapping = read(ROOT / MAP)['members']
    receipts = []
    for name, source in sorted(mapping.items()):
        path = safe_child(destination, PACKAGE + '/' + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, path)
        if path.suffix == '.sh':
            path.chmod(path.stat().st_mode | 0o111)
        if sha(path) != sha(ROOT / source):
            raise ValueError('Restored member differs')
        receipts.append({'runtime_path': PACKAGE + '/' + name, 'sha256': sha(path)})
    write(destination / 'materialization.json', {'line': LINE, 'stage': 'v1',
           'archive_stage': ARCHIVE_STAGE, 'files': receipts, 'training_started': False,
           'historical_routes_copied': False, 'original_bytes_modified': False})
    return {'line': LINE, 'stage': 'v1', 'runtime': str(destination), 'package': str(destination / PACKAGE),
            'files_verified': len(receipts), 'historical_routes_copied': False, 'training_started': False}


def doctor(package):
    package = Path(package).resolve()
    manifest = read(package / 'release_manifest.json')
    problems = [name for name, h in manifest['files_sha256'].items()
                if not (package / name).is_file() or sha(package / name) != h]
    data = Path(os.environ.get('VG_DATA', '/data/xhw/czsl_repro/data/ut-zap50k'))
    checkpoint = Path(os.environ.get('VG_CLIP', '/data/xhw/.cache/clip/ViT-L-14.pt'))
    inputs = read(package / 'assets/historical_inputs.json')
    external = []
    for name, h in inputs.items():
        p = data / name
        if not p.is_file() or sha(p) != h:
            external.append(str(p))
    if not checkpoint.is_file() or sha(checkpoint) != CLIP_SHA:
        external.append(str(checkpoint))
    if not any((data / folder).is_dir() for folder in ('images', '_images_extract/ut-zap50k-images-square')):
        external.append(str(data / 'images'))
    return {'line': LINE, 'stage': 'v1', 'package': str(package), 'changed_package_members': problems,
            'missing_or_changed_external_inputs': external, 'ready_for_package_preflight': not (problems or external),
            'not_checked': ['all individual image bytes', 'CUDA execution', 'training accuracy'], 'training_started': False}


def finalize():
    inputs = read(ROOT / STAGE / 'inputs.json')
    all_paths = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / LINE).rglob('*') if p.is_file())
    owned = set(all_paths)
    inputs['configs'] = sorted(p for p in all_paths if '/configs/' in p and not p.endswith('.md'))
    inputs['code_files'] = sorted(p for p in all_paths if '/code/' in p)
    inputs['documents'] = sorted(p for p in all_paths if p.endswith('.md'))
    inputs['data_files'] = sorted(set(inputs['data_files']) | {p for p in all_paths if '/results/' in p})
    write(ROOT / STAGE / 'inputs.json', inputs)
    stages = read(ROOT / 'provenance/stages.json')
    for s in stages:
        if s['id'] == ARCHIVE_STAGE:
            s.update(configs=len(inputs['configs']), data_files=len(inputs['data_files']), code_files=len(inputs['code_files']))
    write(ROOT / 'provenance/stages.json', stages)
    owned.update(('README.md', 'environment/README.md', 'tools/reproduce.py', 'tools/register_vgcpod_v1.py',
                  'provenance/layout.json', 'provenance/stages.json', RECEIPT,
                  'provenance/documents/vgcpod_v1_formulas_code.json'))
    current = {r['path']: r for r in read(ROOT / 'provenance/current_manifest.json')}
    for name in sorted(owned):
        path = ROOT / name
        current[name] = {'path': name, 'sha256': sha(path), 'bytes': path.stat().st_size}
    write(ROOT / 'provenance/current_manifest.json', [current[p] for p in sorted(current)])
    return check()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path)
    p.add_argument('--archive', type=Path)
    p.add_argument('--check', action='store_true')
    p.add_argument('--finalize', action='store_true')
    p.add_argument('--restore-check', action='store_true')
    args = p.parse_args()
    if args.check:
        result = check()
    elif args.finalize:
        result = finalize()
    elif args.restore_check:
        with tempfile.TemporaryDirectory(prefix='vgcpod-restore-') as temp:
            result = materialize(Path(temp) / 'independent')
    elif args.package and args.archive:
        result = {'registered_members': len(register(args.package, args.archive)), 'training_started': False}
    else:
        p.error('Choose --check/--finalize/--restore-check or --package with --archive')
    print(json.dumps(result, ensure_ascii=False))
