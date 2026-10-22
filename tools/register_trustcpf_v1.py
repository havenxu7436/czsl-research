"""Register, verify and restore the independent TRUST-CPF CLIP v1 release."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LINE = 'TRUST-CPF方案'
STAGE = LINE + '/v1'
PACKAGE = 'trustcpf_clip_v1'
ARCHIVE_STAGE = '32'
MAP = STAGE + '/results/release_member_map.json'
RECEIPT = 'provenance/updates/20261021_trustcpf_v1.json'
CLIP_SHA = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
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
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe package member: ' + name)
    return path


def destination(name):
    path = safe_name(name)
    if name.startswith('assets/knowledge/'):
        return STAGE + '/configs/' + PACKAGE + '/' + name, 'knowledge'
    if name.startswith(('configs/', 'assets/')):
        return STAGE + '/configs/' + PACKAGE + '/' + name, 'configs'
    if path.suffix in ('.py', '.sh') or name in ('requirements.txt', 'vendor/CLIP_LICENSE'):
        return STAGE + '/code/' + PACKAGE + '/' + name, 'code_files'
    if path.suffix == '.md':
        return STAGE + '/results/package_support/' + PACKAGE + '/' + name + '.txt', 'data_files'
    if name.startswith('validation/'):
        return STAGE + '/results/local-validation-20261021/' + path.name, 'data_files'
    return STAGE + '/results/' + PACKAGE + '/' + name, 'data_files'


def register(package, archive):
    package, archive = Path(package), Path(archive)
    if (ROOT / MAP).exists():
        raise FileExistsError('Release already registered; use --check')
    expected = read(package / 'release_manifest.json')
    names = sorted(expected) + ['release_manifest.json']
    for name, digest in expected.items():
        safe_name(name)
        if sha(package / name) != digest:
            raise ValueError('Source member changed: ' + name)
    with zipfile.ZipFile(archive) as zipped:
        if set(zipped.namelist()) != set(names) or zipped.testzip() is not None:
            raise ValueError('ZIP member set differs')
        for name in names:
            if zipped.read(name) != (package / name).read_bytes():
                raise ValueError('ZIP member changed: ' + name)
    stages = read(ROOT / 'provenance/stages.json')
    if any(s['id'] == ARCHIVE_STAGE or s.get('line') == LINE for s in stages):
        raise ValueError('Stage identity is already used')
    mapping, records = {}, []
    categories = {k: [] for k in ('knowledge', 'configs', 'code_files', 'data_files')}
    for name in names:
        target, category = destination(name)
        path = ROOT / target
        if path.exists():
            raise FileExistsError('Target exists: ' + target)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / name, path)
        mapping[name] = target
        categories[category].append(target)
        records.append(dict(original_path=archive.name + '::' + name,
            runtime_path=PACKAGE + '/' + name, current_path=target,
            sha256=sha(path), bytes=path.stat().st_size,
            stage=ARCHIVE_STAGE, kind='trustcpf_v1_release_member'))
    deployment = ROOT / STAGE / 'results/deployment' / archive.name
    deployment.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, deployment)
    deployment.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='utf-8')
    write(ROOT / MAP, dict(source_zip_sha256=sha(archive), source_zip_root='.',
          runtime_package=PACKAGE, members=mapping))
    layout = read(ROOT / 'provenance/layout.json')
    layout['files'].extend(records)
    write(ROOT / 'provenance/layout.json', layout)
    stages.append(dict(id=ARCHIVE_STAGE, line=LINE, number='v1', version='v1',
        title='v1 · CLIP可迁移关系效用与风险预算选图', path=STAGE,
        primary_package=PACKAGE, active_package=PACKAGE,
        status='CLIP方案、实现、13项测试与CUDA合成验证已归档；待服务器正式训练',
        completed_runs=0, planned_main_runs=21, local_tests=13))
    write(ROOT / 'provenance/stages.json', stages)
    write(ROOT / STAGE / 'inputs.json', dict(categories, line=LINE, stage='v1',
        archive_stage=ARCHIVE_STAGE, primary_package=PACKAGE,
        status='clip_implementation_and_local_validation',
        formal_training_results_included=False, completed_runs=0,
        planned_main_runs=21, seeds=[3351, 3352, 3353],
        variants=['r0', 'capacity', 'dense', 'drop', 'free', 'mean', 'robust'],
        release_member_map=MAP, code_and_package_members_preserved=True))
    write(ROOT / RECEIPT, dict(date='2026-10-21', line=LINE, stage='v1',
        archive_stage=ARCHIVE_STAGE, source_zip_sha256=sha(archive),
        release_members=len(names), original_member_bytes_modified=False,
        local_tests=13, cuda_validation='synthetic 768D CLIP-space tensors',
        formal_training_results_included=False, completed_runs=0,
        planned_main_runs=21, training_started_by_import=False, mapping=MAP))
    return {'registered_members': len(names), 'training_started': False}


def check():
    record = read(ROOT / MAP)
    mapping = record['members']
    manifest = read(ROOT / mapping['release_manifest.json'])
    if set(mapping) != set(manifest) | {'release_manifest.json'}:
        raise ValueError('Incomplete release mapping')
    records = [r for r in read(ROOT / 'provenance/layout.json')['files']
               if r['stage'] == ARCHIVE_STAGE and r['kind'] == 'trustcpf_v1_release_member']
    if len(records) != len(mapping):
        raise ValueError('Layout count differs')
    for row in records:
        name = row['runtime_path'].split('/', 1)[1]
        if row['current_path'] != mapping[name] or sha(ROOT / row['current_path']) != row['sha256']:
            raise ValueError('Layout member differs: ' + name)
    for name, digest in manifest.items():
        if sha(ROOT / mapping[name]) != digest:
            raise ValueError('Manifest member differs: ' + name)
    deployment = next((ROOT / STAGE / 'results/deployment').glob('*.zip'))
    if sha(deployment) != record['source_zip_sha256']:
        raise ValueError('Deployment ZIP changed')
    with zipfile.ZipFile(deployment) as zipped:
        if set(zipped.namelist()) != set(mapping):
            raise ValueError('Deployment member set differs')
        for name, target in mapping.items():
            safe_name(name)
            if zipped.read(name) != (ROOT / target).read_bytes():
                raise ValueError('Deployment bytes differ: ' + name)
    inputs = read(ROOT / STAGE / 'inputs.json')
    indexed = set()
    for key in ('knowledge', 'configs', 'code_files', 'data_files', 'documents'):
        for name in inputs.get(key, []):
            safe_name(name)
            if not (ROOT / name).is_file():
                raise ValueError('Indexed file missing: ' + name)
            indexed.add(name)
    if not set(mapping.values()) <= indexed:
        raise ValueError('Unindexed release members')
    return dict(passed=True, line=LINE, stage='v1', release_members=len(mapping),
        manifest_hashes_verified=len(manifest), indexed_files=len(indexed),
        byte_identity_preserved=True,
        formal_training_results_included=inputs.get('formal_training_results_included', False),
        completed_runs=inputs.get('completed_runs', 0), training_started=False)


def materialize(destination):
    check()
    destination = Path(destination).expanduser()
    if not destination.is_absolute():
        raise ValueError('Choose an absolute destination')
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents or destination.exists():
        raise FileExistsError('Choose a new directory outside repository: ' + str(destination))
    from reproduce import safe_child
    receipts = []
    for name, source in sorted(read(ROOT / MAP)['members'].items()):
        path = safe_child(destination, PACKAGE + '/' + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, path)
        if path.suffix == '.sh':
            path.chmod(path.stat().st_mode | 0o111)
        if sha(path) != sha(ROOT / source):
            raise ValueError('Restored member differs: ' + name)
        receipts.append(dict(runtime_path=PACKAGE + '/' + name, sha256=sha(path)))
    write(destination / 'materialization.json', dict(line=LINE, stage='v1',
        archive_stage=ARCHIVE_STAGE, files=receipts, training_started=False,
        historical_routes_copied=False, original_bytes_modified=False))
    return dict(line=LINE, stage='v1', runtime=str(destination),
        package=str(destination / PACKAGE), files_verified=len(receipts),
        training_started=False)


def doctor(package):
    package = Path(package).resolve()
    problems = [name for name, digest in read(package / 'release_manifest.json').items()
        if not (package / name).is_file() or sha(package / name) != digest]
    data = Path(os.environ.get('DATA_ROOT', '/srv/czsl-project/data/ut-zap50k'))
    checkpoint = Path(os.environ.get('CLIP_CKPT', str(Path.home() / '.cache/clip/ViT-L-14.pt')))
    external = [str(data / name) for name, digest in read(package / 'assets/historical_inputs.json').items()
        if not (data / name).is_file() or sha(data / name) != digest]
    if not checkpoint.is_file() or sha(checkpoint) != CLIP_SHA:
        external.append(str(checkpoint))
    if not any((data / name).is_dir() for name in ('images', '_images_extract/ut-zap50k-images-square')):
        external.append(str(data / 'images'))
    return dict(line=LINE, stage='v1', package=str(package),
        changed_package_members=problems, missing_or_changed_external_inputs=external,
        ready_for_package_preflight=not (problems or external), training_started=False)


def finalize():
    inputs = read(ROOT / STAGE / 'inputs.json')
    paths = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / STAGE).rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    inputs['knowledge'] = [p for p in paths if '/assets/knowledge/' in p]
    inputs['configs'] = [p for p in paths if '/configs/' in p and not p.endswith('.md') and p not in inputs['knowledge']]
    inputs['code_files'] = [p for p in paths if '/code/' in p and not p.endswith('.md')]
    inputs['documents'] = [p for p in paths if p.endswith('.md')]
    used = set().union(*(set(inputs[k]) for k in ('knowledge', 'configs', 'code_files', 'documents')))
    inputs['data_files'] = [p for p in paths if p not in used and not p.endswith(('/inputs.json', '/runbook.json'))]
    write(ROOT / STAGE / 'inputs.json', inputs)
    stages = read(ROOT / 'provenance/stages.json')
    for stage in stages:
        if stage['id'] == ARCHIVE_STAGE:
            stage.update({k: len(inputs[k]) for k in ('knowledge', 'configs', 'code_files', 'data_files')})
    write(ROOT / 'provenance/stages.json', stages)
    owned = set(paths) | {'README.md', LINE + '/README.md', 'environment/README.md', 'tools/reproduce.py',
        'tools/register_trustcpf_v1.py', 'provenance/layout.json', 'provenance/stages.json', RECEIPT}
    owned.update(p.relative_to(ROOT).as_posix() for p in (ROOT / 'provenance/documents').glob('trustcpf_*.json'))
    current = {r['path']: r for r in read(ROOT / 'provenance/current_manifest.json')}
    for name in sorted(owned):
        path = ROOT / name
        current[name] = dict(path=name, sha256=sha(path), bytes=path.stat().st_size)
    write(ROOT / 'provenance/current_manifest.json', [current[p] for p in sorted(current)])
    return check()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--restore-check', action='store_true')
    args = parser.parse_args()
    if args.check:
        result = check()
    elif args.finalize:
        result = finalize()
    elif args.restore_check:
        with tempfile.TemporaryDirectory(prefix='trustcpf-restore-') as temporary:
            result = materialize(Path(temporary) / 'release')
    elif args.package and args.archive:
        result = register(args.package, args.archive)
    else:
        parser.error('Choose --check/--finalize/--restore-check or --package with --archive')
    print(json.dumps(result, ensure_ascii=False))
