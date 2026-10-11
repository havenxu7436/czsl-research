"""Stdlib-only, hash-sealed pilot export and offline ZIP verification."""
import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from common import ROOT, read_json, sha, zip_destination


def allowed_work_file(name):
    roots = {'protocol.json', 'environment.json', 'baseline.json', 'labels.json',
             'candidates_cw.json', 'candidates_ow.json', 'val_image_ids.json', 'val_truth.npz',
             'mechanics_complete.json', 'summary.json', 'execution_status.json'}
    if name in roots or (name.endswith('.partial') and name[:-8] in roots):
        return True
    variant = r'P_(?:name|K|generic|perm1|perm2)'
    attempt = r'attempt_[0-9]{4,}'
    patterns = [
        r'mechanics/' + attempt + r'/(?:fold|summary|' + variant + r')\.json(?:\.partial)?',
        r'mechanics/' + attempt + '/' + variant + r'_hold_predictions\.npz',
        r'runs/' + variant + r'/(?:last\.pt|best_auc\.pt|history\.json|complete\.json)(?:\.partial)?',
        r'runs/' + variant + r'/epochs/epoch_[0-9]{4,}/' + attempt
        + r'/(?:grid_scores\.npy|conditional_predictions\.npz|evaluation\.json)(?:\.partial)?',
        r'failures/' + attempt + r'/failure\.json(?:\.partial)?',
        r'resume_selfchecks/' + attempt + r'/selfcheck\.json(?:\.partial)?']
    return any(re.fullmatch(pattern, name) for pattern in patterns)


def export_work(work, destination):
    """Caller must own exclusive pilot.lock; standalone main acquires it."""
    work = Path(work).resolve()
    ref = read_json(ROOT / 'reference.json')
    destination = zip_destination(destination, ref, [work, Path(ref['stageA_base']).parent])
    partial = destination.with_name(destination.name + '.partial')
    if partial.exists():
        raise FileExistsError('Preserve partial export; choose another output name')
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in sorted(work.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink in owned results')
        if not path.is_file() or path.name == 'pilot.lock':
            continue
        name = path.relative_to(work).as_posix()
        if not allowed_work_file(name):
            raise ValueError('Unrecognized results file; refusing to export: ' + name)
        files['pilot/' + name] = path
    manifest = read_json(ROOT / 'release_manifest.json')
    for name, expected in manifest.items():
        if Path(name).name != name or sha(ROOT / name) != expected:
            raise ValueError('Executed release changed before export: ' + name)
        files['release/' + name] = ROOT / name
    files['release/release_manifest.json'] = ROOT / 'release_manifest.json'
    size = sum(p.stat().st_size for p in files.values())
    if size > 512 * 1024**2:
        raise ValueError('Pilot export exceeds the fixed 512 MiB uncompressed budget')
    if shutil.disk_usage(destination.parent).free < size + 128 * 1024**2:
        raise RuntimeError('Insufficient space; no results removed')
    inventory = {name: dict(sha256=sha(path), bytes=path.stat().st_size) for name, path in files.items()}
    status = read_json(work / 'execution_status.json') if (work / 'execution_status.json').exists() else {}
    passed = status.get('state') == 'complete'
    summary = read_json(work / 'summary.json') if passed else {}
    if passed and (not summary.get('passed') or summary.get('protocol_sha256') != status.get('protocol_sha256')):
        raise ValueError('Completion status and summary disagree')
    specification = dict(schema='conditional_attribute_pilot_export_v1', execution_passed=passed,
                         pilot_budget_gate_passed=summary.get('decision', {}).get('pilot_budget_gate_passed') if passed else None,
                         includes_frozen_base_weights=False, includes_feature_caches=False,
                         includes_test_predictions=False, files=inventory)
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in files.items():
            archive.write(path, name)
            if sha(path) != inventory[name]['sha256']:
                raise ValueError('Results changed while exporting: ' + name)
        archive.writestr('manifest.json', json.dumps(specification, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    verify_zip(partial, require_sidecar=False)
    if destination.exists():
        raise FileExistsError('Preserve concurrently created destination')
    partial.rename(destination)
    with destination.with_name(destination.name + '.sha256').open('x', encoding='ascii') as handle:
        handle.write(sha(destination) + '  ' + destination.name + '\n')
    print('PILOT_EXPORT_COMPLETE', str(destination), 'execution_passed', passed, flush=True)
    return specification


def verify_zip(path, require_sidecar=True):
    import hashlib
    path = Path(path)
    if require_sidecar:
        expected = path.with_name(path.name + '.sha256').read_text(encoding='ascii').split()[0]
        if sha(path) != expected:
            raise ValueError('ZIP SHA256 differs')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe ZIP path')
        if sum(x.file_size for x in archive.infolist()) > 520 * 1024**2:
            raise ValueError('ZIP expanded size exceeds budget')
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC failed')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['schema'] != 'conditional_attribute_pilot_export_v1':
            raise ValueError('Not a pilot export')
        if set(names) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('ZIP inventory differs')
        for name, expected in manifest['files'].items():
            info = archive.getinfo(name)
            if info.file_size != expected['bytes'] or hashlib.sha256(archive.read(name)).hexdigest() != expected['sha256']:
                raise ValueError('ZIP member hash/size differs: ' + name)
            if name.startswith('pilot/') and not allowed_work_file(name[len('pilot/'):]):
                raise ValueError('Unrecognized pilot artifact')
            if name.startswith('release/') and len(PurePosixPath(name).parts) != 2:
                raise ValueError('Unexpected release path')
            if not name.startswith(('pilot/', 'release/')):
                raise ValueError('Unexpected ZIP member')
    print('PILOT_ZIP_VERIFIED', len(names), 'members', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-zip', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.verify_zip:
        verify_zip(args.verify_zip)
        return
    if not args.work or not args.out:
        parser.error('Choose --verify-zip, or both --work and --out')
    import fcntl
    work = args.work.resolve()
    protocol = read_json(work / 'protocol.json')
    if protocol.get('schema') != 'conditional_attribute_pilot_v1':
        raise ValueError('Not an owned pilot directory')
    with (work / 'pilot.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        export_work(work, args.out)


if __name__ == '__main__':
    main()
