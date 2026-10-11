"""Stdlib-only guarded export and offline verification for generic controls."""
import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from bootstrap import ROOT, child, guard_path, read_json, sha, verify_release


def allowed(name):
    suffix = r'(?:\.partial)?'
    root = {'protocol.json', 'environment.json', 'baseline.json', 'labels.json',
            'candidates_cw.json', 'candidates_ow.json', 'val_image_ids.json', 'val_truth.npz',
            'reference_checks_complete.json', 'mechanics_complete.json', 'summary.json', 'execution_status.json'}
    if name in root or (name.endswith('.partial') and name[:-8] in root):
        return True
    attempt, variant = r'attempt_[0-9]{4,}', r'G_(?:K|name|perm1|perm2)'
    patterns = [
        r'reference_checks/' + attempt + r'/replay\.json' + suffix,
        r'mechanics/' + attempt + r'/(?:fold|summary|' + variant + r')\.json' + suffix,
        r'mechanics/' + attempt + r'/' + variant + r'_hold_predictions\.npz',
        r'runs/G_K/complete\.json' + suffix,
        r'runs/G_K/reference_replays/' + attempt + r'/(?:history\.json|selected_reference\.pt)' + suffix,
        r'runs/G_K/reference_replays/' + attempt + r'/epoch_[0-9]{4}/(?:grid_scores\.npy|unmixed_grid_scores\.npy|conditional_predictions\.npz|evaluation\.json)' + suffix,
        r'runs/G_(?:name|perm1|perm2)/(?:last\.pt|best_auc\.pt|history\.json|complete\.json)' + suffix,
        r'runs/G_(?:name|perm1|perm2)/epochs/epoch_[0-9]{4,}/' + attempt + r'/(?:grid_scores\.npy|unmixed_grid_scores\.npy|conditional_predictions\.npz|evaluation\.json)' + suffix,
        r'failures/' + attempt + r'/failure\.json' + suffix,
        r'resume_selfchecks/' + attempt + r'/selfcheck\.json' + suffix]
    return any(re.fullmatch(p, name) for p in patterns)


def export_work(work, destination):
    work = Path(work).resolve()
    ref = read_json(ROOT / 'legacy/reference.json')
    experiment = read_json(ROOT / 'experiment.json')
    destination = guard_path(destination, ref, experiment, [work])
    sidecar = destination.with_name(destination.name + '.sha256')
    partial = destination.with_name(destination.name + '.partial')
    if destination.exists() or sidecar.exists() or partial.exists():
        raise FileExistsError('Preserve existing export/sidecar/partial')
    files = {}
    for path in sorted(work.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink in owned work')
        if not path.is_file() or path.name == 'controls.lock':
            continue
        name = path.relative_to(work).as_posix()
        if not allowed(name):
            raise ValueError('Unrecognized work artifact: ' + name)
        files['controls/' + name] = path
    verify_release()
    for name in read_json(ROOT / 'release_manifest.json'):
        files['release/' + name] = child(ROOT, name)
    files['release/release_manifest.json'] = ROOT / 'release_manifest.json'
    size = sum(path.stat().st_size for path in files.values())
    if size > 512*1024**2:
        raise ValueError('Expanded export exceeds 512 MiB budget')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < size + 128*1024**2:
        raise RuntimeError('Insufficient export space; no files removed')
    inventory = {name: dict(sha256=sha(path), bytes=path.stat().st_size) for name, path in files.items()}
    status = read_json(work / 'execution_status.json') if (work / 'execution_status.json').exists() else {}
    passed = status.get('state') == 'complete'
    summary = read_json(work / 'summary.json') if passed else {}
    if passed and (not summary.get('passed') or summary.get('protocol_sha256') != status.get('protocol_sha256')):
        raise ValueError('Completion status/summary mismatch')
    manifest = dict(schema='generic_mixture_controls_export_v1', execution_passed=passed,
                    generic_budget_gate_passed=summary.get('decision', {}).get('generic_budget_gate_passed') if passed else None,
                    new_formal_optimizer_steps=summary.get('new_formal_optimizer_steps') if passed else None,
                    source_correct_run_reused=True, includes_frozen_base_weights=False,
                    includes_feature_caches=False, includes_test_predictions=False, files=inventory)
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in files.items():
            archive.write(path, name)
            if sha(path) != inventory[name]['sha256']:
                raise ValueError('Artifact changed during export: ' + name)
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    verify_zip(partial, require_sidecar=False)
    partial.rename(destination)
    with sidecar.open('x', encoding='ascii') as handle:
        handle.write(sha(destination) + '  ' + destination.name + '\n')
    print('GENERIC_CONTROLS_EXPORT_COMPLETE', destination, 'execution_passed', passed, flush=True)
    return manifest


def verify_zip(path, require_sidecar=True):
    path = Path(path)
    if require_sidecar:
        expected = path.with_name(path.name+'.sha256').read_text(encoding='ascii').split()[0]
        if sha(path) != expected:
            raise ValueError('ZIP SHA256 differs')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError('ZIP duplicate member or CRC failure')
        if sum(info.file_size for info in archive.infolist()) > 520*1024**2:
            raise ValueError('Expanded ZIP too large')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe ZIP path')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['schema'] != 'generic_mixture_controls_export_v1' or set(names) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('ZIP schema/inventory differs')
        for name, expected in manifest['files'].items():
            data = archive.read(name)
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('ZIP member identity differs: ' + name)
            if name.startswith('controls/') and not allowed(name[len('controls/'):]):
                raise ValueError('Unrecognized work member')
            if not name.startswith(('controls/', 'release/')):
                raise ValueError('Unexpected ZIP root')
    print('GENERIC_CONTROLS_ZIP_VERIFIED', len(names), 'members', flush=True)
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
        parser.error('Choose --verify-zip or both --work and --out')
    import fcntl
    with (args.work.resolve() / 'controls.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        export_work(args.work, args.out)


if __name__ == '__main__':
    main()
