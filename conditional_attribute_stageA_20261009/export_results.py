"""Export Stage A diagnostic evidence, without backbone or feature-cache weights."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

from common import ROOT, read_json, sha, verify_release, zip_destination


def verify_zip(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or z.testzip():
            raise ValueError('Invalid ZIP inventory/CRC')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:
                raise ValueError('Unsafe member')
        manifest = json.loads(z.read('manifest.json'))['files']
        if set(names) != set(manifest) | {'manifest.json'}:
            raise ValueError('Manifest inventory mismatch')
        for name, spec in manifest.items():
            data = z.read(name)
            if len(data) != spec['size'] or hashlib.sha256(data).hexdigest() != spec['sha256']:
                raise ValueError('Invalid fingerprint: ' + name)
    print('STAGE_A_ZIP_VERIFIED', sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--logs', type=Path)
    parser.add_argument('--verify-zip', type=Path)
    args = parser.parse_args()
    if args.verify_zip:
        verify_zip(args.verify_zip)
        return
    if args.base is None or args.probe is None or args.out is None:
        parser.error('--base, --probe and --out are required for exporting')
    verify_release()
    ref = read_json(ROOT / 'reference.json')
    archive = zip_destination(args.out, ref, [args.base, args.probe] + ([args.logs] if args.logs else []))
    sidecar = archive.with_name(archive.name + '.sha256')
    archive.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    roots = [('stageA', args.base.resolve()), ('bf_probe', args.probe.resolve()), ('release', ROOT)]
    if args.logs is not None:
        roots.append(('logs', args.logs.resolve()))
    for prefix, root in roots:
        if not root.is_dir():
            raise FileNotFoundError('Evidence directory missing: ' + str(root))
        for path in sorted(root.iterdir()):
            allowed = {'.json', '.npy', '.npz'} if prefix != 'release' else {'.json', '.py'}
            if prefix == 'logs':
                allowed = {'.json', '.log'}
            if not path.is_file() or path.suffix not in allowed:
                continue
            if path.is_symlink() or path.stat().st_size > 64 * 1024 ** 2:
                raise ValueError('Unsafe or oversized evidence file')
            entries[prefix + '/' + path.name] = path
    if sum(p.stat().st_size for p in entries.values()) > 128 * 1024 ** 2:
        raise ValueError('Evidence size exceeds 128 MiB')
    manifest = {'schema': 'conditional_attribute_stageA_export_v1', 'files': {}}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in entries.items():
            before = sha(path)
            z.write(path, name)
            if sha(path) != before:
                raise RuntimeError('Evidence changed mid-export; preserve partial ZIP')
            manifest['files'][name] = dict(size=path.stat().st_size, sha256=before)
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    sidecar.write_text(sha(archive) + '  ' + archive.name + '\n', encoding='utf-8')
    verify_zip(archive)
    prepared, probe = args.base / 'prepared.json', args.probe / 'probe.json'
    print('STAGE_A_COMPLETE', prepared.is_file() and read_json(prepared).get('passed') is True)
    print('BF_PROBE_COMPLETE', probe.is_file() and read_json(probe).get('passed') is True)
    print('No TEST images or model weights exported.')


if __name__ == '__main__':
    main()
