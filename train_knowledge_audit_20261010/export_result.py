"""Self-contained audit report export; never includes checkpoint or old cache."""
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from common import ROOT, read_json, require, sha, valid_relative


def verify(path):
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None, 'ZIP CRC failure')
        names = z.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        for name in names:
            valid_relative(name)
        manifest = json.loads(z.read('export_manifest.json'))
        require(set(names) == set(manifest['files_sha256']) | {'export_manifest.json'}, 'Member set differs')
        for name, expected in manifest['files_sha256'].items():
            with z.open(name) as f:
                h = hashlib.sha256()
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            require(h.hexdigest() == expected, 'ZIP SHA differs: ' + name)
    print('TRAIN_AUDIT_ZIP_VERIFIED', len(names), 'members', flush=True)


def export(work, target):
    require(target.parent == work.parent and target.name.startswith(work.name), 'Export scope')
    require(not target.exists() and not Path(str(target) + '.sha256').exists(), 'Export already exists')
    files = {}
    for prefix, root in (('result', work), ('source', ROOT)):
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                files[prefix + '/' + p.relative_to(root).as_posix()] = p
    manifest = {'files_sha256': {n: sha(p) for n, p in sorted(files.items())},
                'train_contact_sheets_included': True, 'raw_images_included': False,
                'clip_weights_included': False, 'old_feature_cache_included': False}
    temp = Path(str(target) + '.partial')
    require(not temp.exists(), 'Partial ZIP exists; preserve it')
    with zipfile.ZipFile(temp, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n, p in sorted(files.items()):
            z.write(p, n)
        z.writestr('export_manifest.json', json.dumps(manifest, indent=2))
    verify(temp)
    os.replace(str(temp), str(target))
    Path(str(target) + '.sha256').write_text(sha(target) + '  ' + target.name + '\n', encoding='ascii')
    passed = read_json(work / 'complete.json')['execution_passed'] if (work / 'complete.json').exists() else False
    print('TRAIN_AUDIT_EXPORT_COMPLETE', str(target), 'execution_passed', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', required=True)
    verify(Path(parser.parse_args().verify))
