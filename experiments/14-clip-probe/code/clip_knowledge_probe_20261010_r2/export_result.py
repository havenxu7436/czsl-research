"""Self-contained results, no original images/model weights/site-packages."""
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from common import ROOT, read_json, require, sha, write_json


def export(work, target):
    work, target = Path(work), Path(target)
    require(target.parent == work.parent and target.name.startswith(work.name),
            'Export must be a sibling ZIP bearing the work-directory prefix')
    require(not target.exists() and not Path(str(target) + '.sha256').exists(), 'Export exists')
    files = {}
    for p in work.rglob('*'):
        if p.is_file() and not any(part in ('deps', 'model', '__pycache__') for part in p.relative_to(work).parts) and p.name != 'run.lock':
            files['result/' + str(p.relative_to(work)).replace('\\', '/')] = p
    for p in ROOT.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
            files['source/' + str(p.relative_to(ROOT)).replace('\\', '/')] = p
    manifest = {'files_sha256': {name: sha(p) for name, p in sorted(files.items())},
                'original_images_included': False, 'clip_weights_included': False}
    temporary = Path(str(target) + '.partial')
    require(not temporary.exists(), 'Partial export exists; choose another export suffix')
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, p in sorted(files.items()):
            z.write(p, name)
        z.writestr('export_manifest.json', json.dumps(manifest, indent=2))
    verify(temporary)
    os.replace(str(temporary), str(target))
    Path(str(target) + '.sha256').write_text(sha(target) + '  ' + target.name + '\n', encoding='ascii')
    passed = read_json(work / 'complete.json').get('execution_passed') if (work / 'complete.json').exists() else False
    print('KNOWLEDGE_PROBE_EXPORT_COMPLETE', str(target), 'execution_passed', passed, flush=True)


def verify(path):
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None, 'ZIP CRC error')
        names = z.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        manifest = json.loads(z.read('export_manifest.json'))
        require(set(names) == set(manifest['files_sha256']) | {'export_manifest.json'}, 'ZIP membership mismatch')
        for name, expected in manifest['files_sha256'].items():
            with z.open(name) as f:
                h = hashlib.sha256()
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            require(h.hexdigest() == expected, 'ZIP content mismatch: ' + name)
    print('KNOWLEDGE_PROBE_ZIP_VERIFIED', len(names), 'members', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', required=True)
    verify(parser.parse_args().verify)
