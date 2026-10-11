import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(pending, path)


def verify_release(bundle):
    for name, fingerprint in read(bundle / 'release_manifest.json').items():
        if sha(bundle / name) != fingerprint:
            raise ValueError('Release changed: ' + name)


def verify_files(mapping):
    for name, fingerprint in mapping.items():
        if sha(name) != fingerprint:
            raise ValueError('Frozen input changed: ' + name)
