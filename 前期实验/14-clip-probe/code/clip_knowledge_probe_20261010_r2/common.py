"""Small, Python 3.8-compatible helpers; no training-project imports."""
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    os.replace(str(temporary), str(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_release():
    manifest = read_json(ROOT / 'release_manifest.json')
    for relative, expected in manifest['files_sha256'].items():
        require(sha(ROOT / relative) == expected, 'Release changed: ' + relative)
    return manifest


def trusted_torch_load(path, expected):
    # Only load a locally archived, exactly SHA-verified file. Never use this
    # fallback on arbitrary checkpoints or downloaded unverified pickle data.
    import inspect
    import torch
    require(sha(path) == expected, 'Trusted input changed: ' + str(path))
    kw = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kw['weights_only'] = False
    return torch.load(str(path), **kw)
