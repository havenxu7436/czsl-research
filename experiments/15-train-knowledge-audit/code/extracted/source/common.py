"""Read-only inputs, atomic generated artifacts, Python 3.8 compatible."""
import hashlib
import io
import json
import os
import uuid
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


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
    temp = path.with_name(path.name + '.writing-' + uuid.uuid4().hex)
    with open(temp, 'x', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(str(temp), str(path))


def write_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    temp = path.with_name(path.name + '.writing-' + uuid.uuid4().hex)
    with open(temp, 'xb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(str(temp), str(path))


def valid_relative(name):
    p = PurePosixPath(name)
    require(name and '\\' not in name and not p.is_absolute() and
            '..' not in p.parts and ':' not in name, 'Unsafe relative path: ' + name)


class Inputs:
    """A closed allowlist: requesting VAL, TEST or other files is an error."""
    def __init__(self, expected, root=None, archive=None):
        require((root is None) != (archive is None), 'Choose one input backend')
        self.expected, self.root, self.archive = expected, Path(root) if root else None, archive
        self.reads = set()
        for name in expected:
            valid_relative(name)
            require(not any(s in name.lower() for s in ('val_', '/val/', 'test_', '/test/')),
                    'Non-TRAIN input in allowlist: ' + name)

    def blob(self, name):
        require(name in self.expected, 'Input not on TRAIN allowlist: ' + name)
        if self.archive is not None:
            value = self.archive.read('result/' + name)
        else:
            p = (self.root / name).resolve()
            require(self.root.resolve() in p.parents, 'Input escapes probe root')
            value = p.read_bytes()
        require(hashlib.sha256(value).hexdigest() == self.expected[name], 'Input SHA mismatch: ' + name)
        self.reads.add(name)
        return value

    def js(self, name):
        return json.loads(self.blob(name))

    def arrays(self, name):
        import numpy as np
        with np.load(io.BytesIO(self.blob(name)), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def verify_all(self):
        for name in sorted(self.expected):
            self.blob(name)


def verify_release():
    manifest = read_json(ROOT / 'release_manifest.json')
    for name, expected in manifest['files_sha256'].items():
        valid_relative(name)
        require(sha(ROOT / name) == expected, 'Release SHA mismatch: ' + name)
    return manifest
