"""Stdlib bootstrap and immutable-release/path checks; no Torch needed for export."""
import sys
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'legacy'))
from common import read_json, sha, verify_files, write_json


def child(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe relative artifact path')
    path = (Path(root) / name).resolve()
    if Path(root).resolve() not in path.parents:
        raise ValueError('Artifact escaped owned root')
    return path


def verify_release():
    manifest = read_json(ROOT / 'release_manifest.json')
    for name, expected in manifest.items():
        path = child(ROOT, name)
        if path.is_symlink() or sha(path) != expected:
            raise ValueError('Release identity changed: ' + name)
    return sha(ROOT / 'release_manifest.json')


def protected_roots(reference, experiment):
    return [ROOT.resolve(), Path('/data').resolve(), Path(experiment['source_work']).resolve(),
            Path(reference['stageA_base']).parent.resolve(), Path(reference['c_source']).parent.resolve(),
            Path(reference['bf_source']).parent.resolve()]


def guard_path(path, reference, experiment, extra=()):
    path = Path(path).resolve()
    for root in protected_roots(reference, experiment) + [Path(x).resolve() for x in extra]:
        if path == root or root in path.parents:
            raise ValueError('Destination inside protected original/input directory')
    return path
