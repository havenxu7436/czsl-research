"""Verify package files using only Python's standard library."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'release_manifest.json').read_text(encoding='utf-8'))
for name, expected in manifest['files_sha256'].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('Package content differs: ' + name)
print('RELEASE_VERIFIED', len(manifest['files_sha256']), 'files')
