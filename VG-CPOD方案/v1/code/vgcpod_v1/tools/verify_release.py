"""Verify release bytes with Python standard library only."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'release_manifest.json').read_text(encoding='utf-8'))
failures = []
for name, expected in manifest['files_sha256'].items():
    path = root / name
    if not path.is_file():
        failures.append('missing: ' + name)
        continue
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != expected:
        failures.append('changed: ' + name)
if failures:
    raise SystemExit('\n'.join(failures))
print('RELEASE_VERIFIED', len(manifest['files_sha256']), 'files')
