"""Verify the release inventory without importing project code."""
import hashlib
from pathlib import Path

root=Path(__file__).resolve().parent
lines=(root/'SHA256SUMS').read_text(encoding='ascii').splitlines()
expected={}
for line in lines:
    digest,name=line.split('  ',1)
    expected[name]=digest
actual={p.name for p in root.iterdir() if p.is_file() and p.name!='SHA256SUMS'}
if set(expected)!=actual:
    raise SystemExit('RELEASE INVENTORY FAILED')
for name,digest in expected.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise SystemExit('RELEASE CHECKSUM FAILED: '+name)
print('TRAIN SUPPORT RELEASE CHECKSUM PASSED',len(expected))
