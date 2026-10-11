import hashlib
from pathlib import Path

root=Path(__file__).resolve().parent
count=0
for line in (root/'SHA256SUMS').read_text(encoding='ascii').splitlines():
    digest,name=line.split('  ',1)
    path=root/name
    if len(digest)!=64 or path.resolve().parent!=root:
        raise ValueError('Invalid release manifest entry')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('Diagnostic release checksum mismatch: '+name)
    count+=1
print('DIAGNOSTIC RELEASE CHECKSUM PASSED',count,flush=True)
