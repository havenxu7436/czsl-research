import hashlib
from pathlib import Path

root=Path(__file__).resolve().parent
for line in (root/'SHA256SUMS').read_text(encoding='ascii').splitlines():
    digest,name=line.split('  ',1)
    path=(root/name).resolve()
    if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('Release checksum failed: '+name)
print('NAME RELEASE CHECKSUM PASSED',flush=True)
