import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
entries = []
for line in (root/'SHA256SUMS').read_text(encoding='ascii').splitlines():
    expected, name = line.split('  ', 1)
    path = root/name
    if path.resolve().parent != root or len(expected) != 64:
        raise ValueError('Invalid release entry')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('V2 release checksum mismatch: ' + name)
    entries.append(name)
if len(entries) != len(set(entries)):
    raise ValueError('Duplicate release entry')
print('V2 RELEASE CHECKSUM PASSED', len(entries), flush=True)
