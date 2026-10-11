#!/usr/bin/env python3
"""Check extracted release contents before running any preparation code."""
from pathlib import Path
from common import sha


def main():
    root = Path(__file__).resolve().parent
    sums = root / 'SHA256SUMS'
    count = 0
    for line in sums.read_text(encoding='utf-8').splitlines():
        fingerprint, name = line.split('  ', 1)
        path = (root / name).resolve()
        if root not in path.parents:
            raise ValueError('Unsafe release path')
        if sha(path) != fingerprint:
            raise ValueError('Release checksum mismatch: ' + name)
        count += 1
    print('RELEASE_VERIFIED', root, 'files=', count)


if __name__ == '__main__':
    main()
