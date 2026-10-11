#!/usr/bin/env python3
"""Mechanical UTF-8/LF normalization, checksums and ZIP packaging of this release."""
import argparse
import hashlib
import zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    if a.out.exists():
        raise FileExistsError('Archive exists; preserve it or use a new version')
    files=[f for f in sorted(root.rglob('*')) if f.is_file() and '__pycache__' not in f.parts
           and f.name!='SHA256SUMS' and f.suffix!='.pyc']
    for f in files:
        if f.suffix in ('.py','.md','.json','.sh'):
            f.write_bytes(f.read_bytes().replace(b'\r\n',b'\n'))
    sums=''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.relative_to(root).as_posix()+'\n' for f in files)
    (root/'SHA256SUMS').write_bytes(sums.encode('utf-8'))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.out,'x',zipfile.ZIP_DEFLATED) as z:
        for f in files+[root/'SHA256SUMS']:
            z.write(f,arcname=root.name+'/'+f.relative_to(root).as_posix())
    fingerprint=hashlib.sha256(a.out.read_bytes()).hexdigest()
    sidecar=a.out.with_suffix(a.out.suffix+'.sha256')
    if sidecar.exists():
        raise FileExistsError('Checksum exists')
    with sidecar.open('x',encoding='ascii',newline='\n') as handle:
        handle.write(fingerprint+'  '+a.out.name+'\n')
    print('PACKAGED',a.out,'sha256=',fingerprint,'files=',len(files)+1)


if __name__=='__main__':
    main()
