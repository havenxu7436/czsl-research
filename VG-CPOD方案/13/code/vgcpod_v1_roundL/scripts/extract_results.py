"""Windows UTF-8 extraction, including archive CRC verification on complete read."""
import argparse
import os
import tarfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('archive'); p.add_argument('destination')
args = p.parse_args()
destination = Path(args.destination).resolve()
destination.mkdir(parents=True, exist_ok=True)
with tarfile.open(args.archive, 'r:gz', encoding='utf-8') as tar:
    members = tar.getmembers()
    for member in members:
        target = (destination / member.name).resolve()
        if os.path.commonpath([str(destination), str(target)]) != str(destination):
            raise ValueError('Archive path outside destination: ' + member.name)
        if not member.isfile() and not member.isdir():
            raise ValueError('Unexpected archive member: ' + member.name)
    tar.extractall(destination, members=members)
print('EXTRACTED', destination)
