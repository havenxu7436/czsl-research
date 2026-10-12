"""Package a checked Stage A release; refuse overwriting an existing archive."""
import argparse
import json
import zipfile
from pathlib import Path
from common import ROOT, read_json, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or Path(str(args.out) + '.sha256').exists():
        raise FileExistsError('Release ZIP already exists')
    if read_json(ROOT / 'local_test_report.json')['passed'] is not True:
        raise ValueError('Local tests failed')
    files = sorted(p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py', '.json'} and p.name != 'release_manifest.json')
    write_json(ROOT / 'release_manifest.json', {p.name: sha(p) for p in files})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files + [ROOT / 'release_manifest.json']:
            z.write(p, ROOT.name + '/' + p.name)
    Path(str(args.out) + '.sha256').write_text(sha(args.out) + '  ' + args.out.name + '\n', encoding='utf-8')
    print('RELEASE_BUILT', sha(args.out), args.out)


if __name__ == '__main__':
    main()
