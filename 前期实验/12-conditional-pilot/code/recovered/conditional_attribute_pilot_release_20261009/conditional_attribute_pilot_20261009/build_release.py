"""One-shot immutable release ZIP builder, after successful local synthetic tests."""
import argparse
import ast
import zipfile
from pathlib import Path
from common import ROOT, read_json, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    destination = args.out.resolve()
    sidecar = destination.with_name(destination.name + '.sha256')
    if destination.exists() or sidecar.exists() or (ROOT / 'release_manifest.json').exists():
        raise FileExistsError('Do not overwrite a sealed release')
    report = read_json(ROOT / 'local_test_report.json')
    if not report['passed'] or report['tests_run'] < 23 or report['real_cuda_pilot_run']:
        raise ValueError('Need complete CPU-only synthetic tests, not a fabricated CUDA claim')
    reference = read_json(ROOT / 'reference.json')
    if sha(ROOT / 'head.py') != reference['audited_head_sha256']:
        raise ValueError('The Stage A head must remain unchanged')
    files = sorted(ROOT.glob('*.py')) + [ROOT / name for name in ('reference.json', 'stageA_audit.json', 'local_test_report.json')]
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=path.name, feature_version=(3, 8))
    write_json(ROOT / 'release_manifest.json', {path.name: sha(path) for path in files})
    files.append(ROOT / 'release_manifest.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, ROOT.name + '/' + path.name)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Release ZIP CRC failed')
        for path in files:
            if archive.read(ROOT.name + '/' + path.name) != path.read_bytes():
                raise ValueError('Release contents differ')
    with sidecar.open('x', encoding='ascii') as handle:
        handle.write(sha(destination) + '  ' + destination.name + '\n')
    print('PILOT_RELEASE_SEALED', sha(destination), destination.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
