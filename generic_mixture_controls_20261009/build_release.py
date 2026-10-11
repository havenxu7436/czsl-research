"""One-shot release builder after local CPU tests."""
import argparse
import ast
import zipfile
from pathlib import Path
from bootstrap import ROOT, child, read_json, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    sidecar = out.with_name(out.name+'.sha256')
    if out.exists() or sidecar.exists() or (ROOT/'release_manifest.json').exists():
        raise FileExistsError('Preserve sealed release; use a fresh source revision')
    report = read_json(ROOT/'local_test_report.json')
    if not report['passed'] or report['tests_run'] < 24 or report['real_cuda_run']:
        raise ValueError('Need at least 24 passing CPU-only tests')
    experiment = read_json(ROOT/'experiment.json')
    if experiment['source_old_gate'] or experiment['alpha'] != .25:
        raise ValueError('Unexpected source conclusion/readout')
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=path.name, feature_version=(3,8))
    files = sorted(ROOT.glob('*.py')) + [ROOT/'experiment.json', ROOT/'local_test_report.json']
    files += sorted((ROOT/'legacy').iterdir())
    files = [p for p in files if p.is_file() and p.name != 'release_manifest.json']
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    write_json(ROOT/'release_manifest.json', manifest)
    files.append(ROOT/'release_manifest.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, ROOT.name+'/'+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out) as archive:
        if archive.testzip() is not None:
            raise ValueError('Release ZIP CRC failed')
        for path in files:
            if archive.read(ROOT.name+'/'+path.relative_to(ROOT).as_posix()) != path.read_bytes():
                raise ValueError('Release member differs')
    with sidecar.open('x', encoding='ascii') as handle:
        handle.write(sha(out)+'  '+out.name+'\n')
    print('GENERIC_CONTROLS_RELEASE_SEALED', sha(out), out.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
