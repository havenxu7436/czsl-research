"""Create the v1 B-group upload archive and its content manifest."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'vgcpod_v1_roundB'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    shutil.copy2(ROOT / 'VG-CPOD_v1_B组实验实施方案_20261019.html', PACKAGE / 'EXPERIMENT_B_v1.html')
    shutil.copy2(ROOT / 'analysis_roundB_20261019/qa/layout.json', PACKAGE / 'validation/document_layout.json')
    text = (PACKAGE / 'EXPERIMENT_B_v1.md').read_text(encoding='utf-8')
    sys.path.insert(0, str(ROOT / 'tmp/round_a_bash_validation'))
    import bashlex
    bash_blocks = re.findall(r'```bash\n(.*?)\n```', text, re.S)
    for block in bash_blocks:
        bashlex.parse(block)
    files = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
                   p.suffix != '.pyc' and p.name != 'release_manifest.json')
    manifest = dict(version='v1', stage='roundB', date='2026-10-19',
        experiments=['B1', 'B2', 'B3'], clip='frozen OpenAI CLIP ViT-L/14; reuse cache',
        validation=dict(unit_and_integration_tests=22, legacy_bitwise_variants=9,
                        cli_cpu_parallel_jobs=3, cli_repeat_skipped=3, documented_bash_blocks=len(bash_blocks)),
        files_sha256={p.relative_to(PACKAGE).as_posix(): sha(p) for p in files})
    target = PACKAGE / 'release_manifest.json'
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    files.append(target)
    archive = ROOT / 'vgcpod_v1_roundB_20261019.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in sorted(files):
            info = zipfile.ZipInfo.from_file(file, arcname=file.relative_to(ROOT).as_posix())
            info.create_system = 3
            info.external_attr = (0o100755 if file.suffix == '.sh' else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, file.read_bytes())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for file, expected in manifest['files_sha256'].items():
            assert hashlib.sha256(z.read(PACKAGE.name + '/' + file)).hexdigest() == expected, file
    archive.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='ascii')
    print(json.dumps(dict(archive=archive.name, bytes=archive.stat().st_size, files=len(files),
                         sha256=sha(archive), bash_blocks=len(bash_blocks)), indent=2))


if __name__ == '__main__':
    main()
