import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

base = Path(__file__).resolve().parents[1]
release = base / 'trustcpf_state_v1_10ep'
output = base / 'output'
output.mkdir(exist_ok=True)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def include(path):
    rel = path.relative_to(release)
    return (path.is_file() and '__pycache__' not in rel.parts
            and rel.parts[0] not in ('work', 'logs')
            and not rel.as_posix().startswith('validation/real_fit/')
            and rel.as_posix() not in ('server_env.sh', 'setup.json', 'download.json', 'MANIFEST.json')
            and path.suffix not in ('.pyc', '.pyo', '.writing', '.lock'))

files = sorted(p for p in release.rglob('*') if include(p))
manifest = {p.relative_to(release).as_posix(): digest(p.read_bytes()) for p in files}
(release / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='ascii', newline='\n')
files.append(release / 'MANIFEST.json')
archive = output / (release.name + '.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files:
        arcname = release.name + '/' + p.relative_to(release).as_posix()
        assert arcname.isascii(), arcname
        z.write(p, arcname)
sha256 = digest(archive.read_bytes())
Path(str(archive) + '.sha256').write_bytes((sha256 + '  ' + archive.name + '\n').encode('ascii'))

with tempfile.TemporaryDirectory(prefix='trustcpf_release_verify_') as temp:
    target = Path(temp)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert not any('real_fit/' in n or '__pycache__' in n or n.endswith('/server_env.sh') for n in names)
        z.extractall(target)
    extracted = target / release.name
    for name, expected in manifest.items():
        assert digest((extracted / name).read_bytes()) == expected, name
    assert json.loads((extracted / 'configs/default.json').read_text())['adapt_epochs'] == 10
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, 'run.py', '--help'], cwd=extracted, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    check = subprocess.run([sys.executable, '-c',
        "from pathlib import Path; from trustcpf import state_workflow, state; "
        "assert Path(state_workflow.__file__).resolve().parent.parent == Path.cwd(); "
        "assert state.SEEDS == (3351,3352,3353); print('EXTRACTED_IMPORT_OK')"],
        cwd=extracted, env=env, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr

copies = {
    'SERVER_COMMANDS.md': 'TRUST-CPF_第一批_10epoch_服务器实践命令.md',
    'FORMULAS_AND_CODE.md': 'TRUST-CPF_第一批_10epoch_公式与核心代码.md',
    'README.md': 'TRUST-CPF_第一批_10epoch_交付说明.md',
}
for name, destination in copies.items():
    shutil.copyfile(release / name, output / destination)
info = dict(archive=str(archive), sha256=sha256, bytes=archive.stat().st_size,
            files=len(files), manifest_verified=True, zip_crc_verified=True,
            ascii_paths=True, checksum_lf=True, extracted_cli_and_import=True,
            excluded_generated_setup=True)
(Path(__file__).parent / 'release_info.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf8')
print(json.dumps(info, indent=2))
