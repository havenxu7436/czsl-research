"""Verify packaged bytes and entrypoints, not just the development directory."""
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'output/0601_formula_release_20261011_r2.zip'
expected = archive.with_suffix('.zip.sha256').read_text(encoding='utf-8').split()[0]
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise ValueError('ZIP checksum failed')
with tempfile.TemporaryDirectory(prefix='0601_formula_delivery_') as temporary:
    dest = Path(temporary)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failed')
        manifest = json.loads(z.read('faithful_0601_20261011/release_manifest.json'))
        for name, digest in manifest.items():
            if hashlib.sha256(z.read('faithful_0601_20261011/' + name)).hexdigest() != digest:
                raise ValueError('Manifest failed: ' + name)
        z.extractall(dest)
    release = dest / 'faithful_0601_20261011'
    help_checks = []
    for name in ('prepare.py', 'run_suite.py', 'report.py', 'status.py'):
        result = subprocess.run([sys.executable, str(release / name), '--help'], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(name + ': ' + result.stderr)
        help_checks.append(name)
    result = subprocess.run([sys.executable, str(release / 'test_formula.py')], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    tests = json.loads((release / 'local_test.json').read_text(encoding='utf-8'))
report = {'zip_sha256': expected, 'manifest_files_verified': len(manifest), 'zip_crc_passed': True,
          'extracted_entrypoint_help_passed': help_checks, 'extracted_cpu_tests': tests,
          'server_cuda_training_performed': False}
output = ROOT / 'output/0601_formula_delivery_verification_20261011.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
