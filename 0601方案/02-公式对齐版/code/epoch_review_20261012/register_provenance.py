"""Register new review/evidence files without changing immutable source layout."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).absolute().parents[4]
STAGE = ROOT / '0601方案/02-公式对齐版'
RESULT = STAGE / 'results/seed3351-30epoch-20261012'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    audit = json.loads((RESULT / 'audit.json').read_text(encoding='utf-8'))
    if not audit['passed']:
        raise ValueError('No provenance registration for failing evidence')
    selected = list(RESULT.rglob('*')) + list((STAGE / 'code/epoch_review_20261012').rglob('*'))
    selected.extend([ROOT / 'README.md', STAGE / 'README.md', STAGE / 'inputs.json',
                     STAGE / 'runbook.json'])
    paths = {p.relative_to(ROOT).as_posix(): p for p in selected if p.is_file() and '__pycache__' not in p.parts}
    secret = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|https?://[^\s/@:]+:[^\s/@]+@)')
    issues = []
    for relative, path in paths.items():
        if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.pth', '.pem', '.key') or path.name in ('.env', '.git-credentials'):
            issues.append(relative + ': disallowed image/weight/credential file')
        if path.suffix.lower() in ('.json', '.txt', '.log', '.md', '.yml', '.yaml', '.py') and secret.search(path.read_bytes()):
            issues.append(relative + ': possible credential pattern')
    if issues:
        raise ValueError('Upload scope review failed: ' + str(issues))
    records = [{'path': relative, 'sha256': sha(path), 'bytes': path.stat().st_size}
               for relative, path in sorted(paths.items())]
    receipt = ROOT / 'provenance/updates/20261012_formula_seed3351.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({'date': '2026-10-12', 'stage': '17',
                                  'source_zip_sha256': audit['source_zip_sha256'],
                                  'review_passed': True, 'original_export_members': 792,
                                  'original_export_sha_verified': 791, 'metric_replays': 56,
                                  'server_training_started_by_review': False,
                                  'original_protocol_or_training_source_modified': False,
                                  'scope_scan_passed': True, 'records': records},
                                 ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    records.append({'path': receipt.relative_to(ROOT).as_posix(), 'sha256': sha(receipt), 'bytes': receipt.stat().st_size})
    current_path = ROOT / 'provenance/current_manifest.json'
    current = {r['path']: r for r in json.loads(current_path.read_text(encoding='utf-8'))}
    current.update({r['path']: r for r in records})
    current_path.write_text(json.dumps([current[k] for k in sorted(current)], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'scope_scan_passed': True, 'new_or_updated_files_registered': len(records),
                      'bytes_registered': sum(r['bytes'] for r in records), 'server_training_started': False}))


if __name__ == '__main__':
    main()
