"""Read-only identity check for the existing CPF server inputs (no training)."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/xhw/czsl_repro'))
    parser.add_argument('--release-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'code/faithful_0601_20261011')
    args = parser.parse_args()
    release, root = args.release_dir.resolve(), args.root.resolve()
    manifest = json.loads((release / 'assets/legacy_cache_manifest.json').read_text(encoding='utf-8'))
    origins = json.loads((release / 'assets/source_origin.json').read_text(encoding='utf-8'))
    cache = root / 'knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k'
    targets = {cache / 'manifest.json': digest(release / 'assets/legacy_cache_manifest.json')}
    targets.update({cache / name: value for name, value in manifest['files_sha256'].items()})
    original_root = '/data/xhw/czsl_repro/'
    for name, value in manifest['input_sha256'].items():
        if not name.startswith(original_root):
            raise ValueError('Unexpected input outside original project: ' + name)
        targets[root / name[len(original_root):]] = value
    for name in ('models/ddm.py', 'models/word_embedding_utils.py', 'models/ddm_utils.py',
                 'dataset.py', 'evaluator_ge.py', 'config/defaults.py'):
        targets[root / 'src/KCC0601_v1' / name] = origins[name]
    targets[root / 'data/ut-zap50k/metadata_compositional-split-natural.t7'] = (
        '7fbe052a107170ad28f5b85d2237e584c7ea306ca791844e691606e89635aa5b')
    issues = []
    for path, expected in targets.items():
        if not path.is_file():
            issues.append({'path': str(path), 'problem': 'missing'})
        elif digest(path) != expected:
            issues.append({'path': str(path), 'problem': 'sha256 mismatch'})
    result = {'read_only': True, 'files_checked': len(targets), 'passed': not issues,
              'issues': issues, 'cuda_training_performed': False,
              'notice': 'Does not verify all images, pretrained weights, imports, CUDA or relocated prepare compatibility.'}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
