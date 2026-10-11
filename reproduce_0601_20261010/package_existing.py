"""Package the verified final source, preserving the earlier development ZIP."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path
from build_release import HERE, ROOT, OLD, patch_model, patch_cache, patch_train, sha


def main():
    for name, fn in [('models/kcc_cpf.py',patch_model), ('models/kcc_cache.py',patch_cache), ('train.py',patch_train)]:
        expected = fn((OLD / name).read_text(encoding='utf-8'))
        actual = (HERE / 'source' / name).read_text(encoding='utf-8')
        if expected != actual:
            raise ValueError('Generated source differs from tested patch ' + name)
        ast.parse(actual, feature_version=8)
    for name, target in [('full_feature.py','models/full_feature.py'), ('smoke_full.py','tests/smoke_full.py')]:
        if sha(HERE/name) != sha(HERE/'source'/target):
            raise ValueError('Generated helper differs ' + name)
    result = json.loads((HERE/'local_test.json').read_text(encoding='utf-8'))
    if result['passed'] is not True or result['real_cuda_training_performed'] is not False:
        raise ValueError('Missing honest passing local tests')
    hashes = {str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in sorted(HERE.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'release_manifest.json'}
    (HERE/'release_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    archive = ROOT/'output/0601_priority_release_20261010_r2.zip'
    if archive.exists():
        raise FileExistsError(str(archive))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(hashes):
            z.write(HERE/name,HERE.name+'/'+name)
        z.write(HERE/'release_manifest.json',HERE.name+'/release_manifest.json')
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print('FINAL_RELEASE_BUILT', archive, 'sha256', sha(archive))


if __name__ == '__main__':
    main()
