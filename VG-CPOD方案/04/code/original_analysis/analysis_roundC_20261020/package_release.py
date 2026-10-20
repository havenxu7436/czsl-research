"""Assemble the C-group v1 upload archive with its documentation and validation records."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'vgcpod_v1_roundC'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    shutil.copy2(ROOT/'VG-CPOD_v1_C组实验实施方案_20261020.html',PACKAGE/'EXPERIMENT_C_v1.html')
    shutil.copy2(ROOT/'analysis_roundC_20261020/qa/layout.json',PACKAGE/'validation/document_layout.json')
    text = (PACKAGE/'EXPERIMENT_C_v1.md').read_text(encoding='utf-8')
    sys.path.insert(0,str(ROOT/'tmp/round_a_bash_validation'))
    import bashlex
    blocks = re.findall(r'```bash\n(.*?)\n```',text,re.S)
    for block in blocks:
        bashlex.parse(block)
    files = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
                   p.suffix!='.pyc' and p.name!='release_manifest.json')
    manifest = dict(version='v1',stage='roundC',date='2026-10-20',experiments=['C1','C2'],
        companion_job=dict(case='B3',seed=3352,source='/data/xhw/vgcpod_project/vgcpod_v1_roundB'),
        clip='frozen OpenAI CLIP ViT-L/14, reuse existing cache',
        validation=dict(tests=26,bitwise_legacy_cases=12,cli_cpu_jobs=3,repeated_jobs_skipped=3,
                        b3_original_source_verified=True,documented_bash_blocks=len(blocks)),
        files_sha256={p.relative_to(PACKAGE).as_posix():sha(p) for p in files})
    target = PACKAGE/'release_manifest.json'
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files.append(target)
    archive = ROOT/'vgcpod_v1_roundC_20261020.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(files):
            info = zipfile.ZipInfo.from_file(file,arcname=file.relative_to(ROOT).as_posix())
            info.create_system = 3
            info.external_attr = (0o100755 if file.suffix=='.sh' else 0o100644)<<16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info,file.read_bytes())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,fingerprint in manifest['files_sha256'].items():
            assert hashlib.sha256(z.read(PACKAGE.name+'/'+name)).hexdigest()==fingerprint,name
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(dict(archive=archive.name,bytes=archive.stat().st_size,files=len(files),
                         sha256=sha(archive),bash_blocks=len(blocks)),indent=2))


if __name__=='__main__':
    main()
