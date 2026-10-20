"""Package the D v1 sources, formula guide, and verification records."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'vgcpod_v1_roundD'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    shutil.copy2(ROOT/'VG-CPOD_v1_D组实验实施方案_20261020.html',PACKAGE/'EXPERIMENT_D_v1.html')
    shutil.copy2(ROOT/'analysis_roundD_20261020/qa/layout.json',PACKAGE/'validation/document_layout.json')
    files = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and p.suffix!='.pyc' and p.name!='release_manifest.json')
    manifest = dict(version='v1',stage='roundD',date='2026-10-20',experiments=['D1'],
        c2_repeats=dict(seeds=[3352,3353],source='/data/xhw/vgcpod_project/vgcpod_v1_roundC'),
        mean_gate_checkpoints=dict(C2=[3351,3352,3353],D1=[3351]),
        clip='frozen OpenAI CLIP ViT-L/14; existing UT-Zappos cache',
        validation=dict(tests=30,cli_cpu_training_jobs=3,mean_gate_evaluations=4,
            repeat_training_skipped=3,repeat_evaluation_skipped=4,original_c_results_unchanged=True),
        files_sha256={p.relative_to(PACKAGE).as_posix():sha(p) for p in files})
    target = PACKAGE/'release_manifest.json'
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files.append(target)
    archive = ROOT/'vgcpod_v1_roundD_20261020.zip'
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
    print(json.dumps(dict(archive=archive.name,files=len(files),bytes=archive.stat().st_size,sha256=sha(archive)),indent=2))


if __name__=='__main__':
    main()
