"""Create the F v1 upload archive with source and document checksums."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'vgcpod_v1_roundF'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    shutil.copy2(ROOT/'VG-CPOD_v1_F组实验实施方案_20261020.html',PACKAGE/'EXPERIMENT_F_v1.html')
    shutil.copy2(ROOT/'analysis_roundF_20261020/qa/layout.json',PACKAGE/'validation/document_layout.json')
    files = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and p.suffix!='.pyc' and p.name!='release_manifest.json')
    manifest = dict(version='v1',stage='roundF',date='2026-10-20',case='F1',rho=.15,beta=.05,
        seeds=[3351,3352,3353],clip='frozen OpenAI CLIP ViT-L/14; reuse UT-Zappos cache',
        primary_selection='first strict max VAL CW AUC_native',
        secondary_selection='first strict max VAL CW AUC_exact',
        calibration='Each checkpoint/condition uses its own VAL CW/OW native bias on TEST',
        reference_runs=17,validation=dict(unit_tests=36,cpu_training_jobs=3,cpu_mean_gate_jobs=6,
            repeat_training_skipped=3,repeat_mean_gate_skipped=6,model_py_unchanged=True,
            legacy_parameters_adamw_and_torch_rng_bitwise_equal=True),
        files_sha256={p.relative_to(PACKAGE).as_posix():sha(p) for p in files})
    target = PACKAGE/'release_manifest.json'
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files.append(target)
    archive = ROOT/'vgcpod_v1_roundF_20261020.zip'
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
    print(json.dumps(dict(archive=archive.name,bytes=archive.stat().st_size,files=len(files),sha256=sha(archive)),indent=2))


if __name__=='__main__':
    main()
