"""Package the validated J1 code, formula guide and result snapshots."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'vgcpod_v1_roundJ'
HERE=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    shutil.copy2(ROOT/'VG-CPOD_v1_J组实验实施方案_20261021.html',PACKAGE/'EXPERIMENT_J_v1.html')
    layout=json.loads((HERE/'qa/layout.json').read_text(encoding='utf-8'))
    layout['visual_review']='Formula, loss, TRAIN-mean, RMS and command screenshots inspected'
    (PACKAGE/'validation/document_layout.json').write_text(json.dumps(layout,indent=2)+'\n',encoding='utf-8')
    text=(PACKAGE/'validation/unit_tests.txt').read_text(encoding='utf-8-sig')
    match=re.search(r'Ran (\d+) tests in ([0-9.]+)s\s+OK',text)
    assert match, 'Unit test log must finish successfully'
    tests=dict(tests=int(match[1]),seconds=float(match[2]),status='passed',device='CPU')
    (PACKAGE/'validation/unit_test_summary.json').write_text(json.dumps(tests,indent=2)+'\n',encoding='utf-8')
    refs=json.loads((PACKAGE/'references/pre_j_complete.json').read_text(encoding='utf-8'))['runs']
    integration=json.loads((PACKAGE/'validation/result.json').read_text(encoding='utf-8'))
    files=sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and p.suffix!='.pyc' and p.name!='release_manifest.json')
    manifest=dict(version='v1',stage='roundJ',date='2026-10-21',cases=['J1'],
        readouts=dict(static='Wc',dynamic='Wd',mode='separate',shared_factors=['Au','Av']),
        trainable_parameters=147456,added_parameters=24576,rho=.15,beta=.075,
        gate_input='train_residual',tau_gamma=.56,gate_eta=.25,rank=32,adapter_rank=8,
        seeds=[3351,3352,3353],epochs=10,
        clip='frozen OpenAI CLIP ViT-L/14; reuse UT-Zappos cache',
        gate_statistics='Shared all-TRAIN query mean, copied from completed I group when available',
        primary_selection='first strict max VAL CW AUC_native',
        secondary_selection='first strict max VAL CW AUC_exact',
        reference_runs=len(refs),reference_cases=sorted({r['case'] for r in refs}),
        tests=tests,integration=integration,
        files={p.relative_to(PACKAGE).as_posix():digest(p) for p in files})
    (PACKAGE/'release_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    archive=ROOT/'vgcpod_v1_roundJ_20261021.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files+[PACKAGE/'release_manifest.json']:
            name=p.relative_to(ROOT).as_posix()
            info=zipfile.ZipInfo(name)
            info.date_time=(2026,10,21,12,0,0)
            info.create_system=3
            info.external_attr=(0o100755 if p.suffix=='.sh' else 0o100644)<<16
            z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,expected in manifest['files'].items():
            assert hashlib.sha256(z.read(PACKAGE.name+'/'+name)).hexdigest()==expected,name
        assert len(z.namelist())==len(files)+1
        assert b'\r\n' not in z.read(PACKAGE.name+'/scripts/start_round_j_tmux.sh')
    sha=digest(archive)
    archive.with_suffix('.zip.sha256').write_text(sha+'  '+archive.name+'\n',encoding='ascii')
    report=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha,
        verified_files=len(files)+1,reference_runs=len(refs),unit_tests=tests['tests'])
    (HERE/'delivery.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
