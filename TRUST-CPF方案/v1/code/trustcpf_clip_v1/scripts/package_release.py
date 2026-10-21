"""Create the server ZIP and verify every archived file against its manifest."""
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for p in (ROOT/'scripts').glob('*.sh'):
        p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    log=(ROOT/'validation/tests_cpu.log').read_text(encoding='utf-8',errors='replace')
    count=int(re.findall(r'Ran (\d+) tests',log)[-1])
    assert re.search(r'\nOK\s*$',log)
    result={'test_count':count,'cpu_tests':'passed','cuda_smoke':json.loads((ROOT/'validation/smoke_cuda.json').read_text(encoding='utf-8')),
            'formula_render':json.loads((ROOT/'validation/formula_render_checks.json').read_text(encoding='utf-8')),
            'experiment_results':'UT training is run by the server commands',
            'python':sys.version}
    (ROOT/'validation/result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    inputs=[Path(r'C:\Users\llenovo\Desktop\USG\idea\USG-CPF_3.0_完整实验方案_v1.md'),
            Path(r'C:\Users\llenovo\Desktop\VG-CPOD\v1\VG-CPOD公式及代码v1.md')]
    references={str(p):sha(p) for p in inputs if p.is_file()}
    if not references and (ROOT/'provenance.json').exists():
        references=json.loads((ROOT/'provenance.json').read_text(encoding='utf-8'))['reference_documents']
    provenance={'date':'2026-10-21','method':'TRUST-CPF / CLIP v1',
        'reference_documents':references,
        'reused_modules':'UT data mapping, CLIP extraction, AUC from local VG-CPOD v1; revised in this release',
        'clip_commit':'a1d071733d7111c9c014f024669f959182114e33',
        'knowledge':{p.name:sha(p) for p in (ROOT/'assets/knowledge').glob('*.json')}}
    (ROOT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    paths=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
           and p.suffix!='.pyc' and p.name!='release_manifest.json']
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    mp=ROOT/'release_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    target=ROOT.parent/'output/TRUST-CPF_CLIP_方案v1.zip';target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in paths+[mp]:archive.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name,fingerprint in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==fingerprint,name
    target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n',encoding='utf-8')
    print('RELEASE',len(manifest)+1,'files',target.stat().st_size,'bytes',target)


if __name__=='__main__':main()
