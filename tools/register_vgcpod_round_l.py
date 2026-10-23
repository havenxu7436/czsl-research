"""Verify, restore and index VG-CPOD experiment 13 (L group)."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'VG-CPOD方案/13'
PACKAGE='vgcpod_v1_roundL'
ORIGINAL='trustcpf_conditional_v2'
ARCHIVE='trustcpf_conditional_v2_results_20261022_232124.tar.gz'
TITLE='G0条件知识残差与语义附件对照'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pointer(p):
    return p.stat().st_size<200 and p.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1')

def finalize():
    paths=sorted(p for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
    inputs=dict(line='VG-CPOD方案',stage='13',version='v1',archive_stage='33',title=TITLE,primary_package=PACKAGE,
        completed_runs=9,model_epochs=90,predictions=18,checkpoint_files=18,initial_checkpoints=3,
        completed_cases=['L0','L1','L2'],completed_seeds=[3351,3352,3353],gpu_schedule={'1':3351,'2':3352,'3':3353},
        formal_training_results_included=True,original_server_members=181,
        run_matrix=[dict(case=c,variant=v,seed=s) for c,v in [('L0','base'),('L1','knowledge'),('L2','shuffle')] for s in [3351,3352,3353]],
        package_member_map='package_members.json',experiment_inventory='configs/experiment_inventory.json')
    for key,sub in [('code_files','code'),('configs','configs'),('data_files','results')]:
        inputs[key]=[p.relative_to(STAGE).as_posix() for p in paths if p.relative_to(STAGE).parts[0]==sub]
    inputs['documents']=[p.relative_to(STAGE).as_posix() for p in paths if p.suffix=='.md']
    write(STAGE/'inputs.json',inputs)
    commands=[]
    for name in ['README.md','SERVER_COMMANDS.md']:
        for language,text in re.findall(r'```(bash|powershell)\r?\n(.*?)```',(STAGE/name).read_text(encoding='utf8'),re.S):
            commands.append(dict(source=name,language=language,text=text))
    write(STAGE/'runbook.json',dict(line='VG-CPOD方案',stage='13',primary_package=PACKAGE,commands=commands))
    stages=read(ROOT/'provenance/stages.json')
    stages=[s for s in stages if s['id']!='33']
    stages.append(dict(id='33',line='VG-CPOD方案',number='13',version='v1',title='13 · '+TITLE,
        path='VG-CPOD方案/13',primary_package=PACKAGE,active_package=PACKAGE,
        status='L0/L1/L2各三seed、完整结果与分析归档；下一步视觉比较仅为提案',
        completed_runs=9,model_epochs=90,predictions=18,checkpoint_files=18))
    write(ROOT/'provenance/stages.json',stages)
    update=ROOT/'provenance/updates/20261023_vgcpod_13.json'
    write(update,dict(date='2026-10-23',line='VG-CPOD方案',number='13',version='v1',archive_stage='33',title=TITLE,
        reclassified_experiment=True,original_runtime_label=ORIGINAL,current_runtime_label=PACKAGE,
        original_server_archive_sha256=sha(STAGE/'results/archives'/ARCHIVE),original_server_members=181,
        original_scientific_bytes_preserved=True,active_scientific_operations_changed=False,
        completed_runs=9,completed_cases=['L0','L1','L2'],seeds=[3351,3352,3353],
        removed_route_version=True,planned_only=['Trainable CLIP Global/Pool-first/Compare-first, 9 proposed fits; no implementation or training']))
    layout=read(ROOT/'provenance/layout.json')
    layout['files']=[r for r in layout['files'] if r.get('kind')!='vgcpod_round_l_server_member']
    for r in read(STAGE/'results/raw_export_members.json'):
        layout['files'].append(dict(original_path=ARCHIVE+'::'+r['runtime_path'],runtime_path=r['runtime_path'],
            current_path='VG-CPOD方案/13/'+r['path'],sha256=r['sha256'],bytes=r['bytes'],stage='33',kind='vgcpod_round_l_server_member'))
    write(ROOT/'provenance/layout.json',layout)
    summary=read(ROOT/'provenance/structure_summary.json')
    summary.update(documentation_updated='2026-10-23',research_lines=['前期实验','0601方案','VG-CPOD方案','TRUST-CPF方案'])
    summary['line_numbering']={'前期实验':'01–15','0601方案':'01–04','VG-CPOD方案':'01–13','TRUST-CPF方案':'01 / v1'}
    write(ROOT/'provenance/structure_summary.json',summary)
    idx=ROOT/'provenance/metrics_index.csv'
    with idx.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=[r for r in reader if r.get('route')!='VG-CPOD-L']
    for run in read(STAGE/'configs/experiment_inventory.json'):
        m=read(STAGE/run['path']/'metrics.json')
        for phase in ['val','test']:
            for world in ['cw','ow']:
                metric=m[phase][world];row={k:'' for k in fields}
                row.update(stage='33',route='VG-CPOD-L',dataset='ut-zap50k',seed=run['seed'],variant=run['case']+'/'+run['variant'],
                    phase=phase,setting=world,selected_epoch=run['selected_epoch'],auc_pct=metric['AUC_native'],
                    hm_pct=metric['at_val_bias']['HM'],source='VG-CPOD方案/13/'+run['path']+'/metrics.json',note='Native AUC; TEST HM at VAL bias')
                rows.append(row)
    with idx.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    # Remove stale references from v1 indexes while preserving all v1 scientific results.
    v1=ROOT/'TRUST-CPF方案/v1'
    ip=v1/'inputs.json';iv=read(ip)
    for key in ['knowledge','configs','code_files','data_files','documents']:
        iv[key]=[p for p in iv.get(key,[]) if (ROOT/p).is_file()]
    write(ip,iv)
    vp=v1/'results/server-results-20261022/file_manifest.json';vm=read(vp)
    for row in vm:
        p=ROOT/row['path']
        if p.is_file() and not pointer(p):row.update(sha256=sha(p),bytes=p.stat().st_size)
    write(vp,[r for r in vm if (ROOT/r['path']).is_file()])
    for p in (ROOT/'provenance/documents').glob('vgcpod_13_*.json'):
        x=read(p);x['published_sha256']=sha(ROOT/x['document']);write(p,x)
    paths=sorted(p for p in STAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
    write(STAGE/'file_manifest.json',[dict(path=p.relative_to(STAGE).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
        for p in paths if p.name!='file_manifest.json'])
    current={r['path']:r for r in read(ROOT/'provenance/current_manifest.json') if (ROOT/r['path']).is_file()}
    modified=subprocess.check_output(['git','diff','--name-only','-z'],cwd=ROOT).decode('utf8').split('\0')
    owned=set(modified)|{p.relative_to(ROOT).as_posix() for p in paths}|{
        'VG-CPOD方案/13/file_manifest.json','tools/register_vgcpod_round_l.py','provenance/updates/20261023_vgcpod_13.json'}
    owned.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'provenance/documents').glob('vgcpod_13_*.json'))
    for relative in owned:
        if not relative or relative=='provenance/current_manifest.json':continue
        p=ROOT/relative
        if p.is_file() and not pointer(p):current[relative]=dict(path=relative,sha256=sha(p),bytes=p.stat().st_size)
    write(ROOT/'provenance/current_manifest.json',[current[p] for p in sorted(current)])

def check():
    for row in read(STAGE/'file_manifest.json'):
        assert sha(STAGE/row['path'])==row['sha256'],row['path']
    raw_members=read(STAGE/'results/raw_export_members.json')
    assert len(raw_members)==181
    for r in raw_members:assert sha(STAGE/r['path'])==r['sha256'],r['path']
    arc=STAGE/'results/archives'/ARCHIVE
    assert sha(arc)==arc.with_name(arc.name+'.sha256').read_text().split()[0]
    original=read(STAGE/'results/original_release_member_map.json')
    manifest=read(ROOT/original['members']['MANIFEST.json'])
    assert len(original['members'])==40
    for n,h in manifest.items():assert sha(ROOT/original['members'][n])==h,n
    inv=read(STAGE/'configs/experiment_inventory.json');assert len(inv)==9
    for r in inv:
        folder=STAGE/r['path'];m=read(folder/'metrics.json');done=read(folder/'DONE.json');history=read(folder/'history.json')
        assert done['metrics_sha256']==sha(folder/'metrics.json')
        assert len(history)==10 and m['selected_epoch']==max(range(10),key=lambda i:history[i]['val_cw_auc'])+1
        for name in ['best.pt','last.pt','val_predictions.npz','test_predictions.npz','val_records.json','test_records.json']:assert (folder/name).is_file()
    mapping=read(STAGE/'package_members.json')['members']
    with zipfile.ZipFile(STAGE/'results/deployment'/f'{PACKAGE}.zip') as z:
        assert set(z.namelist())=={PACKAGE+'/'+n for n in mapping}
        for n,p in mapping.items():assert z.read(PACKAGE+'/'+n)==(STAGE/p).read_bytes(),n
    active=read(STAGE/mapping['MANIFEST.json'])
    for n,h in active.items():assert sha(STAGE/mapping[n])==h,n
    docs=[p for p in STAGE.rglob('*.md') if 'raw_export' not in p.parts]
    for p in docs:
        text=p.read_text(encoding='utf8');assert text.count('```')%2==0,p
        for link in re.findall(r'\]\((<[^>]+>|[^)\n]+)\)',text):
            link=link.strip('<>').split('#')[0]
            if link and not re.match(r'^[a-z]+://',link):assert (p.parent/link).exists(),(p,link)
    assert not (ROOT/'TRUST-CPF方案/v2').exists()
    trust_docs=list((ROOT/'TRUST-CPF方案').rglob('*.md'))
    forbidden=re.compile(r'TRUST[- ]?CPF\s*v2|TRUST-CPF方案[/\\]v2|\[v2[^\]]*\]|conditional_knowledge_v2|后续\s*v2')
    for p in trust_docs:assert not forbidden.search(p.read_text(encoding='utf8')),p
    for folder in ['code','analysis']:
        for p in (STAGE/folder).rglob('*.py'):
            if '__pycache__' not in p.parts:ast.parse(p.read_text(encoding='utf-8-sig'))
    return dict(passed=True,stage='VG-CPOD方案/13',completed_runs=9,original_server_members=181,
        original_release_members=40,active_release_members=len(mapping),predictions=18,checkpoint_files=18,
        original_bytes_preserved=True,active_scientific_operations_changed=False,retired_route_removed=True,training_started=False)

def materialize(destination):
    destination=Path(destination).expanduser()
    if not destination.is_absolute():raise ValueError('Choose an absolute destination')
    destination=destination.resolve()
    if destination.exists() or destination==ROOT or ROOT in destination.parents:raise FileExistsError(destination)
    mapping=read(STAGE/'package_members.json')['members']
    for n,p in mapping.items():
        target=destination/PACKAGE/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(STAGE/p,target)
        if target.suffix=='.sh':target.chmod(target.stat().st_mode|0o111)
    assert doctor(destination/PACKAGE)['ready_for_package_preflight']
    write(destination/'materialization.json',dict(line='VG-CPOD方案',stage='13',package=PACKAGE,files_verified=len(mapping),training_started=False))
    return dict(line='VG-CPOD方案',stage='13',package=str(destination/PACKAGE),files_verified=len(mapping),training_started=False)

def doctor(package):
    package=Path(package).resolve()
    changed=[n for n,h in read(package/'MANIFEST.json').items() if not (package/n).is_file() or sha(package/n)!=h]
    return dict(line='VG-CPOD方案',stage='13',package=str(package),changed_package_members=changed,ready_for_package_preflight=not changed,training_started=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');ap.add_argument('--finalize',action='store_true');ap.add_argument('--materialize',type=Path)
    args=ap.parse_args()
    if args.finalize:finalize();result=check()
    elif args.materialize:result=materialize(args.materialize)
    elif args.check:result=check()
    else:ap.error('Choose --check, --finalize or --materialize')
    print(json.dumps(result,ensure_ascii=False))
