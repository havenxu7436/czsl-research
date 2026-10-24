"""Verify and restore the completed TRUST-CPF strong-baseline series, stages 02–04."""
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
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
SPECS={'02':('34','strong','强基底完整矩阵'), '03':('35','modelcond','模型响应条件化'), '04':('36','compact','Compact 紧凑效用监督')}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def stage(n):return ROOT/'TRUST-CPF方案'/n
def files(folder):return sorted(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'recomputed' not in p.parts)
def safe_new(destination):
    p=Path(destination).expanduser()
    if not p.is_absolute():raise ValueError('Destination must be absolute')
    p=p.resolve()
    if p.exists() or p==ROOT or ROOT in p.parents:raise FileExistsError('Choose a new directory outside the repository: '+str(p))
    return p
def safe_child(parent,relative):
    p=Path(relative)
    if p.is_absolute() or p.drive or '..' in p.parts:raise ValueError('Invalid relative path')
    return parent/p
def doctor(number,package):
    p=Path(package);m=read(p/'MANIFEST.json')
    changed=[n for n,h in m.items() if not safe_child(p,n).is_file() or sha(p/n)!=h]
    return dict(stage=number,ready_for_package_preflight=not changed,changed_package_members=changed,files_verified=len(m),training_started=False)
def materialize(number,destination):
    s=stage(number);d=safe_new(destination);m=read(s/'package_members.json');pkg=m['package']
    for n,p in m['members'].items():
        out=safe_child(d/pkg,n);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/p,out)
        if out.suffix=='.sh':out.chmod(out.stat().st_mode|0o111)
    result=doctor(number,d/pkg)
    if not result['ready_for_package_preflight']:raise ValueError(result)
    result.update(package=str(d/pkg),line='TRUST-CPF方案')
    write(d/'materialization.json',result)
    return result
def restore_archive(number,destination):
    s=stage(number);d=safe_new(destination);a=read(s/'results/archives/archive.json');d.mkdir(parents=True)
    out=d/a['name'];whole=hashlib.sha256();count=0
    with out.open('wb') as f:
        for part in a['parts']:
            h=hashlib.sha256();size=0
            with (s/part['path']).open('rb') as src:
                for b in iter(lambda:src.read(8*1024**2),b''):f.write(b);h.update(b);whole.update(b);size+=len(b)
            if h.hexdigest()!=part['sha256'] or size!=part['bytes']:raise ValueError('Archive part mismatch: '+part['path'])
            count+=size
    if count!=a['bytes'] or whole.hexdigest()!=a['sha256']:raise ValueError('Whole archive mismatch')
    out.with_name(out.name+'.sha256').write_text(a['sha256']+'  '+out.name+'\n',encoding='ascii')
    return dict(stage=number,archive=str(out),bytes=count,sha256=whole.hexdigest(),verified=True)
def check(number=None):
    results=[]
    for n in ([number] if number else SPECS):
        s=stage(n);member_map=read(s/'package_members.json');pkg=member_map['package'];mapping=member_map['members']
        for row in read(s/'file_manifest.json'):
            p=s/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
        raw=read(s/'results/raw_export_members.json')
        assert len(raw)=={'02':985,'03':325,'04':307}[n]
        for row in raw:assert sha(s/row['path'])==row['sha256'],row['path']
        a=read(s/'results/archives/archive.json');whole=hashlib.sha256();total=0
        for part in a['parts']:
            h=hashlib.sha256();size=0
            with (s/part['path']).open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):h.update(b);whole.update(b);size+=len(b)
            assert h.hexdigest()==part['sha256'] and size==part['bytes'];total+=size
        assert whole.hexdigest()==a['sha256'] and total==a['bytes']
        manifest=read(s/mapping['MANIFEST.json'])
        for name,h in manifest.items():assert sha(s/mapping[name])==h,name
        with zipfile.ZipFile(s/'results/deployment'/f'{pkg}.zip') as z:
            names=[x for x in z.namelist() if not x.endswith('/')]
            assert set(names)=={pkg+'/'+name for name in mapping}
            for name,p in mapping.items():assert z.read(pkg+'/'+name)==(s/p).read_bytes(),name
        inv=read(s/'configs/experiment_inventory.json');assert len(inv)==(30 if n=='02' else 9)
        for r in inv:
            folder=s/r['path'];m=read(folder/'metrics.json');assert sha(folder/'metrics.json')==r['metrics_sha256']
            assert (m['seed'],m['variant'],m['epoch'])==(r['seed'],r['variant'],r['selected_epoch'])
            for phase in ['val','test']:assert (folder/(phase+'_predictions.npz')).is_file()
            if 'reused_from' in r:
                for name in r['reuse_verified_files']:assert sha(folder/name)==sha(ROOT/r['reused_from']/name)
        status=read(s/f'results/raw_export/{pkg}/work/status.json');assert status['complete'] and not status['missing']
        for p in s.rglob('*.md'):
            t=p.read_text(encoding='utf8');assert t.count('```')%2==0,p
            prose=re.sub(r'(?:```[^\n]*\n[\s\S]*?```|~~~[^\n]*\n[\s\S]*?~~~)','',t)
            for link in re.findall(r'\]\((<[^>]+>|[^)\n]+)\)',prose):
                link=unquote(link.strip('<>')).split('#')[0]
                if link and not re.match(r'^[a-z]+://',link):assert (p.parent/link).exists(),(p,link)
        for p in (s/'code').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
        results.append(dict(stage=n,passed=True,server_members=len(raw),release_members=len(mapping),metrics=len(inv),reused=sum(x['result_kind']=='reused_control' for x in inv),status=status))
    return dict(passed=True,stages=results,original_bytes_preserved=True,training_started=False)
def finalize():
    stages=read(ROOT/'provenance/stages.json');stages=[r for r in stages if r['id'] not in ['34','35','36']]
    layout=read(ROOT/'provenance/layout.json');layout['files']=[r for r in layout['files'] if r.get('kind')!='trustcpf_strong_series_server_member']
    idx=ROOT/'provenance/metrics_index.csv'
    with idx.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=[r for r in reader if not r['route'].startswith('TRUST-strong-')]
    receipts=[]
    for n,(sid,key,title) in SPECS.items():
        s=stage(n);pkg=f'trustcpf_{key}_v1';inv=read(s/'configs/experiment_inventory.json');status=read(s/f'results/raw_export/{pkg}/work/status.json')
        inputs=dict(line='TRUST-CPF方案',stage=n,archive_stage=sid,title=title,primary_package=pkg,seeds=[3351,3352,3353],
            completed_results=len(inv),reused_controls=sum(x['result_kind']=='reused_control' for x in inv),formal_training_results_included=True,
            package_member_map='package_members.json',experiment_inventory='configs/experiment_inventory.json',status=status)
        for k,part in [('code_files','code'),('configs','configs'),('data_files','results'),('knowledge',f'configs/{pkg}/assets')]:inputs[k]=[p.relative_to(s).as_posix() for p in files(s/part)]
        inputs['documents']=[p.relative_to(s).as_posix() for p in s.rglob('*.md')];write(s/'inputs.json',inputs)
        commands=[]
        for name in ['README.md','SERVER_COMMANDS.md','analysis/REPRODUCE.md']:
            for language,cmd in re.findall(r'```(bash|powershell)\n(.*?)```',(s/name).read_text(encoding='utf8'),re.S):commands.append(dict(source=name,language=language,text=cmd))
        write(s/'runbook.json',dict(line='TRUST-CPF方案',stage=n,primary_package=pkg,commands=commands))
        stages.append(dict(id=sid,line='TRUST-CPF方案',number=n,version='v1',title=n+' · '+title,path='TRUST-CPF方案/'+n,primary_package=pkg,active_package=pkg,status='Completed; full source, results and analyses archived',completed_results=len(inv),reused_controls=inputs['reused_controls']))
        arc=read(s/'results/archives/archive.json')
        for r in read(s/'results/raw_export_members.json'):
            layout['files'].append(dict(original_path=arc['name']+'::'+r['runtime_path'],runtime_path=r['runtime_path'],current_path='TRUST-CPF方案/'+n+'/'+r['path'],sha256=r['sha256'],bytes=r['bytes'],stage=sid,kind='trustcpf_strong_series_server_member'))
        for run in inv:
            m=read(s/run['path']/'metrics.json')
            for phase in ['val','test']:
                for world in ['cw','ow']:
                    metric=m[phase][world];row={k:'' for k in fields}
                    row.update(stage=sid,route='TRUST-strong-'+n,dataset='ut-zap50k',seed=run['seed'],variant=run['variant'],phase=phase,setting=world,selected_epoch=run['selected_epoch'],auc_pct=metric['AUC_native'],hm_pct=metric['at_val_bias']['HM'],source='TRUST-CPF方案/'+n+'/'+run['path']+'/metrics.json',note='Native AUC; '+run['result_kind'])
                    rows.append(row)
        receipts.append(dict(stage=n,archive=arc,status=status,original_scientific_bytes_preserved=True))
        write(s/'file_manifest.json',[dict(path=p.relative_to(s).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files(s) if p.name!='file_manifest.json'])
    write(ROOT/'provenance/stages.json',stages);write(ROOT/'provenance/layout.json',layout)
    with idx.open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary=read(ROOT/'provenance/structure_summary.json');summary['line_numbering']['TRUST-CPF方案']='01 / v1、02–04';summary['documentation_updated']='2026-10-24';write(ROOT/'provenance/structure_summary.json',summary)
    receipt='provenance/updates/20261024_trustcpf_strong_series.json'
    write(ROOT/receipt,dict(date='2026-10-24',stages=receipts,total_result_records=48,reused_result_records=9,training_started=False,planned_only='Compact type calibration and half-density alignment; no new experiment claimed'))
    for p in (ROOT/'provenance/documents').glob('trustcpf_0*.json'):
        r=read(p);r['published_sha256']=sha(ROOT/r['document']);write(p,r)
    current={r['path']:r for r in read(ROOT/'provenance/current_manifest.json')}
    changed=set(subprocess.check_output(['git','diff','--name-only','-z','HEAD'],cwd=ROOT).decode('utf8').split('\0'))
    changed.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode('utf8').split('\0'))
    changed.update(p.relative_to(ROOT).as_posix() for n in SPECS for p in files(stage(n)))
    changed.update([receipt,'tools/register_trustcpf_strong_series.py','tools/analyze_trustcpf_strong_series.py','tools/check_trustcpf_web.py'])
    for relative in changed:
        p=ROOT/relative
        if relative and relative!='provenance/current_manifest.json' and p.is_file():current[relative]=dict(path=relative,sha256=sha(p),bytes=p.stat().st_size)
    write(ROOT/'provenance/current_manifest.json',[current[k] for k in sorted(current)])
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=list(SPECS));ap.add_argument('--check',action='store_true');ap.add_argument('--finalize',action='store_true');ap.add_argument('--materialize');ap.add_argument('--restore-archive')
    a=ap.parse_args()
    if a.finalize:finalize();result=check(a.stage)
    elif a.materialize or a.restore_archive:
        if not a.stage:ap.error('--stage is required')
        result=materialize(a.stage,a.materialize) if a.materialize else restore_archive(a.stage,a.restore_archive)
    elif a.check:result=check(a.stage)
    else:ap.error('Choose --check, --finalize, --materialize or --restore-archive')
    print(json.dumps(result,ensure_ascii=False))
