"""Verify, restore and register completed TRUST-CPF stages 05 (Align) and 06 (Competition)."""
import argparse,ast,csv,hashlib,json,re,shutil,subprocess,zipfile
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[1]
SPECS={'05':('37','trustcpf_align_v1','Align 半密度对齐',18,9),
       '06':('38','trustcpf_competition_v1_10ep','Competition 三目标与 10 epoch',42,15)}
_CACHE={}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):
 p=Path(p);st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
 if key not in _CACHE:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
  _CACHE[key]=h.hexdigest()
 return _CACHE[key]
def stage(n):return ROOT/'TRUST-CPF方案'/n
def files(p):return sorted(x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts)
def safe_child(p,relative):
 relative=Path(relative)
 if relative.is_absolute() or relative.drive or '..' in relative.parts:raise ValueError('Unsafe relative path')
 return p/relative
def safe_new(destination):
 p=Path(destination).expanduser()
 if not p.is_absolute():raise ValueError('Destination must be absolute')
 p=p.resolve()
 if p.exists() or p==ROOT or ROOT in p.parents:raise FileExistsError('Choose a new directory outside the repository')
 return p
def doctor(number,package):
 p=Path(package);m=read(p/'MANIFEST.json');changed=[n for n,h in m.items() if not safe_child(p,n).is_file() or sha(p/n)!=h]
 return dict(stage=number,ready_for_package_preflight=not changed,changed_package_members=changed,files_verified=len(m),training_started=False)
def materialize(number,destination):
 s=stage(number);dest=safe_new(destination);mapping=read(s/'package_members.json');pkg=mapping['package']
 for n,p in mapping['members'].items():
  out=safe_child(dest/pkg,n);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/p,out)
  if out.suffix=='.sh':out.chmod(out.stat().st_mode|0o111)
 result=doctor(number,dest/pkg)
 if not result['ready_for_package_preflight']:raise ValueError(result)
 result.update(package=str(dest/pkg),line='TRUST-CPF方案');write(dest/'materialization.json',result);return result
def restore_archive(number,destination):
 s=stage(number);dest=safe_new(destination);a=read(s/'results/archives/archive.json');dest.mkdir(parents=True);out=dest/a['name'];whole=hashlib.sha256();size=0
 with out.open('wb') as target:
  for part in a['parts']:
   h=hashlib.sha256();n=0
   with (s/part['path']).open('rb') as src:
    for b in iter(lambda:src.read(8*1024**2),b''):target.write(b);h.update(b);whole.update(b);n+=len(b)
   if n!=part['bytes'] or h.hexdigest()!=part['sha256']:raise ValueError('Archive part differs: '+part['path'])
   size+=n
 if whole.hexdigest()!=a['sha256'] or size!=a['bytes']:raise ValueError('Whole archive differs')
 out.with_name(out.name+'.sha256').write_bytes((a['sha256']+'  '+out.name+'\n').encode('ascii'))
 return dict(stage=number,archive=str(out),bytes=size,sha256=whole.hexdigest(),verified=True)
def check(number=None):
 output=[]
 for n in ([number] if number else SPECS):
  s=stage(n);sid,pkg,title,nstages,nmetrics=SPECS[n];mapping=read(s/'package_members.json')['members']
  for f in read(s/'file_manifest.json'):
   p=s/f['path'];assert p.is_file() and p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],str(p)
  raw=read(s/'results/raw_export_members.json');assert len(raw)=={'05':648,'06':1197}[n]
  for f in raw:assert sha(s/f['path'])==f['sha256'],f['path']
  a=read(s/'results/archives/archive.json');whole=hashlib.sha256();size=0
  for part in a['parts']:
   assert sha(s/part['path'])==part['sha256']
   with (s/part['path']).open('rb') as src:
    for b in iter(lambda:src.read(8*1024**2),b''):whole.update(b);size+=len(b)
  assert size==a['bytes'] and whole.hexdigest()==a['sha256']
  for name,h in read(s/mapping['MANIFEST.json']).items():assert sha(s/mapping[name])==h,name
  with zipfile.ZipFile(s/'results/deployment'/(pkg+'.zip')) as z:
   assert set(x for x in z.namelist() if not x.endswith('/'))=={pkg+'/'+x for x in mapping}
   for name,p in mapping.items():assert z.read(pkg+'/'+name)==(s/p).read_bytes(),name
  inv=read(s/'configs/experiment_inventory.json');assert len(inv)==nmetrics
  for item in inv:
   folder=s/item['path'];metric=read(folder/'metrics.json');assert sha(folder/'metrics.json')==item['metrics_sha256']
   assert metric['epoch']==item['selected_epoch']
   for snapshot in ['epoch0','best']:
    for bank in ['1701','2701','3701']:
     for phase in ['val','test']:assert (folder/snapshot/('bank'+bank)/(phase+'_predictions.npz')).is_file()
   if 'reused_from' in item:
    for f in item['reuse_verified_files']:assert sha(folder/f)==sha(ROOT/item['reused_from']/f)
  status=read(s/f'results/raw_export/{pkg}/work/status.json');assert status['complete'] and status['completed_stages']==nstages and not status['missing']
  for p in s.rglob('*.md'):
   t=p.read_text(encoding='utf8');assert t.count('```')%2==0,p
   assert not any(ord(c)<32 and c not in '\n\r\t' for c in t),p
   prose=re.sub(r'```[^\n]*\n[\s\S]*?```','',t)
   for link in re.findall(r'\]\((<[^>]+>|[^)\n]+)\)',prose):
    link=unquote(link.strip('<>')).split('#')[0]
    if not link or re.match(r'^[a-z]+://',link):continue
    assert not re.match(r'^[A-Za-z]:',link) and (p.parent/link).exists(),(str(p),link)
  for p in (s/'code').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
  output.append(dict(stage=n,passed=True,server_members=len(raw),release_members=len(mapping),metrics=nmetrics,reused_uniform=sum(x['result_kind']=='reused_uniform' for x in inv),status=status))
 return dict(passed=True,stages=output,scientific_bytes_preserved=True,training_started=False)
def finalize():
 stages=[x for x in read(ROOT/'provenance/stages.json') if x['id'] not in ['37','38']]
 layout=read(ROOT/'provenance/layout.json');layout['files']=[x for x in layout['files'] if x.get('kind')!='trustcpf_followup_server_member']
 idx=ROOT/'provenance/metrics_index.csv'
 with idx.open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;metrics=[x for x in reader if not x['route'].startswith('TRUST-followup-')]
 receipts=[]
 for n,(sid,pkg,title,nstages,nmetrics) in SPECS.items():
  s=stage(n);status=read(s/f'results/raw_export/{pkg}/work/status.json');inv=read(s/'configs/experiment_inventory.json')
  inputs=dict(line='TRUST-CPF方案',stage=n,archive_stage=sid,title=title,primary_package=pkg,seeds=[3351,3352,3353],completed_results=nmetrics,reused_controls=sum(x['result_kind']=='reused_uniform' for x in inv),formal_training_results_included=True,package_member_map='package_members.json',experiment_inventory='configs/experiment_inventory.json',status=status)
  for k,part in [('code_files','code'),('configs','configs'),('data_files','results'),('knowledge',f'configs/{pkg}/assets')]:inputs[k]=[p.relative_to(s).as_posix() for p in files(s/part)]
  inputs['documents']=[p.relative_to(s).as_posix() for p in s.rglob('*.md')];write(s/'inputs.json',inputs)
  commands=[]
  for name in ['README.md','SERVER_COMMANDS.md','analysis/REPRODUCE.md']:
   for language,cmd in re.findall(r'```(bash|powershell)\n(.*?)```',(s/name).read_text(encoding='utf8'),re.S):commands.append(dict(source=name,language=language,text=cmd))
  write(s/'runbook.json',dict(line='TRUST-CPF方案',stage=n,primary_package=pkg,commands=commands))
  stages.append(dict(id=sid,line='TRUST-CPF方案',number=n,version='v1',title=n+' · '+title,path='TRUST-CPF方案/'+n,primary_package=pkg,active_package=pkg,status='Completed; full source, raw results and reproducible analyses archived',completed_results=nmetrics,reused_controls=inputs['reused_controls']))
  arc=read(s/'results/archives/archive.json')
  for f in read(s/'results/raw_export_members.json'):layout['files'].append(dict(original_path=arc['name']+'::'+f['runtime_path'],runtime_path=f['runtime_path'],current_path='TRUST-CPF方案/'+n+'/'+f['path'],sha256=f['sha256'],bytes=f['bytes'],stage=sid,kind='trustcpf_followup_server_member'))
  invix={(x['seed'],x['variant']):x for x in inv}
  with (s/f'results/raw_export/{pkg}/work/summary.csv').open(encoding='utf-8-sig',newline='') as f:
   for item in csv.DictReader(f):
    run=invix[(int(item['seed']),item['variant'])];row={k:'' for k in fields}
    row.update(stage=sid,route='TRUST-followup-'+n,dataset='ut-zap50k',seed=item['seed'],variant=item['variant'],phase=item['phase'],setting=item['world'],selected_epoch=item['epoch'],auc_pct=item['AUC_native'],hm_pct=item['HM'],source='TRUST-CPF方案/'+n+'/'+run['path']+'/metrics.json',note='Native AUC; '+run['result_kind']+'; snapshot='+item['snapshot']+'; graph_bank='+item['bank'])
    metrics.append(row)
  receipts.append(dict(stage=n,archive=arc,status=status,scientific_bytes_preserved=True))
  write(s/'file_manifest.json',[dict(path=p.relative_to(s).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files(s) if p.name!='file_manifest.json'])
 write(ROOT/'provenance/stages.json',stages);write(ROOT/'provenance/layout.json',layout)
 with idx.open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(metrics)
 summary=read(ROOT/'provenance/structure_summary.json');summary['line_numbering']['TRUST-CPF方案']='01 / v1、02–06';summary['documentation_updated']='2026-10-24';write(ROOT/'provenance/structure_summary.json',summary)
 receipt='provenance/updates/20261024_trustcpf_align_competition.json';write(ROOT/receipt,dict(date='2026-10-24',stages=receipts,completed_result_records=24,reused_uniform_records=13,new_adaptations=11,training_started=False,planned_only='R5 relative utility fitting; superseded 3-epoch Competition release has no formal server result.'))
 for p in (ROOT/'provenance/documents').glob('trustcpf_0*.json'):
  doc=read(p);doc['published_sha256']=sha(ROOT/doc['document']);write(p,doc)
 current={x['path']:x for x in read(ROOT/'provenance/current_manifest.json')}
 changed=set(subprocess.check_output(['git','diff','--name-only','-z','HEAD'],cwd=ROOT).decode('utf8').split('\0'))
 changed.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode('utf8').split('\0'))
 changed.update(p.relative_to(ROOT).as_posix() for n in SPECS for p in files(stage(n)))
 for rel in changed:
  p=ROOT/rel
  if rel and rel!='provenance/current_manifest.json' and p.is_file():current[rel]=dict(path=rel,sha256=sha(p),bytes=p.stat().st_size)
 write(ROOT/'provenance/current_manifest.json',[current[k] for k in sorted(current)])
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=list(SPECS));ap.add_argument('--check',action='store_true');ap.add_argument('--finalize',action='store_true');ap.add_argument('--materialize');ap.add_argument('--restore-archive');a=ap.parse_args()
 if a.finalize:finalize();result=check(a.stage)
 elif a.materialize or a.restore_archive:
  if not a.stage:ap.error('--stage is required')
  result=materialize(a.stage,a.materialize) if a.materialize else restore_archive(a.stage,a.restore_archive)
 elif a.check:result=check(a.stage)
 else:ap.error('Choose --check, --finalize, --materialize or --restore-archive')
 print(json.dumps(result,ensure_ascii=False))
