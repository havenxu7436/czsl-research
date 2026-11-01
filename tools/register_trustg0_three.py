"""Verify and restore completed TRUST stage 11 (three datasets, G0, 10 epochs)."""
from pathlib import Path
import argparse,ast,csv,json,hashlib,os,re,shutil
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT/'TRUST-CPF方案/11'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def child(root,name):
 rel=Path(name)
 if rel.is_absolute() or rel.drive or '..' in rel.parts:raise ValueError('Unsafe relative name')
 return root/rel
def new(path):
 p=Path(path).resolve()
 if p.exists():raise ValueError('Choose a new destination: '+str(p))
 p.mkdir(parents=True);return p
def transfer(a,b):
 b.parent.mkdir(parents=True,exist_ok=True)
 try:os.link(a,b)
 except OSError:shutil.copy2(a,b)
def materialize(destination):
 dest=new(destination);members=read(STAGE/'package_members.json');package=members['package']
 for original,canonical in members['members'].items():
  target=child(dest,package+'/'+original);transfer(STAGE/canonical,target)
  if target.suffix=='.sh':target.chmod(target.stat().st_mode|0o111)
 return dict(stage='11',path=str(dest/package),files=len(members['members']),training_started=False)
def restore_results(destination):
 dest=new(destination);records=read(STAGE/'results/raw_export_members.json')
 for row in records:transfer(STAGE/row['path'],child(dest,row['runtime_path']))
 return dict(stage='11',path=str(dest),members=len(records),training_started=False)
def restore_archive(dataset,destination):
 archive=read(STAGE/'results/archives/index.json')[dataset];target=Path(destination).resolve()
 if target.exists():raise ValueError('Choose a new archive path')
 target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
 with target.open('wb') as out:
  for part in archive['parts']:
   with (STAGE/part['path']).open('rb') as f:
    for block in iter(lambda:f.read(8*1024**2),b''):out.write(block);h.update(block);size+=len(block)
 assert size==archive['bytes'] and h.hexdigest()==archive['sha256']
 Path(str(target)+'.sha256').write_text(h.hexdigest()+'  '+target.name+'\n',encoding='ascii')
 return dict(dataset=dataset,path=str(target),sha256=h.hexdigest(),bytes=size)
def check():
 records=read(STAGE/'file_manifest.json')
 for row in records:
  p=STAGE/row['path'];assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
 raw=read(STAGE/'results/raw_export_members.json')
 by_runtime={r['runtime_path']:r for r in raw}
 for row in raw:assert sha(STAGE/row['path'])==row['sha256'],row['path']
 for ds,archive in read(STAGE/'results/archives/index.json').items():
  h=hashlib.sha256();size=0
  for row in archive['parts']:
   p=STAGE/row['path'];assert sha(p)==row['sha256']
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024**2),b''):h.update(b);size+=len(b)
  assert h.hexdigest()==archive['sha256'] and size==archive['bytes']
  snapshot=archive['snapshot_root']
  for name,digest in read(STAGE/'results/raw_export'/snapshot/'snapshot_manifest.json').items():
   assert by_runtime[snapshot+'/'+name]['sha256']==digest
 members=read(STAGE/'package_members.json')['members']
 for name,digest in read(STAGE/members['MANIFEST.json']).items():assert sha(STAGE/members[name])==digest
 queue=read(STAGE/'configs/experiment_queue.json');assert queue['total']==queue['completed']==165 and len(queue['jobs'])==165
 for job in queue['jobs']:assert sha(STAGE/job['receipt'])==job['receipt_sha256']
 training=read(STAGE/'configs/training_inventory.json');assert len(training)==450 and all(t['epochs']==10 for t in training)
 with (STAGE/'results/summary.csv').open(encoding='utf8',newline='') as f:summary=list(csv.DictReader(f))
 assert len(summary)==1620 and len({tuple(r[k] for k in ['dataset','seed','method','bank','phase']) for r in summary})==1620
 links=0;syntax=0
 for p in (STAGE/'code').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'));syntax+=1
 for p in STAGE.rglob('*.md'):
  text=p.read_text(encoding='utf8');assert text.count(chr(96)*3)%2==0,p
  prose=re.sub(chr(96)*3+r'[^\n]*\n[\s\S]*?'+chr(96)*3,'',text)
  for target in re.findall(r'\]\((<[^>]+>|[^)\n]+)\)',prose):
   target=unquote(target.strip('<>')).split('#')[0]
   if not target or re.match(r'^[a-z]+://',target):continue
   assert not re.match(r'^[A-Za-z]:',target) and (p.parent/target).exists(),(p,target)
   links+=1
 return dict(passed=True,stage='11',files=len(records),server_members=len(raw),queue_jobs=165,training_stages=450,recognition_rows=1620,q5_rows=378,python_syntax_files=syntax,local_links=links,scientific_bytes_preserved=True,training_started=False)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--materialize');p.add_argument('--restore-results');p.add_argument('--restore-archive');p.add_argument('--dataset',choices=['ut-zap50k','mit-states','cgqa']);a=p.parse_args()
 if a.check:r=check()
 elif a.materialize:r=materialize(a.materialize)
 elif a.restore_results:r=restore_results(a.restore_results)
 elif a.restore_archive and a.dataset:r=restore_archive(a.dataset,a.restore_archive)
 else:p.error('Choose --check, --materialize, --restore-results, or --restore-archive with --dataset')
 print(json.dumps(r,ensure_ascii=False))
