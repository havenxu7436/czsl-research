"""Publish a completed archive, then an atomic JSON download descriptor."""
import hashlib,json,os,tarfile
from datetime import datetime,timedelta,timezone
from pathlib import Path
root=Path(__file__).resolve().parents[1];work=Path(os.environ['CPF_WORK']).resolve()
stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
out=root.parent/('trustcpf_align_v1_results_'+stamp+'.tar.gz')
partial=Path(str(out)+'.part')
entries=[]
for p in sorted(root.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or work in p.parents:continue
    if p.parts[len(root.parts)] in ('work','logs'):continue
    if p.name in ('download.json','latest_archive.txt') or p.name.endswith(('.lock','.writing')):continue
    entries.append((p,'trustcpf_align_v1/'+p.relative_to(root).as_posix()))
for folder,name in [(work,'work'),(root/'logs','logs')]:
    if folder.exists():
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not p.name.endswith(('.lock','.writing','.pid')):
                entries.append((p,'trustcpf_align_v1/'+name+'/'+p.relative_to(folder).as_posix()))
print('PACKING',len(entries),'files; includes Q5 teacher, predictors, policy scans, models, and raw audit blocks.',flush=True)
with tarfile.open(partial,'w:gz',encoding='utf8') as tar:
    tar.dereference=True
    for i,(p,name) in enumerate(entries,1):
        tar.add(p,arcname=name,recursive=False)
        if i%25==0 or i==len(entries):print('PACK_PROGRESS',i,'/',len(entries),flush=True)
print('HASHING_ARCHIVE',flush=True)
h=hashlib.sha256()
with partial.open('rb') as f:
    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
os.replace(str(partial),str(out))
Path(str(out)+'.sha256').write_bytes((h.hexdigest()+'  '+out.name+'\n').encode('ascii'))
descriptor=dict(schema=1,project='trustcpf_align_v1',archive=str(out),name=out.name,bytes=out.stat().st_size,sha256=h.hexdigest())
tmp=root/'download.json.writing'
tmp.write_text(json.dumps(descriptor,indent=2)+'\n',encoding='utf8')
os.replace(str(tmp),str(root/'download.json'))
print('PACK_COMPLETE',out,flush=True)
print('DOWNLOAD_READY',root/'download.json',flush=True)
