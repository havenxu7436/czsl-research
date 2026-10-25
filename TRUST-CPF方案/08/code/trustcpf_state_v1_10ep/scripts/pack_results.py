"""Atomic fixed-name archive. Download commands need no remote path parsing."""
import hashlib,json,os,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
work=Path(os.environ['CPF_WORK']).resolve()
status=json.loads((work/'status.json').read_text(encoding='utf-8-sig'))
if not status['complete']:
    raise SystemExit('Experiment not complete. Run bash scripts/status.sh to see missing stages.')
out=root.parent/'trustcpf_state_v1_10ep_results.tar.gz'
partial=Path(str(out)+'.part')
entries=[]
for p in sorted(root.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or work in p.parents:continue
    relative=p.relative_to(root)
    if relative.parts[0] in ('work','logs'):continue
    if p.name in ('download.json','server_env.sh','setup.json') or p.name.endswith(('.writing','.lock','.pid')):continue
    if not relative.as_posix().isascii():continue
    entries.append((p,'trustcpf_state_v1_10ep/'+relative.as_posix()))
for folder,name in ((work,'work'),(root/'logs','logs')):
    if folder.exists():
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not p.name.endswith(('.writing','.lock','.pid')):
                entries.append((p,'trustcpf_state_v1_10ep/'+name+'/'+p.relative_to(folder).as_posix()))
print('PACKING',len(entries),'files ->',out,flush=True)
with tarfile.open(str(partial),'w:gz',compresslevel=1,encoding='utf8',dereference=True) as archive:
    for i,(file,name) in enumerate(entries,1):
        archive.add(str(file),arcname=name,recursive=False)
        if i%50==0 or i==len(entries):print('PACK_PROGRESS',i,'/',len(entries),flush=True)
h=hashlib.sha256()
print('HASHING_ARCHIVE',flush=True)
with partial.open('rb') as f:
    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
os.replace(str(partial),str(out))
Path(str(out)+'.sha256').write_bytes((h.hexdigest()+'  '+out.name+'\n').encode('ascii'))
(root/'download.json').write_bytes((json.dumps(dict(archive=str(out),name=out.name,
    sha256=h.hexdigest(),bytes=out.stat().st_size),indent=2)+'\n').encode('utf8'))
print('PACK_COMPLETE',out,flush=True)
print('SHA256',h.hexdigest(),flush=True)

