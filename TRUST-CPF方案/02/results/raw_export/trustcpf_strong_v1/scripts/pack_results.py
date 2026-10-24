import hashlib,os,tarfile
from datetime import datetime,timedelta,timezone
from pathlib import Path
root=Path(__file__).resolve().parents[1]
work=Path(os.environ['CPF_WORK']).resolve()
stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
out=root.parent/('trustcpf_strong_v1_results_'+stamp+'.tar.gz')
with tarfile.open(out,'w:gz',encoding='utf8') as tar:
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or work==p or work in p.parents:continue
        if p.name.endswith(('.lock','.writing')):continue
        if any(ord(c)>127 for c in p.relative_to(root).as_posix()):continue
        tar.add(p,arcname='trustcpf_strong_v1/'+p.relative_to(root).as_posix(),recursive=False)
    for p in sorted(work.rglob('*')):
        if p.is_file() and not p.name.endswith(('.lock','.writing')):
            # Dereference locally hardlinked measurements so Windows extraction is self-contained.
            tar.dereference=True
            tar.add(p,arcname='trustcpf_strong_v1/work/'+p.relative_to(work).as_posix(),recursive=False)
h=hashlib.sha256()
with out.open('rb') as f:
    for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
Path(str(out)+'.sha256').write_text(h.hexdigest()+'  '+out.name+'\n',encoding='ascii')
(root/'latest_archive.txt').write_text(str(out)+'\n',encoding='utf8')
print(out)
