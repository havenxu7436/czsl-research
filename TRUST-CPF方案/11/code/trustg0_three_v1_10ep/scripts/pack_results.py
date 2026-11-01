import argparse
import hashlib
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trustg0.common import read,sha,write
from trustg0 import workflow


def pack(root,archive):
    root=Path(root);archive=Path(archive);report=read(root/'work/report.json')
    if not report['complete']:raise RuntimeError('Complete all pipelines before final export')
    files=[]
    for p in root.rglob('*'):
        if not p.is_file():continue
        r=p.relative_to(root)
        if any(v in r.parts for v in ('__pycache__','.git','vendor')) or p.suffix in ('.pyc','.lock','.tmp'):continue
        if 'cache' in r.parts and p.name not in ('complete.json','progress.json','prompts.json'):continue
        if p.name in ('last.pt','partial.pt','result_manifest.json') or 'validation' in r.parts:continue
        files.append(p)
    manifest={p.relative_to(root).as_posix():sha(p) for p in files};write(root/'result_manifest.json',manifest);files.append(root/'result_manifest.json')
    temp=archive.with_name(archive.name+'.tmp');print('PACKING',len(files),'files',round(sum(p.stat().st_size for p in files)/2**30,2),'GiB',flush=True)
    with tarfile.open(temp,'w:gz',compresslevel=1) as tar:
        for i,p in enumerate(files):
            tar.add(p,arcname='trustg0_three_v1_10ep/'+p.relative_to(root).as_posix(),recursive=False)
            if (i+1)%200==0:print('PACK',i+1,'/',len(files),flush=True)
    temp.replace(archive);digest=sha(archive);archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    print('ARCHIVE_READY',archive,'SHA256',digest,flush=True);return archive


if __name__=='__main__':
    args=argparse.Namespace(work=str(ROOT/'work'));workflow.report(args)
    pack(ROOT,ROOT.parent/'trustg0_three_v1_10ep_results.tar.gz')
