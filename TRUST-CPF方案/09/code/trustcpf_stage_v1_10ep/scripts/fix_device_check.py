"""Repair the teacher CPU/CUDA comparison and migrate only its recorded digest."""
import argparse,contextlib,hashlib,json,os
from pathlib import Path

KEY='trustcpf/matched_train.py'
OLD_SHA='d8297152a836229f1249e7b3fc9c21fac8c400019da27b5f012219f2e4ded128'
NEW_SHA='57c66f5b06f90f196a92e289a7b3cdf19d600afb82d9ca21430b006a78a8fc47'
OLD="torch.equal(v,st['model'][k]) and torch.equal(v,st['best_model'][k])"
NEW="torch.equal(v,st['model'][k].detach().cpu()) and torch.equal(v,st['best_model'][k].detach().cpu())"

def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def atomic(path,data):
    tmp=path.with_name(path.name+'.devicefix.tmp');tmp.write_bytes(data);os.replace(str(tmp),str(path))
def json_bytes(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf8')
def backup(path):
    target=path.with_name(path.name+'.before_device_fix')
    if not target.exists():target.write_bytes(path.read_bytes())

@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if os.name=='posix':
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        else:
            import msvcrt
            f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        yield

def repair(project,work):
    project=Path(project).resolve();work=Path(work).resolve();target=project/KEY
    with contextlib.ExitStack() as locks:
        locks.enter_context(locked(project/'logs/queue/.lock'))
        locks.enter_context(locked(work/'progress/global/.lock'))
        for seed in (3351,3352,3353):locks.enter_context(locked(work/('progress/seed%d/.lock'%seed)))
        old=target.read_bytes();current=sha(old)
        if current not in (OLD_SHA,NEW_SHA):raise ValueError('Unexpected matched_train.py version; no changes made')
        new=old.replace(OLD.encode(),NEW.encode()) if current==OLD_SHA else old
        if sha(new)!=NEW_SHA:raise ValueError('Patch digest mismatch')
        ip=work/'progress/global/identity.json';identity=read(ip) if ip.exists() else None
        if identity is not None:
            actual={p.relative_to(project).as_posix():sha(p.read_bytes()) for p in
                    list((project/'trustcpf').glob('*.py'))+[project/'run.py']}
            recorded=identity['implementation']
            if set(actual)!=set(recorded):raise ValueError('Other source files changed; identity not migrated')
            for name,digest in actual.items():
                valid=recorded[name] in (OLD_SHA,NEW_SHA) if name==KEY else recorded[name]==digest
                if not valid:raise ValueError('Unrelated source change: '+name)
        mp=project/'MANIFEST.json';manifest=read(mp) if mp.exists() else None
        if manifest is not None and manifest.get(KEY) not in (OLD_SHA,NEW_SHA):
            raise ValueError('Unexpected manifest version')
        if current==OLD_SHA:backup(target);atomic(target,new)
        if manifest is not None and manifest[KEY]!=NEW_SHA:
            backup(mp);manifest[KEY]=NEW_SHA;atomic(mp,json_bytes(manifest))
        if identity is not None and identity['implementation'][KEY]!=NEW_SHA:
            backup(ip);identity['implementation'][KEY]=NEW_SHA;atomic(ip,json_bytes(identity))
        record=dict(fix='CPU/CUDA-safe frozen backbone comparison',file=KEY,old_sha256=OLD_SHA,new_sha256=NEW_SHA,
                    checkpoint_files_modified=False,identity_migrated=identity is not None)
        atomic(work/'progress/global/device_fix.json',json_bytes(record))
    print('DEVICE_FIX_COMPLETE',flush=True)
    print('Checkpoints preserved. Resume: bash scripts/run_all.sh 1 2 3',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',default='.')
    p.add_argument('--work',default=os.getenv('CPF_WORK'))
    a=p.parse_args();repair(a.project,a.work or str(Path(a.project)/'work'))
