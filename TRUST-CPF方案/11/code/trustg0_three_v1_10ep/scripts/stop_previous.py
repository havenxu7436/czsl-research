#!/usr/bin/env python3
"""Stop only the requested CPF repair or previous Pure project; preserve all result files."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

def snapshot():
    out={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split()
            if v[0]=='Z':continue
            out[int(p.name)]=dict(pid=int(p.name),ppid=int(v[1]),start=v[19],
                argv=(p/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0'),cwd=Path(os.readlink(p/'cwd')))
        except (OSError,ValueError,IndexError):pass
    return out

def flag(argv,key,default):
    return argv[argv.index(key)+1] if key in argv and argv.index(key)+1<len(argv) else default

def target(r,project):
    for a in r['argv'][1:]:
        if a.endswith(('.py','.sh')):
            path=Path(a);path=(r['cwd']/path).resolve() if not path.is_absolute() else path.resolve()
            if path in [project/'run.py',project/'scripts/run_queue.py',project/'scripts/run_all.sh']:return True
            if path.name=='TRUST_CPF_four_gpu_20261027.py' and 'run' in r['argv']:
                return Path(flag(r['argv'],'--project',str(project))).resolve()==project
    return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',default='/var/tmp/xhw/cpf_repair_v1_10ep');p.add_argument('--apply',action='store_true');a=p.parse_args()
    project=Path(a.project).resolve()
    if project.name not in ('cpf_repair_v1_10ep','trustcpf_pure_v1_10ep') or not (project/'work/identity.json').is_file():raise RuntimeError('Expected initialized CPF repair or previous Pure project')
    records=snapshot();roots={pid for pid,r in records.items() if target(r,project)}
    print('PROJECT',project,flush=True)
    if not roots:print('ALREADY_STOPPED',flush=True);return
    for pid in sorted(roots):print('TARGET',pid,' '.join(records[pid]['argv']),flush=True)
    if not a.apply:print('PLAN_ONLY: use --apply to stop these project processes');return
    # Freeze dispatchers first so they cannot replenish the queue during stop.
    queues={pid for pid in roots if any(Path(x).name in ('run_queue.py','TRUST_CPF_four_gpu_20261027.py','run_all.sh') for x in records[pid]['argv'])}
    for pid in queues:
        try:os.kill(pid,signal.SIGSTOP)
        except ProcessLookupError:pass
    time.sleep(.2);records=snapshot();roots={pid for pid,r in records.items() if target(r,project)}
    selected=set(roots)
    while True:
        more={pid for pid,r in records.items() if r['ppid'] in selected}
        if more<=selected:break
        selected|=more
    for pid in sorted(selected):
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    for pid in queues:
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        now=snapshot();remaining={pid for pid in selected if pid in now and now[pid]['start']==records[pid]['start']}
        if not remaining:break
        time.sleep(.25)
    for pid in remaining:
        print('STOP_UNRESPONSIVE',pid,flush=True)
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    time.sleep(.3);now=snapshot()
    remaining={pid for pid in selected if pid in now and now[pid]['start']==records[pid]['start']}
    if remaining:raise RuntimeError('Still exiting: '+str(sorted(remaining)))
    print('CPF_PROJECT_STOPPED; all checkpoints and results preserved',flush=True)

if __name__=='__main__':main()
