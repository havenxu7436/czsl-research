"""One process per physical GPU, dependency-aware resume, GPUs 0/1/2 only."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trustg0.common import read,write,lock


def jobs(work,identity):
    result=[];cfg=identity['config']
    for ds in identity['datasets']:
        dp=work/ds;prepare='prepare_'+ds
        result.append(dict(id=prepare,command=['prepare','--dataset',ds],done=dp/'groups.json',deps=[]))
        for seed in cfg['seeds']:
            label=ds+'_'+str(seed);w=dp/('seed%d'%seed);base='base_'+label
            result.append(dict(id=base,command=['train-base','--dataset',ds,'--seed',str(seed)],done=w/'base_complete.json',deps=[dp/'groups.json']))
            teachers=[]
            for t in range(1,cfg['teacher_initializations']+1):
                for q in range(1,6):
                    done=w/'teachers'/('t%d_q%d'%(t,q))/'complete.json';teachers.append(done)
                    result.append(dict(id='teacher_'+label+'_t%d_q%d'%(t,q),command=['teacher','--dataset',ds,'--seed',str(seed),'--teacher',str(t),'--fold',str(q)],done=done,deps=[dp/'groups.json']))
            result.append(dict(id='fit_'+label,command=['fit','--dataset',ds,'--seed',str(seed)],done=w/'fit_complete.json',deps=teachers))
            result.append(dict(id='finish_'+label,command=['finish','--dataset',ds,'--seed',str(seed)],done=w/'complete.json',deps=[w/'base_complete.json',w/'fit_complete.json']))
    # Prepare all datasets, then formal bases, then round-robin teachers across datasets/seeds.
    priority={'prepare':0,'train-base':1,'teacher':2,'fit':3,'finish':4}
    return sorted(result,key=lambda j:(priority[j['command'][0]], j['command'][-1] if j['command'][0]=='teacher' else ''))


def externally_busy(gpus):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    uuids={line.split(',')[1].strip():int(line.split(',')[0]) for line in output.splitlines()}
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return {uuids[line.split(',')[0].strip()] for line in output.splitlines() if line.split(',')[0].strip() in uuids}&set(gpus)


def disk_preflight(work,identity):
    remaining=0
    for ds in identity['datasets']:
        folder=work/ds/'cache';index=read(work/ds/'index.json');total=len(index['rows'])*257*768*2+len(index['attrs'])*len(index['objs'])*768*4+64*2**20
        allocated=sum(p.stat().st_blocks*512 if hasattr(p.stat(),'st_blocks') else p.stat().st_size for p in folder.glob('*') if p.is_file())
        remaining+=max(0,total-allocated)
    free=shutil.disk_usage(work).free;reserve=identity['config']['cache_reserve_gb']*2**30
    if free<remaining+reserve:raise RuntimeError('Need %.1f GiB more cache + %.1f GiB reserve; free %.1f GiB'%(remaining/2**30,reserve/2**30,free/2**30))


def main():
    p=argparse.ArgumentParser();p.add_argument('gpus',type=int,nargs='*',default=[0,1,2]);p.add_argument('--work',default=str(ROOT/'work'));a=p.parse_args()
    if not a.gpus or len(a.gpus)!=len(set(a.gpus)) or not set(a.gpus)<={0,1,2}:raise ValueError('Use only distinct GPUs from 0,1,2')
    work=Path(a.work).resolve();identity=read(work/'identity.json');disk_preflight(work,identity)
    logs=ROOT/'logs';logs.mkdir(exist_ok=True);alljobs=jobs(work,identity);active={};failed=[];stopping=[False]
    def stop(sig,frame):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with lock(work/'queue.lock'):
        write(logs/'queue_pid.json',dict(pid=os.getpid(),gpus=a.gpus))
        while True:
            for gpu,(proc,job,handle) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                handle.close();ok=code==0 and job['done'].exists()
                print('FINISH',job['id'],'gpu',gpu,'exit',code,'ok',ok,flush=True)
                if not ok:failed.append(job['id'])
                del active[gpu]
            current={j['id'] for p,j,h in active.values()};pending=[j for j in alljobs if not j['done'].exists() and j['id'] not in current]
            if stopping[0]:
                for proc,j,h in active.values():
                    if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
                for proc,j,h in active.values():
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    h.close()
                print('QUEUE_STOPPED; rerun the same command to resume',flush=True);return 2
            if not failed:
                busy=externally_busy(a.gpus)
                for gpu in a.gpus:
                    if gpu in active or gpu in busy:continue
                    job=next((j for j in pending if all(f.exists() for f in j['deps'])),None)
                    if job is None:break
                    handle=open(logs/(job['id']+'.log'),'a',buffering=1);handle.write('\nQUEUE '+datetime.datetime.now().isoformat()+' GPU '+str(gpu)+'\n')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'run.py'),*job['command'],'--work',str(work)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    active[gpu]=(proc,job,handle);pending.remove(job)
                    print('START',job['id'],'physical_gpu',gpu,'pid',proc.pid,flush=True)
            write(work/'queue_status.json',dict(total=len(alljobs),finished=sum(j['done'].exists() for j in alljobs),failed=failed,
                active=[dict(gpu=g,pid=p.pid,job=j['id'],log=str(logs/(j['id']+'.log'))) for g,(p,j,h) in active.items()],updated=datetime.datetime.now().isoformat()))
            if not active and (failed or not pending):break
            time.sleep(5)
        subprocess.run([sys.executable,str(ROOT/'run.py'),'report','--work',str(work)],check=True)
        if failed:print('QUEUE_FAILED',failed,'Inspect task logs; rerun to resume.',flush=True);return 1
        if not read(work/'report.json')['complete']:raise RuntimeError('Queue done but report incomplete')
        print('ALL_G0_TRUST_EXPERIMENTS_COMPLETE',flush=True);return 0


if __name__=='__main__':sys.exit(main())
