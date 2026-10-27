"""Three GPU workers with pooled-fit and Q4 selection barriers."""
import os,subprocess,sys
from datetime import datetime
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from trustcpf.runtime import lock
from trustcpf.common import read_json,write_json
gpus=sys.argv[1:] or ['1','2','3']
if len(gpus)!=3 or len(set(gpus))!=3:raise SystemExit('Provide three distinct GPU IDs: 1 2 3 or 0 1 2')
work=Path(os.environ['CPF_WORK']);logs=root/'logs';logs.mkdir(exist_ok=True)
if not (work/'progress/global/import.json').exists():raise SystemExit('Run python run.py setup first')
def workers(stage):
    processes=[];handles=[]
    try:
        for gpu,seed in zip(gpus,(3351,3352,3353)):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu;env['PYTHONUNBUFFERED']='1'
            h=(logs/('seed%d.log'%seed)).open('a',encoding='utf8');handles.append(h)
            h.write('\nQUEUE '+datetime.now().isoformat()+' '+stage+' physical_gpu='+gpu+'\n');h.flush()
            p=subprocess.Popen([sys.executable,'-u','run.py',stage,'--seed',str(seed)],cwd=str(root),env=env,
                stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT);processes.append(p)
            (logs/('seed%d.pid'%seed)).write_text(str(p.pid)+'\n',encoding='ascii')
            print('START',stage,'seed',seed,'physical_gpu',gpu,'pid',p.pid,flush=True)
        return [p.wait() for p in processes]
    except BaseException:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait()
        raise
    finally:
        for h in handles:h.close()
with lock(logs/'queue'):
    result=dict(gpus=gpus,complete=False)
    for stage,kind in [('prepare-seed','workers'),('fit','global'),('q4-seed','workers'),
                       ('select','global'),('evaluate-seed','workers')]:
        codes=workers(stage) if kind=='workers' else [subprocess.run([sys.executable,'-u','run.py',stage],cwd=str(root)).returncode]
        result[stage]=codes;write_json(logs/'queue_result.json',result)
        if any(codes):
            subprocess.run([sys.executable,'run.py','report'],cwd=str(root))
            raise SystemExit('QUEUE_FAILED at '+stage+': inspect seed logs; rerun same queue to resume')
    result['report_code']=subprocess.run([sys.executable,'run.py','report'],cwd=str(root)).returncode
    status=read_json(work/'status.json')
    result['complete']=result['report_code']==0 and status['complete']
    write_json(logs/'queue_result.json',result)
    print('ALL_ABC_EXPERIMENTS_COMPLETE' if result['complete'] else 'QUEUE_INCOMPLETE',flush=True)
    raise SystemExit(0 if result['complete'] else 1)
