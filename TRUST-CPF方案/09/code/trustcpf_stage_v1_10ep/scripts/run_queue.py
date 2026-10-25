"""One command: three teacher workers, global fit barrier, three evaluation workers."""
import os,subprocess,sys
from datetime import datetime
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from trustcpf.runtime import lock
from trustcpf.common import read_json,write_json
gpus=sys.argv[1:] or ['0','1','2']
if len(gpus)!=3 or len(set(gpus))!=3:raise SystemExit('Provide three distinct GPUs, e.g. 0 1 2')
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
    result={};result['teacher_codes']=workers('train-seed')
    if any(result['teacher_codes']):
        subprocess.run([sys.executable,'run.py','report'],cwd=str(root));write_json(logs/'queue_result.json',result)
        raise SystemExit('TEACHER_QUEUE_FAILED: inspect logs; rerun the same queue to resume')
    print('ALL_TEACHERS_READY: fitting one shared predictor',flush=True)
    result['fit_code']=subprocess.run([sys.executable,'-u','run.py','fit'],cwd=str(root)).returncode
    if result['fit_code']:
        write_json(logs/'queue_result.json',result);raise SystemExit('GLOBAL_FIT_FAILED')
    result['evaluation_codes']=workers('evaluate-seed')
    result['report_code']=subprocess.run([sys.executable,'run.py','report'],cwd=str(root)).returncode
    status=read_json(work/'status.json') if (work/'status.json').exists() else {'complete':False}
    result['complete']=not any(result['evaluation_codes']) and result['report_code']==0 and status['complete']
    write_json(logs/'queue_result.json',dict(result,gpus=gpus))
    print('QUEUE_COMPLETE' if result['complete'] else 'QUEUE_FAILED: inspect seed logs; rerun to resume',flush=True)
    raise SystemExit(0 if result['complete'] else 1)
