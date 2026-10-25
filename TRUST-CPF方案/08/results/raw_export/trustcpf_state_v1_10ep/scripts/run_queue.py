"""Three independent GPU workers; global pooled fitting is completed by setup."""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from trustcpf.runtime import lock
from trustcpf.common import read_json,write_json
gpus=sys.argv[1:] or ['0','1','2']
if len(gpus)!=3 or len(set(gpus))!=3:
    raise SystemExit('Usage: bash scripts/run_all.sh 0 1 2 (three distinct GPUs)')
work=Path(os.environ['CPF_WORK'])
if not (work/'progress/global/fit.json').exists():
    raise SystemExit('Run python run.py setup first.')
logs=root/'logs';logs.mkdir(exist_ok=True)
with lock(logs/'queue'):
    processes=[];handles=[]
    try:
        for gpu,seed in zip(gpus,(3351,3352,3353)):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu;env['PYTHONUNBUFFERED']='1'
            logfile=logs/('seed%d.log'%seed);handle=logfile.open('a',encoding='utf8');handles.append(handle)
            handle.write('\nQUEUE_START '+datetime.now().isoformat()+' physical_gpu='+gpu+'\n');handle.flush()
            process=subprocess.Popen([sys.executable,'-u','run.py','run-seed','--seed',str(seed)],
                cwd=str(root),env=env,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            processes.append(process)
            (logs/('seed%d.pid'%seed)).write_text(str(process.pid)+'\n',encoding='ascii')
            print('START seed=%d physical_gpu=%s pid=%d'%(seed,gpu,process.pid),flush=True)
        codes=[p.wait() for p in processes]
    except KeyboardInterrupt:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait()
        raise
    finally:
        for h in handles:h.close()
    result=subprocess.run([sys.executable,'run.py','report'],cwd=str(root)).returncode
    status=read_json(work/'status.json') if (work/'status.json').exists() else {'complete':False}
    success=not any(codes) and result==0 and status['complete']
    write_json(logs/'queue_result.json',dict(gpus=gpus,exit_codes=codes,complete=success))
    print('QUEUE_COMPLETE' if success else 'QUEUE_FAILED: inspect seed logs; rerun the same command to resume.',flush=True)
    raise SystemExit(0 if success else 1)

