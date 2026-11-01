import datetime
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trustg0.common import read

work=ROOT/'work'
if not (work/'identity.json').exists():raise SystemExit('Run setup first.')
identity=read(work/'identity.json');q=read(work/'queue_status.json') if (work/'queue_status.json').exists() else {}
print('QUEUE',q.get('finished',0),'/',q.get('total','?'),'updated',q.get('updated','not started'),'failed',q.get('failed',[]))
for task in q.get('active',[]):print('RUNNING','GPU',task['gpu'],'PID',task['pid'],task['job'],'LOG',task['log'])
for ds in identity['datasets']:
    dp=work/ds;cache=dp/'cache/progress.json'
    print('\nDATASET',ds,'cache',read(cache) if cache.exists() else 'pending')
    for seed in identity['config']['seeds']:
        w=dp/('seed%d'%seed);tc=len(list((w/'teachers').glob('*/complete.json')));rc=len(list((w/'recognition').glob('*/*/metrics.json')))
        print('SEED',seed,'base',(w/'base_complete.json').exists(),'teachers',str(tc)+'/15','fit',(w/'fit_complete.json').exists(),'readouts',str(rc)+'/45','complete',(w/'complete.json').exists())
        updates=sorted(w.rglob('progress.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        if updates:print('  LATEST',updates[0].relative_to(w),read(updates[0]))
if (work/'report.json').exists():print('\nREPORT complete',read(work/'report.json')['complete'])
