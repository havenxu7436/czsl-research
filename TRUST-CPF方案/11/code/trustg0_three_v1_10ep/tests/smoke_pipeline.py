"""End-to-end 10-epoch synthetic pipeline, one independent repetition."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_core import fixture
from trustg0 import workflow
from trustg0.common import read,write

p=argparse.ArgumentParser();p.add_argument('--work',required=True);a=p.parse_args()
work=Path(a.work).resolve()
if (work/'identity.json').exists():raise RuntimeError('Use a new synthetic work directory')
fixture(work)
args=SimpleNamespace(work=str(work),dataset='toy',seed=5351,device='cpu')
workflow.train_base(args)
for q in range(1,6):
    args.teacher=1;args.fold=q;workflow.teacher(args)
workflow.fit(args);workflow.finish(args);result=workflow.report(args)
assert result['complete'] and result['metric_rows']==36,result
for f in work.rglob('complete.json'):
    d=read(f)
    if 'epochs' in d:assert d['epochs']==10
write(work/'smoke_pass.json',dict(complete=True,epochs=10,pipelines=1,teachers=5,metric_rows=36))
print('SYNTHETIC_FULL_PIPELINE_PASS')
