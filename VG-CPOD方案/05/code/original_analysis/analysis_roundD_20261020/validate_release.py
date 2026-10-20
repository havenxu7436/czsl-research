"""Check the complete D workflow on a small CPU cache and repeat it."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'vgcpod_v1_roundD'
OLD = ROOT/'vgcpod_v1_roundC'
sys.path.insert(0,str(PACKAGE))
import torch
from tests.test_math import synthetic_cache
from vgcpod import engine
from vgcpod.common import read_json

torch.set_num_threads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    runtime = list((PACKAGE/'vgcpod').glob('*.py'))+[PACKAGE/'run.py']
    for path in runtime:
        assert path.read_bytes()==(OLD/path.relative_to(PACKAGE)).read_bytes(),path
    with tempfile.TemporaryDirectory(prefix='vgcpod_d_cli_') as folder:
        temp = Path(folder)
        d,c = temp/'D_source',temp/'C_source'
        ignore = shutil.ignore_patterns('__pycache__','*.pyc','validation')
        shutil.copytree(PACKAGE,d,ignore=ignore)
        shutil.copytree(OLD,c,ignore=ignore)
        small = synthetic_cache(temp/'cache')
        for source in (d,c):
            for sub in ('round_c','round_d'):
                for path in (source/'configs'/sub).glob('*.json'):
                    cfg = read_json(path)
                    cfg.update(rank=small['rank'],adapter_rank=small['adapter_rank'],workers=0,
                        batch_size=8,eval_batch_size=8,epochs=2)
                    path.write_text(json.dumps(cfg)+'\n',encoding='utf-8')
        for case in ('A5','A6'):
            cfg = read_json(PACKAGE/'configs/round_a'/(case+'.json'))
            cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=2)
            for seed in (3351,3352,3353):
                engine.train(argparse.Namespace(cache=str(temp/'cache'),out=str(temp/'A_out'/case),
                    seed=seed,device='cpu',resume=False),cfg)
        baseline = [sys.executable,'-u',str(c/'round_c.py'),'run','--cache',str(temp/'cache'),
            '--out',str(temp/'C_out'),'--a-out',str(temp/'A_out'),'--b-out',str(temp/'B_out'),
            '--skip-b3','--gpus','1','2','--cases','C1','C2','--seeds','3351','--device','cpu']
        result = subprocess.run(baseline,cwd=c,capture_output=True,text=True,encoding='utf-8',timeout=120)
        assert result.returncode==0,result.stdout+result.stderr
        logs = ['C1/C2 baseline\n'+result.stdout]
        original = {p.relative_to(temp/'C_out').as_posix():sha(p) for p in (temp/'C_out').rglob('*')
                    if p.is_file() and 'seed3351' in p.parts}
        command = [sys.executable,'-u',str(d/'round_d.py'),'run','--cache',str(temp/'cache'),
            '--out',str(temp/'D_out'),'--c-out',str(temp/'C_out'),'--c-root',str(c),
            '--a-out',str(temp/'A_out'),'--b-out',str(temp/'B_out'),'--gpus','1','2','3','--device','cpu']
        for invocation in ('first','repeat'):
            result = subprocess.run(command,cwd=d,capture_output=True,text=True,encoding='utf-8',timeout=180)
            logs.append(invocation+'\n'+result.stdout+result.stderr)
            assert result.returncode==0,'\n'.join(logs)
            events = read_json(temp/'D_out/round_d_status.json')
            self_status = 'complete' if invocation=='first' else 'already_complete'
            assert len(events)==3 and all(e['status']==self_status for e in events),events
            assert {(e['case'],e['seed'],str(e['gpu'])) for e in events}=={
                ('C2',3352,'1'),('C2',3353,'2'),('D1',3351,'3')}
            assert len(read_json(temp/'D_out/mean_gate_status.json'))==4
            assert 'ROUND_D_COMPLETE' in result.stdout
        for name,fingerprint in original.items():
            assert sha(temp/'C_out'/name)==fingerprint,name
        summary = read_json(temp/'D_out/round_d_summary.json')
        assert len(summary['runs'])==5
        assert len(summary['paired_deltas'])==13,summary['paired_deltas']
        c2 = next(r for r in summary['by_case'] if r['case']=='C2')
        assert c2['n_seeds']==3 and c2['seeds']==[3351,3352,3353]
        d1 = next(r for r in summary['by_case'] if r['case']=='D1')
        assert d1['n_seeds']==1 and d1['val_cw_AUC_native_sd'] is None
        assert len(summary['mean_gate_effects'])==16
        assert summary['per_pair_rows']>0
        for case,seeds,source in [('C2',[3351,3352,3353],c),('D1',[3351],d)]:
            for seed in seeds:
                record = read_json(temp/'D_out/mean_gate'/case/('seed%d'%seed)/'metrics.json')
                assert record['source_root']==str(source.resolve())
                assert record['train_images']==16 and record['optimizer_steps']==0
                assert record['mean_shape']==[2,3] and record['mean_mode_average_max_error']<1e-6
                for condition in ('dynamic','train_mean'):
                    metrics = record['metrics'][condition]
                    for world in ('cw','ow'):
                        assert metrics['test_'+world]['fixed_val_bias']==metrics['val_'+world]['bias_native']
                assert all(abs(v)<1e-6 for v in record['dynamic_minus_saved_auc'].values())
                log = (temp/'D_out/logs'/('mean_gate_%s_seed%d.log'%(case,seed))).read_text(encoding='utf-8')
                assert 'MEAN_GATE_ALREADY_COMPLETE' in log
        (PACKAGE/'validation/cli_output.txt').write_text('\n'.join(logs),encoding='utf-8')
    sys.path.insert(0,str(ROOT/'tmp/round_a_bash_validation'))
    import bashlex
    scripts = []
    for path in sorted((PACKAGE/'scripts').glob('*.sh')):
        bashlex.parse(path.read_text(encoding='utf-8'))
        assert b'\r\n' not in path.read_bytes()
        scripts.append(path.name)
    python_files = list((PACKAGE/'vgcpod').glob('*.py'))+list(PACKAGE.glob('*.py'))
    for path in python_files:
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,8))
    report = dict(fixture='synthetic cache, CPU',python=sys.version.split()[0],torch=torch.__version__,
        training_jobs=3,repeat_training_skipped=3,mean_gate_evaluations=4,repeat_mean_gate_skipped=4,
        original_c_source_used=True,original_c_runs_unchanged=True,unchanged_runtime_files=len(runtime),
        combined_metric_runs=5,paired_comparisons=13,mean_gate_metric_rows=16,
        python38_syntax_files=len(python_files),bash_scripts=scripts)
    (PACKAGE/'validation/result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
