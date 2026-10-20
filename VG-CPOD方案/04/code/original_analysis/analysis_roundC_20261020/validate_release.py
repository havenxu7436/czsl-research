"""CPU verification of centered gates, original B-source delegation, and CLI restart."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'vgcpod_v1_roundC'
OLD = ROOT / 'vgcpod_v1_roundB'
sys.path.insert(0,str(PACKAGE))
import torch
from tests.test_math import fixture,synthetic_cache
from vgcpod import engine
from vgcpod.common import read_json

torch.set_num_threads(1)


def main():
    spec = importlib.util.spec_from_file_location('legacy_b_model',OLD/'vgcpod/model.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    compatibility = []
    for name in list(legacy.VARIANTS)+['B1','B2','B3']:
        current,cfg,x,y,mask = fixture('V1' if name.startswith('B') else name)
        if name.startswith('B'):
            cfg.update(read_json(OLD/'configs/round_b'/(name+'.json')))
            cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=2)
            current = type(current)(current.ea,current.eo,current.ec,cfg,3351)
        old = legacy.VGCPOD(current.ea,current.eo,current.ec,cfg,3351)
        optimizers = [engine.make_optimizer(m,cfg) for m in (old,current)]
        for step in range(4):
            results = []
            for model,opt in zip((old,current),optimizers):
                engine.set_learning_rates(opt,cfg,step,4)
                opt.zero_grad(set_to_none=True)
                out = model(*x)
                loss = model.objective(out,*y,mask)['total']
                loss.backward()
                engine.clip_total_norm([p for p in model.parameters() if p.requires_grad],cfg['gradient_clip'])
                opt.step()
                results.append((out['joint'].detach(),loss.detach()))
            assert all(torch.equal(a,b) for a,b in zip(*results)),name
            assert all(torch.equal(v,current.state_dict()[k]) for k,v in old.state_dict().items()),name
        compatibility.append(dict(case=name,steps=4,bitwise_equal=True))
    (PACKAGE/'validation/legacy_compatibility.json').write_text(json.dumps(compatibility,indent=2)+'\n',encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='vgcpod_c_cli_') as temp:
        temp = Path(temp)
        stage,old_stage = temp/'C_source',temp/'B_source'
        ignore = shutil.ignore_patterns('__pycache__','*.pyc','validation')
        shutil.copytree(PACKAGE,stage,ignore=ignore)
        shutil.copytree(OLD,old_stage,ignore=ignore)
        small = synthetic_cache(temp/'cache')
        for path in list((stage/'configs/round_c').glob('*.json'))+[old_stage/'configs/round_b/B3.json']:
            cfg = read_json(path)
            cfg.update(rank=small['rank'],adapter_rank=small['adapter_rank'],workers=0,
                       batch_size=8,eval_batch_size=8,epochs=2)
            path.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
        for name in ['A5','A6']:
            cfg = read_json(PACKAGE/'configs/round_a'/(name+'.json'))
            cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=2)
            for seed in [3351,3352]:
                args = argparse.Namespace(cache=str(temp/'cache'),out=str(temp/'A_out'/name),
                                          seed=seed,device='cpu',resume=False)
                engine.train(args,cfg)
        old_command = [sys.executable,'-u',str(old_stage/'round_b.py'),'run','--cache',str(temp/'cache'),
            '--out',str(temp/'B_out'),'--reference-out',str(temp/'A_out'),
            '--gpus','3','--cases','B3','--seeds','3351','--device','cpu']
        old_result = subprocess.run(old_command,cwd=old_stage,env=dict(os.environ),stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,encoding='utf-8',errors='replace',timeout=120)
        assert old_result.returncode==0,old_result.stdout
        command = [sys.executable,'-u',str(stage/'round_c.py'),'run','--cache',str(temp/'cache'),
            '--out',str(temp/'C_out'),'--a-out',str(temp/'A_out'),'--b-out',str(temp/'B_out'),
            '--b-root',str(old_stage),'--gpus','1','2','3','--cases','C1','C2','--seeds','3351',
            '--b3-seed','3352','--device','cpu']
        logs = ['B3 seed3351 reference\n'+old_result.stdout]
        for invocation in ['first','repeat']:
            result = subprocess.run(command,cwd=stage,env=dict(os.environ),stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,encoding='utf-8',errors='replace',timeout=120)
            logs.append(invocation+'\n'+result.stdout)
            assert result.returncode==0,result.stdout
            events = read_json(temp/'C_out/round_c_status.json')
            expected = 'complete' if invocation=='first' else 'already_complete'
            assert len(events)==3 and all(r['status']==expected for r in events),events
            assert 'ROUND_C_COMPLETE' in result.stdout and 'jobs 3' in result.stdout
        summary = read_json(temp/'C_out/round_c_summary.json')
        assert len(summary['runs'])==4 and len(summary['paired_deltas'])==11
        assert summary['per_pair_rows']>0
        for seed in [3351,3352]:
            protocol = read_json(temp/f'B_out/B3/ut-zap50k/V1/seed{seed}/protocol.json')
            expected = hashlib.sha256((OLD/'vgcpod/model.py').read_bytes()).hexdigest()
            assert protocol['source_sha256']['vgcpod/model.py']==expected
            assert protocol['source_sha256']['vgcpod/model.py']!=hashlib.sha256((PACKAGE/'vgcpod/model.py').read_bytes()).hexdigest()
        (PACKAGE/'validation/cli_output.txt').write_text('\n'.join(logs),encoding='utf-8')
    sys.path.insert(0,str(ROOT/'tmp/round_a_bash_validation'))
    import bashlex
    scripts = []
    for path in sorted((PACKAGE/'scripts').glob('*.sh')):
        bashlex.parse(path.read_text(encoding='utf-8'))
        assert b'\r\n' not in path.read_bytes()
        scripts.append(path.name)
    runtime = list((PACKAGE/'vgcpod').glob('*.py'))+[PACKAGE/'run.py',PACKAGE/'round_c.py',PACKAGE/'round_b.py']
    for path in runtime:
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,8))
    report = dict(local_validation=True,fixture='synthetic cache, CPU',python=sys.version.split()[0],
        torch=torch.__version__,legacy_bitwise_cases=len(compatibility),optimizer_steps_per_case=4,
        concurrent_jobs=3,repeat_jobs_skipped=3,b3_original_source_verified=True,
        combined_metric_runs=4,paired_comparisons=11,python38_syntax_files=len(runtime),bash_scripts=scripts)
    (PACKAGE/'validation/result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
