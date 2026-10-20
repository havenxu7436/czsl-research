"""Exercise three F seeds, both selections, six gate evaluations and legacy equivalence."""
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
PACKAGE = ROOT/'vgcpod_v1_roundF'
sys.path.insert(0,str(PACKAGE))
import torch
from tests.test_math import synthetic_cache
from vgcpod import engine
from vgcpod.common import load_torch,read_json,sha

torch.set_num_threads(1)


def main():
    assert (PACKAGE/'vgcpod/model.py').read_bytes()==(ROOT/'vgcpod_v1_roundE/vgcpod/model.py').read_bytes()
    with tempfile.TemporaryDirectory(prefix='vgcpod_f_cli_') as folder:
        temp = Path(folder)
        stage,legacy = temp/'F_source',temp/'E_source'
        ignore = shutil.ignore_patterns('__pycache__','*.pyc','validation')
        shutil.copytree(PACKAGE,stage,ignore=ignore)
        shutil.copytree(ROOT/'vgcpod_v1_roundE',legacy,ignore=ignore)
        synthetic_cache(temp/'cache')
        cfg = read_json(stage/'configs/round_f/F1.json')
        cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3)
        (stage/'configs/round_f/F1.json').write_text(json.dumps(cfg)+'\n',encoding='utf-8')
        old_config = legacy/'test_f1.json'
        old_config.write_text(json.dumps(cfg)+'\n',encoding='utf-8')
        for case,sub,output in [('A5','round_a','A_out'),('C2','round_c','C_out'),('D1','round_d','D_out'),('E1','round_e','E_out')]:
            ref = read_json(PACKAGE/'configs'/sub/(case+'.json'))
            ref.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3)
            for seed in ((3351,3352,3353) if case=='E1' else (3351,)):
                engine.train(argparse.Namespace(cache=str(temp/'cache'),out=str(temp/output/case),
                    seed=seed,device='cpu',resume=False),ref)
        old_command = [sys.executable,'-u',str(legacy/'run.py'),'train','--cache',str(temp/'cache'),
            '--out',str(temp/'legacy'),'--config',str(old_config),'--seed','3351','--device','cpu']
        old = subprocess.run(old_command,cwd=legacy,capture_output=True,text=True,encoding='utf-8',timeout=120)
        assert old.returncode==0,old.stdout+old.stderr
        command = [sys.executable,'-u',str(stage/'round_f.py'),'run','--cache',str(temp/'cache'),
            '--out',str(temp/'F_out'),'--a-out',str(temp/'A_out'),'--c-out',str(temp/'C_out'),
            '--d-out',str(temp/'D_out'),'--e-out',str(temp/'E_out'),'--gpus','1','2','3','--device','cpu']
        logs = ['Legacy E trainer at beta=.05\n'+old.stdout]
        for invocation in ('first','repeat'):
            result = subprocess.run(command,cwd=stage,capture_output=True,text=True,encoding='utf-8',
                errors='replace',timeout=180)
            logs.append(invocation+'\n'+result.stdout+result.stderr)
            assert result.returncode==0,'\n'.join(logs)
            status = read_json(temp/'F_out/round_f_status.json')
            assert len(status)==3 and all(r['status']=='complete' for r in status),status
            assert {(r['seed'],r['gpu']) for r in status}=={(3351,'1'),(3352,'2'),(3353,'3')}
            expected = 'complete' if invocation=='first' else 'already_complete'
            assert all(r['training']['status']==expected for r in status)
            assert all(len(r['mean_gate'])==2 and all(e['exit_code']==0 for e in r['mean_gate']) for r in status)
            assert 'ROUND_F_COMPLETE' in result.stdout
        summary = read_json(temp/'F_out/round_f_summary.json')
        assert len(summary['runs'])==3 and summary['by_case'][0]['seeds']==[3351,3352,3353]
        assert len(summary['exact_selection_runs'])==3 and len(summary['mean_gate_effects'])==24
        assert len(summary['beta_native_runs'])==6
        assert {r['case'] for r in summary['beta_native_runs']}=={'E1','F1'}
        assert len(summary['support_training_runs'])==12
        assert len([r for r in summary['paired_deltas'] if r['comparison']=='F1-E1'])==3
        assert len(summary['paired_deltas'])==6
        assert summary['per_pair_rows']>0
        new_run = temp/'F_out/F1/ut-zap50k/V1/seed3351'
        old_run = temp/'legacy/ut-zap50k/V1/seed3351'
        for name in ('best.pt','last.pt'):
            a,b = load_torch(old_run/'checkpoints'/name),load_torch(new_run/'checkpoints'/name)
            assert a['epoch']==b['epoch'] and a['step']==b['step']
            for key,value in a['model'].items():
                assert torch.equal(value,b['model'][key]),(name,key)
            assert torch.equal(a['rng']['torch'],b['rng']['torch'])
            assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
            for parameter,state in a['optimizer']['state'].items():
                for key,value in state.items():
                    other = b['optimizer']['state'][parameter][key]
                    assert torch.equal(value,other) if torch.is_tensor(value) else value==other
        for row in summary['support_training_runs']:
            expected = 1.5*row['beta']/(1+row['beta'])
            assert abs(row['support_weight']-expected)<1e-14
            assert abs(row['pair_weighted']-1.5*row['pair'])<1e-12
            assert abs(row['support_weighted']-expected*row['support'])<1e-12
            assert row['task_epoch_reconstruction_error']<1e-5
        for seed in (3351,3352,3353):
            run = temp/'F_out/F1/ut-zap50k/V1'/('seed%d'%seed)
            assert sha(run/'checkpoints/best.pt')==sha(run/'checkpoints/best_native.pt')
            history = read_json(run/'history.json')
            for selection,criterion,folder in [('native','AUC_native','mean_gate'),('exact','AUC_exact','mean_gate_exact')]:
                checkpoint = load_torch(run/'checkpoints'/('best_'+selection+'.pt'))
                epoch = max(range(len(history)),key=lambda i:history[i]['val_cw'][criterion])+1
                assert checkpoint['epoch']==epoch
                means = read_json(temp/'F_out'/folder/'F1'/('seed%d'%seed)/'metrics.json')
                assert means['selected_epoch']==epoch and means['selection']==selection
                assert means['identity']['checkpoint_sha256']==sha(run/'checkpoints'/('best_'+selection+'.pt'))
                assert max(abs(v) for v in means['dynamic_minus_saved_auc'].values())==0
                for condition in ('dynamic','train_mean'):
                    for world in ('cw','ow'):
                        metrics = means['metrics'][condition]
                        assert metrics['test_'+world]['fixed_val_bias']==metrics['val_'+world]['bias_native']
                log = (temp/'F_out/logs'/('mean_gate_%s_F1_seed%d.log'%(selection,seed))).read_text(encoding='utf-8')
                assert 'MEAN_GATE_ALREADY_COMPLETE' in log
        (PACKAGE/'validation/cli_output.txt').write_text('\n'.join(logs),encoding='utf-8')
    sys.path.insert(0,str(ROOT/'tmp/round_a_bash_validation'))
    import bashlex
    scripts = list((PACKAGE/'scripts').glob('*.sh'))
    for p in scripts:
        bashlex.parse(p.read_text(encoding='utf-8'))
        assert b'\r\n' not in p.read_bytes()
    sources = list((PACKAGE/'vgcpod').glob('*.py'))+list(PACKAGE.glob('*.py'))
    for p in sources:
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p),feature_version=(3,8))
    report = dict(fixture='CPU synthetic CLIP cache',training_jobs=3,mean_gate_jobs=6,
        repeat_training_skipped=3,repeat_mean_gate_skipped=6,selections=['native','exact'],
        model_py_unchanged=True,legacy_beta005_bitwise_parameters=True,legacy_rng_equal=True,legacy_optimizer_equal=True,support_logs_valid=True,
        beta_comparison_cases=['E1','F1'],independent_val_biases=True,
        python38_syntax_files=len(sources),bash_scripts=[p.name for p in scripts])
    (PACKAGE/'validation/result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
