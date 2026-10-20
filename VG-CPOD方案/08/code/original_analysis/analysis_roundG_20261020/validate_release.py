"""Exercise six G runs, six gate evaluations, repeat skipping and the original trainer."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'vgcpod_v1_roundG'
sys.path.insert(0,str(PACKAGE))
import torch
from tests.test_math import synthetic_cache
from vgcpod import engine
from vgcpod.common import load_torch,read_json,sha

torch.set_num_threads(1)


def run(command,cwd):
    result=subprocess.run(command,cwd=cwd,capture_output=True,text=True,encoding='utf-8',
        errors='replace',timeout=240)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout+result.stderr


def same_state(left,right):
    assert left['epoch']==right['epoch'] and left['step']==right['step']
    for key,value in left['model'].items():assert torch.equal(value,right['model'][key]),key
    assert torch.equal(left['rng']['torch'],right['rng']['torch'])
    assert left['optimizer']['param_groups']==right['optimizer']['param_groups']
    for parameter,state in left['optimizer']['state'].items():
        for key,value in state.items():
            other=right['optimizer']['state'][parameter][key]
            assert torch.equal(value,other) if torch.is_tensor(value) else value==other


def main():
    runtime=list((PACKAGE/'vgcpod').glob('*.py'))+[PACKAGE/'run.py',PACKAGE/'mean_gate_eval.py']
    for path in runtime:assert path.read_bytes()==(ROOT/'vgcpod_v1_roundF'/path.relative_to(PACKAGE)).read_bytes()
    logs=[]
    with tempfile.TemporaryDirectory(prefix='vgcpod_g_cli_') as folder:
        temp=Path(folder);stage=temp/'G_source';legacy=temp/'F_source'
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','validation')
        shutil.copytree(PACKAGE,stage,ignore=ignore)
        shutil.copytree(ROOT/'vgcpod_v1_roundF',legacy,ignore=ignore)
        synthetic_cache(temp/'cache')
        for case in ('G0','G1'):
            path=stage/'configs/round_g'/(case+'.json');cfg=read_json(path)
            cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3)
            path.write_text(json.dumps(cfg)+'\n',encoding='utf-8')
            command=[sys.executable,'-u',str(legacy/'run.py'),'train','--cache',str(temp/'cache'),
                '--out',str(temp/'legacy'/case),'--config',str(path),'--seed','3351','--device','cpu']
            logs.append('Legacy '+case+'\n'+run(command,legacy))
        for case,sub,output in [('A5','round_a','A_out'),('C2','round_c','C_out'),('D1','round_d','D_out'),
                                ('E1','round_e','E_out'),('F1','round_f','F_out')]:
            cfg=read_json(PACKAGE/'configs'/sub/(case+'.json'))
            cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3)
            for seed in ((3351,3352,3353) if case in ('E1','F1') else (3351,)):
                engine.train(argparse.Namespace(cache=str(temp/'cache'),out=str(temp/output/case),
                    seed=seed,device='cpu',resume=False),cfg)
        reference_files=[p for base in ('A_out','C_out','D_out','E_out','F_out') for p in (temp/base).rglob('*') if p.is_file()]
        reference_hashes={p:sha(p) for p in reference_files}
        command=[sys.executable,'-u',str(stage/'round_g.py'),'run','--cache',str(temp/'cache'),
            '--out',str(temp/'G_out'),'--gpus','1','2','3','--device','cpu']
        for name in ('a','c','d','e','f'):command+=['--'+name+'-out',str(temp/(name.upper()+'_out'))]
        checkpoint_hashes=None
        for invocation in ('first','repeat'):
            output=run(command,stage);logs.append(invocation+'\n'+output)
            status=read_json(temp/'G_out/round_g_status.json')
            assert len(status)==3 and all(e['status']=='complete' for e in status)
            assert {(e['seed'],e['gpu']) for e in status}=={(3351,'1'),(3352,'2'),(3353,'3')}
            expected='complete' if invocation=='first' else 'already_complete'
            assert all([r['case'] for r in e['trainings']]==['G0','G1'] for e in status)
            assert all(len(e['trainings'])==2 and all(t['status']==expected for t in e['trainings']) for e in status)
            assert all(len(e['mean_gate'])==2 and all(m['exit_code']==0 for m in e['mean_gate']) for e in status)
            assert 'ROUND_G_COMPLETE' in output and 'training_jobs 6 mean_gate_jobs 6' in output
            hashes={p.relative_to(temp/'G_out').as_posix():sha(p) for p in (temp/'G_out').rglob('*.pt')}
            if checkpoint_hashes is not None:assert hashes==checkpoint_hashes
            checkpoint_hashes=hashes
        assert reference_hashes=={p:sha(p) for p in reference_files}
        summary=read_json(temp/'G_out/round_g_summary.json')
        assert len(summary['runs'])==6 and len(summary['exact_selection_runs'])==6
        assert len(summary['mean_gate_effects'])==24 and len(summary['beta_native_runs'])==9
        assert len(summary['support_training_runs'])==24
        for comparison in ('G1-G0','G1-E1','G1-F1'):
            assert len([r for r in summary['paired_deltas'] if r['comparison']==comparison])==3
        assert summary['per_pair_rows']>0
        diag=read_json(temp/'G_out/ranking_diagnostics.json')
        assert summary['ranking_diagnostic_rows']==len(diag['runs'])>0
        assert {r['selection'] for r in diag['runs'] if r['case']=='G0'}=={'native','exact'}
        assert {r['condition'] for r in diag['runs'] if r['case']=='G1'}=={'dynamic','train_mean'}
        assert all(r['extra_support_pct']==0 and r['extra_dominates_valid']==0 for r in diag['runs'])
        assert len([r for r in diag['paired_deltas'] if r['comparison']=='G1-G0'])==36
        for case in ('G0','G1'):
            run_path=temp/'G_out'/case/'ut-zap50k/V1/seed3351'
            old=temp/'legacy'/case/'ut-zap50k/V1/seed3351'
            for checkpoint in ('best.pt','best_exact.pt','last.pt'):
                same_state(load_torch(old/'checkpoints'/checkpoint),load_torch(run_path/'checkpoints'/checkpoint))
            for seed in (3351,3352,3353):
                run_path=temp/'G_out'/case/'ut-zap50k/V1'/('seed%d'%seed)
                assert sha(run_path/'checkpoints/best.pt')==sha(run_path/'checkpoints/best_native.pt')
                history=read_json(run_path/'history.json')
                if case=='G0':assert all(h['train']['orth']==0 and h['train']['gamma_image_variance']==0 for h in history)
                for selection,criterion in [('native','AUC_native'),('exact','AUC_exact')]:
                    epoch=max(range(len(history)),key=lambda i:history[i]['val_cw'][criterion])+1
                    checkpoint=load_torch(run_path/'checkpoints'/('best_'+selection+'.pt'))
                    assert checkpoint['epoch']==epoch
                    if case=='G1':
                        folder='mean_gate' if selection=='native' else 'mean_gate_exact'
                        means=read_json(temp/'G_out'/folder/'G1'/('seed%d'%seed)/'metrics.json')
                        assert means['selected_epoch']==epoch
                        assert max(abs(v) for v in means['dynamic_minus_saved_auc'].values())==0
                        for condition in ('dynamic','train_mean'):
                            for world in ('cw','ow'):
                                metrics=means['metrics'][condition]
                                assert metrics['test_'+world]['fixed_val_bias']==metrics['val_'+world]['bias_native']
                        log=temp/'G_out/logs'/('mean_gate_%s_G1_seed%d.log'%(selection,seed))
                        assert 'MEAN_GATE_ALREADY_COMPLETE' in log.read_text(encoding='utf-8')
        (PACKAGE/'validation/cli_output.txt').write_text('\n'.join(logs),encoding='utf-8')
    sys.path.insert(0,str(ROOT/'tmp/round_a_bash_validation'))
    import bashlex
    scripts=list((PACKAGE/'scripts').glob('*.sh'))
    for path in scripts:
        bashlex.parse(path.read_text(encoding='utf-8'));assert b'\r\n' not in path.read_bytes()
    sources=list((PACKAGE/'vgcpod').glob('*.py'))+list(PACKAGE.glob('*.py'))
    for path in sources:ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,8))
    report=dict(fixture='CPU synthetic CLIP cache',training_jobs=6,mean_gate_jobs=6,
        repeat_training_skipped=6,repeat_mean_gate_skipped=6,runtime_identical_to_F=True,
        legacy_G0_G1_parameters_adamw_and_torch_rng_bitwise_equal=True,static_gamma_one_orth_zero=True,
        native_exact_selection_verified=True,independent_val_biases=True,reference_files_unchanged=True,
        ranking_diagnostics_verified=True,python38_syntax_files=len(sources),bash_scripts=[p.name for p in scripts])
    (PACKAGE/'validation/result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
