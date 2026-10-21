"""Exercise the H1 launcher twice, historical comparisons and legacy G1 equivalence."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'vgcpod_v1_roundH'
sys.path.insert(0,str(PACKAGE))
import numpy as np
import torch
from tests.test_math import synthetic_cache
from vgcpod import engine
from vgcpod.common import load_torch,read_json,sha

torch.set_num_threads(1)


def run(command,cwd):
    result=subprocess.run(command,cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout+result.stderr


def same_state(left,right):
    assert left['epoch']==right['epoch'] and left['step']==right['step']
    assert set(left['model'])==set(right['model'])
    for key,value in left['model'].items():assert torch.equal(value,right['model'][key]),key
    assert torch.equal(left['rng']['torch'],right['rng']['torch'])
    assert left['optimizer']['param_groups']==right['optimizer']['param_groups']
    for parameter,state in left['optimizer']['state'].items():
        for key,value in state.items():
            other=right['optimizer']['state'][parameter][key]
            assert torch.equal(value,other) if torch.is_tensor(value) else value==other


def main():
    logs=[]
    with tempfile.TemporaryDirectory(prefix='vgcpod_h_cli_') as folder:
        temp=Path(folder);stage=temp/'H_source';legacy=temp/'G_source'
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','validation')
        shutil.copytree(PACKAGE,stage,ignore=ignore)
        shutil.copytree(ROOT/'vgcpod_v1_roundG',legacy,ignore=ignore)
        synthetic_cache(temp/'cache')
        path=stage/'configs/round_h/H1.json';cfg=read_json(path)
        cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3)
        path.write_text(json.dumps(cfg)+'\n',encoding='utf-8')
        for case,stage_name in [('G0','g'),('G1','g'),('E1','e'),('F1','f')]:
            config=read_json(PACKAGE/('configs/round_'+stage_name)/(case+'.json'))
            config.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3)
            for seed in (3351,3352,3353):
                engine.train(argparse.Namespace(cache=str(temp/'cache'),out=str(temp/(stage_name.upper()+'_out')/case),
                    seed=seed,device='cpu',resume=False),config)
            if case=='G1':
                old_config=temp/'g1_config.json';old_config.write_text(json.dumps(config),encoding='utf-8')
                command=[sys.executable,'-u',str(legacy/'run.py'),'train','--cache',str(temp/'cache'),
                    '--out',str(temp/'legacy'),'--config',str(old_config),'--seed','3351','--device','cpu']
                logs.append('Legacy G1\n'+run(command,legacy))
                for checkpoint in ('best.pt','best_exact.pt','last.pt'):
                    sub=Path('ut-zap50k/V1/seed3351/checkpoints')/checkpoint
                    same_state(load_torch(temp/'legacy'/sub),load_torch(temp/'G_out/G1'/sub))
        reference_files=[p for name in ('G_out','E_out','F_out') for p in (temp/name).rglob('*') if p.is_file()]
        reference_hashes={p:sha(p) for p in reference_files}
        command=[sys.executable,'-u',str(stage/'round_h.py'),'run','--cache',str(temp/'cache'),
            '--out',str(temp/'H_out'),'--gpus','1','2','3','--device','cpu']
        for name in ('a','c','d','e','f','g'):command+=['--'+name+'-out',str(temp/(name.upper()+'_out'))]
        checkpoint_hashes=None;mean_sha=None
        for invocation in ('first','repeat'):
            output=run(command,stage);logs.append(invocation+'\n'+output)
            status=read_json(temp/'H_out/round_h_status.json')
            assert len(status)==3 and all(e['status']=='complete' for e in status)
            assert {(e['seed'],e['gpu']) for e in status}=={(3351,'1'),(3352,'2'),(3353,'3')}
            expected='complete' if invocation=='first' else 'already_complete'
            assert all(e['training']['status']==expected for e in status)
            assert all(len(e['mean_gate'])==2 and all(m['exit_code']==0 for m in e['mean_gate']) for e in status)
            assert 'ROUND_H_COMPLETE' in output and 'training_jobs 3 mean_gate_jobs 6' in output
            hashes={p.relative_to(temp/'H_out').as_posix():sha(p) for p in (temp/'H_out').rglob('*.pt')}
            if checkpoint_hashes is not None:assert hashes==checkpoint_hashes
            checkpoint_hashes=hashes
            current_mean=sha(temp/'H_out/feature_statistics/train_object_query_mean.npz')
            if mean_sha is not None:assert mean_sha==current_mean
            mean_sha=current_mean
        assert reference_hashes=={p:sha(p) for p in reference_files}
        summary=read_json(temp/'H_out/round_h_summary.json')
        assert len(summary['runs'])==3 and len(summary['exact_selection_runs'])==3
        assert len(summary['mean_gate_effects'])==24 and len(summary['matched_native_runs'])==9
        assert len(summary['support_training_runs'])==30
        for comparison in ('H1-G0','H1-G1','H1-E1','H1-F1'):
            assert len([r for r in summary['paired_deltas'] if r['comparison']==comparison])==3
        assert len(summary['exact_reference_deltas'])==6 and summary['per_pair_rows']>0
        diag=read_json(temp/'H_out/ranking_diagnostics.json')
        assert summary['ranking_diagnostic_rows']==len(diag['runs'])>0
        assert {r['selection'] for r in diag['runs'] if r['case']=='G1'}=={'native','exact'}
        assert {r['condition'] for r in diag['runs'] if r['case']=='H1'}=={'dynamic','train_mean'}
        for comparison in ('H1-G0','H1-G1','H1-H1_mean'):
            assert len([r for r in diag['paired_deltas'] if r['comparison']==comparison])==36
        assert all(r['extra_support_pct']==0 and r['extra_dominates_valid']==0 for r in diag['runs'])
        expected=np.load(temp/'cache/train/fo.npy').astype(np.float64).mean(0).astype(np.float32)
        config=read_json(temp/'H_out/configs/H1.json')
        assert config['gate_statistics_sha256']==mean_sha
        for seed in (3351,3352,3353):
            run_path=temp/'H_out/H1/ut-zap50k/V1'/('seed%d'%seed)
            assert sha(run_path/'checkpoints/best.pt')==sha(run_path/'checkpoints/best_native.pt')
            history=read_json(run_path/'history.json')
            for selection,criterion in [('native','AUC_native'),('exact','AUC_exact')]:
                epoch=max(range(len(history)),key=lambda i:history[i]['val_cw'][criterion])+1
                checkpoint=load_torch(run_path/'checkpoints'/('best_'+selection+'.pt'))
                assert checkpoint['epoch']==epoch
                np.testing.assert_array_equal(checkpoint['model']['gate_fo_mean'].numpy(),expected)
                assert checkpoint['config']['gate_statistics_sha256']==mean_sha
                folder='mean_gate' if selection=='native' else 'mean_gate_exact'
                means=read_json(temp/'H_out'/folder/'H1'/('seed%d'%seed)/'metrics.json')
                assert means['selected_epoch']==epoch
                assert max(abs(v) for v in means['dynamic_minus_saved_auc'].values())==0
                assert means['train_images']==16
                for condition in ('dynamic','train_mean'):
                    for world in ('cw','ow'):
                        metrics=means['metrics'][condition]
                        assert metrics['test_'+world]['fixed_val_bias']==metrics['val_'+world]['bias_native']
                log=temp/'H_out/logs'/('mean_gate_%s_H1_seed%d.log'%(selection,seed))
                assert 'MEAN_GATE_ALREADY_COMPLETE' in log.read_text(encoding='utf-8')
        (PACKAGE/'validation/cli_output.txt').write_text('\n'.join(logs),encoding='utf-8')
    sys.path.insert(0,str(ROOT/'tmp/round_a_bash_validation'))
    import bashlex
    scripts=list((PACKAGE/'scripts').glob('*.sh'))
    for path in scripts:
        bashlex.parse(path.read_text(encoding='utf-8'));assert b'\r\n' not in path.read_bytes()
    sources=list((PACKAGE/'vgcpod').glob('*.py'))+list(PACKAGE.glob('*.py'))+list((PACKAGE/'tests').glob('*.py'))
    for path in sources:ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,8))
    report=dict(fixture='CPU synthetic CLIP cache',training_jobs=3,mean_gate_jobs=6,
        repeat_training_skipped=3,repeat_mean_gate_skipped=6,shared_train_mean_identical_across_seeds=True,
        legacy_G1_parameters_adamw_and_torch_rng_bitwise_equal=True,
        native_exact_selection_verified=True,independent_val_biases=True,reference_files_unchanged=True,
        ranking_diagnostics_verified=True,exact_reference_deltas=6,
        python38_syntax_files=len(sources),bash_scripts=[p.name for p in scripts])
    (PACKAGE/'validation/result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
