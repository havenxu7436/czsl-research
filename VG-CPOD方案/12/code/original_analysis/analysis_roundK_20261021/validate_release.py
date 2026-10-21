"""Exercise six K trainings, twelve interventions, repeat reuse and legacy J1 equivalence."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'vgcpod_v1_roundK'
sys.path.insert(0,str(PACKAGE))
import numpy as np
import torch
from tests.test_math import synthetic_cache
from vgcpod import engine
from vgcpod.common import load_torch,read_json,sha
from vgcpod.gate_features import prepare_gate_statistics

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
    with tempfile.TemporaryDirectory(prefix='vgcpod_k_cli_') as folder:
        temp=Path(folder);stage=temp/'K_source';legacy=temp/'J_source'
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','validation')
        shutil.copytree(PACKAGE,stage,ignore=ignore);shutil.copytree(ROOT/'vgcpod_v1_roundJ',legacy,ignore=ignore)
        synthetic_cache(temp/'cache')
        statistics=prepare_gate_statistics(temp/'cache',temp/'J_out/feature_statistics')
        for case in ('K1','K2'):
            path=stage/'configs/round_k'/(case+'.json');cfg=read_json(path)
            cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3)
            path.write_text(json.dumps(cfg)+'\n',encoding='utf-8')
        for case,stage_name in [('G0','g'),('G1','g'),('H1','h'),('I1','i'),('I2','i'),('J1','j')]:
            config=read_json(PACKAGE/('configs/round_'+stage_name)/(case+'.json'))
            config.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3)
            if case in ('H1','I1','I2','J1'):config.update(gate_statistics_path=str(statistics),gate_statistics_sha256=sha(statistics))
            for seed in (3351,3352,3353):
                engine.train(argparse.Namespace(cache=str(temp/'cache'),out=str(temp/(stage_name.upper()+'_out')/case),
                    seed=seed,device='cpu',resume=False),config)
            if case=='J1':
                path=temp/'j1_config.json';path.write_text(json.dumps(config),encoding='utf-8')
                command=[sys.executable,'-u',str(legacy/'run.py'),'train','--cache',str(temp/'cache'),
                    '--out',str(temp/'legacy'),'--config',str(path),'--seed','3351','--device','cpu']
                logs.append('Legacy J1\n'+run(command,legacy))
                for checkpoint in ('best.pt','best_exact.pt','last.pt'):
                    sub=Path('ut-zap50k/V1/seed3351/checkpoints')/checkpoint
                    same_state(load_torch(temp/'legacy'/sub),load_torch(temp/'J_out/J1'/sub))
        reference_files=[p for name in ('G_out','H_out','I_out','J_out') for p in (temp/name).rglob('*') if p.is_file()]
        reference_hashes={p:sha(p) for p in reference_files}
        command=[sys.executable,'-u',str(stage/'round_k.py'),'run','--cache',str(temp/'cache'),
            '--out',str(temp/'K_out'),'--gpus','1','2','3','--device','cpu']
        for name in ('a','c','d','e','f','g','h','i','j'):command+=['--'+name+'-out',str(temp/(name.upper()+'_out'))]
        hashes_before=None
        for invocation in ('first','repeat'):
            output=run(command,stage);logs.append(invocation+'\n'+output)
            status=read_json(temp/'K_out/round_k_status.json')
            assert len(status)==3 and all(e['status']=='complete' for e in status)
            assert {(e['seed'],e['gpu']) for e in status}=={(3351,'1'),(3352,'2'),(3353,'3')}
            expected='complete' if invocation=='first' else 'already_complete'
            assert all([t['case'] for t in e['trainings']]==['K1','K2'] for e in status)
            assert all(all(t['status']==expected for t in e['trainings']) for e in status)
            assert all(len(e['mean_gate'])==4 and all(m['exit_code']==0 for m in e['mean_gate']) for e in status)
            assert 'ROUND_K_COMPLETE' in output and 'training_jobs 6 mean_gate_jobs 12' in output
            hashes={p.relative_to(temp/'K_out').as_posix():sha(p) for p in (temp/'K_out').rglob('*.pt')}
            if hashes_before is not None:assert hashes==hashes_before
            hashes_before=hashes
            assert sha(temp/'K_out/feature_statistics/train_object_query_mean.npz')==sha(statistics)
        assert 'TRAIN_GATE_MEAN_COPIED_FROM_J' in logs[1]
        assert reference_hashes=={p:sha(p) for p in reference_files}
        summary=read_json(temp/'K_out/round_k_summary.json')
        assert len(summary['runs'])==6 and len(summary['exact_selection_runs'])==6
        assert len(summary['mean_gate_effects'])==48 and len(summary['matched_native_runs'])==24
        assert len(summary['support_training_runs'])==48
        assert len(summary['readout_training_runs'])==18
        assert len(summary['readout_optimizer_runs'])==54
        for r in summary['readout_optimizer_runs']:
            if r['case'] in ('K1','K2') and r['parameter']=='Wd':
                assert r['optimizer_group']=='dynamic_cp'
                assert abs(r['lr_scale']-(.09 if r['case']=='K1' else .03))<1e-12
        assert all(r['cp_dynamic_rms']>0 and r['cp_static_rms']>0 for r in summary['readout_training_runs'])
        for comparison in ('K1-J1','K2-J1','K1-K2','K1-I2','K2-I2','K1-G0','K2-G0'):
            assert len([r for r in summary['paired_deltas'] if r['comparison']==comparison])==3
        assert len(summary['exact_reference_deltas'])==39 and summary['per_pair_rows']>0
        diag=read_json(temp/'K_out/ranking_diagnostics.json')
        assert summary['ranking_diagnostic_rows']==len(diag['runs'])>0
        for comparison in ('K1-J1','K2-J1','K1-K2','K1-I2','K2-I2','K1-K1_mean','K2-K2_mean'):
            assert len([r for r in diag['paired_deltas'] if r['comparison']==comparison])==36,comparison
        expected_mean=np.load(temp/'cache/train/fo.npy').astype(np.float64).mean(0).astype(np.float32)
        configs=[read_json(temp/'K_out/configs'/(c+'.json')) for c in ('K1','K2')]
        assert configs[0]['gate_statistics_sha256']==sha(statistics)
        for case in ('K1','K2'):
            for seed in (3351,3352,3353):
                run_path=temp/'K_out'/case/'ut-zap50k/V1'/('seed%d'%seed)
                assert sha(run_path/'checkpoints/best.pt')==sha(run_path/'checkpoints/best_native.pt')
                history=read_json(run_path/'history.json')
                for selection,criterion in [('native','AUC_native'),('exact','AUC_exact')]:
                    epoch=max(range(len(history)),key=lambda i:history[i]['val_cw'][criterion])+1
                    checkpoint=load_torch(run_path/'checkpoints'/('best_'+selection+'.pt'))
                    assert checkpoint['epoch']==epoch
                    np.testing.assert_array_equal(checkpoint['model']['gate_fo_mean'].numpy(),expected_mean)
                    folder='mean_gate' if selection=='native' else 'mean_gate_exact'
                    means=read_json(temp/'K_out'/folder/case/('seed%d'%seed)/'metrics.json')
                    assert means['selected_epoch']==epoch and means['train_images']==16
                    assert max(abs(v) for v in means['dynamic_minus_saved_auc'].values())==0
                    for condition in ('dynamic','train_mean'):
                        for world in ('cw','ow'):
                            metrics=means['metrics'][condition]
                            assert metrics['test_'+world]['fixed_val_bias']==metrics['val_'+world]['bias_native']
                    log=temp/'K_out/logs'/('mean_gate_%s_%s_seed%d.log'%(selection,case,seed))
                    assert 'MEAN_GATE_ALREADY_COMPLETE' in log.read_text(encoding='utf-8')
        (PACKAGE/'validation/cli_output.txt').write_text('\n'.join(logs),encoding='utf-8')
    sys.path.insert(0,str(ROOT/'tmp/round_a_bash_validation'))
    import bashlex
    scripts=list((PACKAGE/'scripts').glob('*.sh'))
    for path in scripts:
        bashlex.parse(path.read_text(encoding='utf-8'));assert b'\r\n' not in path.read_bytes()
    sources=list((PACKAGE/'vgcpod').glob('*.py'))+list(PACKAGE.glob('*.py'))+list((PACKAGE/'tests').glob('*.py'))
    for path in sources:ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,8))
    report=dict(fixture='CPU synthetic CLIP cache',training_jobs=6,mean_gate_jobs=12,
        repeat_training_skipped=6,repeat_mean_gate_skipped=12,shared_mean_copied_byte_for_byte_from_J=True,
        legacy_J1_parameters_adamw_and_torch_rng_bitwise_equal=True,native_exact_selection_verified=True,
        independent_val_biases=True,reference_files_unchanged=True,ranking_diagnostics_verified=True,
        exact_reference_deltas=39,readout_rms_summary_rows=18,optimizer_diagnostic_rows=54,python38_syntax_files=len(sources),bash_scripts=[p.name for p in scripts])
    (PACKAGE/'validation/result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
