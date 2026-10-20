"""Exercise the next-round launcher, paired interaction summary and restart."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'vgcpod_next_v1_20261020'
sys.path.insert(0,str(ROOT/'vgcpod_v1_roundD'))
import torch
from vgcpod import engine
from vgcpod.common import read_json
from tests.test_math import synthetic_cache

torch.set_num_threads(1)


def main():
    with tempfile.TemporaryDirectory(prefix='vgcpod_next_') as folder:
        temp = Path(folder)
        for stage in ('C','D'):
            target = temp/('source'+stage)
            shutil.copytree(ROOT/('vgcpod_v1_round'+stage),target,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc','validation'))
            for sub in ('round_c','round_d'):
                for config in (target/'configs'/sub).glob('*.json'):
                    cfg = read_json(config)
                    cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=2)
                    config.write_text(json.dumps(cfg),encoding='utf-8')
        synthetic_cache(temp/'cache')
        for case,seeds,group,output in [('A5',[3351,3352,3353],'round_a','A'),
                                       ('C1',[3351],'round_c','C'),
                                       ('C2',[3351,3352,3353],'round_c','C'),
                                       ('D1',[3351],'round_d','D')]:
            config = temp/'sourceD/configs'/group/(case+'.json')
            cfg = read_json(config)
            cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=2)
            for seed in seeds:
                engine.train(argparse.Namespace(cache=str(temp/'cache'),out=str(temp/output/case),
                    device='cpu',seed=seed,resume=False),cfg)
        command = [sys.executable,'-u',str(PACKAGE/'next_round.py'),'--cache',str(temp/'cache'),
            '--d-root',str(temp/'sourceD'),'--c-root',str(temp/'sourceC'),'--d-out',str(temp/'D'),
            '--c-out',str(temp/'C'),'--a-out',str(temp/'A'),'--b-out',str(temp/'B'),
            '--out',str(temp/'next'),'--device','cpu']
        logs = []
        for invocation in ('first','repeat'):
            process = subprocess.run(command,cwd=PACKAGE,capture_output=True,text=True,
                encoding='utf-8',errors='replace',timeout=180)
            logs.append(invocation+'\n'+process.stdout+process.stderr)
            assert process.returncode==0,'\n'.join(logs)
            status = read_json(temp/'next/status.json')
            assert len(status)==3 and all(r['status']!='failed' for r in status),status
            for event in status:
                if event['case']=='D1':
                    assert event['training']['status']==('complete' if invocation=='first' else 'already_complete')
                else:
                    assert event['status']==('complete' if invocation=='first' else 'already_complete')
            assert {(r['case'],r['seed'],r['gpu']) for r in status}=={
                ('D1',3352,'1'),('D1',3353,'2'),('C1',3352,'3')}
            assert 'NEXT_COMPLETE' in process.stdout
        summary = read_json(temp/'D/round_d_summary.json')
        assert len(summary['runs'])==8
        d1 = next(r for r in summary['by_case'] if r['case']=='D1')
        assert d1['seeds']==[3351,3352,3353]
        interaction = read_json(temp/'next/interaction.json')
        assert interaction['seeds']==[3351,3352] and len(interaction['rows'])==2
        mapped = {(r['case'],r['seed']):r for r in summary['runs']}
        for row in interaction['rows']:
            seed = row['seed']
            a5 = read_json(temp/'A/A5/ut-zap50k/V1'/('seed%d'%seed)/'metrics.json')['metrics']
            expected = mapped[('D1',seed)]['test_cw_AUC_native']-mapped[('C2',seed)]['test_cw_AUC_native']-mapped[('C1',seed)]['test_cw_AUC_native']+a5['test_cw']['AUC_native']
            assert row['test_cw_AUC_native']==expected
        for seed in (3352,3353):
            mean = read_json(temp/'D/mean_gate/D1'/('seed%d'%seed)/'metrics.json')
            assert mean['optimizer_steps']==0 and mean['train_images']==16
            log = (temp/'D/logs'/('mean_gate_D1_seed%d.log'%seed)).read_text(encoding='utf-8')
            assert 'MEAN_GATE_ALREADY_COMPLETE' in log
        (PACKAGE/'validation_cli.txt').write_text('\n'.join(logs),encoding='utf-8')
    sys.path.insert(0,str(ROOT/'tmp/round_a_bash_validation'))
    import bashlex
    script = PACKAGE/'start_tmux.sh'
    bashlex.parse(script.read_text(encoding='utf-8'))
    assert b'\r\n' not in script.read_bytes()
    ast.parse((PACKAGE/'next_round.py').read_text(encoding='utf-8'),feature_version=(3,8))
    result = dict(fixture='CPU synthetic CLIP feature cache',training_jobs=3,mean_gate_jobs=2,
        repeat_training_skipped=3,repeat_mean_gate_skipped=2,
        d1_summary_seeds=[3351,3352,3353],interaction_seeds=[3351,3352],
        python38_syntax=True,bash_syntax=True)
    (PACKAGE/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
