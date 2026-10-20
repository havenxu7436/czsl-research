"""Prepare the G0/G1 v1 experiment using the frozen F runtime."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'vgcpod_v1_roundF'
NEW=ROOT/'vgcpod_v1_roundG'


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8',newline='\n')


def main():
    NEW.mkdir(exist_ok=False)
    for name in ('assets','vendor','vgcpod','tests','configs','references'):
        shutil.copytree(OLD/name,NEW/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('run.py','round_a.py','round_b.py','round_c.py','round_d.py','round_e.py','round_f.py',
                 'mean_gate_eval.py','requirements.txt','NOTICE.md'):
        shutil.copy2(OLD/name,NEW/name)
    (NEW/'validation').mkdir()
    cfg=json.loads((OLD/'configs/round_f/F1.json').read_text(encoding='utf-8'))
    cfg.update(task_aux_beta=.075,experiment_id='G1')
    write(NEW/'configs/round_g/G1.json',json.dumps(cfg,indent=2)+'\n')
    cfg.update(gate_input='none',experiment_id='G0')
    write(NEW/'configs/round_g/G0.json',json.dumps(cfg,indent=2)+'\n')
    script=(OLD/'scripts/start_round_f_tmux.sh').read_text(encoding='utf-8')
    for a,b in [('round_f','round_g'),('VG_F','VG_G'),('roundF','roundG'),('ROUND_F','ROUND_G'),('vgcpod_F','vgcpod_G')]:
        script=script.replace(a,b)
    write(NEW/'scripts/start_round_g_tmux.sh',script)
    refs=json.loads((OLD/'references/pre_f_complete.json').read_text(encoding='utf-8'))
    for seed in (3351,3352,3353):
        run=ROOT/'server_results_roundF_20261020/roundF_20261020/F1/ut-zap50k/V1'/('seed%d'%seed)
        protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
        saved=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
        history=json.loads((run/'history.json').read_text(encoding='utf-8'))
        refs['runs'].append(dict(case='F1',seed=seed,config=protocol['config'],
            feature_manifest_sha256=protocol['feature_manifest_sha256'],selected_epoch=saved['selected_epoch'],
            metrics=saved['metrics'],train=history[saved['selected_epoch']-1]['train']))
    assert len(refs['runs'])==20
    write(NEW/'references/pre_g_complete.json',json.dumps(refs,ensure_ascii=False,indent=2)+'\n')
    print('Prepared',NEW)


if __name__=='__main__':
    main()
