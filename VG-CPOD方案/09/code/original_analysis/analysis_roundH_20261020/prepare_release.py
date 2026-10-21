"""Create H1 v1 from G and retain all completed native references."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'vgcpod_v1_roundG'
NEW=ROOT/'vgcpod_v1_roundH'


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8',newline='\n')


def main():
    NEW.mkdir(exist_ok=False)
    for name in ('assets','vendor','vgcpod','tests','configs','references'):
        shutil.copytree(OLD/name,NEW/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('run.py','round_a.py','round_b.py','round_c.py','round_d.py','round_e.py','round_f.py','round_g.py',
                 'ranking_diagnostics.py','mean_gate_eval.py','requirements.txt','NOTICE.md'):
        shutil.copy2(OLD/name,NEW/name)
    (NEW/'validation').mkdir()
    cfg=json.loads((OLD/'configs/round_g/G1.json').read_text(encoding='utf-8'))
    cfg.update(experiment_id='H1',gate_input='train_residual',gate_residual_eps=1e-12)
    write(NEW/'configs/round_h/H1.json',json.dumps(cfg,indent=2)+'\n')
    script=(OLD/'scripts/start_round_g_tmux.sh').read_text(encoding='utf-8')
    for a,b in [('round_g','round_h'),('VG_G','VG_H'),('roundG','roundH'),('ROUND_G','ROUND_H'),('vgcpod_G','vgcpod_H')]:
        script=script.replace(a,b)
    write(NEW/'scripts/start_round_h_tmux.sh',script)
    refs=json.loads((OLD/'references/pre_g_complete.json').read_text(encoding='utf-8'))
    for case in ('G0','G1'):
        for seed in (3351,3352,3353):
            run=ROOT/'server_results_roundG_20261020/roundG_20261020'/case/'ut-zap50k/V1'/('seed%d'%seed)
            load=lambda path:json.loads(path.read_text(encoding='utf-8'))
            protocol,saved,exact,history=load(run/'protocol.json'),load(run/'metrics.json'),load(run/'exact_selection/metrics.json'),load(run/'history.json')
            refs['runs'].append(dict(case=case,seed=seed,config=protocol['config'],
                feature_manifest_sha256=protocol['feature_manifest_sha256'],selected_epoch=saved['selected_epoch'],
                metrics=saved['metrics'],train=history[saved['selected_epoch']-1]['train'],
                exact_selection=exact,train_exact=history[exact['selected_epoch']-1]['train']))
    assert len(refs['runs'])==26
    write(NEW/'references/pre_h_complete.json',json.dumps(refs,ensure_ascii=False,indent=2)+'\n')
    print('Prepared',NEW)


if __name__=='__main__':
    main()
