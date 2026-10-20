"""Prepare F1 from the completed E1 implementation and reference records."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'vgcpod_v1_roundE'
NEW = ROOT/'vgcpod_v1_roundF'


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')


def main():
    NEW.mkdir(exist_ok=False)
    for folder in ('assets','vendor','vgcpod','tests','configs','references'):
        shutil.copytree(OLD/folder,NEW/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('run.py','round_a.py','round_b.py','round_c.py','round_d.py','round_e.py',
                 'mean_gate_eval.py','requirements.txt','NOTICE.md'):
        shutil.copy2(OLD/name,NEW/name)
    (NEW/'validation').mkdir()
    cfg = json.loads((OLD/'configs/round_e/E1.json').read_text(encoding='utf-8'))
    cfg.update(task_aux_beta=.05,experiment_id='F1')
    write(NEW/'configs/round_f/F1.json',json.dumps(cfg,indent=2)+'\n')
    engine = (NEW/'vgcpod/engine.py').read_text(encoding='utf-8')
    helper = '''def task_decomposition(logs, beta):
    """Detached scalar diagnostics; preserve the objective's operation order."""
    support_weight = 1.5 * beta / (1.0 + beta)
    pair_weighted = 1.5 * logs['pair']
    support_weighted = support_weight * logs['support']
    return dict(support_weight=support_weight, pair_weighted=pair_weighted,
                support_weighted=support_weighted,
                task_reconstruction_error=abs(logs['task'] - pair_weighted - support_weighted))


'''
    engine = engine.replace('def train(args, cfg):',helper+'def train(args, cfg):')
    engine = engine.replace("            logs.update(gradient_norm=float(norm)",
        "            logs.update(task_decomposition(logs, cfg.get('task_aux_beta', 0.5)))\n"
        "            logs.update(gradient_norm=float(norm)")
    write(NEW/'vgcpod/engine.py',engine)
    script = (OLD/'scripts/start_round_e_tmux.sh').read_text(encoding='utf-8')
    for a,b in [('round_e','round_f'),('VG_E','VG_F'),('roundE','roundF'),('ROUND_E','ROUND_F'),('vgcpod_E','vgcpod_F')]:
        script = script.replace(a,b)
    write(NEW/'scripts/start_round_f_tmux.sh',script)
    refs = json.loads((OLD/'references/pre_e_complete.json').read_text(encoding='utf-8'))
    for seed in (3351,3352,3353):
        run = ROOT/'server_results_roundE_20261020/roundE_20261020/E1/ut-zap50k/V1'/('seed%d'%seed)
        protocol = json.loads((run/'protocol.json').read_text(encoding='utf-8'))
        saved = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
        history = json.loads((run/'history.json').read_text(encoding='utf-8'))
        refs['runs'].append(dict(case='E1',seed=seed,config=protocol['config'],
            feature_manifest_sha256=protocol['feature_manifest_sha256'],
            selected_epoch=saved['selected_epoch'],metrics=saved['metrics'],
            train=history[saved['selected_epoch']-1]['train']))
    assert len(refs['runs'])==17
    write(NEW/'references/pre_f_complete.json',json.dumps(refs,ensure_ascii=False,indent=2)+'\n')
    print('Prepared',NEW)


if __name__=='__main__':
    main()
