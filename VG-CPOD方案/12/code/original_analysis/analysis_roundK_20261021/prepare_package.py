"""Create the K1/K2 experiment package from J, retaining the v1 model."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'vgcpod_v1_roundJ'
PACKAGE = ROOT/'vgcpod_v1_roundK'


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')


def main():
    shutil.copytree(SOURCE, PACKAGE, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', 'validation', 'release_manifest.json',
        'EXPERIMENT_J_v1.md', 'EXPERIMENT_J_v1.html', 'README.md'))
    (PACKAGE/'validation').mkdir()
    for case, multiplier in [('K1', .3), ('K2', .1)]:
        config = json.loads((SOURCE/'configs/round_j/J1.json').read_text(encoding='utf-8'))
        config.update(experiment_id=case, dynamic_readout_lr_multiplier=multiplier)
        write(PACKAGE/'configs/round_k'/(case+'.json'), json.dumps(config, indent=2)+'\n')
    text = (SOURCE/'round_j.py').read_text(encoding='utf-8')
    text = text.replace('J1 separate static/dynamic CP readouts: three trainings and six gate evaluations.',
        'K1/K2 dynamic readout step sizes: six trainings and twelve gate evaluations.')
    text = text.replace('roundJ', 'roundK').replace('round_j', 'round_k').replace('ROUND_J', 'ROUND_K')
    text = text.replace('pre_j_complete', 'pre_k_complete')
    text = text.replace("CASES=('J1',)", "CASES=('K1','K2')")
    text = text.replace("stage=='I'", "stage in ('I','J')")
    text = text.replace("('A','C','D','E','F','G','H','I')", "('A','C','D','E','F','G','H','I','J')")
    text = text.replace("('I',('I1','I2'))]", "('I',('I1','I2')),('J',('J1',))]")
    text = text.replace("('G0','G1','H1','I1','I2')", "('G0','G1','H1','I1','I2','J1')")
    text = text.replace('No completed J1 results', 'No completed K1/K2 results')
    text = text.replace("targets=('I2','G0','G1','H1','I1','D1','E1','F1','A5','A6','C2','C1')",
        "targets=('J1','I2','G0','G1','H1','I1','D1','E1','F1','A5','A6','C2','C1')\n        if row['case']=='K1':targets+=('K2',)")
    text = text.replace("targets=('I2_exact','G0_exact','G1_exact','H1_exact','I1_exact')",
        "targets=('J1_exact','I2_exact','G0_exact','G1_exact','H1_exact','I1_exact')\n        if row['case']=='K1_exact':targets+=('K2_exact',)")
    text = text.replace("('G0','G1','H1','I1','I2','J1') and", "('G0','G1','H1','I1','I2','J1','K1','K2') and")
    text = text.replace("('E1','F1','G0','G1','H1','I1','I2','J1')", "('E1','F1','G0','G1','H1','I1','I2','J1','K1','K2')")
    text = text.replace("glob('J*/seed*/metrics.json')", "glob('K*/seed*/metrics.json')")
    text = text.replace('J1-I2: separate versus shared CP readouts; J1-G0/G1: matched beta/rho controls;',
        'K1/K2-J1: Wd learning rate only; K1-K2: dynamic step size; I2/G0/G1: matched controls;')
    text = text.replace('reference=args.i_out/', 'reference=args.j_out/')
    text = text.replace('TRAIN_GATE_MEAN_COPIED_FROM_I', 'TRAIN_GATE_MEAN_COPIED_FROM_J')
    text = text.replace("'h_out','i_out')", "'h_out','i_out','j_out')")
    text = text.replace('training_jobs 3 mean_gate_jobs 6', 'training_jobs 6 mean_gate_jobs 12')
    text = text.replace('from round_g import completed', 'from round_g import completed\nfrom readout_diagnostics import summarize_readout_optimizer')
    text = text.replace('    mean_effects=[]', "    optimizer_rows=[]\n    for row in refs.values():\n        if row['case'] in ('J1','K1','K2') and 'run_dir' in row and row['feature_manifest_sha256'] in cache_ids:\n            optimizer_rows.extend(summarize_readout_optimizer(row))\n    mean_effects=[]")
    text = text.replace("('mean_gate_effects',mean_effects),('readout_training_summary',readout_rows)",
        "('mean_gate_effects',mean_effects),('readout_training_summary',readout_rows),\n                      ('readout_optimizer_summary',optimizer_rows)")
    text = text.replace('readout_training_runs=readout_rows,', 'readout_training_runs=readout_rows,readout_optimizer_runs=optimizer_rows,')
    write(PACKAGE/'round_k.py', text)
    text = (SOURCE/'ranking_diagnostics.py').read_text(encoding='utf-8')
    text = text.replace("('G1','H1','I1','I2','J1')", "('G1','H1','I1','I2','J1','K1','K2')")
    text = text.replace("            if left['selection']=='native':", "            if left['case'] in ('K1','K2'):\n                targets+=[('J1','dynamic'),('I2','dynamic'),('G1','dynamic'),('H1','dynamic'),('I1','dynamic')]\n                if left['case']=='K1':targets.append(('K2','dynamic'))\n            if left['selection']=='native':")
    write(PACKAGE/'ranking_diagnostics.py', text)
    text = (SOURCE/'scripts/start_round_j_tmux.sh').read_text(encoding='utf-8')
    text = text.replace('round_j','round_k').replace('roundJ','roundK').replace('ROUND_J','ROUND_K')
    text = text.replace('VG_J','VG_K').replace('vgcpod_J','vgcpod_K')
    write(PACKAGE/'scripts/start_round_k_tmux.sh', text)
    refs = json.loads((SOURCE/'references/pre_j_complete.json').read_text(encoding='utf-8'))
    downloaded = ROOT/'server_results_roundJ_20261021/roundJ_20261021'
    for seed in (3351,3352,3353):
        run = downloaded/'J1/ut-zap50k/V1'/('seed%d'%seed)
        read = lambda name: json.loads((run/name).read_text(encoding='utf-8'))
        protocol, metrics, history = read('protocol.json'), read('metrics.json'), read('history.json')
        refs['runs'].append(dict(case='J1', seed=seed, config=protocol['config'],
            selected_epoch=metrics['selected_epoch'],metrics=metrics['metrics'],
            feature_manifest_sha256=protocol['feature_manifest_sha256'],
            train=history[metrics['selected_epoch']-1]['train'],exact_selection=read('exact_selection/metrics.json')))
    write(PACKAGE/'references/pre_k_complete.json',json.dumps(refs,indent=2,ensure_ascii=False)+'\n')
    print(PACKAGE)


if __name__=='__main__':main()
