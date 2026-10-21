"""Create the J1 release from the completed I release and its result snapshots."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vgcpod_v1_roundI'
PACKAGE = ROOT / 'vgcpod_v1_roundJ'


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')


def main():
    shutil.copytree(SOURCE, PACKAGE, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', 'validation', 'release_manifest.json',
        'EXPERIMENT_I_v1.md', 'EXPERIMENT_I_v1.html', 'README.md'))
    (PACKAGE / 'validation').mkdir()
    config = json.loads((SOURCE / 'configs/round_i/I2.json').read_text(encoding='utf-8'))
    config.update(experiment_id='J1', cp_readout_mode='separate')
    write(PACKAGE / 'configs/round_j/J1.json', json.dumps(config, indent=2) + '\n')
    text = (SOURCE / 'round_i.py').read_text(encoding='utf-8')
    text = text.replace('I1/I2 residual magnitude experiments: three GPUs, six trainings and twelve gate evaluations.',
        'J1 separate static/dynamic CP readouts: three trainings and six gate evaluations.')
    text = text.replace('roundI', 'roundJ').replace('round_i', 'round_j').replace('ROUND_I', 'ROUND_J')
    text = text.replace('pre_i_complete', 'pre_j_complete')
    text = text.replace("CASES=('I1','I2')", "CASES=('J1',)")
    text = text.replace("'20261019' if stage=='A' else '20261020'", "'20261019' if stage=='A' else '20261021' if stage=='I' else '20261020'")
    text = text.replace("('A','C','D','E','F','G','H')", "('A','C','D','E','F','G','H','I')")
    text = text.replace("('H',('H1',))]", "('H',('H1',)),('I',('I1','I2'))]")
    text = text.replace("('G0','G1','H1')", "('G0','G1','H1','I1','I2')")
    text = text.replace("No completed I1/I2 results", "No completed J1 results")
    text = text.replace("targets=('H1','G1','G0','E1','F1','D1','A5','A6','C2','C1')\n        if row['case']=='I1':targets+=('I2',)",
        "targets=('I2','G0','G1','H1','I1','D1','E1','F1','A5','A6','C2','C1')")
    text = text.replace("targets=('G0_exact','G1_exact','H1_exact')\n        if row['case']=='I1_exact':targets+=('I2_exact',)",
        "targets=('I2_exact','G0_exact','G1_exact','H1_exact','I1_exact')")
    text = text.replace("('G0','G1','H1','I1','I2') and", "('G0','G1','H1','I1','I2','J1') and")
    text = text.replace("('E1','F1','G0','G1','H1','I1','I2')", "('E1','F1','G0','G1','H1','I1','I2','J1')")
    text = text.replace("glob('I*/seed*/metrics.json')", "glob('J*/seed*/metrics.json')")
    text = text.replace("comparison_rule='I1/I2-H1: residual normalization and temperature; I1-I2: magnitude vs direction at representative scale; I1/I2-G0/G1: matched controls; native/exact references matched by selection'",
        "comparison_rule='J1-I2: separate versus shared CP readouts; J1-G0/G1: matched beta/rho controls; D1: performance reference; native/exact references matched by selection'")
    text = text.replace("reference=args.h_out/", "reference=args.i_out/")
    text = text.replace('TRAIN_GATE_MEAN_COPIED_FROM_H', 'TRAIN_GATE_MEAN_COPIED_FROM_I')
    text = text.replace("'f_out','g_out','h_out')", "'f_out','g_out','h_out','i_out')")
    text = text.replace('training_jobs 6 mean_gate_jobs 12', 'training_jobs 3 mean_gate_jobs 6')
    write(PACKAGE / 'round_j.py', text)
    script = (SOURCE / 'scripts/start_round_i_tmux.sh').read_text(encoding='utf-8')
    script = script.replace('round_i', 'round_j').replace('roundI', 'roundJ').replace('ROUND_I', 'ROUND_J')
    script = script.replace('VG_I', 'VG_J').replace('vgcpod_I', 'vgcpod_J')
    write(PACKAGE / 'scripts/start_round_j_tmux.sh', script)
    refs = json.loads((SOURCE / 'references/pre_i_complete.json').read_text(encoding='utf-8'))
    downloaded = ROOT / 'server_results_roundI_20261021/roundI_20261021'
    for case in ('I1', 'I2'):
        for seed in (3351, 3352, 3353):
            run = downloaded / case / 'ut-zap50k/V1' / ('seed%d' % seed)
            read = lambda name: json.loads((run / name).read_text(encoding='utf-8'))
            protocol, metrics, history = read('protocol.json'), read('metrics.json'), read('history.json')
            refs['runs'].append(dict(case=case, seed=seed, config=protocol['config'],
                selected_epoch=metrics['selected_epoch'], metrics=metrics['metrics'],
                feature_manifest_sha256=protocol['feature_manifest_sha256'],
                train=history[metrics['selected_epoch']-1]['train'], exact_selection=read('exact_selection/metrics.json')))
    write(PACKAGE / 'references/pre_j_complete.json', json.dumps(refs, indent=2, ensure_ascii=False) + '\n')
    print(PACKAGE)


if __name__ == '__main__':
    main()
