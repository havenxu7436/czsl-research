"""Read-only verification and optional CPU metric replay of the two-seed export."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).absolute().parents[4]
STAGE = ROOT / '0601方案/02-公式对齐版'
RESULT = STAGE / 'results/seeds3352-3353-10epoch-20261012'
ORDER = list('ABCDEFGH') + ['J0', 'Jtext', 'H_no_pair', 'H_shuffle', 'H_random_prior', 'H_uniform_prior']
FIELDS = ['AUC', 'best_hm', 'best_seen', 'best_unseen', 'hm_seen', 'hm_unseen']


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''): h.update(b)
    return h.hexdigest()


def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(result, replay=False):
    import numpy as np
    import yaml
    raw = result / 'raw_export'
    mapping = read(result / 'export_member_map.json')
    manifest = read(raw / 'export_manifest.json')
    checks = 0

    def require(value, message):
        nonlocal checks
        checks += 1
        if not value: raise ValueError(message)

    expected = manifest['files_sha256']
    require(set(mapping['members']) == set(expected) | {'export_manifest.json'}, 'Export member set differs')
    require(manifest['checkpoints_included'] is False, 'Unexpected model weights in export')
    for member, digest in expected.items():
        rel = Path(mapping['members'][member])
        require(not rel.is_absolute() and not rel.drive and '..' not in rel.parts, 'Unsafe member mapping')
        require(sha(ROOT / rel) == digest, 'Original export bytes differ: ' + member)
    old = STAGE / 'results/seed3351-30epoch-20261012/raw_export'
    old_mapping = read(old.parent / 'export_member_map.json')['members']
    source = {n for n in expected if n.startswith('source/')}
    require(source == {n for n in old_mapping if n.startswith('source/')}, 'Source member set differs from r2')
    for name in source:
        require(sha(ROOT / old_mapping[name]) == expected[name], 'Training source differs from r2: ' + name)
    protocol, summary = read(raw / 'protocol.json'), read(raw / 'summary/summary.json')
    require(protocol['dataset'] == 'ut-zappos' and protocol['epochs'] == 10 and protocol['seeds'] == [3352, 3353], 'Dataset/budget/seeds differ')
    jobs = protocol['jobs']
    require(len(jobs) == 28 and len({j['variant'] for j in jobs}) == 28, 'Unexpected job count')
    for seed in (3352, 3353):
        selected = [j for j in jobs if j['seed'] == seed]
        require(len(selected) == 14 and {j['base_variant'] for j in selected} == set(ORDER), 'Variant matrix incomplete')
    require(summary['core_complete'] and summary['all_planned_complete'], 'Batch incomplete')
    require(summary['protocol_sha256'] == sha(raw / 'protocol.json'), 'Summary protocol SHA differs')
    require(len(summary['rows']) == 112 and len(summary['states']) == 28, 'Evaluation/state count differs')
    require({s['variant'] for s in summary['states']} == {j['variant'] for j in jobs}
            and all(s['state'] == 'complete' for s in summary['states']), 'Summary states differ')
    work = protocol['source'].rsplit('/source', 1)[0] + '/'
    for name, digest in protocol['frozen_inputs'].items():
        if name.startswith(work):
            member = name[len(work):]
            require(member in mapping['members'] and sha(ROOT / mapping['members'][member]) == digest, 'Frozen input differs: ' + member)
    split_dir = ROOT / '前期实验/01-baseline/results/data/ut-zap50k/compositional-split-natural'
    splits = {}
    for phase in ('train', 'val', 'test'):
        path = split_dir / (phase + '_pairs.txt')
        key = protocol['root'] + '/data/ut-zap50k/compositional-split-natural/' + path.name
        require(sha(path) == protocol['frozen_inputs'][key], 'Dataset split differs: ' + phase)
        splits[phase] = [tuple(line.split()) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    old_protocol = read(old / 'protocol.json')

    def normalize(value, prefix):
        if isinstance(value, dict):
            return {k: ('<SEED>' if k == 'seed' else '<EPOCHS>' if k in ('max_epoch', 'final_max_epoch') else normalize(v, prefix)) for k, v in value.items()}
        if isinstance(value, list): return [normalize(v, prefix) for v in value]
        if isinstance(value, str): return re.sub(r'_s335[123](?=\b|/|$)', '_sSEED', value.replace(prefix, '<WORK>/'))
        return value

    if replay:
        import numpy as np
        import torch
        torch.set_num_threads(2)
        require(not torch.cuda.is_available(), 'CPU replay requires CUDA_VISIBLE_DEVICES=-1')
        helper = load(ROOT / '0601方案/01-残差版/code/review_0601_20261011/audit.py', 'two_seed_replay_helpers')
        ev = load(ROOT / mapping['members']['source/evaluator_ge.py'], 'two_seed_native_evaluator')
        require(ev.device == 'cpu', 'Evaluator is not CPU-only')
    labels = read(old / 'runs/A_s3351/meta/labels.json')
    lookup = {(r['variant'], r['phase'], r['setting']): r for r in summary['rows']}
    require(len(lookup) == 112, 'Duplicate summary evaluation keys')
    metrics, curves, replays = [], [], []
    for job in jobs:
        name, base, seed = job['variant'], job['base_variant'], job['seed']
        run = raw / 'runs' / name
        config_path = raw / 'configs' / (name + '.yml')
        cfg = yaml.safe_load(config_path.read_text(encoding='utf8'))
        require(sha(config_path) == job['cfg_sha256'], 'Config hash differs: ' + name)
        require(config_path.read_bytes() == (run / 'config_requested.yml').read_bytes(), 'Requested config differs: ' + name)
        require(cfg['TRAIN']['seed'] == seed and cfg['TRAIN']['max_epoch'] == cfg['TRAIN']['final_max_epoch'] == 10, 'Run seed/budget differs: ' + name)
        baseline = yaml.safe_load((old / ('runs/' + base + '_s3351/config_requested.yml')).read_text(encoding='utf8'))
        require(normalize(cfg, work) == normalize(baseline, old_protocol['source'].rsplit('/source', 1)[0] + '/'), 'Non-seed/budget config change: ' + name)
        state, trained, ckpt = read(run / 'status.json'), read(run / 'train_complete.json'), read(run / 'meta/checkpoint.json')
        require(state['state'] == 'complete' and state['epochs'] == 10 and trained['epochs'] == list(range(1, 11)), 'Incomplete training ledger: ' + name)
        require(state['checkpoint_sha256'] == trained['best_sha256'] == ckpt['sha256'], 'Checkpoint identities differ: ' + name)
        log = (run / 'train.log').read_text(encoding='utf8')
        finished = [int(e) for e in re.findall(r'FULL_EPOCH_FINISHED epoch=(\d+)', log)]
        require(finished == list(range(1, 11)), 'Training log epochs differ: ' + name)
        curve = [{'epoch': int(e), 'val_cw_auc_pct': 100*float(a), 'val_cw_hm_pct': 100*float(h)} for e,a,h in re.findall(r'Val/Test Epoch (\d+): AUC=([\d.eE+-]+) \| best_hm=([\d.eE+-]+)', log)]
        require(len(curve) == 10 and curve == summary['val_curves'][name], 'VAL curve differs: ' + name)
        events = [(int(e), float(a)) for e,a in re.findall(r'Saved new best checkpoint: epoch=(\d+), AUC=([\d.eE+-]+)', log)]
        loaded = [int(e) for e in re.findall(r'Loading validation-best checkpoint from epoch (\d+)', log)]
        require(bool(events) and len(loaded) == 1 and loaded[0] == events[-1][0] and 1 <= loaded[0] <= 10, 'Selected checkpoint event differs: ' + name)
        require(abs(curve[loaded[0]-1]['val_cw_auc_pct'] - max(c['val_cw_auc_pct'] for c in curve)) < 0.011, 'Selected epoch is not logged maximum: ' + name)
        require(read(run / 'meta/labels.json') == labels, 'Primitive ordering differs: ' + name)
        model = cfg['MODEL']
        if model['kcc_feature'] or model['kcc_logic'] or model['formula_joint']:
            smoke = read(run / 'smoke.json')
            require(all(smoke[k] is True for k in ('passed','same_native_initialization','native_cpu_cuda_rng_preserved','joint_formula_exact')), 'Smoke check failed: ' + name)
            require(smoke['optimizer_steps'] == smoke['test_images_read'] == 0 and smoke['train_infer_formula_identity'] == bool(model['formula_joint']), 'Smoke scope differs: ' + name)
            require(all(v > 0 and v < float('inf') for v in smoke['gradients'].values()), 'Nonfinite or missing gradients: ' + name)
        if model['kcc_feature']: require('FULL_GRADIENT epoch=10' in log, 'Final head gradient missing: ' + name)
        if model['kcc_logic']: require('BF_LOGIC epoch=10' in log, 'Final logic evidence missing: ' + name)
        curves.append({'variant': base, 'seed': seed, 'best_epoch': loaded[0], 'actual_epochs': 10, 'val_curve': curve, 'checkpoint_sha256': ckpt['sha256']})
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                key = (name, phase, setting)
                folder = run / ('eval_' + phase + '_' + setting)
                meta = read(folder / 'metrics.json')
                require(meta['seed'] == seed and meta['best_epoch'] == loaded[0] and meta['checkpoint_sha256'] == ckpt['sha256'], 'Eval checkpoint/seed differs: ' + str(key))
                require(meta['config_sha256'] == sha(folder / 'config_eval_effective.yml'), 'Eval config SHA differs: ' + str(key))
                reference = old / ('runs/A_s3351/eval_' + phase + '_' + setting)
                require((folder / 'candidates.json').read_bytes() == (reference / 'candidates.json').read_bytes(), 'Candidate order differs: ' + str(key))
                require((folder / 'image_ids.json').read_bytes() == (reference / 'image_ids.json').read_bytes(), 'Image order differs: ' + str(key))
                with np.load(folder / 'labels.npz', allow_pickle=False) as truth, np.load(reference / 'labels.npz', allow_pickle=False) as baseline_truth:
                    require({'attr','obj','pair'} <= set(truth.files) and all(
                        truth[k].dtype == baseline_truth[k].dtype and np.array_equal(truth[k],baseline_truth[k])
                        for k in ('attr','obj','pair')), 'Ground truth arrays differ: ' + str(key))
                shape = [3214 if phase == 'val' else 2914, 116 if setting == 'cw' else 192]
                require(meta['score_shape'] == shape and meta['score_dtype'] == 'float32' and (folder / 'scores.dat').stat().st_size == shape[0]*shape[1]*4, 'Score shape/size differs: ' + str(key))
                values = {f:float(meta['metrics_raw'][f]) for f in FIELDS}
                if replay:
                    ds = helper.dataset(labels, splits, phase, setting)
                    require(read(folder / 'candidates.json') == [list(p) for p in ds.pairs], 'Native candidate order differs')
                    with np.load(folder / 'labels.npz', allow_pickle=False) as truth:
                        matrix = np.memmap(folder / 'scores.dat', mode='r', dtype=np.float32, shape=tuple(shape))
                        require(np.isfinite(matrix).all(), 'Nonfinite scores: ' + str(key))
                        computed = helper.evaluate(matrix, truth, ds, ev.Evaluator)
                    difference = max(abs(computed[f]-values[f]) for f in FIELDS)
                    require(difference < 2e-6, 'Native metrics differ: ' + str(key))
                    replays.append({'variant':name,'phase':phase,'setting':setting,'max_abs_difference':difference})
                summary_row = lookup[key]
                require(summary_row['seed'] == seed and summary_row['best_epoch'] == loaded[0] and summary_row['checkpoint_sha256'] == ckpt['sha256'], 'Summary checkpoint differs: ' + str(key))
                require(all(abs(summary_row[f+'_pct']-100*values[f]) < 0.0002 for f in FIELDS[:4]), 'Summary metrics differ: ' + str(key))
                metrics.append({'variant':base,'seed':seed,'phase':phase,'setting':setting,'best_epoch':loaded[0], **{f+'_pct':100*v for f,v in values.items()}})
        print('VERIFIED', name, 'best_epoch', loaded[0], flush=True)
    return {'passed':True,'date':'2026-10-12','checks':checks,'source_zip_sha256':mapping['source_zip_sha256'],
            'original_export_files_verified':len(expected),'source_files_matched_r2':len(source),
            'seeds':[3352,3353],'completed_runs':28,'actual_epochs_per_run':10,'completed_epochs':280,
            'evaluations_verified':112,'evaluations_replayed':len(replays),
            'max_metric_replay_abs_difference':max((r['max_abs_difference'] for r in replays),default=None),
            'training_started':False,'config_difference_from_seed3351':'Seed, budget and run paths only',
            'native_metric_replays':replays,'metrics':metrics,'epoch_review':curves}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,default=RESULT)
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--audit-out',type=Path,help='Optional JSON report; never changes original evidence')
    args=parser.parse_args()
    report=verify(args.result.absolute(),args.replay)
    if args.audit_out:
        args.audit_out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','epoch_review','native_metric_replays')},ensure_ascii=False))


if __name__ == '__main__': main()
