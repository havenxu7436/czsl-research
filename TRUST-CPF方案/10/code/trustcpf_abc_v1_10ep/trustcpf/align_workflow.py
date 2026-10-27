"""First-batch queue: import, frozen policy scan, three adaptations, Q5 audit."""
import copy
import csv
import os
import platform
import shlex
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, read_json, write_json, load_torch, save_torch, sha, source_hashes
from .runtime import lock
from . import align, decision_audit

SEEDS = (3351, 3352, 3353)
STAGES = ('import', 'scan', 'adapt_uniform', 'adapt_align', 'adapt_permuted', 'decision')
COPY_FILES = ('reference/last.pt', 'folds.json', 'responses/reference.pt', 'responses/fold5.pt',
              'utility/compact/model.pt', 'utility_all/compact/model.pt', 'utility/selection.json')


def sources(args):
    seed = 'seed%d' % args.seed
    return Path(args.source).resolve() / 'work' / seed, Path(args.teacher_source).resolve() / 'work' / seed


def validate_sources(args):
    src, teacher = sources(args)
    bank = load_torch(src / 'runs/compact/bank.pt')
    ref = src / 'reference/last.pt'; q5 = teacher / 'teachers/5/last.pt'
    if sha(ref) != sha(teacher / 'reference/last.pt'):
        raise ValueError('Compact and Strong reference differ')
    folds = read_json(src / 'folds.json')
    if folds != read_json(teacher / 'folds.json'):
        raise ValueError('Source folds differ')
    for name, checkpoint in [('reference', ref), ('fold5', q5)]:
        value = load_torch(src / 'responses' / (name + '.pt'))
        if value['checkpoint_sha256'] != sha(checkpoint):
            raise ValueError('Response checkpoint mismatch: ' + name)
        if tuple(value['features'].shape) != (len(bank['types']), 8) or not torch.isfinite(value['features']).all():
            raise ValueError('Invalid eight-dimensional response: ' + name)
        if name == 'fold5' and set(map(tuple, value['train_pairs'])) & set(map(tuple, folds['query'][4])):
            raise ValueError('Q5 teacher response overlaps query pairs')
    audit = load_torch(src / 'utility/compact/model.pt')
    final = load_torch(src / 'utility_all/compact/model.pt')
    if audit['fit_fold_ids'] != [1, 2, 3] or final['fit_fold_ids'] != [1, 2, 3, 4, 5]:
        raise ValueError('Unexpected audit/final Compact fit folds')
    if audit['alpha'] != final['alpha']:
        raise ValueError('Audit and final predictors must share selected alpha')
    for checkpoint in (ref, q5):
        state = load_torch(checkpoint)
        for key in ('anchors', 'source', 'target', 'types', 'degree'):
            if not torch.equal(state['model']['graph.' + key], bank[key]):
                raise ValueError('Teacher graph differs: ' + key)
    q5_train = set(map(tuple, load_torch(q5)['train_pairs']))
    if q5_train & set(map(tuple, folds['query'][4])):
        raise ValueError('Teacher trained on Q5 pairs')
    return bank


def setup(args):
    from .cache_validation import validate_cache
    src, teacher, cache = Path(args.source).resolve(), Path(args.teacher_source).resolve(), Path(args.cache).resolve()
    for base in (src, teacher):
        if not read_json(base / 'work/status.json')['complete']:
            raise ValueError('Source experiment incomplete: ' + str(base))
        for name in ('model.py', 'g0.py', 'engine.py', 'evaluation.py', 'metrics.py'):
            if sha(base / 'trustcpf' / name) != sha(ROOT / 'trustcpf' / name):
                raise ValueError('Source classifier/evaluator differs: ' + name)
        if sha(base / 'assets/knowledge.pt') != sha(ROOT / 'assets/knowledge.pt'):
            raise ValueError('Knowledge bank differs')
    if (ROOT / 'MANIFEST.json').exists():
        for name, h in read_json(ROOT / 'MANIFEST.json').items():
            if sha(ROOT / name) != h:
                raise ValueError('Release file differs: ' + name)
    print('CHECKING_FEATURE_CACHE', cache, flush=True)
    validate_cache(cache)
    for seed in SEEDS:
        if sha(src / 'initial' / ('g0_seed%d.pt' % seed)) != sha(ROOT / 'initial' / ('g0_seed%d.pt' % seed)):
            raise ValueError('Initial CPF checkpoint differs')
        a = copy.copy(args); a.seed = seed
        validate_sources(a)
        print('SOURCE_READY', seed, flush=True)
    paths = dict(CPF_PY=sys.executable, CPF_SOURCE=str(src), CPF_TEACHER_SOURCE=str(teacher),
                 CPF_CACHE=str(cache), CPF_WORK=str(Path(args.work).resolve()), CPF_CONFIG=str(Path(args.config).resolve()))
    (ROOT / 'server_env.sh').write_bytes(''.join('export ' + k + '=' + shlex.quote(v) + '\n' for k, v in paths.items()).encode('utf8'))
    write_json(ROOT / 'setup.json', dict(paths=paths, python=platform.python_version(), torch=torch.__version__,
               numpy=np.__version__, cuda=torch.cuda.is_available(), gpu_count=torch.cuda.device_count()))
    print('SETUP_COMPLETE', paths, flush=True)
    return paths


def identity(args):
    src, teacher = sources(args)
    files = list(COPY_FILES) + ['runs/compact/bank.pt', 'recipe.json', 'reference_benchmarks.json']
    files += ['runs/%s/metrics.json' % v for v in ('drop', 'compact', 'shuffled_utility')]
    return dict(seed=args.seed, source=str(src), teacher_source=str(teacher), config=read_json(args.config),
        source_files={n: sha(src / n) for n in files}, teacher_sha256=sha(teacher / 'teachers/5/last.pt'),
        implementation=source_hashes(), cache_manifest=sha(Path(args.cache) / 'features_manifest.json'),
        knowledge=sha(ROOT / 'assets/knowledge.pt'), initial=sha(ROOT / 'initial' / ('g0_seed%d.pt' % args.seed)))


def prepare(args):
    src, teacher = sources(args); work = Path(args.work)
    bank = validate_sources(args)
    cfg = dict(read_json(src / 'recipe.json'))
    cfg.update(read_json(args.config)); cfg.update(seed=args.seed, variant='align', readout='probability')
    cfg['fixed_type_budgets'] = {str(t): float((bank['types'] == t).sum()) / 2 for t in range(3)}
    if cfg['graph_seed'] != cfg['eval_graph_seeds'][0]:
        raise ValueError('Selection bank must be first evaluation bank')
    if cfg['audit_graphs'] < 2 or len(cfg['audit_eval_seeds']) < 2:
        raise ValueError('Audit requires multiple graphs and independent banks')
    write_json(work / 'recipe.json', cfg)
    for name in COPY_FILES:
        dest = work / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / name, dest)
    dest = work / 'teacher_q5/last.pt'; dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(teacher / 'teachers/5/last.pt', dest)
    save_torch(work / 'inputs/bank.pt', bank)
    history = read_json(src / 'reference_benchmarks.json')
    for variant in ('drop', 'compact', 'shuffled_utility'):
        history['old_' + variant] = read_json(src / 'runs' / variant / 'metrics.json')
    write_json(work / 'reference_benchmarks.json', history)
    write_json(work / 'reuse.json', dict(source_seed=str(src), teacher_seed=str(teacher),
        reference_sha256=sha(src / 'reference/last.pt'), q5_teacher_sha256=sha(teacher / 'teachers/5/last.pt'),
        final_predictor='Q1-Q5 refit', audit_predictor='Q1-Q3 fit, Q4 alpha selection',
        graph_budget=cfg['fixed_type_budgets']))


def outputs(stage, folder):
    if stage == 'import':
        names = list(COPY_FILES) + ['recipe.json', 'inputs/bank.pt', 'teacher_q5/last.pt',
                                   'reference_benchmarks.json', 'reuse.json']
    elif stage == 'scan':
        names = ['selection.json', 'policies.pt'] + [p.relative_to(folder).as_posix() for p in (folder / 'scan').rglob('*.json')]
    elif stage == 'decision':
        names = [p.relative_to(folder).as_posix() for p in (folder / 'decision').rglob('*') if p.is_file() and not p.name.endswith('.writing')]
    else:
        root = folder / 'runs' / stage[len('adapt_'):]
        names = [p.relative_to(folder).as_posix() for p in root.rglob('*') if p.is_file() and not p.name.endswith('.writing')]
    return sorted(names)


def run_seed(args):
    root = Path(args.work).resolve(); folder = root / ('seed%d' % args.seed)
    marks = root / 'progress' / ('seed%d' % args.seed)
    with lock(marks):
        ident = identity(args)
        if (marks / 'identity.json').exists() and read_json(marks / 'identity.json') != ident:
            raise ValueError('Inputs or recipe changed; select a new --work directory')
        write_json(marks / 'identity.json', ident)
        a = copy.copy(args); a.work = str(folder)
        for stage in STAGES:
            marker = marks / (stage + '.json')
            if marker.exists():
                for rel, h in read_json(marker)['outputs'].items():
                    if not (folder / rel).is_file() or sha(folder / rel) != h:
                        raise ValueError('Completed output changed: ' + rel)
                print('REUSE_STAGE', args.seed, stage, flush=True)
                continue
            align.synchronize(args.device); tic = time.perf_counter()
            if str(args.device).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(args.device)
            print('START_STAGE', args.seed, stage, flush=True)
            if stage == 'import':
                prepare(a)
            elif stage == 'scan':
                align.frozen_scan(a)
            elif stage == 'decision':
                decision_audit.run(a)
            else:
                align.adapt_and_evaluate(a, stage[len('adapt_'):])
            align.synchronize(args.device)
            write_json(marker, dict(stage=stage, seconds=time.perf_counter() - tic, completed=True,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else 0,
                outputs={p: sha(folder / p) for p in outputs(stage, folder)}))
            print('END_STAGE', args.seed, stage, flush=True)
        write_json(marks / 'DONE.json', dict(seed=args.seed, completed=True, stages=len(STAGES)))
    print('SEED_COMPLETE', args.seed, flush=True)


def csv_write(path, rows):
    if rows:
        with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def report(args):
    root = Path(args.work).resolve(); root.mkdir(parents=True, exist_ok=True)
    rows, historical, scans, risks, costs, deployed, utility, paired = [], [], [], [], [], [], [], []
    latency, decisions, adapted, stability = [], [], [], []
    missing = []; stages = metrics_count = audits = 0
    for seed in SEEDS:
        folder = root / ('seed%d' % seed)
        for stage in STAGES:
            marker = root / 'progress' / ('seed%d' % seed) / (stage + '.json')
            if not marker.exists():
                missing.append('%d/%s' % (seed, stage)); continue
            record = read_json(marker); stages += 1
            costs.append(dict(seed=seed, stage=stage, seconds=record['seconds'], peak_gpu_bytes=record['peak_gpu_bytes']))
            for name, h in record['outputs'].items():
                if not (folder / name).is_file() or sha(folder / name) != h:
                    missing.append('%d/changed:%s' % (seed, name))
        for variant in align.VARIANTS:
            file = folder / 'runs' / variant / 'metrics.json'
            if not file.exists():
                continue
            result = read_json(file); metrics_count += 1
            speed = result['inference_latency']
            latency.append(dict(seed=seed, variant=variant, batch_images=speed['batch_images'], graphs=speed['graphs'],
                                milliseconds_per_image=speed['milliseconds_per_image'], repeats=speed['repeats']))
            selection = read_json(folder / 'selection.json')
            for snapshot, banks in result['results'].items():
                for bank_seed, b in banks.items():
                    for phase in ('val', 'test'):
                        for world in ('cw', 'ow'):
                            m = b[phase][world]
                            rows.append(dict(seed=seed, variant=variant, snapshot=snapshot, bank=int(bank_seed),
                                epoch=b['epoch'], phase=phase, world=world, AUC_native=m['AUC_native'], AUC_exact=m['AUC_exact'],
                                HM=m['at_val_bias']['HM'], NLL=m['calibrated']['NLL'], raw_NLL=m['unbiased']['NLL'],
                                ECE=m['calibrated']['ECE'], seen=m['at_val_bias']['seen'], unseen=m['at_val_bias']['unseen'],
                                gt_attr_unseen=m['gt_object_attr_acc_unseen'], graph_variance=m['graph_variance']))
                    for ty, d in b['deployment'].items():
                        deployed.append(dict(seed=seed, variant=variant, snapshot=snapshot, bank=int(bank_seed),
                            epoch=b['epoch'], strength=selection['strength'] if variant != 'uniform' else 0., type=int(ty),
                            expected_edges=d['expected_edges'], probability_std=d['probability_std'],
                            probability_min=d['probability_min'], probability_max=d['probability_max'], kl=d['kl'],
                            expected_covered_nodes=float(sum(d['expected_node_coverage'])),
                            prototype_displacement=d['prototype_displacement_mean']))
        if (folder / 'scan/results.json').exists():
            for r in read_json(folder / 'scan/results.json'):
                scans.append(dict(seed=seed, strength=r['strength'], permutation_seed=r['permutation_seed'],
                    key=r['key'], VAL_CW_AUC=r['val']['cw']['AUC_native'], VAL_OW_AUC=r['val']['ow']['AUC_native']))
        if (folder / 'decision/audit.json').exists():
            a = read_json(folder / 'decision/audit.json'); audits += 1
            risks.extend(dict(seed=seed, **r) for r in a['rows'])
            for name, p in a['policies'].items():
                r = dict(seed=seed, policy=name, eta=p['eta'], local=p['local'], permuted=p['permuted'])
                for metric in ('single_nll', 'mixture_nll', 'single_gain', 'mixture_gain'):
                    for stat in ('mean', 'sd', 'mcse'):
                        r[metric + '_' + stat] = p[metric][stat]
                for target, value in p['proxy'].items():
                    r['additive_proxy_' + target] = value
                decisions.append(r)
            for ty, values in a['correlations'].items():
                for name, v in values.items():
                    utility.append(dict(seed=seed, type=int(ty), comparison=name, **v))
        if (folder / 'reference_benchmarks.json').exists():
            for name, value in read_json(folder / 'reference_benchmarks.json').items():
                for world in ('cw', 'ow'):
                    m = value['test'][world]
                    historical.append(dict(seed=seed, variant=name, world=world, AUC_native=m['AUC_native'], AUC_exact=m['AUC_exact']))
    index = {(r['seed'], r['variant'], r['snapshot'], r['bank'], r['phase'], r['world']): r for r in rows}
    for r in rows:
        if r['snapshot'] != 'best':
            continue
        key = (r['seed'], r['variant'], 'epoch0', r['bank'], r['phase'], r['world'])
        if key in index:
            b = index[key]
            adapted.append(dict(seed=r['seed'], variant=r['variant'], bank=r['bank'], phase=r['phase'], world=r['world'],
                                epoch=r['epoch'], AUC_gain=r['AUC_native']-b['AUC_native'], HM_gain=r['HM']-b['HM']))
    bank_groups = {}
    for r in rows:
        key = (r['seed'], r['variant'], r['snapshot'], r['phase'], r['world'])
        bank_groups.setdefault(key, []).append(r)
    for key, items in bank_groups.items():
        x = np.array([r['AUC_native'] for r in items])
        stability.append(dict(zip(('seed','variant','snapshot','phase','world'),key),
            graph_banks=len(x), AUC_bank_mean=float(x.mean()), AUC_bank_sd=float(x.std(ddof=1)) if len(x)>1 else None))
    for r in rows:
        if r['variant'] != 'align':
            continue
        for control in ('uniform', 'permuted'):
            key = (r['seed'], control, r['snapshot'], r['bank'], r['phase'], r['world'])
            if key in index:
                b = index[key]
                paired.append(dict(seed=r['seed'], comparison='align-minus-' + control, snapshot=r['snapshot'],
                    bank=r['bank'], phase=r['phase'], world=r['world'],
                    AUC_native=r['AUC_native']-b['AUC_native'], AUC_exact=r['AUC_exact']-b['AUC_exact'],
                    HM=r['HM']-b['HM'], raw_NLL_gain=b['raw_NLL']-r['raw_NLL']))
    aggregates = []
    for table, name, group_keys in [(rows, 'recognition', ('variant', 'snapshot', 'bank', 'phase', 'world')),
                                   (paired, 'paired', ('comparison', 'snapshot', 'bank', 'phase', 'world'))]:
        groups = {}
        for r in table:
            groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
        for key, items in groups.items():
            x = np.array([r['AUC_native'] for r in items])
            aggregates.append(dict(table=name, group=dict(zip(group_keys, key)), n_seeds=len(x),
                AUC_mean=float(x.mean()), AUC_sd=float(x.std(ddof=1)) if len(x)>1 else None,
                seeds=[r['seed'] for r in items]))
    for name, table in [('summary', rows), ('historical', historical), ('scan', scans), ('decision_risks', risks),
                        ('utility_correlations', utility), ('deployment', deployed), ('paired_comparisons', paired), ('costs', costs),
                        ('inference_latency', latency), ('decision_summary', decisions), ('adaptation_gain', adapted),
                        ('bank_stability', stability)]:
        csv_write(root / (name + '.csv'), table)
    write_json(root / 'aggregates.json', aggregates)
    status = dict(completed_stages=stages, expected_stages=len(SEEDS)*len(STAGES), metrics=metrics_count,
                  expected_metrics=9, utility_audits=audits, expected_utility_audits=3,
                  summary_rows=len(rows), scan_rows=len(scans), decision_bank_rows=len(risks), missing=missing)
    status['complete'] = stages == status['expected_stages'] and metrics_count == 9 and audits == 3 and not missing
    write_json(root / 'status.json', status)
    print('REPORT', root / 'summary.csv', flush=True)
    print('STATUS', stages, '/', status['expected_stages'], 'models', metrics_count, '/9', 'audits', audits, '/3',
          'complete', status['complete'], flush=True)
    return status
