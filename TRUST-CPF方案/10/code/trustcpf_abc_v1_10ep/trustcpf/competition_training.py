"""Shared dual-group teachers, three utility labels, ridge and Q5 decisions."""
import math
from pathlib import Path
import numpy as np
import torch
from .common import read_json, write_json, load_torch, save_torch, sha
from . import align, engine, strong, compact, response, competition as cmp
from .data import FeatureDataset
from .folds import pair_ids
from .metrics import native_curve, exact_curve


def teacher(args, fold):
    work, cfg, manifest, text, bank = align.context(args)
    split = read_json(work / 'splits' / ('fold%d.json' % fold))
    out = work / 'teachers' / str(fold)
    local_cfg = dict(cfg, _cold_backbone=True)
    initial = strong.fold_backbone(args, local_cfg, text, bank, manifest, split['inner_pairs'],
                                   out / 'backbone', train_ids=split['train_ids'])
    engine.train_model(args, local_cfg, text, bank, manifest, split['inner_pairs'], out,
                       initial=initial, train_ids=split['train_ids'])
    model = align.restore(text, bank, local_cfg, args.device, out / 'last.pt')
    build_response(args, cfg, manifest, bank, model, split, work / 'responses' / ('fold%d.pt' % fold), out / 'last.pt')
    measure(args, fold, model)


@torch.no_grad()
def build_response(args, cfg, manifest, bank, model, split, output, checkpoint):
    if output.exists():
        return
    rows = read_json(Path(args.cache) / 'train_records.json')
    allowed_ids = split['train_ids']
    local_ids, weights = response.select_support([rows[i] for i in allowed_ids], split['inner_pairs'],
                        manifest['labels'], cfg['response_images_per_pair'], cfg['response_seed']+split['fold'])
    ids = [allowed_ids[i] for i in local_ids]
    ds = FeatureDataset(args.cache, 'train'); samples = [ds[i] for i in ids]
    attrs = torch.tensor([s[3] for s in samples]); objs = torch.tensor([s[4] for s in samples])
    candidates = torch.tensor(pair_ids(manifest['labels'], manifest['pairs']['train']), device=args.device)
    index = {int(v): i for i, v in enumerate(candidates.cpu())}
    truth = torch.tensor([index[int(o)*len(manifest['labels']['attrs'])+int(a)] for a, o in zip(attrs, objs)])
    size = cfg['eval_batch_size']
    batches = [[torch.stack([s[k] for s in samples[start:start+size]]) for k in range(3)]
               for start in range(0, len(samples), size)]
    x, count = response.measure_responses(model, batches, truth, weights, attrs, objs, candidates,
                     bank, cfg['response_backgrounds'], cfg['response_seed']+split['fold'])
    save_torch(output, dict(features=x, names=response.FEATURE_NAMES, support_indices=ids, weights=weights,
        train_pairs=split['inner_pairs'], checkpoint_sha256=sha(checkpoint), head_forwards=count,
        backgrounds=cfg['response_backgrounds'], seed=cfg['response_seed']+split['fold']))
    print('RESPONSE_COMPLETE', args.seed, split['fold'], len(ids), flush=True)


@torch.no_grad()
def prepared(args, cfg, manifest, model, split, partition):
    ids = split[partition]['seen'] + split[partition]['unseen']
    base = {}; attrs = []; objs = []
    for batch in engine.batches(args.cache, 'train', cfg, ids):
        b = model.base(*[x.to(args.device) for x in batch[:3]])
        for key, value in b.items(): base.setdefault(key, []).append(value)
        attrs.extend(batch[3].tolist()); objs.extend(batch[4].tolist())
    base = {k: torch.cat(v) for k, v in base.items()}
    candidates = torch.tensor(pair_ids(manifest['labels'], manifest['pairs']['train']), device=args.device)
    index = {int(v): i for i, v in enumerate(candidates.cpu())}
    truth = torch.tensor([index[o*len(manifest['labels']['attrs'])+a] for a, o in zip(attrs, objs)], device=args.device)
    inner = set(map(tuple, split['inner_pairs']))
    seen = torch.tensor([tuple(p) in inner for p in manifest['pairs']['train']], device=args.device)
    return base, candidates, truth, seen


@torch.no_grad()
def score(model, base, mask, candidates, batch_size):
    count = len(next(iter(base.values())))
    return torch.cat([model.score({k: v[start:start+batch_size] for k, v in base.items()}, mask)[0]
                      .flatten(1)[:, candidates].double().log_softmax(-1)
                      for start in range(0, count, batch_size)])


@torch.no_grad()
def measure(args, fold, model):
    work, cfg, manifest, text, bank = align.context(args)
    split = read_json(work / 'splits' / ('fold%d.json' % fold))
    base, candidates, truth, seen = prepared(args, cfg, manifest, model, split, 'measure')
    sm = seen[truth]; edges = len(bank['types'])
    k = cfg['q5_measurement_blocks'] if fold == 5 else cfg['measurement_blocks']
    values = []
    for block in range(k):
        file = work / 'measurements' / ('fold%d' % fold) / ('block%02d.pt' % block)
        if file.exists():
            saved = load_torch(file)
        else:
            gen = torch.Generator().manual_seed(cfg['measurement_seed']+1000*fold+block)
            mask = (torch.rand(edges, generator=gen)<.5).float().to(args.device)
            off, on = [], []
            for edge in range(edges):
                m = mask.clone(); m[edge] = 0
                off.append(cmp.sufficient(score(model, base, m, candidates, cfg['eval_batch_size']), truth, seen))
                m[edge] = 1
                on.append(cmp.sufficient(score(model, base, m, candidates, cfg['eval_batch_size']), truth, seen))
            off, on = torch.stack(off), torch.stack(on)
            diff = torch.stack([cmp.risks(a, sm, cfg['competition_temperature'], cfg['risk_pair_chunk'])-
                               cmp.risks(b, sm, cfg['competition_temperature'], cfg['risk_pair_chunk'])
                               for a, b in zip(off, on)])
            saved = dict(mask=mask.cpu(), off=off.cpu(), on=on.cpu(), effects=diff.cpu(),
                         truth=truth.cpu(), seen_candidates=seen.cpu(), sample_seen=sm.cpu(),
                         columns=['true_logprob', 'within_margin', 'crossing', 'hard_gate'], targets=cmp.TARGETS)
            save_torch(file, saved)
        values.append(saved['effects'].double().numpy())
        print('UTILITY_BLOCK', args.seed, 'fold', fold, block+1, '/', k, flush=True)
    x = np.stack(values)
    save_torch(work / 'measurements' / ('fold%d.pt' % fold),
               dict(target=x.mean(0), se2=x.var(0, ddof=1)/k, blocks=x, targets=cmp.TARGETS,
                    teacher_sha256=sha(work / 'teachers' / str(fold) / 'last.pt'), split=split))


def fit(args):
    work, cfg, manifest, text, bank = align.context(args)
    types = bank['types'].numpy()
    xs = [load_torch(work / 'responses' / ('fold%d.pt' % j))['features'].double().numpy() for j in range(1, 6)]
    ys = [load_torch(work / 'measurements' / ('fold%d.pt' % j))['target'] for j in range(1, 6)]
    xr = load_torch(work / 'responses/reference.pt')['features'].double().numpy()
    selections, predictions, audit = {}, {}, {}
    for qi, name in enumerate(cmp.TARGETS):
        y = [v[:, qi] for v in ys]; trials = []
        for alpha in cfg['ridge_alphas']:
            model = compact.fit_ridge(xs[:3], y[:3], types, alpha)
            p = compact.predict(model, xs[3], types)
            trials.append(dict(alpha=alpha, q4_error=cmp.centered_type_error(p, y[3], types, model['scales'])))
        selected = min(trials, key=lambda v: (v['q4_error'], -v['alpha']))
        selections[name] = dict(selected_alpha=selected['alpha'], trials=trials,
            rule='Q4 centered type-normalized MSE; ties larger alpha; absolute ridge fit')
        for n, dest in [(3, 'utility'), (5, 'utility_all')]:
            model = compact.fit_ridge(xs[:n], y[:n], types, selected['alpha'])
            model.update(fit_fold_ids=list(range(1, n+1)), target=name)
            save_torch(work / dest / name / 'model.pt', model)
            if n == 3:
                p5 = compact.predict(model, xs[4], types)
            else:
                pr = compact.predict(model, xr, types)
        predictions[name] = dict(q5=p5, reference=pr, q5_target=y[4])
        audit[name] = dict(all=compact.point_metrics(p5, y[4]),
            by_type={str(t): compact.point_metrics(p5[types == t], y[4][types == t]) for t in range(3)},
            q5_to_reference_rank={str(t): compact.point_metrics(p5[types == t], pr[types == t]) for t in range(3)})
        print('UTILITY_FIT', args.seed, name, 'alpha', selected['alpha'], flush=True)
    write_json(work / 'utility/selection.json', selections)
    write_json(work / 'utility/audit.json', audit)
    save_torch(work / 'utility/predictions.pt', predictions)


@torch.no_grad()
def audit(args):
    work, cfg, manifest, text, bank = align.context(args)
    types = bank['types'].numpy(); split = read_json(work / 'splits/fold5.json')
    model = align.restore(text, bank, cfg, args.device, work / 'teachers/5/last.pt')
    base, candidates, truth, seen = prepared(args, cfg, manifest, model, split, 'evaluate')
    sm = seen[truth]; predictions = load_torch(work / 'utility/predictions.pt')
    policies = {'uniform': torch.full((len(types),), .5, dtype=torch.double)}
    perm = align.permutation(types, cfg['permutation_seeds'][0])
    for target in cmp.TARGETS:
        for source, key in [('predicted', 'q5'), ('measured', 'q5_target')]:
            p, _ = align.policy(predictions[target][key], types, 1.)
            name = source+'_'+target
            policies[name] = p; policies[name+'_permuted'] = p[perm]
    rows = []
    for graph_seed in cfg['audit_eval_seeds']:
        file = work / 'decision/blocks' / ('evaluate%d.pt' % graph_seed)
        if file.exists():
            saved = load_torch(file)
        else:
            gen = torch.Generator().manual_seed(graph_seed)
            uniforms = torch.rand(cfg['audit_graphs'], len(types), generator=gen).double()
            results = {}
            for name, p in policies.items():
                masks = (uniforms < p[None, :]).float().to(args.device)
                logs = torch.stack([score(model, base, m, candidates, cfg['eval_batch_size']) for m in masks])
                single_stats = torch.stack([cmp.sufficient(z, truth, seen) for z in logs])
                single = [cmp.diagnostics(s, sm, cfg['competition_temperature'], cfg['risk_pair_chunk']) for s in single_stats]
                mixture_logs = logs.logsumexp(0)-math.log(len(logs))
                mix_stats = cmp.sufficient(mixture_logs, truth, seen)
                mix = cmp.diagnostics(mix_stats, sm, cfg['competition_temperature'], cfg['risk_pair_chunk'])
                mix['AUC_native'] = native_curve(mixture_logs.cpu().numpy(), truth.cpu().numpy(), seen.cpu().numpy())['AUC']
                exact = exact_curve(mixture_logs.cpu().numpy(), truth.cpu().numpy(), seen.cpu().numpy())['AUC']
                if abs(exact-mix['AUC_exact']) > 1e-9:
                    raise ArithmeticError('Competition hard-area identity differs from exact evaluator')
                results[name] = dict(single={key: float(np.mean([s[key] for s in single])) for key in single[0]},
                    mixture=mix, single_statistics=single_stats.cpu(), mixture_logprobs=mixture_logs.cpu(), masks=masks.cpu())
            saved = dict(results=results, truth=truth.cpu(), seen=seen.cpu(), policies=policies, graph_seed=graph_seed)
            save_torch(file, saved)
        baseline = saved['results']['uniform']
        for name, value in saved['results'].items():
            row = dict(policy=name, bank=graph_seed)
            for readout in ('single', 'mixture'):
                for metric, val in value[readout].items():
                    row[readout+'_'+metric] = val
                for target in cmp.TARGETS:
                    row[readout+'_'+target+'_gain'] = baseline[readout][target]-value[readout][target]
                row[readout+'_AUC_gain'] = value[readout]['AUC_exact']-baseline[readout]['AUC_exact']
            rows.append(row)
        print('Q5_DECISION_BANK', args.seed, graph_seed, flush=True)
    summary = {}
    for name in policies:
        selected = [r for r in rows if r['policy'] == name]
        summary[name] = {}
        for key in selected[0]:
            if key in ('policy', 'bank'): continue
            values = np.array([r[key] for r in selected])
            summary[name][key] = dict(mean=float(values.mean()), sd=float(values.std(ddof=1)),
                                     mcse=float(values.std(ddof=1)/np.sqrt(len(values))))
    write_json(work / 'decision/audit.json', dict(rows=rows, summary=summary, graphs=cfg['audit_graphs'],
        images={g: len(split['evaluate'][g]) for g in ('seen', 'unseen')},
        policy_strength=1., uncertainty='Graph-bank MCSE conditional on teacher and risk images',
        predictor_fit_folds=[1, 2, 3], selection_fold=4, evaluation_fold=5))


def scan(args):
    work, cfg, manifest, text, bank = align.context(args)
    types = bank['types'].numpy(); predictions = load_torch(work / 'utility/predictions.pt')
    model = align.restore(text, bank, cfg, args.device, work / 'reference/last.pt')
    rows, scanned, selected = [], {}, {}
    for target in cmp.TARGETS:
        mu = predictions[target]['reference']
        for strength in sorted(cfg['policy_strengths']):
            p, info = align.policy(mu, types, strength)
            seeds = [None]+(cfg['permutation_seeds'] if target == 'competition' else [])
            for perm_seed in seeds:
                pi = p if perm_seed is None else p[align.permutation(types, perm_seed)]
                key = '%s_k%g_%s' % (target, strength, 'real' if perm_seed is None else perm_seed)
                cachekey = 'uniform' if strength == 0 else key
                file = work / 'scan' / (cachekey+'.json')
                if file.exists():
                    metrics = read_json(file)
                else:
                    metrics = engine.evaluate_model(model, args.cache, cfg, manifest, pi.float(), 'val')
                    write_json(file, metrics)
                rows.append(dict(target=target, key=key, strength=float(strength), permutation_seed=perm_seed, val=metrics))
                scanned[key] = dict(pi=pi, type_scale=info)
                print('FROZEN_VAL', args.seed, key, metrics['cw']['AUC_native'], flush=True)
        choices = [r for r in rows if r['target'] == target and r['permutation_seed'] is None]
        best = max(choices, key=lambda r: (r['val']['cw']['AUC_native'], -r['strength']))
        selected[target] = dict(key=best['key'], strength=best['strength'], val_auc=best['val']['cw']['AUC_native'])
    formal = {'uniform': torch.full((len(types),), .5, dtype=torch.double)}
    formal.update({t: scanned[selected[t]['key']]['pi'] for t in cmp.TARGETS})
    formal['permuted'] = formal['competition'][align.permutation(types, cfg['permutation_seeds'][0])]
    permutation_optima = {}
    for ps in cfg['permutation_seeds']:
        best = max([r for r in rows if r['permutation_seed'] == ps],
                   key=lambda r: (r['val']['cw']['AUC_native'], -r['strength']))
        permutation_optima[str(ps)] = dict(key=best['key'], strength=best['strength'], val_auc=best['val']['cw']['AUC_native'])
    write_json(work / 'scan/results.json', rows)
    write_json(work / 'selection.json', dict(targets=selected, permutation_seed=cfg['permutation_seeds'][0],
        permutation_optima=permutation_optima, rule='epoch0 VAL CW native AUC; ties smaller strength',
        permuted='same selected Competition probability multiset; type-wise permutation'))
    save_torch(work / 'policies.pt', dict(formal=formal, scanned=scanned, types=types))
