"""Independent-image Q5 decision audit for single-graph and mixture utility."""
import math
from pathlib import Path
import numpy as np
import torch
from .common import read_json, write_json, load_torch, save_torch, sha
from . import compact, engine
from .align import context, restore, policy, permutation, local_policy
from .folds import pair_ids
from .model import candidate_logprob


def split_query(rows, labels, query, seed=8101, limit=8):
    """Same retained pairs on both sides, equal pair then equal image weights."""
    rng = np.random.RandomState(seed)
    result = dict(measure=[], evaluate=[], excluded=[], pairs=[])
    for pair in sorted(map(tuple, query)):
        a, o = labels['attrs'].index(pair[0]), labels['objs'].index(pair[1])
        ids = [i for i, r in enumerate(rows) if r['attr'] == a and r['object'] == o]
        if len(ids) < 2:
            result['excluded'].append(dict(pair=pair, count=len(ids)))
            continue
        ids = rng.permutation(ids).tolist(); mid = len(ids) // 2
        for name, chosen in [('measure', ids[:mid][:limit]), ('evaluate', ids[mid:][:limit])]:
            result[name].extend(dict(row_id=i, image=rows[i]['image'], pair=list(pair),
                                     weight=1 / len(chosen)) for i in chosen)
        result['pairs'].append(pair)
    if not result['pairs']:
        raise ValueError('No Q5 pair with two images')
    for name in ('measure', 'evaluate'):
        for r in result[name]:
            r['weight'] /= len(result['pairs'])
    assert not set(r['row_id'] for r in result['measure']) & set(r['row_id'] for r in result['evaluate'])
    result.update(seed=seed, limit_per_pair=limit, weighting='equal pairs; equal images within each pair')
    return result


@torch.no_grad()
def prepare_base(args, cfg, model, records, manifest):
    device = next(model.parameters()).device
    parts = {}; attrs = []; objs = []
    for batch in engine.batches(args.cache, 'train', cfg, [r['row_id'] for r in records]):
        values = model.base(*[x.to(device) for x in batch[:3]])
        for key, value in values.items():
            parts.setdefault(key, []).append(value)
        attrs.append(batch[3]); objs.append(batch[4])
    base = {k: torch.cat(v) for k, v in parts.items()}
    candidates = torch.tensor(pair_ids(manifest['labels'], manifest['pairs']['train']), device=device)
    targets = torch.cat(objs) * len(manifest['labels']['attrs']) + torch.cat(attrs)
    index = {int(c): i for i, c in enumerate(candidates.cpu())}
    truth = torch.tensor([index[int(t)] for t in targets], device=device)
    weights = torch.tensor([r['weight'] for r in records], device=device, dtype=torch.float64)
    return base, candidates, truth, weights


@torch.no_grad()
def true_logs(model, base, masks, candidates, truth, batch_size):
    """Score bounded image batches; one frozen base encoding serves all graphs."""
    logs = []
    for mask in masks:
        chunks = []
        for start in range(0, len(truth), batch_size):
            end = start + batch_size
            b = {k: v[start:end] for k, v in base.items()}
            joint = model.score(b, mask)[0]
            lp = candidate_logprob(joint.double(), candidates)
            chunks.append(lp.gather(1, truth[start:end, None])[:, 0])
        logs.append(torch.cat(chunks))
    return torch.stack(logs)


def block_effect(log_off, log_on, log_companion_sum, weights):
    single = ((log_on - log_off) * weights).sum(-1)
    mixture = ((torch.logaddexp(log_companion_sum, log_on)
                - torch.logaddexp(log_companion_sum, log_off)) * weights).sum(-1)
    return single, mixture


def summarize_effect(single_blocks, mixture_blocks, graphs):
    single = np.asarray(single_blocks, dtype=np.float64)
    mixture = np.asarray(mixture_blocks, dtype=np.float64)
    k = len(single)
    if k < 2:
        raise ValueError('At least two independent measurement blocks required')
    return dict(single=single.mean(0), mixture=graphs * mixture.mean(0),
                single_se2=single.var(0, ddof=1) / k,
                mixture_se2=graphs ** 2 * mixture.var(0, ddof=1) / k,
                single_blocks=single, mixture_slot_blocks=mixture)


@torch.no_grad()
def measure(args, cfg, model, prepared, types, work):
    base, candidates, truth, weights = prepared
    device = next(model.parameters()).device
    k, m, edges = cfg['audit_blocks'], cfg['audit_graphs'], len(types)
    singles, mixtures = [], []
    for block in range(k):
        file = work / 'decision/blocks' / ('measure_%02d.pt' % block)
        if file.exists():
            value = load_torch(file)
        else:
            seed = cfg['audit_measure_seed'] + block
            gen = torch.Generator(device='cpu').manual_seed(seed)
            masks = (torch.rand((m, edges), generator=gen) < .5).float().to(device)
            companion = true_logs(model, base, masks[1:], candidates, truth, cfg['eval_batch_size'])
            logsum = companion.logsumexp(0)
            off = masks[0].repeat(edges, 1); on = off.clone()
            ii = torch.arange(edges, device=device); off[ii, ii] = 0; on[ii, ii] = 1
            p0 = true_logs(model, base, off, candidates, truth, cfg['eval_batch_size'])
            p1 = true_logs(model, base, on, candidates, truth, cfg['eval_batch_size'])
            d1, dm = block_effect(p0, p1, logsum, weights)
            value = dict(seed=seed, graphs=m, masks=masks.cpu(), log_off=p0.cpu(), log_on=p1.cpu(),
                         log_companion=companion.cpu(), weights=weights.cpu(), truth=truth.cpu(),
                         d_single=d1.cpu().numpy(), d_mixture_slot=dm.cpu().numpy())
            save_torch(file, value)
        singles.append(value['d_single']); mixtures.append(value['d_mixture_slot'])
        print('Q5_MEASURE_BLOCK', args.seed, block + 1, '/', k, flush=True)
    return summarize_effect(singles, mixtures, m)


def make_policies(predicted, effects, types, cfg):
    utilities = dict(predicted=np.asarray(predicted), single=effects['single'], mixture=effects['mixture'])
    policies = dict(uniform=torch.full((len(types),), .5, dtype=torch.float64))
    metadata = dict(uniform=dict(eta=1., source='uniform', permuted=False, local=False))
    indices = permutation(types, cfg['permutation_seeds'][0])
    for name, u in utilities.items():
        p, _ = policy(u, types, 1.)
        for permuted in (False, True):
            key = name + ('_permuted' if permuted else '')
            prob = p[indices] if permuted else p
            policies[key] = prob
            small, eta = local_policy(prob, cfg['audit_local_radius'])
            policies[key + '_local'] = small
            metadata[key] = dict(source=name, permuted=permuted, local=False, eta=1.)
            metadata[key + '_local'] = dict(source=name, permuted=permuted, local=True, eta=eta)
    for name, p in policies.items():
        metadata[name]['proxy'] = {u: float(np.dot(v, p.numpy() - .5)) for u, v in utilities.items()}
    return policies, metadata


@torch.no_grad()
def evaluate_policies(args, cfg, model, prepared, policies, work):
    base, candidates, truth, weights = prepared
    device = next(model.parameters()).device
    records = []
    for seed in cfg['audit_eval_seeds']:
        file = work / 'decision/blocks' / ('evaluate_%d.pt' % seed)
        if file.exists():
            value = load_torch(file)
        else:
            gen = torch.Generator(device='cpu').manual_seed(seed)
            uniforms = torch.rand((cfg['audit_graphs'], len(policies['uniform'])), generator=gen).double()
            values = {}; masks_all = {}; logs_all = {}
            for name, pi in policies.items():
                masks = (uniforms < pi[None, :]).float().to(device)
                lp = true_logs(model, base, masks, candidates, truth, cfg['eval_batch_size'])
                single = float(-(lp.mean(0) * weights).sum())
                mixture = float(-((lp.logsumexp(0) - math.log(len(lp))) * weights).sum())
                values[name] = dict(single_nll=single, mixture_nll=mixture)
                masks_all[name] = masks.cpu(); logs_all[name] = lp.cpu()
            value = dict(seed=seed, values=values, masks=masks_all, true_logprobs=logs_all,
                         weights=weights.cpu(), truth=truth.cpu())
            save_torch(file, value)
        baseline = value['values']['uniform']
        for name, risks in value['values'].items():
            records.append(dict(policy=name, bank=seed, **risks,
                single_gain=baseline['single_nll'] - risks['single_nll'],
                mixture_gain=baseline['mixture_nll'] - risks['mixture_nll']))
        print('Q5_EVALUATE_BANK', args.seed, seed, flush=True)
    return records


def run(args):
    work, cfg, manifest, text, bank = context(args)
    types = bank['types'].numpy()
    teacher = work / 'teacher_q5/last.pt'
    model = restore(text, bank, cfg, args.device, teacher)
    split = split_query(read_json(Path(args.cache) / 'train_records.json'), manifest['labels'],
                        read_json(work / 'folds.json')['query'][4], cfg['audit_split_seed'], cfg['audit_images_per_pair'])
    split.update(teacher_sha256=sha(teacher), candidate_pairs=manifest['pairs']['train'],
                 candidate_ids=pair_ids(manifest['labels'], manifest['pairs']['train']),
                 predictor_sha256=sha(work / 'utility/compact/model.pt'),
                 response_sha256=sha(work / 'responses/fold5.pt'))
    write_json(work / 'decision/sampling.json', split)
    measured_base = prepare_base(args, cfg, model, split['measure'], manifest)
    effects = measure(args, cfg, model, measured_base, types, work)
    del measured_base
    response = load_torch(work / 'responses/fold5.pt')
    pred = compact.predict(load_torch(work / 'utility/compact/model.pt'), response['features'].numpy(), types)
    policies, metadata = make_policies(pred, effects, types, cfg)
    eval_base = prepare_base(args, cfg, model, split['evaluate'], manifest)
    rows = evaluate_policies(args, cfg, model, eval_base, policies, work)
    summary = {}
    for name in policies:
        selected = [r for r in rows if r['policy'] == name]
        summary[name] = dict(metadata[name])
        for metric in ('single_nll', 'mixture_nll', 'single_gain', 'mixture_gain'):
            x = np.array([r[metric] for r in selected])
            summary[name][metric] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)),
                mcse=float(x.std(ddof=1) / math.sqrt(len(x))), banks=len(x))
    comparisons = []
    for name in ('predicted', 'single', 'mixture'):
        for suffix in ('', '_local'):
            real, perm = name + suffix, name + '_permuted' + suffix
            for seed in cfg['audit_eval_seeds']:
                a = next(r for r in rows if r['policy'] == real and r['bank'] == seed)
                b = next(r for r in rows if r['policy'] == perm and r['bank'] == seed)
                comparisons.append(dict(policy=real, bank=seed,
                    single_gain_vs_permuted=b['single_nll']-a['single_nll'],
                    mixture_gain_vs_permuted=b['mixture_nll']-a['mixture_nll']))
    correlations = {}
    for t in sorted(np.unique(types)):
        ix = types == t
        correlations[str(t)] = dict(predicted_single=compact.point_metrics(pred[ix], effects['single'][ix]),
            predicted_mixture=compact.point_metrics(pred[ix], effects['mixture'][ix]),
            single_mixture=compact.point_metrics(effects['single'][ix], effects['mixture'][ix]))
    save_torch(work / 'decision/effects.pt', dict(effects, predicted=pred, types=types, policies=policies))
    write_json(work / 'decision/audit.json', dict(rows=rows, policies=summary, correlations=correlations,
        paired_vs_permutation=comparisons, blocks=cfg['audit_blocks'], graphs=cfg['audit_graphs'],
        evaluation_banks=cfg['audit_eval_seeds'], retained_pairs=len(split['pairs']),
        measure_images=len(split['measure']), evaluate_images=len(split['evaluate']),
        uncertainty='MCSE over independent graph banks conditional on fixed teacher/images; not training-seed uncertainty'))
