"""TRUST-Align: output-scale calibration, matched permutations, and adaptation."""
import copy
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .common import read_json, write_json, load_torch, save_torch, sha
from . import compact, engine
from .data import load_cache
from .evaluation import evaluate_model

VARIANTS = ('uniform', 'align', 'permuted')


def policy(mu, types, strength):
    """Unique equality-budget KL solution in float64; return CPU probabilities."""
    mu = np.asarray(mu, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    if mu.shape != types.shape or not np.isfinite(mu).all() or strength < 0:
        raise ValueError('Invalid utility vector or strength')
    prob = np.full(mu.shape, .5, dtype=np.float64)
    info = {}
    for t in sorted(np.unique(types)):
        ix = types == t
        mean, std = float(mu[ix].mean()), float(mu[ix].std(ddof=0))
        r = (mu[ix] - mean) / std if std > 1e-12 else np.zeros(ix.sum())
        z = strength * r
        gamma = 0.
        if strength > 0 and std > 1e-12:
            lo, hi = float(z.min() - 40), float(z.max() + 40)
            for _ in range(200):
                gamma = (lo + hi) / 2
                p = 1 / (1 + np.exp(np.clip(gamma - z, -700, 700)))
                residual = p.sum() - ix.sum() / 2
                if abs(residual) <= 1e-12:
                    break
                if residual > 0:
                    lo = gamma
                else:
                    hi = gamma
            prob[ix] = p
        error = abs(prob[ix].sum() - ix.sum() / 2)
        if error > 1e-10:
            raise ArithmeticError('Equality budget did not converge')
        info[str(t)] = dict(mu_mean=mean, mu_std=std, gamma=gamma,
                            budget=float(ix.sum() / 2), budget_error=float(error))
    return torch.from_numpy(prob), info


def permutation(types, seed):
    types = np.asarray(types)
    rng = np.random.RandomState(seed)
    indices = np.arange(len(types))
    for t in sorted(np.unique(types)):
        ix = np.flatnonzero(types == t)
        indices[ix] = rng.permutation(ix)
    return indices


def local_policy(prob, radius=.1):
    prob = torch.as_tensor(prob, dtype=torch.float64)
    distance = float((prob - .5).abs().max())
    eta = min(1., radius / distance) if distance else 1.
    return .5 + eta * (prob - .5), eta


def context(args):
    work = Path(args.work)
    cfg = read_json(work / 'recipe.json')
    manifest, text = load_cache(args.cache)
    bank = load_torch(work / 'inputs/bank.pt')
    return work, cfg, manifest, text, bank


def restore(text, bank, cfg, device, checkpoint):
    model = engine.new_model(text, bank, cfg, device)
    model.load_state_dict(load_torch(checkpoint)['model'])
    return model.eval()


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def frozen_scan(args):
    work, cfg, manifest, text, bank = context(args)
    types = bank['types'].numpy()
    predictor = load_torch(work / 'utility_all/compact/model.pt')
    response = load_torch(work / 'responses/reference.pt')
    mu = compact.predict(predictor, response['features'].numpy(), types)
    model = restore(text, bank, cfg, args.device, work / 'reference/last.pt')
    rows, policies = [], {}
    for strength in sorted(cfg['policy_strengths']):
        prob, info = policy(mu, types, strength)
        for permseed in [None] + cfg['permutation_seeds']:
            key = 'k%g_%s' % (strength, 'real' if permseed is None else permseed)
            indices = np.arange(len(types)) if permseed is None else permutation(types, permseed)
            pi = prob[indices]
            # All zero-strength policies are exactly the same computation.
            cachekey = 'k0_real' if strength == 0 else key
            file = work / 'scan' / (cachekey + '.json')
            if file.exists():
                value = read_json(file)
            else:
                synchronize(args.device); tic = time.perf_counter()
                metrics = evaluate_model(model, args.cache, cfg, manifest, pi.float(), 'val')
                synchronize(args.device)
                value = dict(val=metrics, seconds=time.perf_counter() - tic)
                write_json(file, value)
            rows.append(dict(key=key, strength=float(strength), permutation_seed=permseed,
                             reused_from=cachekey if key != cachekey else None, **value))
            policies[key] = dict(pi=pi, types=types, permutation=indices, type_scale=info)
            print('FROZEN_VAL', args.seed, key, value['val']['cw']['AUC_native'], flush=True)
    real = [r for r in rows if r['permutation_seed'] is None]
    selected = max(real, key=lambda r: (r['val']['cw']['AUC_native'], -r['strength']))
    chosen = policies[selected['key']]['pi']
    permseed = cfg['permutation_seeds'][0]
    formal = dict(uniform=torch.full_like(chosen, .5), align=chosen,
                  permuted=chosen[permutation(types, permseed)])
    permutation_optima = {}
    for ps in cfg['permutation_seeds']:
        items = [r for r in rows if r['permutation_seed'] == ps]
        permutation_optima[str(ps)] = max(items, key=lambda r: (r['val']['cw']['AUC_native'], -r['strength']))['key']
    write_json(work / 'scan/results.json', rows)
    write_json(work / 'selection.json', dict(strength=selected['strength'], key=selected['key'],
        val_auc=selected['val']['cw']['AUC_native'], permutation_seed=permseed,
        permutation_optima=permutation_optima, alpha=float(predictor['alpha']),
        rule='epoch0 VAL CW native AUC; ties choose smaller strength', graph_seed=cfg['graph_seed']))
    save_torch(work / 'policies.pt', dict(mu=mu, types=types, scanned=policies, formal=formal))


@torch.no_grad()
def deployment_stats(model, pi, bank, cfg):
    p = pi.double().cpu().numpy(); types = bank['types'].numpy(); targets = bank['target'].numpy()
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(cfg['graph_seed'])
    masks = torch.stack([(torch.rand(pi.shape, generator=gen, device=device) < pi.to(device)).float()
                         for _ in range(cfg['graphs'])])
    ps = np.clip(p, 1e-15, 1 - 1e-15)
    kl = ps * np.log(2 * ps) + (1 - ps) * np.log(2 * (1 - ps))
    result = {}
    anchor = model.graph.anchors
    displacements = torch.stack([(model.graph(m) - anchor).norm(dim=-1).cpu() for m in masks])
    for t in sorted(np.unique(types)):
        ix = types == t
        nodes = sorted(set(targets[ix].tolist()))
        coverage = [1 - float(np.prod(1 - p[ix & (targets == n)])) for n in nodes]
        realized = [[int((m[torch.as_tensor(ix & (targets == n), device=device)] > 0).any())
                     for n in nodes] for m in masks]
        result[str(t)] = dict(expected_edges=float(p[ix].sum()), probability_std=float(p[ix].std()),
            probability_min=float(p[ix].min()), probability_max=float(p[ix].max()), kl=float(kl[ix].sum()),
            sampled_edges=masks[:, torch.as_tensor(ix, device=device)].sum(1).cpu().tolist(),
            target_nodes=nodes, expected_node_coverage=coverage, sampled_node_coverage=realized,
            prototype_displacement_mean=float(displacements[:, nodes].mean()))
    return result


@torch.no_grad()
def inference_latency(args, cfg, model, manifest, pi):
    """Warm single-batch forward timing, including base and both candidate worlds."""
    device = next(model.parameters()).device
    data = next(iter(engine.batches(args.cache, 'val', cfg)))
    inputs = [v.to(device) for v in data[:3]]
    specs = [engine.candidate_spec(manifest, 'val', w)[1].to(device) for w in ('cw', 'ow')]
    gen = torch.Generator(device=device).manual_seed(cfg['graph_seed'])
    masks = [(torch.rand(pi.shape, generator=gen, device=device) < pi.to(device)).float() for _ in range(cfg['graphs'])]
    from .model import candidate_logprob
    timings = []
    for repeat in range(4):
        synchronize(device); tic = time.perf_counter()
        base = model.base(*inputs)
        joints = [model.score(base, m)[0] for m in masks]
        for ids in specs:
            torch.stack([candidate_logprob(j, ids) for j in joints]).logsumexp(0) - math.log(len(masks))
        synchronize(device)
        if repeat:
            timings.append(time.perf_counter() - tic)
    return dict(batch_images=len(inputs[0]), graphs=len(masks), repeats=3, seconds=timings,
        milliseconds_per_image=1000 * float(np.mean(timings)) / len(inputs[0]),
        scope='warm GPU/CPU forward, both CW/OW; excludes disk, transfer, calibration and curve integration')


def adapt_and_evaluate(args, variant):
    work, cfg, manifest, text, bank = context(args)
    cfg = dict(cfg, variant=variant)
    policy_data = load_torch(work / 'policies.pt')
    pi = policy_data['formal'][variant].float()
    output = work / 'runs' / variant
    if variant != 'uniform' and torch.equal(pi, policy_data['formal']['uniform'].float()):
        shutil.copytree(work / 'runs/uniform', output, dirs_exist_ok=True)
        write_json(output / 'reuse.json', dict(from_variant='uniform', reason='identical probabilities and training RNG',
                                              requested_variant=variant))
        return
    # An interrupted first epoch has no optimizer/RNG checkpoint yet. Restart
    # that epoch and epoch-0 selection together instead of using an orphan best.
    if (output / 'best.pt').exists() and not (output / 'last.pt').exists():
        (output / 'best.pt').unlink()
    engine.train_model(args, cfg, text, bank, manifest, manifest['pairs']['train'], output,
                       pi=pi, initial=work / 'reference/last.pt', select=True)
    chosen = load_torch(output / 'best.pt')
    original = load_torch(work / 'reference/last.pt')['model']
    if not all(torch.equal(v, chosen['model'][k]) for k, v in original.items() if k.startswith('backbone.')):
        raise ValueError('Frozen CPF backbone changed')
    results = {}
    for snapshot, checkpoint in [('epoch0', work / 'reference/last.pt'), ('best', output / 'best.pt')]:
        model = restore(text, bank, cfg, args.device, checkpoint)
        results[snapshot] = {}
        for graph_seed in cfg['eval_graph_seeds']:
            evcfg = dict(cfg, graph_seed=graph_seed)
            dest = output / snapshot / ('bank%d' % graph_seed)
            file = dest / 'metrics.json'
            if file.exists():
                entry = read_json(file)
            else:
                synchronize(args.device); tic = time.perf_counter()
                val = evaluate_model(model, args.cache, evcfg, manifest, pi, 'val', save=dest)
                test = evaluate_model(model, args.cache, evcfg, manifest, pi, 'test', calibration=val, save=dest)
                synchronize(args.device)
                entry = dict(val=val, test=test, epoch=0 if snapshot == 'epoch0' else chosen['epoch'],
                    graph_seed=graph_seed, graphs=cfg['graphs'], seconds=time.perf_counter() - tic,
                    checkpoint_sha256=sha(checkpoint), deployment=deployment_stats(model, pi, bank, evcfg))
                write_json(file, entry)
            results[snapshot][str(graph_seed)] = entry
            print('EVALUATED', args.seed, variant, snapshot, graph_seed,
                  'TEST_CW_AUC', entry['test']['cw']['AUC_native'], flush=True)
    latency = inference_latency(args, cfg, model, manifest, pi)
    write_json(output / 'metrics.json', dict(variant=variant, epoch=chosen['epoch'], results=results,
        initial_sha256=sha(work / 'reference/last.pt'), backbone_identical=True, inference_latency=latency))
