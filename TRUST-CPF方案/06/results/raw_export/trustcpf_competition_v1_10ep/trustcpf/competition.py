"""Paired relation utility for unseen NLL, balanced NLL and competition area."""
import numpy as np
import torch

TARGETS = ('unll', 'balanced', 'competition')
VARIANTS = ('uniform',) + TARGETS + ('permuted',)


def split_risk(rows, labels, train_pairs, query, fold, cfg):
    """Hold out Inner images before either teacher training phase."""
    q = set(map(tuple, query))
    inner = [p for p in train_pairs if tuple(p) not in q]
    rng = np.random.RandomState(cfg['risk_split_seed'] + fold)
    train, seen, unseen, singletons = [], [], [], []
    for pair in sorted(map(tuple, train_pairs)):
        a, o = labels['attrs'].index(pair[0]), labels['objs'].index(pair[1])
        ids = [i for i, r in enumerate(rows) if r['attr'] == a and r['object'] == o]
        if not ids:
            raise ValueError('Composition has no image: ' + str(pair))
        if pair in q:
            unseen.extend(ids)
        elif len(ids) == 1:
            train.extend(ids); singletons.append(pair)
        else:
            ids = rng.permutation(ids).tolist()
            n = min(len(ids)-1, max(1, int(np.ceil(len(ids)*cfg['risk_holdout_fraction']))))
            seen.extend(ids[:n]); train.extend(ids[n:])
    measure, evaluate = {}, {}
    rng = np.random.RandomState(cfg['risk_sample_seed'] + fold)
    for name, pool in [('seen', seen), ('unseen', unseen)]:
        ids = rng.permutation(sorted(pool)).tolist()
        if fold == 5:
            mid = len(ids)//2
            measure[name] = ids[:mid][:cfg['risk_images_per_group']]
            evaluate[name] = ids[mid:][:cfg['risk_images_per_group']]
        else:
            measure[name] = ids[:cfg['risk_images_per_group']]
        if not measure[name] or (fold == 5 and not evaluate[name]):
            raise ValueError('Risk group has too few independent images: ' + name)
    return dict(fold=fold, inner_pairs=inner, query_pairs=query, train_ids=sorted(train),
                seen_pool=sorted(seen), unseen_pool=sorted(unseen), singleton_inner=singletons,
                measure=measure, evaluate=evaluate, weighting='equal images within each group',
                candidate_pairs=train_pairs, split_seed=cfg['risk_split_seed']+fold,
                sample_seed=cfg['risk_sample_seed']+fold)


def sufficient(logs, truth, seen):
    """Per-image [log P(true), within-group margin, crossing threshold, hard gate].

    These four numbers reproduce every supervision risk without storing C scores
    for every intervention. Hard gates use the same first-index tie rule as exact_curve.
    """
    z = logs.double().log_softmax(-1)
    y = truth.to(z.device); seen = seen.to(z.device)
    sm = seen[y]; ar = torch.arange(len(z), device=z.device)
    same = seen[None, :] == sm[:, None]
    own = z.masked_fill(~same, -torch.inf)
    gate = own.argmax(-1) == y
    other_max = z.masked_fill(same, -torch.inf).max(-1).values
    competitors = own.clone(); competitors[ar, y] = -torch.inf
    true = z[ar, y]
    margin = true - competitors.max(-1).values
    crossing = torch.where(sm, true-other_max, other_max-true)
    return torch.stack((true, margin, crossing, gate.double()), -1)


def area(stats, sample_seen, temperature=1., chunk=128, hard=False):
    """Exact blockwise pair sum, with no sampling of seen/unseen image pairs."""
    s, u = stats[sample_seen], stats[~sample_seen]
    if not len(s) or not len(u) or temperature <= 0:
        raise ValueError('Both risk groups and positive temperature are required')
    gs = s[:, 3] if hard else torch.sigmoid(s[:, 1]/temperature)
    gu = u[:, 3] if hard else torch.sigmoid(u[:, 1]/temperature)
    total = stats.new_zeros(())
    for start in range(0, len(s), chunk):
        d = s[start:start+chunk, 2, None] - u[None, :, 2]
        cross = (d > 0).double() + .5*(d == 0).double() if hard else torch.sigmoid(d/temperature)
        total += (gs[start:start+chunk, None]*gu[None, :]*cross).sum()
    return total/(len(s)*len(u))


def risks(stats, sample_seen, temperature=1., chunk=128):
    ns = -stats[sample_seen, 0].mean(); nu = -stats[~sample_seen, 0].mean()
    return torch.stack((nu, (ns+nu)/2, 1-area(stats, sample_seen, temperature, chunk)))


def diagnostics(stats, sample_seen, temperature=1., chunk=128):
    r = risks(stats, sample_seen, temperature, chunk)
    finite = torch.isfinite(stats[:, 1])
    return dict(zip(TARGETS, r.cpu().tolist()),
                smooth_area=float(1-r[2]), AUC_exact=float(100*area(stats, sample_seen, temperature, chunk, True)),
                seen_nll=float(-stats[sample_seen, 0].mean()),
                seen_gate=float(torch.sigmoid(stats[sample_seen, 1]/temperature).mean()),
                unseen_gate=float(torch.sigmoid(stats[~sample_seen, 1]/temperature).mean()),
                within_margin_abs_mean=float(stats[finite, 1].abs().mean()) if finite.any() else 0.,
                crossing_sd=float(stats[:, 2].std(unbiased=False)))


def centered_type_error(pred, target, types, scales):
    values = []
    for t in range(3):
        r = (pred[types == t]-target[types == t])/scales[t]
        values.append(float(np.mean((r-r.mean())**2)))
    return float(np.mean(values))
