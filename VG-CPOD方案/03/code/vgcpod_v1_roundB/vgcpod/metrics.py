"""Native finite CPF scan plus complete critical-bias curve; all metrics in percent."""
import numpy as np
import torch


def hm(s, u):
    s, u = np.asarray(s), np.asarray(u)
    return np.divide(2 * s * u, s + u, out=np.zeros_like(s + u, dtype=float), where=(s + u) != 0)


def native_curve(scores, truth, seen_classes):
    # Same top-1 CPF scan: correct unseen differences -1e-4, ~20 bins, endpoint bias=1000.
    x = torch.as_tensor(scores, dtype=torch.float32)
    y = torch.as_tensor(truth, dtype=torch.long)
    seen = torch.as_tensor(seen_classes, dtype=torch.bool)
    is_seen = seen[y]
    if not is_seen.any() or is_seen.all():
        raise ValueError('AUC requires seen and unseen evaluation images')
    endpoint = x.clone()
    endpoint[:, ~seen] += 1000.0
    pred = endpoint.topk(1, dim=1).indices[:, 0]
    correct = pred.eq(y)
    un = ~is_seen
    delta = x[un][:, seen].topk(1, dim=1).values[:, 0] - x[un, y[un]]
    diffs = torch.sort(delta[correct[un]] - 1e-4).values
    step = max(len(diffs) // 20, 1)
    biases = [float(b) for b in diffs[::step]] + [1000.0]
    ss, uu = [], []
    for bias in biases:
        z = x.clone()
        z[:, ~seen] += bias
        correct = z.topk(1, dim=1).indices[:, 0].eq(y)
        ss.append(float(correct[is_seen].float().mean()))
        uu.append(float(correct[un].float().mean()))
    h = hm(ss, uu)
    best = int(np.argmax(h))
    trapz = getattr(np, 'trapezoid', np.trapz)
    return {'AUC': float(trapz(ss, uu) * 100), 'best_hm': float(h[best] * 100),
            'best_bias': biases[best], 'best_seen': max(ss) * 100, 'best_unseen': max(uu) * 100,
            'biases': biases, 'seen': (100 * np.array(ss)).tolist(), 'unseen': (100 * np.array(uu)).tolist()}


def exact_curve(scores, truth, seen_classes):
    x = np.asarray(scores, dtype=np.float64)
    seen = np.asarray(seen_classes, dtype=bool)
    y = np.asarray(truth)
    si, ui = np.flatnonzero(seen), np.flatnonzero(~seen)
    if not len(si) or not len(ui):
        raise ValueError('Curve needs both candidate partitions')
    # A single fixed winner within each partition; events switch seen -> unseen.
    sp = si[x[:, si].argmax(1)]
    up = ui[x[:, ui].argmax(1)]
    d = x[np.arange(len(x)), sp] - x[np.arange(len(x)), up]
    sm, um = seen[y], ~seen[y]
    ns, nu = sm.sum(), um.sum()
    if ns == 0 or nu == 0:
        raise ValueError('Curve needs both sample partitions')
    cs, cu = (sp == y) & sm, (up == y) & um
    order = np.argsort(d, kind='stable')
    thresholds, starts = np.unique(d[order], return_index=True)
    loss_s = np.add.reduceat(cs[order].astype(int), starts)
    gain_u = np.add.reduceat(cu[order].astype(int), starts)
    ss = np.r_[cs.sum(), cs.sum() - np.cumsum(loss_s)] / ns
    uu = np.r_[0, np.cumsum(gain_u)] / nu
    h = hm(ss, uu)
    # Choose interval interiors, so no prediction depends on equality at a threshold.
    low = thresholds[0] - max(1.0, abs(thresholds[0]) * 1e-6)
    high = thresholds[-1] + max(1.0, abs(thresholds[-1]) * 1e-6)
    midpoints = thresholds[:-1] + (thresholds[1:] - thresholds[:-1]) * 0.5
    biases = np.r_[low, midpoints, high]
    best = int(np.argmax(h))
    trapz = getattr(np, 'trapezoid', np.trapz)
    return {'AUC': float(trapz(ss, uu) * 100), 'best_hm': float(h[best] * 100),
            'best_bias': float(biases[best]), 'thresholds': thresholds.tolist(),
            'biases': biases.tolist(), 'seen': (ss * 100).tolist(), 'unseen': (uu * 100).tolist()}


def evaluate_scores(joint, cond, attrs, objs, labels, pairs, phase, world, fixed_bias=None):
    m, n = len(labels['attrs']), len(labels['objs'])
    train = {tuple(p) for p in pairs['train']}
    allowed = train | {tuple(p) for p in pairs[phase]} if world == 'cw' else {
        (a, o) for a in labels['attrs'] for o in labels['objs']}
    # Preserve the evaluator's alphabetical attribute/object pair ordering for topk ties.
    candidates = sorted(allowed)
    ai = np.array([labels['attrs'].index(a) for a, o in candidates])
    oi = np.array([labels['objs'].index(o) for a, o in candidates])
    scores = joint[:, oi, ai].astype(np.float32)
    index = {p: i for i, p in enumerate(candidates)}
    truth = np.array([index[(labels['attrs'][a], labels['objs'][o])] for a, o in zip(attrs, objs)])
    seen = np.array([p in train for p in candidates])
    native = native_curve(scores, truth, seen)
    exact = exact_curve(scores, truth, seen)
    max_scores = scores.max(1, keepdims=True)
    log_probs = scores - (max_scores + np.log(np.exp(scores - max_scores).sum(1, keepdims=True)))
    pred = torch.from_numpy(scores).topk(1, dim=1).indices[:, 0].numpy()
    correct = pred == truth
    attr_correct, obj_correct = ai[pred] == attrs, oi[pred] == objs
    conditional = cond[np.arange(len(attrs)), objs]
    z = conditional.max(1, keepdims=True)
    cond_log = conditional - z - np.log(np.exp(conditional - z).sum(1, keepdims=True))
    present = np.unique(truth)
    metrics = {'images': len(truth), 'candidates': len(candidates),
               'AUC_native': native['AUC'], 'AUC_exact': exact['AUC'],
               'HM_oracle_native': native['best_hm'], 'HM_oracle_exact': exact['best_hm'],
               'bias_native': native['best_bias'], 'bias_exact': exact['best_bias'],
               'pair_acc_unbiased': float(correct.mean() * 100),
               'pair_macro_unbiased': float(np.mean([correct[truth == k].mean() for k in present]) * 100),
               'pair_nll_unbiased': float(-log_probs[np.arange(len(truth)), truth].mean()),
               'gt_object_attr_acc': float((conditional.argmax(1) == attrs).mean() * 100),
               'gt_object_attr_nll': float(-cond_log[np.arange(len(attrs)), attrs].mean()),
               'errors_unbiased': {'both_correct': int((attr_correct & obj_correct).sum()),
                                    'attribute_wrong_object_correct': int((~attr_correct & obj_correct).sum()),
                                    'attribute_correct_object_wrong': int((attr_correct & ~obj_correct).sum()),
                                    'both_wrong': int((~attr_correct & ~obj_correct).sum())}}
    if fixed_bias is not None:
        biased = torch.from_numpy(scores.copy())
        biased[:, ~torch.from_numpy(seen)] += fixed_bias
        ok = biased.topk(1, dim=1).indices[:, 0].numpy() == truth
        s, u = ok[seen[truth]].mean(), ok[~seen[truth]].mean()
        metrics['fixed_val_bias'] = float(fixed_bias)
        metrics['seen_at_val_bias'] = float(s * 100)
        metrics['unseen_at_val_bias'] = float(u * 100)
        metrics['HM_at_val_bias'] = float(hm(s, u) * 100)
    return metrics, {'native': native, 'exact': exact}, np.array(candidates), log_probs
