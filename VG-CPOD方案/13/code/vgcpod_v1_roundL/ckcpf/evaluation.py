import numpy as np
import torch
from .data import loader, train_mask
from .metrics import evaluate_scores, native_curve, hm


@torch.no_grad()
def predict(model, cache, phase, cfg, seed, device):
    model.eval()
    result = {k: [] for k in ('joint', 'cond', 'obj', 'attrs', 'objs', 'delta_abs')}
    for g, a, o in loader(cache, phase, cfg, seed):
        out = model(g.to(device))
        for k in ('joint', 'cond', 'obj'):
            result[k].append(out[k].cpu().numpy())
        result['attrs'].append(a.numpy()); result['objs'].append(o.numpy())
        result['delta_abs'].append(out['delta'].abs().mean((1, 2)).cpu().numpy())
    return {k: np.concatenate(v) for k, v in result.items()}


def candidate_scores(pred, manifest, phase, world):
    labels, pairs = manifest['labels'], manifest['pairs']
    train = set(map(tuple, pairs['train']))
    candidates = sorted(train | set(map(tuple, pairs[phase]))) if world == 'cw' else sorted((a, o) for a in labels['attrs'] for o in labels['objs'])
    ai = np.array([labels['attrs'].index(a) for a, o in candidates])
    oi = np.array([labels['objs'].index(o) for a, o in candidates])
    index = {p: i for i, p in enumerate(candidates)}
    truth = np.array([index[(labels['attrs'][a], labels['objs'][o])] for a, o in zip(pred['attrs'], pred['objs'])])
    return pred['joint'][:, oi, ai], truth, np.array([p in train for p in candidates])


def selection_auc(pred, manifest):
    return native_curve(*candidate_scores(pred, manifest, 'val', 'cw'))['AUC']


def calibrated(scores, truth, seen, bias, temperature):
    z = torch.as_tensor(scores, dtype=torch.float64).clone()
    z[:, ~seen] += bias
    lp = (z / temperature).log_softmax(-1)
    conf, pred = lp.exp().topk(1, dim=1)
    conf, pred = conf[:, 0], pred[:, 0]
    ok = pred == torch.as_tensor(truth)
    sm = torch.as_tensor(seen[truth])
    s, u = float(ok[sm].double().mean()), float(ok[~sm].double().mean())
    nll = -lp[torch.arange(len(truth)), truth]
    ece = 0.
    for b in range(15):
        mask = (conf >= b / 15) & (conf < (b + 1) / 15 if b < 14 else conf <= 1)
        if mask.any():
            ece += float(mask.double().mean() * (ok[mask].double().mean() - conf[mask].mean()).abs())
    return dict(bias=float(bias), temperature=float(temperature), seen=100*s, unseen=100*u,
                HM=float(hm(s, u)*100), NLL=float(nll.mean()), NLL_unseen=float(nll[~sm].mean()), ECE=100*ece)


def evaluate(pred, manifest, phase, calibration=None):
    results, curves = {}, {}
    a, o = pred['attrs'], pred['objs']
    sm = train_mask(manifest).numpy()[o, a]
    cond = pred['cond'][np.arange(len(a)), o]
    true = cond[np.arange(len(a)), a]
    wrong = cond.copy(); wrong[np.arange(len(a)), a] = -np.inf
    margin = true - wrong.max(-1)
    attr_ok = cond.argmax(-1) == a
    for world in ('cw', 'ow'):
        m, curve, _, _ = evaluate_scores(pred['joint'], pred['cond'], a, o,
                                         manifest['labels'], manifest['pairs'], phase, world)
        scores, truth, seen = candidate_scores(pred, manifest, phase, world)
        if calibration is None:
            bias = m['bias_exact']
            temp = min(np.exp(np.linspace(np.log(.1), np.log(10.), 41)),
                       key=lambda t: calibrated(scores, truth, seen, bias, t)['NLL'])
        else:
            bias, temp = calibration[world]['calibrated']['bias'], calibration[world]['calibrated']['temperature']
        m['at_val_bias'] = calibrated(scores, truth, seen, bias, 1.)
        m['calibrated'] = calibrated(scores, truth, seen, bias, temp)
        m['object_marginal_accuracy'] = float((pred['obj'].argmax(-1) == o).mean()*100)
        err = m['errors_unbiased']
        m['joint_object_accuracy'] = (err['both_correct'] + err['attribute_wrong_object_correct']) / len(a)*100
        m['delta_mean_abs'] = float(pred['delta_abs'].mean())
        for label, mask in [('seen', sm), ('unseen', ~sm)]:
            m['gt_object_attr_acc_' + label] = float(attr_ok[mask].mean()*100)
            m['gt_object_margin_' + label] = float(margin[mask].mean())
            m['gt_object_positive_margin_' + label] = float((margin[mask] > 0).mean()*100)
        results[world], curves[world] = m, curve
    return results, curves
