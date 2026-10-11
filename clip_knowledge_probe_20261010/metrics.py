"""Same-object diagnostics, not CPF CW/OW scores. No hyperparameter fitting."""
import numpy as np
from common import require


def distribution(cosine, temperature=0.05):
    x = np.asarray(cosine, dtype=np.float64) / temperature
    x -= x.max(axis=-1, keepdims=True)
    q = np.exp(x)
    return q / q.sum(axis=-1, keepdims=True)


def measure(q, attrs, objects, mask):
    attrs, objects, mask = np.asarray(attrs), np.asarray(objects), np.asarray(mask)
    require(q.ndim == 3 and q.shape[1:] == (12, 16), 'Expected N x objects x attrs')
    require(np.isfinite(q).all() and np.allclose(q.sum(-1), 1, atol=1e-5), 'Invalid distribution')
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        return {'images': 0}
    rows = q[ids, objects[ids]]
    pred = rows.argmax(-1)
    correct = pred == attrs[ids]
    pair_ids = objects[ids] * 16 + attrs[ids]
    macro_pair = np.mean([correct[pair_ids == p].mean() for p in np.unique(pair_ids)])
    per_attr = []
    for a in range(16):
        use = attrs[ids] == a
        per_attr.append({'index': a, 'images': int(use.sum()),
                         'accuracy_pct': float(100 * correct[use].mean()) if use.any() else None})
    confusion = np.zeros((16, 16), dtype=np.int64)
    np.add.at(confusion, (attrs[ids], pred), 1)
    return {'images': len(ids), 'pairs': int(len(np.unique(pair_ids))),
            'accuracy_pct': float(100 * correct.mean()),
            'pair_macro_accuracy_pct': float(100 * macro_pair),
            'nll': float(-np.log(np.maximum(rows[np.arange(len(ids)), attrs[ids]], 1e-30)).mean()),
            'per_attribute': per_attr, 'confusion': confusion.tolist()}


def transitions(reference_q, new_q, attrs, objects, mask):
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        return {'images': 0}
    ref = reference_q[ids, objects[ids]].argmax(-1) == attrs[ids]
    new = new_q[ids, objects[ids]].argmax(-1) == attrs[ids]
    repair, damage = int((~ref & new).sum()), int((ref & ~new).sum())
    return {'images': len(ids), 'repaired': repair, 'damaged': damage,
            'net_accuracy_pp': float(100 * (repair - damage) / len(ids))}


def pair_bootstrap(q1, q0, attrs, objects, mask, repeats=2000):
    # Cluster the resampling by composition, not image. Descriptive interval
    # only; it is NOT a proof of significance/generalization across datasets.
    ids = np.flatnonzero(mask)
    a, o = attrs[ids], objects[ids]
    d = (q1[ids, o].argmax(-1) == a).astype(float) - (q0[ids, o].argmax(-1) == a)
    pairs = o * 16 + a
    means = np.array([d[pairs == p].mean() for p in np.unique(pairs)])
    if len(means) == 0:
        return None
    rng = np.random.RandomState(10102026)
    draws = means[rng.randint(0, len(means), size=(repeats, len(means)))].mean(1) * 100
    return {'macro_delta_pp': float(100 * means.mean()),
            'pair_cluster_bootstrap_95pct_pp': [float(x) for x in np.percentile(draws, [2.5, 97.5])],
            'repeats': repeats, 'seed': 10102026, 'interval_is_descriptive': True}


def family_gate(summaries, family, wrong1, wrong2):
    unseen = {k: v['val']['unseen'] for k, v in summaries.items()}
    k, name = unseen[family], unseen['T_name']
    require(k['images'] > 0, 'No official VAL unseen images')
    deltas = {c: k['pair_macro_accuracy_pct'] - unseen[c]['pair_macro_accuracy_pct']
              for c in ('T_name', wrong1, wrong2)}
    micro = k['accuracy_pct'] - name['accuracy_pct']
    train_delta = (summaries[family]['train']['all']['pair_macro_accuracy_pct'] -
                   summaries['T_name']['train']['all']['pair_macro_accuracy_pct'])
    passed = all(v >= 1.0 for v in deltas.values()) and micro >= 0 and train_delta >= 0
    return {'passed': bool(passed), 'macro_deltas_pp': deltas,
            'unseen_micro_minus_name_pp': micro, 'train_macro_minus_name_pp': train_delta,
            'rule': 'VAL unseen pair macro >= name AND each wrong control +1.0pp; '
                    'VAL unseen micro >= name; TRAIN pair macro >= name',
            'budget_screen_not_significance': True}
