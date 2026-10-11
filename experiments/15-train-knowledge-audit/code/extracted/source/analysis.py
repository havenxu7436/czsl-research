"""Descriptive TRAIN diagnostics only. No fitting, selection or VAL gate."""
import hashlib
import numpy as np
from common import require

VARIANTS = ('T_name', 'T_K', 'T_perm1', 'T_perm2', 'T_AE', 'T_AE_perm1', 'T_AE_perm2')
FAMILIES = (('OE_AE', ('T_K', 'T_perm1', 'T_perm2'), tuple(range(9))),
            ('AE_only', ('T_AE', 'T_AE_perm1', 'T_AE_perm2'), (0, 3, 6)))


def make_prompts(protocol, ae, oe):
    labels, names = protocol['labels'], protocol['names']
    result = {}
    for v in VARIANTS:
        key = 'P_perm1' if v.endswith('perm1') else ('P_perm2' if v.endswith('perm2') else 'P_K')
        perm = protocol['permutations'][key]
        rows = []
        for oi, obj in enumerate(labels['objs']):
            for ai, attr in enumerate(labels['attrs']):
                base = 'A photo of {} made of {}.'.format(names['o'][oi], names['a'][ai])
                for aj in range(3):
                    for oj in range(3):
                        t = base
                        if v != 'T_name':
                            t += ' It has {}.'.format(ae[labels['attrs'][perm['a'][ai]]][aj])
                            if not v.startswith('T_AE'):
                                t += ' Object shape: {}.'.format(oe[labels['objs'][perm['o'][oi]]][oj])
                        rows.append(t)
        result[v] = rows
    return result


def metrics(scores, a, o):
    require(scores.shape == (len(a), 16) and np.isfinite(scores).all(), 'Bad diagnostic scores')
    pred = scores.argmax(1)
    ok = pred == a
    pairs = sorted(set(zip(o.tolist(), a.tolist())))
    macro = np.mean([ok[(o == oi) & (a == ai)].mean() for oi, ai in pairs])
    competitor = scores.copy()
    competitor[np.arange(len(a)), a] = -np.inf
    margins = scores[np.arange(len(a)), a] - competitor.max(1)
    confusion = np.zeros((16, 16), dtype=np.int64)
    np.add.at(confusion, (a, pred), 1)
    return {'images': len(a), 'pairs': len(pairs), 'correct': int(ok.sum()),
            'micro_pct': float(100 * ok.mean()), 'pair_macro_pct': float(100 * macro),
            'mean_gt_margin': float(margins.mean()), 'confusion': confusion.tolist()}, pred, margins


def transition(pred, reference, a):
    ok, old = pred == a, reference == a
    repair, damage = int((ok & ~old).sum()), int((old & ~ok).sum())
    return {'repair': repair, 'damage': damage, 'net': repair - damage}


def select_samples(records, per_attribute=6):
    """Class stratification + object round-robin; never uses success/failure."""
    selected = []
    for ai in range(16):
        pools = {}
        for i, r in enumerate(records):
            if r['attr'] == ai:
                pools.setdefault(r['object'], []).append(i)
        for oi, pool in pools.items():
            pool.sort(key=lambda i: hashlib.sha256(('train-audit-1010|' + records[i]['image']).encode()).hexdigest())
        chosen = []
        while len(chosen) < per_attribute and any(pools.values()):
            for oi in sorted(pools):
                if pools[oi] and len(chosen) < per_attribute:
                    chosen.append(pools[oi].pop(0))
        selected.extend(chosen)
    require(len(selected) == len(set(selected)) and len(selected) <= 96, 'Sampling budget violation')
    return selected


def load_train(inputs, protocol):
    records = inputs.js('train_records.json')
    require(len(records) == 22998 and len({r['image'] for r in records}) == len(records), 'TRAIN identity count')
    truth = inputs.arrays('teacher/train_truth.npz')
    a, o = truth['attrs'], truth['objects']
    require(np.array_equal(a, [r['attr'] for r in records]) and
            np.array_equal(o, [r['object'] for r in records]), 'TRAIN label order')
    require(np.all(truth['seen']) and all(r['seen'] for r in records), 'Non-TRAIN categories')
    require(len(set(zip(o.tolist(), a.tolist()))) == 83, 'TRAIN pair count')
    require(np.all((a >= 0) & (a < 16)) and np.all((o >= 0) & (o < 12)), 'Label index range')
    chunks = []
    for start in range(0, len(records), 512):
        end = min(start + 512, len(records))
        stem = 'features/train/%06d' % start
        receipt = inputs.js(stem + '.json')
        shard = inputs.arrays(stem + '.npz')
        identity = hashlib.sha256('\n'.join(r['image'] + ' ' + r['image_sha256']
                                            for r in records[start:end]).encode()).hexdigest()
        require(receipt['start'] == start and receipt['end'] == end and
                receipt['identity_sha256'] == identity == str(shard['identity_sha256'].item()), 'Shard identity')
        require(receipt['sha256'] == inputs.expected[stem + '.npz'], 'Shard receipt SHA')
        x = shard['features']
        require(x.shape == (end - start, 768) and x.dtype == np.float32 and np.isfinite(x).all(), 'Shard shape')
        require(np.allclose(np.linalg.norm(x, axis=1), 1, atol=2e-3), 'Image feature normalization')
        chunks.append(x)
    old_tables = inputs.arrays('text_features.npz')
    scores = {}
    for v in VARIANTS:
        data = inputs.arrays('teacher/train_' + v + '.npz')
        require(data['cosine'].shape == (len(a), 12, 16), 'Saved score shape')
        scores[v] = data['cosine'][np.arange(len(a)), o]
        require(np.array_equal(scores[v].argmax(1), data['q'][np.arange(len(a)), o].argmax(1)), 'q/score top1')
    return records, np.concatenate(chunks), a, o, old_tables, scores


def pooled_report(scores, a, o, labels):
    summary, preds, margins = {}, {}, {}
    for v in VARIANTS:
        summary[v], preds[v], margins[v] = metrics(scores[v], a, o)
    for v in VARIANTS:
        summary[v]['vs_name'] = transition(preds[v], preds['T_name'], a)
    attributes = []
    for ai, attr in enumerate(labels['attrs']):
        mask = a == ai
        require(mask.any(), 'Empty TRAIN attribute')
        row = {'attr': attr, 'images': int(mask.sum()), 'variants': {}}
        for v in VARIANTS:
            m, _, _ = metrics(scores[v][mask], a[mask], o[mask])
            row['variants'][v] = {'correct': m['correct'], 'micro_pct': m['micro_pct'],
                                 'pair_macro_pct': m['pair_macro_pct'], 'mean_gt_margin': m['mean_gt_margin'],
                                 'predicted_on_all_train': int((preds[v] == ai).sum()),
                                 'vs_name': transition(preds[v][mask], preds['T_name'][mask], a[mask])}
        attributes.append(row)
    pairs = []
    for oi, ai in sorted(set(zip(o.tolist(), a.tolist()))):
        mask = (o == oi) & (a == ai)
        pairs.append({'attr': labels['attrs'][ai], 'object': labels['objs'][oi], 'images': int(mask.sum()),
                      'correct': {v: int((preds[v][mask] == ai).sum()) for v in VARIANTS},
                      'K_net_vs_name': int((preds['T_K'][mask] == ai).sum() - (preds['T_name'][mask] == ai).sum())})
    return {'scope': 'TRAIN descriptive; not held-out generalization', 'metrics': summary,
            'per_attribute': attributes, 'per_pair': pairs}, preds


def unit_scores(features, tables, o):
    require(tables.shape == (12, 16, 9, 768), 'Unit text shape')
    result = np.empty((len(o), 16, 9), dtype=np.float32)
    for oi in range(12):
        ix = np.flatnonzero(o == oi)
        result[ix] = (features[ix] @ tables[oi].reshape(144, 768).T).reshape(-1, 16, 9)
    return result


def sentence_report(features, unit_tables, old_tables, scores, a, o, labels):
    replay = {}
    for v in VARIANTS:
        mean = unit_tables[v].mean(axis=2, dtype=np.float32)
        d = float(np.abs(mean - old_tables[v]).max())
        require(d < 2e-5, 'Re-encoded mean text vector differs: ' + v)
        calculated = np.empty_like(scores[v])
        for oi in range(12):
            ix = np.flatnonzero(o == oi)
            calculated[ix] = features[ix] @ mean[oi].T
        error = float(np.abs(calculated - scores[v]).max())
        mismatch = int((calculated.argmax(1) != scores[v].argmax(1)).sum())
        require(error < 2e-5 and mismatch == 0, 'Pooled TRAIN prediction replay differs: ' + v)
        replay[v] = {'text_abs_diff_max': d, 'cosine_abs_diff_max': error, 'top1_mismatches': mismatch}
    name_summary, name_pred, _ = metrics(scores['T_name'], a, o)
    families = {}
    for family, variants, slots in FAMILIES:
        arrays = {v: unit_scores(features, unit_tables[v], o) for v in variants}
        rows = []
        for slot in slots:
            row = {'slot': slot, 'ae_index': slot // 3, 'oe_index': slot % 3 if family == 'OE_AE' else None,
                   'variants': {}}
            for v in variants:
                m, pred, _ = metrics(arrays[v][:, :, slot], a, o)
                m['vs_name'] = transition(pred, name_pred, a)
                m['per_attribute'] = []
                for ai, attr in enumerate(labels['attrs']):
                    mask = a == ai
                    local, _, _ = metrics(arrays[v][mask, :, slot], a[mask], o[mask])
                    m['per_attribute'].append({'attr': attr, 'images': int(mask.sum()),
                                               'micro_pct': local['micro_pct'],
                                               'mean_gt_margin': local['mean_gt_margin'],
                                               'vs_name': transition(pred[mask], name_pred[mask], a[mask])})
                row['variants'][v] = m
            correct, p1, p2 = (row['variants'][v] for v in variants)
            row['correct_macro_deltas_pp'] = {
                'name': correct['pair_macro_pct'] - name_summary['pair_macro_pct'],
                'perm1': correct['pair_macro_pct'] - p1['pair_macro_pct'],
                'perm2': correct['pair_macro_pct'] - p2['pair_macro_pct']}
            rows.append(row)
        families[family] = rows
    return {'scope': 'TRAIN sentence-position diagnostics, NOT a prompt search or a proceed gate',
            'replay': replay, 'families': families, 'weights_fitted': False,
            'optimizer_steps': 0, 'val_used': False, 'test_used': False}
