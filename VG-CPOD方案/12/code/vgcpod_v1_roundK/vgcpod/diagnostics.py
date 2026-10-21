"""Export VAL gate/partner summaries and fixed paired per-composition gains."""
import csv
from pathlib import Path

import numpy as np
import torch

from .common import load_torch, read_json, seed_all, write_json
from .engine import check_run_identity, loader, make_model


@torch.no_grad()
def diagnose(args):
    run = Path(args.run)
    state = load_torch(run / 'checkpoints' / 'best.pt')
    check_run_identity(args.cache, run, state)
    cfg = state['config']
    seed_all(state['seed'])
    model, manifest = make_model(args.cache, cfg, state['seed'], args.device)
    model.load_state_dict(state['model'])
    model.eval()
    gs, g2s, ps, p2s, count = None, None, None, None, 0
    for g, fa, fo, attrs, objs in loader(args.cache, 'val', cfg):
        out = model(g.to(args.device), fa.to(args.device), fo.to(args.device))
        gate = out['gamma'].double()
        if gs is None:
            gs, g2s = gate.sum(0), gate.square().sum(0)
        else:
            gs += gate.sum(0)
            g2s += gate.square().sum(0)
        if 'pi' in out:
            pi = out['pi'].double()
            if ps is None:
                ps, p2s = pi.sum(0), pi.square().sum(0)
            else:
                ps += pi.sum(0)
                p2s += pi.square().sum(0)
        count += len(attrs)
    folder = run / 'diagnostics'
    folder.mkdir(exist_ok=True)
    arrays = {'gamma_mean': (gs / count).cpu().numpy(),
              'gamma_variance': (g2s / count - (gs / count).square()).clamp_min(0).cpu().numpy()}
    if ps is not None:
        arrays.update(partner_mean=(ps / count).cpu().numpy(),
                      partner_variance=(p2s / count - (ps / count).square()).clamp_min(0).cpu().numpy())
    np.savez(folder / 'val_mechanisms.npz', **arrays)
    write_json(folder / 'axes.json', {'objects': manifest['labels']['objs'], 'attributes': manifest['labels']['attrs'],
                                     'gate_axes': ['object', 'mode'], 'partner_axes': ['attribute', 'object'],
                                     'validation_images': count, 'selected_epoch': state['epoch']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key in ('gamma_mean', 'gamma_variance', 'partner_mean'):
        if key not in arrays:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        view = ax.imshow(arrays[key], aspect='auto', interpolation='nearest')
        fig.colorbar(view, ax=ax)
        ax.set_title(key + ' (VAL)')
        ax.set_xlabel('object' if key == 'partner_mean' else 'mode')
        ax.set_ylabel('attribute' if key == 'partner_mean' else 'object')
        fig.tight_layout()
        fig.savefig(folder / (key + '.png'), dpi=160)
        plt.close(fig)
    print('DIAGNOSTICS_WRITTEN', folder, flush=True)


def compare(args):
    root = Path(args.out) / 'ut-zap50k'
    rows = []
    for high, low in [('V1L', 'V1'), ('V2', 'V1L'), ('V5', 'V4')]:
        for seed in args.seeds:
            for setting in ('val_cw', 'val_ow', 'test_cw', 'test_ow'):
                left = root / high / ('seed%d' % seed) / 'predictions' / (setting + '.npz')
                right = root / low / ('seed%d' % seed) / 'predictions' / (setting + '.npz')
                if not left.exists() or not right.exists():
                    continue
                with np.load(left, allow_pickle=False) as l, np.load(right, allow_pickle=False) as r:
                    if not np.array_equal(l['image_ids'], r['image_ids']) or not np.array_equal(l['candidate_pairs'], r['candidate_pairs']):
                        raise ValueError('Prediction identities/axes differ')
                    # Candidate labels are strings; truths use archived integer primitive axes.
                    manifest = read_json(Path(args.cache) / 'features_manifest.json')
                    labels = manifest['labels']
                    ai = np.array([labels['attrs'].index(p[0]) for p in l['candidate_pairs']])
                    oi = np.array([labels['objs'].index(p[1]) for p in l['candidate_pairs']])
                    a, o = l['attribute_truth'], l['object_truth']
                    if not np.array_equal(a, r['attribute_truth']) or not np.array_equal(o, r['object_truth']):
                        raise ValueError('Prediction truths differ')
                    lp, rp = l['log_probabilities'].argmax(1), r['log_probabilities'].argmax(1)
                    lc, rc = (ai[lp] == a) & (oi[lp] == o), (ai[rp] == a) & (oi[rp] == o)
                    lcond = l['conditional_logits_gt_object'].argmax(1) == a
                    rcond = r['conditional_logits_gt_object'].argmax(1) == a
                    for oa in sorted(set(zip(o.tolist(), a.tolist()))):
                        mask = (o == oa[0]) & (a == oa[1])
                        rows.append({'comparison': high + '-' + low, 'seed': seed, 'setting': setting,
                                     'attribute': labels['attrs'][oa[1]], 'object': labels['objs'][oa[0]],
                                     'images': int(mask.sum()), 'pair_gain_unbiased_pct': float((lc[mask].mean() - rc[mask].mean()) * 100),
                                     'gt_object_attr_gain_pct': float((lcond[mask].mean() - rcond[mask].mean()) * 100)})
    if not rows:
        raise FileNotFoundError('No complete paired prediction exports')
    target = Path(args.out) / 'per_pair_gains.csv'
    with open(target, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('PAIR_GAINS_WRITTEN', target, flush=True)
