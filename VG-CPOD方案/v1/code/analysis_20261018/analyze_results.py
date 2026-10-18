"""Reproduce the VG-CPOD empirical audit from archived JSON/NPZ/checkpoints."""
import csv
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / 'server_results_20261018'
RUNS = DATA / 'vgcpod_v1_runs'
SOURCE = DATA / 'vgcpod_project/vgcpod_v1'
OUT = Path(__file__).resolve().parent
TABLES = OUT / 'tables'
FIGURES = OUT / 'figures'
SEEDS = (3351, 3352, 3353)
VARIANTS = ('V0', 'V1', 'V1L', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7')
SETTINGS = ('val_cw', 'val_ow', 'test_cw', 'test_ow')
PAIRS = (('V1', 'V0'), ('V1L', 'V1'), ('V2', 'V1L'), ('V5', 'V4'),
         ('V5', 'V2'), ('V2', 'V3'), ('V2', 'V6'), ('V5', 'V7'), ('V2', 'V0'), ('V5', 'V0'))
KEYS = ('AUC_native', 'AUC_exact', 'HM_at_val_bias', 'HM_oracle_native',
        'pair_acc_unbiased', 'pair_macro_unbiased', 'pair_nll_unbiased',
        'gt_object_attr_acc', 'gt_object_attr_nll', 'seen_at_val_bias', 'unseen_at_val_bias')
torch.set_num_threads(1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_write(name, rows):
    if not rows:
        return
    with open(TABLES / name, 'w', encoding='utf-8-sig', newline='') as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def mean_sd(values):
    return {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}


def checkpoint(path):
    kw = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kw['weights_only'] = False
    return torch.load(path, **kw)


def softmax(x):
    z = np.asarray(x, dtype=np.float64)
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def index_predictions(path, labels, train_pairs, cw_pairs):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    a, o = data['attribute_truth'], data['object_truth']
    cand = [tuple(p) for p in data['candidate_pairs'].tolist()]
    ai = np.array([labels['attrs'].index(p[0]) for p in cand])
    oi = np.array([labels['objs'].index(p[1]) for p in cand])
    pos = {p: i for i, p in enumerate(cand)}
    truth_pairs = [(labels['attrs'][ia], labels['objs'][jo]) for ia, jo in zip(a, o)]
    truth = np.array([pos[p] for p in truth_pairs])
    seen = np.array([p in train_pairs for p in cand])
    cw = np.array([p in cw_pairs for p in cand])
    pred = torch.from_numpy(data['log_probabilities']).topk(1, dim=1).indices[:, 0].numpy()
    data.update(a=a, o=o, ai=ai, oi=oi, truth=truth, true_pairs=truth_pairs,
                seen_classes=seen, seen_images=seen[truth], cw_classes=cw, pred=pred,
                correct=pred == truth)
    return data


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    archive = DATA / 'VG-CPOD_v1_results_20261018.tar.gz'
    expected_archive = (DATA / (archive.name + '.sha256')).read_text().split()[0]
    assert sha(archive) == expected_archive, 'Archive SHA256 differs'
    cache = RUNS / 'cache_ut_clip_l14'
    manifest = read(cache / 'features_manifest.json')
    labels, splits = manifest['labels'], manifest['pairs']
    train_pairs = set(map(tuple, splits['train']))
    cache_sha = sha(cache / 'features_manifest.json')
    audit = {'archive_sha256': expected_archive, 'archive_checksum': 'passed',
             'errors': [], 'main_models': 27, 'sensitivity_models': 6, 'baselines': 2,
             'cache_files_verified': 0, 'omitted_feature_arrays': [], 'prediction_files': 0}
    for rel, expected in manifest['files_sha256'].items():
        p = cache / rel
        if not p.exists():
            audit['omitted_feature_arrays'].append(rel)
        else:
            assert sha(p) == expected, rel
            audit['cache_files_verified'] += 1
    release = read(SOURCE / 'release_manifest.json')
    release_changes, filename_aliases = [], {}
    source_files = [p for p in SOURCE.rglob('*') if p.is_file()]
    for rel, expected in release['files_sha256'].items():
        p = SOURCE / rel
        if not p.exists():
            candidates = [q for q in source_files if q.suffix == p.suffix and sha(q) == expected]
            assert len(candidates) == 1, ('Missing source file', rel)
            p = candidates[0]
            filename_aliases[rel] = p.relative_to(SOURCE).as_posix()
        if sha(p) != expected:
            release_changes.append(rel)
    audit['source_document_filename_aliases'] = filename_aliases
    audit['release_changed_files'] = release_changes
    assert not release_changes
    spec = importlib.util.spec_from_file_location('verified_archived_metrics', SOURCE / 'vgcpod/metrics.py')
    metrics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_module)
    all_runs = sorted((RUNS / 'main/ut-zap50k').glob('V*/seed*/metrics.json'))
    assert len(all_runs) == 27
    all_runs += sorted((RUNS / 'sensitivity').glob('*/ut-zap50k/*/seed*/metrics.json'))
    assert len(all_runs) == 33
    all_runs += [RUNS / 'main/ut-zap50k' / name / 'metrics.json' for name in ('name_clip', 'chain_zeroshot')]
    numeric_rows, history_rows, protocols, mechanisms, prediction_diagnostics = [], [], {}, [], []
    stats, histories, predictions, per_pair_rows, primitive_rows = {}, {}, {}, [], []
    max_errors = {'normalization': 0., 'conditional_probability': 0., 'native_auc': 0., 'exact_auc': 0.,
                  'unbiased_accuracy': 0., 'fixed_bias_hm': 0., 'nll': 0., 'cw_from_ow_log_probability': 0.}
    reference_ids = {}
    total_epoch_seconds = 0.
    full_source_hashes = None
    for run_index, mp in enumerate(all_runs):
        folder = mp.parent
        rel = folder.relative_to(RUNS).as_posix()
        result, p = read(mp), read(folder / 'protocol.json')
        cfg = p['config']
        assert p['feature_manifest_sha256'] == cache_sha
        for name, expected in p['source_sha256'].items():
            assert sha(SOURCE / name) == expected, (rel, name)
        if full_source_hashes is None:
            full_source_hashes = p['source_sha256']
        assert full_source_hashes == p['source_sha256']
        protocols[rel] = p
        if rel.startswith('main/'):
            group = 'main'
            variant = folder.parent.name if folder.name.startswith('seed') else folder.name
        else:
            group = folder.parents[2].name
            variant = folder.parent.name
        seed = p['seed']
        stats[rel] = result
        is_trained = folder.name.startswith('seed')
        if is_trained:
            h = read(folder / 'history.json')
            assert [r['epoch'] for r in h] == list(range(1, 11)), rel
            assert h[-1]['step'] == 3600 and cfg['epochs'] == 10
            best_epoch = int(np.argmax([r['val_cw']['AUC_native'] for r in h])) + 1
            assert result['selected_epoch'] == best_epoch
            assert abs(h[best_epoch - 1]['val_cw']['AUC_native'] - result['metrics']['val_cw']['AUC_native']) < 1e-9
            assert sha(folder / 'checkpoints/best.pt') == result['checkpoint_sha256']
            best = checkpoint(folder / 'checkpoints/best.pt')
            last = checkpoint(folder / 'checkpoints/last.pt')
            assert best['epoch'] == best_epoch and last['epoch'] == 10
            assert last['step'] == 3600
            assert last['history'] == h
            for key in ('ea', 'eo', 'ec'):
                with np.load(cache / 'text_features.npz') as texts:
                    np.testing.assert_allclose(best['model'][key].numpy(), texts[key], atol=0, rtol=0)
            for r in h:
                history_rows.append(dict(group=group, variant=variant, seed=seed, epoch=r['epoch'],
                     val_auc=r['val_cw']['AUC_native'], val_auc_exact=r['val_cw']['AUC_exact'],
                     val_nll=r['val_cw']['pair_nll_unbiased'], **r['train'], seconds=r['seconds']))
                total_epoch_seconds += r['seconds']
                assert all(math.isfinite(x) for x in r['train'].values())
            histories[rel] = h
            b = best['model']['Bvis'].double().numpy()
            gram = b @ b.T
            eig = np.linalg.eigvalsh(gram)
            row_norm = np.sqrt(np.diag(gram))
            corr = gram / (row_norm[:, None] * row_norm[None, :])
            state = best['model']
            u = (state['ea'] @ state['Au'].T).numpy()
            v = (state['eo'] @ state['Av'].T).numpy()
            delta = np.einsum('ir,jr,dr->ijd', u, v, state['Wc'].numpy())
            hs = h[best_epoch - 1]['train']
            mechanism = {'group': group, 'variant': variant, 'seed': seed, 'selected_epoch': best_epoch,
                'rank': cfg['rank'], 'tau_pi': cfg['tau_pi'], 'parameters': p['trainable_parameters'],
                'gamma_mean_selected_epoch': hs['gamma_mean'], 'gamma_variance_selected_epoch': hs['gamma_variance'],
                'orth_selected_epoch_average': hs['orth'], 'orth_checkpoint': float(((gram - np.eye(len(b))) ** 2).mean()),
                'mode_basis_effective_rank': float(eig.sum() ** 2 / (eig * eig).sum()),
                'mode_basis_row_norm_mean': float(row_norm.mean()), 'mode_basis_row_norm_max': float(row_norm.max()),
                'mode_basis_abs_cosine_off_diagonal': float(np.abs(corr[~np.eye(len(b), dtype=bool)]).mean()),
                'delta_reference_gamma1_norm_mean': float(np.linalg.norm(delta, axis=-1).mean()),
                'regularizer_fraction_selected': (cfg['lambda_orth'] if variant != 'V7' else 0) * hs['orth'] / hs['total']}
            if 'partner_entropy' in hs:
                mechanism.update(partner_entropy=hs['partner_entropy'],
                    partner_normalized_entropy=hs['partner_entropy'] / math.log(12),
                    partner_effective_count=math.exp(hs['partner_entropy']),
                    partner_kl_to_uniform=math.log(12) - hs['partner_entropy'],
                    prim_weighted=cfg['lambda_prim'] * hs['prim'], cent_weighted=cfg['lambda_cent'] * hs['cent'])
            mechanisms.append(mechanism)
        for setting in SETTINGS:
            phase, world = setting.split('_')
            cw_pairs = train_pairs | set(map(tuple, splits[phase]))
            data = index_predictions(folder / 'predictions' / (setting + '.npz'), labels, train_pairs, cw_pairs)
            audit['prediction_files'] += 1
            m = result['metrics'][setting]
            a, o, y = data['a'], data['o'], data['truth']
            records = read(cache / (phase + '_records.json'))
            assert data['image_ids'].tolist() == [r['image'] for r in records]
            assert a.tolist() == [r['attr'] for r in records] and o.tolist() == [r['object'] for r in records]
            assert m['images'] == len(y) and m['candidates'] == len(data['candidate_pairs'])
            lp, prob = data['log_probabilities'], data['probabilities']
            max_errors['normalization'] = max(max_errors['normalization'], float(abs(prob.sum(1) - 1).max()))
            np.testing.assert_allclose(prob, np.exp(lp), atol=1e-8)
            base_acc = data['correct'].mean() * 100
            max_errors['unbiased_accuracy'] = max(max_errors['unbiased_accuracy'], abs(base_acc - m['pair_acc_unbiased']))
            max_errors['nll'] = max(max_errors['nll'], abs(float(-lp[np.arange(len(y)), y].mean()) - m['pair_nll_unbiased']))
            native = metrics_module.native_curve(lp, y, data['seen_classes'])
            exact = metrics_module.exact_curve(lp, y, data['seen_classes'])
            max_errors['native_auc'] = max(max_errors['native_auc'], abs(native['AUC'] - m['AUC_native']))
            max_errors['exact_auc'] = max(max_errors['exact_auc'], abs(exact['AUC'] - m['AUC_exact']))
            bias = result['metrics']['val_' + world]['bias_native']
            assert m['fixed_val_bias'] == bias
            z = torch.from_numpy(lp.copy())
            z[:, ~torch.from_numpy(data['seen_classes'])] += bias
            pred_cal = z.topk(1, dim=1).indices[:, 0].numpy()
            correct_cal = pred_cal == y
            sm = data['seen_images']
            s, un = correct_cal[sm].mean(), correct_cal[~sm].mean()
            hvalue = float(metrics_module.hm(s, un) * 100)
            max_errors['fixed_bias_hm'] = max(max_errors['fixed_bias_hm'], abs(hvalue - m['HM_at_val_bias']))
            data.update(pred_cal=pred_cal, correct_cal=correct_cal)
            gtcond = data['conditional_logits_gt_object']
            correct_cond = gtcond.argmax(1) == a
            d = {'group': group, 'variant': variant, 'seed': seed, 'setting': setting,
                 'seen_images': int(sm.sum()), 'unseen_images': int((~sm).sum()),
                 'gt_object_attr_acc_seen': float(correct_cond[sm].mean() * 100),
                 'gt_object_attr_acc_unseen': float(correct_cond[~sm].mean() * 100),
                 'pair_acc_unbiased_seen': float(data['correct'][sm].mean() * 100),
                 'pair_acc_unbiased_unseen': float(data['correct'][~sm].mean() * 100),
                 'max_seen_accuracy': float(exact['seen'][0]), 'max_unseen_accuracy': float(exact['unseen'][-1]),
                 'pair_argmax_object_accuracy': float((data['oi'][data['pred']] == o).mean() * 100),
                 'extra_candidates': int((~data['cw_classes']).sum()),
                 'extra_candidate_pred_unbiased_pct': float((~data['cw_classes'][data['pred']]).mean() * 100),
                 'extra_candidate_pred_calibrated_pct': float((~data['cw_classes'][pred_cal]).mean() * 100)}
            if world == 'ow':
                object_prob = np.stack([prob[:, data['oi'] == j].sum(1) for j in range(len(labels['objs']))], axis=-1)
                derived_cond = np.stack([prob[k, data['oi'] == j] / object_prob[k, j] for k, j in enumerate(o)])
                max_errors['conditional_probability'] = max(max_errors['conditional_probability'],
                                     float(abs(derived_cond - softmax(gtcond)).max()))
                # Candidates are alphabetical by attribute then object, so each object has attribute axis order.
                d.update(object_marginal_accuracy=float((object_prob.argmax(1) == o).mean() * 100),
                         extra_candidate_probability_mass=float(prob[:, ~data['cw_classes']].sum(1).mean()),
                         extra_candidate_calibrated_mass=float(softmax(z.numpy())[:, ~data['cw_classes']].sum(1).mean()))
                best_unseen = np.flatnonzero(~data['seen_classes'])[lp[:, ~data['seen_classes']].argmax(1)]
                d['unseen_limit_extra_candidate_pct'] = float((~data['cw_classes'][best_unseen[~sm]]).mean() * 100)
            prediction_diagnostics.append(d)
            data['correct_cond'] = correct_cond
            predictions[(rel, setting)] = data
            row = dict(group=group, variant=variant, seed=seed, setting=setting,
                       selected_epoch=result.get('selected_epoch'), **{k: m[k] for k in KEYS}, bias=bias)
            numeric_rows.append(row)
            if is_trained and group == 'main':
                for pair in sorted(set(data['true_pairs'])):
                    mask = np.array([t == pair for t in data['true_pairs']])
                    per_pair_rows.append(dict(variant=variant, seed=seed, setting=setting,
                        attribute=pair[0], object=pair[1], seen=pair in train_pairs, images=int(mask.sum()),
                        pair_acc_unbiased=float(data['correct'][mask].mean() * 100),
                        pair_acc_calibrated=float(correct_cal[mask].mean() * 100),
                        gt_object_attr_acc=float(correct_cond[mask].mean() * 100)))
                if setting == 'test_ow':
                    for i, name in enumerate(labels['attrs']):
                        mask = a == i
                        if mask.any():
                            primitive_rows.append(dict(variant=variant, seed=seed, attribute=name, images=int(mask.sum()),
                                                       gt_object_attr_acc=float(correct_cond[mask].mean() * 100)))
        for phase in ('val', 'test'):
            cw = predictions[(rel, phase + '_cw')]
            ow = predictions[(rel, phase + '_ow')]
            projected = ow['log_probabilities'][:, ow['cw_classes']]
            lse = np.log(np.exp(projected - projected.max(1, keepdims=True)).sum(1, keepdims=True)) + projected.max(1, keepdims=True)
            max_errors['cw_from_ow_log_probability'] = max(max_errors['cw_from_ow_log_probability'],
                                          float(abs(projected - lse - cw['log_probabilities']).max()))
        print('AUDITED', run_index + 1, '/', len(all_runs), rel, flush=True)
    audit.update(max_prediction_recomputation_errors=max_errors, model_epochs=330,
                 recorded_epoch_seconds=total_epoch_seconds, source_protocol_hashes='all identical and verified',
                 selection='all first strict maximum VAL CW AUC_native', epoch_completion='all 33 models: epochs 1-10, step3600')
    csv_write('all_runs_metrics.csv', numeric_rows)
    csv_write('training_history.csv', history_rows)
    csv_write('mechanisms.csv', mechanisms)
    csv_write('prediction_diagnostics.csv', prediction_diagnostics)
    csv_write('per_pair_metrics.csv', per_pair_rows)
    csv_write('attribute_accuracy.csv', primitive_rows)
    aggregate = []
    for v in VARIANTS:
        for setting in SETTINGS:
            group = [r for r in numeric_rows if r['group'] == 'main' and r['variant'] == v and r['setting'] == setting]
            row = dict(variant=v, setting=setting)
            for k in KEYS:
                for key, value in mean_sd([r[k] for r in group]).items():
                    row[k + '_' + key] = value
            aggregate.append(row)
    csv_write('main_mean_sd.csv', aggregate)
    paired = []
    for hi, lo in PAIRS:
        for setting in SETTINGS:
            a = {r['seed']: r for r in numeric_rows if r['group'] == 'main' and r['variant'] == hi and r['setting'] == setting}
            b = {r['seed']: r for r in numeric_rows if r['group'] == 'main' and r['variant'] == lo and r['setting'] == setting}
            for key in KEYS:
                delta = [a[s][key] - b[s][key] for s in SEEDS]
                paired.append(dict(comparison=hi + '-' + lo, setting=setting, metric=key,
                    **{'seed' + str(s): delta[i] for i, s in enumerate(SEEDS)},
                    **mean_sd(delta), positive_seeds=sum(x > 0 for x in delta)))
    csv_write('paired_deltas.csv', paired)
    # Recheck existing summaries against independent aggregation.
    saved = read(RUNS / 'main/summary.json')['by_variant']
    for r in aggregate:
        for k in KEYS:
            if k in saved[r['variant'] + '/' + r['setting']]:
                assert abs(saved[r['variant'] + '/' + r['setting']][k]['mean'] - r[k + '_mean']) < 1e-9
    audit['saved_summary_recomputed'] = 'passed'
    write(OUT / 'audit.json', audit)
    write(OUT / 'analysis_data.json', {'main_aggregate': aggregate, 'paired': paired,
        'mechanisms': mechanisms, 'prediction_diagnostics': prediction_diagnostics,
        'baselines': [r for r in numeric_rows if r['variant'] in ('name_clip', 'chain_zeroshot')],
        'sensitivity': [r for r in numeric_rows if r['group'] != 'main'], 'audit': audit})
    plots(aggregate, paired, history_rows, mechanisms, numeric_rows, prediction_diagnostics)
    print('ANALYSIS_READY', OUT, flush=True)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def plots(aggregate, paired, history_rows, mechanisms, numeric_rows, diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 140})
    colors = {'V0': '#556270', 'V1': '#236b8e', 'V1L': '#76a5af', 'V2': '#d78128', 'V3': '#baa762',
              'V4': '#7d78aa', 'V5': '#b94a57', 'V6': '#7c996f', 'V7': '#a19aa5'}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, setting, metric, title in zip(axs, ['val_cw', 'test_cw', 'test_ow'],
                    ['AUC_native'] * 3, ['VAL CW (selection)', 'TEST CW', 'TEST OW']):
        rows = [next(r for r in aggregate if r['variant'] == v and r['setting'] == setting) for v in VARIANTS]
        ax.bar(VARIANTS, [r[metric + '_mean'] for r in rows], yerr=[r[metric + '_sd'] for r in rows],
               color=[colors[v] for v in VARIANTS], capsize=3)
        ax.axhline(rows[0][metric + '_mean'], color=colors['V0'], ls='--', lw=1)
        ax.set_title(title)
        ax.set_ylabel('AUC_native (area x 100)')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 48)
    fig.suptitle('Main matrix: mean +/- sample SD over 3 seeds')
    fig.tight_layout()
    fig.savefig(FIGURES / '01_main_auc.png')
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    comparisons = ['V1L-V1', 'V2-V1L', 'V5-V4']
    for ax, setting, title in zip(axs, ['test_cw', 'test_ow'], ['TEST CW', 'TEST OW']):
        for i, c in enumerate(comparisons):
            r = next(r for r in paired if r['comparison'] == c and r['setting'] == setting and r['metric'] == 'AUC_native')
            ax.scatter(np.arange(3) * .08 + i - .08, [r['seed' + str(s)] for s in SEEDS], s=35, color='#74899b')
            ax.errorbar(i, r['mean'], yerr=r['sd'], fmt='D', color='#b94a57', capsize=5, ms=5)
        ax.axhline(0, color='#333', lw=1)
        ax.set_xticks(range(3), ['Local readout\nV1L - V1', 'Visual gating\nV2 - V1L', 'Visual partners\nV5 - V4'])
        ax.set_title(title)
        ax.set_ylabel('Paired AUC_native difference (points)')
    fig.suptitle('Each dot is one matched seed; diamonds show mean +/- SD')
    fig.tight_layout()
    fig.savefig(FIGURES / '02_core_paired_deltas.png')
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for v in ['V0', 'V1', 'V1L', 'V2', 'V5', 'V6']:
        grid = [[next(r for r in history_rows if r['group'] == 'main' and r['variant'] == v and r['seed'] == s and r['epoch'] == e)
                 for e in range(1, 11)] for s in SEEDS]
        auc = np.array([[r['val_auc'] for r in rows] for rows in grid])
        loss = np.array([[r['pair'] + .5*r['cond'] + .5*r['object'] for r in rows] for rows in grid])
        for ax, values in zip(axs, [auc, loss]):
            ax.plot(range(1, 11), values.mean(0), label=v, color=colors[v])
            ax.fill_between(range(1, 11), values.mean(0) - values.std(0, ddof=1), values.mean(0) + values.std(0, ddof=1),
                            color=colors[v], alpha=.09)
    axs[0].set_ylabel('VAL CW AUC_native')
    axs[1].set_ylabel('Common task loss (auxiliary terms excluded)')
    for ax in axs:
        ax.set_xlabel('Epoch')
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / '03_training_curves.png')
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for v in ['V2', 'V5', 'V6', 'V7']:
        grid = [[next(r for r in history_rows if r['group'] == 'main' and r['variant'] == v and r['seed'] == s and r['epoch'] == e)
                 for e in range(1, 11)] for s in SEEDS]
        for ax, field in [(axs[0, 0], 'gamma_mean'), (axs[0, 1], 'gamma_variance'), (axs[1, 0], 'orth')]:
            values = np.array([[r[field] for r in rows] for rows in grid])
            ax.plot(range(1, 11), values.mean(0), label=v, color=colors[v])
    axs[0, 0].set_ylabel('Mean gamma (training batches)')
    axs[0, 0].axhline(2, lw=1, color='#777', ls='--')
    axs[0, 1].set_ylabel('Gamma variance (batch/object/mode pooled)')
    axs[1, 0].set_ylabel('Unweighted orthogonality loss')
    for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
    names, entropy = [], []
    for group, v in [('main', 'V4'), ('tau_pi_0.2', 'V5'), ('main', 'V5'), ('tau_pi_0.05', 'V5')]:
        r = next(r for r in mechanisms if r['group'] == group and r['variant'] == v and r['seed'] == 3351)
        names.append('Uniform' if v == 'V4' else 'tau=' + str(r['tau_pi']))
        entropy.append(r['partner_effective_count'])
    axs[1, 1].bar(names, entropy, color=['#7d78aa', '#c9a0a5', '#b94a57', '#80414c'])
    axs[1, 1].axhline(12, color='#777', ls='--', lw=1)
    axs[1, 1].set_ylabel('Effective partners exp(mean entropy)')
    axs[1, 1].set_ylim(0, 13)
    fig.tight_layout()
    fig.savefig(FIGURES / '04_mechanisms.png')
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, v in zip(axs[:2], ['V2', 'V5']):
        for setting, label, color in [('val_cw', 'VAL CW', '#236b8e'), ('test_cw', 'TEST CW', '#d78128'), ('test_ow', 'TEST OW', '#b94a57')]:
            values = []
            for rank in (16, 32, 64):
                group = 'main' if rank == 32 else 'rank' + str(rank)
                values.append(next(r['AUC_native'] for r in numeric_rows if r['group'] == group and r['variant'] == v and r['seed'] == 3351 and r['setting'] == setting))
            ax.plot([16, 32, 64], values, marker='o', label=label, color=color)
        ax.set_title(v + ': rank')
        ax.set_xticks([16, 32, 64])
        ax.set_xlabel('Rank')
        ax.set_ylabel('AUC_native')
        ax.legend(fontsize=8)
    for setting, label, color in [('val_cw', 'VAL CW', '#236b8e'), ('test_cw', 'TEST CW', '#d78128'), ('test_ow', 'TEST OW', '#b94a57')]:
        values = []
        for tau in (.05, .1, .2):
            group = 'main' if tau == .1 else 'tau_pi_' + str(tau)
            values.append(next(r['AUC_native'] for r in numeric_rows if r['group'] == group and r['variant'] == 'V5' and r['seed'] == 3351 and r['setting'] == setting))
        axs[2].plot([.05, .1, .2], values, marker='o', label=label, color=color)
    axs[2].set_title('V5: partner temperature')
    axs[2].set_xticks([.05, .1, .2])
    axs[2].set_xlabel('tau_pi')
    axs[2].legend(fontsize=8)
    fig.suptitle('Sensitivity: seed3351; select configuration using VAL CW')
    fig.tight_layout()
    fig.savefig(FIGURES / '05_sensitivity.png')
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for setting, label, color in [('test_cw', 'CW', '#236b8e'), ('test_ow', 'OW', '#b94a57')]:
        values = [np.mean([r['max_unseen_accuracy'] for r in diag if r['group'] == 'main' and r['variant'] == v and r['setting'] == setting]) for v in VARIANTS]
        axs[0].plot(VARIANTS, values, marker='o', label=label, color=color)
    axs[0].set_ylabel('Unseen-only ranking accuracy (%)')
    axs[0].set_title('Positive-bias limit: only unseen candidates compete')
    axs[0].legend()
    values = [np.mean([r['unseen_limit_extra_candidate_pct'] for r in diag if r['group'] == 'main' and r['variant'] == v and r['setting'] == 'test_ow']) for v in VARIANTS]
    axs[1].bar(VARIANTS, values, color=[colors[v] for v in VARIANTS])
    axs[1].set_ylabel('Extra-candidate win rate (%)')
    axs[1].set_title('Where the extra 91 OW candidates win')
    for ax in axs:
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES / '06_open_world_competition.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
