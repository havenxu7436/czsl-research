"""Analyze the downloaded B pilot against seed-matched A controls and archived predictions."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
BR = ROOT / 'server_results_roundB_20261019/roundB_20261019'
BS = ROOT / 'server_results_roundB_20261019/vgcpod_v1_roundB'
AR = ROOT / 'server_results_roundA_confirm_20261019/roundA_20261019'
OLD = ROOT / 'server_results_20261018/vgcpod_v1_runs'
CASES = ['D0', 'V1', 'A1', 'A5', 'A6', 'B1', 'B2', 'B3']
SETTINGS = ['val_cw', 'val_ow', 'test_cw', 'test_ow']
torch.set_num_threads(1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv_write(name, rows):
    if not rows:
        return
    with (OUT / 'tables' / name).open('w', encoding='utf-8-sig', newline='') as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def folder(case):
    if case in ('D0', 'V1'):
        return OLD / 'main/ut-zap50k' / ('V0' if case == 'D0' else 'V1') / 'seed3351'
    return next(((BR if case.startswith('B') else AR) / case).glob('ut-zap50k/V*/seed3351'))


def main():
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    helper = module('old_prediction_tools', ROOT / 'analysis_20261018/analyze_results.py')
    evaluator = module('b_evaluator', BS / 'vgcpod/metrics.py')
    cache = OLD / 'cache_ut_clip_l14'
    manifest = read(cache / 'features_manifest.json')
    labels = manifest['labels']
    train = set(map(tuple, manifest['pairs']['train']))
    cache_hash = sha(cache / 'features_manifest.json')
    audit = dict(archive_sha256=sha(ROOT / 'vgcpod_roundB_results_20261019.tar.gz'),
                 feature_manifest_sha256=cache_hash, new_runs=3, seed=3351, prediction_files=0,
                 errors=[], source_files_matched=0)
    assert (BR / 'launcher.exitcode').read_text().strip() == '0'
    status = read(BR / 'round_b_status.json')
    assert len(status) == 3 and {r['case'] for r in status} == {'B1', 'B2', 'B3'}
    assert all(r['status'] == 'complete' and r['exit_code'] == 0 for r in status)
    errors = dict(native_auc=0., exact_auc=0., hm=0., probability_normalization=0., support_mass=0., loss_identity=0.)
    rows, diagnostics, history, gates, geometry, pair_rows = [], [], [], [], [], []
    predictions = {}
    for case in CASES:
        run = folder(case)
        m, p, h = read(run / 'metrics.json'), read(run / 'protocol.json'), read(run / 'history.json')
        cfg = p['config']
        epoch = int(np.argmax([r['val_cw']['AUC_native'] for r in h])) + 1
        assert epoch == m['selected_epoch'] and sha(run / 'checkpoints/best.pt') == m['checkpoint_sha256']
        assert p['seed'] == 3351 and p['feature_manifest_sha256'] == cache_hash
        assert len(h) == 10 and h[-1]['step'] == 3600
        if case.startswith('B'):
            assert cfg == read(ROOT / 'vgcpod_v1_roundB/configs/round_b' / (case + '.json'))
            for name, fingerprint in p['source_sha256'].items():
                assert sha(BS / name) == fingerprint == sha(ROOT / 'vgcpod_v1_roundB' / name)
                audit['source_files_matched'] += 1
            best = helper.checkpoint(run / 'checkpoints/best.pt')
            last = helper.checkpoint(run / 'checkpoints/last.pt')
            assert best['epoch'] == epoch and last['epoch'] == 10 and last['history'] == h
            for group in last['optimizer']['param_groups']:
                expected = cfg['min_lr'] * (cfg['cp_lr_scale'] if group['name'] == 'cp' else 1)
                assert abs(group['lr'] - expected) < 1e-15
            if case in ('B2', 'B3'):
                b = best['model']['Bvis'].double()
                gram = b @ b.T
                norm = b.norm(dim=-1)
                off = ~torch.eye(len(b), dtype=torch.bool)
                cosine = gram / (norm[:, None] * norm[None, :])
                eigen = torch.linalg.eigvalsh(gram).clamp_min(0)
                geometry.append(dict(case=case, epoch=epoch, row_norm_mean=float(norm.mean()),
                    off_diagonal_cosine_mean=float(cosine[off].mean()),
                    mean_row_norm=float(b.mean(0).norm()),
                    orthogonality_loss=float((gram - torch.eye(len(b))).square().mean()),
                    first_eigenvalue_fraction=float(eigen[-1] / eigen.sum())))
        for entry in h:
            t = entry['train']
            hr = dict(case=case, epoch=entry['epoch'], chosen=entry['epoch'] == epoch,
                val_cw=entry['val_cw']['AUC_native'], val_ow=entry.get('val_ow', {}).get('AUC_native'),
                val_attr_unseen=entry.get('val_diagnostics', {}).get('gt_object_attr_acc_unseen'),
                delta_rms=t.get('centered_delta_rms'), clipped_fraction=t.get('clipped_step_fraction'))
            history.append(hr)
            if case.startswith('B'):
                beta = cfg['task_aux_beta']
                errors['loss_identity'] = max(errors['loss_identity'], abs(t['task'] - 1.5*t['pair'] - 1.5*beta/(1+beta)*t['support']))
                eta = cfg['gate_eta']
                gates.append(dict(case=case, epoch=entry['epoch'], chosen=entry['epoch'] == epoch,
                    gamma_mean=t['gamma_mean'], gamma_std=float(np.sqrt(t['gamma_variance'])),
                    gamma_image_std=float(np.sqrt(t['gamma_image_variance'])),
                    normalized_gate_shift=(t['gamma_mean'] - 1)/eta if eta else 0,
                    raw_sigmoid_mean=((t['gamma_mean'] - 1)/eta + 1)/2 if eta else None,
                    gamma_abs_deviation=t['gamma_abs_deviation'],
                    effective_scalar_rho=cfg['rho']*t['gamma_mean'],
                    orth_weighted=cfg['lambda_orth']*t['orth'], task=t['task'],
                    clipped_fraction=t['clipped_step_fraction']))
        for setting in SETTINGS:
            phase, world = setting.split('_')
            mm = m['metrics'][setting]
            cw = train | set(map(tuple, manifest['pairs'][phase]))
            d = helper.index_predictions(run / 'predictions' / (setting + '.npz'), labels, train, cw)
            audit['prediction_files'] += 1
            rec = read(cache / (phase + '_records.json'))
            assert d['image_ids'].tolist() == [r['image'] for r in rec]
            assert d['a'].tolist() == [r['attr'] for r in rec] and d['o'].tolist() == [r['object'] for r in rec]
            n = evaluator.native_curve(d['log_probabilities'], d['truth'], d['seen_classes'])
            exact = evaluator.exact_curve(d['log_probabilities'], d['truth'], d['seen_classes'])
            errors['native_auc'] = max(errors['native_auc'], abs(n['AUC'] - mm['AUC_native']))
            errors['exact_auc'] = max(errors['exact_auc'], abs(exact['AUC'] - mm['AUC_exact']))
            errors['probability_normalization'] = max(errors['probability_normalization'], float(abs(d['probabilities'].sum(-1)-1).max()))
            logits = torch.from_numpy(d['log_probabilities'].copy())
            bias = m['metrics']['val_' + world]['bias_native']
            assert mm['fixed_val_bias'] == bias
            logits[:, ~torch.from_numpy(d['seen_classes'])] += bias
            ok = logits.topk(1, dim=1).indices[:, 0].numpy() == d['truth']
            sm = d['seen_images']
            hm = float(evaluator.hm(ok[sm].mean(), ok[~sm].mean()) * 100)
            errors['hm'] = max(errors['hm'], abs(hm - mm['HM_at_val_bias']))
            exact_logits = torch.from_numpy(d['log_probabilities'].copy())
            exact_logits[:, ~torch.from_numpy(d['seen_classes'])] += m['metrics']['val_' + world]['bias_exact']
            exact_ok = exact_logits.topk(1, dim=1).indices[:, 0].numpy() == d['truth']
            row = dict(case=case, seed=3351, setting=setting, epoch=epoch,
                       **{k:v for k,v in mm.items() if isinstance(v, (int, float))})
            row['HM_at_exact_val_bias'] = float(evaluator.hm(exact_ok[sm].mean(), exact_ok[~sm].mean()) * 100)
            rows.append(row)
            if world == 'ow':
                probability = d['probabilities'].astype(np.float64)
                support = probability[:, d['seen_classes']].sum(-1)
                cond = helper.softmax(d['conditional_logits_gt_object'])
                co = cond.argmax(-1) == d['a']
                op = np.stack([probability[:, d['oi'] == j].sum(-1) for j in range(len(labels['objs']))], axis=-1)
                diagnostics.append(dict(case=case, phase=phase, attr_seen=float(co[sm].mean()*100),
                    attr_unseen=float(co[~sm].mean()*100), attr_all=float(co.mean()*100),
                    seen_correct=int(co[sm].sum()), unseen_correct=int(co[~sm].sum()),
                    support_unseen=float(support[~sm].mean()*100),
                    object_acc=float((op.argmax(-1) == d['o']).mean()*100),
                    object_acc_unseen=float((op.argmax(-1)[~sm] == d['o'][~sm]).mean()*100),
                    unseen_limit_acc=exact['unseen'][-1],
                    unseen_cond_nll=float(-np.log(cond[np.arange(len(cond)), d['a']])[~sm].mean())))
                if case not in ('D0', 'V1'):
                    errors['support_mass'] = max(errors['support_mass'], abs(support[~sm].mean() - mm['support_mass_unseen']))
                predictions[(case, phase)] = dict(correct=co, probabilities=cond, truth=d['a'],
                                                 seen=sm, pairs=d['true_pairs'], calibrated=ok)
                for pair in sorted(set(d['true_pairs'])):
                    mask = np.array([p == pair for p in d['true_pairs']])
                    pair_rows.append(dict(case=case, phase=phase, attribute=pair[0], object=pair[1],
                        seen=pair in train, images=int(mask.sum()), correct=int(co[mask].sum()),
                        cond_acc=float(co[mask].mean()*100), calibrated_pair_acc=float(ok[mask].mean()*100)))
        print('ANALYZED', case, epoch, flush=True)
    comparisons = []
    for high, low in [('B1','A5'),('B2','A5'),('B3','A5'),('B3','A6'),('B3','B2')]:
        for setting in SETTINGS:
            left = next(r for r in rows if r['case'] == high and r['setting'] == setting)
            right = next(r for r in rows if r['case'] == low and r['setting'] == setting)
            comparisons.append(dict(comparison=high+'-'+low, seed=3351, setting=setting,
                **{k:left[k]-right[k] for k in ['AUC_native','AUC_exact','HM_at_val_bias','HM_at_exact_val_bias',
                                               'gt_object_attr_acc','gt_object_attr_acc_unseen','pair_nll_unbiased']}))
    changes, pair_deltas = [], []
    for high, low in [('B1','A5'),('B2','A5'),('B3','A5'),('B3','A6')]:
        for phase in ('val','test'):
            left, right = predictions[(high,phase)], predictions[(low,phase)]
            a, b = left['correct'], right['correct']
            for group in ('all','seen','unseen'):
                mask = np.ones_like(a, dtype=bool) if group == 'all' else (left['seen'] if group == 'seen' else ~left['seen'])
                changes.append(dict(comparison=high+'-'+low, phase=phase, group=group, images=int(mask.sum()),
                    recovered=int((a & ~b & mask).sum()), regressed=int((~a & b & mask).sum()),
                    net_correct=int(a[mask].sum())-int(b[mask].sum()),
                    top1_agreement=float((left['probabilities'].argmax(-1)[mask] == right['probabilities'].argmax(-1)[mask]).mean()*100),
                    mean_total_variation=float((abs(left['probabilities'][mask]-right['probabilities'][mask]).sum(-1)/2).mean())))
            for pair in sorted(set(left['pairs'])):
                mask = np.array([p == pair for p in left['pairs']])
                lrow = next(r for r in pair_rows if r['case']==high and r['phase']==phase and (r['attribute'],r['object'])==pair)
                rrow = next(r for r in pair_rows if r['case']==low and r['phase']==phase and (r['attribute'],r['object'])==pair)
                pair_deltas.append(dict(comparison=high+'-'+low,phase=phase,attribute=pair[0],object=pair[1],
                    seen=pair in train, images=int(mask.sum()), left_acc=lrow['cond_acc'], right_acc=rrow['cond_acc'],
                    accuracy_delta=lrow['cond_acc']-rrow['cond_acc'],correct_delta=lrow['correct']-rrow['correct'],
                    calibrated_pair_delta=lrow['calibrated_pair_acc']-rrow['calibrated_pair_acc']))
    assert errors['native_auc'] < .03 and errors['exact_auc'] < .001 and errors['hm'] < 1e-8
    assert errors['probability_normalization'] < 1e-5 and errors['support_mass'] < 1e-6 and errors['loss_identity'] < 1e-5
    for row in read(BR / 'round_b_summary.json')['runs']:
        for setting in SETTINGS:
            chosen = next(r for r in rows if r['case']==row['case'] and r['setting']==setting)
            assert chosen['AUC_native'] == row[setting+'_AUC_native']
    audit['max_recompute_errors'] = errors
    for name, data in [('all_metrics',rows),('diagnostics',diagnostics),('history',history),('gates',gates),
                        ('gate_geometry',geometry),('per_pair',pair_rows),('comparisons',comparisons),
                        ('conditional_changes',changes),('per_pair_deltas',pair_deltas)]:
        csv_write(name+'.csv',data)
    data = dict(metrics=rows,diagnostics=diagnostics,history=history,gates=gates,geometry=geometry,
                comparisons=comparisons,changes=changes,pair_deltas=pair_deltas,audit=audit)
    write(OUT/'analysis.json',data)
    write(OUT/'audit.json',audit)
    plots(data)
    print(json.dumps(dict(audit=audit,selected_gates=[r for r in gates if r['chosen']],geometry=geometry),indent=2))


def plots(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.dpi':140})
    cases = ['A5','A6','B1','B2','B3']
    colors = ['#5d9b91','#8983a8','#8b9ba9','#367fa2','#145b7c']
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    for ax, setting, title in zip(axes,['val_cw','test_cw','test_ow'],['VAL CW','TEST CW','TEST OW']):
        values = [next(r['AUC_native'] for r in data['metrics'] if r['case']==c and r['setting']==setting) for c in cases]
        ax.bar(cases,values,color=colors)
        ax.set_ylim(0,50 if setting=='val_cw' else 40)
        ax.set_title(title)
        ax.set_ylabel('AUC_native (area x 100)')
        for k,value in enumerate(values): ax.text(k,value+.4,f'{value:.2f}',ha='center',fontsize=9)
    fig.suptitle('Seed 3351: paired A5/A6 controls and B pilot')
    fig.tight_layout();fig.savefig(OUT/'figures/01_results.png');plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(12,7))
    color = {'A5':'#5d9b91','B1':'#8b9ba9','B2':'#367fa2','B3':'#145b7c'}
    for c in color:
        h = [r for r in data['history'] if r['case']==c]
        for ax,key,title in zip(axes[0],['val_cw','val_ow'],['VAL CW learning curve','VAL OW learning curve']):
            ax.plot([r['epoch'] for r in h],[r[key] for r in h],label=c,color=color[c])
            ch = next(r for r in h if r['chosen'])
            ax.scatter(ch['epoch'],ch[key],color=color[c],s=30)
            ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel('AUC_native');ax.legend()
    for c in ['B2','B3']:
        g = [r for r in data['gates'] if r['case']==c]
        axes[1,0].plot([r['epoch'] for r in g],[r['normalized_gate_shift'] for r in g],label=c,color=color[c])
        axes[1,1].plot([r['epoch'] for r in g],[r['gamma_image_std'] for r in g],label=c,color=color[c])
    axes[1,0].axhline(1,color='#8b9ba9',ls='--',lw=1)
    axes[1,0].set_ylim(.8,1.01);axes[1,0].set_title('Training-batch mean gate shift / eta')
    axes[1,0].set_ylabel('(mean gamma - 1) / eta')
    axes[1,1].set_title('Training-batch cross-image gate variation');axes[1,1].set_ylabel('sqrt(mean batch image variance)')
    for ax in axes[1]: ax.set_xlabel('Epoch');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'figures/02_learning_and_gate.png');plt.close(fig)
    candidates = [r for r in data['pair_deltas'] if r['comparison']=='B3-A5' and r['phase']=='test' and not r['seen']]
    candidates = sorted(candidates,key=lambda r:r['correct_delta'])
    candidates = [r for r in candidates if r['correct_delta']!=0]
    fig,ax = plt.subplots(figsize=(10,4.5))
    vals=[r['correct_delta'] for r in candidates]
    labels=[r['attribute']+' / '+r['object'].replace('Shoes.','') for r in candidates]
    ax.barh(labels,vals,color=['#b44d5c' if v<0 else '#5d9b91' for v in vals])
    ax.axvline(0,color='#333',lw=1);ax.set_xlabel('B3 - A5: conditional correct image count')
    ax.set_title('TEST unseen compositions, seed 3351')
    fig.tight_layout();fig.savefig(OUT/'figures/03_unseen_changes.png');plt.close(fig)


if __name__ == '__main__':
    main()
