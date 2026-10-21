import csv
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .common import digest, load_torch, read_json, save_torch, seed_all, sha, source_hashes, write_json
from .data import FeatureDataset, load_cache, mask_from_pairs
from .metrics import evaluate_scores
from .model import VGCPOD


@torch.no_grad()
def clip_total_norm(parameters, maximum):
    gradients = [p.grad for p in parameters if p.grad is not None]
    norm = torch.stack([g.norm(2) for g in gradients]).norm(2)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient norm')
    coefficient = (maximum / (norm + 1e-12)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient)
    return norm


def make_optimizer(model, cfg):
    """Separate adapter and CP step sizes while preserving parameter order."""
    base_names = {'Bo', 'Co', 'Bc', 'Cc'}
    groups = []
    for label, scale in [('adapter', 1.0), ('cp', cfg.get('cp_lr_scale', 1.0))]:
        parameters = [p for name, p in model.named_parameters() if p.requires_grad
                      and ((name in base_names) == (label == 'adapter'))]
        if parameters:
            groups.append({'params': parameters, 'name': label, 'lr_scale': scale,
                           'lr': cfg['lr'] * scale})
    return torch.optim.AdamW(groups, betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'],
                             weight_decay=cfg['weight_decay'])


def set_learning_rates(optimizer, cfg, step, total_steps):
    eta = cfg['min_lr'] + 0.5 * (cfg['lr'] - cfg['min_lr']) * (
        1 + math.cos(math.pi * step / max(total_steps - 1, 1)))
    for group in optimizer.param_groups:
        group['lr'] = eta * group['lr_scale']
    return eta


def distribution_diagnostics(joint, cond, attrs, objs, labels, pairs):
    """Full joint probabilities before CW restriction or calibration bias."""
    mask = mask_from_pairs(labels, pairs['train']).numpy()
    seen_images = mask[objs, attrs]
    support = np.exp(joint.astype(np.float64))[:, mask].sum(-1)
    attr_ok = cond[np.arange(len(attrs)), objs].argmax(-1) == attrs
    result = {'support_mass_all': float(support.mean())}
    for name, sm in [('seen', seen_images), ('unseen', ~seen_images)]:
        result['support_mass_' + name] = float(support[sm].mean()) if sm.any() else None
        result['gt_object_attr_acc_' + name] = float(attr_ok[sm].mean() * 100) if sm.any() else None
    return result


def make_model(cache, cfg, seed, device, verify=True, gate_mean=None):
    manifest, text = load_cache(cache, verify=verify)
    if abs(manifest['tau_v'] - cfg['tau_v']) > 1e-12:
        raise ValueError('Cached tau_v and training config differ')
    if cfg.get('gate_input') in ('train_residual', 'train_centered') and gate_mean is None:
        from .gate_features import load_gate_mean
        gate_mean = load_gate_mean(cache, cfg, manifest)
    model = VGCPOD(text['ea'], text['eo'], text['ec'], cfg, seed, gate_mean=gate_mean).to(device)
    return model, manifest


def loader(cache, phase, cfg, epoch=None, seed=0, limit=None):
    data = FeatureDataset(cache, phase)
    if limit is not None:
        data = Subset(data, range(min(limit, len(data))))
    gen = torch.Generator().manual_seed(seed + (0 if epoch is None else epoch * 1000003))
    return DataLoader(data, batch_size=cfg['batch_size'] if epoch is not None else cfg['eval_batch_size'],
                      shuffle=epoch is not None, num_workers=cfg['workers'], generator=gen,
                      drop_last=False, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict(model, cache, phase, cfg, device):
    model.eval()
    joint, cond, attrs, objs = [], [], [], []
    for g, fa, fo, a, o in loader(cache, phase, cfg):
        out = model(g.to(device), fa.to(device), fo.to(device))
        joint.append(out['joint'].cpu().numpy())
        cond.append(out['cond'].cpu().numpy())
        attrs.append(a.numpy())
        objs.append(o.numpy())
    return np.concatenate(joint), np.concatenate(cond), np.concatenate(attrs), np.concatenate(objs)


def protocol(cache, cfg, seed, model):
    return {'schema': 'vgcpod_run_v1', 'idea': 'VG-CPOD', 'version': 'v1',
            'config': cfg, 'seed': seed, 'feature_manifest_sha256': sha(Path(cache) / 'features_manifest.json'),
            'source_sha256': source_hashes(), 'trainable_parameters': model.trainable_count(),
            'axes': ['batch', 'object', 'attribute'],
            'selection': 'first strict maximum VAL CW AUC_native; TEST never used for selection',
            'v6_rule': 'roll gate inputs by +1 within each batch; evaluation metadata order, batch size fixed',
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                            'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
                            'device': str(next(model.parameters()).device),
                            'gpu': torch.cuda.get_device_name() if next(model.parameters()).is_cuda else None},
            'cache_path_at_run': str(Path(cache).resolve())}


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def task_decomposition(logs, beta):
    """Detached scalar diagnostics; preserve the objective's operation order."""
    support_weight = 1.5 * beta / (1.0 + beta)
    pair_weighted = 1.5 * logs['pair']
    support_weighted = support_weight * logs['support']
    return dict(support_weight=support_weight, pair_weighted=pair_weighted,
                support_weighted=support_weighted,
                task_reconstruction_error=abs(logs['task'] - pair_weighted - support_weighted))


def train(args, cfg):
    seed_all(args.seed)
    run = Path(args.out) / cfg['dataset'] / cfg['variant'] / ('seed%d' % args.seed)
    run.mkdir(parents=True, exist_ok=True)
    model, manifest = make_model(args.cache, cfg, args.seed, args.device)
    current = protocol(args.cache, cfg, args.seed, model)
    p = run / 'protocol.json'
    if p.exists():
        if read_json(p) != current:
            raise ValueError('Run identity differs; use a new --out directory')
        if not args.resume:
            raise FileExistsError('Run exists; use --resume: ' + str(run))
    else:
        if args.resume:
            raise FileNotFoundError('No protocol to resume: ' + str(run))
        write_json(p, current)
    mask = mask_from_pairs(manifest['labels'], manifest['pairs']['train']).to(args.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, cfg)
    steps_per_epoch = math.ceil(manifest['counts']['train'] / cfg['batch_size'])
    total_steps = steps_per_epoch * cfg['epochs']
    start_epoch, step, best, best_exact, history = 0, 0, -math.inf, -math.inf, []
    last = run / 'checkpoints' / 'last.pt'
    recovery = last if last.exists() else run / 'checkpoints' / 'best.pt'
    if args.resume and recovery.exists():
        state = load_torch(recovery)
        if state['protocol_sha256'] != digest(current):
            raise ValueError('Checkpoint protocol differs')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for state_item in optimizer.state.values():
            for key, value in state_item.items():
                if torch.is_tensor(value) and key != 'step':
                    state_item[key] = value.to(args.device)
        start_epoch, step, best, history = state['epoch'], state['step'], state['best_auc'], state['history']
        best_exact = state['best_exact_auc']
        restore_rng(state['rng'])
    elif args.resume:
        print('RESTART_FIRST_EPOCH_FROM_MATCHED_SEED', run, flush=True)
    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        tic = time.time()
        sums, count, clipped_steps, epoch_steps = {}, 0, 0, 0
        for g, fa, fo, attrs, objs in loader(args.cache, 'train', cfg, epoch=epoch, seed=args.seed):
            eta = set_learning_rates(optimizer, cfg, step, total_steps)
            optimizer.zero_grad(set_to_none=True)
            attrs, objs = attrs.to(args.device), objs.to(args.device)
            out = model(g.to(args.device), fa.to(args.device), fo.to(args.device))
            losses = model.objective(out, attrs, objs, mask)
            if not torch.isfinite(losses['total']):
                raise FloatingPointError('Nonfinite loss at epoch %d step %d' % (epoch + 1, step))
            losses['total'].backward()
            norm = clip_total_norm(parameters, cfg['gradient_clip'])
            clipped_steps += int(norm > cfg['gradient_clip'])
            epoch_steps += 1
            optimizer.step()
            batch = len(attrs)
            logs = {key: float(value.detach()) for key, value in losses.items()}
            logs.update(task_decomposition(logs, cfg.get('task_aux_beta', 0.5)))
            logs.update(gradient_norm=float(norm), gamma_mean=float(out['gamma'].mean()),
                        gamma_variance=float(out['gamma'].var(unbiased=False)))
            logs['gamma_image_variance'] = float(out['gamma'].var(dim=0, unbiased=False).mean())
            logs['gamma_abs_deviation'] = float((out['gamma'] - 1).abs().mean())
            logs['gamma_mode_mean_deviation'] = float((out['gamma'].mean(-1) - 1).abs().mean())
            delta = (out['cond'] - out['base']).detach()
            logs['centered_delta_mse'] = float((delta - delta.mean(-1, keepdim=True)).square().mean())
            for name in ('cp_static', 'cp_dynamic'):
                if name in out:
                    component = out[name].detach()
                    logs[name + '_mse'] = float(component.square().mean())
                    logs[name + '_centered_mse'] = float(
                        (component - component.mean(-1, keepdim=True)).square().mean())
            logs['support_mass'] = float(out['joint'].detach().exp()[:, mask].sum(-1).mean())
            if 'pi' in out:
                pi = out['pi']
                logs['partner_entropy'] = float(-(pi * pi.clamp_min(1e-30).log()).sum(-1).mean())
            for key, value in logs.items():
                sums[key] = sums.get(key, 0.0) + value * batch
            count += batch
            step += 1
        joint, cond, attrs, objs = predict(model, args.cache, 'val', cfg, args.device)
        metrics, _, _, _ = evaluate_scores(joint, cond, attrs, objs, manifest['labels'],
                                           manifest['pairs'], 'val', 'cw')
        val_ow, _, _, _ = evaluate_scores(joint, cond, attrs, objs, manifest['labels'],
                                          manifest['pairs'], 'val', 'ow')
        diagnostics = distribution_diagnostics(joint, cond, attrs, objs, manifest['labels'], manifest['pairs'])
        selected = metrics['AUC_native'] > best
        selected_exact = metrics['AUC_exact'] > best_exact
        if selected:
            best = metrics['AUC_native']
        if selected_exact:
            best_exact = metrics['AUC_exact']
        entry = {'epoch': epoch + 1, 'step': step, 'lr_last': eta,
                 'train': {k: v / count for k, v in sums.items()}, 'val_cw': metrics,
                 'val_ow': val_ow, 'val_diagnostics': diagnostics,
                 'lr_groups': {g['name']: g['lr'] for g in optimizer.param_groups},
                 'selected': selected, 'selected_exact': selected_exact, 'seconds': time.time() - tic}
        entry['train'].update(clipped_steps=clipped_steps, batches=epoch_steps,
                              clipped_step_fraction=clipped_steps / epoch_steps,
                              centered_delta_rms=math.sqrt(entry['train']['centered_delta_mse']))
        for name in ('cp_static', 'cp_dynamic', 'cp_static_centered', 'cp_dynamic_centered'):
            if name + '_mse' in entry['train']:
                entry['train'][name + '_rms'] = math.sqrt(entry['train'][name + '_mse'])
        history.append(entry)
        state = {'epoch': epoch + 1, 'step': step, 'best_auc': best, 'best_exact_auc': best_exact, 'history': history,
                 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state(),
                 'protocol_sha256': digest(current), 'config': cfg, 'seed': args.seed}
        if selected:
            native = run / 'checkpoints' / 'best_native.pt'
            save_torch(native, state)
            alias = run / 'checkpoints' / 'best.pt'
            writing = alias.with_name(alias.name + '.writing')
            shutil.copyfile(str(native), str(writing))
            os.replace(str(writing), str(alias))
        if selected_exact:
            save_torch(run / 'checkpoints' / 'best_exact.pt', state)
        save_torch(last, state)
        write_json(run / 'history.json', history)
        print('EPOCH', epoch + 1, cfg.get('experiment_id', cfg['variant']), cfg['variant'], 'seed', args.seed,
              'loss', round(entry['train']['total'], 6), 'VAL_CW_AUC_native', round(metrics['AUC_native'], 6),
              'VAL_OW_AUC_native', round(val_ow['AUC_native'], 6),
              'VAL_unseen_attr', diagnostics['gt_object_attr_acc_unseen'],
              'VAL_unseen_support', diagnostics['support_mass_unseen'],
              'best', round(best, 6), 'seconds', round(entry['seconds'], 1), flush=True)
    evaluate_run(args.cache, run, args.device)
    evaluate_run(args.cache, run, args.device, checkpoint_name='best_exact.pt', out=run / 'exact_selection')
    print('TRAIN_COMPLETE', run, flush=True)


def write_evaluation(cache, run, manifest, cfg, phase, joint, cond, attrs, objs, world, fixed_bias):
    metrics, curves, candidates, log_probs = evaluate_scores(joint, cond, attrs, objs,
                manifest['labels'], manifest['pairs'], phase, world, fixed_bias=fixed_bias)
    metrics.update(distribution_diagnostics(joint, cond, attrs, objs, manifest['labels'], manifest['pairs']))
    key = phase + '_' + world
    folder = Path(run) / 'predictions'
    folder.mkdir(parents=True, exist_ok=True)
    rows = read_json(Path(cache) / (phase + '_records.json'))
    np.savez_compressed(folder / (key + '.npz'), image_ids=np.array([r['image'] for r in rows]),
                        candidate_pairs=candidates, attribute_truth=attrs, object_truth=objs,
                        log_probabilities=log_probs, probabilities=np.exp(log_probs),
                        conditional_logits_gt_object=cond[np.arange(len(attrs)), objs])
    write_json(Path(run) / 'curves' / (key + '.json'), curves)
    return metrics


def evaluate_run(cache, run, device, checkpoint_name='best.pt', out=None):
    run = Path(run)
    checkpoint_path = run / 'checkpoints' / checkpoint_name
    checkpoint = load_torch(checkpoint_path)
    destination = Path(out) if out is not None else run
    cfg, seed = checkpoint['config'], checkpoint['seed']
    seed_all(seed)
    model, manifest = make_model(cache, cfg, seed, device, gate_mean=checkpoint['model'].get('gate_fo_mean'))
    check_run_identity(cache, run, checkpoint)
    model.load_state_dict(checkpoint['model'])
    results = {'selected_epoch': checkpoint['epoch'],
               'selection': 'VAL CW AUC_exact' if checkpoint_name == 'best_exact.pt' else 'VAL CW AUC_native',
               'checkpoint_name': checkpoint_name, 'checkpoint_sha256': sha(checkpoint_path), 'metrics': {}}
    biases = {}
    for phase in ('val', 'test'):
        joint, cond, attrs, objs = predict(model, cache, phase, cfg, device)
        for world in ('cw', 'ow'):
            fixed = biases.get(world)
            if phase == 'val':
                provisional, _, _, _ = evaluate_scores(joint, cond, attrs, objs, manifest['labels'],
                                                        manifest['pairs'], phase, world)
                fixed = biases[world] = provisional['bias_native']
            results['metrics'][phase + '_' + world] = write_evaluation(cache, destination, manifest, cfg,
                                    phase, joint, cond, attrs, objs, world, fixed)
    write_json(destination / 'metrics.json', results)
    return results


def check_run_identity(cache, run, checkpoint):
    archived_protocol = read_json(Path(run) / 'protocol.json')
    if checkpoint['protocol_sha256'] != digest(archived_protocol):
        raise ValueError('Checkpoint identity differs from its protocol')
    if archived_protocol['feature_manifest_sha256'] != sha(Path(cache) / 'features_manifest.json'):
        raise ValueError('Evaluation cache differs from the training cache')
    if archived_protocol['source_sha256'] != source_hashes():
        raise ValueError('Evaluation source differs from training; keep the archived source')


def baselines(args, cfg):
    seed_all(3351)
    cfg = dict(cfg, variant='V0')
    model, manifest = make_model(args.cache, cfg, 3351, args.device)
    model.requires_grad_(False)
    for name in ('name_clip', 'chain_zeroshot'):
        run = Path(args.out) / cfg['dataset'] / name
        run.mkdir(parents=True, exist_ok=True)
        results, biases = {}, {}
        for phase in ('val', 'test'):
            joint, cond, attrs, objs = predict(model, args.cache, phase, cfg, args.device)
            if name == 'name_clip':
                # V0 adapter output is exactly zero at initialization.
                joint = cond.copy()
            for world in ('cw', 'ow'):
                if phase == 'val':
                    metric, _, _, _ = evaluate_scores(joint, cond, attrs, objs, manifest['labels'],
                                                        manifest['pairs'], phase, world)
                    biases[world] = metric['bias_native']
                results[phase + '_' + world] = write_evaluation(args.cache, run, manifest, cfg,
                                            phase, joint, cond, attrs, objs, world, biases[world])
        write_json(run / 'metrics.json', {'optimizer_steps': 0, 'metrics': results})
        baseline_protocol = protocol(args.cache, cfg, 3351, model)
        baseline_protocol.update(baseline=name, optimizer_steps=0, selection='frozen zero-shot; no checkpoint selection')
        write_json(run / 'protocol.json', baseline_protocol)
        print('BASELINE_COMPLETE', name, flush=True)


def smoke(args, cfg):
    seed_all(args.seed)
    cfg = dict(cfg, variant='V5', workers=0)
    model, manifest = make_model(args.cache, cfg, args.seed, args.device)
    mask = mask_from_pairs(manifest['labels'], manifest['pairs']['train']).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'])
    report = {}
    # Real cached TRAIN images, both 4 examples and one complete batch.
    for size in (4, cfg['batch_size']):
        g, fa, fo, attrs, objs = next(iter(loader(args.cache, 'train', cfg, limit=size)))
        g, fa, fo, attrs, objs = [x.to(args.device) for x in (g, fa, fo, attrs, objs)]
        optimizer.zero_grad(set_to_none=True)
        out = model(g, fa, fo)
        losses = model.objective(out, attrs, objs, mask)
        p = out['joint'].exp()
        norm_error = float((p.sum((1, 2)) - 1).abs().max())
        marginal_error = float((p.sum(-1) - out['log_o'].exp()).abs().max())
        if norm_error >= 1e-5 or marginal_error >= 1e-5:
            raise AssertionError('Probability normalization failed')
        if size == 4 and float((out['base'] - out['cond']).abs().max()) != 0:
            raise AssertionError('Zero initialized CP fails D0 recovery')
        losses['total'].backward()
        wc_grad = float(model.Wc.grad.norm())
        if not wc_grad > 0:
            raise AssertionError('Wc has no task gradient')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        out = model(g, fa, fo)
        model.objective(out, attrs, objs, mask)['total'].backward()
        norms = {name: float(getattr(model, name).grad.norm()) for name in ('Au', 'Av', 'Bvis')}
        if not all(value > 0 and math.isfinite(value) for value in norms.values()):
            raise AssertionError('Updated CP failed to propagate gradients')
        report[str(size)] = {'normalization_max_error': norm_error, 'object_marginal_max_error': marginal_error,
                             'Wc_first_gradient_norm': wc_grad, 'second_gradients': norms}
    write_json(Path(args.out) / 'real_train_smoke.json', report)
    print('REAL_TRAIN_SMOKE_PASSED', report, flush=True)


def summarize(args):
    root = Path(args.out)
    rows = []
    for p in sorted(root.glob('ut-zap50k/V*/seed*/metrics.json')):
        result = read_json(p)
        variant, seed = p.parent.parent.name, int(p.parent.name[4:])
        for setting, metrics in result['metrics'].items():
            rows.append(dict(variant=variant, seed=seed, setting=setting,
                             selected_epoch=result['selected_epoch'], **{k: v for k, v in metrics.items()
                                  if not isinstance(v, dict)}))
    if not rows:
        raise FileNotFoundError('No completed run metrics under ' + str(root))
    fields = list(rows[0])
    with open(root / 'summary.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped, paired = {}, {}
    keys = ('AUC_native', 'AUC_exact', 'HM_at_val_bias', 'pair_nll_unbiased', 'gt_object_attr_acc')
    for variant in sorted({r['variant'] for r in rows}):
        for setting in ('val_cw', 'val_ow', 'test_cw', 'test_ow'):
            group = [r for r in rows if r['variant'] == variant and r['setting'] == setting]
            grouped[variant + '/' + setting] = {k: {'mean': float(np.mean([r[k] for r in group])),
                 'sd': float(np.std([r[k] for r in group], ddof=1)) if len(group) > 1 else None,
                 'seeds': [r['seed'] for r in group]} for k in keys}
    for high, low in [('V1L', 'V1'), ('V2', 'V1L'), ('V5', 'V4'), ('V5', 'V2'), ('V2', 'V3'), ('V2', 'V6'), ('V5', 'V7')]:
        for setting in ('val_cw', 'val_ow', 'test_cw', 'test_ow'):
            left = {r['seed']: r for r in rows if r['variant'] == high and r['setting'] == setting}
            right = {r['seed']: r for r in rows if r['variant'] == low and r['setting'] == setting}
            seeds = sorted(left.keys() & right.keys())
            if seeds:
                paired[high + '-' + low + '/' + setting] = {k: {'seeds': seeds,
                        'deltas': [left[s][k] - right[s][k] for s in seeds],
                        'sd_delta': float(np.std([left[s][k] - right[s][k] for s in seeds], ddof=1)) if len(seeds) > 1 else None,
                        'mean_delta': float(np.mean([left[s][k] - right[s][k] for s in seeds]))} for k in keys}
    write_json(root / 'summary.json', {'by_variant': grouped, 'paired_deltas': paired})
    print('SUMMARY_WRITTEN', root / 'summary.csv', root / 'summary.json', flush=True)
