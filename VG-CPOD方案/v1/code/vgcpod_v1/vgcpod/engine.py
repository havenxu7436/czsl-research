import csv
import math
import os
import platform
import random
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


def make_model(cache, cfg, seed, device, verify=True):
    manifest, text = load_cache(cache, verify=verify)
    if abs(manifest['tau_v'] - cfg['tau_v']) > 1e-12:
        raise ValueError('Cached tau_v and training config differ')
    model = VGCPOD(text['ea'], text['eo'], text['ec'], cfg, seed).to(device)
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
    optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], betas=tuple(cfg['adam_betas']),
                                 eps=cfg['adam_eps'], weight_decay=cfg['weight_decay'])
    steps_per_epoch = math.ceil(manifest['counts']['train'] / cfg['batch_size'])
    total_steps = steps_per_epoch * cfg['epochs']
    start_epoch, step, best, history = 0, 0, -math.inf, []
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
        restore_rng(state['rng'])
    elif args.resume:
        print('RESTART_FIRST_EPOCH_FROM_MATCHED_SEED', run, flush=True)
    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        tic = time.time()
        sums, count = {}, 0
        for g, fa, fo, attrs, objs in loader(args.cache, 'train', cfg, epoch=epoch, seed=args.seed):
            # t=step+1 gives cosine endpoints eta_1=lr and eta_T=min_lr.
            eta = cfg['min_lr'] + 0.5 * (cfg['lr'] - cfg['min_lr']) * (
                1 + math.cos(math.pi * step / max(total_steps - 1, 1)))
            for group in optimizer.param_groups:
                group['lr'] = eta
            optimizer.zero_grad(set_to_none=True)
            attrs, objs = attrs.to(args.device), objs.to(args.device)
            out = model(g.to(args.device), fa.to(args.device), fo.to(args.device))
            losses = model.objective(out, attrs, objs, mask)
            if not torch.isfinite(losses['total']):
                raise FloatingPointError('Nonfinite loss at epoch %d step %d' % (epoch + 1, step))
            losses['total'].backward()
            norm = clip_total_norm(parameters, cfg['gradient_clip'])
            optimizer.step()
            batch = len(attrs)
            logs = {key: float(value.detach()) for key, value in losses.items()}
            logs.update(gradient_norm=float(norm), gamma_mean=float(out['gamma'].mean()),
                        gamma_variance=float(out['gamma'].var(unbiased=False)))
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
        selected = metrics['AUC_native'] > best
        if selected:
            best = metrics['AUC_native']
        entry = {'epoch': epoch + 1, 'step': step, 'lr_last': eta,
                 'train': {k: v / count for k, v in sums.items()}, 'val_cw': metrics,
                 'selected': selected, 'seconds': time.time() - tic}
        history.append(entry)
        state = {'epoch': epoch + 1, 'step': step, 'best_auc': best, 'history': history,
                 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state(),
                 'protocol_sha256': digest(current), 'config': cfg, 'seed': args.seed}
        if selected:
            save_torch(run / 'checkpoints' / 'best.pt', state)
        save_torch(last, state)
        write_json(run / 'history.json', history)
        print('EPOCH', epoch + 1, cfg['variant'], 'seed', args.seed,
              'loss', round(entry['train']['total'], 6), 'VAL_CW_AUC_native', round(metrics['AUC_native'], 6),
              'best', round(best, 6), 'seconds', round(entry['seconds'], 1), flush=True)
    evaluate_run(args.cache, run, args.device)
    print('TRAIN_COMPLETE', run, flush=True)


def write_evaluation(cache, run, manifest, cfg, phase, joint, cond, attrs, objs, world, fixed_bias):
    metrics, curves, candidates, log_probs = evaluate_scores(joint, cond, attrs, objs,
                manifest['labels'], manifest['pairs'], phase, world, fixed_bias=fixed_bias)
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


def evaluate_run(cache, run, device):
    run = Path(run)
    checkpoint = load_torch(run / 'checkpoints' / 'best.pt')
    cfg, seed = checkpoint['config'], checkpoint['seed']
    seed_all(seed)
    model, manifest = make_model(cache, cfg, seed, device)
    check_run_identity(cache, run, checkpoint)
    model.load_state_dict(checkpoint['model'])
    results = {'selected_epoch': checkpoint['epoch'], 'selection': 'VAL CW AUC_native',
               'checkpoint_sha256': sha(run / 'checkpoints' / 'best.pt'), 'metrics': {}}
    biases = {}
    for phase in ('val', 'test'):
        joint, cond, attrs, objs = predict(model, cache, phase, cfg, device)
        for world in ('cw', 'ow'):
            fixed = biases.get(world)
            if phase == 'val':
                provisional, _, _, _ = evaluate_scores(joint, cond, attrs, objs, manifest['labels'],
                                                        manifest['pairs'], phase, world)
                fixed = biases[world] = provisional['bias_native']
            results['metrics'][phase + '_' + world] = write_evaluation(cache, run, manifest, cfg,
                                    phase, joint, cond, attrs, objs, world, fixed)
    write_json(run / 'metrics.json', results)
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
