"""Evaluate dynamic vs TRAIN-mean gates with one fixed native/exact checkpoint.

The source root is selected before importing the archived training runtime.
This evaluator lives outside that runtime; the original run is read only.
"""
import argparse
import hashlib
import sys
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_gate_moments(model, batches, device):
    """All TRAIN images contribute to every object query, without labels."""
    import torch
    count, total, squares = 0, None, None
    model.eval()
    with torch.no_grad():
        for g, fa, fo, _, _ in batches:
            gamma = model(g.to(device), fa.to(device), fo.to(device))['gamma'].double()
            batch_sum, batch_square = gamma.sum(0), gamma.square().sum(0)
            total = batch_sum if total is None else total + batch_sum
            squares = batch_square if squares is None else squares + batch_square
            count += len(gamma)
    if not count:
        raise ValueError('TRAIN cache has no images')
    mean = total / count
    return mean, (squares / count - mean.square()).clamp_min(0), count


def mean_gate_model(model, mean):
    import torch
    from torch import nn

    if model.readout_feature != 'global' or not model.gate_center_modes or not model.dynamic:
        raise ValueError('This intervention requires a global readout and centered dynamic gates')

    class FixedTrainMean(nn.Module):
        def __init__(self):
            super().__init__()
            self.dynamic_model = model
            self.register_buffer('mean_gamma', mean.to(device=model.Wc.device, dtype=model.Wc.dtype))

        def forward(self, g, fa, fo):
            m = self.dynamic_model
            out = m(g, fa, fo)
            u, v = m.ea @ m.Au.T, m.eo @ m.Av.T
            gamma = self.mean_gamma[None].expand(len(g), -1, -1)
            if getattr(m, 'cp_readout_mode', 'shared') == 'separate':
                dynamic_modes = (g[:, None, :] @ m.Wd) * (gamma - 1) * v[None]
                scale = m.cfg['rho'] * m.cfg['kappa']
                static = out['cp_static']
                dynamic = scale * torch.einsum('bor,ir->boi', dynamic_modes, u)
                cond = out['base'] + static + dynamic
                out.update(cp_static=static, cp_dynamic=dynamic)
            else:
                modes = (g[:, None, :] @ m.Wc) * gamma * v[None]
                cond = out['base'] + m.cfg['rho'] * m.cfg['kappa'] * torch.einsum('bor,ir->boi', modes, u)
            log_a = cond.log_softmax(-1)
            return dict(out, gamma=gamma, cond=cond, log_a=log_a, joint=out['log_o'][:, :, None]+log_a)

    return FixedTrainMean()


def evaluate_condition(model, cache, out, manifest, cfg, device, engine):
    metrics, biases = {}, {}
    for phase in ('val', 'test'):
        joint, cond, attrs, objs = engine.predict(model, cache, phase, cfg, device)
        for world in ('cw', 'ow'):
            if phase == 'val':
                provisional, _, _, _ = engine.evaluate_scores(joint, cond, attrs, objs,
                    manifest['labels'], manifest['pairs'], phase, world)
                biases[world] = provisional['bias_native']
            metrics[phase+'_'+world] = engine.write_evaluation(cache, out, manifest, cfg,
                phase, joint, cond, attrs, objs, world, biases[world])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--selection', choices=['native', 'exact'], default='native')
    args = parser.parse_args()
    args.run, args.out = args.run.resolve(), args.out.resolve()
    if args.out == args.run or args.run in args.out.parents:
        parser.error('--out must be outside the original training run')
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    from vgcpod import engine
    from vgcpod.common import load_torch, read_json, seed_all, sha, source_hashes, write_json

    checkpoint_name = 'best_exact.pt' if args.selection == 'exact' else 'best.pt'
    best = args.run / 'checkpoints' / checkpoint_name
    checkpoint = load_torch(best)
    engine.check_run_identity(args.cache, args.run, checkpoint)
    cfg, seed = checkpoint['config'], checkpoint['seed']
    identity = dict(checkpoint_sha256=sha(best), selection=args.selection, source_sha256=source_hashes(),
        evaluator_sha256=file_sha(__file__), feature_manifest_sha256=sha(args.cache/'features_manifest.json'))
    target = args.out / 'metrics.json'
    if target.exists() and read_json(target).get('identity') == identity:
        print('MEAN_GATE_ALREADY_COMPLETE', cfg.get('experiment_id'), seed, args.out, flush=True)
        return
    seed_all(seed)
    model, manifest = engine.make_model(args.cache, cfg, seed, args.device,
        gate_mean=checkpoint['model'].get('gate_fo_mean'))
    model.load_state_dict(checkpoint['model'])
    mean, variance, count = train_gate_moments(model, engine.loader(args.cache, 'train', cfg), args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out/'train_mean_gamma.npz', mean_gamma=mean.cpu().numpy(),
        image_variance=variance.cpu().numpy(), train_images=np.array(count),
        object_names=np.asarray(manifest['labels']['objs']))
    conditions = {}
    for name, evaluated in [('dynamic', model), ('train_mean', mean_gate_model(model, mean))]:
        conditions[name] = evaluate_condition(evaluated, args.cache, args.out/name, manifest, cfg, args.device, engine)
        write_json(args.out/name/'metrics.json', dict(selected_epoch=checkpoint['epoch'],
            identity=identity, condition=name, optimizer_steps=0, metrics=conditions[name]))
    keys = ('AUC_native','AUC_exact','HM_at_val_bias','gt_object_attr_acc',
            'gt_object_attr_acc_unseen','support_mass_unseen')
    deltas = {setting:{key:conditions['dynamic'][setting][key]-conditions['train_mean'][setting][key]
        if conditions['dynamic'][setting][key] is not None and conditions['train_mean'][setting][key] is not None
        else None for key in keys} for setting in conditions['dynamic']}
    reference = args.run/'exact_selection' if args.selection == 'exact' else args.run
    original = read_json(reference/'metrics.json')['metrics']
    drift = {setting:conditions['dynamic'][setting]['AUC_native']-original[setting]['AUC_native']
             for setting in conditions['dynamic']}
    write_json(target, dict(schema='vgcpod_mean_gate_v1', case=cfg['experiment_id'], seed=seed,
        selected_epoch=checkpoint['epoch'], selection=args.selection, checkpoint_name=checkpoint_name,
        identity=identity, source_root=str(args.source_root.resolve()),
        run_dir=str(args.run), config=cfg, optimizer_steps=0, train_images=count,
        mean_shape=list(mean.shape), mean_mode_average_max_error=float((mean.mean(-1)-1).abs().max()),
        mean_image_variance=float(variance.mean()),
        calibration='Each condition selects CW/OW biases on its own VAL; TEST uses those fixed biases',
        mean_origin='All TRAIN images, every object query; no labels, VAL, or TEST used in moments',
        metrics=conditions, dynamic_minus_train_mean=deltas, dynamic_minus_saved_auc=drift))
    print('MEAN_GATE_COMPLETE', cfg['experiment_id'], seed, args.out, flush=True)


if __name__ == '__main__':
    main()
