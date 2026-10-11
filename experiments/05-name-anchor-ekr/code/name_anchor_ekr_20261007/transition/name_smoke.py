"""GPU smoke against the real A_unit checkpoint; consumes only two TRAIN images."""
from pathlib import Path
import torch
import torch.nn.functional as F
from .common import candidates, move, seed_all, sha, state_digest, write_json
from .runtime import base_model, read_cfg, trainset
from .name_runtime import BranchModel
from .name_plan import validate_config


def smoke(cfg_path, output):
    cfg = read_cfg(cfg_path)
    validate_config(cfg, require_replay=True)
    out = Path(output)
    if out.exists():
        raise FileExistsError('Preserve the existing smoke report; use a new output')
    seed_all(int(cfg.TRAIN.seed))
    ds = trainset(cfg)
    batch = move(next(iter(torch.utils.data.DataLoader(ds, batch_size=2,
                         shuffle=False, num_workers=0))), 'cuda:0')
    ref_cfg = read_cfg(cfg.MODEL.branch_anchor_config)
    reference = base_model(ds, ref_cfg, cfg.MODEL.branch_anchor)
    reference.eval()
    ai, oi, _ = candidates(reference, ds, train=True)
    with torch.no_grad():
        expected = reference.raw_forward(batch)['pred']
    del reference
    model = BranchModel(ds, cfg)
    ai, oi, _ = candidates(model, ds, train=True)
    state = state_digest(model.base)
    with torch.no_grad():
        initial = model.score(batch, ai, oi)
    error = float((initial - expected).abs().max())
    if error > 1e-6:
        raise ValueError('Initial residual does not reproduce the actual A_unit anchor')
    model.train(True)
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = []
    for _ in range(3):
        result = model.score(batch, ai, oi)
        loss = F.cross_entropy(result / float(cfg.MODEL.cosine_cls_temp), batch['pair'])
        optimizer.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.router.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise ValueError('Missing/nonfinite branch gradients')
        counts.append(sum(int(g.abs().sum() > 0) for g in grads))
        if any(p.grad is not None for p in model.base.parameters()):
            raise ValueError('Frozen anchor received gradients')
        optimizer.step()
    if state_digest(model.base) != state or max(counts[1:]) < 2:
        raise ValueError('Anchor changed or router did not start learning')
    with torch.no_grad():
        off = model.score(batch, ai, oi, force_off=True)
    off_error = float((off - expected).abs().max())
    if off_error > 1e-6:
        raise ValueError('Force-off does not restore A_unit after updates')
    write_json(out, {'passed': True, 'base_state_unchanged': True,
        'config_sha256': sha(cfg_path), 'checkpoint_sha256': sha(cfg.MODEL.branch_anchor),
        'bank_manifest_sha256': sha(Path(cfg.MODEL.description_bank) / 'manifest.json'),
        'initial_max_abs_error': error, 'off_max_abs_error': off_error,
        'nonzero_gradient_parameter_counts': counts, 'branch_mode': cfg.MODEL.branch_mode,
        'new_trainable_params': sum(p.numel() for p in model.router.parameters())})
    print('NAME GPU SMOKE PASSED', out, flush=True)
