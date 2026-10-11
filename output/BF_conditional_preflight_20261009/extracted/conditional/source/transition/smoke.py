from pathlib import Path

import torch
import torch.nn.functional as F

from .common import candidates, load_bank, move, seed_all, sha, state_digest, write_json
from .runtime import BranchModel, base_model, read_cfg, trainset


def smoke(args):
    from models.kcc_cpf import DDM as OriginalKCC
    cfg = read_cfg(args.cfg)
    seed_all(int(cfg.TRAIN.seed))
    ds = trainset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = move(next(iter(loader)), "cuda:0")
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = cfg.MODEL.branch_anchor
    original_cfg = cfg.clone()
    mode = getattr(cfg.MODEL, "semantic_text", "correct")
    if mode not in {"raw", "correct"}:
        raise ValueError("Fidelity smoke uses A_raw or C_text, not unit/shuffled inputs")
    original_cfg.MODEL.kcc_text = (mode == "correct")
    original = OriginalKCC(ds, original_cfg).to("cuda:0")
    original.load_state_dict(torch.load(str(checkpoint), map_location="cuda:0"), strict=True)
    original.eval()
    ai, oi, pairs = candidates(original, ds, train=True)
    with torch.no_grad():
        expected = original.val_forward(batch)["pred"]
    base = base_model(ds, cfg, checkpoint)
    base.eval()
    candidates(base, ds, train=True)
    with torch.no_grad():
        actual = base.raw_forward(batch)
    error = float((actual["pred"] - expected).abs().max())
    if error > 1e-6:
        raise ValueError("C/base fidelity failed: " + str(error))
    del original, base
    checks = {"base_original_max_abs_error": error, "checkpoint_sha256": sha(checkpoint),
              "bank_manifest_sha256": sha(Path(cfg.MODEL.description_bank) / "manifest.json"),
              "branch_mode": getattr(cfg.MODEL, "branch_mode", None),
              "regime": getattr(cfg.MODEL, "branch_regime", None)}
    if hasattr(cfg.MODEL, "branch_mode"):
        model = BranchModel(ds, cfg)
        if not model.frozen or model.off_mode:
            raise ValueError("GPU branch smoke should use the frozen non-off pilot config")
        ai, oi, _ = candidates(model, ds, train=True)
        initial = state_digest(model.base)
        with torch.no_grad():
            score = model.score(batch, ai, oi)
        checks["initial_branch_max_abs_error"] = float((score - expected).abs().max())
        if checks["initial_branch_max_abs_error"] > 1e-6:
            raise ValueError("Zero residual does not reproduce C")
        model.train(True)
        optimizer = torch.optim.Adam(model.router.parameters(), lr=1e-3)
        counts = []
        for step in range(3):
            result = model.score(batch, ai, oi)
            loss = F.cross_entropy(result / float(cfg.MODEL.cosine_cls_temp), batch["pair"])
            optimizer.zero_grad()
            loss.backward()
            grads = [p.grad for p in model.router.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise ValueError("GPU branch gradient is missing/nonfinite")
            counts.append(sum(int(g.abs().sum() > 0) for g in grads))
            optimizer.step()
        if max(counts[1:]) < 2:
            raise ValueError("Router does not learn after initial gain update")
        if initial != state_digest(model.base):
            raise ValueError("Frozen C state changed during smoke")
        with torch.no_grad():
            off = model.score(batch, ai, oi, force_off=True)
        if (off - expected).abs().max() > 1e-6:
            raise ValueError("Off state differs from C after router updates")
        checks.update(base_state_unchanged=True, nonzero_grad_parameter_counts=counts,
                      learned_eta=float(model.router.eta.detach()))
    checks["passed"] = True
    output = Path(args.out)
    if output.exists():
        raise FileExistsError("Audit file exists; use a new audit version")
    write_json(output, checks)
    print("GPU SMOKE PASSED", checks, flush=True)
