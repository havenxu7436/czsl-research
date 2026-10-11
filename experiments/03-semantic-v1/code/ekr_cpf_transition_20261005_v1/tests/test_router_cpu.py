"""CPU mathematical and gradient checks; no images or experiment results used."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from transition.router import Router
from transition.bank import derangement
from transition.runtime import BranchModel


def synthetic_bank(m=7, n=5):
    torch.manual_seed(914)
    units = F.normalize(torch.randn(m, 4, 300), dim=-1)
    mask = torch.ones(m, 4, dtype=torch.bool)
    mask[0, 2:] = False
    units[0, 2:] = 0
    mean = units.sum(1) / mask.sum(1, keepdim=True)
    residual = (units - mean[:, None]) * mask[..., None]
    perm = torch.tensor(derangement(m, 1729))
    shuffled = units[perm].clone()
    shuffled[:, 0] = units[:, 0]
    return {"a": {"units": units, "mask": mask, "residual": residual,
                   "mean": mean, "correct": F.normalize(mean, dim=-1),
                   "shuffle_units": shuffled, "shuffle_mask": mask[perm]},
            "o": {"correct": F.normalize(torch.randn(n, 300), dim=-1)}}


def main():
    bank = synthetic_bank()
    feature = torch.randn(3, 300)
    ai = torch.tensor([2, 0, 6, 1, 2, 4, 0, 3, 5])
    oi = torch.tensor([4, 0, 1, 2, 4, 3, 0, 1, 2])
    assert bank["a"]["residual"].sum(1).abs().max() < 1e-6
    evidence = torch.einsum("bd,mrd->bmr", F.normalize(feature, dim=-1),
                            bank["a"]["residual"] / 2)
    assert evidence.sum(-1).abs().max() < 1e-6
    for mode in ("adapter", "attention", "ekr", "static", "no_object",
                 "no_null", "shuffle", "name_residual", "raw_values"):
        torch.manual_seed(3351)
        router = Router(bank, mode, object_chunk=2)
        zero = router(feature, ai, oi)
        assert torch.equal(zero, torch.zeros_like(zero)), mode + " initial off"
        router.raw_gain.data.fill_(0.4)
        small = router(feature, ai, oi)
        router.object_chunk = 99
        large = router(feature, ai, oi)
        assert torch.allclose(small, large, atol=1e-6, rtol=1e-5), mode + " chunk"
        perm = torch.tensor([8, 1, 6, 2, 7, 3, 0, 4, 5])
        reordered = router(feature, ai[perm], oi[perm])
        assert torch.allclose(large[:, perm], reordered, atol=1e-6), mode + " pair order"
        assert large.abs().max() <= 0.1 + 1e-6, mode + " score bound"
        assert torch.equal(router(feature, ai, oi, force_off=True), torch.zeros_like(large))
        loss = F.cross_entropy(large / 0.05, torch.tensor([0, 1, 2]))
        loss.backward()
        grads = [p.grad for p in router.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), mode + " finite grad"
        assert any(g.abs().sum() > 0 for g in grads), mode + " nonzero grad"
        print("PASS", mode)
    # Name-only primitive: genuine difference is exactly zero.
    one = synthetic_bank()
    one["a"]["mask"][:] = False
    one["a"]["mask"][:, 0] = True
    one["a"]["units"][:, 1:] = 0
    for mode in ("ekr", "no_null", "name_residual"):
        router = Router(one, mode)
        router.raw_gain.data.fill_(0.7)
        assert router(feature, ai, oi).abs().max() == 0
    assert not any(i == j for i, j in enumerate(derangement(16, 1729)))
    # Corruption preserves base/bank and RNG; changes only temporary router input.
    model = BranchModel.__new__(BranchModel)
    torch.nn.Module.__init__(model)
    model.bank = synthetic_bank()
    model.router = Router(model.bank, "ekr")
    model.off_mode = False
    old = model.router.key_source.clone()
    rng = torch.get_rng_state().clone()
    with model.corrupt(0.3, 11):
        assert model.last_corruption["actual_fraction"] == 2 / 7
        assert all(i != j for i, j in zip(model.last_corruption["receiver_attr_ids"],
                                        model.last_corruption["donor_attr_ids"]))
        assert not torch.equal(old, model.router.key_source)
        assert torch.equal(rng, torch.get_rng_state())
        model.router.raw_gain.data.fill_(0.3)
        model.router(feature, ai, oi).sum().backward()
    assert torch.equal(old, model.router.key_source)
    assert torch.equal(rng, torch.get_rng_state())
    print("PASS corruption isolation and RNG preservation")
    print("ALL CPU ROUTER CHECKS PASSED", torch.__version__)


if __name__ == "__main__":
    main()
