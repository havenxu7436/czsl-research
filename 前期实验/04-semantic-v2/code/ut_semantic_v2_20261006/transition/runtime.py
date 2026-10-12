import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import load_bank, sha
from .router import Router


def read_cfg(path):
    from config import cfg as original
    cfg = original.clone()
    cfg.merge_from_file(str(path))
    if bool(cfg.MODEL.kcc_calib) or float(cfg.MODEL.kcc_lambda) != 0:
        raise ValueError("Static K must remain disabled")
    return cfg


def trainset(cfg):
    from dataset import CompositionDataset
    return CompositionDataset(phase="train", split=cfg.DATASET.splitname,
                              open_world=False, cfg=cfg)


def base_model(ds, cfg, checkpoint):
    from models.ekr_base import DDM
    model = DDM(ds, cfg).to("cuda:0")
    model.load_state_dict(torch.load(str(checkpoint), map_location="cuda:0"), strict=True)
    return model


class BranchModel(torch.nn.Module):
    def __init__(self, ds, cfg):
        super().__init__()
        if sha(cfg.MODEL.branch_anchor) != cfg.MODEL.branch_anchor_sha256:
            raise ValueError("C anchor checkpoint changed")
        self.base = base_model(ds, cfg, cfg.MODEL.branch_anchor)
        self.frozen = str(getattr(cfg.MODEL, "branch_regime", "frozen")) == "frozen"
        self.off_mode = cfg.MODEL.branch_mode == "off"
        self.bank, self.bank_manifest = load_bank(cfg.MODEL.description_bank, ds.attrs, ds.objs)
        self.router = Router(self.bank, "adapter" if self.off_mode else cfg.MODEL.branch_mode,
                             cfg.MODEL.branch_rank, cfg.MODEL.branch_eta_max,
                             cfg.MODEL.branch_temperature, cfg.MODEL.branch_object_chunk).to("cuda:0")
        if self.frozen:
            self.base.requires_grad_(False)
        if self.off_mode:
            self.router.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.base.eval()
        self.router.train(mode)
        return self

    def components(self, batch):
        context = torch.no_grad() if self.frozen else contextlib.nullcontext()
        with context:
            values = self.base.raw_forward(batch)
        return values["pred"], values["attr_feature"]

    def score(self, batch, ai, oi, force_off=False, return_diag=False):
        base, feature = self.components(batch)
        result = self.router(feature, ai, oi, force_off=(force_off or self.off_mode),
                             return_diag=return_diag)
        if return_diag:
            residual, diagnostics = result
            return base + residual, diagnostics
        return base + result

    @contextlib.contextmanager
    def corrupt(self, ratio, seed):
        """New branch input only: frozen base C and bank files stay unchanged."""
        if not 0 <= ratio <= 1:
            raise ValueError("Corruption ratio must be in [0,1]")
        if ratio == 0:
            self.last_corruption = {"nominal_fraction": 0.0, "actual_fraction": 0.0,
                                    "receiver_attr_ids": [], "donor_attr_ids": [], "seed": seed}
            yield
            return
        if self.router.mode == "shuffle":
            raise ValueError("Noise tests use clean-knowledge routers; R7 is a separate shuffled control")
        original = {k: getattr(self.router, k) for k in ("key_source", "value_source", "slot_mask")}
        a = self.bank["a"]
        units, mask = a["units"].clone(), a["mask"].clone()
        n = units.size(0)
        rng = np.random.RandomState(seed)
        count = max(1, int(round(n * ratio)))
        receivers = rng.permutation(n)[:count]
        donors = []
        for i in receivers:
            donor = int((i + rng.randint(1, n)) % n)
            donors.append(donor)
            units[i, 1:] = a["units"][donor, 1:]
            mask[i, 1:] = a["mask"][donor, 1:]
        self.last_corruption = {"nominal_fraction": ratio, "actual_fraction": count / n,
                                "receiver_attr_ids": receivers.tolist(), "donor_attr_ids": donors,
                                "seed": seed}
        altered = {"a": dict(a, units=units, mask=mask), "o": self.bank["o"]}
        altered["a"]["correct"] = F.normalize(
            (units * mask[..., None]).sum(1) / mask.sum(1, keepdim=True), dim=-1)
        # Only replace input buffers. Do not consume training/dropout RNG state.
        with torch.random.fork_rng(devices=[]):
            replacement = Router(altered, "adapter" if self.off_mode else self.router.mode,
                                 self.router.qx.out_features)
        try:
            device = self.router.key_source.device
            for key in original:
                setattr(self.router, key, getattr(replacement, key).to(device))
            yield
        finally:
            for key, value in original.items():
                setattr(self.router, key, value)


def load_branch(cfg, ds, checkpoint):
    saved = torch.load(str(checkpoint), map_location="cpu")
    if saved["anchor_sha256"] != cfg.MODEL.branch_anchor_sha256:
        raise ValueError("Branch checkpoint C anchor differs from config")
    if saved["bank_manifest_sha256"] != sha(Path(cfg.MODEL.description_bank) / "manifest.json"):
        raise ValueError("Branch checkpoint bank differs from config")
    if saved["branch_mode"] != cfg.MODEL.branch_mode:
        raise ValueError("Branch checkpoint mode differs from config")
    if saved["regime"] != getattr(cfg.MODEL, "branch_regime", "frozen"):
        raise ValueError("Branch checkpoint training regime differs from config")
    model = BranchModel(ds, cfg)
    model.router.load_state_dict(saved["router_state"], strict=True)
    if "base_state" in saved:
        model.base.load_state_dict(saved["base_state"], strict=True)
    model.eval()
    return model, saved
