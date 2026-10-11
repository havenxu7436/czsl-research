"""Frozen-cache loading and pair calibration for the existing GloVe CPF."""
import hashlib
import json
import math
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class KCCCacheMixin:
    def _setup_kcc(self, dset, cfg):
        self.kcc_text = bool(getattr(cfg.MODEL, "kcc_text", False))
        self.kcc_calib = bool(getattr(cfg.MODEL, "kcc_calib", False))
        self.kcc_lambda = float(getattr(cfg.MODEL, "kcc_lambda", 0.0))
        self.kcc_cache_manifest_sha256 = None
        if not math.isfinite(self.kcc_lambda) or self.kcc_lambda < 0:
            raise ValueError("kcc_lambda must be finite and nonnegative")
        if getattr(cfg.MODEL, "kcc_score_mode", "legacy") != "legacy":
            raise NotImplementedError("this MVP implements legacy scores only")
        if getattr(cfg.MODEL, "kcc_control", "none") != "none":
            raise NotImplementedError("negative controls need their own implementation")
        if not (self.kcc_text or self.kcc_calib):
            return
        if self.word_dim != 300:
            raise ValueError("GloVe cache requires a 300-dimensional baseline")
        cache = Path(cfg.MODEL.kcc_cache).resolve()
        manifest_path = cache / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["attrs"] != list(dset.attrs) or manifest["objs"] != list(dset.objs):
            raise ValueError("cache primitive labels/order mismatch")
        if manifest["pair_axis_order"] != ["object", "attribute"]:
            raise ValueError("cache K must have [object, attribute] axes")
        if manifest["dimension"] != 300 or manifest["dtype"] != "float32":
            raise ValueError("cache dimension/dtype mismatch")
        # Verify the small MVP tensors; pair blocks are not consumed by this model.
        names = ["W_o_name_raw.pt", "W_a_name_raw.pt", "labels.json"]
        if self.kcc_text:
            names += ["W_o_K.pt", "W_a_K.pt"]
        if self.kcc_calib:
            names += ["K_oa_text.pt", "K_oa_prior_centered.pt"]
        for name in names:
            if file_sha256(cache / name) != manifest["files_sha256"][name]:
                raise ValueError("cache file checksum mismatch: " + name)
        labels = json.loads((cache / "labels.json").read_text(encoding="utf-8"))
        if labels != {"attrs": list(dset.attrs), "objs": list(dset.objs)}:
            raise ValueError("cache labels.json mismatch")
        raw_obj = torch.load(cache / "W_o_name_raw.pt", map_location="cpu")
        raw_attr = torch.load(cache / "W_a_name_raw.pt", map_location="cpu")
        for raw, original, kind in ((raw_obj, self.obj_embedder.weight, "object"),
                                     (raw_attr, self.attr_embedder.weight, "attribute")):
            if raw.shape != original.shape or not torch.allclose(
                    raw, original.detach().cpu(), atol=1e-6, rtol=1e-5):
                raise ValueError(kind + " cached baseline name vectors differ from initializer")
        if self.kcc_text:
            for name, filename, size in (("W_o_K", "W_o_K.pt", self.num_objs),
                                        ("W_a_K", "W_a_K.pt", self.num_attrs)):
                value = torch.load(cache / filename, map_location="cpu").float()
                if tuple(value.shape) != (size, self.word_dim) or not torch.isfinite(value).all():
                    raise ValueError("bad knowledge tensor: " + filename)
                if not torch.allclose(value.norm(dim=-1), torch.ones(size), atol=1e-5):
                    raise ValueError("knowledge embedding unit norm failure: " + filename)
                self.register_buffer(name, value, persistent=False)
        if self.kcc_calib:
            k = torch.load(cache / "K_oa_text.pt", map_location="cpu").float()
            prior = torch.load(cache / "K_oa_prior_centered.pt", map_location="cpu").float()
            if tuple(k.shape) != (self.num_objs, self.num_attrs) or prior.shape != k.shape:
                raise ValueError("bad K shape")
            if not torch.isfinite(k).all() or not torch.isfinite(prior).all():
                raise ValueError("non-finite K")
            expected = torch.zeros_like(k) if k.std(unbiased=False) < 1e-8 else (k - k.mean()) / k.std(unbiased=False)
            if not torch.allclose(expected, prior, atol=1e-6, rtol=1e-5):
                raise ValueError("K prior is not the population-standardized K")
            self.register_buffer("K_oa_prior", prior, persistent=False)
        self.kcc_cache_manifest_sha256 = file_sha256(manifest_path)
        print("KCC CACHE LOADED: {} text={} calib={} lambda={} status={}".format(
            cache, self.kcc_text, self.kcc_calib, self.kcc_lambda,
            manifest.get("knowledge_status")), flush=True)

    def _kcc_obj_embeddings(self, indices):
        if self.kcc_text:
            return self.W_o_K.index_select(0, indices)
        return self.obj_embedder(indices)

    def _kcc_attr_embeddings(self, indices):
        if self.kcc_text:
            return self.W_a_K.index_select(0, indices)
        return self.attr_embedder(indices)

    def _kcc_calibrate(self, scores, obj_ids, attr_ids):
        if not self.kcc_calib or self.kcc_lambda == 0.0:
            return scores
        prior = self.K_oa_prior[obj_ids, attr_ids]
        if scores.size(1) != prior.numel():
            raise ValueError("K prior candidate order/shape mismatch")
        return scores + self.kcc_lambda * prior.unsqueeze(0)
