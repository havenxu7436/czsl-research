#!/usr/bin/env python3
"""Real-batch baseline equivalence, prior ordering and A/C/D/E gradient checks."""
import argparse
import gc
import json
import random
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg as default_cfg
from dataset import CompositionDataset
from models.ddm import DDM as OriginalDDM
from models.kcc_cpf import DDM as KCCDDM


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def candidates(model, pairs, dataset, device):
    ids = torch.tensor([(dataset.attr2idx[a], dataset.obj2idx[o]) for a, o in pairs], device=device)
    model.val_attrs, model.val_objs = ids[:, 0], ids[:, 1]
    model.val_pairs = pairs
    model.pairs = ids
    model.pair2idx = {p: i for i, p in enumerate(pairs)}


def tensor_batch(dataset, device, count):
    loader = torch.utils.data.DataLoader(dataset, batch_size=count, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def prepare_train(model, state, cfg):
    model.load_state_dict(state, strict=True)
    model.zero_grad(set_to_none=True)
    model.train()
    if not cfg.TRAIN.finetune_backbone:
        model.feat_extractor.eval()
        model.feat_extractor.requires_grad_(False)
    seed(int(cfg.TRAIN.seed))


def near(actual, expected, label, tolerance=1e-6):
    diff = (actual - expected).abs().max().item()
    if diff > tolerance:
        raise AssertionError("{} max difference {} > {}".format(label, diff, tolerance))
    return diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("batch-size >= 2 is required for BatchNorm training")
    if args.out.exists():
        raise FileExistsError("refusing to overwrite smoke result: " + str(args.out))
    if not torch.cuda.is_available():
        raise RuntimeError("run smoke on an available CUDA GPU")
    cfg = default_cfg.clone()
    cfg.merge_from_file(args.cfg)
    cfg.EVAL.open_world = False
    cfg.MODEL.kcc_text = False
    cfg.MODEL.kcc_calib = False
    cfg.MODEL.kcc_feature = False
    cfg.MODEL.kcc_logic = False
    cfg.MODEL.kcc_lambda = 0.0
    cfg.MODEL.kcc_score_mode = "legacy"
    cfg.MODEL.kcc_control = "none"
    cfg.MODEL.kcc_cache = args.cache
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    seed(int(cfg.TRAIN.seed))
    trainset = CompositionDataset(phase="train", split=cfg.DATASET.splitname, open_world=False, cfg=cfg)
    valset = CompositionDataset(phase="val", split=cfg.DATASET.splitname, open_world=False, cfg=cfg)
    train_batch = tensor_batch(trainset, device, args.batch_size)
    val_batch = tensor_batch(valset, device, args.batch_size)
    state = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("expected baseline state_dict checkpoint")
    old = OriginalDDM(trainset, cfg).to(device)
    old.load_state_dict(state, strict=True)
    old.eval()
    pair_sets = {"cw": list(valset.pairs), "ow": list(product(valset.attrs, valset.objs))}
    references = {}
    with torch.no_grad():
        for setting, pairs in pair_sets.items():
            candidates(old, pairs, valset, device)
            references[setting] = old(val_batch)["pred"].cpu()
    prepare_train(old, state, cfg)
    original_loss = old(train_batch)["loss_total"].detach().cpu()
    del old
    gc.collect()
    torch.cuda.empty_cache()
    cfg.MODEL.kcc_text, cfg.MODEL.kcc_calib, cfg.MODEL.kcc_lambda = True, True, .1
    model = KCCDDM(trainset, cfg).to(device)
    if set(model.state_dict()) != set(state):
        raise AssertionError("new model broke strict baseline checkpoint compatibility")
    report = {"baseline_checkpoint": args.checkpoint, "cache": args.cache,
              "cache_manifest_sha256": model.kcc_cache_manifest_sha256,
              "batch_size": args.batch_size, "settings": {}, "train": {}}
    for setting, pairs in pair_sets.items():
        model.load_state_dict(state, strict=True)
        model.eval()
        candidates(model, pairs, valset, device)
        with torch.no_grad():
            model.kcc_text, model.kcc_calib = False, False
            a = model(val_batch)["pred"]
            expected_a = references[setting].to(device)
            difference = near(a, expected_a, setting + " A vs original")
            if not torch.equal(a.argmax(1), expected_a.argmax(1)):
                raise AssertionError("A top1 changed")
            model.kcc_calib, model.kcc_lambda = True, 0.
            near(model(val_batch)["pred"], a, setting + " lambda0")
            model.kcc_lambda = .1
            d = model(val_batch)["pred"]
            prior = model.K_oa_prior[model.val_objs, model.val_attrs]
            near(d, a + .1 * prior.unsqueeze(0), setting + " D pair ordering")
            model.kcc_text, model.kcc_calib = True, False
            c = model(val_batch)["pred"]
            model.kcc_calib = True
            e = model(val_batch)["pred"]
            near(e, c + .1 * prior.unsqueeze(0), setting + " E pair ordering")
            saved = model.K_oa_prior.clone()
            model.K_oa_prior.zero_()
            near(model(val_batch)["pred"], c, setting + " uniform prior")
            model.K_oa_prior.copy_(saved)
            # Reordered candidates must retain the same per-pair scores.
            candidates(model, list(reversed(pairs)), valset, device)
            near(model(val_batch)["pred"], e.flip(1), setting + " candidate permutation", 1e-5)
        report["settings"][setting] = {"A_max_abs_score_difference": difference,
                                       "A_top1_identical": True, "candidate_count": len(pairs),
                                       "lambda0_identity": True, "uniform_identity": True,
                                       "D_E_prior_ordering": True, "candidate_permutation": True}
        print("PASS {}: baseline identity / lambda0 / uniform / prior ordering / candidate permutation".format(setting), flush=True)
    for variant, flags in (("A", (False, False)), ("C", (True, False)),
                           ("D", (False, True)), ("E", (True, True))):
        model.kcc_text, model.kcc_calib = flags
        model.kcc_lambda = .1 if flags[1] else 0.
        prepare_train(model, state, cfg)
        captured = []
        hook = model.classifier.register_forward_hook(lambda module, inp, output: captured.append(output))
        output = model(train_batch)
        hook.remove()
        loss = output["loss_total"]
        if not torch.isfinite(loss):
            raise AssertionError(variant + " non-finite loss")
        if variant == "A":
            near(loss.detach().cpu(), original_loss, "A train loss vs original", 1e-5)
        if len(captured) != 3:
            raise AssertionError("composition classifier call count changed")
        raw = captured[2]
        if flags[1]:
            raw = raw + model.kcc_lambda * model.K_oa_prior[model.train_objs, model.train_attrs].unsqueeze(0)
        expected_loss = (F.cross_entropy(raw, train_batch["pair"])
                         + .4 * F.cross_entropy(captured[0], train_batch["obj"])
                         + .6 * F.cross_entropy(captured[1], train_batch["attr"]))
        near(loss.detach(), expected_loss.detach(), variant + " train prior CE placement")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise AssertionError(variant + " missing/non-finite gradients")
        if not any(g.abs().sum().item() > 0 for g in gradients):
            raise AssertionError(variant + " zero gradients")
        for name in ("W_o_K", "W_a_K", "K_oa_prior"):
            value = getattr(model, name)
            if value.requires_grad or value.grad is not None:
                raise AssertionError("knowledge tensor is trainable: " + name)
        report["train"][variant] = {"loss": loss.item(), "finite_nonzero_gradients": True,
                                    "prior_CE_placement": True, "frozen_knowledge": True}
        print("PASS {}: finite loss/backward, frozen knowledge, correct train prior placement".format(variant), flush=True)
        del captured, loss, output, raw, expected_loss, gradients
        model.zero_grad(set_to_none=True)
    report["status"] = "MVP_SMOKE_PASSED"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("MVP SMOKE PASSED: " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
