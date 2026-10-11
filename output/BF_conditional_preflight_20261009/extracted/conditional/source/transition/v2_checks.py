"""Real server GPU smoke and post-run integrity audit; never substitutes for results."""
import argparse
import gc
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import yaml

from .common import code_manifest, move, read_json, seed_all, sha, write_json
from .runtime import read_cfg, trainset
from .v2_prepare import require, verify_prepared


def verify_native_log(run, cfg):
    # Native print(cfg) is not cfg.dump(): 1e-05 and None can lose scalar types.
    # Reproduce the exact YACS printer, retaining strict comparison of every field.
    text = (run / "train.log").read_text(encoding="utf-8", errors="replace")
    require("Random seed:" in text, "Native merged config missing from training log")
    logged = yaml.safe_load(text.split("Random seed:", 1)[0])
    require(logged == yaml.safe_load(str(cfg)), "Native logged training config differs: " + run.name)


def paired_bias0(index, root):
    """Verify matched VAL identities and recompute paired bias-zero top1 error counts."""
    references, rows = {}, {}
    for rid, path in index.items():
        cfg = read_cfg(path)
        run = Path(cfg.TRAIN.kcc_run_dir)
        for setting in ("cw", "ow"):
            folder = run / ("eval_val_" + setting)
            ids = read_json(folder / "image_ids.json")
            candidates = read_json(folder / "candidates.json")
            with np.load(str(folder / "labels.npz"), allow_pickle=False) as stored:
                arrays = {key: stored[key].copy() for key in ("attr", "obj", "pair", "top1_at_bias0")}
            require(len(ids) == len(set(ids)) == 3214, "VAL identity/length mismatch: " + rid)
            require(all(v.shape == (3214,) and np.issubdtype(v.dtype, np.integer) for v in arrays.values()),
                    "VAL label/prediction array shape or dtype mismatch")
            require(np.all((arrays["pair"] >= 0) & (arrays["pair"] < len(candidates))) and
                    np.all((arrays["top1_at_bias0"] >= 0) & (arrays["top1_at_bias0"] < len(candidates))),
                    "VAL candidate index outside valid range")
            eval_cfg = cfg.clone()
            eval_cfg.EVAL.open_world = setting == "ow"
            record = read_json(folder / "metrics.json")
            import hashlib
            require(record["config_sha256"] == hashlib.sha256(eval_cfg.dump().encode()).hexdigest() and
                    record["n_images"] == len(ids) and record["n_candidates"] == len(candidates),
                    "Evaluation config/dimensions differ")
            if setting not in references:
                references[setting] = (ids, candidates, arrays)
            ref_ids, ref_candidates, ref_arrays = references[setting]
            require(ids == ref_ids and candidates == ref_candidates and
                    all(np.array_equal(arrays[k], ref_arrays[k]) for k in ("attr", "obj", "pair")),
                    "Paired validation images/candidates/gold labels differ: " + rid)
            rows[int(cfg.TRAIN.seed), cfg.TRAIN.kcc_variant, setting] = arrays
    # Check those shared identities/labels against immutable original metadata, not just each other.
    data = root / "data/ut-zap50k"
    metadata = torch.load(data / "metadata_compositional-split-natural.t7", map_location="cpu")
    labels = read_json(Path(read_cfg(next(iter(index.values()))).MODEL.kcc_cache) / "labels.json")
    attr_idx = {v:i for i,v in enumerate(labels["attrs"])}
    obj_idx = {v:i for i,v in enumerate(labels["objs"])}
    train_pairs = {tuple(x.split()) for x in (data / "compositional-split-natural/train_pairs.txt").read_text().splitlines()}
    unseen = {}
    for setting, (ids, candidates, arrays) in references.items():
        pair_idx = {tuple(v):i for i,v in enumerate(candidates)}
        gold = [r for r in metadata if r["set"] == "val" and (r["attr"],r["obj"]) in pair_idx]
        require(ids == [r["image"] for r in gold], "VAL image order differs from source metadata")
        require(np.array_equal(arrays["attr"], [attr_idx[r["attr"]] for r in gold]) and
                np.array_equal(arrays["obj"], [obj_idx[r["obj"]] for r in gold]) and
                np.array_equal(arrays["pair"], [pair_idx[r["attr"],r["obj"]] for r in gold]),
                "VAL gold labels differ from source metadata")
        unseen[setting] = np.array([(r["attr"],r["obj"]) not in train_pairs for r in gold])
    comparisons = []
    for seed in (0,3351,2026):
        for setting in ("cw","ow"):
            current = rows[seed,"C_text",setting]
            correct = current["top1_at_bias0"] == current["pair"]
            for variant in ("A_raw","A_unit","C_shuffle"):
                reference = rows[seed,variant,setting]
                old_correct = reference["top1_at_bias0"] == reference["pair"]
                improved = (~old_correct) & correct
                worsened = old_correct & (~correct)
                comparisons.append({"seed":seed,"setting":setting,"comparison":"C_text-"+variant,
                    "wrong_to_correct":int(improved.sum()),"correct_to_wrong":int(worsened.sum()),
                    "net_corrected":int(improved.sum()-worsened.sum()),
                    "unseen_net_corrected":int((improved & unseen[setting]).sum()-(worsened & unseen[setting]).sum())})
    return {"matched_validation_identities":True,"comparisons":comparisons,
        "limits":"Bias-zero top1 counts do not recompute AUC/HM and are not causal semantic proof."}


def gpu_smoke(project, root, work):
    frozen, tables = verify_prepared(project, root, work)
    import torchvision
    require(sys.version_info[:2] == (3, 8), "Use the existing CPF Python3.8 environment")
    require(torch.__version__.startswith("1.13.1") and torchvision.__version__.startswith("0.14.1"),
            "Use the frozen torch1.13.1 / torchvision0.14.1 environment")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    destination = work / "gpu_smoke.json"
    if destination.exists():
        prior = read_json(destination)
        require(prior["prepared_sha256"] == sha(work / "prepared.json") and prior["passed"],
                "Existing smoke belongs to another preparation")
        print("GPU SMOKE ALREADY COMPLETE", destination, flush=True)
        return
    from .smoke import smoke
    legacy = root / "experiments/kcc0601_glove_mvp_v1/runs"
    fidelity = {}
    for variant, letter in (("A_raw", "A"), ("C_text", "C")):
        cfg_path = work / "experiments/semantic_v2/configs" / ("semantic_ut-zappos_{}_s3351_e10.yml".format(variant))
        checkpoint = legacy / ("mvp_ut-zappos_{}_legacy_s3351_l0".format(letter)) / "checkpoints/best_auc.pth"
        out = work / "smoke" / (variant + "_fidelity.json")
        if not out.exists():
            smoke(SimpleNamespace(cfg=cfg_path, checkpoint=checkpoint, out=out))
        evidence = read_json(out)
        require(evidence["passed"] and evidence["checkpoint_sha256"] == sha(checkpoint) and
                evidence["bank_manifest_sha256"] == sha(work / "bank/ut-zap50k/manifest.json") and
                evidence["base_original_max_abs_error"] <= 1e-6,
                "Fidelity smoke identity/error mismatch")
        fidelity[variant] = evidence
        gc.collect()
        torch.cuda.empty_cache()
    # These steps are disposable interface checks; no checked weights are reused in training.
    from models.semantic_control import DDM
    gradient_checks = {}
    parameter_counts = []
    for variant in frozen["variants"]:
        seed_all(3351)
        path = work / "experiments/semantic_v2/configs" / ("semantic_ut-zappos_{}_s3351_e10.yml".format(variant))
        cfg = read_cfg(path)
        ds = trainset(cfg)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        batch = move(next(iter(loader)), "cuda:0")
        model = DDM(ds, cfg).to("cuda:0")
        model.train()
        output = model(batch)
        loss = output["loss_total"]
        require(bool(torch.isfinite(loss)), "Nonfinite native loss: " + variant)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads = [(name, parameter) for name, parameter in model.named_parameters() if parameter.grad is not None]
        require(grads and all(bool(torch.isfinite(p.grad).all()) for name, p in grads),
                "Nonfinite/missing gradient: " + variant)
        active = [(name, p) for name, p in grads if bool((p.grad != 0).any())]
        require(active, "All gradients are zero: " + variant)
        # One small head parameter update avoids duplicating full ViT Adam state for a smoke.
        chosen = next(((name, p) for name, p in active if name.startswith("objc_embedder")), active[-1])
        name, parameter = chosen
        before = parameter.detach().clone()
        optimizer = torch.optim.Adam([parameter], lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.wd)
        optimizer.step()
        changed = float((before - parameter.detach()).abs().max())
        require(changed > 0, "Smoke head parameter did not update")
        count = sum(p.numel() for p in model.parameters())
        parameter_counts.append(count)
        gradient_checks[variant] = {"loss_total": float(loss.detach()), "parameters": count,
            "parameters_with_finite_gradient": len(grads), "parameters_with_nonzero_gradient": len(active),
            "smoke_updated_parameter": name, "update_max_abs": changed,
            "training_weights_saved_or_reused": False}
        print("NATIVE ONE-BATCH SMOKE PASSED", variant, flush=True)
        del optimizer, before, parameter, chosen, active, grads, output, loss, model, batch, loader, ds
        gc.collect()
        torch.cuda.empty_cache()
    require(len(set(parameter_counts)) == 1, "Control architectures have different parameter counts")
    record = {"passed": True, "prepared_sha256": sha(work / "prepared.json"),
        "python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0), "fidelity": fidelity, "one_batch": gradient_checks,
        "purpose": "Interface fidelity and finite native training gradients; no performance evidence"}
    write_json(destination, record)
    print("ALL V2 GPU SMOKE CHECKS PASSED", destination, flush=True)


def post_audit(project, root, work, out):
    frozen, tables = verify_prepared(project, root, work)
    index = read_json(work / "experiments/semantic_v2/config_index_semantic_e10.json")
    completed = {p.name for p in (work / "experiments/semantic_v2/runs").iterdir()
                 if (p / "status.json").is_file() and read_json(p / "status.json")["status"] == "train_and_val_complete"}
    require(completed == set(index), "Unexpected or missing completed run; refuse a mixed summary")
    checks = []
    for rid, path in index.items():
        cfg = read_cfg(path)
        run = Path(cfg.TRAIN.kcc_run_dir)
        require(read_json(run / "status.json")["status"] == "train_and_val_complete", "Run incomplete: " + rid)
        require((run / "effective_config.yml").read_text(encoding="utf-8") == cfg.dump(), "Effective config differs: " + rid)
        require(read_json(run / "meta/code_manifest.json") == frozen["code_manifest"], "Run code differs: " + rid)
        require(read_json(run / "meta/cache_manifest.json") == read_json(work / "cache/ut-zap50k/manifest.json"),
                "Run cache differs: " + rid)
        require(read_json(run / "meta/bank_manifest.json") == read_json(work / "bank/ut-zap50k/manifest.json"),
                "Run bank differs: " + rid)
        complete = read_json(run / "train_complete.json")
        checkpoint = run / "checkpoints/best_auc.pth"
        require(complete["epochs"] == 10 and isinstance(complete["best_epoch"], int) and
                1 <= complete["best_epoch"] <= 10, "Epoch completion record invalid: " + rid)
        require(sha(checkpoint) == complete["checkpoint_sha256"], "Checkpoint changed: " + rid)
        # Recover the actual merged native log, not just the requested YAML.
        verify_native_log(run, cfg)
        inputs = {}
        for setting in ("cw", "ow"):
            path = run / ("eval_val_" + setting) / "metrics.json"
            m = read_json(path)
            require(m["checkpoint_sha256"] == complete["checkpoint_sha256"] and
                    m["bank_manifest_sha256"] == sha(work / "bank/ut-zap50k/manifest.json") and
                    m["phase"] == "val" and m["setting"] == setting and m["variant"] == cfg.TRAIN.kcc_variant and
                    int(m["seed"]) == int(cfg.TRAIN.seed), "Evaluation identity differs: " + rid)
            require(m["training_regime"] == "native_semantic", "Unexpected evaluation regime")
            for key in ("AUC", "best_hm", "best_seen", "best_unseen"):
                require(0 <= m["metrics_raw"][key] <= 1, "Metric outside expected fraction range")
            inputs[str(path)] = sha(path)
        checks.append({"run_id": rid, "best_epoch": complete["best_epoch"],
            "checkpoint_sha256": complete["checkpoint_sha256"], "evaluation_sha256": inputs,
            "native_config_matched": True})
    require(len(checks) == 12, "Need all 12 matched runs")
    paired = paired_bias0(index, root)
    record = {"integrity_passed": True, "runs_checked": checks, "paired_bias0_errors": paired,
        "prepared_sha256": sha(work / "prepared.json"), "code_manifest": code_manifest(project),
        "not_a_research_gate": True, "test_evaluation_performed": False,
        "native_log_limit": "YACS print representation checked; effective YAML remains the scalar-type authority"}
    if out.exists():
        require(read_json(out) == record, "Post audit changed; use a new report version")
    else:
        write_json(out, record)
    print("POST-RUN INTEGRITY PASSED: 12 native logs/configs, checkpoints and 24 VAL evaluations", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["gpu-smoke", "post-audit"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.command == "gpu-smoke":
        gpu_smoke(project, args.root.resolve(), args.work.resolve())
    else:
        if not args.out:
            p.error("post-audit requires --out")
        post_audit(project, args.root.resolve(), args.work.resolve(), args.out)


if __name__ == "__main__":
    main()
