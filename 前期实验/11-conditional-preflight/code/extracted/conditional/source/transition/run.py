import contextlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import (candidates, code_manifest, empty_output, metrics_pct, move,
                     read_json, seed_all, seed_worker, sha, state_digest, write_json)
from .evaluate import evaluate
from .runtime import BranchModel, base_model, load_branch, read_cfg, trainset


def metadata(run, cfg):
    (run / "meta").mkdir(parents=True, exist_ok=True)
    (run / "effective_config.yml").write_text(cfg.dump(), encoding="utf-8")
    write_json(run / "meta/code_manifest.json", code_manifest(Path.cwd()))
    shutil.copy2(Path(cfg.MODEL.kcc_cache) / "cache_manifest.json", run / "meta/cache_manifest.json")
    shutil.copy2(Path(cfg.MODEL.description_bank) / "manifest.json", run / "meta/bank_manifest.json")
    write_json(run / "meta/environment.json", {"python": sys.version,
               "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0)})


def eval_or_read(model, cfg, setting, out, force_off=False):
    path = Path(out) / "metrics.json"
    if path.is_file():
        value = read_json(path)
        if value["phase"] != "val" or value["setting"] != setting:
            raise ValueError("Existing evaluation identity mismatch")
        if value["bank_manifest_sha256"] != sha(Path(cfg.MODEL.description_bank) / "manifest.json"):
            raise ValueError("Existing evaluation bank mismatch")
        expected = bool(force_off or (hasattr(model, "off_mode") and model.off_mode))
        if bool(value.get("force_off", False)) != expected:
            raise ValueError("Existing evaluation off-state mismatch")
        expected_checkpoint = getattr(model, "evaluation_checkpoint_sha256", None)
        if value.get("checkpoint_sha256") != expected_checkpoint:
            raise ValueError("Existing evaluation checkpoint mismatch")
        return value
    return evaluate(model, cfg, "val", setting, out, force_off=force_off)


def semantic(args):
    cfg = read_cfg(args.cfg)
    run = Path(cfg.TRAIN.kcc_run_dir)
    if args.finish_only:
        if (run / "effective_config.yml").read_text(encoding="utf-8") != cfg.dump():
            raise ValueError("Finish-only config differs from original effective config")
        if read_json(run / "meta/code_manifest.json") != code_manifest(Path.cwd()):
            raise ValueError("Code changed since training; use a separately documented evaluation version")
    if not args.finish_only:
        empty_output(run)
        metadata(run, cfg)
        write_json(run / "status.json", {"status": "training"})
        with (run / "train.log").open("w", encoding="utf-8") as log:
            command = [sys.executable, "-u", "train_semantic.py", "--cfg", str(args.cfg)]
            write_json(run / "command_train.json", command)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError("Native semantic training failed; preserve this run for diagnosis")
        native = run / "checkpoints" / (cfg.config_name + "_" + cfg.TRAIN.train_name) / "best_auc.pth"
        target = run / "checkpoints/best_auc.pth"
        shutil.copy2(native, target)
        matches = re.findall(r"Loading validation-best checkpoint from epoch (\d+)",
                             (run / "train.log").read_text())
        write_json(run / "train_complete.json", {"checkpoint_sha256": sha(target),
                   "best_epoch": int(matches[-1]) if matches else None,
                   "epochs": int(cfg.TRAIN.max_epoch)})
    if not (run / "train_complete.json").is_file():
        raise ValueError("No successful training completion record; finish-only cannot resume training")
    target = run / "checkpoints/best_auc.pth"
    complete = read_json(run / "train_complete.json")
    if sha(target) != complete["checkpoint_sha256"]:
        raise ValueError("Semantic checkpoint changed")
    model = base_model(trainset(cfg), cfg, target)
    model.evaluation_checkpoint_sha256 = sha(target)
    for setting in ("cw", "ow"):
        record = eval_or_read(model, cfg, setting, run / ("eval_val_" + setting))
        record.update(checkpoint_sha256=sha(target), best_epoch=complete["best_epoch"],
                      training_regime="native_semantic")
        write_json(run / ("eval_val_" + setting) / "metrics.json", record)
    write_json(run / "status.json", {"status": "train_and_val_complete"})


def snapshot_branch(model, cfg, epoch):
    value = {"router_state": model.router.state_dict(), "epoch": epoch,
             "anchor_sha256": cfg.MODEL.branch_anchor_sha256,
             "bank_manifest_sha256": sha(Path(cfg.MODEL.description_bank) / "manifest.json"),
             "branch_mode": cfg.MODEL.branch_mode,
             "regime": getattr(cfg.MODEL, "branch_regime", "frozen")}
    if not model.frozen:
        value["base_state"] = model.base.state_dict()
    return value


def branch(args):
    cfg = read_cfg(args.cfg)
    if bool(getattr(cfg.TRAIN, "branch_smoke_only", False)):
        raise ValueError("This is an interface audit config; formal training is disabled")
    gate_file = getattr(cfg.TRAIN, "semantic_gate_file", "")
    if not gate_file or sha(gate_file) != cfg.TRAIN.semantic_gate_sha256:
        raise ValueError("Semantic gate is absent/changed; regenerate configs after the valid decision")
    if not read_json(gate_file)["datasets"][cfg.TRAIN.semantic_gate_dataset]["proceed"]:
        raise ValueError("Semantic screening gate did not pass")
    run = Path(cfg.TRAIN.kcc_run_dir)
    if args.finish_only:
        if (run / "effective_config.yml").read_text(encoding="utf-8") != cfg.dump():
            raise ValueError("Finish-only config differs from original effective config")
        if read_json(run / "meta/code_manifest.json") != code_manifest(Path.cwd()):
            raise ValueError("Code changed since training; preserve the old run and version the evaluation")
    if not args.finish_only:
        empty_output(run)
        metadata(run, cfg)
        seed_all(int(cfg.TRAIN.seed))
        ds = trainset(cfg)
        model = BranchModel(ds, cfg)
        if model.frozen and model.off_mode:
            raise ValueError("Frozen off mode has no trainable branch; use C0 evaluation")
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.TRAIN.batch_size,
                    shuffle=True, num_workers=cfg.TRAIN.num_workers,
                    pin_memory=True, worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(int(cfg.TRAIN.seed)))
        groups = []
        if not model.off_mode:
            groups.append({"params": [p for p in model.router.parameters() if p.requires_grad],
                           "lr": cfg.TRAIN.branch_lr})
        if not model.frozen:
            encoder, rest = [], []
            for name, p in model.base.named_parameters():
                if p.requires_grad:
                    (encoder if name.startswith("feat_extractor") else rest).append(p)
            if encoder:
                groups.append({"params": encoder, "lr": cfg.TRAIN.lr_encoder})
            if rest:
                groups.append({"params": rest, "lr": cfg.TRAIN.lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg.TRAIN.branch_weight_decay)
        start_state = state_digest(model.base) if model.frozen else None
        write_json(run / "meta/branch_identity.json", {
            "anchor": cfg.MODEL.branch_anchor, "anchor_sha256": cfg.MODEL.branch_anchor_sha256,
            "anchor_seed": int(cfg.MODEL.branch_anchor_seed),
            "branch_mode": cfg.MODEL.branch_mode,
            "training_regime": "frozen_branch" if model.frozen else "joint_ce",
            "new_registered_params": sum(p.numel() for p in model.router.parameters()),
            "new_trainable_params": sum(p.numel() for p in model.router.parameters() if p.requires_grad),
            "base_trainable_params": sum(p.numel() for p in model.base.parameters() if p.requires_grad)})
        best = -float("inf")
        best_path = run / "checkpoints/best_branch.pth"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(run / "status.json", {"status": "training"})
        started = time.time()
        training_peak_mib = 0.0
        tau = float(cfg.MODEL.cosine_cls_temp)
        for epoch in range(1, int(cfg.TRAIN.max_epoch) + 1):
            model.train(True)
            ai, oi, pairs = candidates(model, ds, train=True)
            loss_total, n, nonzero_grad_params = 0.0, 0, 0
            torch.cuda.reset_peak_memory_stats()
            with (run / "train.log").open("a", encoding="utf-8") as log:
                for index, batch in enumerate(loader):
                    batch = move(batch, "cuda:0")
                    base_scores, feature = model.components(batch)
                    residual = model.router(feature, ai, oi, force_off=model.off_mode)
                    clean_loss = F.cross_entropy((base_scores + residual) / tau, batch["pair"])
                    loss = clean_loss
                    beta = float(getattr(cfg.TRAIN, "branch_fallback_beta", 0.0))
                    if beta > 0 and (index % 5 == 0):
                        noise_seed = int(cfg.TRAIN.seed) + 100000 * epoch + index
                        with model.corrupt(0.1, noise_seed):
                            noisy = base_scores + model.router(feature, ai, oi, force_off=model.off_mode)
                            fallback = F.kl_div(F.log_softmax(noisy / tau, dim=1),
                                                F.softmax(base_scores.detach() / tau, dim=1),
                                                reduction="batchmean")
                            loss = loss + beta * fallback
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite branch loss")
                    optimizer.zero_grad()
                    loss.backward()
                    grads = [p.grad for p in model.router.parameters() if p.grad is not None]
                    if any(not torch.isfinite(g).all() for g in grads):
                        raise ValueError("Non-finite branch gradient")
                    nonzero_grad_params = sum(int(g.abs().sum() > 0) for g in grads)
                    if model.frozen and any(p.grad is not None for p in model.base.parameters()):
                        raise ValueError("Frozen C unexpectedly received gradients")
                    optimizer.step()
                    size = batch["pair"].size(0)
                    n += size
                    loss_total += float(loss.detach()) * size
                    if index % 25 == 0:
                        line = "epoch={} batch={}/{} loss={:.6f} eta={:.6f} grad_params={}\n".format(
                            epoch, index + 1, len(loader), float(loss.detach()),
                            float(model.router.eta.detach()), nonzero_grad_params)
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
            epoch_train_peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
            training_peak_mib = max(training_peak_mib, epoch_train_peak_mib)
            record = evaluate(model, cfg, "val", "cw", run / "epochs" / ("epoch_%02d_cw" % epoch))
            auc = record["metrics_raw"]["AUC"]
            # First strict maximum; no post-hoc epoch/OW scan.
            if auc > best:
                best = auc
                torch.save(snapshot_branch(model, cfg, epoch), best_path)
            epoch_record = {"epoch": epoch, "loss_mean": loss_total / n,
                            "val_cw_auc_pct": auc * 100,
                            "val_cw_hm_pct": record["metrics_raw"]["best_hm"] * 100,
                            "eta": float(model.router.eta.detach()),
                            "last_nonzero_gradient_parameter_count": nonzero_grad_params,
                            "training_peak_gpu_mib": epoch_train_peak_mib,
                            "elapsed_seconds": time.time() - started}
            with (run / "epoch_metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(epoch_record) + "\n")
        end_state = state_digest(model.base) if model.frozen else None
        if model.frozen and start_state != end_state:
            raise ValueError("Frozen C state changed")
        write_json(run / "train_complete.json", {"checkpoint_sha256": sha(best_path),
                   "epochs": int(cfg.TRAIN.max_epoch), "base_state_unchanged": model.frozen,
                   "base_state_start": start_state, "base_state_end": end_state,
                   "training_peak_gpu_mib": training_peak_mib,
                   "train_seconds": time.time() - started})
        del optimizer, model
        torch.cuda.empty_cache()
    if not (run / "train_complete.json").is_file():
        raise ValueError("No successful branch training record")
    target = run / "checkpoints/best_branch.pth"
    if sha(target) != read_json(run / "train_complete.json")["checkpoint_sha256"]:
        raise ValueError("Branch checkpoint changed")
    model, saved = load_branch(cfg, trainset(cfg), target)
    model.evaluation_checkpoint_sha256 = sha(target)
    values = {}
    for setting in ("cw", "ow"):
        record = eval_or_read(model, cfg, setting, run / ("eval_val_" + setting))
        record.update(checkpoint_sha256=sha(target), best_epoch=saved["epoch"],
                      anchor_sha256=saved["anchor_sha256"], anchor_seed=cfg.MODEL.branch_anchor_seed,
                      training_regime="frozen_branch" if model.frozen else "joint_ce")
        write_json(run / ("eval_val_" + setting) / "metrics.json", record)
        values[setting] = record
    if model.frozen:
        anchors = {}
        for setting in ("cw", "ow"):
            anchor_cfg = cfg.clone()
            anchor_cfg.TRAIN.kcc_variant = "C0"
            record = eval_or_read(model, anchor_cfg, setting,
                                 run / ("anchor_val_" + setting), force_off=True)
            anchors[setting] = record
        dw = metrics_pct(values["cw"])
        do = metrics_pct(values["ow"])
        aw = metrics_pct(anchors["cw"])
        ao = metrics_pct(anchors["ow"])
        conditions = {"cw_auc_strict_gain": dw["auc_pct"] > aw["auc_pct"] + 1e-6,
                      "cw_hm_guard": dw["hm_pct"] >= aw["hm_pct"] - 0.5,
                      "ow_auc_guard": do["auc_pct"] >= ao["auc_pct"] - 0.5,
                      "ow_unseen_guard": do["best_unseen_pct"] >= ao["best_unseen_pct"] - 2}
        write_json(run / "selected_state.json", {
            "selection": "CW-best epoch, then guards; no other OW epoch scan",
            "conditions": conditions, "adopted": all(conditions.values()),
            "state": "trained_branch" if all(conditions.values()) else "residual_off",
            "candidate_checkpoint": str(target), "anchor": cfg.MODEL.branch_anchor,
            "delta_cw_auc_pp": dw["auc_pct"] - aw["auc_pct"],
            "delta_ow_auc_pp": do["auc_pct"] - ao["auc_pct"]})
    else:
        write_json(run / "selected_state.json", {"state": "candidate_requires_joint_controls",
                   "candidate_checkpoint": str(target), "warning": "Compare with matched C_budget before adopting"})
    write_json(run / "status.json", {"status": "train_and_val_complete"})
