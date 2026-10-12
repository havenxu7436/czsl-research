import argparse
import json
import sys
from pathlib import Path

from .common import DATASETS, code_manifest, read_json, sha, write_json


def preflight(args):
    import numpy
    import torch
    import torchvision
    import scipy
    import yaml
    import yacs
    from torch.nn.modules.pooling import MaxPool2d
    root = Path(args.root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; stop before training")
    if not torch.__version__.startswith("1.13.1") or not torchvision.__version__.startswith("0.14.1"):
        raise RuntimeError("Environment differs from the audited torch1.13.1/vision0.14.1 protocol")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Expected the existing Python3.8 CPF environment")
    for name, (folder, seed) in DATASETS.items():
        for path in [root / "data" / folder / "images",
                     root / "data" / folder / "compositional-split-natural/train_pairs.txt",
                     root / "knowledge/kcc0601_v1/trusted_v1" / folder / "ae_dict.json",
                     root / "knowledge/kcc0601_v1/cache_trusted_v1" / folder / "manifest.json"]:
            if not path.exists():
                raise FileNotFoundError(str(path))
    print("PREFLIGHT PASSED", sys.executable, torch.__version__, torchvision.__version__,
          "GPU", torch.cuda.get_device_name(0), flush=True)


def evaluate_command(args):
    from .runtime import base_model, load_branch, read_cfg, trainset
    from .evaluate import evaluate
    cfg = read_cfg(args.cfg)
    off = args.force_off
    if args.phase == "test":
        if not args.protocol:
            raise ValueError("Final test requires freeze-final protocol.json")
        protocol = read_json(args.protocol)
        if protocol["code_manifest"] != code_manifest(Path.cwd()):
            raise ValueError("Code changed after final protocol was frozen")
        identity = (sha(args.cfg), sha(args.load))
        matches = [r for r in protocol["runs"]
                   if (r["config_sha256"], r["checkpoint_sha256"]) == identity]
        if len(matches) != 1:
            raise ValueError("Config/checkpoint absent from frozen final protocol")
        for path, digest in matches[0]["artifacts_sha256"].items():
            if sha(path) != digest:
                raise ValueError("Frozen final protocol artifact changed: " + path)
        cfg.TRAIN.transition_test_allowed = True
        off = matches[0]["force_off"]
        if args.noise_ratio != 0:
            raise ValueError("Final test protocol does not include post-hoc noise changes")
    ds = trainset(cfg)
    if hasattr(cfg.MODEL, "branch_mode"):
        model, saved = load_branch(cfg, ds, args.load)
    else:
        model = base_model(ds, cfg, args.load)
    model.evaluation_checkpoint_sha256 = sha(args.load)
    if args.noise_ratio and not hasattr(model, "router"):
        raise ValueError("Noise evaluation here alters only the supplemental branch")
    if hasattr(model, "router"):
        with model.corrupt(args.noise_ratio, args.noise_seed):
            value = evaluate(model, cfg, args.phase, args.setting, args.out,
                             save_scores=args.save_scores, force_off=off)
    else:
        value = evaluate(model, cfg, args.phase, args.setting, args.out,
                         save_scores=args.save_scores, force_off=off)
    value.update(noise_ratio=args.noise_ratio, noise_seed=args.noise_seed,
                 corruption=getattr(model, "last_corruption", None),
                 final_protocol_sha256=sha(args.protocol) if args.protocol else None)
    write_json(Path(args.out) / "metrics.json", value)


def freeze_final(args):
    runs = []
    for name in args.runs:
        run = Path(args.exp) / "runs" / name
        cfg = Path(args.exp) / "configs" / (name + ".yml")
        legacy_reused = False
        if not (run / "status.json").is_file():
            if not args.legacy_exp:
                raise ValueError("Run absent; native A/C reuse requires --legacy-exp: " + name)
            import yaml
            from .configs import recover_config
            values = yaml.safe_load(cfg.read_text())
            dataset, seed = values["DATASET"]["name"], int(values["TRAIN"]["seed"])
            variant = values["TRAIN"]["kcc_variant"]
            if variant not in {"A_raw", "C_text"} or seed != DATASETS[dataset][1] or "branch_mode" in values["MODEL"]:
                raise ValueError("Only unchanged native-seed A/C can be reused for final tests")
            old_variant = "A" if variant == "A_raw" else "C"
            run = Path(args.legacy_exp) / "runs" / "mvp_{}_{}_legacy_s{}_l0".format(dataset, old_variant, seed)
            old_cfg = recover_config(run)
            if (run / "status.txt").read_text().strip() != "train_and_val_complete":
                raise ValueError("Legacy run incomplete")
            for section, keys in (("MODEL", ("wordembs", "cosine_cls_temp", "kcc_text", "kcc_calib", "kcc_lambda")),
                                  ("TRAIN", ("seed", "max_epoch", "batch_size", "lr", "lr_encoder", "wd")),
                                  ("DATASET", ("name", "splitname"))):
                if any(old_cfg[section][key] != values[section][key] for key in keys):
                    raise ValueError("Legacy final-test config differs from training protocol")
            legacy_reused = True
        elif read_json(run / "status.json")["status"] != "train_and_val_complete":
            raise ValueError("Run incomplete: " + name)
        checkpoint = run / "checkpoints" / ("best_branch.pth" if
                     (run / "checkpoints/best_branch.pth").is_file() else "best_auc.pth")
        selected = read_json(run / "selected_state.json") if (run / "selected_state.json").is_file() else {}
        complete = (read_json(run / "eval_val_cw/metrics.json") if legacy_reused
                    else read_json(run / "train_complete.json"))
        if sha(checkpoint) != complete["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed before final freeze")
        if legacy_reused:
            artifacts = [run / "train.log", run / "status.txt"]
            artifacts.append(Path(values["MODEL"]["description_bank"]) / "manifest.json")
        else:
            artifacts = [run / "effective_config.yml", run / "meta/code_manifest.json",
                         run / "meta/bank_manifest.json", run / "meta/cache_manifest.json"]
        artifacts += [run / "eval_val_cw/metrics.json", run / "eval_val_ow/metrics.json"]
        if (run / "selected_state.json").is_file():
            artifacts.append(run / "selected_state.json")
        runs.append({"run_id": name, "config": str(cfg), "checkpoint": str(checkpoint),
                     "config_sha256": sha(cfg), "checkpoint_sha256": sha(checkpoint),
                     "legacy_reused": legacy_reused,
                     "artifacts_sha256": {str(p): sha(p) for p in artifacts},
                     "force_off": selected.get("state") == "residual_off",
                     "validation_selection": selected})
    if Path(args.out).exists():
        raise FileExistsError("Final protocol exists; do not overwrite after inspecting test")
    write_json(args.out, {"version": "final_test_freeze_v1", "runs": runs,
                         "code_manifest": code_manifest(Path.cwd()),
                         "rule": "No model/text/epoch changes after final test begins"})
    print("FINAL PROTOCOL FROZEN", args.out)


def parser():
    p = argparse.ArgumentParser(description="CPF semantic attribution -> EKR -> mechanism verification")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("preflight")
    s.add_argument("--root", required=True)
    s = sub.add_parser("build-bank")
    for name in ("project", "cache-root", "knowledge-root", "out-root", "glove"):
        s.add_argument("--" + name, required=True)
    s.add_argument("--datasets", nargs="+", choices=[v[0] for v in DATASETS.values()],
                   default=[v[0] for v in DATASETS.values()])
    s.add_argument("--permutation-seed", type=int, default=1729)
    s = sub.add_parser("make-configs")
    for name in ("root", "exp", "legacy-exp", "bank-root"):
        s.add_argument("--" + name, required=True)
    s.add_argument("--stage", choices=["semantic", "branch"], required=True)
    s.add_argument("--cache-root", help="Explicit isolated v2 cache root; use with v2 banks")
    s.add_argument("--semantic-exp", help="Separate completed v2 semantic experiment supplying C anchors")
    s.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=["ut-zappos"])
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 3351, 2026])
    s.add_argument("--epochs", type=int, default=10)
    s.add_argument("--variants", nargs="+")
    s.add_argument("--use-legacy-anchor", action="store_true")
    s.add_argument("--regime", choices=["frozen", "joint"], default="frozen")
    s.add_argument("--smoke-only", action="store_true")
    s.add_argument("--gate")
    s.add_argument("--transfer-from", choices=list(DATASETS))
    s = sub.add_parser("smoke")
    s.add_argument("--cfg", required=True)
    s.add_argument("--checkpoint")
    s.add_argument("--out", required=True)
    for command in ("run-semantic", "train-branch"):
        s = sub.add_parser(command)
        s.add_argument("--cfg", required=True)
        s.add_argument("--finish-only", action="store_true")
    s = sub.add_parser("eval")
    for name in ("cfg", "load", "out"):
        s.add_argument("--" + name, required=True)
    s.add_argument("--phase", choices=["val", "test"], default="val")
    s.add_argument("--setting", choices=["cw", "ow"], required=True)
    s.add_argument("--force-off", action="store_true")
    s.add_argument("--save-scores", action="store_true")
    s.add_argument("--noise-ratio", type=float, default=0.0)
    s.add_argument("--noise-seed", type=int, default=11)
    s.add_argument("--protocol")
    s = sub.add_parser("summarize")
    s.add_argument("--exp", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--legacy-exp")
    s = sub.add_parser("gate")
    s.add_argument("--report", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=["ut-zappos"])
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 3351, 2026])
    s = sub.add_parser("freeze-final")
    s.add_argument("--exp", required=True)
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--legacy-exp")
    s = sub.add_parser("bias0")
    for name in ("reference", "treatment", "train-pairs", "out"):
        s.add_argument("--" + name, required=True)
    s = sub.add_parser("run-plan")
    s.add_argument("--configs", nargs="+", required=True)
    s = sub.add_parser("noise-summary")
    s.add_argument("--exp", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--runs", nargs="+")
    s.add_argument("--noise-version", default="noise_v1")
    return p


def main():
    args = parser().parse_args()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "build-bank":
        from .bank import build
        build(args)
    elif args.command == "make-configs":
        from .configs import make
        make(args)
    elif args.command == "smoke":
        from .smoke import smoke
        smoke(args)
    elif args.command == "run-semantic":
        from .run import semantic
        semantic(args)
    elif args.command == "train-branch":
        from .run import branch
        branch(args)
    elif args.command == "eval":
        evaluate_command(args)
    elif args.command == "summarize":
        from .summary import summarize
        summarize(args)
    elif args.command == "gate":
        from .summary import gate
        gate(args)
    elif args.command == "freeze-final":
        freeze_final(args)
    elif args.command == "bias0":
        from .diagnostics import bias0
        bias0(args)
    elif args.command == "run-plan":
        from .queue import run_plan
        run_plan(args)
    elif args.command == "noise-summary":
        from .noise_summary import summarize_noise
        summarize_noise(args)


if __name__ == "__main__":
    main()
