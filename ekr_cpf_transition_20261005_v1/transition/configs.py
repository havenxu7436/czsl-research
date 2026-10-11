from pathlib import Path
import yaml

from .common import (BRANCH_MODES, DATASETS, SEMANTIC_MODES, read_json,
                     sha, write_json)


def recover_config(path):
    path = Path(path)
    text = (path / "train.log").read_text(encoding="utf-8", errors="replace")
    if "Random seed:" not in text:
        raise ValueError("Merged training config cannot be recovered: " + str(path))
    cfg = yaml.safe_load(text.split("Random seed:", 1)[0])
    for key in ("DATASET", "MODEL", "TRAIN", "EVAL"):
        if not isinstance(cfg.get(key), dict):
            raise ValueError("Invalid merged config section: " + key)
    if int(cfg["TRAIN"]["max_epoch"]) != 10 or int(cfg["TRAIN"]["aug_epoch"]) <= 10:
        raise ValueError("This protocol requires the audited 10-epoch non-augmentation setup")
    if cfg["MODEL"]["wordembs"] != "glove":
        raise ValueError("Only the audited GloVe protocol is supported")
    for key in ("lr", "lr_encoder", "lr_word_embedding", "wd", "decay_factor"):
        if key in cfg["TRAIN"]:
            cfg["TRAIN"][key] = float(cfg["TRAIN"][key])
    return cfg


def make(args):
    root, exp, legacy = Path(args.root), Path(args.exp), Path(args.legacy_exp)
    requested = {}
    gate_value = None
    if args.stage == "branch" and not args.smoke_only:
        if not args.gate:
            raise ValueError("Formal branch configs require a semantic gate decision")
        gate_value = read_json(args.gate)
    for dataset in args.datasets:
        folder, native_seed = DATASETS[dataset]
        original_c = legacy / "runs" / "mvp_{}_C_legacy_s{}_l0".format(dataset, native_seed)
        template = recover_config(original_c)
        bank = Path(args.bank_root) / folder
        if not (bank / "manifest.json").is_file():
            raise FileNotFoundError("Verified bank required: " + str(bank))
        for seed in args.seeds:
            variants = SEMANTIC_MODES if args.stage == "semantic" else BRANCH_MODES
            default_variants = list(variants) if args.stage == "semantic" else ["R0", "R3", "R5"]
            for variant in args.variants or default_variants:
                if variant not in variants:
                    raise ValueError("Unknown stage variant: " + variant)
                cfg = yaml.safe_load(yaml.safe_dump(template))
                stage_label = "joint" if args.stage == "branch" and args.regime == "joint" else args.stage
                if args.stage == "branch" and args.smoke_only:
                    stage_label = "smoke"
                if variant == "C_budget" and stage_label != "joint":
                    raise ValueError("C_budget is a joint-training control, not a frozen branch")
                rid = "{}_{}_{}_s{}_e{}".format(stage_label, dataset, variant, seed, args.epochs)
                run = exp / "runs" / rid
                cfg["DATASET"]["root_dir"] = str(root / "data" / folder)
                cfg["EVAL"].update(open_world=False, topk=1)
                cfg["MODEL"].update(kcc_calib=False, kcc_lambda=0.0,
                                    kcc_feature=False, kcc_logic=False, kcc_control="none",
                                    kcc_cache=str(root / "knowledge/kcc0601_v1/cache_trusted_v1" / folder),
                                    description_bank=str(bank))
                cfg["TRAIN"].update(seed=seed, max_epoch=args.epochs, final_max_epoch=args.epochs,
                                    start_epoch=1, kcc_test_at_end=False,
                                    kcc_run_dir=str(run), checkpoint_dir=str(run / "checkpoints"),
                                    log_dir=str(run / "logs"), train_name=rid,
                                    kcc_stage=stage_label, kcc_variant=variant)
                if args.stage == "semantic":
                    mode = SEMANTIC_MODES[variant]
                    cfg["MODEL"].update(semantic_text=mode, kcc_text=(mode != "raw"))
                else:
                    gate_dataset = args.transfer_from or dataset
                    if gate_value is not None and not gate_value["datasets"][gate_dataset]["proceed"]:
                        raise ValueError("Semantic screening gate did not pass: " + gate_dataset)
                    anchor_seed = seed
                    anchor = exp / "runs" / "semantic_{}_C_text_s{}_e10".format(dataset, seed) / "checkpoints/best_auc.pth"
                    anchor_cfg = exp / "configs" / "semantic_{}_C_text_s{}_e10.yml".format(dataset, seed)
                    if seed == native_seed and args.use_legacy_anchor:
                        anchor = original_c / "checkpoints/best_auc.pth"
                        anchor_cfg = original_c / "config_requested.yml"
                    if not anchor.is_file():
                        raise FileNotFoundError("C anchor missing; complete matched semantic C first: " + str(anchor))
                    cfg["MODEL"].update(kcc_text=True, semantic_text="correct",
                                        branch_mode=BRANCH_MODES[variant], branch_rank=32,
                                        branch_eta_max=0.1, branch_temperature=0.2,
                                        branch_object_chunk=16, branch_anchor=str(anchor),
                                        branch_regime=args.regime,
                                        branch_anchor_config=str(anchor_cfg),
                                        branch_anchor_seed=anchor_seed,
                                        branch_anchor_sha256=sha(anchor))
                    cfg["TRAIN"].update(branch_lr=1e-3, branch_weight_decay=1e-4,
                                        branch_deterministic=True,
                                        branch_fallback_beta=0.05 if variant in {"R6", "R0noise", "R3noise"} else 0.0,
                                        branch_smoke_only=args.smoke_only,
                                        semantic_gate_file=str(args.gate or ""),
                                        semantic_gate_sha256=sha(args.gate) if args.gate else "",
                                        semantic_gate_dataset=gate_dataset,
                                        cross_dataset_transfer=bool(args.transfer_from and args.transfer_from != dataset))
                requested[exp / "configs" / (rid + ".yml")] = yaml.safe_dump(
                    cfg, allow_unicode=True, sort_keys=True)
    # Check every destination before writing any config.
    for path, content in requested.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError("Different config already exists: " + str(path))
    for path, content in requested.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        print(path)
    stage_label = "joint" if args.stage == "branch" and args.regime == "joint" else args.stage
    if args.stage == "branch" and args.smoke_only:
        stage_label = "smoke"
    write_json(exp / "config_index_{}_e{}.json".format(stage_label, args.epochs),
               {path.stem: str(path) for path in requested})
