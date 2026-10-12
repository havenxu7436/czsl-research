#!/usr/bin/env python3
"""Make A/C/D/E configs from the baseline's effective train.log settings."""
import argparse
import json
from pathlib import Path

import yaml

RUNS = {"ut-zappos": ("cpf_ut_zappos_seed3351", "ut-zap50k"),
        "mit-states": ("cpf_mit_states_seed0", "mit-states"),
        "cgqa": ("cpf_cgqa_seed3351_gpu2", "cgqa")}
FLAGS = {"A": (False, False), "C": (True, False),
         "D": (False, True), "E": (True, True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exp", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=RUNS, default="ut-zappos")
    parser.add_argument("--seed", type=int, default=3351)
    parser.add_argument("--lambda-k", type=float, default=0.1)
    args = parser.parse_args()
    if args.lambda_k < 0:
        parser.error("lambda-k must be nonnegative")
    old_run, folder = RUNS[args.dataset]
    log = args.root / "runs" / old_run / "train.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    if "Random seed:" not in text:
        raise ValueError("cannot identify baseline effective config: " + str(log))
    baseline = yaml.safe_load(text.split("Random seed:", 1)[0])
    for section in ("MODEL", "DATASET", "TRAIN", "EVAL"):
        if not isinstance(baseline.get(section), dict):
            raise ValueError("invalid baseline config section: " + section)
    if baseline["MODEL"]["wordembs"] != "glove":
        raise ValueError("requires the actual GloVe baseline")
    for key in ("lr", "lr_encoder", "lr_word_embedding", "wd", "decay_factor"):
        if key in baseline["TRAIN"]:
            baseline["TRAIN"][key] = float(baseline["TRAIN"][key])
    cache = args.cache_root.resolve() / folder
    if not (cache / "manifest.json").is_file():
        raise FileNotFoundError("knowledge cache missing: " + str(cache))
    if int(baseline["TRAIN"]["aug_epoch"]) <= 10:
        raise ValueError("baseline augmentation enters the first 10 epochs; audit before this MVP")
    requested = {}
    for variant, (txt, cal) in FLAGS.items():
        c = yaml.safe_load(yaml.safe_dump(baseline))
        lam = args.lambda_k if cal else 0.0
        rid = "mvp_{}_{}_legacy_s{}_l{}".format(args.dataset, variant, args.seed, format(lam, ".6g"))
        run = args.exp.resolve() / "runs" / rid
        c["DATASET"]["root_dir"] = str(args.root.resolve() / "data" / folder)
        c["EVAL"].update(open_world=False, topk=1)
        c["TRAIN"].update(train_name=rid, seed=args.seed, start_epoch=1,
                          max_epoch=10, final_max_epoch=10,
                          start_epoch_validate=1, eval_every_epoch=1,
                          log_dir=str(run / "logs"), checkpoint_dir=str(run / "checkpoints"),
                          kcc_run_dir=str(run), kcc_stage="mvp", kcc_variant=variant,
                          kcc_test_at_end=False)
        c["MODEL"].update(kcc_text=txt, kcc_calib=cal, kcc_feature=False, kcc_logic=False,
                          kcc_lambda=lam, kcc_beta=0.0, kcc_score_mode="legacy", kcc_control="none",
                          kcc_cache=str(cache))
        path = args.exp.resolve() / "configs" / (rid + ".yml")
        requested[path] = yaml.safe_dump(c, allow_unicode=True, sort_keys=True)
    # Preflight all paths before any writes.
    for path, content in requested.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError("different config already exists: " + str(path))
    for path, content in requested.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        print(path)
    index = args.exp.resolve() / "configs" / ("index_{}_s{}.json".format(args.dataset, args.seed))
    index.write_text(json.dumps({v: str(p) for v, p in zip(FLAGS, requested)}, indent=2), encoding="utf-8")
    print("INDEX: " + str(index))


if __name__ == "__main__":
    main()
