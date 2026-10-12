import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .common import DATASETS, metrics_pct, read_json, sha, write_json


def write_csv(path, rows):
    columns = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(args):
    exp, out = Path(args.exp), Path(args.out)
    rows, inputs = [], {}
    for run in sorted((exp / "runs").glob("*")):
        if not (run / "status.json").is_file():
            continue
        if read_json(run / "status.json")["status"] != "train_and_val_complete":
            continue
        cfg = __import__("yaml").safe_load((run / "effective_config.yml").read_text())
        for setting in ("cw", "ow"):
            path = run / ("eval_val_" + setting) / "metrics.json"
            m = read_json(path)
            inputs[str(path)] = sha(path)
            row = {"dataset": m["dataset"], "variant": m["variant"], "seed": m["seed"],
                   "phase": m["phase"], "setting": setting, "run_id": run.name,
                   "regime": m["training_regime"], "epochs": cfg["TRAIN"]["max_epoch"],
                   "checkpoint_sha256": m["checkpoint_sha256"], "reused_legacy": False,
                   "anchor_sha256": m.get("anchor_sha256", ""),
                   "anchor_seed": m.get("anchor_seed", ""), **metrics_pct(m)}
            for key in ("best_epoch", "eval_seconds", "peak_gpu_mib", "bank_manifest_sha256"):
                row[key] = m.get(key, "")
            for filename, key in (("effective_config.yml", "training_config_sha256"),
                                  ("meta/code_manifest.json", "code_manifest_sha256")):
                if (run / filename).is_file():
                    row[key] = sha(run / filename)
                    inputs[str(run / filename)] = row[key]
            if (run / "train_complete.json").is_file():
                complete = read_json(run / "train_complete.json")
                row["training_stage_seconds_including_cw_val"] = complete.get("train_seconds", "")
                row["training_peak_gpu_mib"] = complete.get("training_peak_gpu_mib", "")
            if (run / "meta/branch_identity.json").is_file():
                identity = read_json(run / "meta/branch_identity.json")
                for key in ("new_registered_params", "new_trainable_params", "base_trainable_params"):
                    row[key] = identity[key]
            row.update(m.get("diagnostics", {}))
            if (run / "selected_state.json").is_file():
                row["adopted"] = read_json(run / "selected_state.json").get("adopted", "")
                inputs[str(run / "selected_state.json")] = sha(run / "selected_state.json")
            rows.append(row)
            anchor = run / ("anchor_val_" + setting) / "metrics.json"
            if anchor.is_file():
                a = read_json(anchor)
                inputs[str(anchor)] = sha(anchor)
                anchor_row = dict(row, variant="C0", checkpoint_sha256=m["anchor_sha256"],
                                  **metrics_pct(a))
                anchor_row.update(eta=0.0, mean_abs_residual=0.0, max_abs_residual=0.0,
                                  null_mean="", adopted="", best_epoch="",
                                  new_registered_params=0, new_trainable_params=0,
                                  base_trainable_params=0, training_peak_gpu_mib="",
                                  training_stage_seconds_including_cw_val="",
                                  eval_seconds=a.get("eval_seconds", ""),
                                  peak_gpu_mib=a.get("peak_gpu_mib", ""))
                rows.append(anchor_row)
    if args.legacy_exp:
        for ds, (folder, seed) in DATASETS.items():
            for old_variant, new_variant in (("A", "A_raw"), ("C", "C_text")):
                run = Path(args.legacy_exp) / "runs" / "mvp_{}_{}_legacy_s{}_l0".format(ds, old_variant, seed)
                for setting in ("cw", "ow"):
                    identity = (ds, new_variant, seed, "val", setting, "native_semantic", 10)
                    if any((r["dataset"], r["variant"], r["seed"], r["phase"], r["setting"],
                            r["regime"], r["epochs"]) == identity for r in rows):
                        continue
                    path = run / ("eval_val_" + setting) / "metrics.json"
                    if not path.is_file():
                        continue
                    m = read_json(path)
                    from .configs import recover_config
                    old_cfg = recover_config(run)
                    if (m["phase"] != "val" or m["setting"] != setting or
                            m["dataset"] != ds or int(m["seed"]) != seed or
                            m.get("score_mode", "legacy") != "legacy" or
                            old_cfg["DATASET"]["name"] != ds or
                            int(old_cfg["TRAIN"]["seed"]) != seed or
                            bool(old_cfg["MODEL"]["kcc_calib"]) or
                            float(old_cfg["MODEL"]["kcc_lambda"]) != 0 or
                            bool(old_cfg["MODEL"]["kcc_text"]) != (old_variant == "C")):
                        raise ValueError("Legacy protocol mismatch: " + str(run))
                    inputs[str(run / "train.log")] = sha(run / "train.log")
                    checkpoint = run / "checkpoints/best_auc.pth"
                    if sha(checkpoint) != m["checkpoint_sha256"]:
                        raise ValueError("Legacy checkpoint hash mismatch: " + str(run))
                    inputs[str(path)] = sha(path)
                    rows.append({"dataset": ds, "variant": new_variant, "seed": seed,
                                 "phase": "val", "setting": setting,
                                 "run_id": run.name, "regime": "native_semantic", "epochs": 10,
                                 "checkpoint_sha256": m["checkpoint_sha256"],
                                 "reused_legacy": True, "anchor_sha256": "", "anchor_seed": "",
                                 **metrics_pct(m)})
    # Deduplicate C0 observations only; they are the same frozen anchor, not independent repeats.
    unique = {}
    for r in rows:
        key = tuple(r.get(k, "") for k in ("dataset", "variant", "seed", "phase", "setting",
                                            "regime", "epochs", "anchor_sha256"))
        if key in unique:
            if any(abs(unique[key][metric] - r[metric]) > 1e-6
                   for metric in ("auc_pct", "hm_pct", "best_unseen_pct")):
                raise ValueError("Duplicate identity with inconsistent metrics")
        else:
            unique[key] = r
    rows = list(unique.values())
    comparisons = [("A_unit", "A_raw"), ("C_text", "A_unit"), ("C_text", "C_shuffle"),
                   ("C_text", "A_raw"), ("R5", "C0"), ("R5", "R0"), ("R5", "R3"),
                   ("R5", "R1"), ("R5", "R2"), ("R5", "R4"), ("R5", "R7"),
                   ("R5", "Rname"), ("R5", "Rvalue"), ("R6", "R5"),
                   ("R6", "R0noise"), ("R6", "R3noise"), ("R5", "C_budget")]
    paired = []
    for treatment, reference in comparisons:
        for t in [r for r in rows if r["variant"] == treatment]:
            matched = [r for r in rows if r["variant"] == reference and all(
                r.get(k) == t.get(k) for k in ("dataset", "seed", "phase", "setting", "regime",
                                             "epochs", "anchor_sha256"))]
            if len(matched) == 1:
                r = matched[0]
                paired.append({k: t.get(k, "") for k in ("dataset", "seed", "phase", "setting",
                               "regime", "epochs", "anchor_sha256")})
                paired[-1].update(comparison=treatment + "-" + reference,
                    delta_auc_pp=t["auc_pct"] - r["auc_pct"],
                    delta_hm_pp=t["hm_pct"] - r["hm_pct"],
                    delta_unseen_pp=t["best_unseen_pct"] - r["best_unseen_pct"])
    stats = []
    group_keys = ("dataset", "comparison", "phase", "setting", "regime", "epochs")
    groups = {}
    for row in paired:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    for key, items in groups.items():
        record = dict(zip(group_keys, key), n_seeds=len(items), seeds=";".join(str(r["seed"]) for r in items))
        for metric in ("delta_auc_pp", "delta_hm_pp", "delta_unseen_pp"):
            a = np.array([r[metric] for r in items])
            record[metric + "_mean"] = float(a.mean())
            record[metric + "_sample_std"] = float(a.std(ddof=1)) if len(a) > 1 else ""
        stats.append(record)
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if out.exists() and any(out.iterdir()):
        existing = out / "collection.json"
        if existing.is_file() and read_json(existing)["fingerprint"] == fingerprint:
            print("SUMMARY ALREADY COMPLETE", out)
            return
        raise FileExistsError("New inputs need a new report version: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", rows)
    write_csv(out / "paired_deltas.csv", paired)
    write_csv(out / "paired_summary.csv", stats)
    write_json(out / "collection.json", {"inputs": inputs, "fingerprint": fingerprint})
    print("SUMMARY", out, "rows", len(rows), "paired comparisons", len(paired))


def gate(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be distinct")
    report = Path(args.report)
    with (report / "runs.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for dataset in args.datasets:
        subset = [r for r in rows if r["dataset"] == dataset and r["phase"] == "val" and
                  r["regime"] == "native_semantic" and int(r["epochs"]) == 10]
        values = {(r["variant"], int(r["seed"]), r["setting"]): r for r in subset}
        missing = [(v, s, setting) for v in ("A_raw", "A_unit", "C_text", "C_shuffle")
                   for s in args.seeds for setting in ("cw", "ow")
                   if (v, s, setting) not in values]
        if missing:
            result[dataset] = {"proceed": False, "reason": "Incomplete matched semantic controls",
                               "missing": missing}
            continue
        def differences(t, r, setting, metric):
            return [float(values[t, s, setting][metric]) - float(values[r, s, setting][metric])
                    for s in args.seeds]
        text = differences("C_text", "A_unit", "cw", "auc_pct")
        shuffled = differences("C_text", "C_shuffle", "cw", "auc_pct")
        conditions = {"at_least_three_seeds": len(args.seeds) >= 3,
            "mean_text_over_unit_at_least_0.5pp": np.mean(text) >= 0.5,
            "mean_text_over_shuffle_at_least_0.5pp": np.mean(shuffled) >= 0.5,
            "text_positive_on_two_thirds_seeds": sum(x > 0 for x in text) >= math.ceil(2 * len(text) / 3),
            "shuffle_positive_on_two_thirds_seeds": sum(x > 0 for x in shuffled) >= math.ceil(2 * len(shuffled) / 3),
            "C_not_below_Araw_mean": np.mean(differences("C_text", "A_raw", "cw", "auc_pct")) >= 0,
            "cw_hm_guard": np.mean(differences("C_text", "A_unit", "cw", "hm_pct")) >= -0.5,
            "ow_auc_guard": np.mean(differences("C_text", "A_unit", "ow", "auc_pct")) >= -0.5,
            "ow_unseen_guard": np.mean(differences("C_text", "A_unit", "ow", "best_unseen_pct")) >= -2}
        conditions = {k: bool(v) for k, v in conditions.items()}
        result[dataset] = {"proceed": all(conditions.values()), "conditions": conditions,
                           "C_minus_Aunit_cw_auc_pp": text,
                           "C_minus_shuffle_cw_auc_pp": shuffled,
                           "interpretation": "Operational screening gate, not statistical proof"}
    output = Path(args.out)
    if output.exists():
        raise FileExistsError("Gate decision exists; use a new decision version")
    write_json(output, {"planned_seeds": args.seeds, "datasets": result,
                        "report_fingerprint": read_json(report / "collection.json")["fingerprint"]})
    print(json.dumps(result, indent=2), flush=True)
