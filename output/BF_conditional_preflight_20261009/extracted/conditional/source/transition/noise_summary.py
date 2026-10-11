"""Keep perturbation repeats separate from independent training seeds."""
from pathlib import Path

import numpy as np

from .common import empty_output, metrics_pct, read_json, sha, write_json
from .summary import write_csv


def summarize_noise(args):
    exp = Path(args.exp)
    rows, inputs, missing = [], {}, []
    run_names = args.runs or [p.name for p in sorted((exp / "runs").glob("*"))
                             if (p / args.noise_version).is_dir()]
    if not run_names:
        raise ValueError("No noise runs found")
    for name in run_names:
        run = exp / "runs" / name
        for setting in ("cw", "ow"):
            clean_path = run / ("eval_val_" + setting) / "metrics.json"
            clean = read_json(clean_path)
            cm = metrics_pct(clean)
            inputs[str(clean_path)] = sha(clean_path)
            for ratio in (0.1, 0.3):
                for seed in (11, 22, 33):
                    path = run / args.noise_version / ("p{}_n{}_{}".format(ratio, seed, setting)) / "metrics.json"
                    if not path.is_file():
                        missing.append(str(path))
                        continue
                    value = read_json(path)
                    for key in ("dataset", "variant", "seed", "phase", "setting", "checkpoint_sha256"):
                        if value[key] != clean[key]:
                            raise ValueError("Noise/clean identity mismatch: " + str(path))
                    if value["phase"] != "val" or value["noise_ratio"] != ratio or value["noise_seed"] != seed:
                        raise ValueError("Noise experiment identity mismatch")
                    nm = metrics_pct(value)
                    inputs[str(path)] = sha(path)
                    row = {"run_id": name, "dataset": value["dataset"], "variant": value["variant"],
                           "training_seed": value["seed"], "noise_seed": seed,
                           "nominal_noise_ratio": ratio, "setting": setting,
                           "actual_noise_ratio": (value.get("corruption") or {}).get("actual_fraction", ratio),
                           "checkpoint_sha256": value["checkpoint_sha256"],
                           "anchor_sha256": clean["anchor_sha256"], **nm}
                    for metric in ("auc_pct", "hm_pct", "best_unseen_pct"):
                        row["clean_" + metric] = cm[metric]
                        row["drop_" + metric + "_pp"] = cm[metric] - nm[metric]
                    rows.append(row)
    if missing:
        raise ValueError("Noise matrix incomplete; first missing: " + missing[0])
    groups = {}
    keys = ("dataset", "variant", "training_seed", "setting", "nominal_noise_ratio")
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    averages = []
    for key, items in groups.items():
        record = dict(zip(keys, key), n_noise_seeds=len(items))
        for metric in ("auc_pct", "hm_pct", "best_unseen_pct"):
            values = np.array([r[metric] for r in items])
            drops = np.array([r["drop_" + metric + "_pp"] for r in items])
            record[metric + "_mean"] = float(values.mean())
            record[metric + "_noise_sample_std"] = float(values.std(ddof=1))
            record["drop_" + metric + "_pp_mean"] = float(drops.mean())
        averages.append(record)
    paired = []
    for treatment, reference in (("R6", "R5"), ("R6", "R0noise"), ("R6", "R3noise")):
        for t in [r for r in rows if r["variant"] == treatment]:
            match = [r for r in rows if r["variant"] == reference and all(
                r[k] == t[k] for k in ("dataset", "training_seed", "noise_seed", "setting",
                                       "nominal_noise_ratio", "anchor_sha256"))]
            if len(match) == 1:
                r = match[0]
                paired.append({k: t[k] for k in ("dataset", "training_seed", "noise_seed",
                               "setting", "nominal_noise_ratio")})
                paired[-1].update(comparison=treatment + "-" + reference,
                    delta_auc_pp=t["auc_pct"] - r["auc_pct"],
                    delta_hm_pp=t["hm_pct"] - r["hm_pct"])
    out = empty_output(args.out)
    write_csv(out / "noise_metrics.csv", rows)
    write_csv(out / "noise_summary_by_training_seed.csv", averages)
    write_csv(out / "noise_paired_deltas.csv", paired)
    write_json(out / "collection.json", {"inputs": inputs,
        "meaning": "Noise-seed SD measures perturbation variation, not independent training SD"})
    print("NOISE SUMMARY", out, len(rows), "observations", flush=True)
