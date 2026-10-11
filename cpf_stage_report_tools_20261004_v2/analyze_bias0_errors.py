#!/usr/bin/env python3
"""Compare saved A/C top-1 predictions at bias=0. Requires numpy, no GPU/torch.

Uses a collected report, verifies candidate/image/label order, and reads ONLY
train_pairs.txt for seen membership. Does not reconstruct AUC or best-HM bias.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--setting", choices=["cw", "ow"], default="ow")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error("output exists; choose a new version")
    import numpy as np
    with (args.report / "tables" / "ac_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        matched = {r["dataset"] for r in csv.DictReader(f) if r["setting"] == args.setting and r["comparison_status"] == "matched"}
    args.out.mkdir(parents=True)
    summaries, per_attr, failures, source_hashes = [], [], [], []
    for ds, seed, folder in [("ut-zappos", 3351, "ut-zap50k"), ("mit-states", 0, "mit-states"), ("cgqa", 3351, "cgqa")]:
        if ds not in matched:
            failures.append(dict(dataset=ds, reason="A/C pair not verified as matched; skipped"))
            continue
        try:
            runs = {v: args.report / "evidence" / "runs" / ("mvp_{}_{}_legacy_s{}_l0".format(ds, v, seed)) for v in "AC"}
            roots = {v: runs[v] / ("eval_val_" + args.setting) for v in "AC"}
            for v in "AC":
                for filename in ["labels.npz", "candidates.json", "image_ids.json"]:
                    path = roots[v] / filename
                    source_hashes.append(dict(dataset=ds, source=str(path), sha256=digest(path)))
            ids = {v: read_json(roots[v] / "image_ids.json") for v in "AC"}
            candidates = {v: read_json(roots[v] / "candidates.json") for v in "AC"}
            manifest = {v: read_json(runs[v] / "meta" / "cache_manifest.json") for v in "AC"}
            if ids["A"] != ids["C"] or candidates["A"] != candidates["C"]:
                raise ValueError("image or candidate order differs; cannot pair by row")
            if manifest["A"]["attrs"] != manifest["C"]["attrs"] or manifest["A"]["objs"] != manifest["C"]["objs"]:
                raise ValueError("primitive label order differs")
            data = {}
            for v in "AC":
                with np.load(str(roots[v] / "labels.npz"), allow_pickle=False) as f:
                    data[v] = {k: f[k].copy() for k in ["attr", "obj", "pair", "top5_indices_at_bias0"]}
            for k in ["attr", "obj", "pair"]:
                if not np.array_equal(data["A"][k], data["C"][k]):
                    raise ValueError("ground-truth labels differ: " + k)
            n = len(ids["A"])
            if not n or any(len(data[v][k]) != n for v in "AC" for k in data[v]):
                raise ValueError("prediction/label/image lengths differ")
            attrs, objs = manifest["A"]["attrs"], manifest["A"]["objs"]
            for v in "AC":
                top = data[v]["top5_indices_at_bias0"]
                if top.ndim != 2 or top.shape[1] < 1 or not np.issubdtype(top.dtype, np.integer) or np.any(top < 0) or np.any(top >= len(candidates[v])):
                    raise ValueError("invalid top-5 candidate indices")
                if np.any(data[v]["attr"] < 0) or np.any(data[v]["attr"] >= len(attrs)) or np.any(data[v]["obj"] < 0) or np.any(data[v]["obj"] >= len(objs)):
                    raise ValueError("invalid primitive label indices")
            truth = [(attrs[int(a)], objs[int(o)]) for a, o in zip(data["A"]["attr"], data["A"]["obj"])]
            pred = {v: [tuple(candidates[v][int(i)]) for i in data[v]["top5_indices_at_bias0"][:, 0]] for v in "AC"}
            trainfile = args.data_root / folder / "compositional-split-natural" / "train_pairs.txt"
            trainhash = digest(trainfile)
            for v in "AC":
                declarations = [line.split(None, 1)[0] for line in (runs[v] / "meta" / "dataset_hashes.txt").read_text().splitlines()
                                if line.replace("\\", "/").rstrip().endswith("/train_pairs.txt")]
                if declarations != [trainhash]:
                    raise ValueError("current train pair file differs from run dataset hash declaration")
            seen = set()
            for line in trainfile.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    parts = line.split()
                    if len(parts) != 2:
                        raise ValueError("invalid train pair line")
                    seen.add(tuple(parts))
            source_hashes.append(dict(dataset=ds, source=str(trainfile), sha256=trainhash))
            cases = []
            for i, t in enumerate(truth):
                a, c = pred["A"][i], pred["C"][i]
                ac, cc = a == t, c == t
                category = "both_correct" if ac and cc else "A_wrong_C_correct" if not ac and cc else "A_correct_C_wrong" if ac and not cc else "both_wrong"
                cases.append(dict(dataset=ds, seed=seed, phase="val", setting=args.setting, prediction_bias=0, image_id=ids["A"][i],
                                  group="seen" if t in seen else "val_unseen", gt_attr=t[0], gt_obj=t[1], A_attr=a[0], A_obj=a[1],
                                  C_attr=c[0], C_obj=c[1], A_pair_correct=ac, C_pair_correct=cc, A_attr_correct=a[0] == t[0],
                                  C_attr_correct=c[0] == t[0], A_obj_correct=a[1] == t[1], C_obj_correct=c[1] == t[1], change=category))
            save_csv(args.out / (ds + "_image_cases.csv"), cases)
            for group in ["all", "seen", "val_unseen"]:
                subset = cases if group == "all" else [r for r in cases if r["group"] == group]
                if not subset:
                    continue
                row = dict(dataset=ds, seed=seed, phase="val", setting=args.setting, prediction_bias=0, group=group, n=len(subset))
                for v in "AC":
                    for metric in ["pair", "attr", "obj"]:
                        row[v + "_" + metric + "_accuracy_pct"] = 100 * sum(r[v + "_" + metric + "_correct"] for r in subset) / len(subset)
                for category in ["both_correct", "A_wrong_C_correct", "A_correct_C_wrong", "both_wrong"]:
                    row[category] = sum(r["change"] == category for r in subset)
                row["delta_pair_accuracy_pp"] = row["C_pair_accuracy_pct"] - row["A_pair_accuracy_pct"]
                row["net_corrected_images"] = row["A_wrong_C_correct"] - row["A_correct_C_wrong"]
                summaries.append(row)
            for attr in sorted({r["gt_attr"] for r in cases}):
                subset = [r for r in cases if r["gt_attr"] == attr]
                aa = 100 * sum(r["A_pair_correct"] for r in subset) / len(subset)
                cc = 100 * sum(r["C_pair_correct"] for r in subset) / len(subset)
                per_attr.append(dict(dataset=ds, seed=seed, setting=args.setting, prediction_bias=0, attribute=attr, n=len(subset),
                                     A_pair_accuracy_pct=aa, C_pair_accuracy_pct=cc, delta_pair_accuracy_pp=cc-aa))
        except Exception as e:
            failures.append(dict(dataset=ds, reason=type(e).__name__ + ": " + str(e)))
    save_csv(args.out / "bias0_summary.csv", summaries)
    save_csv(args.out / "per_attribute_bias0.csv", per_attr)
    save_csv(args.out / "issues.csv", failures)
    (args.out / "sources.json").write_text(json.dumps(source_hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text("# Bias=0 逐图诊断\n\n只比较原始分数 bias=0 的 top-1。AUC、best-HM、best-seen/best-unseen 对应的曲线或 operating point 无法由这些 top-5 重建。\n\n类别表同时记录样本量，低支持度类别不据此宣称稳定增益。各数据集失败原因见 issues.csv。\n", encoding="utf-8")
    print("BIAS0 OUTPUT:", args.out)
    print("DATASETS ANALYZED:", len({r["dataset"] for r in summaries}), "ISSUES:", len(failures))
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
