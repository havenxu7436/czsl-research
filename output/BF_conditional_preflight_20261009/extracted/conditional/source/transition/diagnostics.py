"""Paired bias=0 analysis, distinct from evaluator's bias-swept AUC/HM."""
import csv
from pathlib import Path

import numpy as np

from .common import empty_output, read_json, sha, write_json
from .summary import write_csv


def bias0(args):
    reference, treatment = Path(args.reference), Path(args.treatment)
    rm, tm = read_json(reference / "metrics.json"), read_json(treatment / "metrics.json")
    for key in ("dataset", "phase", "setting", "n_images", "n_candidates"):
        if rm[key] != tm[key]:
            raise ValueError("Paired prediction identity differs: " + key)
    if rm["phase"] != "val":
        raise ValueError("Mechanism screening uses validation predictions")
    images = read_json(reference / "image_ids.json")
    pairs = read_json(reference / "candidates.json")
    if images != read_json(treatment / "image_ids.json") or pairs != read_json(treatment / "candidates.json"):
        raise ValueError("Images/candidates are not identically ordered")
    r, t = np.load(reference / "labels.npz"), np.load(treatment / "labels.npz")
    for key in ("attr", "obj", "pair"):
        if not np.array_equal(r[key], t[key]):
            raise ValueError("Paired truth differs: " + key)
    def top1(z):
        if "top1_at_bias0" in z:
            return z["top1_at_bias0"]
        return z["top5_indices_at_bias0"][:, 0]
    rp, tp, truth = top1(r), top1(t), r["pair"]
    seen_pairs = set()
    for line in Path(args.train_pairs).read_text().splitlines():
        if line.strip():
            a, o = line.split()
            seen_pairs.add((a, o))
    records = []
    for i, image in enumerate(images):
        a, o = pairs[int(truth[i])]
        records.append({"image_id": image, "attribute": a, "object": o,
                        "seen": (a, o) in seen_pairs, "reference_correct": bool(rp[i] == truth[i]),
                        "treatment_correct": bool(tp[i] == truth[i]),
                        "reference_pair": " | ".join(pairs[int(rp[i])]),
                        "treatment_pair": " | ".join(pairs[int(tp[i])])})
    def aggregate(name, items):
        count = len(items)
        corrected = sum(not x["reference_correct"] and x["treatment_correct"] for x in items)
        damaged = sum(x["reference_correct"] and not x["treatment_correct"] for x in items)
        return {"group": name, "count": count, "wrong_to_correct": corrected,
                "correct_to_wrong": damaged, "net_corrected": corrected - damaged,
                "reference_accuracy_pct": 100 * sum(x["reference_correct"] for x in items) / count if count else "",
                "treatment_accuracy_pct": 100 * sum(x["treatment_correct"] for x in items) / count if count else "",
                "delta_accuracy_pp": 100 * (corrected - damaged) / count if count else ""}
    groups = [aggregate("all", records),
              aggregate("seen", [r for r in records if r["seen"]]),
              aggregate("val_unseen", [r for r in records if not r["seen"]])]
    attrs = [aggregate(a, [r for r in records if r["attribute"] == a])
             for a in sorted({r["attribute"] for r in records})]
    out = empty_output(args.out)
    write_csv(out / "paired_images.csv", records)
    write_csv(out / "group_summary.csv", groups)
    write_csv(out / "by_attribute.csv", attrs)
    write_json(out / "identity.json", {"reference": str(reference), "treatment": str(treatment),
                "reference_checkpoint_sha256": rm["checkpoint_sha256"],
                "treatment_checkpoint_sha256": tm["checkpoint_sha256"],
                "train_pairs_sha256": sha(args.train_pairs),
                "meaning": "Raw score top1 at bias=0; not best-HM or best-unseen"})
    print("BIAS0 PAIRED ANALYSIS", groups, flush=True)
