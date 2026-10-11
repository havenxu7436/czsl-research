"""Read-only UT semantic attribution audit. Python 3.8+, CPU only.

Existing runs, gate decisions, configs and knowledge are never written.
Only --out (which must not exist) is created. No training or test evaluation.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from yacs.config import CfgNode

MODES = {"A_raw": "raw", "A_unit": "unit", "C_text": "correct", "C_shuffle": "shuffle"}
ALIASES = {"A_raw": "A", "C_text": "C"}
EXPECTED_DIFFERENCES = {
    "MODEL.semantic_text", "MODEL.kcc_text", "MODEL.description_bank",
    "TRAIN.seed", "TRAIN.kcc_run_dir", "TRAIN.checkpoint_dir", "TRAIN.log_dir",
    "TRAIN.train_name", "TRAIN.kcc_stage", "TRAIN.kcc_variant",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def flatten(value, prefix=""):
    result = {}
    for k, v in value.items():
        key = prefix + k
        if isinstance(v, dict):
            result.update(flatten(v, key + "."))
        else:
            result[key] = v
    return result


def printed_config(value):
    """Match native print(cfg), which differs from cfg.dump() for 1e-05/None.

    Use the same installed YACS printer, rather than ignoring mismatched fields
    or broadly coercing arbitrary strings. Printed logs are inherently less typed
    than the authoritative effective_config.yml.
    """
    return yaml.safe_load(str(CfgNode(value)))


def raw_differences(a, b, intersect=False):
    fa, fb = flatten(a), flatten(b)
    keys = set(fa) & set(fb) if intersect else set(fa) | set(fb)
    return {k: {"reference": fa.get(k), "observed": fb.get(k),
                "reference_type": type(fa.get(k)).__name__, "observed_type": type(fb.get(k)).__name__}
            for k in sorted(keys) if k not in fa or k not in fb or fa[k] != fb[k]}


def config_differences(a, b, intersect=False):
    diffs = raw_differences(printed_config(a), printed_config(b), intersect)
    return {k: v for k, v in diffs.items() if k not in EXPECTED_DIFFERENCES}


def verify_logged_config(logged, recorded):
    differences = raw_differences(logged, printed_config(recorded))
    need(not differences, "Actual logged/effective config mismatch: " + json.dumps(differences, sort_keys=True))
    return raw_differences(recorded, logged)


def table_from_rows(rows, seeds):
    selected = [r for r in rows if r["dataset"] == "ut-zappos" and r["phase"] == "val"
                and r["regime"] == "native_semantic" and int(r["epochs"]) == 10
                and int(r["seed"]) in seeds and r["variant"] in MODES]
    values = {}
    for r in selected:
        k = (int(r["seed"]), r["variant"], r["setting"])
        need(k not in values, "Duplicate report identity: " + str(k))
        for metric in ("auc_pct", "hm_pct", "best_unseen_pct"):
            need(math.isfinite(float(r[metric])), "Non-finite report metric: " + str(k))
        values[k] = r
    expected = {(s, v, setting) for s in seeds for v in MODES for setting in ("cw", "ow")}
    need(set(values) == expected, "Missing/unexpected UT report identities: " + str(expected ^ set(values)))
    return values


def recompute_gate(values, seeds):
    def delta(t, r, setting="cw", metric="auc_pct"):
        return [float(values[s, t, setting][metric]) - float(values[s, r, setting][metric])
                for s in seeds]
    text, shuffle = delta("C_text", "A_unit"), delta("C_text", "C_shuffle")
    conditions = {
        "at_least_three_seeds": len(seeds) >= 3,
        "mean_text_over_unit_at_least_0.5pp": np.mean(text) >= 0.5,
        "mean_text_over_shuffle_at_least_0.5pp": np.mean(shuffle) >= 0.5,
        "text_positive_on_two_thirds_seeds": sum(x > 0 for x in text) >= math.ceil(2 * len(seeds) / 3),
        "shuffle_positive_on_two_thirds_seeds": sum(x > 0 for x in shuffle) >= math.ceil(2 * len(seeds) / 3),
        "C_not_below_Araw_mean": np.mean(delta("C_text", "A_raw")) >= 0,
        "cw_hm_guard": np.mean(delta("C_text", "A_unit", metric="hm_pct")) >= -0.5,
        "ow_auc_guard": np.mean(delta("C_text", "A_unit", "ow")) >= -0.5,
        "ow_unseen_guard": np.mean(delta("C_text", "A_unit", "ow", "best_unseen_pct")) >= -2,
    }
    conditions = {k: bool(v) for k, v in conditions.items()}
    return {"proceed": all(conditions.values()), "conditions": conditions,
            "C_minus_Aunit_cw_auc_pp": text, "C_minus_shuffle_cw_auc_pp": shuffle,
            "mean_C_minus_Aunit_pp": float(np.mean(text)),
            "mean_C_minus_shuffle_pp": float(np.mean(shuffle)),
            "by_seed": [{"seed": s, "C_minus_Aunit_pp": t, "C_minus_shuffle_pp": q}
                        for s, t, q in zip(seeds, text, shuffle)]}


def parse_curve(text):
    pattern = r"Val/Test Epoch (\d+): AUC=([0-9.eE+-]+) \| best_hm=([0-9.eE+-]+)"
    rows = [{"epoch": int(e), "auc_pct_rounded": 100 * float(a),
             "hm_pct_rounded": 100 * float(h)} for e, a, h in re.findall(pattern, text)]
    # Native training prints ten validation records. Do not confuse batches with epochs.
    need([r["epoch"] for r in rows] == list(range(1, 11)),
         "Training log does not contain exactly validation epochs 1..10")
    best = re.findall(r"Loading validation-best checkpoint from epoch (\d+)", text)
    need(bool(best), "No final validation-best checkpoint loading record")
    need("Done:" in text, "No native training completion record")
    epoch = int(best[-1])
    need(1 <= epoch <= 10, "Best epoch outside training budget")
    return {"best_epoch": epoch, "best_at_budget_boundary": epoch == 10, "epochs": rows}


def check_tensor_table(table, raw, old, seed):
    units, mask, perm = table["units"], table["mask"], table["permutation"]
    n = len(raw)
    need(units.ndim == 3 and units.shape[0] == n and units.shape[-1] == 300,
         "Description tensor shape mismatch")
    need(mask.dtype == torch.bool and tuple(mask.shape) == tuple(units.shape[:2]), "Invalid mask")
    need(torch.isfinite(units).all().item() and mask[:, 0].all().item(), "Invalid name units")
    need(sorted(perm.tolist()) == list(range(n)) and not (perm == torch.arange(n)).any().item(),
         "Permutation is not a derangement")
    order = np.random.RandomState(seed).permutation(n)
    expected = np.empty(n, dtype=np.int64)
    expected[order] = np.roll(order, 1)
    need(perm.tolist() == expected.tolist(), "Permutation differs from the frozen permutation seed")
    checks = {"unit": (table["unit"], F.normalize(raw, dim=-1)),
              "name_slot": (units[:, 0], F.normalize(raw, dim=-1)),
              "correct_original_cache": (table["correct"], old)}
    counts = mask.sum(1)
    expected_mask = torch.arange(units.shape[1])[None, :] < counts[:, None]
    need(torch.equal(mask, expected_mask), "Description slots are not contiguous")
    need((units[~mask].abs() < 1e-7).all().item(), "Nonzero padded description units")
    need(torch.allclose(units[mask].norm(dim=-1), torch.ones_like(units[mask].norm(dim=-1)),
                        atol=1e-5), "Valid units are not L2 normalized")
    mean = (units * mask[..., None]).sum(1) / counts[:, None]
    checks.update(mean=(table["mean"], mean), correct_mean=(table["correct"], F.normalize(mean, dim=-1)),
                  residual=(table["residual"], (units - mean[:, None]) * mask[..., None]))
    shuffled = torch.zeros_like(units)
    smask = torch.zeros_like(mask)
    for i, donor in enumerate(perm.tolist()):
        k = int(counts[donor])
        shuffled[i, 0] = units[i, 0]
        shuffled[i, 1:k] = units[donor, 1:k]
        smask[i, :k] = True
    need(torch.equal(table["shuffle_mask"], smask), "Shuffle mask mismatch")
    smean = shuffled.sum(1) / smask.sum(1)[:, None]
    checks.update(shuffle_units=(table["shuffle_units"], shuffled),
                  shuffle_mean=(table["shuffle_mean"], smean),
                  shuffle=(table["shuffle"], F.normalize(smean, dim=-1)))
    errors = {}
    for name, (observed, reference) in checks.items():
        need(tuple(observed.shape) == tuple(reference.shape), "Shape mismatch: " + name)
        error = float((observed - reference).abs().max())
        need(math.isfinite(error) and error <= 1e-6, "Tensor mismatch: {} max_abs={}".format(name, error))
        errors[name] = error
    cosine = (table["correct"] * table["shuffle"]).sum(-1)
    return {"max_abs_errors": errors, "description_counts": (counts - 1).tolist(),
            "correct_shuffle_cosine_mean": float(cosine.mean()),
            "correct_shuffle_cosine_min": float(cosine.min()),
            "correct_shuffle_cosine_max": float(cosine.max()),
            "residual_norm_mean": float(table["residual"].norm(dim=-1)[mask].mean()),
            "permutation": perm.tolist()}


def saved_top1(value):
    """Support the two audited evaluator schemas without renaming old artifacts."""
    if "top1_at_bias0" in value:
        top, field = value["top1_at_bias0"], "top1_at_bias0"
    elif "top5_indices_at_bias0" in value:
        top5 = value["top5_indices_at_bias0"]
        need(top5.ndim == 2 and top5.shape[1] >= 1 and np.issubdtype(top5.dtype, np.integer),
             "Invalid saved top5 array")
        top, field = top5[:, 0], "top5_indices_at_bias0[:,0]"
    else:
        raise ValueError("No supported bias-zero prediction field; available=" + str(sorted(value)))
    need(top.ndim == 1 and np.issubdtype(top.dtype, np.integer), "Invalid saved top1 array")
    return top, field


def paired_top1(left, right, labels, train_pairs):
    a_ids, b_ids = read(left / "image_ids.json"), read(right / "image_ids.json")
    ac, bc = read(left / "candidates.json"), read(right / "candidates.json")
    need(a_ids == b_ids, "Validation image order mismatch; refuse unaligned pairing")
    need(ac == bc, "Candidate order mismatch; refuse unaligned pairing")
    need(len(a_ids) == len(set(a_ids)), "Duplicate validation image identifiers")
    with np.load(str(left / "labels.npz"), allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    with np.load(str(right / "labels.npz"), allow_pickle=False) as z:
        b = {k: z[k].copy() for k in z.files}
    for key in ("attr", "obj", "pair"):
        need(np.array_equal(a[key], b[key]), "Ground-truth order mismatch: " + key)
        need(a[key].shape == (len(a_ids),), "Invalid ground-truth vector shape")
    truth = [(labels["attrs"][int(x)], labels["objs"][int(y)]) for x, y in zip(a["attr"], a["obj"])]
    predictions, prediction_fields = [], []
    for v in (a, b):
        top, field = saved_top1(v)
        need(top.shape == (len(truth),) and np.issubdtype(top.dtype, np.integer), "Invalid top1 indices")
        need(((top >= 0) & (top < len(ac))).all(), "Top1 candidate index outside range")
        predictions.append([tuple(ac[int(i)]) for i in top])
        prediction_fields.append(field)
    correct = [np.array([p == t for p, t in zip(ps, truth)]) for ps in predictions]
    seen = np.array([p in train_pairs for p in truth])
    def stats(indices):
        x, y = correct[0][indices], correct[1][indices]
        return {"n_images": len(x), "C_only_correct": int((x & ~y).sum()),
                "reference_only_correct": int((~x & y).sum()),
                "both_correct": int((x & y).sum()), "both_wrong": int((~x & ~y).sum()),
                "delta_bias0_top1_pp": float(100 * (x.mean() - y.mean())) if len(x) else None}
    groups = {"all": stats(np.ones(len(truth), dtype=bool)), "seen": stats(seen), "unseen": stats(~seen)}
    groups["by_attribute"] = {name: stats(a["attr"] == i) for i, name in enumerate(labels["attrs"])
                              if (a["attr"] == i).any()}
    groups["by_object"] = {name: stats(a["obj"] == i) for i, name in enumerate(labels["objs"])
                           if (a["obj"] == i).any()}
    groups["saved_prediction_fields"] = dict(C_text=prediction_fields[0], reference=prediction_fields[1])
    return groups


class Audit:
    def __init__(self):
        self.checks, self.digests = [], {}

    def digest(self, path):
        key = str(Path(path).resolve())
        if key not in self.digests:
            self.digests[key] = sha(path)
        return self.digests[key]

    def section(self, name, fn):
        try:
            value = fn()
            self.checks.append({"check": name, "status": "PASS"})
            print("PASS", name, flush=True)
            return value
        except Exception as e:
            self.checks.append({"check": name, "status": "FAIL", "error": type(e).__name__ + ": " + str(e)})
            print("FAIL", name, str(e), flush=True)
            return None


def run(args):
    out, root, exp, legacy, project, report = map(Path, (args.out, args.root, args.exp,
                                                      args.legacy_exp, args.project, args.report))
    need(not out.exists(), "Output already exists; use a new audit timestamp")
    # Writing inside the code or any original result/knowledge would break preservation.
    protected = [project, report, exp / "runs", legacy, root / "knowledge", exp / "protocol"]
    need(not any(out.resolve() == p.resolve() or p.resolve() in out.resolve().parents for p in protected),
         "Audit output must be separate from protected inputs")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    audit = Audit()
    result = {"version": "semantic_gate_audit_v2", "seeds": args.seeds,
              "inputs": {"root": str(root), "report": str(report), "gate": args.gate},
              "checks": audit.checks, "run_audits": [], "paired_bias0": []}
    result["audit_tool_sha256"] = audit.digest(Path(__file__))
    bank_dir = root / "knowledge/ekr0601_transition_v1/description_bank_v1/ut-zap50k"
    cache_dir = root / "knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k"
    knowledge = root / "knowledge/kcc0601_v1/trusted_v1/ut-zap50k"

    def collection_check():
        c = read(report / "collection.json")
        fingerprint = hashlib.sha256(json.dumps(c["inputs"], sort_keys=True).encode()).hexdigest()
        need(fingerprint == c["fingerprint"], "Report collection fingerprint mismatch")
        for path, expected in c["inputs"].items():
            need(audit.digest(path) == expected, "Report source file changed: " + path)
        with (report / "runs.csv").open(encoding="utf-8-sig", newline="") as f:
            values = table_from_rows(list(csv.DictReader(f)), args.seeds)
        gate = recompute_gate(values, args.seeds)
        if args.gate:
            original = read(args.gate)
            need(original["report_fingerprint"] == fingerprint, "Gate is from another report")
            need(original["planned_seeds"] == args.seeds, "Gate seed order differs; pass its planned_seeds order")
            ds = original["datasets"]["ut-zappos"]
            for key in ("conditions", "proceed"):
                need(ds[key] == gate[key], "Saved gate differs from recomputation: " + key)
            for key in ("C_minus_Aunit_cw_auc_pp", "C_minus_shuffle_cw_auc_pp"):
                need(np.allclose(ds[key], gate[key], atol=1e-8, rtol=0), "Saved gate deltas differ: " + key)
            result["original_gate_sha256"] = audit.digest(args.gate)
        result["gate_recomputed"] = gate
        print("GATE_PROCEED", gate["proceed"], "(research gate, not an integrity failure)", flush=True)
        for row in gate["by_seed"]:
            print("SEED", row["seed"], "C_MINUS_UNIT_pp", round(row["C_minus_Aunit_pp"], 6),
                  "C_MINUS_SHUFFLE_pp", round(row["C_minus_shuffle_pp"], 6), flush=True)
        return values

    values = audit.section("report_sources_and_gate", collection_check)

    def bank_check():
        manifest, cm = read(bank_dir / "manifest.json"), read(cache_dir / "manifest.json")
        need(audit.digest(cache_dir / "manifest.json") == manifest["original_cache_manifest_sha256"],
             "Bank uses a different original cache manifest")
        for name, digest in manifest["files_sha256"].items():
            need(audit.digest(bank_dir / name) == digest, "Bank file checksum mismatch: " + name)
        for name, digest in manifest["knowledge_inputs"].items():
            need(audit.digest(knowledge / name) == digest, "Frozen knowledge source changed: " + name)
        labels = read(bank_dir / "labels.json")
        need(labels == read(cache_dir / "labels.json"), "Original cache label order differs")
        # Only load our own hash-verified cache/bank tensor files. No model checkpoint unpickling.
        bank = torch.load(str(bank_dir / "bank.pt"), map_location="cpu")
        tables, review = {}, ["# Frozen descriptions and shuffle donors", "",
                             "Review semantics against label meaning and training examples.",
                             "Do not use validation errors as description rewrite targets.", ""]
        for offset, (kind, axis) in enumerate((("a", "attrs"), ("o", "objs"))):
            filename = "ae_dict.json" if kind == "a" else "oe_dict.json"
            texts = read(knowledge / filename)
            need(list(texts) == labels[axis], "Knowledge label order mismatch")
            tensors = []
            for suffix in ("name_raw", "K"):
                name = "W_{}_{}.pt".format(kind, suffix)
                need(audit.digest(cache_dir / name) == cm["files_sha256"][name], "Cache tensor changed: " + name)
                tensors.append(torch.load(str(cache_dir / name), map_location="cpu"))
            tables[kind] = check_tensor_table(bank[kind], *tensors, manifest["permutation_seed"] + offset)
            need(tables[kind]["description_counts"] == [len(texts[x]) for x in labels[axis]],
                 "Description count differs from original text source")
            records = read(bank_dir / "descriptions.json")[kind]
            need([(r["label"], r["text"]) for r in records] ==
                 [(label, text) for label in labels[axis] for text in texts[label]], "Bank text records differ")
            tables[kind]["fallback_sentences"] = sum(bool(r["fallback"]) for r in records)
            tables[kind]["total_sentences"] = len(records)
            tables[kind]["oov_tokens"] = sum(len(r["oov"]) for r in records)
            tables[kind]["total_tokens"] = sum(len(r["tokens"]) for r in records)
            tables[kind]["duplicate_sentences"] = sorted({r["text"] for r in records
                if sum(q["text"] == r["text"] for q in records) > 1})
            for i, name in enumerate(labels[axis]):
                donor = labels[axis][tables[kind]["permutation"][i]]
                review += ["## {}: {} (shuffle donor: {})".format(kind, name, donor), ""]
                review += ["- " + t.replace("\n", " ") for t in texts[name]]
                review += ["", "Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate", ""]
        (out / "description_review.md").write_text("\n".join(review), encoding="utf-8")
        result["bank_audit"] = tables
        result["knowledge_status"] = manifest.get("knowledge_status")
        return labels

    labels = audit.section("frozen_bank_and_shuffle_integrity", bank_check)
    configs, runs = {}, {}
    baseline_run = legacy / "runs/mvp_ut-zappos_C_legacy_s3351_l0"

    def baseline_config():
        text = (baseline_run / "train.log").read_text(encoding="utf-8", errors="replace")
        need("Random seed:" in text, "Cannot recover baseline config")
        return yaml.safe_load(text.split("Random seed:", 1)[0])

    baseline = audit.section("recover_original_protocol", baseline_config)
    if values is not None:
        for seed in args.seeds:
            for variant in MODES:
                row = values[seed, variant, "cw"]
                is_old = str(row.get("reused_legacy", "False")).lower() == "true"
                run_dir = (legacy if is_old else exp) / "runs" / row["run_id"]
                runs[seed, variant] = run_dir

                def run_check(run_dir=run_dir, seed=seed, variant=variant, is_old=is_old):
                    text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
                    log_cfg = yaml.safe_load(text.split("Random seed:", 1)[0])
                    cfg = log_cfg if is_old else yaml.safe_load((run_dir / "effective_config.yml").read_text())
                    need(isinstance(cfg, dict), "Invalid training config")
                    print_differences = verify_logged_config(log_cfg, cfg)
                    if print_differences:
                        print("CONFIG_PRINT_COMPAT", seed, variant, ",".join(print_differences), flush=True)
                    tr, mo, ev = cfg["TRAIN"], cfg["MODEL"], cfg["EVAL"]
                    need(int(tr["seed"]) == seed and int(tr["max_epoch"]) == 10 and int(tr["start_epoch"]) == 1,
                         "Seed/budget/start epoch mismatch")
                    need(cfg["DATASET"]["name"] == "ut-zappos" and not ev["open_world"] and int(ev["topk"]) == 1,
                         "Training/evaluation protocol mismatch")
                    need(not mo["kcc_calib"] and float(mo["kcc_lambda"]) == 0 and mo["wordembs"] == "glove",
                         "K calibration or embedding protocol differs")
                    need(not mo["kcc_feature"] and not mo["kcc_logic"] and mo["kcc_control"] == "none",
                         "Unexpected active K feature/logic/control")
                    need(not tr["kcc_test_at_end"] and int(tr["aug_epoch"]) > 10, "Unexpected test/augmentation training protocol")
                    need(bool(mo["kcc_text"]) == (variant != "A_raw"), "kcc_text flag mismatch")
                    need(tr["kcc_variant"] == (ALIASES[variant] if is_old else variant), "Variant mismatch")
                    need(Path(tr["kcc_run_dir"]).resolve() == run_dir.resolve(), "Training output identity mismatch")
                    need(Path(mo["kcc_cache"]).resolve() == cache_dir.resolve(), "Different original knowledge cache")
                    if not is_old:
                        need(mo["semantic_text"] == MODES[variant], "Semantic mode mismatch")
                        need(Path(mo["description_bank"]).resolve() == bank_dir.resolve(), "Different description bank")
                        need(read(run_dir / "status.json")["status"] == "train_and_val_complete", "Run is not complete")
                        need(read(run_dir / "meta/bank_manifest.json") == read(bank_dir / "manifest.json"), "Training bank manifest differs")
                        cm = read(run_dir / "meta/code_manifest.json")
                        for name, digest in cm.items():
                            need(audit.digest(project / name) == digest, "Code changed since training: " + name)
                    if baseline is not None:
                        differences = config_differences(baseline, cfg, intersect=True)
                        need(not differences, "Different original scientific protocol: " + json.dumps(differences))
                    curve = parse_curve(text)
                    digest = audit.digest(run_dir / "checkpoints/best_auc.pth")
                    if not is_old:
                        complete = read(run_dir / "train_complete.json")
                        need(complete["checkpoint_sha256"] == digest and complete["epochs"] == 10
                             and complete["best_epoch"] == curve["best_epoch"], "Training completion/checkpoint mismatch")
                    for setting in ("cw", "ow"):
                        m = read(run_dir / ("eval_val_" + setting) / "metrics.json")
                        need((m["dataset"], m["phase"], m["setting"], int(m["seed"])) ==
                             ("ut-zappos", "val", setting, seed), "Evaluation identity mismatch")
                        need(m["variant"] == (ALIASES[variant] if is_old else variant), "Evaluation variant mismatch")
                        need(m["checkpoint_sha256"] == digest and m["best_epoch"] == curve["best_epoch"],
                             "Evaluated checkpoint/best epoch mismatch")
                        need(m.get("score_mode", "legacy") == "legacy", "Evaluation score mode mismatch")
                        if not is_old:
                            need(m["bank_manifest_sha256"] == audit.digest(bank_dir / "manifest.json"), "Evaluation bank differs")
                        r = values[seed, variant, setting]
                        for metric, rawkey in (("auc_pct", "AUC"), ("hm_pct", "best_hm"), ("best_unseen_pct", "best_unseen")):
                            need(abs(float(r[metric]) - 100 * m["metrics_raw"][rawkey]) < 1e-6, "CSV and metrics differ")
                        if setting == "cw":
                            selected = curve["epochs"][curve["best_epoch"] - 1]
                            need(abs(selected["auc_pct_rounded"] - float(r["auc_pct"])) < 0.02,
                                 "Best epoch logged AUC differs from evaluated AUC")
                    configs[seed, variant] = cfg
                    record = dict(seed=seed, variant=variant, run=str(run_dir), reused_legacy=is_old,
                                  checkpoint_sha256=digest, curve=curve,
                                  console_serialization_differences=print_differences)
                    result["run_audits"].append(record)
                audit.section("run_s{}_{}".format(seed, variant), run_check)

    def matched_config_check():
        need(len(configs) == 4 * len(args.seeds), "Some per-run audits failed; cannot establish matched protocol")
        reference = next(cfg for (s, v), cfg in configs.items() if "semantic_text" in cfg["MODEL"])
        for (s, v), cfg in configs.items():
            old = "semantic_text" not in cfg["MODEL"]
            diff = config_differences(reference, cfg, intersect=old)
            need(not diff, "Unmatched config seed={} variant={}: {}".format(s, v, json.dumps(diff)))
    audit.section("matched_configs_across_variants_and_seeds", matched_config_check)

    if labels is not None:
        def pairing_check():
            split = root / "data/ut-zap50k/compositional-split-natural"
            train_pairs = {tuple(line.split()) for line in (split / "train_pairs.txt").read_text().splitlines() if line.strip()}
            need(all(len(p) == 2 for p in train_pairs), "Invalid train pair definitions")
            for seed in args.seeds:
                for setting in ("cw", "ow"):
                    for ref in ("A_unit", "C_shuffle"):
                        a, b = [runs[seed, v] / ("eval_val_" + setting) for v in ("C_text", ref)]
                        for directory in (a, b):
                            for filename in ("image_ids.json", "candidates.json", "labels.npz"):
                                audit.digest(directory / filename)
                        summary = paired_top1(a, b, labels, train_pairs)
                        result["paired_bias0"].append({"seed": seed, "setting": setting,
                                                       "comparison": "C_text-" + ref, **summary})
        audit.section("paired_validation_bias0_errors", pairing_check)
    else:
        audit.checks.append({"check": "paired_validation_bias0_errors", "status": "SKIP", "reason": "Bank audit failed"})

    result["integrity_passed"] = all(c["status"] == "PASS" for c in audit.checks)
    result["input_files_sha256"] = audit.digests
    result["limits"] = ["CPU artifact audit only; no GPU checkpoint replay or training",
                        "Bank tensor arithmetic verified; GloVe file not rescanned or sentence embeddings rebuilt",
                        "Bias-zero top1 error counts do not recompute AUC/HM and are not causal semantic proof",
                        "Legacy top1 uses the first saved sorted top5 index; exact score ties may differ from new argmax tie handling",
                        "Native log comparison reproduces YACS print format; printed logs cannot recover all original scalar types",
                        "Single frozen shuffle mapping; three training seeds do not replicate shuffle mappings",
                        "Native cuDNN benchmark setting permits numerical nondeterminism",
                        "No gate or original artifact is modified; integrity pass does not imply research gate pass"]
    write(out / "audit.json", result)
    text = ["# UT semantic attribution audit", "", "INTEGRITY_PASSED: " + str(result["integrity_passed"]), ""]
    g = result.get("gate_recomputed")
    if g:
        text += ["RESEARCH_GATE_PROCEED: " + str(g["proceed"]), "",
                 "| Seed | C - A_unit CW AUC (pp) | C - C_shuffle CW AUC (pp) |",
                 "| --- | ---: | ---: |"]
        text += ["| {seed} | {C_minus_Aunit_pp:+.6f} | {C_minus_shuffle_pp:+.6f} |".format(**r) for r in g["by_seed"]]
        text += ["", "Mean C - A_unit: {:+.6f} pp".format(g["mean_C_minus_Aunit_pp"]),
                 "Mean C - C_shuffle: {:+.6f} pp".format(g["mean_C_minus_shuffle_pp"]), ""]
    text += ["## Checks", ""] + ["- " + c["status"] + ": " + c["check"] +
                                         (" -- " + c["error"] if "error" in c else "") for c in audit.checks]
    text += ["", "## Best epochs", ""] + ["- s{} {}: epoch {}{}".format(r["seed"], r["variant"],
        r["curve"]["best_epoch"], " (budget boundary)" if r["curve"]["best_at_budget_boundary"] else "")
        for r in result["run_audits"]]
    text += ["", "## Limits", ""] + ["- " + t for t in result["limits"]]
    (out / "audit_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("AUDIT_OUTPUT", out, flush=True)
    print("INTEGRITY_PASSED", result["integrity_passed"], flush=True)
    return 0 if result["integrity_passed"] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/data/xhw/czsl_repro")
    p.add_argument("--exp", default="/data/xhw/czsl_repro/experiments/ekr0601_transition_v1")
    p.add_argument("--legacy-exp", default="/data/xhw/czsl_repro/experiments/kcc0601_glove_mvp_v1")
    p.add_argument("--project", default="/data/xhw/czsl_repro/src/EKR0601_transition_v1")
    p.add_argument("--report", required=True)
    p.add_argument("--gate", required=True)
    p.add_argument("--seeds", nargs="+", type=int, help="Default: planned_seeds from the saved gate")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.seeds is None:
        args.seeds = read(args.gate)["planned_seeds"]
    need(all(isinstance(s, int) for s in args.seeds), "Gate seeds must be integers")
    need(len(args.seeds) == len(set(args.seeds)), "Duplicate seeds")
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("AUDIT_FATAL", type(error).__name__, str(error), file=sys.stderr, flush=True)
        sys.exit(1)
