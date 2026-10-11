#!/usr/bin/env python3
"""Read existing CPF MVP outputs; snapshot evidence and build a stage report.

Python 3.8+, standard library. PyYAML is optional for the full A/C config audit.
Never trains, evaluates, imports torch, or modifies the experiment inputs.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

VERSION = "cpf_stage_report_20261004_v1"
SPECS = [("ut-zappos", 3351, v, "0.1" if v in "DE" else "0") for v in "ACDE"]
SPECS += [(ds, seed, v, "0") for ds, seed in [("mit-states", 0), ("cgqa", 3351)] for v in "AC"]
FIELDS = {"auc_pct": "AUC", "hm_pct": "best_hm", "best_seen_pct": "best_seen",
          "best_unseen_pct": "best_unseen", "hm_seen_pct": "hm_seen", "hm_unseen_pct": "hm_unseen",
          "attr_acc_evaluator_pct": "closed_attr_match", "obj_acc_evaluator_pct": "closed_obj_match"}
FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def text(path):
    return Path(path).read_text(encoding="utf-8-sig", errors="replace")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows, columns=None):
    if columns is None:
        columns = list(dict.fromkeys(k for row in rows for k in row))
    if not columns:
        columns = ["status"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def number(value, key):
    if isinstance(value, bool):
        raise ValueError("boolean metric: " + key)
    n = float(value)
    if not math.isfinite(n) or not 0 <= n <= 1:
        raise ValueError("metric must be a finite fraction in [0,1]: " + key)
    return n


def metrics(meta):
    if meta.get("phase") != "val" or meta.get("setting") not in ("cw", "ow"):
        raise ValueError("expected validation CW/OW metadata")
    raw = meta["metrics_raw"]
    result = {}
    for col, key in FIELDS.items():
        if key in raw:
            result[col] = 100 * number(raw[key], key)
        elif key in ("AUC", "best_hm", "best_seen", "best_unseen"):
            raise ValueError("required metric missing: " + key)
        else:
            result[col] = None
    result["best_hm_bias"] = raw.get("biasterm")
    if result["hm_seen_pct"] is not None and result["hm_unseen_pct"] is not None:
        s, u = result["hm_seen_pct"], result["hm_unseen_pct"]
        expected = 2 * s * u / (s + u) if s + u else 0
        if abs(expected - result["hm_pct"]) > 1e-4:
            raise ValueError("best_hm does not match hm_seen/hm_unseen")
    return result


def parse_log(path, run_id, dataset, seed, variant):
    if not Path(path).is_file():
        return [], None
    value = text(path)
    base = dict(run_id=run_id, dataset=dataset, seed=seed, variant=variant)
    rows = []
    # Last printed cumulative average; not necessarily an exact full-epoch mean.
    training = {}
    rx = r"Epoch: (\d+) Iter: (\d+)/(\d+), Loss: (" + FLOAT + r"), Acc_Obj: (" + FLOAT + r"), Acc_Attr: (" + FLOAT + r"), Acc_Pair: (" + FLOAT + r")"
    for m in re.finditer(rx, value):
        ep, it, total = map(int, m.group(1, 2, 3))
        training[ep] = dict(last_logged_iter=it, total_iters=total, loss_last_logged_avg=float(m.group(4)),
                            train_obj_acc_as_logged=float(m.group(5)), train_attr_acc_as_logged=float(m.group(6)),
                            train_pair_acc_as_logged=float(m.group(7)))
    phase = "unknown"
    for line in value.replace("\r", "\n").splitlines():
        if "Validation set ===>" in line:
            phase = "val"
        elif "Test set with validation-best checkpoint" in line:
            phase = "test"
        match = re.search(r"Val/Test Epoch (\d+): (.+)", line)
        if match:
            ep = int(match.group(1))
            vals = {k: float(v) for k, v in re.findall(r"([A-Za-z_]+)=(" + FLOAT + r")", match.group(2))}
            row = dict(base, epoch=ep, phase=phase, setting="train_config_setting", curve_source="rounded_train_log")
            row.update(training.get(ep, {}))
            for col, key in FIELDS.items():
                row[col] = 100 * vals[key] if key in vals else None
            rows.append(row)
    loaded = re.findall(r"Loading validation-best checkpoint from epoch (\d+)", value)
    return rows, int(loaded[-1]) if loaded else None


class Evidence:
    def __init__(self, out):
        self.out = out
        self.files = []
        self.issues = []

    def issue(self, scope, message):
        self.issues.append(dict(scope=scope, message=message))

    def copy(self, source, relative):
        source = Path(source)
        if not source.is_file():
            return None
        dest = self.out / "evidence" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        shutil.copyfile(str(source), str(dest))
        digest = sha(dest)
        after = source.stat()
        stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        self.files.append(dict(source=str(source.resolve()), snapshot=str(dest.relative_to(self.out)).replace("\\", "/"),
                               sha256=digest, bytes=dest.stat().st_size, source_stable_during_copy=stable))
        if not stable:
            self.issue(str(source), "source changed while copying; snapshot is provisional")
        return dest


def flatten(value, prefix=""):
    result = {}
    for k, v in value.items():
        key = prefix + str(k)
        if isinstance(v, dict):
            result.update(flatten(v, key + "."))
        else:
            result[key] = v
    return result


def config_audit(a, c):
    try:
        import yaml
    except ImportError:
        return "metadata_only", "PyYAML unavailable; manually compare requested configs"
    try:
        fa = flatten(yaml.safe_load(text(a / "config_requested.yml")))
        fc = flatten(yaml.safe_load(text(c / "config_requested.yml")))
        for f, variant in [(fa, "A"), (fc, "C")]:
            if f.get("MODEL.kcc_text") is not (variant == "C") or f.get("MODEL.kcc_calib") is not False:
                raise ValueError("A/C text/calibration flags mismatch")
            if float(f.get("MODEL.kcc_lambda", -1)) != 0 or f.get("TRAIN.kcc_test_at_end") is not False:
                raise ValueError("lambda/test-at-end flag mismatch")
            if int(f.get("TRAIN.max_epoch", -1)) != 10:
                raise ValueError("expected ten-epoch MVP budget")
        allowed = {"MODEL.kcc_text", "TRAIN.train_name", "TRAIN.kcc_variant", "TRAIN.kcc_run_dir",
                   "TRAIN.log_dir", "TRAIN.checkpoint_dir"}
        diff = [k for k in sorted(set(fa) | set(fc)) if fa.get(k) != fc.get(k) and k not in allowed]
        return ("mismatch", "; ".join(diff)) if diff else ("matched", "requested configs match outside intended A/C changes")
    except Exception as e:
        return "unverified", type(e).__name__ + ": " + str(e)


def duration(run):
    try:
        start = datetime.fromisoformat(text(run / "start_time.txt").strip())
        end = datetime.fromisoformat(text(run / "end_time.txt").strip())
        return (end - start).total_seconds()
    except (OSError, ValueError):
        return None


def markdown_table(rows, columns):
    def fmt(v):
        if v is None or v == "":
            return "待完成"
        if isinstance(v, float):
            return "{:.3f}".format(v)
        return str(v).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(label for _, label in columns) + " |\n"
    header += "| " + " | ".join("---" for _ in columns) + " |\n"
    return header + "".join("| " + " | ".join(fmt(row.get(key)) for key, _ in columns) + " |\n" for row in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--lambda-dir", type=Path)
    p.add_argument("--audit-dir", type=Path)
    p.add_argument("--require-complete", action="store_true", help="Write report but exit 2 if eight runs or lambda grid are not fully verified")
    args = p.parse_args()
    if args.out.exists():
        p.error("output exists; choose a new report version: " + str(args.out))
    exp, out = args.experiment.resolve(), args.out.resolve()
    out.mkdir(parents=True)
    tables = out / "tables"
    tables.mkdir()
    ev = Evidence(out)
    all_rows, inventory, curves = [], [], []
    states, usable = {}, {}
    for ds, seed, variant, lam in SPECS:
        rid = "mvp_{}_{}_legacy_s{}_l{}".format(ds, variant, seed, lam)
        run = exp / "runs" / rid
        key = (ds, seed, variant)
        status = text(run / "status.txt").strip() if (run / "status.txt").is_file() else "missing"
        copied = {}
        for rel in ["status.txt", "start_time.txt", "end_time.txt", "train.log", "config_requested.yml", "command_train.txt",
                    "meta/checkpoint.sha256", "meta/source.sha256", "meta/source_snapshot.tar.gz", "meta/cache_manifest.json",
                    "meta/dataset_hashes.txt", "meta/pip_freeze.txt", "meta/nvidia-smi.txt"]:
            copied[rel] = ev.copy(run / rel, Path("runs") / rid / rel)
        for source in sorted((run / "logs").rglob("events.out.tfevents.*")) if (run / "logs").is_dir() else []:
            ev.copy(source, Path("runs") / rid / source.relative_to(run))
        logrows, best_epoch_log = parse_log(copied.get("train.log") or run / "train.log", rid, ds, seed, variant)
        curves.extend(logrows)
        val_epochs = sorted({r["epoch"] for r in logrows if r["phase"] == "val"})
        if status == "train_and_val_complete" and val_epochs != list(range(1, 11)):
            ev.issue(rid, "train log does not confirm validation epochs 1..10: " + str(val_epochs))
        if any(r["phase"] == "test" for r in logrows):
            ev.issue(rid, "unexpected test evaluation in validation-only MVP log")
        valid_settings = []
        row_errors = []
        for setting in ("cw", "ow"):
            for name in ["metrics.json", "config_eval_effective.yml", "candidates.json", "image_ids.json", "labels.npz"]:
                ev.copy(run / ("eval_val_" + setting) / name, Path("runs") / rid / ("eval_val_" + setting) / name)
            path = out / "evidence" / "runs" / rid / ("eval_val_" + setting) / "metrics.json"
            row = dict(run_id=rid, dataset=ds, seed=seed, variant=variant, lambda_k=float(lam), phase="val", setting=setting,
                       status=status, row_status="missing_metrics", metrics_source=str(run / ("eval_val_" + setting) / "metrics.json"))
            if path.is_file():
                try:
                    meta = read_json(path)
                    if meta.get("setting") != setting or meta.get("seed") != seed or meta.get("variant") != variant:
                        raise ValueError("run identity/setting/seed/variant mismatch")
                    if meta.get("run_id") != rid or meta.get("dataset") != ds or meta.get("score_mode") != "legacy":
                        raise ValueError("run_id/dataset/score mode mismatch")
                    if float(meta.get("lambda_k", -1)) != float(lam) or meta.get("topk") != 1:
                        raise ValueError("lambda/topk mismatch")
                    row.update(metrics(meta))
                    row.update({k: meta.get(k) for k in ["best_epoch", "checkpoint_sha256", "config_sha256", "source_manifest_sha256",
                                "knowledge_manifest_sha256", "dataset_manifest_sha256", "n_images", "n_candidates", "eval_seconds",
                                "peak_eval_gpu_allocated_mb", "score_mode", "scores_saved"]})
                    if row["best_epoch"] is None:
                        row["best_epoch"] = best_epoch_log
                    provenance = True
                    for field, rel in [("source_manifest_sha256", "meta/source.sha256"), ("knowledge_manifest_sha256", "meta/cache_manifest.json"),
                                       ("dataset_manifest_sha256", "meta/dataset_hashes.txt")]:
                        snap = copied.get(rel)
                        if not snap or not meta.get(field):
                            provenance = False
                            ev.issue(rid, "missing provenance: " + field)
                        elif sha(snap) != meta[field]:
                            raise ValueError("manifest digest mismatch: " + field)
                    ckfile = copied.get("meta/checkpoint.sha256")
                    if not ckfile or not meta.get("checkpoint_sha256"):
                        provenance = False
                    elif text(ckfile).split()[0] != meta["checkpoint_sha256"]:
                        raise ValueError("checkpoint hash declaration mismatch")
                    cfgpath = path.parent / "config_eval_effective.yml"
                    if not cfgpath.is_file() or not meta.get("config_sha256"):
                        provenance = False
                    elif sha(cfgpath) != meta["config_sha256"]:
                        raise ValueError("effective evaluation config digest mismatch")
                    row["row_status"] = "verified" if status == "train_and_val_complete" and provenance else "provisional"
                    valid_settings.append(setting)
                    if row["row_status"] == "verified":
                        usable[(ds, seed, variant, setting)] = row
                except Exception as e:
                    row["row_status"] = "invalid"
                    row_errors.append(setting + ": " + str(e))
                    ev.issue(rid, row_errors[-1])
            all_rows.append(row)
        states[key] = run
        checkpoint_match = all(usable.get((ds, seed, variant, s)) for s in ("cw", "ow"))
        if checkpoint_match:
            a, b = [usable[(ds, seed, variant, s)] for s in ("cw", "ow")]
            checkpoint_match = a["checkpoint_sha256"] == b["checkpoint_sha256"] and a["best_epoch"] == b["best_epoch"]
            if not checkpoint_match:
                ev.issue(rid, "CW/OW use different checkpoint or best_epoch")
                for s in ("cw", "ow"):
                    usable.pop((ds, seed, variant, s), None)
        inventory.append(dict(run_id=rid, dataset=ds, variant=variant, seed=seed, lambda_k=float(lam), status=status,
                              cw_ow_verified=bool(checkpoint_match), best_epoch=best_epoch_log, observed_val_epochs=len(val_epochs),
                              observed_last_val_epoch=max(val_epochs) if val_epochs else None, train_plus_val_seconds=duration(run),
                              errors="; ".join(row_errors), path=str(run)))
    # Do not use snapshots from files changing during collection in comparisons.
    for fileinfo in ev.files:
        if not fileinfo["source_stable_during_copy"]:
            parts = Path(fileinfo["snapshot"]).parts
            if len(parts) > 2 and parts[1] == "runs":
                for key, row in list(usable.items()):
                    if row["run_id"] == parts[2]:
                        usable.pop(key, None)
    ac, audits = [], []
    for ds, seed in [("ut-zappos", 3351), ("mit-states", 0), ("cgqa", 3351)]:
        ar, cr = states[(ds, seed, "A")], states[(ds, seed, "C")]
        cfgstate, detail = config_audit(out / "evidence" / "runs" / ar.name, out / "evidence" / "runs" / cr.name)
        for setting in ("cw", "ow"):
            a, c = usable.get((ds, seed, "A", setting)), usable.get((ds, seed, "C", setting))
            checks = []
            if not a or not c:
                checks.append("complete verified A/C metrics unavailable")
            else:
                for field in ["source_manifest_sha256", "knowledge_manifest_sha256", "dataset_manifest_sha256", "n_images", "n_candidates", "score_mode"]:
                    if a.get(field) is None or a[field] != c.get(field):
                        checks.append(field + " differs/is missing")
                for name in ["image_ids.json", "candidates.json"]:
                    pa = out / "evidence" / "runs" / ar.name / ("eval_val_" + setting) / name
                    pc = out / "evidence" / "runs" / cr.name / ("eval_val_" + setting) / name
                    if pa.is_file() and pc.is_file() and read_json(pa) != read_json(pc):
                        checks.append(name + " ordering/content differs")
                ea = out / "evidence" / "runs" / ar.name / "meta/pip_freeze.txt"
                ec = out / "evidence" / "runs" / cr.name / "meta/pip_freeze.txt"
                if ea.is_file() and ec.is_file() and sorted(text(ea).splitlines()) != sorted(text(ec).splitlines()):
                    checks.append("pip environment declarations differ")
            comparison = "matched" if not checks and cfgstate == "matched" else "metadata_only" if not checks and cfgstate == "metadata_only" else "not_comparable"
            audits.append(dict(dataset=ds, seed=seed, setting=setting, comparison_status=comparison,
                               config_audit=cfgstate, detail=detail + ("; " + "; ".join(checks) if checks else "")))
            row = dict(dataset=ds, seed=seed, setting=setting, comparison_status=comparison)
            if comparison in ("matched", "metadata_only"):
                for col in FIELDS:
                    row["A_" + col], row["C_" + col] = a.get(col), c.get(col)
                    row["delta_" + col.replace("_pct", "_pp")] = c[col] - a[col] if a.get(col) is not None and c.get(col) is not None else None
            ac.append(row)
    effects = []
    for setting in ("cw", "ow"):
        vals = {v: usable.get(("ut-zappos", 3351, v, setting)) for v in "ACDE"}
        if all(vals.values()):
            consistent = all(len({vals[v].get(field) for v in "ACDE"}) == 1 for field in
                             ["source_manifest_sha256", "knowledge_manifest_sha256", "dataset_manifest_sha256", "n_images", "n_candidates", "score_mode"])
            if not consistent:
                ev.issue("ut_acde_" + setting, "ACDE provenance/candidate metadata differ; arithmetic contrasts withheld")
                continue
            # These are descriptive score contrasts, not causal/ANOVA estimates.
            for metric in ("auc_pct", "hm_pct"):
                a, c, d, e = [vals[v][metric] for v in "ACDE"]
                for name, value in [("C-A", c-a), ("D-A", d-a), ("E-C", e-c), ("E-D", e-d), ("E-A", e-a), ("E-C-D+A", e-c-d+a)]:
                    effects.append(dict(dataset="ut-zappos", seed=3351, setting=setting, metric=metric, contrast=name, value_pp=value))
    diag = (args.lambda_dir or exp / "diagnostics" / "lambda_post_v1").resolve()
    lambda_rows, selection = [], {}
    for filename in ["lambda_selection.json", "lambda_post_results.csv"]:
        ev.copy(diag / filename, Path("lambda_post") / filename)
    csvpath = out / "evidence" / "lambda_post" / "lambda_post_results.csv"
    selpath = csvpath.with_name("lambda_selection.json")
    if csvpath.is_file() and selpath.is_file():
        try:
            with csvpath.open(encoding="utf-8-sig", newline="") as f:
                original = list(csv.DictReader(f))
            expected = {(lab, lam) for lab in ["Dpost", "Epost"] for lam in [0, .025, .05, .1, .2]}
            if len(original) != 10 or {(r["diagnostic"], float(r["lambda"])) for r in original} != expected:
                raise ValueError("lambda grid is incomplete/duplicated")
            selection = read_json(selpath)
            for r in original:
                for field in ["cw_auc_pct", "cw_hm_pct", "ow_auc_pct", "ow_hm_pct", "ow_unseen_pct", "delta_cw_auc_pp", "delta_cw_hm_pp", "delta_ow_auc_pp", "delta_ow_hm_pp", "delta_ow_unseen_pp"]:
                    if not math.isfinite(float(r[field])):
                        raise ValueError("non-finite lambda CSV value: " + field)
                label, lam = r["diagnostic"], float(r["lambda"])
                row = dict(r)
                row["lambda"] = lam
                chosen = float(selection[label]["selected_lambda"])
                row["selected"] = lam == chosen
                constraints = [("delta_cw_hm_pp", -.5), ("delta_ow_auc_pp", -.5), ("delta_ow_unseen_pp", -2)]
                failed = [key for key, threshold in constraints if float(r[key]) < threshold]
                source_eligible = str(r["eligible"]).lower() in ("true", "1")
                if source_eligible != (not failed):
                    raise ValueError("eligible flag contradicts constraints for " + label + " " + str(lam))
                row["failed_constraints"] = ";".join(failed)
                if row["selected"]:
                    if not source_eligible:
                        raise ValueError("declared selected lambda is ineligible")
                    row["selection_explanation"] = "selected_by_existing_rule"
                elif failed:
                    row["selection_explanation"] = "protective_constraints_failed"
                else:
                    selected_auc = float(selection[label]["selected_row"]["cw_auc_pct"])
                    delta = float(r["cw_auc_pct"]) - selected_auc
                    row["selection_explanation"] = "lower_primary_metric" if delta < -1e-8 else "tie_larger_lambda" if abs(delta) <= 1e-8 else "inconsistent_selection_review_required"
                    if delta > 1e-8:
                        ev.issue(label, "eligible candidate exceeds declared selected CW AUC")
                raw_ok = True
                for setting in ["cw", "ow"]:
                    rel = "{}_l{}_{}".format(label, lam, setting)
                    folder = diag / rel
                    for name in ["metrics.json", "config_eval_effective.yml", "candidates.json", "image_ids.json", "labels.npz"]:
                        ev.copy(folder / name, Path("lambda_post") / rel / name)
                    snap = out / "evidence" / "lambda_post" / rel / "metrics.json"
                    if not snap.is_file():
                        raw_ok = False
                        ev.issue(label, "missing candidate metrics: " + rel)
                        continue
                    meta = read_json(snap)
                    mm = metrics(meta)
                    if meta.get("variant") != label or meta.get("setting") != setting or float(meta.get("lambda_k", -1)) != lam:
                        raise ValueError("candidate identity mismatch: " + rel)
                    for suffix, field in [("auc_pct", "auc_pct"), ("hm_pct", "hm_pct")]:
                        if abs(float(r[setting + "_" + suffix]) - mm[field]) > 1e-6:
                            raise ValueError("CSV/raw metric mismatch: " + rel)
                    baseline = "A" if label == "Dpost" else "C"
                    anchor = usable.get(("ut-zappos", 3351, baseline, setting))
                    if anchor:
                        if meta.get("checkpoint_sha256") != anchor["checkpoint_sha256"]:
                            raise ValueError("posthoc candidate uses different anchor checkpoint: " + rel)
                        for suffix, field in [("auc_pp", "auc_pct"), ("hm_pp", "hm_pct")]:
                            if abs(float(r["delta_" + setting + "_" + suffix]) - (mm[field] - anchor[field])) > 1e-6:
                                raise ValueError("posthoc delta is not relative to declared anchor: " + rel)
                        if setting == "ow" and (abs(float(r["ow_unseen_pct"]) - mm["best_unseen_pct"]) > 1e-6 or
                                                abs(float(r["delta_ow_unseen_pp"]) - (mm["best_unseen_pct"] - anchor["best_unseen_pct"])) > 1e-6):
                            raise ValueError("OW unseen CSV/anchor delta mismatch: " + rel)
                    else:
                        raw_ok = False
                    if lam == 0:
                        if not anchor:
                            raw_ok = False
                        elif abs(mm["auc_pct"] - anchor["auc_pct"]) > 1e-6 or abs(mm["hm_pct"] - anchor["hm_pct"]) > 1e-6 or meta.get("checkpoint_sha256") != anchor["checkpoint_sha256"]:
                            raise ValueError("lambda=0 does not reproduce anchor: " + rel)
                row["candidate_raw_metrics_verified"] = raw_ok
                lambda_rows.append(row)
        except Exception as e:
            ev.issue("lambda_grid", type(e).__name__ + ": " + str(e))
    else:
        ev.issue("lambda_grid", "selection JSON or grid CSV missing")
    cache_rows = []
    if args.cache_root:
        for ds, folder in [("ut-zappos", "ut-zap50k"), ("mit-states", "mit-states"), ("cgqa", "cgqa")]:
            root = args.cache_root / folder
            snaps = {name: ev.copy(root / name, Path("knowledge") / folder / name) for name in ["cache_manifest.json", "manifest.json", "report.json", "labels.json"]}
            try:
                man = read_json(snaps["cache_manifest.json"])
                rep = read_json(snaps["report.json"])
                cache_rows.append(dict(dataset=ds, n_attrs=len(man["attrs"]), n_objs=len(man["objs"]), dimension=man["dimension"],
                                       prior_source=man.get("prior_source"), knowledge_status=man.get("knowledge_status"),
                                       description_OOV_fraction=rep.get("description_OOV_fraction"), template_K_degenerate=rep.get("template_K_degenerate"),
                                       description_fallback_sentences=rep.get("description_fallback_sentences"),
                                       template_K_min=rep.get("K_oa_text", {}).get("min"), template_K_max=rep.get("K_oa_text", {}).get("max"),
                                       template_K_mean=rep.get("K_oa_text", {}).get("mean"), template_K_population_std=rep.get("K_oa_text", {}).get("population_std"),
                                       manifest_sha256=sha(snaps["cache_manifest.json"])))
            except Exception as e:
                ev.issue(ds + "_knowledge", "cache report unavailable: " + str(e))
    audit_dir = args.audit_dir or exp / "audit" / "ac_transfer_v1"
    for path in sorted(audit_dir.glob("smoke_*.*")) if audit_dir.is_dir() else []:
        ev.copy(path, Path("audits") / audit_dir.name / path.name)
    for ds in ["mit-states", "cgqa"]:
        path = out / "evidence" / "audits" / audit_dir.name / ("smoke_" + ds + ".json")
        try:
            if read_json(path).get("status") != "MVP_SMOKE_PASSED":
                raise ValueError("smoke did not pass")
        except Exception as e:
            ev.issue(ds + "_smoke", "smoke audit missing/invalid: " + str(e))
    for path in sorted((exp / "audit").glob("smoke_mvp_ut_s3351.*")):
        ev.copy(path, Path("audits") / path.name)
    for path in sorted((exp / "protocol").glob("*")) if (exp / "protocol").is_dir() else []:
        if path.is_file() and path.suffix in (".json", ".sha256", ".txt", ".md"):
            ev.copy(path, Path("protocol") / path.name)
    for row in ac:
        if row["comparison_status"] != "matched":
            ev.issue(row["dataset"] + "_" + row["setting"], "A/C comparison is " + row["comparison_status"])
    complete = len(usable) == 16 and len(lambda_rows) == 10 and all(r["candidate_raw_metrics_verified"] for r in lambda_rows) and all(r["comparison_status"] == "matched" for r in ac) and not ev.issues
    for filename, rows in [("run_inventory.csv", inventory), ("all_val_metrics.csv", all_rows), ("ac_comparison.csv", ac),
                           ("pair_audit.csv", audits), ("ut_acde_effects.csv", effects), ("lambda_analysis.csv", lambda_rows),
                           ("epoch_curves.csv", curves), ("knowledge_summary.csv", cache_rows), ("issues.csv", ev.issues)]:
        save_csv(tables / filename, rows)
    save_json(out / "evidence_manifest.json", ev.files)
    save_json(out / "collection.json", dict(version=VERSION, created_at=datetime.now().astimezone().isoformat(), experiment=str(exp),
                                             expected_unique_mvp_runs=8, expected_mvp_evaluations=16, expected_lambda_evaluations=20,
                                             verified_mvp_evaluations=len(usable), complete=complete, issues=len(ev.issues),
                                             checkpoints_rehashed=False, note="Checkpoint identities checked against saved hash declarations; model weights are not copied or rehashed."))
    report = "# CPF/KCC 阶段实验报告（自动数据草稿）\n\n"
    report += "生成时间：{}。数据状态：{}。\n\n".format(datetime.now().astimezone().isoformat(), "已完整核验" if complete else "存在缺失或待核验项，见 issues.csv")
    report += "本报告仅包含当前 MVP validation 与固定 checkpoint 的 lambda 后处理。历史 baseline test 应另附表；尚未实现的 EKR/USG 不计作完成实验。\n\n"
    report += "## 1. 实验范围与完成状态\n\n"
    report += "预期 8 个独立训练 run：UT 的 A/C/D/E，加 MIT 和 C-GQA 的 A/C；每个 run 的 CW/OW 是同一个 checkpoint 的两种评价。另有 2×5×2=20 次 lambda 后处理评价，不是 20 个训练 run。\n\n"
    report += markdown_table(inventory, [("dataset", "数据集"), ("variant", "变体"), ("seed", "seed"), ("status", "状态"), ("cw_ow_verified", "两评价核验"), ("best_epoch", "最佳epoch")])
    report += "\n## 2. 三数据集 A/C 主结果\n\n"
    report += markdown_table(ac, [("dataset", "数据集"), ("seed", "seed"), ("setting", "设置"), ("A_auc_pct", "A AUC%"), ("C_auc_pct", "C AUC%"), ("delta_auc_pp", "ΔAUC pp"), ("delta_hm_pp", "ΔHM pp"), ("comparison_status", "可比性")])
    report += "\n绝对值为百分数，差值为百分点；每数据集目前只有一对 seed，不能计算 seed 方差或宣称跨 seed 稳定/统计显著。CW/OW、三个数据集均逐项报告，不挑选正结果。\n\n"
    matched = [r for r in ac if r["comparison_status"] == "matched" and r.get("delta_auc_pp") is not None]
    report += "已核验的 A/C AUC 比较 {} 个，其中 C−A 为正 {} 个、为负 {} 个、精确持平 {} 个。这是描述性计数，不是独立重复次数。\n\n".format(len(matched), sum(r["delta_auc_pp"] > 0 for r in matched), sum(r["delta_auc_pp"] < 0 for r in matched), sum(r["delta_auc_pp"] == 0 for r in matched))
    report += "## 3. UT 消融与交互指标\n\n"
    report += markdown_table([r for r in all_rows if r["dataset"] == "ut-zappos"], [("setting", "设置"), ("variant", "变体"), ("auc_pct", "AUC%"), ("hm_pct", "HM%"), ("best_seen_pct", "best-seen%"), ("best_unseen_pct", "best-unseen%"), ("row_status", "数据状态")])
    report += "\nC−A 为文本路径效果，D−A 为当前校准路径效果，E−C 为文本存在时再加校准的效果。E−C−D+A 只作描述性差值，不是已建立的因果交互。完整结果见 ut_acde_effects.csv。\n\n"
    report += "## 4. Lambda 后处理\n\n"
    report += markdown_table([dict(diagnostic=k, selected_lambda=v.get("selected_lambda"), primary=v.get("selection_metric")) for k, v in selection.items()], [("diagnostic", "诊断"), ("selected_lambda", "选择lambda"), ("primary", "主指标")])
    report += "\n逐候选解释见 lambda_analysis.csv。选择零仅说明本知识版本、网格、checkpoint 和保护约束下未采用正强度校准；不证明所有非零候选的所有指标都更差，也不证明外部知识普遍无效。后处理与重训 D/E 分开报告。\n\n"
    report += "## 5. 学习过程与知识质量\n\n"
    report += "epoch_curves.csv 来源于 train.log 的四位小数输出，只用于曲线/收敛诊断；正式表用 metrics.json 的完整精度。loss 为最近一次日志累计均值，未必覆盖整轮。knowledge_summary.csv 记录 OOV、模板 K 与审核标记；bulk_trusted_unverified 不等于人工审核完成。\n\n"
    report += "## 6. 待填写的分析与决定\n\n"
    report += "- 跨数据集方向：逐项说明 C−A，特别记录持平/下降的设置。\n- 收敛：最佳 epoch 是否落在预算末端，是否存在训练下降而 validation 恶化。\n- 错误机制：如运行 bias0 分析，只将其归因于固定 bias=0 的 top-1，不能解释成 best-HM 点预测。\n- 下一步：依据完整 validation 对照决定多 seed、知识质量修订或停止该设置。\n- 限制：单 seed、同一 validation 用于选 checkpoint、重构知识来源与未完成模型变体。\n\n"
    report += "## 7. 异常与证据\n\n"
    report += markdown_table(ev.issues, [("scope", "范围"), ("message", "问题")]) if ev.issues else "未发现本汇总器能够检查的缺失/一致性问题；这不等于已重新运行训练或完成语义审核。\n"
    report += "\n原始证据快照及 SHA256 见 evidence/ 和 evidence_manifest.json。checkpoint 权重与大型 scores.dat 没有复制/重哈希，应继续保留在原 run 路径；不要清理原 run。\n"
    reference = Path(__file__).resolve().parent / "historical_baseline_reference.json"
    if reference.is_file():
        try:
            history = read_json(reference)
            for item in history["sources"]:
                path = Path(__file__).resolve().parent / item["package_file"]
                if sha(path) != item["sha256"]:
                    raise ValueError("historical reference digest mismatch")
                ev.copy(path, Path("historical_baseline") / path.name)
            ev.copy(reference, Path("historical_baseline") / reference.name)
            save_csv(tables / "historical_baseline_test_reference.csv", history["records"])
            report += "\n## 附表：历史 CPF 复现 test（独立参考）\n\n"
            report += markdown_table(history["records"], [("dataset", "数据集"), ("seed", "seed"), ("phase", "phase"), ("setting", "设置"), ("auc_pct", "AUC%"), ("hm_pct", "HM%"), ("best_seen_pct", "Seen%"), ("best_unseen_pct", "Unseen%")])
            report += "\n来源是本机已归档的 final_results.txt，原文已随工具包保留。它们与当前 MVP validation 分表，不计算跨 phase 增益，也不把其中记录的论文参考数字当作已核验的同配置比较。\n"
            save_json(out / "evidence_manifest.json", ev.files)
        except Exception as e:
            raise RuntimeError("historical reference cannot be trusted: " + str(e))
    (out / "阶段实验报告_自动草稿.md").write_text(report, encoding="utf-8")
    output_files = sorted(p for p in out.rglob("*") if p.is_file())
    with (out / "report_files.sha256").open("w", encoding="utf-8") as f:
        for path in output_files:
            f.write(sha(path) + "  " + str(path.relative_to(out)).replace("\\", "/") + "\n")
    print("REPORT:", out)
    print("VERIFIED MVP EVALUATIONS: {}/16; lambda rows: {}/10; issues: {}".format(len(usable), len(lambda_rows), len(ev.issues)))
    print("COMPLETE:", complete)
    if args.require_complete and not complete:
        print("Report saved, but final completeness gate failed. Inspect tables/issues.csv; choose a new output after repair.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
