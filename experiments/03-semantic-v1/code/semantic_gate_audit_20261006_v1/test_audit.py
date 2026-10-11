"""Failure-detection tests and a small end-to-end synthetic audit. No GPU."""
import argparse
import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import audit_semantic_gate as a


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    a.write(path, value)


def tensor_fixture(n, seed):
    raw = torch.randn(n, 300)
    units = F.normalize(torch.randn(n, 4, 300), dim=-1)
    units[:, 0] = F.normalize(raw, dim=-1)
    mask = torch.ones(n, 4, dtype=torch.bool)
    order = np.random.RandomState(seed).permutation(n)
    permutation = np.empty(n, dtype=np.int64)
    permutation[order] = np.roll(order, 1)
    perm = torch.tensor(permutation)
    mean = units.mean(1)
    shuffled = units[perm].clone()
    shuffled[:, 0] = units[:, 0]
    sm = shuffled.mean(1)
    correct = F.normalize(mean, dim=-1)
    return {"units": units, "mask": mask, "mean": mean, "residual": units - mean[:, None],
            "correct": correct, "unit": units[:, 0], "shuffle": F.normalize(sm, dim=-1),
            "shuffle_units": shuffled, "shuffle_mask": mask, "shuffle_mean": sm,
            "permutation": perm}, raw, correct


def make_fixture(root):
    exp, legacy, project, report = [root / name for name in ("exp", "legacy", "project", "report")]
    code = project / "models/semantic_control.py"
    code.parent.mkdir(parents=True)
    code.write_text("# fixture only\n")
    bank_dir = root / "knowledge/ekr0601_transition_v1/description_bank_v1/ut-zap50k"
    cache = root / "knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k"
    knowledge = root / "knowledge/kcc0601_v1/trusted_v1/ut-zap50k"
    labels = {"attrs": ["red", "blue", "rough"], "objs": ["shoe", "boot"]}
    files, bank, records = {}, {}, {}
    for offset, (kind, axis) in enumerate((("a", "attrs"), ("o", "objs"))):
        table, raw, correct = tensor_fixture(len(labels[axis]), 1729 + offset)
        bank[kind] = table
        texts = {x: [x + " desc " + str(i) for i in range(3)] for x in labels[axis]}
        records[kind] = [{"label": x, "text": t, "fallback": False, "oov": [], "tokens": t.split()}
                         for x in texts for t in texts[x]]
        save_json(knowledge / ("ae_dict.json" if kind == "a" else "oe_dict.json"), texts)
        cache.mkdir(parents=True, exist_ok=True)
        for suffix, tensor in (("name_raw", raw), ("K", correct)):
            name = "W_{}_{}.pt".format(kind, suffix)
            torch.save(tensor, cache / name)
            files[name] = a.sha(cache / name)
    save_json(cache / "labels.json", labels)
    save_json(cache / "manifest.json", {"files_sha256": files})
    save_json(bank_dir / "labels.json", labels)
    torch.save(bank, bank_dir / "bank.pt")
    save_json(bank_dir / "descriptions.json", records)
    save_json(bank_dir / "report.json", {})
    manifest = {"permutation_seed": 1729, "knowledge_status": "fixture_only",
        "original_cache_manifest_sha256": a.sha(cache / "manifest.json"),
        "files_sha256": {n: a.sha(bank_dir / n) for n in ("labels.json", "bank.pt", "descriptions.json", "report.json")},
        "knowledge_inputs": {n: a.sha(knowledge / n) for n in ("ae_dict.json", "oe_dict.json")}}
    save_json(bank_dir / "manifest.json", manifest)
    split = root / "data/ut-zap50k/compositional-split-natural"
    split.mkdir(parents=True)
    (split / "train_pairs.txt").write_text("red shoe\n")
    text_d = [0.508647104712125, 4.662171454858033, 2.1254419202762023]
    shuf_d = [-2.0889476341193998, 1.0196149922969298, 0.269668060436139]
    rows, sources = [], {}
    for index, seed in enumerate((0, 3351, 2026)):
        for variant in a.MODES:
            is_old = seed == 3351 and variant in a.ALIASES
            rid = ("mvp_ut-zappos_{}_legacy_s3351_l0".format(a.ALIASES[variant]) if is_old else
                   "semantic_ut-zappos_{}_s{}_e10".format(variant, seed))
            run = (legacy if is_old else exp) / "runs" / rid
            run.mkdir(parents=True)
            cfg = {"DATASET": {"name": "ut-zappos"}, "config_name": "ut-zappos-czsl",
                "EVAL": {"open_world": False, "topk": 1},
                "MODEL": {"kcc_calib": False, "kcc_lambda": 0.0, "wordembs": "glove",
                          "kcc_feature": False, "kcc_logic": False, "kcc_control": "none",
                          "kcc_text": variant != "A_raw", "kcc_cache": str(cache)},
                "TRAIN": {"seed": seed, "max_epoch": 10, "start_epoch": 1, "aug_epoch": 37,
                          "lr": 1e-4, "kcc_test_at_end": False, "kcc_run_dir": str(run),
                          "kcc_variant": a.ALIASES[variant] if is_old else variant}}
            if not is_old:
                cfg["MODEL"].update(semantic_text=a.MODES[variant], description_bank=str(bank_dir))
            cw = {"A_raw": 39, "A_unit": 40, "C_text": 40 + text_d[index],
                  "C_shuffle": 40 + text_d[index] - shuf_d[index]}[variant]
            log = yaml.safe_dump(cfg) + "Random seed: {}\n".format(seed)
            for epoch in range(1, 11):
                log += "Val/Test Epoch {}: AUC={:.4f} | best_hm=0.5500\n".format(epoch, (cw - 10 + epoch) / 100)
            log += "Loading validation-best checkpoint from epoch 10: fixture\nDone: ut-zappos-czsl\n"
            (run / "train.log").write_text(log)
            sources[str(run / "train.log")] = a.sha(run / "train.log")
            ckpt = run / "checkpoints/best_auc.pth"
            ckpt.parent.mkdir()
            ckpt.write_bytes((str(seed) + variant).encode())
            digest = a.sha(ckpt)
            if not is_old:
                (run / "effective_config.yml").write_text(yaml.safe_dump(cfg))
                save_json(run / "status.json", {"status": "train_and_val_complete"})
                save_json(run / "train_complete.json", {"checkpoint_sha256": digest, "epochs": 10, "best_epoch": 10})
                save_json(run / "meta/code_manifest.json", {"models/semantic_control.py": a.sha(code)})
                save_json(run / "meta/bank_manifest.json", manifest)
            for setting in ("cw", "ow"):
                directory = run / ("eval_val_" + setting)
                metric = {"dataset": "ut-zappos", "phase": "val", "setting": setting, "seed": seed,
                          "variant": a.ALIASES[variant] if is_old else variant,
                          "checkpoint_sha256": digest, "best_epoch": 10,
                          "bank_manifest_sha256": a.sha(bank_dir / "manifest.json"),
                          "metrics_raw": {"AUC": cw / 100, "best_hm": .55, "best_unseen": .6}}
                save_json(directory / "metrics.json", metric)
                sources[str(directory / "metrics.json")] = a.sha(directory / "metrics.json")
                save_json(directory / "image_ids.json", ["a.jpg", "b.jpg"])
                save_json(directory / "candidates.json", [["red", "shoe"], ["blue", "boot"]])
                np.savez_compressed(str(directory / "labels.npz"), attr=[0, 1], obj=[0, 1], pair=[0, 1],
                                    top1_at_bias0=[0, 0] if variant == "C_shuffle" else [0, 1])
                rows.append(dict(dataset="ut-zappos", phase="val", setting=setting, seed=seed, variant=variant,
                    regime="native_semantic", epochs=10, run_id=rid, reused_legacy=is_old,
                    auc_pct=cw, hm_pct=55, best_unseen_pct=60))
    report.mkdir()
    with (report / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    save_json(report / "collection.json", {"inputs": sources, "fingerprint": fingerprint})
    gate = a.recompute_gate(a.table_from_rows(rows, [0, 3351, 2026]), [0, 3351, 2026])
    save_json(exp / "protocol/gate.json", {"planned_seeds": [0, 3351, 2026],
             "report_fingerprint": fingerprint, "datasets": {"ut-zappos": gate}})
    return argparse.Namespace(root=str(root), exp=str(exp), legacy_exp=str(legacy), project=str(project),
               report=str(report), gate=str(exp / "protocol/gate.json"), seeds=[0, 3351, 2026], out=str(exp / "diagnostics/audit"))


class AuditTests(unittest.TestCase):
    def test_config_lr_difference_detected(self):
        x = {"TRAIN": {"lr": .0001, "seed": 0, "train_name": "a"}}
        y = {"TRAIN": {"lr": .0002, "seed": 2026, "train_name": "b"}}
        self.assertEqual(list(a.config_differences(x, y)), ["TRAIN.lr"])

    def test_corrupted_shuffle_and_self_donor(self):
        table, raw, old = tensor_fixture(3, 1729)
        self.assertIn("max_abs_errors", a.check_tensor_table(table, raw, old, 1729))
        # A correct shuffle MUST retain its own category name, not the donor's name.
        bad = copy.deepcopy(table)
        bad["shuffle_units"][:, 0] = table["units"][table["permutation"], 0]
        with self.assertRaisesRegex(ValueError, "shuffle_units"):
            a.check_tensor_table(bad, raw, old, 1729)
        bad = copy.deepcopy(table)
        bad["permutation"] = torch.arange(3)
        with self.assertRaisesRegex(ValueError, "derangement"):
            a.check_tensor_table(bad, raw, old, 1729)

    def test_curve_incomplete_and_best_epoch(self):
        real = Path(__file__).resolve().parent / "fixtures/legacy_C_s3351_curve.txt"
        parsed = a.parse_curve(real.read_text(encoding="utf-8"))
        self.assertEqual(parsed["best_epoch"], 6)
        with self.assertRaisesRegex(ValueError, "1..10"):
            a.parse_curve("Val/Test Epoch 1: AUC=0.4 | best_hm=0.55\nDone:\n")

    def test_full_audit_negative_gate_and_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_fixture(Path(tmp))
            before = {str(p): a.sha(p) for p in Path(tmp).rglob("*") if p.is_file()}
            with redirect_stdout(io.StringIO()):
                code = a.run(args)
            self.assertEqual(code, 0)
            result = a.read(Path(args.out) / "audit.json")
            self.assertTrue(result["integrity_passed"])
            self.assertFalse(result["gate_recomputed"]["proceed"])
            self.assertAlmostEqual(result["gate_recomputed"]["mean_C_minus_Aunit_pp"], 2.4320868266154534)
            self.assertAlmostEqual(result["gate_recomputed"]["mean_C_minus_shuffle_pp"], -0.26655486046211035)
            self.assertEqual(result["gate_recomputed"]["by_seed"][0]["seed"], 0)
            self.assertEqual(len(result["run_audits"]), 12)
            self.assertEqual(len(result["paired_bias0"]), 12)
            self.assertTrue(all(a.sha(p) == digest for p, digest in before.items()))
            with self.assertRaisesRegex(ValueError, "already exists"):
                a.run(args)

    def test_changed_checkpoint_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_fixture(Path(tmp))
            bad = Path(args.exp) / "runs/semantic_ut-zappos_C_shuffle_s0_e10/checkpoints/best_auc.pth"
            bad.write_bytes(b"corrupt")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(a.run(args), 2)
            checks = a.read(Path(args.out) / "audit.json")["checks"]
            self.assertTrue(any(c["check"] == "run_s0_C_shuffle" and c["status"] == "FAIL" for c in checks))

    def test_wrong_image_order_refuses_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_fixture(Path(tmp))
            exp = Path(args.exp)
            left = exp / "runs/semantic_ut-zappos_C_text_s0_e10/eval_val_cw"
            right = exp / "runs/semantic_ut-zappos_C_shuffle_s0_e10/eval_val_cw"
            save_json(right / "image_ids.json", ["b.jpg", "a.jpg"])
            with self.assertRaisesRegex(ValueError, "image order"):
                a.paired_top1(left, right, {"attrs": ["red", "blue", "rough"], "objs": ["shoe", "boot"]}, {("red", "shoe")})

    def test_duplicate_report_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_fixture(Path(tmp))
            with (Path(args.report) / "runs.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                a.table_from_rows(rows + [rows[0]], args.seeds)


if __name__ == "__main__":
    torch.manual_seed(114)
    unittest.main(verbosity=2)
