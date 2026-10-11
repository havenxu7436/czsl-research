"""Exercise the real snapshot installer, builder, config and reporting interfaces."""
import argparse
import ast
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
from transition import bank as bank_tools
from transition import configs, summary, cli, noise_summary
from transition.common import DATASETS, read_json, sha, write_json


def fake_merged_config(ds="ut-zappos", seed=3351):
    return {"config_name": "ut-zappos-czsl",
        "DATASET": {"name": ds, "root_dir": "/data/fake", "splitname": "compositional-split-natural"},
        "MODEL": {"wordembs": "glove", "kcc_calib": False, "kcc_lambda": 0.0,
                  "kcc_text": True,
                  "cosine_cls_temp": 0.05},
        "TRAIN": {"seed": seed, "max_epoch": 10, "aug_epoch": 37, "batch_size": 64,
                  "test_batch_size": 64, "num_workers": 4, "lr": 0.0001,
                  "lr_encoder": 1e-5, "lr_word_embedding": 2.5e-6, "wd": 5e-5},
        "EVAL": {"open_world": False, "topk": 1}}


def fake_metrics(ds, variant, seed, setting, auc, hm=55, unseen=60):
    return {"dataset": ds, "variant": variant, "seed": seed, "phase": "val", "setting": setting,
            "training_regime": "native_semantic", "checkpoint_sha256": "fake-checkpoint",
            "metrics_raw": {"AUC": auc / 100, "best_hm": hm / 100,
                            "best_seen": .68, "best_unseen": unseen / 100,
                            "hm_seen": .54, "hm_unseen": .56}}


def test_summary_and_gate(root):
    exp = root / "summary_exp"
    for seed in (0, 3351, 2026):
        for variant, auc in (("A_raw", 39), ("A_unit", 41), ("C_text", 44), ("C_shuffle", 40)):
            run = exp / "runs" / (variant + "_" + str(seed))
            write_json(run / "status.json", {"status": "train_and_val_complete"})
            (run / "effective_config.yml").write_text(yaml.safe_dump(fake_merged_config(seed=seed)))
            for setting in ("cw", "ow"):
                write_json(run / ("eval_val_" + setting) / "metrics.json",
                           fake_metrics("ut-zappos", variant, seed, setting, auc))
    report = root / "summary_report"
    summary.summarize(argparse.Namespace(exp=exp, out=report, legacy_exp=None))
    summary.summarize(argparse.Namespace(exp=exp, out=report, legacy_exp=None))
    decision = root / "decision.json"
    summary.gate(argparse.Namespace(report=report, out=decision,
                                    datasets=["ut-zappos"], seeds=[0, 3351, 2026]))
    assert read_json(decision)["datasets"]["ut-zappos"]["proceed"]
    missing = root / "missing_decision.json"
    summary.gate(argparse.Namespace(report=report, out=missing,
                                    datasets=["cgqa"], seeds=[0, 3351, 2026]))
    assert not read_json(missing)["datasets"]["cgqa"]["proceed"]


def test_config_generation(root):
    legacy = root / "legacy"
    run = legacy / "runs/mvp_ut-zappos_C_legacy_s3351_l0"
    run.mkdir(parents=True)
    (run / "train.log").write_text(yaml.safe_dump(fake_merged_config()) + "Random seed: 3351\n")
    (run / "checkpoints").mkdir()
    (run / "checkpoints/best_auc.pth").write_bytes(b"test checkpoint identity")
    bank_root = root / "config_bank"
    write_json(bank_root / "ut-zap50k/manifest.json", {})
    exp = root / "config_exp"
    args = argparse.Namespace(root=root, exp=exp, legacy_exp=legacy, bank_root=bank_root,
             datasets=["ut-zappos"], seeds=[3351], stage="semantic", variants=["A_unit", "C_shuffle"],
             epochs=10, use_legacy_anchor=True, regime="frozen", smoke_only=False, gate=None,
             transfer_from=None)
    configs.make(args)
    correct = yaml.safe_load((exp / "configs/semantic_ut-zappos_A_unit_s3351_e10.yml").read_text())
    assert correct["MODEL"]["semantic_text"] == "unit"
    assert correct["TRAIN"]["max_epoch"] == 10
    args.stage, args.variants, args.epochs, args.smoke_only = "branch", ["R5"], 3, True
    configs.make(args)
    smoke = yaml.safe_load((exp / "configs/smoke_ut-zappos_R5_s3351_e3.yml").read_text())
    assert smoke["TRAIN"]["branch_smoke_only"]
    gate = root / "config_gate.json"
    write_json(gate, {"datasets": {"ut-zappos": {"proceed": True}}})
    args.smoke_only, args.gate = False, str(gate)
    configs.make(args)
    assert (exp / "configs/branch_ut-zappos_R5_s3351_e3.yml").is_file()
    args.regime, args.variants, args.epochs = "joint", ["C_budget", "R5"], 10
    configs.make(args)
    assert (exp / "configs/joint_ut-zappos_C_budget_s3351_e10.yml").is_file()


def test_real_builder(root, source):
    project = source
    knowledge, caches, output = root / "knowledge", root / "cache", root / "banks"
    ks, cache = knowledge / "toy", caches / "toy"
    ks.mkdir(parents=True)
    cache.mkdir(parents=True)
    labels = {"attrs": ["red", "blue", "rough"], "objs": ["ball", "car"]}
    texts = {"a": {x: [x + " visible", x + " surface", "visible shape"] for x in labels["attrs"]},
             "o": {x: [x + " visible", x + " shape", "surface shape"] for x in labels["objs"]}}
    tokens = sorted(set(" ".join(t for table in texts.values() for sentences in table.values()
                                  for t in sentences).split()))
    torch.manual_seed(114)
    vectors = {t: torch.randn(300) for t in tokens}
    glove = root / "glove.6B.300d.txt"
    glove.write_text("".join(t + " " + " ".join(str(float(v)) for v in vectors[t]) + "\n"
                             for t in tokens))
    spec = importlib.util.spec_from_file_location("original_builder", project / "tools/build_knowledge.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    encoded, digest = builder.load_needed_glove(glove, set(tokens))
    write_json(cache / "labels.json", labels)
    write_json(ks / "ae_dict.json", texts["a"])
    write_json(ks / "oe_dict.json", texts["o"])
    write_json(ks / "oi_dict.json", {"vehicle": ["car"], "thing": ["ball"]})
    write_json(ks / "manifest.json", {"knowledge_status": "bulk_trusted_unverified"})
    for axis, filename in (("attrs", "attributes.txt"), ("objs", "objects.txt")):
        (ks / filename).write_text("\n".join(labels[axis]) + "\n")
    (ks / "review_status.csv").write_text("kind,label,status\n")
    files = {}
    for kind, axis in (("a", "attrs"), ("o", "objs")):
        raw, correct, records = builder.make_primitive_table(labels[axis], texts[kind],
                        {label: [label] for label in labels[axis]}, encoded, kind)
        for suffix, value in (("name_raw", raw), ("K", correct)):
            name = "W_" + kind + "_" + suffix + ".pt"
            torch.save(value, cache / name)
            files[name] = sha(cache / name)
    names = ("attributes.txt", "objects.txt", "ae_dict.json", "oe_dict.json", "oi_dict.json",
             "manifest.json", "review_status.csv")
    write_json(cache / "manifest.json", {"knowledge_status": "bulk_trusted_unverified",
                "files_sha256": files,
                "input_sha256": dict({str(ks / name): sha(ks / name) for name in names},
                                     **{str(glove): digest})})
    args = argparse.Namespace(project=project, cache_root=caches, knowledge_root=knowledge,
             out_root=output, datasets=["toy"], glove=glove, permutation_seed=1729)
    bank_tools.build(args)
    bank = torch.load(output / "toy/bank.pt", map_location="cpu")
    assert torch.equal(bank["a"]["correct"], torch.load(cache / "W_a_K.pt"))
    assert bank["a"]["residual"].sum(1).abs().max() < 1e-6
    assert not torch.allclose(bank["a"]["shuffle"], bank["a"]["correct"])
    try:
        bank_tools.build(args)
        raise AssertionError("Overwrite should be refused")
    except FileExistsError:
        pass


def test_final_freeze(root):
    legacy = root / "legacy"
    run = legacy / "runs/mvp_ut-zappos_C_legacy_s3351_l0"
    (run / "status.txt").write_text("train_and_val_complete\n")
    for setting in ("cw", "ow"):
        metric = fake_metrics("ut-zappos", "C", 3351, setting, 44)
        metric["checkpoint_sha256"] = sha(run / "checkpoints/best_auc.pth")
        write_json(run / ("eval_val_" + setting) / "metrics.json", metric)
    exp = root / "config_exp"
    values = fake_merged_config()
    values["MODEL"]["description_bank"] = str(root / "config_bank/ut-zap50k")
    values["TRAIN"]["kcc_variant"] = "C_text"
    rid = "semantic_ut-zappos_C_text_s3351_e10"
    (exp / "configs" / (rid + ".yml")).write_text(yaml.safe_dump(values))
    output = root / "final_freeze.json"
    cli.freeze_final(argparse.Namespace(exp=exp, runs=[rid], out=output, legacy_exp=legacy))
    frozen = read_json(output)
    assert frozen["runs"][0]["legacy_reused"]
    assert not frozen["runs"][0]["force_off"]
    assert frozen["runs"][0]["config_sha256"] == sha(exp / "configs" / (rid + ".yml"))
    try:
        cli.freeze_final(argparse.Namespace(exp=exp, runs=[rid], out=output, legacy_exp=legacy))
        raise AssertionError("Frozen test protocol must not overwrite")
    except FileExistsError:
        pass


def test_noise_report(root):
    exp = root / "noise_exp"
    names = []
    for variant in ("R5", "R6", "R0noise", "R3noise"):
        names.append(variant + "_3351")
        run = exp / "runs" / names[-1]
        for setting in ("cw", "ow"):
            clean = fake_metrics("ut-zappos", variant, 3351, setting, 44)
            clean["anchor_sha256"] = "same-C-anchor"
            write_json(run / ("eval_val_" + setting) / "metrics.json", clean)
            for ratio in (0.1, 0.3):
                for seed in (11, 22, 33):
                    value = copy.deepcopy(clean)
                    value.update(noise_ratio=ratio, noise_seed=seed)
                    value["metrics_raw"]["AUC"] -= ratio / (2 if variant == "R6" else 1)
                    path = run / "noise_v1" / ("p{}_n{}_{}".format(ratio, seed, setting)) / "metrics.json"
                    if variant == "R3noise" and setting == "ow" and ratio == .3 and seed == 33:
                        final_path, final_value = path, value
                    else:
                        write_json(path, value)
    args = argparse.Namespace(exp=exp, out=root / "noise_report", runs=names, noise_version="noise_v1")
    try:
        noise_summary.summarize_noise(args)
        raise AssertionError("Incomplete perturbation matrix must fail")
    except ValueError as e:
        assert "incomplete" in str(e)
    write_json(final_path, final_value)
    noise_summary.summarize_noise(args)
    import csv
    with (args.out / "noise_summary_by_training_seed.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16 and all(int(r["n_noise_seeds"]) == 3 for r in rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-fixture", required=True, type=Path)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="ekr_integration_") as temp:
        root = Path(temp)
        target = root / "installed"
        subprocess.run([sys.executable, str(BUNDLE / "install.py"), "--source",
                        str(args.source_fixture), "--destination", str(target)], check=True)
        for path in target.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), feature_version=(3, 8))
        assert "def raw_forward" in (target / "models/ekr_base.py").read_text()
        original = ast.parse((target / "models/kcc_cpf.py").read_text())
        generated = ast.parse((target / "models/ekr_base.py").read_text())
        old_method = next(n for n in original.body if isinstance(n, ast.ClassDef)).body[-1]
        new_method = next(n for n in generated.body if isinstance(n, ast.ClassDef)).body[0]
        assert [ast.dump(n) for n in old_method.body[:-1]] == [ast.dump(n) for n in new_method.body[:-1]]
        print("PASS installer and exact feature prefix")
        test_real_builder(root, target)
        print("PASS real builder aggregation, permutation and overwrite protection")
        test_config_generation(root)
        print("PASS semantic/frozen/joint config generation")
        test_summary_and_gate(root)
        print("PASS reporting, paired controls and missing-data gate")
        test_final_freeze(root)
        print("PASS legacy reuse and immutable final-test protocol")
        test_noise_report(root)
        print("PASS complete noise matrix and separate perturbation statistics")
    print("ALL CPU INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
