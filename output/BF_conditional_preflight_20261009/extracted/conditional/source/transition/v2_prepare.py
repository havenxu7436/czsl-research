"""Prepare and verify one frozen UT v2 revision. Python3.8/torch1.13 compatible."""
import argparse
import copy
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml

from .bank import build as build_bank, derangement, source_hash
from .common import SEMANTIC_MODES, code_manifest, load_bank, read_json, sha, write_json
from .configs import make, recover_config

SEEDS = [0, 3351, 2026]
FOLDER = "ut-zap50k"
DATASET = "ut-zappos"
VARIANTS = list(SEMANTIC_MODES)


def native_identity(value):
    """Only explicit path/variant/seed and stage bookkeeping may differ."""
    value = copy.deepcopy(value)
    for key in ("root_dir",):
        value["DATASET"].pop(key, None)
    for key in ("kcc_cache", "kcc_text", "description_bank", "semantic_text"):
        value["MODEL"].pop(key, None)
    for key in ("seed", "checkpoint_dir", "log_dir", "kcc_run_dir", "train_name",
                "kcc_stage", "kcc_variant"):
        value["TRAIN"].pop(key, None)
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def builder_module(project):
    spec = importlib.util.spec_from_file_location("_ut_v2_cache_builder", project / "tools/build_knowledge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_assets(project):
    assets = project / "v2_assets"
    frozen = read_json(assets / "assets_sha256.json")
    for name, digest in frozen.items():
        target = (assets / name).resolve()
        require(assets.resolve() in target.parents and sha(target) == digest,
                "Frozen v2 asset changed: " + name)
    source = assets / "knowledge" / FOLDER
    manifest = read_json(source / "manifest.json")
    require(manifest["status"] == "assistant_reviewed_train_images" and not manifest["human_reviewed"],
            "Review provenance changed")
    for name, digest in manifest["knowledge_sha256"].items():
        require(sha(source / name) == digest, "Knowledge checksum changed: " + name)
    labels = {"attrs": (source / "attributes.txt").read_text().splitlines(),
              "objs": (source / "objects.txt").read_text().splitlines()}
    groups = read_json(assets / "review_evidence/selected_train_images.json")
    group_map = {(g["kind"], g["label"]): g for g in groups}
    review = read_json(source / "item_review.json")
    texts = {"a": read_json(source / "ae_dict.json"), "o": read_json(source / "oe_dict.json")}
    require(len(groups) == 28 and len(review["items"]) == 84, "Review coverage is incomplete")
    identities = set()
    unique_images = set()
    for group in groups:
        require(len(group["selected"]) == 20, "Each frozen board must have 20 panels")
        for row in group["selected"]:
            require(row["split"] == "train", "Non-training review image")
            unique_images.add(row["image"])
    require(len(unique_images) == 528, "Selected training identities changed")
    for kind, axis in (("a", "attrs"), ("o", "objs")):
        require(list(texts[kind]) == labels[axis], "V2 description order differs from labels")
        for label, sentences in texts[kind].items():
            require(len(sentences) == len(set(sentences)) == 3, "Need exactly three distinct sentences")
    for row in review["items"]:
        identity = (row["kind"], row["label"], row["number"])
        require(identity not in identities, "Duplicate review row")
        identities.add(identity)
        require(texts[row["kind"]][row["label"]][row["number"] - 1] == row["revised"],
                "Dictionary differs from reviewed sentence")
        group = group_map[row["kind"], row["label"]]
        require(row["board_sha256"] == sha(assets / "review_evidence" / group["board"]),
                "Review board checksum differs")
        require(row["status"] == "assistant_review_complete" and not row["human_reviewed"],
                "Pending review or unsupported human approval")
        for key in ("support_panels", "exception_or_unresolved_visual_panels"):
            require(all(1 <= n <= 20 for n in row[key]), "Panel reference outside frozen board")
    return assets, source, labels, review, groups


def data_integrity(root, assets, decode_train=False):
    """No image content is used to tune descriptions. Test images are never opened."""
    data = root / "data" / FOLDER
    gallery = read_json(assets / "review_evidence/manifest.json")
    for name in ("metadata_compositional-split-natural.t7", "compositional-split-natural/train_pairs.txt"):
        old_name = "/data/xhw/czsl_repro/data/" + FOLDER + "/" + name
        require(sha(data / name) == gallery["inputs_sha256"][old_name], "Training metadata/split changed: " + name)
    split = data / "compositional-split-natural"
    train_pairs = {tuple(x.split()) for x in (split / "train_pairs.txt").read_text().splitlines() if x.strip()}
    records = torch.load(data / "metadata_compositional-split-natural.t7", map_location="cpu")
    train = [r for r in records if r["set"] == "train" and (r["attr"], r["obj"]) in train_pairs]
    require(len(train) == 22998 and len(train_pairs) == 83, "Training population changed")
    train_map = {r["image"]: r for r in train}
    for group in read_json(assets / "review_evidence/selected_train_images.json"):
        for r in group["selected"]:
            original = train_map.get(r["image"])
            require(original is not None and (original["attr"], original["obj"]) == (r["attr"], r["obj"]),
                    "Review image identity absent from train split")
    failures = []
    from PIL import Image
    for i, row in enumerate(train, 1):
        target = data / "images" / row["image"]
        if not target.is_file():
            failures.append({"image": row["image"], "error": "missing native CPF training image"})
        elif decode_train:
            try:
                with Image.open(target) as image:
                    image.load()
                    image.convert("RGB").load()
            except Exception as error:
                failures.append({"image": row["image"], "error": str(error)})
        if i % 5000 == 0:
            print("TRAIN IMAGE CHECK", i, flush=True)
    # Native validation will need these paths; existence is checked without reading content.
    validation = [r for r in records if r["set"] == "val"]
    for row in validation:
        if not (data / "images" / row["image"]).is_file():
            failures.append({"image": row["image"], "error": "missing native CPF validation image"})
    if failures:
        raise ValueError("Native dataset image check failed; no automatic replacement: " + json.dumps(failures[:10]))
    return {"train_images": len(train), "train_images_decoded": len(train) if decode_train else 0,
            "validation_paths_checked": len(validation), "test_images_opened": 0,
            "selected_training_identities_verified": 528}


def table_integrity(project, root, work, verify_sources=True):
    assets, source, labels, review, groups = verify_assets(project)
    cache = work / "cache" / FOLDER
    old_cache = root / "knowledge/kcc0601_v1/cache_trusted_v1" / FOLDER
    builder = builder_module(project)
    builder.verify(cache, verify_sources=verify_sources)
    cm = read_json(cache / "manifest.json")
    old_cm = read_json(old_cache / "manifest.json")
    require(cm["knowledge_status"] == "assistant_reviewed_train_images", "Cache has wrong review status")
    require(cm["attrs"] == labels["attrs"] and cm["objs"] == labels["objs"], "Cache label order changed")
    require(source_hash(cm, "glove.6B.300d.txt") == source_hash(old_cm, "glove.6B.300d.txt"),
            "GloVe differs from the frozen v1 encoder")
    bank, bm = load_bank(work / "bank" / FOLDER, labels["attrs"], labels["objs"])
    require(bm["original_cache_manifest_sha256"] == sha(cache / "manifest.json"), "Bank uses another cache")
    require(bm["glove_sha256"] == source_hash(cm, "glove.6B.300d.txt"), "Bank encoder mismatch")
    require(bm["knowledge_inputs"] == {n: sha(source / n) for n in bm["knowledge_inputs"]},
            "Bank knowledge changed")
    desc = read_json(work / "bank" / FOLDER / "descriptions.json")
    frozen_perms = read_json(assets / "protocol/shuffle_mapping.json")
    reports = {}
    for offset, (kind, axis) in enumerate((("a", "attrs"), ("o", "objs"))):
        table = bank[kind]
        raw = torch.load(cache / ("W_" + kind + "_name_raw.pt"), map_location="cpu")
        old_name_path = old_cache / ("W_" + kind + "_name_raw.pt")
        require(sha(old_name_path) == old_cm["files_sha256"][old_name_path.name], "V1 name table changed")
        old_raw = torch.load(old_name_path, map_location="cpu")
        require(torch.equal(raw, old_raw), "V2 changed raw name embeddings")
        perm = derangement(len(labels[axis]), 1729 + offset)
        require(perm == frozen_perms[kind]["permutation"] == table["permutation"].tolist(),
                "Frozen shuffle mapping changed")
        require(tuple(table["units"].shape) == (len(labels[axis]), 4, 300), "Bank dimensions changed")
        require(bool(table["mask"].all()), "A description slot is missing")
        require(torch.equal(table["unit"], F.normalize(raw, dim=-1)), "Unit-name control changed")
        require(torch.equal(table["units"][:, 0], table["shuffle_units"][:, 0]), "Shuffle changed names")
        require(torch.equal(table["shuffle_units"][:, 1:], table["units"][perm, 1:]),
                "Shuffle is not the fixed complete description donor mapping")
        require(torch.allclose(table["correct"], F.normalize(table["units"].mean(1), dim=-1), atol=1e-6),
                "Correct aggregation differs")
        require(torch.allclose(table["shuffle"], F.normalize(table["shuffle_units"].mean(1), dim=-1), atol=1e-6),
                "Shuffled aggregation differs")
        require(float(table["residual"].sum(1).abs().max()) < 1e-6, "Residuals do not center to zero")
        for name in ("units", "correct", "unit", "shuffle", "residual"):
            require(bool(torch.isfinite(table[name]).all()), "Nonfinite bank vector")
        rows = desc[kind]
        require(not any(r["fallback"] for r in rows), "Description fallback is not allowed in v2")
        for row in rows:
            require(row["text"] in read_json(source / ("ae_dict.json" if kind == "a" else "oe_dict.json"))[row["label"]],
                    "Bank sentence differs from reviewed dictionary")
            require(len(row["oov"]) / len(row["tokens"]) <= 0.5, "More than half a sentence is OOV")
        total = sum(len(r["tokens"]) for r in rows)
        oov = sum(len(r["oov"]) for r in rows)
        require(oov / total <= 0.10, "More than 10 percent of description tokens are OOV")
        reports[kind] = {"fallback_sentences": 0, "OOV_fraction": oov / total,
                         "name_vectors_equal_v1": True, "permutation": perm,
                         "residual_sum_max_abs": float(table["residual"].sum(1).abs().max())}
    return reports


def config_integrity(project, root, work):
    exp = work / "experiments/semantic_v2"
    index = read_json(exp / "config_index_semantic_e10.json")
    expected_ids = ["semantic_ut-zappos_{}_s{}_e10".format(v, s) for s in SEEDS for v in VARIANTS]
    require(set(index) == set(expected_ids), "Expected exactly 12 matched configs")
    template = read_json(project / "v2_assets/protocol/base_effective_config.json")
    for rid in expected_ids:
        cfg = yaml.safe_load(Path(index[rid]).read_text())
        require(native_identity(cfg) == native_identity(template), "Native protocol differs: " + rid)
        seed = int(cfg["TRAIN"]["seed"])
        variant = cfg["TRAIN"]["kcc_variant"]
        require(rid == "semantic_ut-zappos_{}_s{}_e10".format(variant, seed), "Run identity mismatch")
        require(cfg["MODEL"]["semantic_text"] == SEMANTIC_MODES[variant], "Wrong semantic control mode")
        require(cfg["MODEL"]["kcc_text"] == (variant != "A_raw"), "Wrong KCC text flag")
        require(Path(cfg["MODEL"]["kcc_cache"]).resolve() == (work / "cache" / FOLDER).resolve(), "Old cache reused")
        require(Path(cfg["MODEL"]["description_bank"]).resolve() == (work / "bank" / FOLDER).resolve(), "Old bank reused")
        require(Path(cfg["DATASET"]["root_dir"]).resolve() == (root / "data" / FOLDER).resolve(), "Dataset root changed")
        require(Path(cfg["TRAIN"]["kcc_run_dir"]).resolve() == (exp / "runs" / rid).resolve(), "Run path collision")
        require(cfg["TRAIN"]["max_epoch"] == cfg["TRAIN"]["final_max_epoch"] == 10 and
                cfg["TRAIN"]["start_epoch"] == 1 and not cfg["TRAIN"]["kcc_test_at_end"], "Budget/test policy changed")
        require(not cfg["MODEL"]["kcc_calib"] and cfg["MODEL"]["kcc_lambda"] == 0 and
                not cfg["MODEL"]["kcc_logic"] and not cfg["MODEL"]["kcc_feature"], "Static K re-enabled")
    return {rid: sha(index[rid]) for rid in expected_ids}


def verify_prepared(project, root, work, full=True):
    frozen = read_json(work / "prepared.json")
    require(frozen["code_manifest"] == code_manifest(project), "Installed code changed after preparation")
    require(frozen["assets_manifest_sha256"] == sha(project / "v2_assets/assets_sha256.json"), "Asset freeze changed")
    for path, digest in frozen["lineage_sha256"].items():
        require(sha(path) == digest, "Original lineage input changed: " + path)
    for path, digest in frozen["prepared_artifacts_sha256"].items():
        require(sha(path) == digest, "Prepared artifact changed: " + path)
    configs = config_integrity(project, root, work)
    require(configs == frozen["config_sha256"], "Config changed after preparation")
    tables = table_integrity(project, root, work, verify_sources=full)
    print("V2 CPU INTEGRITY PASSED: 84 reviews, 28 labels, fixed donors, 12 matched configs", flush=True)
    return frozen, tables


def prepare(project, root, work, decode_train=True):
    if (work / "prepared.json").exists():
        verify_prepared(project, root, work)
        print("PREPARATION ALREADY COMPLETE", work, flush=True)
        return
    torch.set_num_threads(1)
    torch.manual_seed(91001)
    assets, source, labels, review, groups = verify_assets(project)
    for name, digest in read_json(project / "transition_install.json")["source_files"].items():
        require(sha(project / name) == digest, "Native source changed: " + name)
    legacy = root / "experiments/kcc0601_glove_mvp_v1"
    original_run = legacy / "runs/mvp_ut-zappos_C_legacy_s3351_l0"
    original = recover_config(original_run)
    template = read_json(assets / "protocol/base_effective_config.json")
    require(native_identity(original) == native_identity(template), "Server baseline differs from frozen week5 protocol")
    old_source = root / "knowledge/kcc0601_v1/trusted_v1" / FOLDER
    source_manifest = read_json(source / "manifest.json")
    for name in ("ae_dict.json", "oe_dict.json", "oi_dict.json"):
        require(sha(old_source / name) == source_manifest["previous_knowledge_sha256"][name], "V1 knowledge lineage changed")
    data_report = data_integrity(root, assets, decode_train)
    work.mkdir(parents=True, exist_ok=True)
    cache_cfg = work / "cache_build_config.yml"
    content = yaml.safe_dump(original, sort_keys=True)
    if cache_cfg.exists():
        require(cache_cfg.read_text() == content, "Cache build config changed")
    else:
        cache_cfg.write_text(content, encoding="utf-8")
    builder = builder_module(project)
    cache = work / "cache" / FOLDER
    glove = root / "data/glove/glove.6B.300d.txt"
    require(glove.is_file(), "Missing original GloVe file")
    old_cache = root / "knowledge/kcc0601_v1/cache_trusted_v1" / FOLDER
    require(sha(glove) == source_hash(read_json(old_cache / "manifest.json"), glove.name),
            "GloVe changed from the frozen v1 cache")
    if (cache / "manifest.json").is_file():
        builder.verify(cache, verify_sources=True)
    else:
        builder.build(SimpleNamespace(project_root=project, cfg=cache_cfg,
            data_root=root / "data" / FOLDER, knowledge_dir=source, output_dir=cache,
            glove_file=glove, descriptions_per_class=3, prior_source="template",
            knowledge_status="assistant_reviewed_train_images", seed=91001))
    if not (work / "bank" / FOLDER / "manifest.json").is_file():
        build_bank(SimpleNamespace(project=project, cache_root=work / "cache",
            knowledge_root=assets / "knowledge", out_root=work / "bank", glove=glove,
            datasets=[FOLDER], permutation_seed=1729))
    table_report = table_integrity(project, root, work)
    exp = work / "experiments/semantic_v2"
    make(SimpleNamespace(root=root, exp=exp, legacy_exp=legacy, bank_root=work / "bank",
        cache_root=work / "cache", stage="semantic", datasets=[DATASET], seeds=SEEDS,
        epochs=10, variants=VARIANTS, use_legacy_anchor=False, regime="frozen",
        smoke_only=False, gate=None, transfer_from=None))
    config_hashes = config_integrity(project, root, work)
    lineage = [original_run / "train.log", old_cache / "manifest.json",
               root / "data" / FOLDER / "metadata_compositional-split-natural.t7"]
    lineage += [old_source / n for n in ("ae_dict.json", "oe_dict.json", "oi_dict.json")]
    prepared = [cache / "manifest.json", work / "bank" / FOLDER / "manifest.json", cache_cfg,
                exp / "config_index_semantic_e10.json"]
    record = {"version": "ut_semantic_v2_20261006", "code_manifest": code_manifest(project),
        "assets_manifest_sha256": sha(assets / "assets_sha256.json"),
        "project": str(project), "root": str(root), "work": str(work),
        "lineage_sha256": {str(p): sha(p) for p in lineage},
        "prepared_artifacts_sha256": {str(p): sha(p) for p in prepared},
        "config_sha256": config_hashes, "seeds": SEEDS, "variants": VARIANTS,
        "data_checks": data_report, "table_checks": table_report, "human_reviewed": False,
        "training_performed": False, "test_images_opened": 0,
        "controls": "Fresh 12 runs; never mix v1 results or legacy C anchors",
        "gate_thresholds_changed": False, "USG_graph_route": False}
    write_json(work / "prepared.json", record)
    verify_prepared(project, root, work, full=False)
    print("PREPARED FOR GPU SMOKE AND 12 MATCHED RUNS", exp, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.verify_only:
        verify_prepared(project, args.root.resolve(), args.work.resolve())
    else:
        prepare(project, args.root.resolve(), args.work.resolve())


if __name__ == "__main__":
    main()
