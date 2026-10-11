#!/usr/bin/env python3
"""CPU-only frozen GloVe knowledge cache for the existing CPF baseline.

Python 3.8 / torch 1.13 compatible. Source knowledge is read-only.
Pair identities are used only for label alignment and coverage reporting.
"""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

VERSION = "glove_template_v1"
DIM = 300
EPS = 1e-8
TEMPLATES = ("a {object} that is {attribute}",
             "the {attribute} {object}",
             "{attribute} appearance of {object}")
FILLERS = (("a", "that", "is"), ("the",), ("appearance", "of"))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def centered_prior(k, eps=EPS):
    k = k.float()
    if not torch.isfinite(k).all():
        raise ValueError("non-finite K")
    std = k.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(k)
    return (k - k.mean()) / std


def normalize(x):
    if not torch.isfinite(x).all() or (x.norm(dim=-1) < EPS).any():
        raise ValueError("non-finite or zero embedding")
    return F.normalize(x, dim=-1)


def label_information(project, cfg_path, data_root):
    # Call the project's actual parse_split, without constructing/loading images.
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
    if cfg["MODEL"]["wordembs"] != "glove":
        raise ValueError("this builder requires the existing glove baseline")
    dataset_source = project / "dataset.py"
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location("_kcc_dataset_labels", dataset_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    split = cfg["DATASET"]["splitname"]
    proxy = SimpleNamespace(
        cfg=SimpleNamespace(DATASET=SimpleNamespace(
            root_dir=str(data_root), name=cfg["DATASET"]["name"])), split=split)
    attrs, objs, _, tr, va, te = module.CompositionDataset.parse_split(proxy)
    inputs = [cfg_path, dataset_source]
    inputs += [data_root / split / (s + "_pairs.txt") for s in ("train", "val", "test")]
    return list(attrs), list(objs), {"train": tr, "val": va, "test": te}, inputs


def baseline_aliases(path):
    # Extract literal maps from the actual initializer; do not import fasttext/gensim.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "load_word_embeddings")
    maps = {}
    for node in function.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("custom_map", "custom_map_vaw"):
                    maps[target.id] = ast.literal_eval(node.value)
    if "custom_map" not in maps or "custom_map_vaw" not in maps:
        raise ValueError("baseline alias maps changed; inspect word_embedding_utils.py")
    return maps["custom_map"], maps["custom_map_vaw"]


def name_tokens(label, aliases, token_aliases):
    # Matches baseline: token alias is applied only inside underscore labels.
    value = aliases.get(label, label).lower()
    if "_" in value:
        return [token_aliases.get(x, x) for x in value.split("_")]
    return [value]


def description_tokens(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def load_needed_glove(path, needed):
    vectors = {}
    digest = hashlib.sha256()
    print("Scanning baseline GloVe once; retaining only required tokens...", flush=True)
    with path.open("rb") as f:
        for number, raw in enumerate(f, 1):
            digest.update(raw)
            parts = raw.split(maxsplit=1)
            if len(parts) != 2:
                continue
            token = parts[0].decode("utf-8")
            if token not in needed:
                continue
            values = [float(x) for x in parts[1].split()]
            if len(values) != DIM:
                raise ValueError("GloVe dimension mismatch at line {}".format(number))
            vector = torch.tensor(values, dtype=torch.float32)
            if not torch.isfinite(vector).all():
                raise ValueError("non-finite GloVe token: " + token)
            vectors[token] = vector
            if number % 100000 == 0:
                print("GloVe lines scanned: {}".format(number), flush=True)
    return vectors, digest.hexdigest()


def make_primitive_table(labels, descriptions, label_tokens, glove, kind):
    raw_names, knowledge, records = [], [], []
    for label in labels:
        missing = [t for t in label_tokens[label] if t not in glove]
        if missing:
            raise ValueError("baseline name OOV {}: {}".format(label, missing))
        raw_name = torch.stack([glove[t] for t in label_tokens[label]]).mean(0)
        name = normalize(raw_name)
        sentence_vectors = [name]
        oov = Counter()
        total, kept, fallback = 0, 0, 0
        for text in descriptions[label]:
            tokens = description_tokens(text)
            known = [glove[t] for t in tokens if t in glove]
            oov.update(t for t in tokens if t not in glove)
            total += len(tokens)
            kept += len(known)
            sentence = torch.stack(known).mean(0) if known else torch.zeros(DIM)
            if sentence.norm() < EPS:
                sentence = raw_name
                fallback += 1
            sentence_vectors.append(normalize(sentence))
        out = normalize(torch.stack(sentence_vectors).mean(0))
        raw_names.append(raw_name)
        knowledge.append(out)
        records.append({"kind": kind, "label": label,
                        "name_tokens": label_tokens[label],
                        "description_tokens_total": total,
                        "description_tokens_kept": kept,
                        "oov_tokens": dict(sorted(oov.items())),
                        "fallback_sentences": fallback})
    return torch.stack(raw_names), torch.stack(knowledge), records


def stats(matrix):
    flat = matrix.flatten().float()
    return {"shape": list(matrix.shape), "min": flat.min().item(),
            "max": flat.max().item(), "mean": flat.mean().item(),
            "population_std": flat.std(unbiased=False).item(),
            "zero_fraction": (flat == 0).float().mean().item(),
            "quantiles": dict(zip(("q00", "q01", "q25", "q50", "q75", "q99", "q100"),
                                   torch.quantile(flat, torch.tensor([0., .01, .25, .5, .75, .99, 1.])).tolist()))}


def load_lexical(path, attrs, objs):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != ["object"] + attrs or len(rows) != len(objs) + 1:
        raise ValueError("lexical K header/shape/order mismatch")
    values = []
    for obj, row in zip(objs, rows[1:]):
        if len(row) != len(attrs) + 1 or row[0] != obj:
            raise ValueError("lexical K object order mismatch")
        values.append([float(x) for x in row[1:]])
    k = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(k).all() or (k < 0).any() or (k > 1).any():
        raise ValueError("invalid lexical K values")
    return k


def build(args):
    project = args.project_root.resolve()
    source = args.knowledge_dir.resolve()
    output = args.output_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("cache output must be separate from source knowledge")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("cache already exists; use a NEW output directory: " + str(output))
    attrs, objs, audit_pairs, inputs = label_information(project, args.cfg.resolve(), args.data_root.resolve())
    for filename, labels in (("attributes.txt", attrs), ("objects.txt", objs)):
        path = source / filename
        if path.read_text(encoding="utf-8-sig").splitlines() != labels:
            raise ValueError(filename + " differs from actual dataset label order")
        inputs.append(path)
    dictionaries = {kind: read_json(source / (kind + "_dict.json")) for kind in ("oi", "oe", "ae")}
    inputs += [source / (kind + "_dict.json") for kind in ("oi", "oe", "ae")]
    for kind, labels in (("oe", objs), ("ae", attrs)):
        if list(dictionaries[kind]) != labels:
            raise ValueError(kind + " label keys/order mismatch")
        for label in labels:
            texts = dictionaries[kind][label]
            if not isinstance(texts, list) or len(texts) != args.descriptions_per_class:
                raise ValueError("description count mismatch: " + label)
            if any(not isinstance(t, str) or not t.strip() for t in texts):
                raise ValueError("invalid description: " + label)
    children = [child for group in dictionaries["oi"].values() for child in group]
    if sorted(children) != sorted(objs) or len(children) != len(set(children)):
        raise ValueError("OI coverage/uniqueness mismatch")
    inputs += [source / "manifest.json", source / "review_status.csv"]
    source_manifest = read_json(source / "manifest.json")
    for filename, expected in source_manifest.get("knowledge_sha256", {}).items():
        if sha256(source / filename) != expected:
            raise ValueError("source knowledge checksum mismatch: " + filename)
    with (source / "review_status.csv").open(encoding="utf-8-sig", newline="") as f:
        reviews = list(csv.DictReader(f))
    identities = {(r["kind"], r["label"]) for r in reviews}
    expected = {(k, x) for k, labels in (("object", objs), ("attribute", attrs)) for x in labels}
    if identities != expected or len(reviews) != len(expected):
        raise ValueError("review_status identity/coverage mismatch")
    if any(r["status"] != "approved" for r in reviews):
        raise ValueError("source has pending entries; select your existing trusted_v1 directory")
    initializer = project / "models" / "word_embedding_utils.py"
    aliases, token_aliases = baseline_aliases(initializer)
    inputs.append(initializer)
    token_maps = {}
    for kind, labels in (("object", objs), ("attribute", attrs)):
        token_maps[kind] = {x: name_tokens(x, aliases, token_aliases) for x in labels}
        canonical = [tuple(token_maps[kind][x]) for x in labels]
        if len(set(canonical)) != len(canonical):
            raise ValueError(kind + " alias collision")
    needed = {t for mapping in token_maps.values() for tokens in mapping.values() for t in tokens}
    needed.update(t for kind in ("oe", "ae") for texts in dictionaries[kind].values()
                  for text in texts for t in description_tokens(text))
    needed.update(t for tokens in FILLERS for t in tokens)
    glove_path = (args.glove_file or (args.data_root.resolve().parent / "glove" / "glove.6B.300d.txt")).resolve()
    glove, glove_hash = load_needed_glove(glove_path, needed)
    missing_fillers = [t for tokens in FILLERS for t in tokens if t not in glove]
    if missing_fillers:
        raise ValueError("template filler OOV: " + str(missing_fillers))
    W_o_name, W_o_K, obj_records = make_primitive_table(objs, dictionaries["oe"], token_maps["object"], glove, "object")
    W_a_name, W_a_K, attr_records = make_primitive_table(attrs, dictionaries["ae"], token_maps["attribute"], glove, "attribute")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".building-", dir=str(output.parent)) as temporary:
        stage = Path(temporary) / "cache"
        stage.mkdir()
        (stage / "W_ao_K").mkdir()
        attr_sums = torch.stack([torch.stack([glove[t] for t in token_maps["attribute"][a]]).sum(0) for a in attrs])
        attr_counts = torch.tensor([len(token_maps["attribute"][a]) for a in attrs], dtype=torch.float32)
        raw_k = torch.empty(len(objs), len(attrs))
        for oi, obj in enumerate(objs):
            obj_tokens = token_maps["object"][obj]
            obj_sum = torch.stack([glove[t] for t in obj_tokens]).sum(0)
            sentences = []
            for fillers in FILLERS:
                filler_sum = torch.stack([glove[t] for t in fillers]).sum(0)
                means = (attr_sums + obj_sum + filler_sum) / (attr_counts + len(obj_tokens) + len(fillers)).unsqueeze(1)
                sentences.append(normalize(means))
            pair_embeddings = normalize(torch.stack(sentences).mean(0))
            proto = normalize(W_o_K[oi].unsqueeze(0) + W_a_K)
            raw_k[oi] = (pair_embeddings * proto).sum(-1)
            torch.save(pair_embeddings, stage / "W_ao_K" / ("obj_{:05d}.pt".format(oi)))
            if (oi + 1) % 50 == 0 or oi + 1 == len(objs):
                print("Pair blocks: {}/{}".format(oi + 1, len(objs)), flush=True)
        raw_range = (raw_k.max() - raw_k.min()).item()
        degenerate = raw_range < EPS or raw_k.std(unbiased=False).item() < EPS
        template_k = torch.full_like(raw_k, .5) if degenerate else (raw_k - raw_k.min()) / (raw_k.max() - raw_k.min())
        k = template_k
        if args.prior_source == "lexical":
            lexical_path = source / "K_oa_text_lexical.csv"
            k = load_lexical(lexical_path, attrs, objs)
            inputs.append(lexical_path)
        prior = centered_prior(k)
        tensors = {"W_o_name_raw.pt": W_o_name, "W_a_name_raw.pt": W_a_name,
                   "W_o_name_unit.pt": normalize(W_o_name), "W_a_name_unit.pt": normalize(W_a_name),
                   "W_o_K.pt": W_o_K, "W_a_K.pt": W_a_K,
                   "K_oa_text_raw.pt": raw_k, "K_oa_text_template.pt": template_k,
                   "K_oa_text.pt": k, "K_oa_prior_centered.pt": prior}
        for filename, tensor in tensors.items():
            torch.save(tensor, stage / filename)
        labels = {"attrs": attrs, "objs": objs}
        write_json(stage / "labels.json", labels)
        all_records = obj_records + attr_records
        total = sum(r["description_tokens_total"] for r in all_records)
        kept = sum(r["description_tokens_kept"] for r in all_records)
        obj_bad = {r["label"] for r in obj_records if r["fallback_sentences"]}
        attr_bad = {r["label"] for r in attr_records if r["fallback_sentences"]}
        coverage = {}
        for split, pairs in audit_pairs.items():
            flags = [o in obj_bad or a in attr_bad for a, o in pairs]
            coverage[split] = {"pairs": len(pairs), "template_coverage": 1.0,
                               "primitive_description_fallback_fraction": sum(flags) / len(flags) if flags else 0.0}
        rows = []
        for mode, descending in (("top", True), ("bottom", False)):
            for index in torch.argsort(k.flatten(), descending=descending)[:10].tolist():
                oi, ai = divmod(index, len(attrs))
                rows.append({"rank_group": mode, "object": objs[oi], "attribute": attrs[ai], "K": k[oi, ai].item()})
        report = {"version": VERSION, "prior_source": args.prior_source,
                  "knowledge_status": args.knowledge_status,
                  "human_review_inferred_from_approved": False,
                  "raw_template_K": stats(raw_k), "K_oa_text": stats(k),
                  "centered_prior": stats(prior), "template_K_degenerate": degenerate,
                  "template_K_near_constant_std_lt_1e-4": raw_k.std(unbiased=False).item() < 1e-4,
                  "uniform_prior_test": bool(torch.equal(centered_prior(torch.ones(2, 3)), torch.zeros(2, 3))),
                  "description_OOV_fraction": (total - kept) / total if total else 0.0,
                  "description_fallback_sentences": sum(r["fallback_sentences"] for r in all_records),
                  "primitive_records": all_records, "split_coverage_audit_only": coverage,
                  "top_bottom_pairs": rows,
                  "warning": "Text similarity is a candidate prior, not LLM feasibility or validated compatibility."}
        write_json(stage / "report.json", report)
        write_json(stage / "knowledge_coverage_report.json", report)
        input_hashes = {str(p.resolve()): sha256(p) for p in inputs}
        input_hashes[str(glove_path)] = glove_hash
        file_hashes = {str(p.relative_to(stage)).replace("\\", "/"): sha256(p)
                       for p in sorted(stage.rglob("*")) if p.is_file()}
        manifest = {"version": VERSION, "attrs": attrs, "objs": objs,
                    "dimension": DIM, "dtype": "float32", "encoder": "glove.6B.300d",
                    "knowledge_status": args.knowledge_status,
                    "source_manifest_status": source_manifest.get("status"),
                    "source_compatible_with_formal_results": source_manifest.get("compatible_with_formal_results"),
                    "prior_source": args.prior_source, "pair_axis_order": ["object", "attribute"],
                    "templates": TEMPLATES, "seed": args.seed,
                    "name_embedding": "actual baseline aliases; lowercase; underscore-token mean; raw and unit variants saved",
                    "description_embedding": "regex tokens; mean in-vocabulary GloVe; L2 normalize each sentence; mean(name plus descriptions); L2 normalize",
                    "OOV_policy": "skip missing description tokens; sentence with no usable vector falls back to name; name OOV is fatal",
                    "K_formula": "cosine(pair-template-set embedding, normalize(W_o_K + W_a_K)); global min-max; constants -> 0.5",
                    "prior_formula": "global (K-mean)/population_std; std<1e-8 -> zeros",
                    "semantic_inputs": "primitive labels, OE/AE descriptions, baseline GloVe; no pair membership",
                    "split_use": "actual dataset label ordering and coverage audit only",
                    "builder_sha256": sha256(Path(__file__)), "input_sha256": input_hashes,
                    "files_sha256": file_hashes}
        write_json(stage / "manifest.json", manifest)
        write_json(stage / "cache_manifest.json", manifest)
        if output.exists():
            output.rmdir()  # Only an empty destination is allowed.
        stage.rename(output)
    print("CACHE BUILT: {}".format(output), flush=True)
    print("W_o_K={} W_a_K={} K={}".format(list(W_o_K.shape), list(W_a_K.shape), list(k.shape)))
    print("K min/max/mean/std: {:.6f}/{:.6f}/{:.6f}/{:.6f}".format(
        k.min().item(), k.max().item(), k.mean().item(), k.std(unbiased=False).item()))
    print("Description OOV fraction: {:.4f}; template degenerate: {}".format(report["description_OOV_fraction"], degenerate))
    verify(output, verify_sources=False)


def verify(output, verify_sources=True):
    manifest = read_json(output / "manifest.json")
    if manifest != read_json(output / "cache_manifest.json"):
        raise ValueError("cache manifest copies differ")
    for filename, expected in manifest["files_sha256"].items():
        path = (output / filename).resolve()
        if output.resolve() not in path.parents or sha256(path) != expected:
            raise ValueError("cache checksum mismatch: " + filename)
    if verify_sources:
        for path, expected in manifest["input_sha256"].items():
            if sha256(Path(path)) != expected:
                raise ValueError("cache source checksum mismatch: " + path)
    n, m = len(manifest["objs"]), len(manifest["attrs"])
    for filename, shape in (("W_o_K.pt", (n, DIM)), ("W_a_K.pt", (m, DIM)),
                            ("K_oa_text.pt", (n, m)), ("K_oa_prior_centered.pt", (n, m))):
        value = torch.load(output / filename, map_location="cpu")
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError("cache tensor mismatch: " + filename)
        if filename.startswith("W_") and not torch.allclose(value.norm(dim=-1), torch.ones(shape[0]), atol=1e-5):
            raise ValueError("embedding unit norm mismatch: " + filename)
    k = torch.load(output / "K_oa_text.pt", map_location="cpu")
    prior = torch.load(output / "K_oa_prior_centered.pt", map_location="cpu")
    if not torch.equal(centered_prior(k), prior):
        raise ValueError("centered prior differs from saved K")
    if not torch.equal(centered_prior(torch.ones(2, 3)), torch.zeros(2, 3)):
        raise AssertionError("uniform prior did not vanish")
    print("CACHE VERIFY PASSED: " + str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--knowledge-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--glove-file", type=Path)
    parser.add_argument("--encoder", choices=["glove"], default="glove")
    parser.add_argument("--descriptions-per-class", type=int, default=3)
    parser.add_argument("--pair-mode", choices=["template"], default="template")
    parser.add_argument("--prior-source", choices=["template", "lexical"], default="template")
    parser.add_argument("--knowledge-status", choices=["bulk_trusted_unverified", "human_reviewed"], default="bulk_trusted_unverified")
    parser.add_argument("--seed", type=int, default=91001)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.verify_only:
        verify(args.output_dir.resolve())
    else:
        for name in ("cfg", "data_root", "knowledge_dir"):
            if getattr(args, name) is None:
                parser.error("--{} is required for building".format(name.replace("_", "-")))
        build(args)


if __name__ == "__main__":
    main()
