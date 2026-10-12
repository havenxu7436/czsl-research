import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import empty_output, read_json, sha, write_json


def derangement(n, seed):
    if n < 2:
        raise ValueError("Description permutation requires at least two labels")
    order = np.random.RandomState(seed).permutation(n)
    result = np.empty(n, dtype=np.int64)
    result[order] = np.roll(order, 1)
    assert not np.any(result == np.arange(n))
    return result.tolist()


def source_hash(manifest, filename):
    matches = [v for k, v in manifest["input_sha256"].items()
               if Path(k).name == filename]
    if len(matches) != 1:
        raise ValueError("Cannot uniquely locate frozen source: " + filename)
    return matches[0]


def construct(labels, texts, raw_names, glove, builder, permutation):
    counts = [1 + len(texts[a]) for a in labels]
    rmax = max(counts)
    units = torch.zeros(len(labels), rmax, 300)
    mask = torch.zeros(len(labels), rmax, dtype=torch.bool)
    records = []
    for i, label in enumerate(labels):
        vectors = [builder.normalize(raw_names[i])]
        for text in texts[label]:
            tokens = builder.description_tokens(text)
            kept = [glove[t] for t in tokens if t in glove]
            sentence = torch.stack(kept).mean(0) if kept else torch.zeros(300)
            fallback = bool(sentence.norm() < builder.EPS)
            if fallback:
                sentence = raw_names[i]
            vectors.append(builder.normalize(sentence))
            records.append({"label": label, "text": text, "tokens": tokens,
                            "oov": [t for t in tokens if t not in glove],
                            "fallback": fallback})
        units[i, :len(vectors)] = torch.stack(vectors)
        mask[i, :len(vectors)] = True
    mean = (units * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    residual = (units - mean[:, None]) * mask[..., None]
    shuffled = torch.zeros_like(units)
    shuffled_mask = torch.zeros_like(mask)
    for i, donor in enumerate(permutation):
        donor_n = counts[donor]
        shuffled[i, 0] = units[i, 0]
        shuffled[i, 1:donor_n] = units[donor, 1:donor_n]
        shuffled_mask[i, :donor_n] = True
    shuffled_mean = shuffled.sum(1) / shuffled_mask.sum(1, keepdim=True)
    return {"units": units, "mask": mask, "mean": mean, "residual": residual,
            "correct": builder.normalize(mean), "unit": units[:, 0],
            "shuffle": builder.normalize(shuffled_mean),
            "shuffle_units": shuffled, "shuffle_mask": shuffled_mask,
            "shuffle_mean": shuffled_mean,
            "permutation": torch.tensor(permutation)}, records


def build(args):
    project = Path(args.project)
    spec = importlib.util.spec_from_file_location("_existing_glove_builder",
                                                project / "tools/build_knowledge.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    needed, prepared = set(), []
    for folder in args.datasets:
        cache = Path(args.cache_root) / folder
        source = Path(args.knowledge_root) / folder
        out = Path(args.out_root) / folder
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Bank exists: " + str(out))
        manifest = read_json(cache / "manifest.json")
        labels = read_json(cache / "labels.json")
        inputs = {}
        for name in ("attributes.txt", "objects.txt", "oi_dict.json", "oe_dict.json",
                     "ae_dict.json", "manifest.json", "review_status.csv"):
            digest = sha(source / name)
            if digest != source_hash(manifest, name):
                raise ValueError("Knowledge differs from original C cache: " + str(source / name))
            inputs[name] = digest
        tables = {}
        for kind, axis in (("a", "attrs"), ("o", "objs")):
            texts = read_json(source / ("ae_dict.json" if kind == "a" else "oe_dict.json"))
            if list(texts) != labels[axis]:
                raise ValueError("Description label order mismatch: " + folder)
            for descriptions in texts.values():
                for text in descriptions:
                    needed.update(builder.description_tokens(text))
            for filename in ("W_" + kind + "_name_raw.pt", "W_" + kind + "_K.pt"):
                if sha(cache / filename) != manifest["files_sha256"][filename]:
                    raise ValueError("Original cache tensor checksum mismatch: " + filename)
            tables[kind] = (texts, torch.load(cache / ("W_" + kind + "_name_raw.pt"),
                                             map_location="cpu"))
        prepared.append((folder, cache, source, out, manifest, labels, tables, inputs))
    glove, glove_hash = builder.load_needed_glove(Path(args.glove), needed)
    for folder, cache, source, out, manifest, labels, tables, inputs in prepared:
        if glove_hash != source_hash(manifest, Path(args.glove).name):
            raise ValueError("GloVe differs from the frozen C encoder")
        result, text_records, reports = {}, {}, {}
        for offset, (kind, axis) in enumerate((("a", "attrs"), ("o", "objs"))):
            permutation = derangement(len(labels[axis]), args.permutation_seed + offset)
            table, records = construct(labels[axis], tables[kind][0], tables[kind][1],
                                       glove, builder, permutation)
            old = torch.load(cache / ("W_" + kind + "_K.pt"), map_location="cpu")
            error = (table["correct"] - old).abs().max().item()
            if error > 1e-6:
                raise ValueError("Original C aggregation not reproduced: {} {} {}".format(folder, kind, error))
            # Preserve the literal original C table, including its floating-point values.
            table["correct"] = old
            result[kind] = table
            text_records[kind] = records
            norms = table["residual"].norm(dim=-1)[table["mask"]]
            reports[kind] = {"old_C_max_abs_error": error,
                             "residual_sum_max_abs": table["residual"].sum(1).abs().max().item(),
                             "residual_norm_median": norms.median().item(),
                             "near_zero_fraction": (norms < 1e-6).float().mean().item(),
                             "fallback_sentences": sum(r["fallback"] for r in records),
                             "oov_tokens": sum(len(r["oov"]) for r in records),
                             "total_tokens": sum(len(r["tokens"]) for r in records)}
        empty_output(out)
        torch.save(result, out / "bank.pt")
        write_json(out / "labels.json", labels)
        write_json(out / "descriptions.json", text_records)
        write_json(out / "report.json", reports)
        write_json(out / "manifest.json", {
            "version": "description_bank_transition_v1", "folder": folder,
            "knowledge_status": manifest["knowledge_status"],
            "original_cache_manifest_sha256": sha(cache / "manifest.json"),
            "knowledge_inputs": inputs, "glove_sha256": glove_hash,
            "permutation_seed": args.permutation_seed,
            "files_sha256": {name: sha(out / name) for name in
                             ("bank.pt", "labels.json", "descriptions.json", "report.json")}})
        print("BANK VERIFIED", folder, reports, flush=True)
