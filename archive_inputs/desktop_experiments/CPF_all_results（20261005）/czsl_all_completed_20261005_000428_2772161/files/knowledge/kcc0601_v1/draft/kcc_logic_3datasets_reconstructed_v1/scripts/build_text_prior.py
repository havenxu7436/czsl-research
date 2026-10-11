#!/usr/bin/env python3
"""Build a deterministic, non-official text-only object-attribute prior.

No pair lists, images, labels, or validation/test scores are read. This is a
lexical candidate prior, not an author-provided compatibility judgment.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


DATASETS = ("ut-zap50k", "mit-states", "cgqa")
STOP = {
    "a", "an", "the", "of", "to", "in", "is", "are", "with", "and", "or",
    "for", "by", "as", "at", "on", "from", "that", "which", "it", "its",
    "can", "be", "has", "have", "this", "object", "objects", "attribute",
    "property", "visual", "semantic", "state", "described", "concept",
    "group", "kind", "type", "category", "belongs", "named", "visible",
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tokens(text):
    return [word for word in re.findall(r"[a-z]+", text.lower())
            if word not in STOP and len(word) > 1]


def char_grams(words):
    grams = []
    for word in words:
        padded = "^" + word + "$"
        grams.extend(padded[i:i + 3] for i in range(len(padded) - 2))
    return grams


def vectors(documents, kind):
    counts = []
    df = Counter()
    for document in documents:
        words = tokens(document)
        items = words if kind == "word" else char_grams(words)
        count = Counter(items)
        counts.append(count)
        df.update(count.keys())
    n = len(documents)
    result = []
    for count in counts:
        weighted = {key: (1.0 + math.log(value)) *
                    (1.0 + math.log((n + 1.0) / (df[key] + 1.0)))
                    for key, value in count.items()}
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        result.append({key: value / norm for key, value in weighted.items()}
                      if norm else {})
    return result


def cosine(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def make_dataset(source):
    attrs = (source / "attributes.txt").read_text(encoding="utf-8").splitlines()
    objs = (source / "objects.txt").read_text(encoding="utf-8").splitlines()
    oe = json.loads((source / "oe_dict.json").read_text(encoding="utf-8"))
    ae = json.loads((source / "ae_dict.json").read_text(encoding="utf-8"))
    documents = [" ".join([label] + oe[label]) for label in objs]
    documents += [" ".join([label] + ae[label]) for label in attrs]
    word = vectors(documents, "word")
    chars = vectors(documents, "char")
    scores = []
    matrix = source / "K_oa_text_lexical.csv"
    with matrix.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["object"] + attrs)
        for oi, obj in enumerate(objs):
            row = []
            for ai in range(len(attrs)):
                j = len(objs) + ai
                value = 0.75 * cosine(word[oi], word[j]) + 0.25 * cosine(chars[oi], chars[j])
                value = max(0.0, min(1.0, value))
                row.append("{:.8f}".format(value))
                scores.append(value)
            writer.writerow([obj] + row)
    report = {
        "status": "unreviewed_lexical_candidate_not_official",
        "formula": "0.75*TFIDF_word_cosine + 0.25*TFIDF_character_trigram_cosine",
        "inputs": "primitive labels and OE/AE descriptions only; no split pair membership",
        "shape_object_by_attribute": [len(objs), len(attrs)],
        "min": min(scores), "max": max(scores),
        "mean": statistics.mean(scores),
        "population_std": statistics.pstdev(scores),
        "zero_fraction": sum(x == 0 for x in scores) / len(scores),
        "unique_8_decimal": len(set(round(x, 8) for x in scores)),
        "warning": "Lexical similarity is not calibrated real-world compatibility. Validate before formal use.",
    }
    (source / "K_oa_text_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["text_prior"] = {
        "filename": matrix.name, "report_filename": "K_oa_text_report.json",
        "status": report["status"], "sha256": digest(matrix),
        "builder_sha256": digest(Path(__file__)),
    }
    manifest["knowledge_sha256"][matrix.name] = digest(matrix)
    manifest["knowledge_sha256"]["K_oa_text_report.json"] = digest(
        source / "K_oa_text_report.json"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print("{}: shape={}x{} mean={:.5f} std={:.5f} unique={}".format(
        source.name, len(objs), len(attrs), report["mean"],
        report["population_std"], report["unique_8_decimal"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    for dataset in args.datasets:
        make_dataset(args.knowledge_root / dataset)


if __name__ == "__main__":
    main()
