#!/usr/bin/env python3
"""Read-only structural and identity check against the actual server datasets."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


DATASETS = ("ut-zap50k", "mit-states", "cgqa")
SPLITS = ("train", "val", "test")


def canonical_digest(pairs):
    text = "".join("{}\t{}\n".format(a, o) for a, o in sorted(set(pairs)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pairs(path):
    pairs = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("{}:{}: malformed pair".format(path, number))
        pairs.append(tuple(parts))
    return pairs


def read_lines(path):
    return path.read_text(encoding="utf-8-sig").splitlines()


def check_dataset(name, data_root, knowledge_root, require_reviewed):
    split_root = data_root / name / "compositional-split-natural"
    source = knowledge_root / name
    if not split_root.is_dir() or not source.is_dir():
        raise FileNotFoundError("missing dataset or knowledge directory: {} / {}".format(split_root, source))
    by_split = {split: read_pairs(split_root / (split + "_pairs.txt")) for split in SPLITS}
    union = [pair for split in SPLITS for pair in by_split[split]]
    attrs = sorted({a for a, _ in union})
    objs = sorted({o for _, o in union})
    if read_lines(source / "attributes.txt") != attrs:
        raise AssertionError("{} attribute vocabulary/order mismatch".format(name))
    if read_lines(source / "objects.txt") != objs:
        raise AssertionError("{} object vocabulary/order mismatch".format(name))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["knowledge_sha256"].items():
        if file_digest(source / filename) != expected:
            raise AssertionError("{} {} SHA256 mismatch".format(name, filename))
    for split in SPLITS:
        got = canonical_digest(by_split[split])
        if got != manifest["canonical_pair_sha256"][split]:
            raise AssertionError("{} {} pair identity mismatch: {}".format(name, split, got))
    oi = json.loads((source / "oi_dict.json").read_text(encoding="utf-8"))
    oe = json.loads((source / "oe_dict.json").read_text(encoding="utf-8"))
    ae = json.loads((source / "ae_dict.json").read_text(encoding="utf-8"))
    children = [child for group in oi.values() for child in group]
    if sorted(children) != objs or len(children) != len(set(children)):
        raise AssertionError("{} OI coverage/uniqueness failure".format(name))
    if list(oe) != objs or list(ae) != attrs:
        raise AssertionError("{} OE/AE key order failure".format(name))
    for label, texts in list(oe.items()) + list(ae.items()):
        if (not isinstance(texts, list) or len(texts) != 3 or
                any(not isinstance(text, str) or not text.strip() for text in texts) or
                len({text.strip().casefold() for text in texts}) != 3):
            raise AssertionError("{}: {} invalid descriptions".format(name, label))
    prior_path = source / "K_oa_text_lexical.csv"
    if prior_path.is_file():
        with prior_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        if rows[0] != ["object"] + attrs or len(rows) != len(objs) + 1:
            raise AssertionError("{} text prior shape/header mismatch".format(name))
        for obj, row in zip(objs, rows[1:]):
            if row[0] != obj or len(row) != len(attrs) + 1:
                raise AssertionError("{} text prior row alignment failure".format(name))
            if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1
                   for value in row[1:]):
                raise AssertionError("{} text prior non-finite/out-of-range".format(name))
    with (source / "review_status.csv").open("r", encoding="utf-8", newline="") as handle:
        review = list(csv.DictReader(handle))
    if len(review) != len(attrs) + len(objs):
        raise AssertionError("{} review table missing rows".format(name))
    pending = [row for row in review if row["status"] != "approved"]
    if require_reviewed and pending:
        raise AssertionError("{} has {} unapproved knowledge entries".format(name, len(pending)))
    if (source / "compatibility.csv").exists():
        raise AssertionError("{} unexpected compatibility.csv; inspect its provenance separately".format(name))
    print("{}: STRUCTURE OK; attrs={} objs={} parents={} pending_review={}".format(
        name, len(attrs), len(objs), len(oi), len(pending)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--knowledge-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()
    for name in args.datasets:
        check_dataset(name, args.data_root, args.knowledge_root, args.require_reviewed)
    print("ALL REQUESTED STRUCTURAL CHECKS PASSED")


if __name__ == "__main__":
    main()
