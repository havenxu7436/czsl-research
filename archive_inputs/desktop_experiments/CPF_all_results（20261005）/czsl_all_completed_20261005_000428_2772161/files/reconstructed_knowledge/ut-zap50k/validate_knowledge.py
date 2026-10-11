#!/usr/bin/env python3
"""Validate reconstructed LOGICZSL knowledge against a UT-Zappos split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_pairs(path: Path) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected two labels, got {fields!r}")
        result.append((fields[0], fields[1]))
    return result


def load_json(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def check_three_descriptions(
    name: str,
    mapping: dict[str, list[str]],
    expected_keys: list[str],
) -> None:
    if list(mapping) != expected_keys:
        missing = sorted(set(expected_keys) - set(mapping))
        extra = sorted(set(mapping) - set(expected_keys))
        raise AssertionError(
            f"{name} key/order mismatch; missing={missing}, extra={extra}"
        )
    for key, descriptions in mapping.items():
        if not isinstance(descriptions, list) or len(descriptions) != 3:
            raise AssertionError(f"{name}[{key!r}] must have exactly three texts")
        normalized = [text.strip().lower() for text in descriptions]
        if any(not text for text in normalized):
            raise AssertionError(f"{name}[{key!r}] contains an empty text")
        if len(set(normalized)) != 3:
            raise AssertionError(f"{name}[{key!r}] contains duplicate texts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    split_root = args.dataset_root / "compositional-split-natural"
    pair_paths = [
        split_root / "train_pairs.txt",
        split_root / "val_pairs.txt",
        split_root / "test_pairs.txt",
    ]
    pairs: list[tuple[str, str]] = []
    for path in pair_paths:
        pairs.extend(read_pairs(path))

    attributes = sorted({attribute for attribute, _ in pairs})
    objects = sorted({obj for _, obj in pairs})
    if len(attributes) != 16 or len(objects) != 12:
        raise AssertionError(
            f"unexpected UT-Zappos label counts: attrs={len(attributes)}, "
            f"objects={len(objects)}"
        )

    knowledge = args.knowledge_dir
    oi = load_json(knowledge / "oi_dict.json")
    oe = load_json(knowledge / "oe_dict.json")
    ae = load_json(knowledge / "ae_dict.json")

    oi_children = [child for children in oi.values() for child in children]
    counts = Counter(oi_children)
    missing = sorted(set(objects) - set(oi_children))
    extra = sorted(set(oi_children) - set(objects))
    duplicate = sorted(name for name, count in counts.items() if count != 1)
    if missing or extra or duplicate:
        raise AssertionError(
            f"oi_dict coverage failure: missing={missing}, extra={extra}, "
            f"duplicate={duplicate}"
        )

    check_three_descriptions("oe_dict", oe, objects)
    check_three_descriptions("ae_dict", ae, attributes)

    compatibility_path = knowledge / "compatibility.csv"
    if compatibility_path.is_file():
        with compatibility_path.open("r", encoding="utf-8", newline="") as handle:
            matrix = [row for row in csv.reader(handle) if row]
        if len(matrix) != len(attributes):
            raise AssertionError(
                f"compatibility rows={len(matrix)}, expected={len(attributes)}"
            )
        if any(len(row) != len(objects) for row in matrix):
            raise AssertionError(
                f"compatibility must be {len(attributes)}x{len(objects)}"
            )
        if any(cell not in {"0", "1"} for row in matrix for cell in row):
            raise AssertionError("compatibility cells must be 0 or 1")
        matrix_pairs = {
            (attributes[i], objects[j])
            for i, row in enumerate(matrix)
            for j, cell in enumerate(row)
            if cell == "1"
        }
        expected_pairs = set(pairs)
        if matrix_pairs != expected_pairs:
            raise AssertionError(
                "compatibility matrix does not equal the split pair union"
            )
    else:
        raise FileNotFoundError(
            f"{compatibility_path} is missing; run build_compatibility.py first"
        )

    print("UT-ZAPPOS RECONSTRUCTED KNOWLEDGE VALIDATION PASSED")
    print(f"attributes={len(attributes)}")
    print(f"objects={len(objects)}")
    print(f"oi_parents={len(oi)}")
    print(f"candidate_pairs={len(set(pairs))}")


if __name__ == "__main__":
    main()
