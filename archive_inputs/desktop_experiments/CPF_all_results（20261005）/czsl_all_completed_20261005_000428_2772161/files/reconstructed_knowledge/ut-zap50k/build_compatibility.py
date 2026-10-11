#!/usr/bin/env python3
"""Build a deterministic UT-Zappos closed-world compatibility matrix.

Rows follow sorted attribute order. Columns follow sorted object order. A cell is
1 when that pair appears in any official train/val/test pair-list file, otherwise
0. This is a reconstructed candidate-pair mask, not an author-supplied artifact.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_pairs(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected two labels, got {fields!r}")
        pairs.add((fields[0], fields[1]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="UT-Zappos root containing compositional-split-natural",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("compatibility.csv"),
    )
    args = parser.parse_args()

    split_root = args.dataset_root / "compositional-split-natural"
    paths = [
        split_root / "train_pairs.txt",
        split_root / "val_pairs.txt",
        split_root / "test_pairs.txt",
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    pairs: set[tuple[str, str]] = set()
    for path in paths:
        pairs.update(read_pairs(path))

    attributes = sorted({attribute for attribute, _ in pairs})
    objects = sorted({obj for _, obj in pairs})
    matrix = [
        [int((attribute, obj) in pairs) for obj in objects]
        for attribute in attributes
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(matrix)

    print(f"attributes={len(attributes)}")
    print(f"objects={len(objects)}")
    print(f"candidate_pairs={len(pairs)}")
    print(f"matrix_shape={len(matrix)}x{len(matrix[0])}")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
