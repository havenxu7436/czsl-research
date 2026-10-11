"""Validate existing A/C weights, then add relative path aliases; never overwrite."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

DATASETS = {"ut-zappos": 3351, "mit-states": 0, "cgqa": 3351}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plan(legacy, datasets):
    result = []
    for dataset in datasets:
        seed = DATASETS[dataset]
        for variant in ("A", "C"):
            rid = "mvp_{}_{}_legacy_s{}_l0".format(dataset, variant, seed)
            run = Path(legacy) / "runs" / rid
            expected = None
            for setting in ("cw", "ow"):
                m = json.loads((run / ("eval_val_" + setting) / "metrics.json").read_text())
                if (m["dataset"], m["variant"], int(m["seed"]), m["phase"], m["setting"]) != (
                        dataset, variant, seed, "val", setting):
                    raise ValueError("Wrong evaluation identity: " + rid)
                digest = m["checkpoint_sha256"]
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Invalid checkpoint digest")
                if expected is not None and expected != digest:
                    raise ValueError("CW/OW checkpoint differs: " + rid)
                expected = digest
            alias = run / "checkpoints/best_auc.pth"
            if alias.exists() or alias.is_symlink():
                if not alias.is_file() or sha(alias) != expected:
                    raise ValueError("Existing alias/file differs; refusing overwrite: " + str(alias))
                print("EXISTING_OK", rid, flush=True)
                continue
            log = (run / "train.log").read_text(encoding="utf-8", errors="replace")
            paths = re.findall(r"Loading validation-best checkpoint from epoch \d+: ([^\r\n]+)", log)
            if not paths:
                raise ValueError("No validation-best checkpoint path in log: " + rid)
            parts = PurePosixPath(paths[-1].strip()).parts
            index = parts.index("checkpoints")
            source = run / "checkpoints" / Path(*parts[index + 1:])
            source.resolve().relative_to((run / "checkpoints").resolve())
            if not source.is_file():
                raise FileNotFoundError("Original logged checkpoint is absent: " + str(source))
            print("CHECKING", source, flush=True)
            if sha(source) != expected:
                raise ValueError("Original checkpoint hash differs: " + str(source))
            result.append((alias, source, expected))
            print("VERIFIED", rid, expected, flush=True)
    return result


def apply(items):
    # All requested weights were checked before any alias is created.
    for alias, source, expected in items:
        if alias.exists() or alias.is_symlink():
            raise FileExistsError("Destination appeared during checking: " + str(alias))
        alias.symlink_to(os.path.relpath(source, alias.parent))
        print("CHECKPOINT_LINKED", alias, "->", source, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-exp", required=True, type=Path)
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    items = plan(args.legacy_exp, args.datasets)
    if args.apply:
        apply(items)
        print("LEGACY CHECKPOINT PATHS READY", flush=True)
    else:
        print("CHECK ONLY PASSED; use --apply to create", len(items), "relative aliases", flush=True)


if __name__ == "__main__":
    main()
