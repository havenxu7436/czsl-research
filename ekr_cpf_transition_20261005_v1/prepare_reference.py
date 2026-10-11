"""Local release preparation: read the supplied week5 snapshot, do not execute it."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--fixture", required=True, type=Path)
    args = p.parse_args()
    names = ["train.py", "dataset.py", "evaluator_ge.py", "models/ddm.py",
             "models/kcc_cpf.py", "models/kcc_cache.py", "models/word_embedding_utils.py",
             "tools/build_knowledge.py", "config/defaults.py"]
    result = {}
    with tarfile.open(args.snapshot, "r:gz") as tar:
        for name in names:
            value = tar.extractfile(name).read()
            result[name] = hashlib.sha256(value).hexdigest()
        for member in tar.getmembers():
            path = Path(member.name)
            if not member.isfile() or path.suffix not in {".py", ".yml"}:
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid snapshot member")
            target = args.fixture / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    (Path(__file__).resolve().parent / "audited_source_sha256.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("REFERENCE PREPARED", len(result), "audited files")


if __name__ == "__main__":
    main()
