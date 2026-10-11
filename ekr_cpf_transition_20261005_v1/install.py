"""Install into a NEW code checkout. Original KCC files are never edited."""
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_model(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DDM")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "val_forward")
    returns = [n for n in method.body if isinstance(n, ast.Return)]
    if len(returns) != 1:
        raise ValueError("val_forward return structure changed; inspect before generating")
    ret = returns[0]
    block = "".join(source.splitlines(True)[method.lineno - 1:ret.lineno - 1])
    for anchor in ("attr_emb = self.attrc_embedder", "pred = comp_pred + attr_pred + obj_pred",
                   "self.val_objs", "self.val_attrs", "scale=False"):
        if anchor not in block:
            raise ValueError("Missing audited feature/score anchor: " + anchor)
    block = block.replace("def val_forward(", "def raw_forward(", 1)
    result = ("from .kcc_cpf import *\nfrom .semantic_control import DDM as SemanticDDM\n\n"
              "class DDM(SemanticDDM):\n" + block +
              "        return {'pred': pred, 'attr_feature': attr_emb}\n")
    compile(result, "models/ekr_base.py", "exec")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    args = p.parse_args()
    bundle = Path(__file__).resolve().parent
    src, dst = args.source.resolve(), args.destination.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("New code directory must be separate from existing source")
    if dst.exists():
        raise FileExistsError("Destination already exists; choose a new version: " + str(dst))
    expected = json.loads((bundle / "audited_source_sha256.json").read_text())
    mismatches = [name for name, digest in expected.items()
                  if not (src / name).is_file() or sha(src / name) != digest]
    if mismatches:
        raise ValueError("Server source differs from the week5 snapshot: " + ", ".join(mismatches))
    native = (src / "train.py").read_text(encoding="utf-8-sig")
    marker = "from models.kcc_cpf import DDM"
    if native.count(marker) != 1:
        raise ValueError("Unexpected training model import")
    raw = raw_model((src / "models/kcc_cpf.py").read_text(encoding="utf-8-sig"))
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", "runs", "experiments", "data", "output", "outputs"))
    shutil.copytree(bundle / "transition", dst / "transition")
    shutil.copy2(bundle / "models/semantic_control.py", dst / "models/semantic_control.py")
    (dst / "models/ekr_base.py").write_text(raw, encoding="utf-8")
    (dst / "train_semantic.py").write_text(native.replace(
        marker, "from models.semantic_control import DDM", 1), encoding="utf-8")
    (dst / "tests").mkdir(exist_ok=True)
    shutil.copy2(bundle / "tests/test_router_cpu.py", dst / "tests/test_router_cpu.py")
    meta = {"source": str(src), "destination": str(dst), "source_files": expected,
            "generation": "original KCC val_forward prefix; only return exposes attr_feature"}
    (dst / "transition_install.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("INSTALLED", dst)
    print("Original source unchanged. Run bank construction and GPU smoke before training.")


if __name__ == "__main__":
    main()
