#!/usr/bin/env python3
"""Generate KCC overrides from actual server ddm.py; preserve the original."""
import argparse
import ast
import difflib
import hashlib
import json
import textwrap
from datetime import datetime
from pathlib import Path


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def method_text(source, node):
    lines = source.splitlines(True)
    end = getattr(node, "end_lineno", None)
    if end is None:
        raise RuntimeError("Python 3.8+ is required")
    return "".join(lines[node.lineno - 1:end])


def generate_model(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DDM")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    overrides = []
    for name in ("compose_word_embeddings", "train_forward", "train_forward_augment", "val_forward"):
        node = methods[name]
        block = method_text(source, node)
        if "_kcc_" in block:
            raise ValueError("original ddm.py already contains KCC edits; inspect before installing")
        if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "obj_embedder" for n in ast.walk(node)):
            raise ValueError("missing object embedding anchor in " + name)
        block = block.replace("self.obj_embedder(", "self._kcc_obj_embeddings(")
        block = block.replace("self.attr_embedder(", "self._kcc_attr_embeddings(")
        if name in ("train_forward", "train_forward_augment"):
            anchors = [n for n in node.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "comp_pred" for t in n.targets)
                       and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                       and n.value.func.attr == "classifier"]
            if len(anchors) != 1:
                raise ValueError("ambiguous composition-logit anchor in " + name)
            anchor = method_text(source, anchors[0])
            block = block.replace(anchor, anchor + "        comp_pred = self._kcc_calibrate(comp_pred, self.train_objs, self.train_attrs)\n", 1)
        if name == "val_forward":
            # Candidate order is taken from the actual evaluation dataset.
            block = block.replace("self.pairs[:, 1]", "self.val_objs")
            block = block.replace("self.pairs[:, 0]", "self.val_attrs")
            marker = "        pred = comp_pred + attr_pred + obj_pred"
            if block.count(marker) != 1:
                raise ValueError("validation score anchor changed")
            prefix = block[:block.index(marker)]
            block = prefix + textwrap.indent(textwrap.dedent('''\
                pred = comp_pred + attr_pred + obj_pred
                pred = self._kcc_calibrate(pred, self.val_objs, self.val_attrs)
                if pred.size(1) != len(self.val_pairs):
                    raise ValueError("evaluation candidate count mismatch")
                return {'pred': pred, 'scores': {
                    pair: pred[:, index] for index, pair in enumerate(self.val_pairs)
                }}
            '''), "        ")
        overrides.append(block.rstrip() + "\n")
    header = '''# Generated from this project's original models/ddm.py.
# A/C/D/E GloVe MVP. Original baseline layers/loss weights are inherited.
from .ddm import *
from .ddm import DDM as BaselineDDM
from .kcc_cache import KCCCacheMixin


class DDM(KCCCacheMixin, BaselineDDM):
    def __init__(self, dset, cfg):
        super().__init__(dset, cfg)
        self._setup_kcc(dset, cfg)

'''
    result = header + "\n".join(overrides)
    compile(result, "models/kcc_cpf.py", "exec")
    return result


def patch_import(source):
    if "from models.kcc_cpf import DDM" in source:
        return source
    marker = "from models.ddm import DDM"
    if source.count(marker) != 1:
        raise ValueError("train/test DDM import differs from the audited baseline")
    return source.replace(marker, "from models.kcc_cpf import DDM", 1)


def guard_final_test(source):
    if "kcc_test_at_end" in source:
        return source
    start_marker = '    print("Test set with validation-best checkpoint ===>")'
    end_marker = "    logger.close()"
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError("train.py final-test markers changed; no files were installed")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    block = source[start:end]
    if "test_auc, test_hm" not in block or "test/auc_at_best_val" not in block:
        raise ValueError("unexpected final-test block")
    wrapped = '    if bool(getattr(cfg.TRAIN, "kcc_test_at_end", False)):\n'
    wrapped += "".join("    " + line if line.strip() else line for line in block.splitlines(True))
    return source[:start] + wrapped + source[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    bundle = Path(__file__).resolve().parent
    base_path = project / "models" / "ddm.py"
    base_source = base_path.read_text(encoding="utf-8-sig")
    destinations = {project / "models" / "kcc_cpf.py": generate_model(base_source),
                    project / "models" / "kcc_cache.py": (bundle / "kcc_cache.py").read_text(encoding="utf-8"),
                    project / "train.py": guard_final_test(patch_import((project / "train.py").read_text(encoding="utf-8-sig"))),
                    project / "test.py": patch_import((project / "test.py").read_text(encoding="utf-8-sig"))}
    for src, dest in (("smoke_mvp.py", "tests/smoke_mvp.py"),
                      ("make_mvp_configs.py", "tools/make_mvp_configs.py"),
                      ("eval_phase.py", "tools/eval_mvp.py"),
                      ("run_mvp.sh", "tools/run_mvp.sh")):
        destinations[project / dest] = (bundle / src).read_text(encoding="utf-8")
    for path, content in destinations.items():
        if path.suffix == ".py":
            compile(content, str(path), "exec")
    changed = {p: s for p, s in destinations.items()
               if not p.exists() or p.read_text(encoding="utf-8-sig") != s}
    if not changed:
        print("Already installed; no files changed.")
        return
    backup = project / ("kcc_mvp_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    backup.mkdir()
    diffs = []
    for path, content in changed.items():
        relative = path.relative_to(project)
        old = path.read_text(encoding="utf-8-sig") if path.exists() else ""
        if path.exists():
            backup_path = backup / relative
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_bytes(path.read_bytes())
        diffs.extend(difflib.unified_diff(old.splitlines(True), content.splitlines(True),
                                         fromfile="before/" + str(relative), tofile="after/" + str(relative)))
    (backup / "changes.diff").write_text("".join(diffs), encoding="utf-8")
    audit = {"baseline_ddm_sha256_lf": sha(base_source),
             "created_files": [str(p.relative_to(project)) for p in changed if not p.exists()],
             "installed_sha256": {str(p.relative_to(project)): sha(s) for p, s in destinations.items()}}
    (backup / "installation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for path, content in changed.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    if base_path.read_text(encoding="utf-8-sig") != base_source:
        raise AssertionError("original ddm.py was modified")
    print("KCC MVP INSTALLED: " + str(project))
    print("BACKUP AND REVIEW DIFF: " + str(backup))
    print("Supported: A/C/D/E; unsupported B/F/G/H fail explicitly.")


if __name__ == "__main__":
    main()
