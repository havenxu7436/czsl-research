import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

DATASETS = {"ut-zappos": ("ut-zap50k", 3351),
            "mit-states": ("mit-states", 0), "cgqa": ("cgqa", 3351)}
SEMANTIC_MODES = {"A_raw": "raw", "A_unit": "unit",
                  "C_text": "correct", "C_shuffle": "shuffle"}
BRANCH_MODES = {"C_budget": "off", "R0": "adapter", "R1": "static", "R2": "no_object",
                "R3": "attention", "R4": "no_null", "R5": "ekr",
                "R7": "shuffle", "Rname": "name_residual",
                "Rvalue": "raw_values", "R6": "ekr",
                "R0noise": "adapter", "R3noise": "attention"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def empty_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Output is not empty; use a new run/version: " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_bank(path, expected_attrs=None, expected_objs=None):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    for name, digest in manifest["files_sha256"].items():
        if sha(path / name) != digest:
            raise ValueError("Description bank checksum mismatch: " + name)
    labels = read_json(path / "labels.json")
    if expected_attrs is not None and labels["attrs"] != list(expected_attrs):
        raise ValueError("Bank attribute order differs from dataset")
    if expected_objs is not None and labels["objs"] != list(expected_objs):
        raise ValueError("Bank object order differs from dataset")
    return torch.load(path / "bank.pt", map_location="cpu"), manifest


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def candidates(model, ds, train=False):
    pairs = ds.train_pairs if train else ds.pairs
    base = getattr(model, "base", model)
    device = next(base.parameters()).device
    base.val_attrs = torch.tensor([ds.attr2idx[a] for a, o in pairs], device=device)
    base.val_objs = torch.tensor([ds.obj2idx[o] for a, o in pairs], device=device)
    base.val_pairs = list(pairs)
    return base.val_attrs, base.val_objs, list(pairs)


def metrics_pct(record):
    m = record["metrics_raw"]
    return {"auc_pct": 100 * m["AUC"], "hm_pct": 100 * m["best_hm"],
            "best_seen_pct": 100 * m["best_seen"],
            "best_unseen_pct": 100 * m["best_unseen"],
            "hm_seen_pct": 100 * m.get("hm_seen", float("nan")),
            "hm_unseen_pct": 100 * m.get("hm_unseen", float("nan"))}


def code_manifest(project):
    project = Path(project)
    names = ["dataset.py", "evaluator_ge.py", "utils.py", "train.py",
             "train_semantic.py", "tools/build_knowledge.py"]
    names += [str(p.relative_to(project)).replace(os.sep, "/")
              for folder in ("models", "config", "utils", "transition")
              for p in sorted((project / folder).glob("*.py"))]
    return {name: sha(project / name) for name in names if (project / name).is_file()}
