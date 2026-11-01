import contextlib
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    os.replace(str(temp), str(path))

def load(path):
    # Input checkpoints/metadata belong to this experiment or the user's datasets.
    try: return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError: return torch.load(path, map_location='cpu')

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp'); torch.save(value, temp)
    os.replace(str(temp), str(path))

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if s['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(s['cuda'])

def fingerprint():
    return {p.relative_to(ROOT).as_posix():sha(p) for folder in ('trustg0','configs','assets')
            for p in sorted((ROOT/folder).rglob('*')) if p.suffix in ('.py','.json')}

@contextlib.contextmanager
def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'a+b')
    if os.name == 'nt':
        import msvcrt
        f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try: yield
    finally: f.close()
