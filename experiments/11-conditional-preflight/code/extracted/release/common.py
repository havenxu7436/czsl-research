"""Read-only native model integration and atomic, isolated output helpers."""
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(temp), str(path))


def verify_files(files):
    for name, expected in sorted(files.items()):
        if sha(name) != expected:
            raise ValueError('SOURCE/INPUT HASH MISMATCH: ' + name)
    return len(files)


def verify_release():
    manifest = read_json(ROOT / 'release_manifest.json')
    for name, expected in manifest.items():
        if Path(name).name != name or sha(ROOT / name) != expected:
            raise ValueError('RELEASE HASH MISMATCH: ' + name)
    return sha(ROOT / 'release_manifest.json')


def zip_destination(path, reference, extra_protected=()):
    path = Path(path).resolve()
    sidecar = path.with_name(path.name + '.sha256')
    if path.exists() or sidecar.exists():
        raise FileExistsError('Preserve existing ZIP/sidecar')
    protected = [ROOT.resolve(), Path('/data').resolve(), Path(reference['c_source']).parent.resolve(),
                 Path(reference['bf_source']).parent.resolve()] + [Path(p).resolve() for p in extra_protected]
    if any(path == p or p in path.parents for p in protected):
        raise ValueError('ZIP destination inside a protected source/input root')
    return path


def new_output(path, reference, free_mib=1024):
    path = Path(path).resolve()
    # No writes inside original experiments, datasets, or the release directory.
    protected = [ROOT.resolve(), Path('/data').resolve(),
                 Path(reference['c_source']).parent.resolve(),
                 Path(reference['bf_source']).parent.resolve()]
    if any(path == p or p in path.parents for p in protected):
        raise ValueError('Output inside a protected source/data/release root')
    if path.exists():
        raise FileExistsError('Preserve existing output; choose a fresh directory: ' + str(path))
    existing = path.parent
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < free_mib * 1024 ** 2:
        raise RuntimeError('Insufficient disk space; nothing will be removed')
    path.mkdir(parents=True)
    return path


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def native_context(source):
    sys.path.insert(0, source)
    os.chdir(source)


def load_cfg(path):
    from config import cfg as original
    cfg = original.clone()
    cfg.merge_from_file(path)
    return cfg


def load_native(ds, cfg, checkpoint, conditional=True):
    import torch
    from unittest.mock import patch
    from models import ddm
    from models.backbone import ViT
    if conditional:
        from models.ekr_base import DDM
    else:
        from models.kcc_cpf import DDM
    def no_download(*args, **kwargs):
        kwargs['pretrained'] = False
        return ViT(*args, **kwargs)
    # Full checkpoint strictly supplies every persistent model tensor; no download
    # or alteration of original source/pretrained-cache files is necessary.
    with patch.object(ddm, 'ViT', no_download):
        model = DDM(ds, cfg).cuda()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.eval()


def deterministic_items(ds):
    """One image per item; native TRAIN __getitem__ otherwise opens a second image."""
    import torch
    from dataset import imagenet_transform
    class Items(torch.utils.data.Dataset):
        def __init__(self):
            self.transform = imagenet_transform('val')
        def __len__(self):
            return len(ds.data)
        def __getitem__(self, index):
            image, attr, obj = ds.data[index]
            return dict(img=self.transform(ds.loader(image)), attr=ds.attr2idx[attr],
                        obj=ds.obj2idx[obj], index=index,
                        pair=(ds.train_pair2idx if ds.phase == 'train' else ds.pair2idx)[(attr, obj)])
    return Items()


def finite(value, name):
    import torch
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite tensor: ' + name)


def tensor_save(path, value):
    import torch
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    torch.save(value, temp)
    os.replace(str(temp), str(path))
