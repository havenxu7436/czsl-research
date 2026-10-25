from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .common import ROOT, read_json, sha


def validate_cache(cache):
    cache = Path(cache).resolve()
    expected = read_json(ROOT / 'assets/expected_cache.json')
    manifest = read_json(cache / 'features_manifest.json')
    if not manifest.get('complete') or manifest['labels'] != expected['labels'] or manifest['pairs'] != expected['pairs']:
        raise ValueError('Feature labels/splits differ from the G0 cache: ' + str(cache))
    # G0 uses g; the original TRUST residual retains the object-query context fo.
    files = ['text_features.npz'] + [f'{p}_records.json' for p in ('train', 'val', 'test')] + [f'{p}/{n}.npy' for p in ('train', 'val', 'test') for n in ('g','fo')]
    for name in files:
        if sha(cache / name) != expected['files_sha256'][name]:
            raise ValueError('G0 input content differs: ' + str(cache / name))
    return manifest


def train_mask(manifest):
    labels = manifest['labels']
    mask = torch.zeros(len(labels['objs']), len(labels['attrs']), dtype=torch.bool)
    for a, o in manifest['pairs']['train']:
        mask[labels['objs'].index(o), labels['attrs'].index(a)] = True
    return mask


class Globals(Dataset):
    def __init__(self, cache, phase):
        self.cache, self.phase = Path(cache), phase
        self.rows = read_json(self.cache / (phase + '_records.json'))
        self.array = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.array is None:
            self.array = np.load(self.cache / self.phase / 'g.npy', mmap_mode='r')
        r = self.rows[i]
        return torch.from_numpy(np.array(self.array[i], dtype=np.float32)), r['attr'], r['object']

    def __getstate__(self):
        return dict(self.__dict__, array=None)


def loader(cache, phase, cfg, seed, epoch=None):
    gen = torch.Generator().manual_seed(seed + (0 if epoch is None else 1000003 * epoch))
    return DataLoader(Globals(cache, phase), batch_size=cfg['batch_size'] if epoch is not None else cfg['eval_batch_size'],
                      shuffle=epoch is not None, generator=gen, num_workers=cfg['workers'], drop_last=False)
