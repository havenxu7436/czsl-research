from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import ROOT, digest, load_torch, read_json, sha


def read_pairs(path):
    rows = [line.split() for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if any(len(row) != 2 for row in rows) or len({tuple(r) for r in rows}) != len(rows):
        raise ValueError('Invalid pair list: ' + str(path))
    return rows


def inspect_dataset(data_root):
    root = Path(data_root)
    split = root / 'compositional-split-natural'
    labels = read_json(ROOT / 'assets' / 'labels.json')
    pairs = {s: read_pairs(split / (s + '_pairs.txt')) for s in ('train', 'val', 'test')}
    for phase, rows in pairs.items():
        if any(a not in labels['attrs'] or o not in labels['objs'] for a, o in rows):
            raise ValueError('Unknown primitive in ' + phase)
    metadata_file = root / 'metadata_compositional-split-natural.t7'
    expected = read_json(ROOT / 'assets' / 'historical_inputs.json')
    # Exact archived TRAIN/VAL split and metadata ensure the first experiment matches history.
    for name, fingerprint in expected.items():
        if sha(root / name) != fingerprint:
            raise ValueError('Archived UT input hash differs: ' + name)
    metadata = load_torch(metadata_file)
    records = {s: [] for s in pairs}
    sets = {s: {tuple(p) for p in rows} for s, rows in pairs.items()}
    all_pairs = set.union(*sets.values())
    identities = set()
    for row in metadata:
        pair, phase = (row['attr'], row['obj']), row['set']
        if row['attr'] == 'NA' or phase == 'NA' or pair not in all_pairs:
            continue
        if phase not in records or pair not in sets[phase]:
            raise ValueError('Metadata pair/split mismatch: ' + str(row))
        identity = str(row['image'])
        if identity in identities:
            raise ValueError('Duplicate image identity across splits: ' + identity)
        identities.add(identity)
        records[phase].append({'image': identity, 'raw_image': row.get('_image', identity),
                               'attr': labels['attrs'].index(pair[0]),
                               'object': labels['objs'].index(pair[1])})
    if len(pairs['train']) != 83 or len(records['train']) != 22998 or len(records['val']) != 3214:
        raise ValueError('Unexpected UT natural split counts')
    seen = sets['train']
    val_seen = sum((labels['attrs'][r['attr']], labels['objs'][r['object']]) in seen for r in records['val'])
    if val_seen != 877:
        raise ValueError('VAL seen/unseen counts differ')
    for phase, rows in records.items():
        covered = {(labels['attrs'][r['attr']], labels['objs'][r['object']]) for r in rows}
        if covered != sets[phase]:
            raise ValueError('Incomplete pair coverage: ' + phase)
    return labels, pairs, records, {p.relative_to(root).as_posix(): sha(p) for p in
                                  [metadata_file] + [split / (s + '_pairs.txt') for s in pairs]}


def image_path(data_root, row):
    root = Path(data_root).resolve()
    aliases = read_json(ROOT / 'assets' / 'path_aliases.json')
    relative = str(row['image']).replace('\\', '/')
    raw = str(row['raw_image']).replace('\\', '/')
    for name in (relative, raw):
        p = PurePosixPath(name)
        if not name or p.is_absolute() or '..' in p.parts or ':' in name:
            raise ValueError('Invalid image path: ' + name)
    if PurePosixPath(relative).name != PurePosixPath(raw).name:
        raise ValueError('Image alias changes filename')
    paths = [root / 'images' / relative,
             root / '_images_extract' / 'ut-zap50k-images-square' / aliases.get(raw, raw)]
    for p in paths:
        resolved = p.resolve()
        if root in resolved.parents and resolved.is_file():
            return resolved
    raise FileNotFoundError('Image absent: ' + relative)


class ImageDataset(Dataset):
    def __init__(self, rows, data_root, preprocess):
        self.rows, self.root, self.preprocess = rows, data_root, preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        p = image_path(self.root, self.rows[index])
        fingerprint = sha(p)
        with Image.open(p) as im:
            tensor = self.preprocess(im.convert('RGB'))
        return tensor, fingerprint


class FeatureDataset(Dataset):
    def __init__(self, cache, phase):
        self.root = Path(cache)
        self.phase = phase
        self.rows = read_json(self.root / (phase + '_records.json'))
        self.arrays = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.arrays is None:
            self.arrays=[np.load(self.root/self.phase/(name+'.npy'),mmap_mode='r') for name in ('g','fo')]
        g,fo=[torch.from_numpy(np.array(x[index],dtype=np.float32)) for x in self.arrays]
        row=self.rows[index]
        return g,g,fo,row['attr'],row['object']

    def __getstate__(self):
        state = dict(self.__dict__)
        state['arrays'] = None
        return state


def mask_from_pairs(labels, pairs):
    mask = torch.zeros(len(labels['objs']), len(labels['attrs']), dtype=torch.bool)
    for a, o in pairs:
        mask[labels['objs'].index(o), labels['attrs'].index(a)] = True
    return mask


def load_cache(cache, verify=False):
    cache = Path(cache)
    manifest = read_json(cache / 'features_manifest.json')
    if not manifest.get('complete'):
        raise ValueError('Feature cache is incomplete')
    if verify:
        for relative, fingerprint in manifest['files_sha256'].items():
            if sha(cache / relative) != fingerprint:
                raise ValueError('Cache hash mismatch: ' + relative)
    with np.load(cache / 'text_features.npz', allow_pickle=False) as z:
        text = {k: torch.from_numpy(z[k].copy()).float() for k in ('ea', 'eo', 'ec')}
    return manifest, text
