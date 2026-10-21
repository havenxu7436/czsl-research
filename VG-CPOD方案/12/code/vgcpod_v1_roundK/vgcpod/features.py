"""Frozen official CLIP ViT-L/14, projected last-layer patches, resumable shards."""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import ROOT, digest, read_json, seed_all, sha, source_hashes, unit, write_json
from .data import ImageDataset, image_path, inspect_dataset

CLIP_SHA = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
CLIP_COMMIT = 'a1d071733d7111c9c014f024669f959182114e33'


def visual_tokens(visual, images):
    """Return CLS and patches after ln_post and projection, in FP32 unit vectors."""
    x = visual.conv1(images.to(visual.conv1.weight.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = torch.cat([cls, x], dim=1)
    x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    x = unit(x.float())
    return x[:, 0], x[:, 1:]


def query_patches(patches, queries, tau):
    alpha = torch.einsum('qd,bld->bql', queries, patches).div(tau).softmax(-1)
    return unit(torch.einsum('bql,bld->bqd', alpha, patches))


def prompts():
    names = read_json(ROOT / 'assets' / 'names.json')
    return {'a': ['a photo of footwear made of {}'.format(a) for a in names['a']],
            'o': ['a photo of {}'.format(o) for o in names['o']],
            'c': ['a photo of {} made of {}'.format(o, a) for o in names['o'] for a in names['a']]}


def preflight(args):
    labels, pairs, records, inputs = inspect_dataset(args.data_root)
    result = {'dataset': 'ut-zap50k', 'labels': labels, 'inputs_sha256': inputs,
              'counts': {s: {'images': len(r), 'pairs': len(pairs[s])} for s, r in records.items()}}
    missing = []
    for phase, rows in records.items():
        for row in rows:
            try:
                image_path(args.data_root, row)
            except FileNotFoundError as e:
                missing.append(str(e))
    result['missing_images'] = len(missing)
    if missing:
        raise FileNotFoundError('{} missing images; first: {}'.format(len(missing), missing[0]))
    if getattr(args, 'checkpoint', None):
        if sha(args.checkpoint) != CLIP_SHA:
            raise ValueError('CLIP ViT-L/14 checkpoint SHA256 mismatch')
        result['clip_checkpoint_sha256'] = CLIP_SHA
    print(result, flush=True)
    return result


def prepare(args):
    seed_all(3351)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    labels, pairs, records, inputs = inspect_dataset(args.data_root)
    if sha(args.checkpoint) != CLIP_SHA:
        raise ValueError('Expected frozen ViT-L/14 checkpoint: ' + CLIP_SHA)
    sys.path.insert(0, str(ROOT / 'vendor'))
    import clip
    from PIL import __version__ as pillow_version
    identity = {'schema': 'vgcpod_features_v1', 'dataset': 'ut-zap50k', 'labels': labels,
                'pairs': pairs, 'split_sha256': inputs, 'prompts': prompts(),
                'checkpoint_sha256': CLIP_SHA, 'clip_commit': CLIP_COMMIT,
                'tau_v': args.tau_v, 'cache_dtype': 'float16', 'encoding_dtype': 'float32',
                'encoding_batch_size': args.batch_size, 'shard_size': args.shard_size,
                'clip_source': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / 'vendor' / 'clip').glob('*.py'))},
                'clip_bpe_sha256': sha(ROOT / 'vendor/clip/bpe_simple_vocab_16e6.txt.gz'),
                'feature_implementation_sha256': {p.name: sha(p) for p in
                    [Path(__file__), ROOT / 'vgcpod/data.py', ROOT / 'vgcpod/common.py']},
                'pillow': pillow_version, 'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
                'gpu': torch.cuda.get_device_name() if str(args.device).startswith('cuda') else None,
                'image_resolution': 224, 'preprocessing': 'official CLIP Resize/CenterCrop/RGB/Normalize',
                'torch': torch.__version__, 'device_type': str(torch.device(args.device).type)}
    import torchvision
    identity['torchvision'] = torchvision.__version__
    identity_path = cache / 'encoding_protocol.json'
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError('Existing cache belongs to a different encoding protocol')
    write_json(identity_path, identity)
    model, preprocess = clip.load(str(args.checkpoint), device=args.device, jit=False)
    model = model.float().eval()
    model.requires_grad_(False)
    if model.visual.input_resolution != 224 or model.visual.proj.shape != (1024, 768):
        raise ValueError('CLIP backbone must be ViT-L/14 at 224px')
    with torch.no_grad():
        text = {}
        for key, rows in identity['prompts'].items():
            chunks = []
            for start in range(0, len(rows), 64):
                tokens = clip.tokenize(rows[start:start + 64], truncate=False).to(args.device)
                chunks.append(unit(model.encode_text(tokens).float()).cpu().numpy())
            text[{'a': 'ea', 'o': 'eo', 'c': 'ec'}[key]] = np.concatenate(chunks)
        text['ec'] = text['ec'].reshape(len(labels['objs']), len(labels['attrs']), 768)
        np.savez(cache / 'text_features.npz', **text)
        ea = torch.from_numpy(text['ea']).to(args.device)
        eo = torch.from_numpy(text['eo']).to(args.device)
        queries = torch.cat([ea, eo], dim=0)
        for phase in ('train', 'val', 'test'):
            rows = records[phase]
            write_json(cache / (phase + '_records.json'), rows)
            folder = cache / phase
            shards = folder / 'shards'
            shards.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(rows), args.shard_size):
                end = min(start + args.shard_size, len(rows))
                target = shards / ('%06d.npz' % start)
                receipt = target.with_suffix('.json')
                row_hash = digest(rows[start:end])
                if target.exists() and receipt.exists():
                    old = read_json(receipt)
                    if old['rows_sha256'] != row_hash or old['sha256'] != sha(target):
                        raise ValueError('Cached shard changed: ' + str(target))
                    print('CACHE_REUSE', phase, start, end, flush=True)
                    continue
                loader = DataLoader(ImageDataset(rows[start:end], args.data_root, preprocess),
                                    batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                    pin_memory=str(args.device).startswith('cuda'))
                global_features, attribute_features, object_features, images_sha = [], [], [], []
                for images, fingerprints in loader:
                    images = images.to(args.device)
                    g, p = visual_tokens(model.visual, images)
                    # One exact CLS comparison per shard catches a changed extraction interface.
                    if not global_features:
                        ref = unit(model.encode_image(images).float())
                        if not torch.allclose(g, ref, atol=2e-5, rtol=2e-5):
                            raise ValueError('Extracted CLS differs from official encode_image')
                    f = query_patches(p, queries, args.tau_v)
                    global_features.append(g.cpu().numpy().astype(np.float16))
                    attribute_features.append(f[:, :len(ea)].cpu().numpy().astype(np.float16))
                    object_features.append(f[:, len(ea):].cpu().numpy().astype(np.float16))
                    images_sha.extend(fingerprints)
                tmp = target.with_suffix('.writing.npz')
                np.savez(tmp, g=np.concatenate(global_features), fa=np.concatenate(attribute_features),
                         fo=np.concatenate(object_features))
                os.replace(str(tmp), str(target))
                write_json(receipt, {'rows_sha256': row_hash, 'sha256': sha(target), 'image_sha256': images_sha})
                print('ENCODED', phase, end, '/', len(rows), flush=True)
            # Rebuild final memory maps from committed shards; interrupted preparation reuses shards.
            for key, shape in [('g', (len(rows), 768)), ('fa', (len(rows), len(ea), 768)),
                               ('fo', (len(rows), len(eo), 768))]:
                target = folder / (key + '.npy')
                temp = target.with_name(key + '.writing.npy')
                array = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float16, shape=shape)
                for start in range(0, len(rows), args.shard_size):
                    with np.load(shards / ('%06d.npz' % start), allow_pickle=False) as z:
                        array[start:start + len(z[key])] = z[key]
                array.flush()
                del array
                os.replace(str(temp), str(target))
    files = [identity_path, cache / 'text_features.npz']
    for phase in ('train', 'val', 'test'):
        files += [cache / (phase + '_records.json')]
        files += list((cache / phase).glob('*.npy'))
        files += list((cache / phase / 'shards').glob('*.json'))
    identity.update(complete=True, counts={s: len(rows) for s, rows in records.items()},
                    files_sha256={p.relative_to(cache).as_posix(): sha(p) for p in sorted(files)})
    write_json(cache / 'features_manifest.json', identity)
    print('FEATURE_CACHE_COMPLETE', cache, flush=True)
