"""Frozen CLIP evidence probe; zero optimizer steps, no TEST images/scores."""
import argparse
import csv
import os
import platform
import sys
import time
import traceback
from pathlib import Path, PurePosixPath

from common import ROOT, read_json, write_json, sha, require, verify_release, trusted_torch_load
from prompts import VARIANTS, make_prompts, check_descriptions


def safe_image(data, item, aliases):
    relative = str(item['image']).replace('\\', '/')
    raw = str(item.get('_image', '')).replace('\\', '/')
    for name in (relative, raw):
        p = PurePosixPath(name)
        require(name and not p.is_absolute() and '..' not in p.parts and ':' not in name,
                'Unsafe metadata image path')
    require(PurePosixPath(relative).name == PurePosixPath(raw).name, 'Raw image identity mismatch')
    candidates = [(data / 'images' / relative, 'cpf_images'),
                  (data / '_images_extract' / 'ut-zap50k-images-square' / aliases.get(raw, raw),
                   'raw_alias' if raw in aliases else 'raw_metadata')]
    for p, method in candidates:
        resolved = p.resolve()
        require(data.resolve() in resolved.parents, 'Image escapes dataset root: ' + str(p))
        if resolved.is_file():
            return resolved, method
    raise FileNotFoundError('No exact image path; no filename search/substitution: ' + relative)


def build_records(metadata, ids, labels, train_pairs, val_pairs, data, aliases):
    index = {}
    for item in metadata:
        # Ignore TEST records altogether. Do not extract its truth or image paths.
        if item.get('set') not in ('train', 'val'):
            continue
        identity = item['image']
        require(identity not in index, 'Duplicate TRAIN/VAL metadata identity')
        index[identity] = item
    result = {}
    require(not set(ids['train']) & set(ids['val']), 'TRAIN/VAL image overlap')
    for phase, allowed in (('train', train_pairs), ('val', train_pairs | val_pairs)):
        rows = []
        require(len(ids[phase]) == len(set(ids[phase])), 'Duplicate cached identity')
        selected = {i['image'] for i in index.values() if i['set'] == phase and
                    (i['attr'], i['obj']) in allowed}
        require(selected == set(ids[phase]), 'Metadata selection differs from StageA ' + phase)
        for n, identity in enumerate(ids[phase]):
            item = index[identity]
            require(item['set'] == phase, 'Image split changed')
            pair = (item['attr'], item['obj'])
            require(pair in allowed, 'Unexpected composition')
            p, method = safe_image(data, item, aliases)
            rows.append({'image': identity, 'attr': labels['attrs'].index(pair[0]),
                         'object': labels['objs'].index(pair[1]), 'seen': pair in train_pairs,
                         'source': str(p), 'source_method': method, 'image_sha256': sha(p)})
            if (n + 1) % 2000 == 0:
                print('IMAGE_PREFLIGHT', phase, n + 1, '/', len(ids[phase]), flush=True)
        result[phase] = rows
    return result


def encode_texts(clip, model, prompts, device, batch_size):
    import numpy as np
    import torch
    all_unique = sorted(set(t for rows in prompts.values() for t in rows))
    # Fail on overflow. Never silently truncate knowledge or one control.
    tokens = clip.tokenize(all_unique, truncate=False)
    features = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            x = model.encode_text(tokens[start:start + batch_size].to(device)).float()
            x = x / x.norm(dim=-1, keepdim=True)
            require(torch.isfinite(x).all().item(), 'Nonfinite text embeddings')
            features.append(x.cpu().numpy())
            done = min(start + batch_size, len(tokens))
            if done % (batch_size * 8) == 0 or done == len(tokens):
                print('CLIP_TEXTS', done, '/', len(tokens), flush=True)
    vectors = np.concatenate(features)
    lookup = {t: i for i, t in enumerate(all_unique)}
    tables = {}
    for variant, rows in prompts.items():
        unit = vectors[[lookup[t] for t in rows]].reshape(12, 16, 9, -1)
        # Average cosine, NOT renormalized average vector; exact equal sentence weights.
        tables[variant] = unit.mean(axis=2, dtype=np.float32)
    return tables, {'unique_texts': len(all_unique), 'tokens_max': int((tokens != 0).sum(1).max()),
                    'units_per_candidate': 9, 'silent_truncation': False}


def image_features(model, preprocess, rows, work, phase, device, batch_size, workers):
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    class Images(Dataset):
        def __init__(self, selected):
            self.rows = selected

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            # Bytes checked again at decode time, guarding changes since preflight.
            require(sha(row['source']) == row['image_sha256'], 'Image changed during encoding')
            with Image.open(row['source']) as im:
                return preprocess(im)

    directory = work / 'features' / phase
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    started = time.monotonic()
    encoded = 0
    shard_size = 512
    for start in range(0, len(rows), shard_size):
        end = min(start + shard_size, len(rows))
        path = directory / ('%06d.npz' % start)
        receipt = path.with_suffix('.json')
        identity_hash = __import__('hashlib').sha256(
            '\n'.join(r['image'] + ' ' + r['image_sha256'] for r in rows[start:end]).encode()).hexdigest()
        if receipt.exists():
            saved = read_json(receipt)
            require(saved['identity_sha256'] == identity_hash and sha(path) == saved['sha256'],
                    'Completed feature shard changed')
        else:
            recorded_batch = batch_size
            if path.exists():
                # Crash after atomic shard commit but before receipt commit:
                # its embedded identity lets us safely finish that transaction.
                with np.load(path, allow_pickle=False) as z:
                    require(str(z['identity_sha256'].item()) == identity_hash, 'Uncommitted shard identity mismatch')
                    require(z['features'].shape == (end - start, 768) and np.isfinite(z['features']).all(), 'Invalid uncommitted shard')
                    recorded_batch = int(z['batch_size'].item())
            else:
                loader = DataLoader(Images(rows[start:end]), batch_size=batch_size,
                                    num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)
                vectors = []
                with torch.no_grad():
                    for images in loader:
                        x = model.encode_image(images.to(device, non_blocking=True)).float()
                        x /= x.norm(dim=-1, keepdim=True)
                        require(torch.isfinite(x).all().item(), 'Nonfinite image embeddings')
                        vectors.append(x.cpu().numpy())
                x = np.concatenate(vectors)
                require(x.shape == (end - start, 768), 'Wrong ViT-L/14 feature shape')
                temporary = directory / (path.name + '.writing')
                with open(temporary, 'wb') as f:
                    np.savez_compressed(f, features=x, identity_sha256=np.array(identity_hash),
                                        batch_size=np.array(batch_size))
                os.replace(str(temporary), str(path))
                encoded += end - start
            write_json(receipt, {'sha256': sha(path), 'identity_sha256': identity_hash,
                                 'start': start, 'end': end, 'batch_size': recorded_batch})
        with np.load(path, allow_pickle=False) as z:
            value = z['features']
        require(value.shape == (end - start, 768) and np.isfinite(value).all(), 'Invalid feature shard')
        require(np.allclose(np.linalg.norm(value, axis=1), 1, atol=2e-3), 'Feature normalization changed')
        outputs.append(value)
        elapsed = time.monotonic() - started
        eta = (len(rows) - end) * elapsed / encoded if encoded else None
        print('CLIP_IMAGES', phase, end, '/', len(rows), 'new_images', encoded,
              'seconds', round(elapsed, 1), 'eta_seconds', round(eta, 1) if eta is not None else 'cached', flush=True)
    return np.concatenate(outputs)


def locked_inputs(protocol):
    for p, expected in protocol['server_files_sha256'].items():
        require(sha(p) == expected, 'Historical input changed: ' + p)
    bank = read_json(Path(protocol['bank']) / 'manifest.json')
    for name in ('ae_dict.json', 'oe_dict.json'):
        require(bank['knowledge_inputs'][name] == protocol['asset_sha256'][name], 'Wrong reviewed knowledge bank')


def execute(args, work, protocol):
    # Old metadata/StageA sources are verified before any pickle load.
    print('VERIFYING_LOCKED_INPUTS no CPF training no TEST images', flush=True)
    locked_inputs(protocol)
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from metrics import distribution, measure, transitions, pair_bootstrap, family_gate
    require(torch.cuda.is_available(), 'GPU required for the real probe; no silent CPU fallback')
    device = torch.device('cuda:0')
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    data = Path(protocol['data'])
    base = Path(protocol['stageA_base'])
    pair_root = data / 'compositional-split-natural'
    pairs = {s: set(tuple(t.split()) for t in (pair_root / (s + '_pairs.txt')).read_text().splitlines() if t.strip())
             for s in ('train', 'val')}
    require(len(pairs['train']) == 83 and len(pairs['val']) == 30, 'Split size mismatch')
    require(not pairs['train'] & pairs['val'], 'TRAIN/VAL unseen pair overlap')
    ids = {s: read_json(base / (s + '_image_ids.json')) for s in ('train', 'val')}
    require([len(ids[s]) for s in ('train', 'val')] == [22998, 3214], 'Image count mismatch')
    labels = read_json(base / 'labels.json')
    require(labels == protocol['labels'], 'Label order mismatch')
    metadata_path = data / 'metadata_compositional-split-natural.t7'
    metadata = trusted_torch_load(metadata_path, protocol['server_files_sha256'][str(metadata_path)])
    records = build_records(metadata, ids, labels, pairs['train'], pairs['val'], data,
                            read_json(ROOT / 'assets' / 'path_aliases.json'))
    caches = {}
    for phase in ('train', 'val'):
        rp = work / (phase + '_records.json')
        if rp.exists():
            require(read_json(rp) == records[phase], 'Images changed since previous attempt')
        else:
            write_json(rp, records[phase])
        cp = base / (phase + '.pt')
        cache = trusted_torch_load(cp, protocol['server_files_sha256'][str(cp)])
        attrs = np.array([r['attr'] for r in records[phase]], dtype=np.int64)
        objects = np.array([r['object'] for r in records[phase]], dtype=np.int64)
        require(np.array_equal(cache['attrs'].numpy(), attrs) and
                np.array_equal(cache['objects'].numpy(), objects), 'Metadata/cache truth differs')
        require(tuple(cache['scores'].shape) == (len(attrs), 12, 16), 'CPF score shape mismatch')
        # Already-audited CPF score-induced conditionals; diagnostic only.
        caches[phase] = distribution(cache['scores'].numpy(), 0.05)
        del cache
    print('IMAGE_PREFLIGHT_PASSED TRAIN 22998 VAL 3214 TEST 0', flush=True)
    from bootstrap import prepare
    checkpoint = prepare(work, protocol, args.checkpoint, args.offline)
    sys.path.insert(0, str(ROOT / 'vendor'))
    import clip
    require(Path(clip.__file__).resolve().parent == (ROOT / 'vendor' / 'clip').resolve(), 'Unexpected CLIP package')
    environment = {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
                   'torchvision': torchvision.__version__, 'numpy': np.__version__,
                   'platform': platform.platform(), 'gpu': torch.cuda.get_device_name(0),
                   'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                   'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                   'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
                   'clip_source_commit': protocol['clip_source_commit'],
                   'batch_size': args.batch_size, 'workers': args.workers,
                   'optimizer_steps': 0, 'test_images_decoded': 0,
                   'preprocess': 'official CLIP Resize(224), CenterCrop(224), RGB, ToTensor, CLIP normalize'}
    # Record attempts separately; resume cannot hide a changed numerical environment.
    ep = work / 'environment.json'
    if ep.exists():
        previous = read_json(ep)
        for key in ('python', 'torch', 'torchvision', 'numpy', 'gpu', 'cuda_visible_devices',
                    'cublas_workspace_config', 'checkpoint_sha256', 'clip_source_commit'):
            require(previous[key] == environment[key], 'Resume environment changed: ' + key)
    else:
        write_json(ep, environment)
    print('CLIP_LOAD_BEGIN ViT-L/14 224px', flush=True)
    # Official SHA-verified OpenAI JIT archive. jit=False reconstructs eager model.
    model, preprocess = clip.load(str(checkpoint), device=device, jit=False)
    model.eval().requires_grad_(False)
    require(int(model.visual.input_resolution) == 224, 'Unexpected image resolution')
    ae, oe = (read_json(ROOT / 'assets' / name) for name in ('ae_dict.json', 'oe_dict.json'))
    check_descriptions(protocol, ae, oe)
    prompts = make_prompts(protocol, ae, oe)
    write_json(work / 'prompts.json', prompts)
    tables, token_audit = encode_texts(clip, model, prompts, device, args.batch_size)
    write_json(work / 'text_audit.json', token_audit)
    np.savez_compressed(work / 'text_features.npz', **tables)
    print('TEXT_AUDIT_PASSED', token_audit, flush=True)
    summaries = {v: {} for v in VARIANTS}
    reports = {}
    result_dir = work / 'teacher'
    result_dir.mkdir(exist_ok=True)
    for phase in ('train', 'val'):
        feats = image_features(model, preprocess, records[phase], work, phase,
                               device, args.batch_size, args.workers)
        attrs = np.array([r['attr'] for r in records[phase]], dtype=np.int64)
        objects = np.array([r['object'] for r in records[phase]], dtype=np.int64)
        seen = np.array([r['seen'] for r in records[phase]], dtype=bool)
        masks = {'all': np.ones(len(attrs), dtype=bool)}
        if phase == 'val':
            masks.update(seen=seen, unseen=~seen)
        np.savez_compressed(result_dir / (phase + '_truth.npz'), attrs=attrs, objects=objects, seen=seen)
        values = {}
        for variant in VARIANTS:
            cosine = (feats @ tables[variant].reshape(192, 768).T).reshape(-1, 12, 16)
            q = distribution(cosine, protocol['temperature']).astype(np.float32)
            require(np.isfinite(cosine).all(), 'Nonfinite teacher scores')
            np.savez_compressed(result_dir / (phase + '_' + variant + '.npz'), cosine=cosine, q=q)
            values[variant] = q
            summaries[variant][phase] = {k: measure(q, attrs, objects, m) for k, m in masks.items()}
            print('TEACHER_METRICS', phase, variant,
                  'GT_OBJECT_ATTR_PCT', summaries[variant][phase]['all']['accuracy_pct'], flush=True)
        report = {'cpf_reference': {k: measure(caches[phase], attrs, objects, m) for k, m in masks.items()},
                  'transitions_vs_name': {}, 'transitions_vs_C_text': {}, 'paired_intervals': {}}
        for variant in VARIANTS:
            report['transitions_vs_name'][variant] = {k: transitions(values['T_name'], values[variant], attrs, objects, m)
                                                      for k, m in masks.items()}
            report['transitions_vs_C_text'][variant] = {k: transitions(caches[phase], values[variant], attrs, objects, m)
                                                        for k, m in masks.items()}
        if phase == 'val':
            for family, controls in (('T_K', ('T_name', 'T_perm1', 'T_perm2')),
                                     ('T_AE', ('T_name', 'T_AE_perm1', 'T_AE_perm2'))):
                report['paired_intervals'][family] = {
                    c: pair_bootstrap(values[family], values[c], attrs, objects, masks['unseen']) for c in controls}
        reports[phase] = report
        write_json(work / (phase + '_diagnostics.json'), report)
    gates = {'OE_AE_primary': family_gate(summaries, 'T_K', 'T_perm1', 'T_perm2'),
             'AE_only_secondary': family_gate(summaries, 'T_AE', 'T_AE_perm1', 'T_AE_perm2')}
    recommendation = ('OE_AE_teacher_controlled_pilot' if gates['OE_AE_primary']['passed'] else (
                      'AE_only_teacher_controlled_pilot' if gates['AE_only_secondary']['passed'] else
                      'stop_expanding_this_teacher_recipe_review_train_failures_once'))
    write_json(work / 'summary.json', {'metrics': summaries, 'gates': gates,
                                     'recommendation': recommendation,
                                     'metric_scope': 'GT-object conditional attribute diagnostic, NOT deployable inference or CPF CW/OW',
                                     'temperature_fitted': False, 'test_used': False,
                                     'official_val_is_development_not_final_test': True})
    with open(work / 'metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'split', 'subset', 'images', 'pairs', 'gt_object_attr_pct', 'pair_macro_pct', 'nll'])
        for variant, phases in summaries.items():
            for phase, subsets in phases.items():
                for subset, v in subsets.items():
                    writer.writerow([variant, phase, subset, v['images'], v.get('pairs'),
                                     v.get('accuracy_pct'), v.get('pair_macro_accuracy_pct'), v.get('nll')])
    # Recheck protected historical files. No writes to CPF/data/previous results.
    locked_inputs(protocol)
    write_json(work / 'complete.json', {'execution_passed': True, 'gates': gates,
                                      'recommendation': recommendation, 'optimizer_steps': 0,
                                      'test_images_decoded': 0, 'historical_inputs_unchanged': True})
    print('KNOWLEDGE_PROBE_COMPLETE', 'OE_AE_GATE', gates['OE_AE_primary']['passed'],
          'AE_GATE', gates['AE_only_secondary']['passed'], 'optimizer_steps 0 TEST_IMAGES 0', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', required=True)
    parser.add_argument('--export', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    require(args.batch_size > 0 and args.workers >= 0, 'Invalid loader settings')
    manifest = verify_release()
    protocol = read_json(ROOT / 'protocol.json')
    work = Path(args.work).resolve()
    # Restrict all generated outputs to this explicitly named new workspace.
    require(work.parent == Path('/var/tmp') and work.name == 'czsl_clip_knowledge_probe_20261010', 'Unexpected work path')
    if args.resume:
        require(work.is_dir(), 'Resume work absent')
    else:
        require(not work.exists(), 'STOP: preserve existing work; use diagnosed resume, not a second start')
        work.mkdir()
    import fcntl  # Server Linux only; an OS lock survives crashes correctly.
    with open(work / 'run.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = work / 'release_identity.json'
        identity = {'release_manifest_sha256': sha(ROOT / 'release_manifest.json'), 'protocol': protocol}
        if receipt.exists():
            require(read_json(receipt) == identity, 'Resume source/protocol changed')
        else:
            require(not args.resume, 'Resume identity receipt absent')
            write_json(receipt, identity)
        from export_result import export
        require(not Path(args.export).exists(), 'Export target already exists; use a new suffix')
        try:
            if (work / 'complete.json').exists():
                require(read_json(work / 'complete.json')['execution_passed'], 'Invalid completion state')
                print('ALREADY_COMPLETE no re-encoding or new evaluation', flush=True)
            else:
                execute(args, work, protocol)
        except Exception:
            message = traceback.format_exc()
            print(message, flush=True)
            write_json(work / ('failure_%d.json' % time.time_ns()), {'execution_passed': False, 'traceback': message})
            export(work, args.export)
            raise
        export(work, args.export)


if __name__ == '__main__':
    main()
