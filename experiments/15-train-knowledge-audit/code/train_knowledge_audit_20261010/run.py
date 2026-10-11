"""One bounded audit: frozen text re-encoding, cached TRAIN features, <=96 images."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from common import ROOT, Inputs, read_json, write_json, write_npz, require, sha, verify_release
from analysis import VARIANTS, make_prompts, load_train, pooled_report, sentence_report
from report import contacts, human_form, html_report


def preflight(probe):
    verify_release()
    protocol = read_json(ROOT / 'protocol.json')
    require(Path(probe).resolve() == Path(protocol['probe_work']).resolve(), 'Use locked r2 work directory')
    expected = read_json(ROOT / 'train_input_manifest.json')['files_sha256']
    inputs = Inputs(expected, root=probe)
    inputs.verify_all()
    require(len(inputs.js('train_records.json')) == 22998, 'TRAIN count changed')
    require(inputs.js('prompts.json') == make_prompts(protocol, read_json(ROOT / 'assets/ae_dict.json'),
                                                    read_json(ROOT / 'assets/oe_dict.json')), 'Prompt recipe changed')
    dependency = inputs.js('dependencies.json')
    for name, digest in dependency['files_sha256'].items():
        from common import valid_relative
        valid_relative(name)
        p = (Path(probe) / 'deps' / name).resolve()
        require((Path(probe) / 'deps').resolve() in p.parents, 'Dependency path escapes')
        require(sha(p) == digest, 'Old isolated dependency changed: ' + name)
    env = inputs.js('environment.json')
    checkpoint = Path(protocol['checkpoint'])
    require(str(checkpoint) == env['checkpoint'] and sha(checkpoint) == protocol['clip_checkpoint_sha256'],
            'Offline CLIP checkpoint absent/changed; no download or fallback')
    print('TRAIN_AUDIT_INPUTS_PASSED TRAIN 22998 VAL_READ 0 TEST_READ 0', flush=True)
    return protocol, inputs


def reencode(probe, protocol, inputs, out, prompts):
    import numpy as np
    unique = sorted(set(t for rows in prompts.values() for t in rows))
    require(len(unique) == 7041, 'Unique text count changed')
    target, receipt = out / 'unit_text_features.npz', out / 'unit_text_receipt.json'
    if target.exists():
        require(receipt.exists(), 'Uncommitted text cache preserved; request diagnosis')
        old = read_json(receipt)
        require(old['sha256'] == sha(target) and old['unique_texts'] == unique, 'Text cache changed')
        with np.load(target, allow_pickle=False) as z:
            vectors = z['features']
    else:
        sys.path.insert(0, str(Path(probe) / 'deps'))
        sys.path.insert(0, str(ROOT / 'vendor'))
        import torch
        import torchvision
        import clip
        original = inputs.js('environment.json')
        require(sys.version_info[:2] == (3, 8) and torch.__version__ == original['torch'] and
                torchvision.__version__ == original['torchvision'] and np.__version__ == original['numpy'],
                'Use existing cpf Python3.8/Torch/Torchvision/NumPy; do not upgrade')
        require(Path(clip.__file__).resolve().parent == (ROOT / 'vendor/clip').resolve(), 'Unexpected CLIP code')
        require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == original['gpu'], 'RTX3090 GPU required')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Missing deterministic CUBLAS setting')
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        write_json(out / 'text_environment.json', {'python': sys.version, 'torch': torch.__version__,
                                                  'torchvision': torchvision.__version__, 'numpy': np.__version__,
                                                  'gpu': torch.cuda.get_device_name(0),
                                                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                                  'clip_source_commit': protocol['clip_source_commit'],
                                                  'clip_checkpoint_sha256': protocol['clip_checkpoint_sha256'],
                                                  'image_encoder_calls': 0, 'optimizer_steps': 0})
        print('TEXT_ONLY_CLIP_LOAD_BEGIN no image encoder forward', flush=True)
        model, _ = clip.load(protocol['checkpoint'], device='cuda:0', jit=False)
        model.eval().requires_grad_(False)
        tokens = clip.tokenize(unique, truncate=False)
        chunks = []
        started = time.monotonic()
        with torch.no_grad():
            for start in range(0, len(unique), 32):
                v = model.encode_text(tokens[start:start + 32].to('cuda:0')).float()
                v /= v.norm(dim=-1, keepdim=True)
                require(torch.isfinite(v).all().item(), 'Nonfinite text vector')
                chunks.append(v.cpu().numpy())
                done = min(start + 32, len(unique))
                if done % 256 == 0 or done == len(unique):
                    print('TEXT_ONLY_ENCODE', done, '/', len(unique), 'seconds', round(time.monotonic() - started, 1), flush=True)
        vectors = np.concatenate(chunks)
        write_npz(target, features=vectors)
        write_json(receipt, {'sha256': sha(target), 'unique_texts': unique, 'batch_size': 32,
                             'checkpoint_sha256': protocol['clip_checkpoint_sha256'], 'image_encoder_calls': 0})
        del model
        torch.cuda.empty_cache()
    require(vectors.shape == (len(unique), 768) and np.isfinite(vectors).all() and
            np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-3), 'Invalid unit text cache')
    lookup = {t: i for i, t in enumerate(unique)}
    return {v: vectors[[lookup[t] for t in rows]].reshape(12, 16, 9, 768) for v, rows in prompts.items()}


def execute(args, protocol, inputs):
    out = Path(args.work)
    require(out.parent == Path('/var/tmp') and out.name == protocol['work_basename'], 'Use locked isolated output directory')
    identity = {'protocol_sha256': sha(ROOT / 'protocol.json'),
                'train_input_manifest_sha256': sha(ROOT / 'train_input_manifest.json'), 'probe_work': str(args.probe_work)}
    if args.resume:
        require(out.is_dir() and read_json(out / 'run_identity.json') == identity, 'Resume identity differs')
        require(not (out / 'complete.json').exists(), 'Already complete; do not repeat')
    else:
        require(not out.exists(), 'Work exists; preserve it and inspect status')
        out.mkdir()
        write_json(out / 'run_identity.json', identity)
    records, features, a, o, old_tables, scores = load_train(inputs, protocol)
    ae, oe = read_json(ROOT / 'assets/ae_dict.json'), read_json(ROOT / 'assets/oe_dict.json')
    pooled, preds = pooled_report(scores, a, o, protocol['labels'])
    write_json(out / 'pooled_train_report.json', pooled)
    prompts = make_prompts(protocol, ae, oe)
    tables = reencode(args.probe_work, protocol, inputs, out, prompts)
    print('TRAIN_SENTENCE_DIAGNOSTICS_BEGIN no optimizer', flush=True)
    sentences = sentence_report(features, tables, old_tables, scores, a, o, protocol['labels'])
    write_json(out / 'sentence_train_report.json', sentences)
    # On resume, never silently overwrite completed pictures; rebuild only if none committed.
    if (out / 'sample_receipt.json').exists():
        sample = read_json(out / 'sample_receipt.json')
        count = sample['train_images_decoded']
        require(count <= 96 and sample['val_images_decoded'] == sample['test_images_decoded'] == 0, 'Sample scope')
        for ai in range(16):
            require((out / 'contact_sheets' / ('attr_%02d.png' % ai)).is_file(), 'Incomplete contact sheets')
    else:
        require(not (out / 'contact_sheets').exists(), 'Partial contact sheets preserved; request diagnosis')
        count = contacts(records, preds, protocol, out)
    human_form(ae, oe, protocol, out)
    html_report(pooled, sentences, protocol, ae, oe, out)
    # Recheck all allowed inputs, never history-mutating write.
    inputs.verify_all()
    write_json(out / 'input_reads.json', {'r2_files_read': sorted(inputs.reads), 'val_files_read': [], 'test_files_read': [],
                                         'r2_inputs_unchanged': True, 'image_encoder_calls': 0,
                                         'train_images_decoded': count, 'optimizer_steps': 0})
    write_json(out / 'complete.json', {'execution_passed': True, 'train_images': len(a), 'train_pairs': 83,
                                      'train_images_decoded': count, 'val_images_decoded': 0, 'test_images_decoded': 0,
                                      'image_encoder_calls': 0, 'optimizer_steps': 0,
                                      'weights_fitted': False, 'teacher_training_authorized_by_this_audit': False,
                                      'next': 'review TRAIN evidence once, then lock a separate minimal protocol or stop recipe'})
    print('TRAIN_KNOWLEDGE_AUDIT_COMPLETE optimizer_steps 0 IMAGE_ENCODER_CALLS 0 TRAIN_DECODED', count,
          'VAL_READ 0 TEST_READ 0', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe-work', default='/var/tmp/czsl_clip_knowledge_probe_20261010_r2')
    p.add_argument('--work')
    p.add_argument('--export')
    p.add_argument('--check-inputs-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    protocol, inputs = preflight(args.probe_work)
    if args.check_inputs_only:
        require(not args.work and not args.export and not args.resume, 'Preflight must not create outputs')
        return
    require(args.work and args.export, 'Need --work and --export')
    out, target = Path(args.work), Path(args.export)
    require(out.parent == Path('/var/tmp') and out.name == protocol['work_basename'], 'Wrong output scope')
    require(target.parent == out.parent and target.name.startswith(out.name) and target.suffix == '.zip', 'Wrong export scope')
    require(not target.exists() and not Path(str(target) + '.sha256').exists() and
            not Path(str(target) + '.partial').exists(), 'Export exists; preserve it')
    if args.resume:
        require(out.is_dir() and read_json(out / 'run_identity.json')['protocol_sha256'] == sha(ROOT / 'protocol.json'),
                'Resume identity differs')
    else:
        require(not out.exists(), 'Work exists; no old outputs modified')
    from export_result import export
    try:
        execute(args, protocol, inputs)
    except Exception:
        out = Path(args.work)
        if (out / 'run_identity.json').is_file() and out.parent == Path('/var/tmp') and out.name == protocol['work_basename']:
            write_json(out / ('failure_' + str(time.time_ns()) + '.json'), {'execution_passed': False,
                                                                        'traceback': traceback.format_exc()})
            export(out, Path(args.export))
        raise
    export(Path(args.work), Path(args.export))


if __name__ == '__main__':
    main()
