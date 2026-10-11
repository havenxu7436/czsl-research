"""Selected B/F checkpoint gradients on 3 shared TRAIN batches; ZERO updates."""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from common import (ROOT, finite, load_cfg, load_native, native_context, new_output,
                    read_json, seed_all, state_sha, verify_files, verify_release, write_json)


def geometry(left, right):
    import torch
    nl, nr = float(torch.linalg.vector_norm(left.double())), float(torch.linalg.vector_norm(right.double()))
    return {'left_norm': nl, 'right_norm': nr,
            'right_over_left_norm': nr / nl if nl > 0 else None,
            'cosine': float(torch.dot(left.double(), right.double()) / (nl * nr)) if nl > 0 and nr > 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    release_sha = verify_release()
    ref = read_json(ROOT / 'reference.json')
    out = new_output(args.out, ref)
    try:
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Use the original CPF CUDA environment')
        verify_files(ref['bf_files'])
        native_context(ref['bf_source'])
        from dataset import CompositionDataset
        from common import deterministic_items
        reports, shared_ids = {}, None
        for variant in ('B', 'F'):
            info = ref['bf'][variant]
            cfg = load_cfg(info['config'])
            if not cfg.MODEL.kcc_logic or float(cfg.MODEL.kcc_beta) != 0.01 or int(cfg.TRAIN.seed) != 3351:
                raise ValueError('Unexpected B/F effective configuration')
            seed_all(3351)
            ds = CompositionDataset(phase='train', split=cfg.DATASET.splitname, open_world=False, cfg=cfg)
            indices = np.random.RandomState(21009).permutation(len(ds))[:12].tolist()
            ids = [ds.data[i][0] for i in indices]
            if shared_ids is not None and ids != shared_ids:
                raise ValueError('B/F image identities differ')
            shared_ids = ids
            items = deterministic_items(ds)
            model = load_native(ds, cfg, info['checkpoint'], conditional=False)
            model.requires_grad_(False)
            names, params, groups = [], [], {'object': [], 'attribute': [], 'joint': []}
            for name, value in model.named_parameters():
                if name.startswith(('objc_embedder.', 'attrc_embedder.')):
                    value.requires_grad_(True)
                    group = 'object' if name.startswith('objc_embedder.') else 'attribute'
                    groups[group].append(len(params))
                    groups['joint'].append(len(params))
                    names.append(name)
                    params.append(value)
            before = state_sha(model)
            observations = []
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(items, indices), batch_size=4, shuffle=False, num_workers=0)
            for batch_id, batch in enumerate(loader):
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                terms = model.train_forward(batch)  # eval mode: dropout/BN updates OFF
                if not torch.allclose(terms['loss_total'], terms['loss_cpf'] + terms['loss_logic_weighted'], atol=1e-6):
                    raise ValueError('Incorrect total loss')
                if not torch.allclose(terms['loss_logic_weighted'], 0.01 * terms['loss_logic_raw'], atol=1e-7):
                    raise ValueError('Logic weight applied incorrectly')
                gradients = {}
                term_names = ('loss_cpf', 'loss_oi', 'loss_oe', 'loss_ae', 'loss_logic_weighted')
                for name in term_names:
                    finite(terms[name], name)
                    if not terms[name].requires_grad:
                        raise ValueError('Disconnected term: ' + name)
                    grad = torch.autograd.grad(terms[name], params, retain_graph=True, allow_unused=True)
                    gradients[name] = [torch.zeros_like(p).cpu().flatten() if g is None else g.detach().cpu().flatten()
                                       for p, g in zip(params, grad)]
                scopes = {}
                for group, positions in groups.items():
                    vectors = {k: torch.cat([v[i] for i in positions]) for k, v in gradients.items()}
                    for k, vector in vectors.items():
                        finite(vector, k + '/' + group)
                    if float(vectors['loss_cpf'].norm()) <= 0:
                        raise ValueError('Zero CPF gradient in ' + group)
                    scopes[group] = {
                        name: geometry(vectors['loss_cpf'], vectors[name])
                        for name in term_names[1:]}
                    scopes[group]['cpf_vs_total'] = geometry(
                        vectors['loss_cpf'], vectors['loss_cpf'] + vectors['loss_logic_weighted'])
                observations.append(dict(batch=batch_id, image_ids=ids[batch_id * 4:(batch_id + 1) * 4],
                                         losses={k: float(terms[k].detach()) for k in term_names}, scopes=scopes))
                del terms, gradients, vectors
            if state_sha(model) != before:
                raise ValueError('Model parameters/buffers mutated during read-only probe')
            reports[variant] = dict(best_epoch=info['best_epoch'], checkpoint_sha256=info['checkpoint_sha256'],
                                   parameters=names, observations=observations, state_unchanged=True)
            write_json(out / (variant + '.json'), reports[variant])
            del model
            torch.cuda.empty_cache()
        verify_files(ref['bf_files'])
        report = dict(passed=True, release_manifest_sha256=release_sha, variants=reports,
                      train_images_decoded_per_model=12, val_images_decoded=0, test_images_decoded=0,
                      optimizer_steps=0, evaluation_mode=True, deterministic_val_transform_on_train=True,
                      projection_gradient_scope_only=True,
                      note='Three sampled TRAIN batches at best weights, not epoch averages, Adam update geometry, or causal efficacy proof.')
        write_json(out / 'probe.json', report)
        print('BF_TRAINED_GRADIENT_PROBE_COMPLETE', flush=True)
    except Exception as error:
        write_json(out / 'failure.json', dict(error=repr(error), optimizer_steps=0, originals_not_overwritten=True))
        raise


if __name__ == '__main__':
    main()
