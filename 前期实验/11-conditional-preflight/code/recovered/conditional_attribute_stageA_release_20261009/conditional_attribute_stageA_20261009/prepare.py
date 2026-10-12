"""Frozen C_text TRAIN/VAL cache + original evaluator replay + real head checks."""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

sys.dont_write_bytecode = True
from common import (ROOT, deterministic_items, finite, load_cfg, load_native, native_context,
                    new_output, read_json, seed_all, sha, state_sha, tensor_save,
                    verify_files, verify_release, write_json)


def set_candidates(model, ds, pairs):
    import torch
    model.val_attrs = torch.tensor([ds.attr2idx[a] for a, o in pairs], device='cuda:0')
    model.val_objs = torch.tensor([ds.obj2idx[o] for a, o in pairs], device='cuda:0')
    model.val_pairs = list(pairs)


def evaluate_cached(base, ds):
    import numpy as np
    import torch
    from evaluator_ge import Evaluator
    ai = torch.tensor([ds.attr2idx[a] for a, o in ds.pairs])
    oi = torch.tensor([ds.obj2idx[o] for a, o in ds.pairs])
    scores = base['scores'][:, oi, ai].contiguous()
    prediction = {pair: scores[:, i] for i, pair in enumerate(ds.pairs)}
    attrs, objects = base['attrs'], base['objects']
    pair = torch.tensor([ds.pair2idx[(ds.attrs[int(a)], ds.objs[int(o)])] for a, o in zip(attrs, objects)])
    evaluator = Evaluator(ds, None)
    results = evaluator.score_model(prediction, objects, bias=1e3, topk=1)
    stats = evaluator.evaluate_predictions(results, attrs, objects, pair, prediction, topk=1)
    metrics = {k: float(v) for k, v in stats.items()}
    for k in ('AUC', 'best_hm', 'best_seen', 'best_unseen', 'hm_seen', 'hm_unseen'):
        if not np.isfinite(metrics[k]):
            raise ValueError('Invalid evaluator metric: ' + k)
    return metrics, scores, pair


def head_smoke(train, bank):
    import torch
    from head import VARIANTS, ConditionalHead, conditional_loss, probability_checks
    reports, shared_state = {}, None
    x, base = train['features'][:16].cuda(), train['scores'][:16].cuda()
    attrs, objects = train['attrs'][:16].cuda(), train['objects'][:16].cuda()
    for variant in VARIANTS:
        seed_all(0)
        head = ConditionalHead(bank, variant).cuda()
        # Generic adds a fusion layer; common input tensors must still match.
        current = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if shared_state is None:
            shared_state = current
        else:
            for k, value in shared_state.items():
                if not torch.equal(value, current[k]):
                    raise ValueError('Unmatched initial tensor: ' + variant + '/' + k)
        before = {k: v.clone() for k, v in head.state_dict().items()}
        with torch.no_grad():
            if not torch.equal(head(x, base), base) or not torch.equal(head(x, base, force_off=True), base):
                raise ValueError('Zero/off branch did not replay full grid')
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        active_gradients = set()
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            delta = head.evidence(x)
            loss = conditional_loss(base, delta, attrs, objects)
            finite(loss, variant + '/conditional_nll')
            loss.backward()
            for name, parameter in head.named_parameters():
                if parameter.grad is not None:
                    finite(parameter.grad, name)
                    if float(parameter.grad.norm()) > 0:
                        active_gradients.add(name)
            if step == 0 and 'output.weight' not in active_gradients:
                raise ValueError('Zero initialization blocks output gradient')
            optimizer.step()
        with torch.no_grad():
            delta = head.evidence(x)
            finite(delta, variant + '/delta')
            errors = probability_checks(base, delta)
            if max(errors.values()) > 3e-6:
                raise ValueError('Probability normalization failure')
            spread = float(delta.std(dim=1, unbiased=False).mean())
            if spread <= 1e-9 or float(delta.abs().max()) > 0.100001:
                raise ValueError('No candidate-object variation or cap violation')
        if active_gradients != {k for k, v in head.named_parameters()}:
            raise ValueError('Inactive trainable parameter: ' + variant)
        reports[variant] = dict(zero_replay_exact=True, force_off_replay_exact=True,
                                active_parameter_count=sum(v.numel() for v in head.parameters()),
                                parameters_with_nonzero_gradients=sorted(active_gradients),
                                candidate_object_evidence_std=spread, probabilities=errors,
                                diagnostic_optimizer_steps=3, permutations=head.permutations)
        # Discard all diagnostic optimization; formal pilot starts from original init.
        head.load_state_dict(before, strict=True)
        if not torch.equal(head(x, base), base):
            raise ValueError('Could not restore initial head')
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        raise ValueError('Extraction batch size must be between 1 and 64')
    release_sha = verify_release()
    ref = read_json(ROOT / 'reference.json')
    out = new_output(args.out, ref, free_mib=2048)
    started = time.time()
    try:
        import numpy as np
        import torch
        from head import VARIANTS
        if not torch.cuda.is_available():
            raise RuntimeError('Use the original CPF CUDA environment')
        verify_files(ref['c_files'])
        native_context(ref['c_source'])
        from dataset import CompositionDataset
        from transition.common import load_bank
        cfg = load_cfg(ref['config'])
        if (cfg.DATASET.name != 'ut-zappos' or int(cfg.TRAIN.seed) != 0
                or bool(cfg.MODEL.kcc_calib) or float(cfg.MODEL.kcc_lambda) != 0
                or bool(cfg.MODEL.kcc_logic) or float(cfg.MODEL.kcc_beta) != 0
                or str(cfg.MODEL.semantic_text) != 'correct'):
            raise ValueError('Expected unmodified strong C_text_v2 seed 0; calibration/logic OFF')
        seed_all(0)
        train_ds = CompositionDataset(phase='train', split=cfg.DATASET.splitname, open_world=False, cfg=cfg)
        val_ds = CompositionDataset(phase='val', split=cfg.DATASET.splitname, open_world=False, cfg=cfg)
        if len(train_ds) != 22998 or len(val_ds) != 3214:
            raise ValueError('Unexpected TRAIN/VAL image counts')
        if train_ds.attrs != ref['labels']['attrs'] or train_ds.objs != ref['labels']['objs']:
            raise ValueError('Primitive order differs from frozen reference')
        if val_ds.attrs != train_ds.attrs or val_ds.objs != train_ds.objs:
            raise ValueError('TRAIN/VAL vocabulary mismatch')
        if [item[0] for item in val_ds.data] != ref['val_evidence']['cw']['image_ids']:
            raise ValueError('VAL identity/order differs from original evaluation')
        bank, _ = load_bank(ref['bank'], train_ds.attrs, train_ds.objs)
        model = load_native(train_ds, cfg, ref['checkpoint'])
        model.requires_grad_(False)
        before = state_sha(model)
        full_pairs = list(product(train_ds.attrs, train_ds.objs))  # native attribute-major grid
        set_candidates(model, train_ds, full_pairs)
        caches = {}
        for phase, ds in (('val', val_ds), ('train', train_ds)):
            loader = torch.utils.data.DataLoader(deterministic_items(ds), batch_size=args.batch_size,
                                                shuffle=False, num_workers=0)
            n = len(ds)
            cache = dict(features=torch.empty(n, 300), scores=torch.empty(n, 12, 16),
                         attrs=torch.empty(n, dtype=torch.long), objects=torch.empty(n, dtype=torch.long))
            offset = 0
            with torch.no_grad():
                for batch in loader:
                    end = offset + len(batch['img'])
                    if batch['index'].tolist() != list(range(offset, end)):
                        raise ValueError('Cache loader order mismatch')
                    values = model.raw_forward({'img': batch['img'].cuda()})
                    score, feature = values['pred'], values['attr_feature']
                    finite(score, phase + '/score')
                    finite(feature, phase + '/feature')
                    if score.shape != (end - offset, 192) or feature.shape != (end - offset, 300):
                        raise ValueError('Unexpected native raw_forward dimensions')
                    cache['scores'][offset:end] = score.reshape(-1, 16, 12).permute(0, 2, 1).cpu()
                    cache['features'][offset:end] = feature.cpu()
                    cache['attrs'][offset:end], cache['objects'][offset:end] = batch['attr'], batch['obj']
                    # Check copied raw_forward and native val_forward on real images.
                    if offset == 0:
                        native = model.val_forward({'img': batch['img'].cuda()})['pred']
                        if not torch.equal(score, native):
                            raise ValueError('raw_forward differs from native val_forward')
                        # CW common-candidate projection should agree with a direct native forward.
                        set_candidates(model, ds, val_ds.pairs)
                        direct = model.val_forward({'img': batch['img'].cuda()})['pred']
                        ai = torch.tensor([ds.attr2idx[a] for a, o in val_ds.pairs])
                        oi = torch.tensor([ds.obj2idx[o] for a, o in val_ds.pairs])
                        projected = cache['scores'][offset:end][:, oi, ai]
                        discrepancy = float((direct.cpu() - projected).abs().max())
                        if discrepancy > 2e-6:
                            raise ValueError('Full grid/CW candidate projection mismatch')
                        set_candidates(model, ds, full_pairs)
                    offset = end
                    if offset % (25 * args.batch_size) == 0 or offset == n:
                        print('CACHE_IMAGES', phase, offset, '/', n, 'seconds', round(time.time() - started, 1), flush=True)
            if offset != n:
                raise ValueError('Incomplete cache extraction')
            tensor_save(out / (phase + '.pt'), cache)
            write_json(out / (phase + '_image_ids.json'), [item[0] for item in ds.data])
            caches[phase] = cache
        if state_sha(model) != before:
            raise ValueError('Frozen base parameters/buffers changed')
        del model
        torch.cuda.empty_cache()
        replay = {}
        for setting in ('cw', 'ow'):
            ds = val_ds if setting == 'cw' else CompositionDataset(phase='val', split=cfg.DATASET.splitname, open_world=True, cfg=cfg)
            if list(map(list, ds.pairs)) != ref['val_evidence'][setting]['candidates']:
                raise ValueError('Original candidate order differs: ' + setting)
            if [item[0] for item in ds.data] != ref['val_evidence'][setting]['image_ids']:
                raise ValueError('Original image order differs: ' + setting)
            metrics, scores, pairs = evaluate_cached(caches['val'], ds)
            labels = np.load(ref['val_evidence'][setting]['labels_file'], allow_pickle=False)
            if (not np.array_equal(labels['attr'], caches['val']['attrs'].numpy())
                    or not np.array_equal(labels['obj'], caches['val']['objects'].numpy())
                    or not np.array_equal(labels['pair'], pairs.numpy())):
                raise ValueError('Original evaluation labels differ: ' + setting)
            differences = {k: metrics[k] - ref['metrics'][setting][k]
                           for k in ('AUC', 'best_hm', 'best_seen', 'best_unseen', 'hm_seen', 'hm_unseen')}
            top1_same = bool(np.array_equal(scores.argmax(1).numpy(), labels['top1_at_bias0']))
            replay[setting] = dict(metrics_raw=metrics, reference_difference_raw=differences,
                                   original_bias0_top1_equal=top1_same, tolerance_raw=1e-5)
            write_json(out / ('replay_' + setting + '.json'), replay[setting])
            np.save(out / ('val_scores_' + setting + '.npy'), scores.numpy(), allow_pickle=False)
            write_json(out / ('candidates_' + setting + '.json'), ds.pairs)
            np.savez_compressed(out / ('labels_' + setting + '.npz'), attr=caches['val']['attrs'].numpy(),
                                obj=caches['val']['objects'].numpy(), pair=pairs.numpy(), top1_at_bias0=scores.argmax(1).numpy())
            if max(abs(v) for v in differences.values()) > 1e-5 or not top1_same:
                raise ValueError('C_text evaluator replay failed; do not relax tolerance or reselect weights')
        checks = head_smoke(caches['train'], bank)
        write_json(out / 'head_smoke.json', checks)
        write_json(out / 'labels.json', ref['labels'])
        # Freeze input permutation identities before any new trained-head VAL evaluation.
        write_json(out / 'control_protocol.json', dict(variants=list(VARIANTS),
                   permutations={k: checks[k]['permutations'] for k in VARIANTS},
                   rank=32, temperature=0.05, evidence_cap=0.10,
                   pending_formal_epochs=10, pending_batch_size=256,
                   pending_lr=1e-3, pending_weight_decay=1e-4,
                   selection='first strict maximum VAL CW AUC over epochs 0..10',
                   generic_active_extra_parameters=checks['P_generic']['active_parameter_count'] - checks['P_K']['active_parameter_count'],
                   original_gates_unchanged=True))
        verify_files(ref['c_files'])
        files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
        report = dict(passed=True, release_manifest_sha256=release_sha, files_sha256=files,
                      cache_shapes={'train': [22998, 12, 16], 'val': [3214, 12, 16]},
                      anchor_checkpoint_sha256=ref['c_reference']['checkpoint_sha256'],
                      base_state_sha256=before, base_state_unchanged=True, replay=replay,
                      native_forward_identity_checked=True, cw_projection_atol=2e-6,
                      diagnostic_new_head_optimizer_steps=15, formal_training_optimizer_steps=0,
                      backbone_optimizer_steps=0, test_images_decoded=0,
                      metadata_includes_test_vocabulary=True, extraction_batch_size=args.batch_size,
                      deterministic_eval_transform_for_train_and_val=True,
                      seconds=time.time() - started,
                      training_status='Stage A only. No formal pilot or TRAIN diagnostic-fold training.',
                      diagnostic_heads_discarded_and_not_saved=True)
        write_json(out / 'prepared.json', report)
        print('CONDITIONAL_STAGE_A_PASSED', json.dumps({'cw_auc_pct': 100 * replay['cw']['metrics_raw']['AUC'],
                   'ow_auc_pct': 100 * replay['ow']['metrics_raw']['AUC'], 'backbone_steps': 0, 'test_images': 0}), flush=True)
    except Exception as error:
        write_json(out / 'failure.json', dict(error=repr(error), formal_training_started=False,
                                             originals_not_overwritten=True, partial_outputs_preserved=True))
        raise


if __name__ == '__main__':
    main()
