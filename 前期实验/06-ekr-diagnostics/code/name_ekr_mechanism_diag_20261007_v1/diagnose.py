"""Read-only diagnostic of completed name-anchor routers; no optimizer or training."""
import argparse
import collections
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F

VERSION = 'name_ekr_mechanism_diag_20261007_v1'
PRIMARY = ('N_name', 'N_clean', 'N_shuffle')
METRICS = ('AUC', 'best_hm', 'best_seen', 'best_unseen', 'hm_seen', 'hm_unseen')
TOLERANCE_PP = 0.05


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path, rows):
    require(bool(rows), 'Empty diagnostic table')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def derangement(n, seed):
    require(n >= 2, 'Derangement needs at least two labels')
    rng = np.random.RandomState(seed)
    for unused in range(10000):
        order = rng.permutation(n)
        if np.all(order != np.arange(n)):
            return order
    raise RuntimeError('Unable to construct the declared input permutation')


def altered_bank(bank, kind, seed):
    """Clone changed buffers only; retain own attribute name in all AE interventions."""
    result = dict(bank)
    result['a'], result['o'] = dict(bank['a']), dict(bank['o'])
    mapping = None
    if kind == 'ae_permutation':
        mapping = derangement(len(bank['a']['unit']), seed)
        for key in ('units', 'mask'):
            value = bank['a'][key].clone()
            value[:, 1:] = bank['a'][key][torch.as_tensor(mapping), 1:]
            result['a'][key] = value
    elif kind == 'object_context_permutation':
        mapping = derangement(len(bank['o']['unit']), seed)
        result['o']['unit'] = bank['o']['unit'][torch.as_tensor(mapping)].clone()
    else:
        require(kind == 'original', 'Unknown input intervention')
    return result, None if mapping is None else mapping.tolist()


@torch.no_grad()
def probe(router, feature, attr_ids, obj_ids, remove_null=False):
    """Reproduce the frozen router and expose candidate attention without changing it."""
    require(router.mode in ('name_only', 'ekr', 'shuffle'), 'Unsupported diagnostic router mode')
    u = F.normalize(feature, dim=-1, eps=1e-6)
    keys = F.normalize(router.pk(router.key_source), dim=-1, eps=1e-6)
    evidence = torch.einsum('bd,ard->bar', u, router.value_source)
    objects = torch.unique(obj_ids, sorted=True)
    local_obj = (obj_ids[:, None] == objects[None]).long().argmax(1)
    context = torch.sigmoid(router.qo(router.object_source[objects]))
    query = F.normalize(router.qx(u)[:, None] * context[None], dim=-1, eps=1e-6)
    logits = torch.einsum('bnk,ark->bnar', query, keys) / router.temperature
    logits = logits.masked_fill(~router.slot_mask[None, None], float('-inf'))
    null = torch.einsum('bnk,k->bn', query, F.normalize(router.null_key, dim=0, eps=1e-6))
    null = null / router.temperature + router.null_bias
    logits = torch.cat([null[:, :, None, None].expand(-1, -1, logits.size(2), 1), logits], -1)
    all_attention = logits.softmax(-1)
    attention = all_attention[:, local_obj, attr_ids]
    if remove_null:
        attention = attention.clone()
        attention[..., 0] = 0
        attention[..., 1:] /= attention[..., 1:].sum(-1, keepdim=True).clamp_min(1e-12)
    delta = (attention[..., 1:] * evidence[:, attr_ids]).sum(-1) * router.eta
    require(torch.isfinite(delta).all() and torch.isfinite(attention).all(), 'Nonfinite probe result')
    return delta, attention


def point_stats(base, residual, attention, gold, allowed, candidate_attr, candidate_obj, gold_attr, gold_obj):
    """Gold is used only in offline summaries, never in the router query or prediction."""
    n = len(gold)
    row = np.arange(n)
    require(np.all(allowed[gold]), 'Gold candidate absent from diagnostic mask')
    base_masked = np.where(allowed[None], base, -np.inf)
    total = np.where(allowed[None], base + residual, -np.inf)
    competitor = np.array(base_masked, copy=True)
    competitor[row, gold] = -np.inf
    competitor_idx = competitor.argmax(1)
    strongest = competitor[row, competitor_idx]
    new_competitor = np.array(total, copy=True)
    new_competitor[row, gold] = -np.inf
    base_margin = base_masked[row, gold] - strongest
    new_margin = total[row, gold] - new_competitor.max(1)
    fixed_contrast = residual[row, gold] - residual[row, competitor_idx]
    bp = base_masked.argmax(1)
    pred = total.argmax(1)
    good_before, good_after = bp == gold, pred == gold
    gold_null = attention[row, gold, 0]
    gold_weights = attention[row, gold, 1:]
    conditional = gold_weights / np.maximum(gold_weights.sum(1, keepdims=True), 1e-12)
    entropy = -(conditional * np.log(np.maximum(conditional, 1e-12))).sum(1)
    # This is a mean over the declared data view; TRAIN is pair-balanced, not population-weighted.
    def ce(scores):
        maximum = scores.max(1)
        return float((maximum + np.log(np.exp(scores - maximum[:, None]).sum(1)) - scores[row, gold]).mean())
    result = dict(n_images=n, n_candidates=len(allowed), n_allowed=int(allowed.sum()),
        base_pair_pct=100*float(good_before.mean()), pair_pct=100*float(good_after.mean()),
        base_ce=ce(base_masked / 0.05), ce=ce(total / 0.05),
        prediction_changed=int((pred != bp).sum()), corrected=int((good_after & ~good_before).sum()),
        broken=int((good_before & ~good_after).sum()), net_correct=int(good_after.sum() - good_before.sum()),
        object_prediction_changed=int((candidate_obj[pred] != candidate_obj[bp]).sum()),
        attribute_prediction_changed=int((candidate_attr[pred] != candidate_attr[bp]).sum()),
        object_pct=100*float((candidate_obj[pred] == gold_obj).mean()),
        attribute_pct=100*float((candidate_attr[pred] == gold_attr).mean()),
        base_margin_mean=float(base_margin.mean()), margin_mean=float(new_margin.mean()),
        margin_delta_mean=float((new_margin - base_margin).mean()),
        fixed_competitor_residual_contrast_mean=float(fixed_contrast.mean()),
        fixed_competitor_positive_contrast_pct=100*float((fixed_contrast > 0).mean()),
        residual_abs_mean=float(np.abs(residual).mean()), residual_abs_max=float(np.abs(residual).max()),
        null_mean=float(attention[..., 0].mean()), gold_null_mean=float(gold_null.mean()),
        gold_null_over_half_pct=100*float((gold_null > 0.5).mean()),
        gold_knowledge_entropy_mean=float(entropy.mean()), gold_knowledge_entropy_max=math.log(4),
        gold_slot_name_mean=float(conditional[:, 0].mean()),
        gold_slot_ae1_mean=float(conditional[:, 1].mean()),
        gold_slot_ae2_mean=float(conditional[:, 2].mean()),
        gold_slot_ae3_mean=float(conditional[:, 3].mean()))
    per_image = dict(base_pred=bp, pred=pred, base_correct=good_before, correct=good_after,
        base_margin=base_margin, new_margin=new_margin, margin_delta=new_margin-base_margin,
        fixed_competitor_idx=competitor_idx, fixed_competitor_residual_contrast=fixed_contrast,
        gold_null=gold_null, gold_knowledge_entropy=entropy)
    return result, per_image


def native_metrics(ds, model, scores, labels):
    import evaluator_ge
    ev = evaluator_ge.Evaluator(ds, model)
    matrix = torch.from_numpy(np.array(scores, dtype=np.float32, copy=True))
    prediction = {tuple(p): matrix[:, i] for i, p in enumerate(ds.pairs)}
    ga, go, gp = [torch.from_numpy(labels[k]) for k in ('attr', 'obj', 'pair')]
    scored = ev.score_model(prediction, go, bias=1e3, topk=1)
    stats = ev.evaluate_predictions(scored, ga, go, gp, prediction, topk=1)
    result = {}
    for key, value in stats.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        a = np.asarray(value)
        if a.size == 1:
            result[key] = float(a.item())
    require(all(k in result and np.isfinite(result[k]) for k in METRICS), 'Invalid native diagnostic metrics')
    return result


def build_case(bank, saved, kind, seed, mode, device, remove_null=False):
    from transition.name_router import NameRouter
    altered, mapping = altered_bank(bank, kind, seed)
    with torch.random.fork_rng(devices=[]):
        router = NameRouter(altered, mode=mode, rank=32, eta_max=0.1, temperature=0.2, object_chunk=16)
    router.load_state_dict(saved['router_state'], strict=True)
    router.to(device).eval().requires_grad_(False)
    return router, mapping, remove_null


def check_inputs(work, reference):
    require(str(work) == reference['work'], 'Use the reviewed work directory')
    mismatches = []
    for path, expected in reference['inputs_sha256'].items():
        if not Path(path).is_file() or sha(path) != expected:
            mismatches.append(path)
    require(not mismatches, 'Reviewed input changed or missing: ' + str(mismatches[:5]))
    from transition.name_plan import validate_protocol, validate_config
    value = validate_protocol(work, full=True)
    require(value['training']['branch_epochs'] == 3 and value['training']['frozen_base'], 'Wrong reviewed pilot')
    gate = read_json(work / 'reports/pilot/decision.json')
    require(gate['stage'] == 'pilot' and gate['seeds'] == [0] and gate['proceed'] is False,
            'This diagnostic pins the reviewed failed seed-0 pilot')
    from transition.name_runtime import read_cfg
    configs, checkpoints = {}, {}
    for variant in PRIMARY:
        run = work / 'experiments/name_ekr/runs' / ('name_ut-zappos_' + variant + '_s0_e3')
        configs[variant] = read_cfg(run / 'effective_config.yml')
        validate_config(configs[variant], require_replay=True)
        complete = read_json(run / 'train_complete.json')
        require(complete['epochs'] == 3 and complete['base_state_unchanged'] and
            complete['base_state_start'] == complete['base_state_end'], 'Pilot base state or budget changed')
        require(read_json(run / 'status.json')['status'] == 'train_and_val_complete', 'Incomplete pilot')
        checkpoints[variant] = run / 'checkpoints/best_branch.pth'
        require(sha(checkpoints[variant]) == complete['checkpoint_sha256'], 'Branch checkpoint changed')
    return value, configs, checkpoints


def cache_view(model, cfg, view, out, seed):
    from dataset import CompositionDataset
    from transition.common import candidates, move
    cfg = cfg.clone()
    cfg.EVAL.open_world = view == 'val_ow'
    ds = CompositionDataset(phase='val', split=cfg.DATASET.splitname, open_world=view == 'val_ow', cfg=cfg)
    if view == 'train_balanced':
        # Reuse deterministic VAL transform for selected TRAIN images. Do not load auxiliary TRAIN views.
        by_pair = collections.defaultdict(list)
        for index, record in enumerate(ds.train_data):
            by_pair[tuple(record[1:])].append(index)
        rng = np.random.RandomState(seed)
        chosen = []
        for pair in sorted(ds.train_pairs):
            pool = by_pair[tuple(pair)]
            require(bool(pool), 'TRAIN pair has no images')
            chosen.extend(rng.permutation(pool)[:min(6, len(pool))].tolist())
        ds.data = [ds.train_data[i] for i in chosen]
        ds.pairs = list(ds.train_pairs)
        ds.pair2idx = {pair: i for i, pair in enumerate(ds.pairs)}
        write_json(out / 'train_selection.json', dict(seed=seed, max_per_pair=6, n_images=len(chosen),
            n_pairs=len(ds.train_pairs), training_record_indices=chosen,
            transform='deterministic VAL transform; TRAIN records only; no auxiliary images'))
    ai, oi, pairs = candidates(model, ds)
    n, nc = len(ds.data), len(pairs)
    folder = out / 'cache' / view
    folder.mkdir(parents=True, exist_ok=False)
    scores = np.lib.format.open_memmap(str(folder / 'base_scores.npy'), mode='w+', dtype=np.float32, shape=(n, nc))
    features = np.lib.format.open_memmap(str(folder / 'attr_features.npy'), mode='w+', dtype=np.float32, shape=(n, 300))
    labels = {key: np.empty(n, np.int64) for key in ('attr', 'obj', 'pair')}
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.TRAIN.test_batch_size, shuffle=False, num_workers=cfg.TRAIN.num_workers)
    offset = 0
    with torch.no_grad():
        for batch in dl:
            batch = move(batch, 'cuda:0')
            base, feature = model.components(batch)
            size = len(feature)
            require(tuple(base.shape) == (size, nc) and tuple(feature.shape) == (size, 300), 'Cache shape mismatch')
            require(torch.isfinite(base).all() and torch.isfinite(feature).all(), 'Nonfinite frozen features')
            scores[offset:offset+size] = base.detach().float().cpu().numpy()
            features[offset:offset+size] = feature.detach().float().cpu().numpy()
            for key in labels:
                labels[key][offset:offset+size] = batch[key].detach().cpu().numpy()
            offset += size
            if offset % 512 < size:
                print('CACHE', view, offset, '/', n, flush=True)
    require(offset == n, 'Cache count mismatch')
    scores.flush(); features.flush()
    np.savez_compressed(str(folder/'labels.npz'), **labels)
    ids = [record[0] for record in ds.data]
    write_json(folder/'image_ids.json', ids)
    write_json(folder/'candidates.json', pairs)
    allowed = np.array([True if view != 'val_cw' else p in set(ds.train_pairs + ds.val_pairs) for p in pairs])
    candidate_attr = ai.detach().cpu().numpy(); candidate_obj = oi.detach().cpu().numpy()
    return dict(ds=ds, ai=ai, oi=oi, base=scores, feature=features, labels=labels, ids=ids, pairs=pairs,
        allowed=allowed, candidate_attr=candidate_attr, candidate_obj=candidate_obj)


def score_case(router, cache, folder, remove_null, batch_size=64):
    folder.mkdir(parents=True, exist_ok=False)
    n, nc = cache['base'].shape
    residual = np.lib.format.open_memmap(str(folder/'residual.npy'), mode='w+', dtype=np.float32, shape=(n, nc))
    attention = np.lib.format.open_memmap(str(folder/'attention.npy'), mode='w+', dtype=np.float32, shape=(n, nc, 5))
    score = np.lib.format.open_memmap(str(folder/'scores.npy'), mode='w+', dtype=np.float32, shape=(n, nc))
    probe_error = 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            feature = torch.from_numpy(np.array(cache['feature'][start:end], copy=True)).to(next(router.parameters()).device)
            delta, weights = probe(router, feature, cache['ai'], cache['oi'], remove_null=remove_null)
            if start == 0 and not remove_null:
                original = router(feature, cache['ai'], cache['oi'])
                probe_error = float((original-delta).abs().max())
                require(probe_error <= 1e-6, 'Probe disagrees with the trained router')
            residual[start:end] = delta.cpu().numpy()
            attention[start:end] = weights.cpu().numpy()
            score[start:end] = cache['base'][start:end] + residual[start:end]
    for array in (residual, attention, score): array.flush()
    stats, per_image = point_stats(cache['base'], residual, attention, cache['labels']['pair'], cache['allowed'],
        cache['candidate_attr'], cache['candidate_obj'], cache['labels']['attr'], cache['labels']['obj'])
    np.savez_compressed(str(folder/'per_image.npz'), **per_image)
    stats['probe_max_error'] = probe_error
    stats['eta'] = float(router.eta)
    return score, residual, attention, stats


def run(args):
    bundle = Path(__file__).resolve().parent
    reference = read_json(bundle/'reference.json')
    work, out = Path(args.work).resolve(), Path(args.out).resolve()
    require(out != work and work not in out.parents and out not in work.parents, 'Diagnostic output must be separate from old work')
    require(not out.exists(), 'Diagnostic output exists; preserve it and use a new output version')
    sys.path.insert(0, str(work/'src'))
    print('VERIFY REVIEWED PILOT INPUTS; NO TRAINING', flush=True)
    value, configs, checkpoints = check_inputs(work, reference)
    require(torch.cuda.is_available(), 'CUDA is required for the frozen image encoder')
    require(torch.__version__ == '1.13.1+cu117', 'Use the reviewed CPF PyTorch 1.13.1+cu117 environment')
    require(shutil.disk_usage(out.parent).free >= 2 * 1024**3, 'Need at least 2 GiB free on the output filesystem')
    from transition.common import seed_all, state_digest
    from transition.name_runtime import load_branch, trainset
    seed_all(0)
    out.mkdir(parents=True)
    write_json(out/'status.json', dict(status='running', version=VERSION))
    started = time.time()
    cfg = configs['N_clean']
    model, clean_saved = load_branch(cfg, trainset(cfg), checkpoints['N_clean'])
    model.eval().requires_grad_(False)
    before = state_digest(model.base)
    recorded_base = read_json(work/'experiments/name_ekr/runs/name_ut-zappos_N_clean_s0_e3/train_complete.json')
    require(before == recorded_base['base_state_start'], 'Loaded base differs from the reviewed training base state')
    saved = {'N_clean': clean_saved}
    for variant in ('N_name', 'N_shuffle'):
        saved[variant] = torch.load(str(checkpoints[variant]), map_location='cpu')
        require(saved[variant]['anchor_sha256'] == cfg.MODEL.branch_anchor_sha256 and
            saved[variant]['bank_manifest_sha256'] == clean_saved['bank_manifest_sha256'] and
            saved[variant]['branch_mode'] == configs[variant].MODEL.branch_mode and
            saved[variant]['regime'] == 'frozen', 'Checkpoint identity differs')
    cases = [('N_name_original','N_name','original',0,False), ('N_clean_original','N_clean','original',0,False),
        ('N_shuffle_original','N_shuffle','original',0,False)]
    cases += [('N_clean_ae_permutation_'+str(s),'N_clean','ae_permutation',s,False) for s in (11,22,33)]
    cases += [('N_name_ae_permutation_11','N_name','ae_permutation',11,False),
        ('N_clean_object_context_permutation_11','N_clean','object_context_permutation',11,False),
        ('N_clean_null_removed','N_clean','original',0,True)]
    protocol = dict(version=VERSION, work=str(work), output=str(out), new_training=False, optimizer_used=False,
        old_pilot_proceed=False, cases=[dict(case_id=c[0],variant=c[1],intervention=c[2],seed=c[3],remove_null=c[4]) for c in cases],
        views=['train_balanced','val_cw','val_ow'], train_max_images_per_pair=6, train_selection_seed=20261007,
        target_metric_replay_tolerance_pp=TOLERANCE_PP, inputs_sha256=reference['inputs_sha256'],
        bundle_sha256={p.name:sha(p) for p in bundle.iterdir() if p.is_file()},
        scientific_status='input intervention on existing weights; failed pilot unchanged; no test; reused VAL development',
        note='Input permutations are diagnostic controls, not random semantic graphs; null removal is frozen inference, not trained ablation')
    write_json(out/'diagnostic_protocol.json', protocol)
    rows, replay_checks, intervention_checks, group_rows = [], [], [], []
    for view in protocol['views']:
        print('START VIEW', view, flush=True)
        cache = cache_view(model, cfg, view, out, protocol['train_selection_seed'])
        if view.startswith('val_'):
            setting = view[4:]
            native_base = native_metrics(cache['ds'], model, cache['base'], cache['labels'])
            target = reference['anchor_metrics'][setting]
            max_pp = max(abs(100*(native_base[k]-target['metrics_raw'][k])) for k in METRICS)
            require(max_pp <= TOLERANCE_PP, 'Frozen base replay metric drift exceeds tolerance')
            oldfolder = work/'diagnostics/full_scores'/('semantic_ut-zappos_A_unit_s0_e10')/setting
            require(cache['ids'] == read_json(oldfolder/'image_ids.json') and cache['pairs'] == [tuple(p) for p in read_json(oldfolder/'candidates.json')], 'VAL IDs/candidates changed')
            with np.load(str(oldfolder/'labels.npz'),allow_pickle=False) as oldlabels:
                require(all(np.array_equal(cache['labels'][k],oldlabels[k]) for k in ('attr','obj','pair')), 'VAL labels changed')
            replay_checks.append(dict(variant='N0',view=view,max_metric_difference_pp=max_pp))
        originals = {}
        for case_id, variant, kind, seed, remove_null in cases:
            router, mapping, remove_null = build_case(model.bank,saved[variant],kind,seed,configs[variant].MODEL.branch_mode,'cuda:0',remove_null)
            router_before = state_digest(router)
            folder = out/'cases'/view/case_id
            scores, residual, attention, stats = score_case(router, cache, folder, remove_null)
            require(state_digest(router) == router_before, 'Router parameters changed in a diagnostic case')
            stats['router_parameters_unchanged'] = True
            stats.update(case_id=case_id,view=view,variant=variant,intervention=kind,mapping=mapping,remove_null=remove_null)
            if kind == 'original' and not remove_null:
                originals[variant] = scores
            if view.startswith('val_'):
                native = native_metrics(cache['ds'],model,scores,cache['labels'])
                stats.update(native_auc_pct=100*native['AUC'],native_hm_pct=100*native['best_hm'],native_best_unseen_pct=100*native['best_unseen'])
                if kind == 'original' and not remove_null:
                    target = reference['branch_metrics'][variant][view[4:]]
                    max_pp = max(abs(100*(native[k]-target['metrics_raw'][k])) for k in METRICS)
                    require(max_pp <= TOLERANCE_PP,'New branch replay metric drift exceeds tolerance: '+case_id)
                    with np.load(str(work/'experiments/name_ekr/runs'/('name_ut-zappos_'+variant+'_s0_e3')/('eval_val_'+view[4:])/'labels.npz'),allow_pickle=False) as oldlabels:
                        raw_mismatches=int((np.asarray(scores).argmax(1)!=oldlabels['top1_at_bias0']).sum())
                    stats['raw_argmax_replay_mismatches']=raw_mismatches
                    replay_checks.append(dict(variant=variant,view=view,max_metric_difference_pp=max_pp,raw_argmax_mismatches=raw_mismatches))
                stats['native_metrics_raw'] = native
            else:
                stats.update(native_auc_pct=None,native_hm_pct=None,native_best_unseen_pct=None)
            if not (kind=='original' and not remove_null):
                delta = np.asarray(scores)-np.asarray(originals[variant])
                stats['intervention_abs_score_change_mean']=float(np.abs(delta).mean())
                stats['intervention_abs_score_change_max']=float(np.abs(delta).max())
                stats['raw_prediction_changed_from_original']=int((np.asarray(scores).argmax(1)!=np.asarray(originals[variant]).argmax(1)).sum())
                if variant=='N_name' and kind=='ae_permutation':
                    require(stats['intervention_abs_score_change_max']<=1e-6,'Name control leaked description input')
                intervention_checks.append(dict(case_id=case_id,view=view,
                    score_change_max=stats['intervention_abs_score_change_max'],
                    score_change_mean=stats['intervention_abs_score_change_mean']))
            write_json(folder/'metrics.json',stats)
            row={k:v for k,v in stats.items() if k not in ('mapping','native_metrics_raw')}
            for key in ('raw_argmax_replay_mismatches','intervention_abs_score_change_mean','intervention_abs_score_change_max','raw_prediction_changed_from_original'):
                row.setdefault(key,None)
            rows.append(row)
            with np.load(str(folder/'per_image.npz'),allow_pickle=False) as p:
                gold_obj=cache['labels']['obj']
                for oi in sorted(set(gold_obj.tolist())):
                    mask=gold_obj==oi
                    group_rows.append(dict(view=view,case_id=case_id,object=cache['ds'].objs[oi],n=int(mask.sum()),
                        pair_pct=100*float(p['correct'][mask].mean()),
                        margin_delta_mean=float(p['margin_delta'][mask].mean()),
                        gold_null_mean=float(p['gold_null'][mask].mean())))
            print('CASE',view,case_id,'PAIR_PCT',round(stats['pair_pct'],4),'NULL_MEAN',round(stats['null_mean'],6),flush=True)
            del router
        del cache, originals
        torch.cuda.empty_cache()
    after=state_digest(model.base)
    require(before==after,'Frozen base state changed during diagnostics')
    check_inputs(work,reference)
    csv_write(out/'diagnostic_summary.csv',rows)
    csv_write(out/'object_groups.csv',group_rows)
    write_json(out/'replay_checks.json',replay_checks)
    write_json(out/'intervention_checks.json',intervention_checks)
    findings={}
    for view in protocol['views']:
        clean=next(r for r in rows if r['view']==view and r['case_id']=='N_clean_original')
        name=next(r for r in rows if r['view']==view and r['case_id']=='N_name_original')
        noise=[r for r in rows if r['view']==view and r['case_id'].startswith('N_clean_ae_permutation_')]
        context=next(r for r in rows if r['view']==view and r['case_id']=='N_clean_object_context_permutation_11')
        no_null=next(r for r in rows if r['view']==view and r['case_id']=='N_clean_null_removed')
        findings[view]=dict(n_images=clean['n_images'],clean_margin_delta_vs_base=clean['margin_delta_mean'],
            clean_ce_delta_vs_base=clean['ce']-clean['base_ce'],clean_margin_delta_vs_name=clean['margin_mean']-name['margin_mean'],
            clean_gold_null_mean=clean['gold_null_mean'],
            ae_permuted_gold_null_mean=float(np.mean([r['gold_null_mean'] for r in noise])),
            ae_permuted_margin_change_mean=float(np.mean([r['margin_mean']-clean['margin_mean'] for r in noise])),
            ae_permuted_score_change_abs_mean=float(np.mean([r['intervention_abs_score_change_mean'] for r in noise])),
            object_context_score_change_abs_mean=context['intervention_abs_score_change_mean'],
            object_context_score_change_relative_to_original_residual=context['intervention_abs_score_change_mean']/max(clean['residual_abs_mean'],1e-12),
            null_removed_score_change_abs_mean=no_null['intervention_abs_score_change_mean'])
        if view.startswith('val_'):
            findings[view]['ae_permuted_native_auc_change_pp_mean']=float(np.mean([r['native_auc_pct']-clean['native_auc_pct'] for r in noise]))
    write_json(out/'mechanism_findings.json',dict(views=findings,interpretation='Descriptive input interventions, not a success gate; no automatic next training',noise_maps_are_training_seeds=False))
    checks=dict(passed=True,new_training=False,old_pilot_proceed=False,base_state_unchanged=True,
        base_state_digest=after,n_cases=len(cases),n_views=3,n_case_views=len(rows),
        replay_max_metric_difference_pp=max(r['max_metric_difference_pp'] for r in replay_checks),
        raw_argmax_replay_mismatch_total=sum(r.get('raw_argmax_mismatches',0) for r in replay_checks),
        raw_argmax_exact_replay=all(r.get('raw_argmax_mismatches',0)==0 for r in replay_checks),
        name_input_control_passed=True,elapsed_seconds=time.time()-started,
        interpretation='Diagnostic integrity only; not a research gate; no automatic new training recommendation')
    write_json(out/'integrity_checks.json',checks)
    write_json(out/'status.json',dict(status='diagnostics_complete',version=VERSION))
    outputs={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='output_manifest.json'}
    write_json(out/'output_manifest.json',dict(files_sha256=outputs,weights_included=False))
    print('DIAGNOSTICS COMPLETE',json.dumps(checks),flush=True)
    pack(out)


def pack(out):
    require(read_json(out/'status.json')['status']=='diagnostics_complete','Diagnostics not complete')
    manifest=read_json(out/'output_manifest.json')
    actual={str(p.relative_to(out)).replace('\\','/') for p in out.rglob('*') if p.is_file()}
    require(actual==set(manifest['files_sha256'])|{'output_manifest.json'},'Diagnostic output inventory changed')
    require(not any(Path(name).suffix in {'.pt','.pth'} for name in actual),'Weights must not enter the review archive')
    for name, expected in manifest['files_sha256'].items():
        require(sha(out/name)==expected,'Diagnostic output changed: '+name)
    target=out.parent/(out.name+'_review.zip')
    require(not target.exists(),'Diagnostic ZIP exists; use it or a new output version')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,str(path.relative_to(out)).replace('\\','/'))
    print('DOWNLOAD THIS FILE:',target,'SHA256',sha(target),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','pack'])
    p.add_argument('--work',default='/var/tmp/czsl_xhw_name_ekr_20261007')
    p.add_argument('--out',default='/var/tmp/czsl_xhw_name_ekr_diag_20261007_v1')
    args=p.parse_args()
    if args.command=='pack':
        pack(Path(args.out).resolve())
    else:
        try:
            run(args)
        except Exception:
            print('DIAGNOSTIC STOPPED; PRESERVE PARTIAL OUTPUT; NO TRAINING WAS REQUESTED',flush=True)
            raise


if __name__=='__main__':
    main()
