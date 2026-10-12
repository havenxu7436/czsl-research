"""Fixed TRAIN mechanics + five 10-epoch frozen-cache conditional heads, VAL only."""
import argparse
import copy
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
from common import ROOT, new_output, read_json, sha, state_sha, tensor_save, verify_files, verify_release, write_json, zip_destination
from head import VARIANTS
from runtime import (FIELDS, conditional_by_seen, cpu_state, environment, fresh_head, load_inputs,
                     matched_initializations, native_metrics, optimizer, paired_predictions,
                     restore_rng, rng_state, scores_and_diagnostics, train_epoch, train_fold)


def owned_child(root, name):
    from pathlib import PurePosixPath
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe owned-artifact path')
    path = (root / name).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('Artifact escaped its owned directory')
    return path


def next_attempt(folder):
    folder.mkdir(parents=True, exist_ok=True)
    i = 0
    while (folder / ('attempt_%04d' % i)).exists():
        i += 1
    target = folder / ('attempt_%04d' % i)
    target.mkdir()
    return target


def cuda_resume_check(bank, controls, cache, settings, device):
    import torch
    head = fresh_head(bank, 'P_K', controls, device)
    opt = optimizer(head, settings)
    tiny = {k: v[:16] for k, v in cache.items()}
    indices = torch.arange(16)
    train_epoch(head, opt, tiny, indices, 1, device, batch_size=16)
    stream = io.BytesIO()
    torch.save(dict(head=cpu_state(head), optimizer=opt.state_dict(), rng=rng_state()), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location='cpu', weights_only=True)
    train_epoch(head, opt, tiny, indices, 2, device, batch_size=16)
    expected = cpu_state(head)
    restored = fresh_head(bank, 'P_K', controls, device)
    restored.load_state_dict(saved['head'], strict=True)
    restored_opt = optimizer(restored, settings)
    restored_opt.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    train_epoch(restored, restored_opt, tiny, indices, 2, device, batch_size=16)
    if not all(torch.equal(value, cpu_state(restored)[k]) for k, value in expected.items()):
        raise ValueError('Optimizer/checkpoint continuation failed on this device')
    return dict(passed=True, device=str(device), next_step_parameter_state_exact=True,
                diagnostic_optimizer_step_calls=3, diagnostic_states_discarded=True)


def mechanics(work, caches, ids, bank, controls, settings, protocol_sha, initializations, device):
    import numpy as np
    import torch
    complete = work / 'mechanics_complete.json'
    if complete.exists():
        record = read_json(complete)
        if not record['passed'] or record['protocol_sha256'] != protocol_sha:
            raise ValueError('TRAIN mechanics used a different protocol')
        verify_files({str(owned_child(work, name)): digest for name, digest in record['files_sha256'].items()})
        print('MECHANICS_REUSED_VERIFIED', flush=True)
        return record
    folder = next_attempt(work / 'mechanics')
    fit, hold, pair_counts = train_fold(caches['train'], ids['train'], settings['train_holdout_fraction'])
    write_json(folder / 'fold.json', dict(fit_indices=fit.tolist(), hold_indices=hold.tolist(), pair_counts=pair_counts,
                                        base_already_trained_on_full_TRAIN=True,
                                        not_independent_validation=True, new_head_holdout_not_optimized=True))
    reports, predictions = {}, {}
    for variant in VARIANTS:
        print('MECHANICS_BEGIN', variant, 'fit', len(fit), 'hold', len(hold), flush=True)
        head = fresh_head(bank, variant, controls, device)
        if state_sha(head) != initializations[variant]['initial_state_sha256']:
            raise ValueError('Diagnostic initialization changed')
        initial = cpu_state(head)
        _, before, _ = scores_and_diagnostics(head, caches['train'], device, hold)
        opt = optimizer(head, settings)
        trained = train_epoch(head, opt, caches['train'], fit, 1, device, settings['batch_size'])
        if set(trained['nonzero_gradient_parameters']) != set(dict(head.named_parameters())):
            raise ValueError('TRAIN mechanics has inactive parameters: ' + variant)
        _, after, vector = scores_and_diagnostics(head, caches['train'], device, hold)
        if after['candidate_object_delta_std'] <= 1e-9:
            raise ValueError('TRAIN mechanics did not connect candidate-object evidence')
        reports[variant] = dict(protocol_sha256=protocol_sha, train_epoch=trained,
                                hold_before=before, hold_after=after, initial_state_sha256=state_sha(fresh_head(bank, variant, controls, 'cpu')),
                                formal_training_started=False, val_forward_calls=0, test_forward_calls=0)
        predictions[variant] = vector
        np.savez_compressed(folder / (variant + '_hold_predictions.npz'), **vector)
        write_json(folder / (variant + '.json'), reports[variant])
        head.load_state_dict(initial, strict=True)
        if state_sha(head) != initializations[variant]['initial_state_sha256']:
            raise ValueError('Could not restore initial diagnostic head')
        del head, opt
    paired = {v: paired_predictions(predictions['P_K'], predictions[v]) for v in ('P_name', 'P_perm1', 'P_perm2', 'P_generic')}
    write_json(folder / 'summary.json', dict(passed=True, reports=reports, pk_paired_holdout=paired,
               numerical_gate_only=True, no_hyperparameter_or_variant_selection_from_holdout=True,
               all_diagnostic_heads_discarded=True))
    files = {str(p.relative_to(work)).replace('\\', '/'): sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    record = dict(passed=True, protocol_sha256=protocol_sha, path=str(folder.relative_to(work)).replace('\\', '/'),
                  files_sha256=files, diagnostic_epochs_per_head=1,
                  actual_diagnostic_optimizer_steps=sum(x['train_epoch']['steps'] for x in reports.values()))
    write_json(complete, record)
    print('MECHANICS_COMPLETE: resetting every head before formal training', flush=True)
    return record


def evaluate_epoch(head, cache, datasets, evaluator, device, folder, epoch):
    import numpy as np
    grid, diagnostics, vectors = scores_and_diagnostics(head, cache, device)
    if epoch == 0 and not __import__('torch').equal(grid, cache['scores']):
        raise ValueError('Formal epoch 0 is not exact frozen-baseline replay')
    metrics = {s: native_metrics(grid, cache['attrs'], cache['objects'], ds, evaluator)
               for s, ds in datasets.items()}
    diagnostics['by_seen_pair'] = conditional_by_seen(vectors, datasets)
    np.save(folder / 'grid_scores.npy', grid.numpy(), allow_pickle=False)
    np.savez_compressed(folder / 'conditional_predictions.npz', **vectors)
    record = dict(epoch=epoch, metrics_raw=metrics, conditional_diagnostics=diagnostics,
                  inference_uses_ground_truth_objects=False,
                  gt_object_used_only_in_conditional_diagnostics=True)
    write_json(folder / 'evaluation.json', record)
    return record


def checkpoint_commit(run, head, opt, epoch, steps, ledger, best, best_state, protocol_sha, variant, initial_sha):
    """last.pt is the atomic commit point; derived best/history repaired on resume."""
    saved = dict(schema='conditional_attribute_pilot_checkpoint_v1', protocol_sha256=protocol_sha,
                 variant=variant, initial_state_sha256=initial_sha, completed_epoch=epoch,
                 cumulative_optimizer_steps=steps, head_state=cpu_state(head), optimizer_state=opt.state_dict(),
                 rng=rng_state(), ledger=ledger, best=best, best_state=best_state)
    tensor_save(run / 'last.pt', saved)
    repair_derived(run, saved)
    return saved


def repair_derived(run, saved):
    tensor_save(run / 'best_auc.pt', dict(schema=saved['schema'], protocol_sha256=saved['protocol_sha256'],
                variant=saved['variant'], initial_state_sha256=saved['initial_state_sha256'],
                best=saved['best'], head_state=saved['best_state']))
    write_json(run / 'history.json', dict(protocol_sha256=saved['protocol_sha256'], variant=saved['variant'],
                                         completed_epoch=saved['completed_epoch'],
                                         cumulative_optimizer_steps=saved['cumulative_optimizer_steps'],
                                         ledger=saved['ledger'], best=saved['best']))


def formal_variant(work, variant, caches, bank, controls, datasets, evaluator, settings,
                   protocol_sha, initialization, device):
    import torch
    run = work / 'runs' / variant
    run.mkdir(parents=True, exist_ok=True)
    head = fresh_head(bank, variant, controls, device)
    initial_sha = state_sha(head)
    if initial_sha != initialization['initial_state_sha256']:
        raise ValueError('Formal initialization not matched to mechanics/protocol')
    opt = optimizer(head, settings)
    last = run / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=True)
        if (saved['protocol_sha256'] != protocol_sha or saved['variant'] != variant
                or saved['initial_state_sha256'] != initial_sha or saved['schema'] != 'conditional_attribute_pilot_checkpoint_v1'):
            raise ValueError('Checkpoint identity mismatch')
        ledger, best, best_state = saved['ledger'], saved['best'], saved['best_state']
        epoch, steps = saved['completed_epoch'], saved['cumulative_optimizer_steps']
        if not isinstance(epoch, int) or not 0 <= epoch <= settings['epochs']:
            raise ValueError('Checkpoint epoch outside fixed budget')
        if [x['epoch'] for x in ledger] != list(range(epoch + 1)):
            raise ValueError('Checkpoint epoch ledger not contiguous')
        expected_steps = epoch * math_steps(len(caches['train']['features']), settings['batch_size'])
        if steps != expected_steps:
            raise ValueError('Checkpoint optimizer budget differs')
        for item in ledger:
            verify_files({str(owned_child(run, item['path'] + '/' + name)): digest for name, digest in item['files_sha256'].items()})
        if best['epoch'] != max(ledger, key=lambda x: x['metrics_raw']['cw']['AUC'])['epoch']:
            raise ValueError('Checkpoint selection is not first strict CW AUC maximum')
        head.load_state_dict(saved['head_state'], strict=True)
        opt.load_state_dict(saved['optimizer_state'])
        restore_rng(saved['rng'])
        repair_derived(run, saved)
        print('FORMAL_RESUME', variant, 'after_epoch', epoch, 'steps', steps, flush=True)
    else:
        epoch, steps, ledger = 0, 0, []
        folder = next_attempt(run / 'epochs' / 'epoch_0000')
        evaluated = evaluate_epoch(head, caches['val'], datasets, evaluator, device, folder, 0)
        evaluated['path'] = str(folder.relative_to(run)).replace('\\', '/')
        evaluated['files_sha256'] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        evaluated['train_epoch'] = None
        ledger.append(evaluated)
        best, best_state = copy.deepcopy(evaluated), cpu_state(head)
        checkpoint_commit(run, head, opt, epoch, steps, ledger, best, best_state, protocol_sha, variant, initial_sha)
    indices = torch.arange(len(caches['train']['features']))
    for epoch in range(epoch + 1, settings['epochs'] + 1):
        started = time.time()
        print('FORMAL_EPOCH_BEGIN', variant, epoch, '/', settings['epochs'], flush=True)
        trained = train_epoch(head, opt, caches['train'], indices, epoch, device, settings['batch_size'])
        steps += trained['steps']
        folder = next_attempt(run / 'epochs' / ('epoch_%04d' % epoch))
        evaluated = evaluate_epoch(head, caches['val'], datasets, evaluator, device, folder, epoch)
        evaluated['path'] = str(folder.relative_to(run)).replace('\\', '/')
        evaluated['files_sha256'] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        evaluated['train_epoch'] = trained
        evaluated['epoch_seconds'] = time.time() - started
        ledger.append(evaluated)
        if evaluated['metrics_raw']['cw']['AUC'] > best['metrics_raw']['cw']['AUC']:
            best, best_state = copy.deepcopy(evaluated), cpu_state(head)
        checkpoint_commit(run, head, opt, epoch, steps, ledger, best, best_state, protocol_sha, variant, initial_sha)
        print('FORMAL_EPOCH_END', variant, epoch, 'CW_AUC_PCT', round(100*evaluated['metrics_raw']['cw']['AUC'], 6),
              'OW_AUC_PCT', round(100*evaluated['metrics_raw']['ow']['AUC'], 6),
              'BEST_EPOCH', best['epoch'], 'SECONDS', round(evaluated['epoch_seconds'], 2), flush=True)
    expected_steps = settings['epochs'] * math_steps(len(caches['train']['features']), settings['batch_size'])
    if steps != expected_steps or len(ledger) != settings['epochs'] + 1:
        raise ValueError('Incomplete formal budget')
    # Read the selected small checkpoint back and replay its actual VAL scores.
    selected = torch.load(run / 'best_auc.pt', map_location='cpu', weights_only=True)
    head.load_state_dict(selected['head_state'], strict=True)
    replay, _, _ = scores_and_diagnostics(head, caches['val'], device)
    import numpy as np
    selected_scores = np.load(owned_child(run, best['path'] + '/grid_scores.npy'), allow_pickle=False)
    if not np.array_equal(replay.numpy(), selected_scores):
        raise ValueError('Selected checkpoint score replay differs')
    record = dict(passed=True, variant=variant, completed_epoch=settings['epochs'],
                  protocol_sha256=protocol_sha, initial_state_sha256=initial_sha,
                  formal_optimizer_steps=steps, backbone_optimizer_steps=0, best_epoch=best['epoch'],
                  selected=best, selected_checkpoint_replay_exact=True,
                  last_checkpoint_sha256=sha(run / 'last.pt'), best_checkpoint_sha256=sha(run / 'best_auc.pt'))
    write_json(run / 'complete.json', record)
    return record


def math_steps(n, batch):
    return (n + batch - 1) // batch


def decision(results, baseline, gate):
    selected = {k: v['selected']['metrics_raw'] for k, v in results.items()}
    def diff(left, right, setting='cw', field='AUC'):
        return 100 * (left[setting][field] - right[setting][field])
    pk = selected['P_K']
    permutation_mean = .5 * (selected['P_perm1']['cw']['AUC'] + selected['P_perm2']['cw']['AUC'])
    deltas = dict(pk_minus_baseline_cw_auc_pp=diff(pk, baseline),
                  pk_minus_name_cw_auc_pp=diff(pk, selected['P_name']),
                  pk_minus_perm1_cw_auc_pp=diff(pk, selected['P_perm1']),
                  pk_minus_perm2_cw_auc_pp=diff(pk, selected['P_perm2']),
                  pk_minus_mean_permutations_cw_auc_pp=100*(pk['cw']['AUC']-permutation_mean),
                  pk_minus_generic_cw_auc_pp=diff(pk, selected['P_generic']),
                  pk_minus_baseline_cw_hm_pp=diff(pk, baseline, field='best_hm'),
                  pk_minus_baseline_ow_auc_pp=diff(pk, baseline, 'ow'),
                  pk_minus_baseline_ow_best_unseen_pp=diff(pk, baseline, 'ow', 'best_unseen'))
    conditions = {k: deltas[k] >= gate[k] for k in ('pk_minus_baseline_cw_auc_pp', 'pk_minus_name_cw_auc_pp', 'pk_minus_mean_permutations_cw_auc_pp')}
    conditions.update(cw_hm=deltas['pk_minus_baseline_cw_hm_pp'] >= gate['cw_hm_floor_pp'],
                      ow_auc=deltas['pk_minus_baseline_ow_auc_pp'] >= gate['ow_auc_floor_pp'],
                      ow_best_unseen=deltas['pk_minus_baseline_ow_best_unseen_pp'] >= gate['ow_best_unseen_floor_pp'])
    return dict(pilot_budget_gate_passed=all(conditions.values()), conditions=conditions, deltas_pp=deltas,
                original_gates_unchanged=True, not_statistical_significance=True,
                generic_requires_its_own_name_and_permutation_controls=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    ref = read_json(ROOT / 'reference.json')
    release_sha = verify_release()
    work = args.work.resolve()
    protected = Path(ref['stageA_base']).parent.resolve()
    if work == protected or protected in work.parents:
        raise ValueError('Never write inside original Stage A')
    export = zip_destination(args.export, ref, [work, protected])
    if args.resume:
        if not work.is_dir() or not (work / 'protocol.json').is_file():
            raise ValueError('Resume requires an existing owned pilot protocol')
        if read_json(work / 'protocol.json').get('schema') != 'conditional_attribute_pilot_v1':
            raise ValueError('Not an owned pilot directory')
        for root in (ROOT, Path('/data'), Path(ref['c_source']).parent, Path(ref['bf_source']).parent):
            if work == root.resolve() or root.resolve() in work.parents:
                raise ValueError('Resume inside a protected original directory')
    else:
        work = new_output(work, ref, free_mib=2048)
    import fcntl
    lock = (work / 'pilot.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        write_json(work / 'execution_status.json', dict(state='running', resume=args.resume))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Real pilot requires the original CPF CUDA environment')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        print('VERIFYING_FIXED_STAGE_A_INPUTS: no image decoding or backbone loading', flush=True)
        caches, ids, bank, datasets, evaluator = load_inputs(ref)
        env = environment()
        if args.resume and read_json(work / 'environment.json') != env:
            raise ValueError('Runtime environment changed; do not silently resume')
        write_json(work / 'environment.json', env)
        settings, controls = ref['pilot_settings'], ref['control_protocol']
        initializations = matched_initializations(bank, controls)
        protocol = dict(schema='conditional_attribute_pilot_v1', release_manifest_sha256=release_sha,
                        stageA_zip_sha256=ref['stageA_zip_sha256'], stageA_files=ref['stageA_files'],
                        anchor_checkpoint_sha256=ref['c_reference']['checkpoint_sha256'],
                        settings=settings, controls=controls, initializations=initializations,
                        budget_gate=ref['budget_gate'], work=str(work), original_gates_unchanged=True,
                        images_decoded=0, test_evaluation_allowed=False,
                        mechanism_fold_not_independent_because_base_saw_full_TRAIN=True)
        if args.resume:
            if read_json(work / 'protocol.json') != protocol:
                raise ValueError('Resume protocol differs')
        else:
            write_json(work / 'protocol.json', protocol)
        protocol_sha = sha(work / 'protocol.json')
        device = torch.device('cuda:0')
        # Repeat the discarded three-step selfcheck on every explicit invocation.
        check = cuda_resume_check(bank, controls, caches['train'], settings, device)
        check['protocol_sha256'] = protocol_sha
        write_json(next_attempt(work / 'resume_selfchecks') / 'selfcheck.json', check)
        # Independent cache-level frozen baseline before any trained-head VAL result.
        baseline = {s: native_metrics(caches['val']['scores'], caches['val']['attrs'], caches['val']['objects'], ds, evaluator)
                    for s, ds in datasets.items()}
        if any(abs(baseline[s][k] - ref['metrics'][s][k]) > 1e-10 for s in baseline for k in FIELDS):
            raise ValueError('Frozen baseline changed')
        write_json(work / 'baseline.json', dict(metrics_raw=baseline, protocol_sha256=protocol_sha, replay_exact_main_metrics=True))
        write_json(work / 'labels.json', ref['labels'])
        for s, ds in datasets.items():
            write_json(work / ('candidates_' + s + '.json'), ds.pairs)
        write_json(work / 'val_image_ids.json', ids['val'])
        import numpy as np
        np.savez_compressed(work / 'val_truth.npz', attrs=caches['val']['attrs'].numpy(), objects=caches['val']['objects'].numpy())
        mechanics(work, caches, ids, bank, controls, settings, protocol_sha, initializations, device)
        results = {}
        for variant in VARIANTS:
            results[variant] = formal_variant(work, variant, caches, bank, controls, datasets, evaluator,
                                            settings, protocol_sha, initializations[variant], device)
        # Compare GT-object attribute corrections/damage using identical VAL identities.
        prediction = {}
        for variant, result in results.items():
            path = owned_child(work / 'runs' / variant, result['selected']['path'] + '/conditional_predictions.npz')
            with np.load(path, allow_pickle=False) as data:
                prediction[variant] = {k: data[k].copy() for k in data.files}
        summary = dict(passed=True, protocol_sha256=protocol_sha, results=results, baseline_metrics_raw=baseline,
                       decision=decision(results, baseline, ref['budget_gate']),
                       pk_paired_val_conditional={v: paired_predictions(prediction['P_K'], prediction[v])
                                                 for v in ('P_name', 'P_perm1', 'P_perm2', 'P_generic')},
                       formal_optimizer_steps=sum(x['formal_optimizer_steps'] for x in results.values()),
                       backbone_optimizer_steps=0, image_decodes=0, test_evaluation=False)
        verify_files(ref['stageA_files'])
        verify_files(ref['c_files'])
        write_json(work / 'summary.json', summary)
        write_json(work / 'execution_status.json', dict(state='complete', protocol_sha256=protocol_sha))
        print('PILOT_COMPLETE', 'formal_steps', summary['formal_optimizer_steps'],
              'BUDGET_GATE', summary['decision']['pilot_budget_gate_passed'], flush=True)
        from export_results import export_work
        export_work(work, export)
    except Exception as error:
        # Attempt-indexed failures retained even when a later explicit resume succeeds.
        folder = next_attempt(work / 'failures')
        write_json(folder / 'failure.json', dict(error=repr(error), originals_not_overwritten=True,
                                               formal_success_not_claimed=True))
        write_json(work / 'execution_status.json', dict(state='failed', error=repr(error)))
        from export_results import export_work
        try:
            export_work(work, export)
        except Exception as export_error:
            print('FORENSIC_EXPORT_FAILED', repr(export_error), flush=True)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

if __name__ == '__main__':
    main()
