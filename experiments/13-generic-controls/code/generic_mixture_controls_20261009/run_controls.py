"""Three new generic content controls + verified reuse of the existing correct run."""
import argparse
import copy
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
from bootstrap import ROOT, child, guard_path, read_json, sha, verify_files, verify_release, write_json
import numpy as np
import torch
from common import state_sha, tensor_load, tensor_save
import pilot as legacy_pilot
import runtime as legacy_runtime
from model import (ALPHA, MODES, NEW_VARIANTS, fresh_head, from_grid, initializations,
                   scored, scores_and_diagnostics)

SCHEMA = 'generic_mixture_controls_protocol_v1'


def evaluate_epoch(head, cache, datasets, evaluator, device, folder, epoch):
    grid, diag, vectors, unmixed = scored(head, cache, device)
    if epoch == 0 and not torch.equal(grid, cache['scores']):
        raise ValueError('Generic epoch zero failed exact baseline identity')
    metrics = {s: legacy_runtime.native_metrics(grid, cache['attrs'], cache['objects'], ds, evaluator)
               for s, ds in datasets.items()}
    diag['by_seen_pair'] = legacy_runtime.conditional_by_seen(vectors, datasets)
    np.save(folder / 'grid_scores.npy', grid.numpy(), allow_pickle=False)
    np.save(folder / 'unmixed_grid_scores.npy', unmixed.numpy(), allow_pickle=False)
    np.savez_compressed(folder / 'conditional_predictions.npz', **vectors)
    record = dict(epoch=epoch, metrics_raw=metrics, conditional_diagnostics=diag,
                  inference_uses_ground_truth_objects=False, gt_object_used_only_in_conditional_diagnostics=True,
                  architecture='original_P_generic', alpha=ALPHA,
                  training_loss='original_unmixed_conditional_CE_not_mixture_NLL')
    write_json(folder / 'evaluation.json', record)
    return record


# Reuse the exact r2 atomic checkpoint/epoch engine; change ONLY head factory and
# evaluation readout. The optimizer, data order and conditional training CE remain
# byte-identical vendored functions, all sealed in this release's manifest.
legacy_pilot.fresh_head = fresh_head
legacy_pilot.scores_and_diagnostics = scores_and_diagnostics
legacy_pilot.evaluate_epoch = evaluate_epoch


def reference_checks(work, caches, bank, controls, settings, experiment, device, protocol_sha):
    complete = work / 'reference_checks_complete.json'
    if complete.exists():
        result = read_json(complete)
        if not result['passed'] or result['protocol_sha256'] != protocol_sha:
            raise ValueError('Reference selfcheck protocol changed')
        verify_files({str(child(work, n)): h for n, h in result['files_sha256'].items()})
        print('REFERENCE_CHECKS_REUSED_VERIFIED', flush=True)
        return result
    folder = legacy_pilot.next_attempt(work / 'reference_checks')
    source = Path(experiment['source_work'])
    old = tensor_load(source / 'runs/P_generic/last.pt')
    old_protocol = read_json(source / 'protocol.json')
    if (old['variant'] != 'P_generic' or old['completed_epoch'] != settings['epochs']
            or old['cumulative_optimizer_steps'] != experiment['source_generic_formal_steps']
            or old['protocol_sha256'] != experiment['source_protocol_sha256']
            or old['initial_state_sha256'] != experiment['source_initial_state_sha256']
            or old_protocol != experiment['source_original_protocol']):
        raise ValueError('Existing generic checkpoint/experiment identity differs')
    head = fresh_head(bank, 'G_K', controls, device)
    if state_sha(head) != experiment['source_initial_state_sha256']:
        raise ValueError('Generic initialization differs: do not reuse old correct run')
    raw, _, _ = legacy_runtime.scores_and_diagnostics(head, caches['val'], device)
    if not torch.equal(raw, caches['val']['scores']):
        raise ValueError('Reference head zero replay failed')
    head.load_state_dict(old['head_state'], strict=True)
    raw, _, _ = legacy_runtime.scores_and_diagnostics(head, caches['val'], device)
    last_item = old['ledger'][-1]
    expected = np.load(child(source / 'runs/P_generic', last_item['path'] + '/grid_scores.npy'), allow_pickle=False)
    if not np.array_equal(raw.numpy(), expected):
        raise ValueError('Actual last generic checkpoint forward not exact')
    # Reproduce first full TRAIN epoch, then discard it. This verifies same
    # initialization, generic evidence, conditional CE, AdamW and minibatch order.
    head = fresh_head(bank, 'G_K', controls, device)
    opt = legacy_runtime.optimizer(head, settings)
    indices = torch.arange(len(caches['train']['features']))
    print('REFERENCE_FIRST_EPOCH_REPRODUCE_BEGIN: diagnostic state will be discarded', flush=True)
    trained = legacy_runtime.train_epoch(head, opt, caches['train'], indices, 1, device, settings['batch_size'])
    item = old['ledger'][1]
    if trained != item['train_epoch']:
        raise ValueError('First TRAIN epoch loss/order/gradient ledger not exact')
    raw, _, _ = legacy_runtime.scores_and_diagnostics(head, caches['val'], device)
    expected = np.load(child(source / 'runs/P_generic', item['path'] + '/grid_scores.npy'), allow_pickle=False)
    if not np.array_equal(raw.numpy(), expected):
        raise ValueError('First generic training epoch scores not exact; cannot reuse old run')
    report = dict(passed=True, protocol_sha256=protocol_sha, initial_parameters_exact=True,
                  last_checkpoint_actual_forward_exact=True, first_epoch_training_ledger_exact=True,
                  first_epoch_VAL_scores_exact=True, diagnostic_optimizer_steps=trained['steps'],
                  all_diagnostic_weights_discarded=True)
    write_json(folder / 'replay.json', report)
    report['files_sha256'] = {str((folder / 'replay.json').relative_to(work)).replace('\\', '/'): sha(folder / 'replay.json')}
    write_json(complete, report)
    print('REFERENCE_REUSE_VERIFIED: first epoch and last checkpoint exact', flush=True)
    return report


def reuse_correct(work, caches, bank, controls, settings, experiment, datasets, evaluator, device, protocol_sha):
    run = work / 'runs/G_K'
    complete = run / 'complete.json'
    if complete.exists():
        result = read_json(complete)
        if not result['passed'] or result['protocol_sha256'] != protocol_sha:
            raise ValueError('Reused correct run protocol changed')
        verify_files({str(child(run, n)): h for n, h in result['owned_files_sha256'].items()})
        print('CORRECT_GENERIC_READOUT_REUSED_VERIFIED', flush=True)
        return result
    source = Path(experiment['source_work'])
    history = read_json(source / 'runs/P_generic/history.json')
    if history['completed_epoch'] != settings['epochs'] or [x['epoch'] for x in history['ledger']] != list(range(settings['epochs']+1)):
        raise ValueError('Original generic ledger incomplete')
    attempt = legacy_pilot.next_attempt(run / 'reference_replays')
    ledger, best = [], None
    for original in history['ledger']:
        epoch = original['epoch']
        folder = attempt / ('epoch_%04d' % epoch)
        folder.mkdir()
        source_path = child(source / 'runs/P_generic', original['path'] + '/grid_scores.npy')
        unmixed = torch.from_numpy(np.load(source_path, allow_pickle=False).copy())
        if unmixed.shape != caches['val']['scores'].shape:
            raise ValueError('Original score shape changed')
        grid, diag, vectors = from_grid(caches['val']['scores'], unmixed, caches['val']['attrs'], caches['val']['objects'])
        if epoch == 0 and not torch.equal(grid, caches['val']['scores']):
            raise ValueError('Reused epoch zero not exact baseline')
        metrics = {s: legacy_runtime.native_metrics(grid, caches['val']['attrs'], caches['val']['objects'], ds, evaluator)
                   for s, ds in datasets.items()}
        diag['by_seen_pair'] = legacy_runtime.conditional_by_seen(vectors, datasets)
        diag['unmixed_head_diagnostics'] = original['conditional_diagnostics']
        shutil.copyfile(str(source_path), str(folder / 'unmixed_grid_scores.npy'))
        np.save(folder / 'grid_scores.npy', grid.numpy(), allow_pickle=False)
        np.savez_compressed(folder / 'conditional_predictions.npz', **vectors)
        record = dict(epoch=epoch, metrics_raw=metrics, conditional_diagnostics=diag,
                      train_epoch=original['train_epoch'], original_source_path=str(source_path),
                      original_source_score_sha256=sha(source_path), new_optimizer_steps=0,
                      inference_uses_ground_truth_objects=False, gt_object_used_only_in_conditional_diagnostics=True)
        write_json(folder / 'evaluation.json', record)
        record['path'] = folder.relative_to(run).as_posix()
        record['files_sha256'] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        ledger.append(record)
        if best is None or metrics['cw']['AUC'] > best['metrics_raw']['cw']['AUC']:
            best = copy.deepcopy(record)
        print('REFERENCE_READOUT_EPOCH', epoch, 'CW_AUC_PCT', round(100*metrics['cw']['AUC'], 6),
              'OW_AUC_PCT', round(100*metrics['ow']['AUC'], 6), flush=True)
    last = tensor_load(source / 'runs/P_generic/last.pt')
    if best['epoch'] == 0:
        head_state = legacy_runtime.cpu_state(fresh_head(bank, 'G_K', controls, device))
    elif best['epoch'] == last['completed_epoch']:
        head_state = last['head_state']
    elif best['epoch'] == last['best']['epoch']:
        head_state = last['best_state']
    else:
        raise ValueError('Selected reference epoch has no archived weights; do not fabricate a checkpoint')
    head = fresh_head(bank, 'G_K', controls, device)
    head.load_state_dict(head_state, strict=True)
    replay, _, _ = scores_and_diagnostics(head, caches['val'], device)
    if not np.array_equal(replay.numpy(), np.load(child(run, best['path'] + '/grid_scores.npy'), allow_pickle=False)):
        raise ValueError('Selected mixed-readout reference checkpoint not exact')
    tensor_save(attempt / 'selected_reference.pt', dict(schema='generic_mixture_reused_checkpoint_v1',
                protocol_sha256=protocol_sha, variant='G_K', head_state=head_state, best=best,
                source_checkpoint_sha256=sha(source / 'runs/P_generic/last.pt'), new_optimizer_steps=0))
    write_json(attempt / 'history.json', dict(protocol_sha256=protocol_sha, ledger=ledger, best=best,
                                            new_optimizer_steps=0, original_optimizer_steps=experiment['source_generic_formal_steps']))
    files = {p.relative_to(run).as_posix(): sha(p) for p in attempt.rglob('*') if p.is_file()}
    result = dict(passed=True, variant='G_K', selected=best, best_epoch=best['epoch'],
                  protocol_sha256=protocol_sha, completed_epoch=settings['epochs'],
                  initial_state_sha256=experiment['source_initial_state_sha256'],
                  formal_optimizer_steps=0, original_source_optimizer_steps=experiment['source_generic_formal_steps'],
                  selected_checkpoint_replay_exact=True, source_reused=True, owned_files_sha256=files)
    write_json(complete, result)
    return result


def mechanics(work, caches, ids, bank, controls, settings, protocol_sha, device):
    complete = work / 'mechanics_complete.json'
    if complete.exists():
        result = read_json(complete)
        if not result['passed'] or result['protocol_sha256'] != protocol_sha:
            raise ValueError('Mechanics identity changed')
        verify_files({str(child(work, n)): h for n, h in result['files_sha256'].items()})
        print('GENERIC_MECHANICS_REUSED_VERIFIED', flush=True)
        return result
    folder = legacy_pilot.next_attempt(work / 'mechanics')
    fit, hold, pairs = legacy_runtime.train_fold(caches['train'], ids['train'], settings['train_holdout_fraction'])
    write_json(folder / 'fold.json', dict(fit_indices=fit.tolist(), hold_indices=hold.tolist(), pair_counts=pairs,
                                        base_already_trained_on_full_TRAIN=True, not_independent_validation=True))
    reports = {}
    for variant in NEW_VARIANTS:
        print('GENERIC_MECHANICS_BEGIN', variant, 'fit', len(fit), 'hold', len(hold), flush=True)
        head = fresh_head(bank, variant, controls, device)
        opt = legacy_runtime.optimizer(head, settings)
        _, before, _ = scores_and_diagnostics(head, caches['train'], device, hold)
        trained = legacy_runtime.train_epoch(head, opt, caches['train'], fit, 1, device, settings['batch_size'])
        if set(trained['nonzero_gradient_parameters']) != set(dict(head.named_parameters())):
            raise ValueError('Generic mechanics has inactive parameters')
        _, after, vector = scores_and_diagnostics(head, caches['train'], device, hold)
        if after['candidate_object_delta_std'] <= 1e-9:
            raise ValueError('Generic object-conditioned evidence not connected')
        np.savez_compressed(folder / (variant + '_hold_predictions.npz'), **vector)
        reports[variant] = dict(before=before, after=after, train_epoch=trained,
                                val_forward_calls=0, test_forward_calls=0, diagnostic_weights_discarded=True)
        write_json(folder / (variant + '.json'), reports[variant])
    write_json(folder / 'summary.json', dict(passed=True, reports=reports, no_hyperparameter_selection=True))
    result = dict(passed=True, protocol_sha256=protocol_sha,
                  actual_diagnostic_optimizer_steps=sum(r['train_epoch']['steps'] for r in reports.values()),
                  files_sha256={p.relative_to(work).as_posix(): sha(p) for p in folder.iterdir() if p.is_file()})
    write_json(complete, result)
    print('GENERIC_MECHANICS_COMPLETE: resetting every formal head', flush=True)
    return result


def decision(results, baseline, gate):
    selected = {k: v['selected']['metrics_raw'] for k, v in results.items()}
    correct = selected['G_K']
    def diff(right, setting='cw', field='AUC'):
        return 100*(correct[setting][field] - right[setting][field])
    d = dict(gk_minus_baseline_cw_auc_pp=diff(baseline),
             gk_minus_name_cw_auc_pp=diff(selected['G_name']),
             gk_minus_perm1_cw_auc_pp=diff(selected['G_perm1']),
             gk_minus_perm2_cw_auc_pp=diff(selected['G_perm2']),
             gk_minus_mean_permutations_cw_auc_pp=100*(correct['cw']['AUC']-.5*(selected['G_perm1']['cw']['AUC']+selected['G_perm2']['cw']['AUC'])),
             gk_minus_baseline_cw_hm_pp=diff(baseline, field='best_hm'),
             gk_minus_baseline_ow_auc_pp=diff(baseline, 'ow'),
             gk_minus_baseline_ow_best_unseen_pp=diff(baseline, 'ow', 'best_unseen'))
    conditions = dict(cw_baseline=d['gk_minus_baseline_cw_auc_pp'] >= gate['pk_minus_baseline_cw_auc_pp'],
                      cw_name=d['gk_minus_name_cw_auc_pp'] >= gate['pk_minus_name_cw_auc_pp'],
                      cw_mean_perm=d['gk_minus_mean_permutations_cw_auc_pp'] >= gate['pk_minus_mean_permutations_cw_auc_pp'],
                      cw_hm=d['gk_minus_baseline_cw_hm_pp'] >= gate['cw_hm_floor_pp'],
                      ow_auc=d['gk_minus_baseline_ow_auc_pp'] >= gate['ow_auc_floor_pp'],
                      ow_best_unseen=d['gk_minus_baseline_ow_best_unseen_pp'] >= gate['ow_best_unseen_floor_pp'])
    return dict(generic_budget_gate_passed=all(conditions.values()), conditions=conditions, deltas_pp=d,
                original_gates_unchanged=True, development_only_not_preregistered=True, not_statistical_significance=True)


def save_same_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Existing metadata changed')
    else:
        write_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    release_sha = verify_release()
    ref = read_json(ROOT / 'legacy/reference.json')
    exp = read_json(ROOT / 'experiment.json')
    work = guard_path(args.work, ref, exp)
    destination = guard_path(args.export, ref, exp, [work])
    for path in (destination, destination.with_name(destination.name+'.sha256'), destination.with_name(destination.name+'.partial')):
        if path.exists():
            raise FileExistsError('Preserve existing export; choose a fresh name')
    if args.resume:
        if not work.is_dir() or read_json(work / 'protocol.json').get('schema') != SCHEMA:
            raise ValueError('Resume requires this owned generic protocol')
    else:
        if work.exists():
            raise FileExistsError('Preserve existing generic work')
        existing = work.parent
        while not existing.exists():
            existing = existing.parent
        if shutil.disk_usage(existing).free < 2*1024**3:
            raise RuntimeError('Need at least 2 GiB free; nothing will be deleted')
        work.mkdir(parents=True)
    import fcntl
    lock = (work / 'controls.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        write_json(work / 'execution_status.json', dict(state='running', resume=args.resume))
        if not torch.cuda.is_available():
            raise RuntimeError('Actual run requires original CPF CUDA environment')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        print('VERIFYING_FIXED_INPUTS_AND_OLD_GENERIC: no image decoding or backbone loading', flush=True)
        verify_files(exp['source_files'])
        caches, ids, bank, datasets, evaluator = legacy_runtime.load_inputs(ref)
        env = legacy_runtime.environment()
        if env != exp['source_environment']:
            raise ValueError('Original CUDA/runtime environment changed; refusing unmatched reuse')
        save_same_json(work / 'environment.json', env)
        settings, controls = ref['pilot_settings'], ref['control_protocol']
        init = initializations(bank, controls)
        if any(x['initial_state_sha256'] != exp['source_initial_state_sha256'] for x in init.values()):
            raise ValueError('Generic initialization is not source-exact')
        protocol = dict(schema=SCHEMA, release_manifest_sha256=release_sha, experiment_sha256=sha(ROOT/'experiment.json'),
                        source_protocol_sha256=exp['source_protocol_sha256'], source_files=exp['source_files'],
                        settings=settings, controls=controls, initializations=init, alpha=ALPHA,
                        formal_variants=list(NEW_VARIANTS), reused_correct_variant='G_K',
                        expected_new_formal_optimizer_steps=2700, source_correct_optimizer_steps=900,
                        training_loss='unchanged_unmixed_conditional_CE', readout='fixed_arithmetic_probability_mixture',
                        budget_gate=ref['budget_gate'], work=str(work), original_gates_unchanged=True,
                        source_old_gate=exp['source_old_gate'], development_only_not_preregistered=True,
                        images_decoded=0, test_evaluation_allowed=False)
        save_same_json(work / 'protocol.json', protocol)
        protocol_sha = sha(work / 'protocol.json')
        device = torch.device('cuda:0')
        check = legacy_pilot.cuda_resume_check(bank, controls, caches['train'], settings, device)
        check['protocol_sha256'] = protocol_sha
        write_json(legacy_pilot.next_attempt(work / 'resume_selfchecks') / 'selfcheck.json', check)
        baseline = {s: legacy_runtime.native_metrics(caches['val']['scores'], caches['val']['attrs'], caches['val']['objects'], ds, evaluator)
                    for s, ds in datasets.items()}
        if any(abs(baseline[s][k]-ref['metrics'][s][k]) > 1e-10 for s in baseline for k in legacy_runtime.FIELDS):
            raise ValueError('Original baseline native metrics changed')
        save_same_json(work / 'baseline.json', dict(metrics_raw=baseline, protocol_sha256=protocol_sha))
        save_same_json(work / 'labels.json', ref['labels'])
        save_same_json(work / 'val_image_ids.json', ids['val'])
        for s, ds in datasets.items():
            save_same_json(work / ('candidates_'+s+'.json'), list(map(list, ds.pairs)))
        truth_path = work / 'val_truth.npz'
        if truth_path.exists():
            with np.load(truth_path, allow_pickle=False) as truth:
                if not np.array_equal(truth['attrs'], caches['val']['attrs'].numpy()) or not np.array_equal(truth['objects'], caches['val']['objects'].numpy()):
                    raise ValueError('VAL truth changed on resume')
        else:
            np.savez_compressed(truth_path, attrs=caches['val']['attrs'].numpy(), objects=caches['val']['objects'].numpy())
        replay_check = reference_checks(work, caches, bank, controls, settings, exp, device, protocol_sha)
        results = {'G_K': reuse_correct(work, caches, bank, controls, settings, exp, datasets, evaluator, device, protocol_sha)}
        mechanical = mechanics(work, caches, ids, bank, controls, settings, protocol_sha, device)
        for variant in NEW_VARIANTS:
            results[variant] = legacy_pilot.formal_variant(work, variant, caches, bank, controls, datasets, evaluator,
                                                          settings, protocol_sha, init[variant], device)
        expected = settings['epochs']*legacy_pilot.math_steps(len(caches['train']['features']), settings['batch_size'])
        for variant in NEW_VARIANTS:
            history = read_json(work / 'runs' / variant / 'history.json')['ledger']
            source_history = read_json(Path(exp['source_work']) / 'runs/P_generic/history.json')['ledger']
            if results[variant]['formal_optimizer_steps'] != expected or any(history[e]['train_epoch']['epoch_order_sha256'] != source_history[e]['train_epoch']['epoch_order_sha256'] for e in range(1, settings['epochs']+1)):
                raise ValueError('Formal control budget/data order not matched to source correct run')
        predictions = {}
        for variant, result in results.items():
            with np.load(child(work / 'runs' / variant, result['selected']['path']+'/conditional_predictions.npz'), allow_pickle=False) as data:
                predictions[variant] = {k: data[k].copy() for k in data.files}
        summary = dict(passed=True, protocol_sha256=protocol_sha, results=results, baseline_metrics_raw=baseline,
                       decision=decision(results, baseline, ref['budget_gate']), source_old_gate=exp['source_old_gate'],
                       gk_paired_val_conditional={v: legacy_runtime.paired_predictions(predictions['G_K'], predictions[v]) for v in NEW_VARIANTS},
                       new_formal_optimizer_steps=sum(x['formal_optimizer_steps'] for x in results.values()),
                       original_correct_formal_optimizer_steps=exp['source_generic_formal_steps'],
                       reference_reproduction_diagnostic_steps=replay_check['diagnostic_optimizer_steps'],
                       mechanics_diagnostic_steps=mechanical['actual_diagnostic_optimizer_steps'],
                       discarded_restore_check_steps_per_invocation=3, backbone_optimizer_steps=0,
                       image_decodes=0, test_evaluation=False)
        if summary['new_formal_optimizer_steps'] != 2700:
            raise ValueError('Unexpected new formal budget')
        verify_files(exp['source_files'])
        verify_files(ref['stageA_files'])
        verify_files(ref['c_files'])
        verify_release()
        write_json(work / 'summary.json', summary)
        write_json(work / 'execution_status.json', dict(state='complete', protocol_sha256=protocol_sha))
        print('GENERIC_CONTROLS_COMPLETE', 'new_formal_steps', summary['new_formal_optimizer_steps'],
              'GENERIC_GATE', summary['decision']['generic_budget_gate_passed'], flush=True)
        from export_controls import export_work
        export_work(work, destination)
    except Exception as error:
        failure = legacy_pilot.next_attempt(work / 'failures')
        write_json(failure / 'failure.json', dict(error=repr(error), originals_not_overwritten=True, success_not_claimed=True))
        write_json(work / 'execution_status.json', dict(state='failed', error=repr(error)))
        try:
            from export_controls import export_work
            export_work(work, destination)
        except Exception as export_error:
            print('FORENSIC_EXPORT_FAILED', repr(export_error), flush=True)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
