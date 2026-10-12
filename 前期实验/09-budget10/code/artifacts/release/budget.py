"""Import a reviewed epoch-3 run without resetting optimizer or RNG state."""
import csv
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile

import numpy as np
import torch


def within(api, root, relative):
    p = PurePosixPath(relative)
    api.require(not p.is_absolute() and '..' not in p.parts and '\\' not in relative,
                'Unsafe relative artifact path')
    target = root.joinpath(*p.parts)
    api.require(root.resolve() in target.resolve().parents, 'Artifact escapes pinned root')
    return target


def validate_history(api):
    ref = api.J(api.BUNDLE/'history_reference.json')
    old = Path(ref['old_output'])
    api.require(api.sha(old/'review_manifest.json') == ref['manifest_sha256'],
                'Reviewed v2 manifest changed or missing')
    manifest = api.J(old/'review_manifest.json')['files_sha256']
    api.require(manifest == ref['files_sha256'], 'Reviewed v2 file inventory changed')
    for relative, expected in manifest.items():
        source = within(api, old, relative)
        api.require(source.is_file() and api.sha(source) == expected,
                    'Reviewed v2 input changed: '+relative)
    protocol = api.J(old/'protocol.json')
    api.require(api.sha(old/'protocol.json') == ref['protocol_sha256'], 'Old v2 protocol changed')
    expected = dict(api.SETTINGS, epochs=3)
    api.require(protocol['settings'] == expected and protocol['variants'] == list(api.VARIANTS),
                'Extension must change only the epoch budget')
    api.require(protocol['release_sha256'] == ref['old_release_sha256'], 'Old v2 release changed')
    api.require(api.sha(api.BUNDLE/'core.py') == ref['core_sha256'], 'Router implementation changed')
    api.require(api.J(old/'reports/decision.json')['proceed'] is False,
                'Pinned old v2 decision changed')
    api.require(api.J(old/'reports/integrity.json')['passed'], 'Old v2 integrity did not pass')
    return old, ref


def verify_import(api, out):
    marker = out/'budget_import.json'
    # Recover the tiny commit marker if a process stopped after the atomic runs rename.
    if not marker.exists() and (out/'runs/_import_manifest.json').is_file():
        api.W(marker, api.J(out/'runs/_import_manifest.json'))
    api.require(marker.is_file(), 'Run the import phase before extending training')
    record = api.J(marker)
    api.require(record['passed'] and record['protocol_sha256'] == api.sha(out/'protocol.json'),
                'Imported state provenance mismatch')
    api.require(record['old_protocol_sha256'] == api.J(api.BUNDLE/'history_reference.json')['protocol_sha256'],
                'Import is not from the pinned reviewed v2 run')
    api.require(record['imported_epochs_each'] == 3 and record['target_epochs_each'] == 10,
                'Imported budget mismatch')
    for relative, expected in record['immutable_files_sha256'].items():
        api.require(api.sha(within(api, out, relative)) == expected, 'Imported evidence changed: '+relative)
    return record


def check_old_checkpoint(api, old, variant, last):
    run = old/'runs'/variant
    complete = api.J(run/'complete.json')
    history = api.J(api.BUNDLE/'history_reference.json')
    api.require(complete['completed_epochs'] == 3 and complete['variant'] == variant,
                'Expected a completed epoch-3 v2 run')
    api.require(last['epoch'] == 3 and last['variant'] == variant and
                last['protocol_sha256'] == history['protocol_sha256'] and
                last['release_sha256'] == history['old_release_sha256'], 'Old last_epoch identity mismatch')
    api.require(last['records'] == api.J(run/'committed_epochs.json') and
                [r['epoch'] for r in last['records']] == [1, 2, 3], 'Old committed records mismatch')
    api.require(last['best_epoch'] == complete['selected_epoch'], 'Old checkpoint selection mismatch')
    selected_path = within(api, run, last['best_checkpoint_relative'])
    api.require(api.sha(selected_path) == last['best_checkpoint_sha256'] == complete['best_checkpoint_sha256'],
                'Old selected checkpoint mismatch')
    selected = torch.load(str(selected_path), map_location='cpu')
    api.require(selected['epoch'] == last['best_epoch'] and selected['variant'] == variant and
                selected['protocol_sha256'] == last['protocol_sha256'] and
                selected['release_sha256'] == last['release_sha256'], 'Old selected state provenance mismatch')
    baseline_auc = api.J(old/'baseline/val_cw/metrics.json')['metrics_raw']['AUC']
    best, epoch = baseline_auc, 0
    for r in last['records']:
        auc = api.J(within(api, run, r['evaluation'])/'metrics.json')['metrics_raw']['AUC']
        api.require(abs(100*auc-r['cw_auc_pct']) <= 1e-10, 'Old epoch metric mismatch')
        if auc > best:
            best, epoch = auc, r['epoch']
    api.require(epoch == last['best_epoch'] and abs(best-last['best_auc']) <= 1e-12,
                'Old first-strict-maximum selection mismatch')
    api.require(set(last['rng']) == {'python','numpy','torch','cuda','loader'}, 'Incomplete RNG state')
    settings = api.J(run/'effective_settings.json')
    steps = int(math.ceil(history['training_images']/api.SETTINGS['batch_size']))*3
    states = last['optimizer']['state']
    api.require(len(states) == (6 if variant.endswith('_fixed') else 8), 'Incomplete optimizer state')
    api.require(all(float(v['step']) == steps for v in states.values()), 'Optimizer step count mismatch')
    for group in last['optimizer']['param_groups']:
        api.require(group['lr'] == api.SETTINGS['learning_rate'] and
                    group['weight_decay'] == api.SETTINGS['weight_decay'], 'Optimizer hyperparameters changed')
    api.require(settings['settings'] == dict(api.SETTINGS, epochs=3), 'Old settings differ beyond epoch count')
    return selected, settings


def replay_last(api, model, cfg, target, previous):
    from transition.evaluate import evaluate
    record = evaluate(model, cfg, 'val', 'cw', target, save_scores=True)
    old_record = api.J(previous/'metrics.json')
    shape = (old_record['n_images'], old_record['n_candidates'])
    api.require(shape == (record['n_images'], record['n_candidates']), 'Epoch-3 score shape changed')
    for name in ('candidates.json', 'image_ids.json'):
        api.require(api.J(target/name) == api.J(previous/name), 'Epoch-3 replay is not paired')
    with np.load(str(target/'labels.npz'), allow_pickle=False) as new, np.load(str(previous/'labels.npz'), allow_pickle=False) as old:
        api.require(all(np.array_equal(new[k], old[k]) for k in ('attr','obj','pair')), 'Epoch-3 labels changed')
    wanted_bytes = int(np.prod(shape))*4
    api.require((target/'scores.dat').stat().st_size == wanted_bytes and
                (previous/'scores.dat').stat().st_size == wanted_bytes, 'Epoch-3 score file size mismatch')
    new = np.memmap(str(target/'scores.dat'), mode='r', dtype=np.float32, shape=shape)
    old = np.memmap(str(previous/'scores.dat'), mode='r', dtype=np.float32, shape=shape)
    api.require(np.isfinite(new).all() and np.isfinite(old).all(), 'Nonfinite epoch-3 replay')
    difference = float(np.max(np.abs(new-old)))
    metric_difference = max(abs(100*(record['metrics_raw'][k]-old_record['metrics_raw'][k])) for k in api.METRICS)
    api.require(difference <= 1e-5 and metric_difference <= .05, 'Epoch-3 checkpoint replay differs from reviewed scores')
    return dict(max_score_difference=difference, max_metric_difference_pp=metric_difference,
                paired=True, score_tolerance=1e-5, metric_tolerance_pp=.05)


def import_previous(api, work, out, reference):
    old, history = validate_history(api)
    api.require(old.resolve() != out.resolve(), 'Never import over the old result')
    if (out/'runs').exists():
        verify_import(api, out)
        print('VERIFIED EXISTING BUDGET IMPORT', flush=True)
        return
    api.require(api.J(out/'prepare_checks.json')['passed'], 'Preparation must pass before import')
    stage = out/('import_staging_%s'%time.time_ns())
    staged_runs = stage/'runs'
    staged_runs.mkdir(parents=True)
    protocol_hash = api.sha(out/'protocol.json')
    rows = []
    for variant in api.VARIANTS:
        old_run = old/'runs'/variant
        old_last = old_run/'checkpoints/last_epoch.pth'
        digest_before = api.sha(old_last)
        last = torch.load(str(old_last), map_location='cpu')
        selected, old_settings = check_old_checkpoint(api, old, variant, last)
        model, ds, original = api.make_model(reference, variant)
        api.require(len(ds) == history['training_images'], 'Training dataset size changed')
        api.require(api.frozen_state(model) == api.J(out/'prepare_checks.json')['frozen_state'], 'Frozen anchor changed')
        model.router.load_state_dict(last['router_state'], strict=True)
        optimizer = torch.optim.AdamW([p for p in model.router.parameters() if p.requires_grad],
                                     lr=api.SETTINGS['learning_rate'], weight_decay=api.SETTINGS['weight_decay'])
        optimizer.load_state_dict(last['optimizer'])
        generator = torch.Generator().manual_seed(api.SETTINGS['seed'])
        api.restore_rng(last['rng'], generator)
        new_run = staged_runs/variant
        new_run.mkdir()
        cfg = api.make_cfg(original, out/'runs'/variant, variant)
        replay_folder = api.next_attempt(out, 'import_replay/'+variant)
        previous = within(api, old_run, last['records'][-1]['evaluation'])
        replay = replay_last(api, model, cfg, replay_folder, previous)
        api.require(api.sha(old_last) == digest_before, 'Old checkpoint changed during replay')
        for r in last['records']:
            prefix = 'runs/'+variant+'/'+r['evaluation']+'/'
            for relative, expected in history['files_sha256'].items():
                if relative.startswith(prefix):
                    src = within(api, old, relative)
                    dst = within(api, new_run, relative[len('runs/'+variant+'/'):])
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    api.require(api.sha(dst) == expected, 'Copied old epoch artifact differs')
        shutil.copy2(old_run/'source_anchor_config.yml', new_run/'source_anchor_config.yml')
        settings = dict(old_settings, settings=dict(api.SETTINGS), imported_epochs=3,
                        old_protocol_sha256=history['protocol_sha256'])
        api.W(new_run/'effective_settings.json', settings)
        selected = dict(selected, protocol_sha256=protocol_hash, release_sha256=api.release_id(),
                        imported_from_protocol_sha256=history['protocol_sha256'],
                        imported_checkpoint_sha256=last['best_checkpoint_sha256'])
        best = new_run/'checkpoints'/('selected_epoch_%02d_imported.pth'%last['best_epoch'])
        api.save_checkpoint(best, selected)
        imported = dict(last, protocol_sha256=protocol_hash, release_sha256=api.release_id(),
                        best_checkpoint_relative=best.relative_to(new_run).as_posix(), best_checkpoint_sha256=api.sha(best),
                        imported_from_protocol_sha256=history['protocol_sha256'],
                        imported_checkpoint_sha256=digest_before)
        saved = new_run/'checkpoints/imported_epoch_03.pth'
        api.save_checkpoint(saved, imported)
        shutil.copy2(saved, new_run/'checkpoints/last_epoch.pth')
        api.W(new_run/'committed_epochs.json', last['records'])
        api.W(new_run/'imported_epochs_01_03.json', last['records'])
        rows.append(dict(variant=variant, imported_epoch=3, selected_epoch_before_extension=last['best_epoch'],
                         old_last_checkpoint_sha256=digest_before, old_selected_checkpoint_sha256=last['best_checkpoint_sha256'],
                         optimizer_and_rng_preserved=True,
                         optimizer_steps_each_parameter=int(math.ceil(history['training_images']/api.SETTINGS['batch_size']))*3,
                         epoch3_replay=replay))
        del model, optimizer
        torch.cuda.empty_cache()
    validate_history(api)
    immutable = {}
    for p in sorted(staged_runs.rglob('*')):
        if p.is_file() and p.name not in ('last_epoch.pth','committed_epochs.json'):
            immutable['runs/'+str(p.relative_to(staged_runs)).replace('\\','/')] = api.sha(p)
    record = dict(passed=True, protocol_sha256=protocol_hash, old_protocol_sha256=history['protocol_sha256'],
                  old_output=str(old), imported_epochs_each=3, target_epochs_each=10,
                  variants=rows, immutable_files_sha256=immutable,
                  old_results_modified=False, import_training_performed=False)
    api.W(staged_runs/'_import_manifest.json', record)
    os.replace(str(staged_runs), str(out/'runs'))
    api.W(out/'budget_import.json', record)
    api.W(out/'status.json', dict(status='epoch3_import_complete', next_epoch=4))
    verify_import(api, out)
    print('BUDGET IMPORT PASSED: FOUR EPOCH-3 STATES; NEXT EPOCH 4; TARGET EPOCH 10', flush=True)


def write_budget_comparison(api, out):
    old, history = validate_history(api)
    curves, comparisons = [], []
    for variant in api.VARIANTS:
        run = out/'runs'/variant
        records = api.J(run/'committed_epochs.json')
        api.require([r['epoch'] for r in records] == list(range(1,11)), 'Final report requires ten committed epochs')
        prior_elapsed = 0
        for r in records:
            metric = api.J(within(api,run,r['evaluation'])/'metrics.json')
            elapsed = r.get('elapsed_epoch_seconds', r['elapsed_attempt_seconds']-prior_elapsed)
            curves.append(dict(variant=variant,epoch=r['epoch'],loss_mean=r['loss_mean'],
                               cw_auc_pct=r['cw_auc_pct'],cw_hm_pct=r['cw_hm_pct'],eta=r['eta'],
                               value_drift_mean=metric['diagnostics']['value_drift_mean'],
                               imported_old_epoch=r['epoch']<=3,elapsed_epoch_seconds=elapsed))
            prior_elapsed=r['elapsed_attempt_seconds']
        new_complete=api.J(run/'complete.json')
        old_complete=api.J(old/'runs'/variant/'complete.json')
        for setting in ('cw','ow'):
            new=api.J(run/new_complete['final_eval_folders'][setting]/'metrics.json')['metrics_raw']
            prior=api.J(old/'runs'/variant/old_complete['final_eval_folders'][setting]/'metrics.json')['metrics_raw']
            comparisons.append(dict(variant=variant,setting=setting,old_selected_epoch=old_complete['selected_epoch'],
                                    new_selected_epoch=new_complete['selected_epoch'],old_budget=3,new_budget=10,
                                    old_auc_pct=100*prior['AUC'],new_auc_pct=100*new['AUC'],
                                    delta_auc_pp=100*(new['AUC']-prior['AUC']),
                                    old_hm_pct=100*prior['best_hm'],new_hm_pct=100*new['best_hm']))
    for name,rows in [('epoch_curves.csv',curves),('budget3_vs_budget10.csv',comparisons)]:
        with (out/'reports'/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    api.W(out/'reports/budget_scope.json',dict(old_protocol_sha256=history['protocol_sha256'],
          epochs_extended_from=3,epochs_extended_to=10,seed=0,old_v2_proceed=False,
          models_and_hyperparameters_unchanged=True,independent_confirmation=False,
          old_static_CPF_results_not_retrained=True,original_gates_not_modified=True,
          convergence_not_assumed_from_epoch_count=True))


def verify_existing_review(api, out, archive):
    wanted={p.relative_to(out).as_posix():api.sha(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.suffix not in ('.pth','.pt','.tmp') and p.name!='review_manifest.json'}
    with zipfile.ZipFile(archive) as z:
        api.require(z.testzip() is None, 'Existing review ZIP CRC failed; preserve it')
        manifest=__import__('json').loads(z.read('review_manifest.json'))
        api.require(manifest['files_sha256']==wanted and manifest['weights_included'] is False,
                    'Existing review ZIP differs from current artifacts; preserve both')
        api.require(len(z.namelist())==len(set(z.namelist())) and
                    set(z.namelist())==set(wanted)|{'review_manifest.json'}, 'Existing review ZIP inventory differs')
        for name,expected in wanted.items():
            api.require(__import__('hashlib').sha256(z.read(name)).hexdigest()==expected,
                        'Existing review ZIP member changed: '+name)
    print('VERIFIED EXISTING BUDGET10 REVIEW ZIP',archive,'SHA256',api.sha(archive),flush=True)
