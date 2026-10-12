"""Independent, bounded UT name-anchor EKR exploration. VAL only."""
import argparse
import copy
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import code_manifest, load_bank, read_json, sha, write_json
from .runtime import base_model, read_cfg, trainset
from .evaluate import evaluate

SEEDS = (0, 3351, 2026)
VARIANTS = {'N_name': 'name_only', 'N_clean': 'ekr', 'N_shuffle': 'shuffle',
            'N_noobj': 'no_object', 'N_nonull': 'no_null'}
PRIMARY = ('N_name', 'N_clean', 'N_shuffle')
METRICS = ('AUC', 'best_hm', 'best_seen', 'best_unseen', 'hm_seen', 'hm_unseen')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_write(path, rows):
    if not rows:
        raise ValueError('Refuse an empty diagnostic table')
    keys = list(rows[0])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def rid(variant, seed):
    return 'name_ut-zappos_{}_s{}_e3'.format(variant, seed)


def protocol_path(work):
    return Path(work) / 'protocol/name_exploration.json'


def old_gate(work):
    return Path(work) / 'experiments/semantic_v2/protocol/semantic_gate_v2.json'


def prepare(args):
    work, previous = Path(args.work).resolve(), Path(args.previous).resolve()
    project = Path.cwd().resolve()
    require(project == work / 'src', 'Run this command from the new work/src directory')
    require(work != previous and previous not in work.parents and work not in previous.parents,
            'Exploration work must be separate from UT v2')
    ref = read_json(project / 'name_reference.json')
    inputs = {previous / 'prepared.json': ref['prepared_json_sha256'],
              old_gate(previous): ref['original_gate_sha256'],
              previous / 'bank/ut-zap50k/manifest.json': ref['bank_manifest_sha256'],
              previous / 'cache/ut-zap50k/cache_manifest.json': ref['cache_manifest_sha256']}
    for name, digest in ref['source_code_manifest'].items():
        inputs[previous / 'src' / name] = digest
    for seed in SEEDS:
        for variant in ('A_unit', 'C_text', 'C_shuffle'):
            name = 'semantic_ut-zappos_{}_s{}_e10'.format(variant, seed)
            cfg = previous / 'experiments/semantic_v2/configs' / (name + '.yml')
            inputs[cfg] = ref['configs_sha256'][name]
            values = yaml.safe_load(cfg.read_text(encoding='utf-8'))
            require(values['TRAIN']['seed'] == seed and values['TRAIN']['max_epoch'] == 10,
                    'Original training seed/budget changed')
            require(values['MODEL']['semantic_text'] == {'A_unit':'unit', 'C_text':'correct',
                    'C_shuffle':'shuffle'}[variant], 'Original semantic input changed')
            root = Path(values['DATASET']['root_dir'])
            for setting in ('cw', 'ow'):
                expected = ref['metrics'][name + '/' + setting]
                m = previous / 'experiments/semantic_v2/runs' / name / ('eval_val_' + setting) / 'metrics.json'
                actual = read_json(m)
                require(actual == expected, 'Original metric record differs from the downloaded review')
                inputs[m] = sha(m)
                for leaf in ('labels.npz','image_ids.json','candidates.json'):
                    p=m.parent/leaf
                    inputs[p]=sha(p)
            checkpoint = previous / 'experiments/semantic_v2/runs' / name / 'checkpoints/best_auc.pth'
            inputs[checkpoint] = expected['checkpoint_sha256']
    for path, digest in inputs.items():
        require(sha(path) == digest, 'Frozen input changed: ' + str(path))
    train_path = root / 'compositional-split-natural/train_pairs.txt'
    val_path = root / 'compositional-split-natural/val_pairs.txt'
    require(sha(train_path) == 'f0d134c668661180ed9464083898e5d7c8631aaa81cc01a061d5bac61aa8340f', 'TRAIN split changed')
    require(sha(val_path) == '26370ddbe6a0eb49f2f5059dffc5d25ea34f6c8bcf2c1040ec7b1e77a3dc3f3c', 'VAL split changed')
    inputs[train_path], inputs[val_path] = sha(train_path), sha(val_path)
    require(not read_json(old_gate(previous))['datasets']['ut-zappos']['proceed'],
            'This protocol documents the reviewed failed original gate')
    bank, _ = load_bank(previous / 'bank/ut-zap50k')
    require(tuple(bank['a']['units'].shape) == (16, 4, 300), 'Unexpected attribute bank')
    require(tuple(bank['o']['unit'].shape) == (12, 300), 'Unexpected object names')
    value = {'version': 'name_anchor_ekr_exploration_20261007',
        'scientific_status': 'new exploratory protocol; reused VAL development evidence',
        'original_gate_passed': False, 'previous': str(previous), 'work': str(work),
        'data_root': str(root), 'reference_sha256': sha(project / 'name_reference.json'),
        'code_manifest': code_manifest(project),
        'frozen_inputs_sha256': {str(k): v for k, v in inputs.items()},
        'anchor': 'same-seed A_unit, existing CW-best 10-epoch checkpoint',
        'knowledge': 'frozen v2 attribute slots; object context always unit names; K off',
        'training': {'seeds': list(SEEDS), 'branch_epochs': 3, 'batch_size': 64,
            'lr': 0.001, 'weight_decay': 0.0001, 'rank': 32, 'eta_max': 0.1,
            'temperature': 0.2, 'frozen_base': True, 'fallback_beta': 0.0},
        'selection': 'each branch first strict maximum VAL CW AUC, same 3-epoch budget',
        'budget': {'seed0_primary_trainings': 3, 'three_seed_primary_total_trainings': 9,
            'conditional_ablations': 6, 'no_automatic_joint_training': True},
        'replay_metric_tolerance_pp': 0.05,
        'pilot_threshold_pp': 0.2, 'confirmation_threshold_pp': 0.5,
        'guards_pp': {'cw_hm': -0.5, 'ow_auc': -0.5, 'ow_best_unseen': -2.0},
        'no_test_evaluation': True, 'no_val_description_edits': True}
    target = protocol_path(work)
    if target.exists():
        require(read_json(target) == value, 'Protocol differs; preserve this work and version a new experiment')
    else:
        write_json(target, value)
    exp = work / 'experiments/name_ekr'
    index = {}
    for seed in SEEDS:
        name = 'semantic_ut-zappos_A_unit_s{}_e10'.format(seed)
        anchor_cfg = previous / 'experiments/semantic_v2/configs' / (name + '.yml')
        anchor = previous / 'experiments/semantic_v2/runs' / name / 'checkpoints/best_auc.pth'
        template = yaml.safe_load(anchor_cfg.read_text(encoding='utf-8'))
        for variant, mode in VARIANTS.items():
            c = copy.deepcopy(template)
            run_id = rid(variant, seed)
            run = exp / 'runs' / run_id
            c['MODEL'].update(semantic_text='unit', kcc_text=True, kcc_calib=False,
                kcc_lambda=0.0, kcc_feature=False, kcc_logic=False, kcc_control='none',
                branch_mode=mode, branch_rank=32, branch_eta_max=0.1,
                branch_temperature=0.2, branch_object_chunk=16, branch_regime='frozen',
                branch_anchor=str(anchor), branch_anchor_sha256=sha(anchor),
                branch_anchor_config=str(anchor_cfg), branch_anchor_seed=seed)
            c['TRAIN'].update(seed=seed, max_epoch=3, final_max_epoch=3, start_epoch=1,
                kcc_run_dir=str(run), checkpoint_dir=str(run / 'checkpoints'),
                log_dir=str(run / 'logs'), train_name=run_id, kcc_stage='name_exploration',
                kcc_variant=variant, kcc_test_at_end=False, branch_lr=1e-3,
                branch_weight_decay=1e-4, branch_deterministic=True,
                branch_fallback_beta=0.0, branch_smoke_only=False,
                name_protocol_file=str(target), name_protocol_sha256=sha(target))
            path = exp / 'configs' / (run_id + '.yml')
            text = yaml.safe_dump(c, allow_unicode=True, sort_keys=True)
            if path.exists():
                require(path.read_text(encoding='utf-8') == text, 'Existing config differs: ' + str(path))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('w', encoding='utf-8', newline='\n') as f:
                    f.write(text)
            index[run_id] = {'config': str(path), 'sha256': sha(path)}
    index_path = work / 'protocol/config_index.json'
    if index_path.exists():
        require(read_json(index_path) == index, 'Config index differs')
    else:
        write_json(index_path, index)
    print('NAME PROTOCOL PREPARED; OLD GATE REMAINS FALSE; 15 CONFIGS FROZEN', target, flush=True)


def validate_protocol(work, full=False):
    value = read_json(protocol_path(work))
    project = Path(value['work']) / 'src'
    require(value['version'] == 'name_anchor_ekr_exploration_20261007' and
            value['original_gate_passed'] is False, 'Wrong exploration identity')
    require(value['code_manifest'] == code_manifest(project), 'Exploration code changed after protocol freeze')
    require(sha(project / 'name_reference.json') == value['reference_sha256'], 'References changed')
    ref = read_json(project / 'name_reference.json')
    require(sha(old_gate(value['previous'])) == ref['original_gate_sha256'], 'Original gate changed')
    if full:
        for path, digest in value['frozen_inputs_sha256'].items():
            require(sha(path) == digest, 'Original input changed: ' + path)
        for entry in read_json(Path(work) / 'protocol/config_index.json').values():
            require(sha(entry['config']) == entry['sha256'], 'Exploration config changed')
    return value


def validate_config(cfg, require_replay=False):
    p = Path(cfg.TRAIN.name_protocol_file)
    require(sha(p) == cfg.TRAIN.name_protocol_sha256, 'Exploration protocol changed')
    value = validate_protocol(p.parent.parent)
    work, previous = Path(value['work']), Path(value['previous'])
    variant, seed = cfg.TRAIN.kcc_variant, int(cfg.TRAIN.seed)
    require(variant in VARIANTS and seed in SEEDS, 'Undeclared variant/seed')
    require(cfg.MODEL.branch_mode == VARIANTS[variant] and cfg.MODEL.branch_regime == 'frozen', 'Branch identity changed')
    require(cfg.MODEL.semantic_text == 'unit' and cfg.MODEL.kcc_text and
            not cfg.MODEL.kcc_calib and cfg.MODEL.kcc_lambda == 0, 'Name anchor or K policy changed')
    name = 'semantic_ut-zappos_A_unit_s{}_e10'.format(seed)
    checkpoint = previous / 'experiments/semantic_v2/runs' / name / 'checkpoints/best_auc.pth'
    require(Path(cfg.MODEL.branch_anchor).resolve() == checkpoint.resolve(), 'Wrong same-seed anchor')
    require(sha(checkpoint) == cfg.MODEL.branch_anchor_sha256 ==
            value['frozen_inputs_sha256'][str(checkpoint)], 'Anchor changed')
    entry = read_json(work / 'protocol/config_index.json')[rid(variant, seed)]
    require(sha(entry['config']) == entry['sha256'], 'Frozen config changed')
    # Compare effective values, not YAML key ordering or scalar print formatting.
    require(cfg.dump() == read_cfg(entry['config']).dump(), 'Effective config differs from frozen config')
    load_bank(cfg.MODEL.description_bank)
    require(sha(Path(cfg.MODEL.description_bank) / 'manifest.json') ==
            read_json(work / 'src/name_reference.json')['bank_manifest_sha256'], 'Bank changed')
    if require_replay:
        result = read_json(work / 'diagnostics/full_scores/replay_audit.json')
        require(result['passed'] and result['protocol_sha256'] == sha(p), 'Full-score replay audit must pass first')
        require(result['inputs_sha256'] == score_inputs(work), 'Full-score diagnostic input changed')
    return value


def replay(args):
    value = validate_protocol(args.work, full=True)
    previous, work = Path(value['previous']), Path(value['work'])
    output = work / 'diagnostics/full_scores'
    for seed in SEEDS:
        for variant in ('A_unit', 'C_text', 'C_shuffle'):
            name = 'semantic_ut-zappos_{}_s{}_e10'.format(variant, seed)
            cfg_path = previous / 'experiments/semantic_v2/configs' / (name + '.yml')
            cfg = read_cfg(cfg_path)
            checkpoint = previous / 'experiments/semantic_v2/runs' / name / 'checkpoints/best_auc.pth'
            model = None
            for setting in ('cw', 'ow'):
                out = output / name / setting
                if (out / 'metrics.json').exists():
                    require((out / 'scores.dat').is_file(), 'Complete metrics without full scores')
                    print('VERIFY EXISTING REPLAY', name, setting, flush=True)
                    continue
                if out.exists() and any(out.iterdir()):
                    raise FileExistsError('Partial replay; preserve it before rerunning: ' + str(out))
                if model is None:
                    model = base_model(trainset(cfg), cfg, checkpoint)
                    model.evaluation_checkpoint_sha256 = sha(checkpoint)
                evaluate(model, cfg, 'val', setting, out, save_scores=True)
            del model
            torch.cuda.empty_cache()
    audit_scores(argparse.Namespace(work=str(work)))


def score_inputs(work):
    out = Path(work) / 'diagnostics/full_scores'
    return {str(p): sha(p) for p in sorted(out.glob('semantic_*/*/*'))
            if p.is_file() and p.name in {'scores.dat','metrics.json','candidates.json',
                                        'labels.npz','image_ids.json','config_eval_effective.yml'}}


def pair_set(path):
    return {tuple(line.split()) for line in Path(path).read_text().splitlines() if line.strip()}


def audit_scores(args):
    value = validate_protocol(args.work, full=True)
    work, previous = Path(value['work']), Path(value['previous'])
    ref = read_json(work / 'src/name_reference.json')
    out = work / 'diagnostics/full_scores'
    train = pair_set(Path(value['data_root']) / 'compositional-split-natural/train_pairs.txt')
    val = pair_set(Path(value['data_root']) / 'compositional-split-natural/val_pairs.txt')
    metric_rows, grid_rows, group_rows = [], [], []
    canonical_ids, canonical_gold, candidates_by_setting = None, None, {}
    max_pp = 0.0
    for seed in SEEDS:
        for variant in ('A_unit', 'C_text', 'C_shuffle'):
            name = 'semantic_ut-zappos_{}_s{}_e10'.format(variant, seed)
            common_bias = ref['metrics']['semantic_ut-zappos_A_unit_s{}_e10/ow'.format(seed)]['metrics_raw']['biasterm']
            for setting in ('cw', 'ow'):
                folder = out / name / setting
                m = read_json(folder / 'metrics.json')
                expected = ref['metrics'][name + '/' + setting]
                require(m['checkpoint_sha256'] == expected['checkpoint_sha256'] and
                        m['config_sha256'] == expected['config_sha256'] and
                        m['bank_manifest_sha256'] == expected['bank_manifest_sha256'], 'Replay identity mismatch')
                require(m['phase'] == 'val' and m['scores_saved'] and m['score_shape'] == expected['score_shape'], 'Replay shape/phase mismatch')
                pairs = [tuple(p) for p in read_json(folder / 'candidates.json')]
                ids = read_json(folder / 'image_ids.json')
                require(len(ids) == 3214 and len(set(ids)) == 3214, 'VAL identity population changed')
                labels = np.load(str(folder / 'labels.npz'), allow_pickle=False)
                gold = labels['pair']; gold_pairs = [pairs[int(i)] for i in gold]
                original = previous / 'experiments/semantic_v2/runs' / name / ('eval_val_' + setting)
                require(ids == read_json(original/'image_ids.json') and
                        pairs == [tuple(p) for p in read_json(original/'candidates.json')],
                        'Replay identity/candidate order differs from the frozen original')
                with np.load(str(original/'labels.npz'),allow_pickle=False) as old:
                    for key in ('attr','obj','pair'):
                        require(np.array_equal(labels[key],old[key]),'Replay gold differs from the frozen original')
                if canonical_ids is None:
                    canonical_ids, canonical_gold = ids, gold_pairs
                require(ids == canonical_ids and gold_pairs == canonical_gold, 'Replay samples are not paired')
                require(candidates_by_setting.setdefault(setting, pairs) == pairs, 'Candidate order differs')
                n, nc = m['score_shape']
                require((folder / 'scores.dat').stat().st_size == 4 * n * nc, 'Wrong full-score byte size')
                scores = np.memmap(str(folder / 'scores.dat'), mode='r', dtype=np.float32, shape=(n, nc))
                require(np.isfinite(scores).all(), 'Nonfinite scores')
                require(np.array_equal(scores.argmax(1), labels['top1_at_bias0']), 'Saved raw argmax mismatch')
                for key in METRICS:
                    delta = 100 * (m['metrics_raw'][key] - expected['metrics_raw'][key])
                    max_pp = max(max_pp, abs(delta))
                    metric_rows.append({'seed': seed, 'variant': variant, 'setting': setting,
                        'metric': key, 'original_pct': 100 * expected['metrics_raw'][key],
                        'replay_pct': 100 * m['metrics_raw'][key], 'delta_pp': delta})
                candidate_seen = np.array([p in train for p in pairs])
                allowed = np.array([True if setting == 'ow' else p in train | val for p in pairs])
                truth_seen = np.array([p in train for p in gold_pairs])
                object_names = sorted({p[1] for p in gold_pairs})
                object_gold = np.array([p[1] for p in gold_pairs])
                object_pred_map = np.array([p[1] for p in pairs])
                for bias in np.linspace(-1.0, 1.0, 81):
                    adjusted = np.array(scores, copy=True)
                    adjusted[:, ~candidate_seen] += np.float32(bias)
                    adjusted[:, ~allowed] = -np.inf
                    prediction = adjusted.argmax(1)
                    correct = prediction == gold
                    s = float(correct[truth_seen].mean()); u = float(correct[~truth_seen].mean())
                    grid_rows.append({'seed':seed,'variant':variant,'setting':setting,'bias':float(bias),
                        'seen_pct':100*s,'unseen_pct':100*u,'hm_pct':100*(2*s*u/(s+u) if s+u else 0),
                        'pair_pct':100*float(correct.mean()),
                        'object_pct':100*float((object_pred_map[prediction]==object_gold).mean())})
                # Common comparison points: zero and same-seed A_unit OW-best-HM bias.
                # The grid and these reference points are diagnostic; neither selects a new checkpoint.
                for point, bias in [('bias0',0.0), ('Aunit_OW_HM_bias',common_bias)]:
                    adjusted = np.array(scores, copy=True)
                    adjusted[:, ~candidate_seen] += np.float32(bias)
                    adjusted[:, ~allowed] = -np.inf
                    prediction = adjusted.argmax(1)
                    for obj in object_names:
                        mask = object_gold == obj
                        group_rows.append({'seed':seed,'variant':variant,'setting':setting,
                            'point':point,'bias':float(bias),'object':obj,'n':int(mask.sum()),
                            'pair_pct':100*float((prediction[mask]==gold[mask]).mean()),
                            'object_pct':100*float((object_pred_map[prediction[mask]]==obj).mean())})
                del scores
                labels.close()
    csv_write(out / 'replay_metric_deltas.csv', metric_rows)
    csv_write(out / 'fixed_bias_grid.csv', grid_rows)
    csv_write(out / 'object_groups_common_bias.csv', group_rows)
    passed = max_pp <= value['replay_metric_tolerance_pp']
    audit = {'passed':passed,'protocol_sha256':sha(protocol_path(work)),
        'evaluations':18,'n_images':3214,'labels_and_candidate_order_paired':True,
        'max_metric_replay_difference_pp':max_pp,
        'tolerance_pp':value['replay_metric_tolerance_pp'], 'inputs_sha256':score_inputs(work),
        'interpretation':'Artifact replay check only; original research gate remains false; no new model result',
        'common_bias_source':'same-seed A_unit OW best-HM bias from existing VAL metrics',
        'grid_is_native_auc':False}
    write_json(out / 'replay_audit.json', audit)
    (out / 'README.md').write_text(
        '# Frozen full-score replay\n\n'
        '18 VAL inferences, no training or test evaluation.\n\n'
        'replay_metric_deltas.csv checks native metrics; fixed_bias_grid.csv is a diagnostic common grid, '
        'not a replacement AUC. object_groups_common_bias.csv compares each group at bias=0 and the '
        'same-seed A_unit OW best-HM bias. These VAL points are development diagnostics.\n\n'
        'Original semantic gate remains false. Replay integrity does not establish semantic attribution.\n', encoding='utf-8')
    print('FULL SCORE REPLAY PASSED' if passed else 'STOP: REPLAY METRICS DIFFER', max_pp, 'pp', flush=True)
    require(passed, 'Resolve replay differences before training the exploration')


def run_plan(args):
    value = validate_protocol(args.work, full=True)
    work = Path(value['work'])
    require(tuple(args.seeds) in {(0,), (3351,2026), SEEDS}, 'Use the declared seed schedule')
    require(all(v in VARIANTS for v in args.variants), 'Undeclared branch variant')
    if tuple(args.seeds) != (0,) or any(v not in PRIMARY for v in args.variants):
        stage = 'confirm' if any(v not in PRIMARY for v in args.variants) else 'pilot'
        gate = read_json(work / 'reports' / stage / 'decision.json')
        require(gate['proceed'] and gate['protocol_sha256'] == sha(protocol_path(work)), 'Preceding exploratory screen did not pass')
        require(gate['inputs_sha256'] == summary_inputs(work, stage), 'Preceding screen evidence changed')
    from .name_run import branch
    for seed in args.seeds:
        for variant in args.variants:
            name = rid(variant, seed)
            entry = read_json(work / 'protocol/config_index.json')[name]
            cfg = read_cfg(entry['config'])
            validate_config(cfg, require_replay=True)
            smoke_path = work / 'smoke' / (name + '.json')
            if not smoke_path.exists():
                from .name_smoke import smoke
                smoke(entry['config'], smoke_path)
            else:
                sm = read_json(smoke_path)
                require(sm['passed'] and sm['config_sha256']==sha(entry['config']) and
                        sm['checkpoint_sha256']==sha(cfg.MODEL.branch_anchor), 'Smoke identity differs')
            run = Path(cfg.TRAIN.kcc_run_dir)
            finish = (run / 'train_complete.json').is_file()
            if not finish and run.exists() and any(run.iterdir()):
                raise FileExistsError('Partial training cannot resume; preserve it and version the run: '+str(run))
            print('VERIFY / FINISH' if finish else 'START', name, flush=True)
            branch(argparse.Namespace(cfg=entry['config'], finish_only=finish))
            torch.cuda.empty_cache()
    print('NAME RUN PLAN COMPLETE', flush=True)


def summary_inputs(work, stage):
    seeds = (0,) if stage == 'pilot' else SEEDS
    result = {str(protocol_path(work)):sha(protocol_path(work))}
    for seed in seeds:
        for variant in PRIMARY:
            run = Path(work)/'experiments/name_ekr/runs'/rid(variant,seed)
            names = ['effective_config.yml','train_complete.json','meta/code_manifest.json',
                'meta/branch_identity.json','status.json','selected_state.json','checkpoints/best_branch.pth']
            names += ['{}/metrics.json'.format(prefix+setting) for prefix in ('eval_val_','anchor_val_') for setting in ('cw','ow')]
            for name in names:
                result[str(run/name)] = sha(run/name)
    return result


def summary(args):
    value = validate_protocol(args.work, full=True)
    work = Path(value['work'])
    stage = args.stage
    seeds = (0,) if stage == 'pilot' else SEEDS
    rows, paired, values = [], [], {}
    capacity = None
    for seed in seeds:
        for variant in PRIMARY:
            run = work / 'experiments/name_ekr/runs' / rid(variant,seed)
            require(read_json(run/'status.json')['status']=='train_and_val_complete', 'Run not complete')
            cfg = read_cfg(run/'effective_config.yml')
            validate_config(cfg,require_replay=True)
            require(read_json(run/'meta/code_manifest.json')==value['code_manifest'], 'Training code mismatch')
            complete = read_json(run/'train_complete.json')
            require(complete['base_state_unchanged'] and complete['base_state_start']==complete['base_state_end'], 'Anchor changed')
            require(sha(run/'checkpoints/best_branch.pth')==complete['checkpoint_sha256'], 'Branch checkpoint changed')
            identity = read_json(run/'meta/branch_identity.json')
            require(identity['base_trainable_params']==0, 'Anchor was trained')
            count = identity['new_trainable_params']
            capacity = count if capacity is None else capacity
            require(count == capacity, 'Controls have unequal parameter counts')
            for setting in ('cw','ow'):
                m = read_json(run/('eval_val_'+setting)/'metrics.json')
                a = read_json(run/('anchor_val_'+setting)/'metrics.json')
                require(m['phase']=='val' and m['seed']==seed and m['variant']==variant and
                    m['checkpoint_sha256']==complete['checkpoint_sha256'] and m['anchor_sha256']==cfg.MODEL.branch_anchor_sha256 and
                    m['bank_manifest_sha256']==read_json(work/'src/name_reference.json')['bank_manifest_sha256'] and
                    not m['force_off'] and a['force_off'], 'Evaluation identity/off state mismatch')
                baseline = read_json(work/'diagnostics/full_scores'/
                    'semantic_ut-zappos_A_unit_s{}_e10'.format(seed)/setting/'metrics.json')
                for key in METRICS:
                    require(abs(100*(a['metrics_raw'][key]-baseline['metrics_raw'][key])) <=
                        value['replay_metric_tolerance_pp'], 'Force-off does not reproduce replayed A_unit')
                values[(seed,variant,setting)] = m['metrics_raw']
                values[(seed,'N0',setting)] = a['metrics_raw']
                rows.append({'seed':seed,'variant':variant,'setting':setting,
                    'auc_pct':100*m['metrics_raw']['AUC'],'hm_pct':100*m['metrics_raw']['best_hm'],
                    'best_unseen_pct':100*m['metrics_raw']['best_unseen'],
                    'new_trainable_params':count,'candidate_reported_even_if_rejected':True,
                    'checkpoint_sha256':complete['checkpoint_sha256'],'anchor_sha256':cfg.MODEL.branch_anchor_sha256})
    conditions={}
    threshold = value['pilot_threshold_pp'] if stage=='pilot' else value['confirmation_threshold_pp']
    for reference in ('N0','N_name','N_shuffle'):
        for setting in ('cw','ow'):
            deltas=[]
            for seed in seeds:
                t=values[(seed,'N_clean',setting)]; r=values[(seed,reference,setting)]
                delta=100*(t['AUC']-r['AUC']); deltas.append(delta)
                paired.append({'seed':seed,'comparison':'N_clean-'+reference,'setting':setting,
                    'delta_auc_pp':delta,'delta_hm_pp':100*(t['best_hm']-r['best_hm']),
                    'delta_unseen_pp':100*(t['best_unseen']-r['best_unseen'])})
            if setting=='cw':
                conditions['mean_cw_over_'+reference] = float(np.mean(deltas)) >= threshold
                conditions['positive_seeds_over_'+reference] = sum(d>0 for d in deltas) >= (1 if stage=='pilot' else 2)
    for reference in ('N0','N_name'):
        for setting,key,bound in [('cw','best_hm',-0.5),('ow','AUC',-0.5),('ow','best_unseen',-2.0)]:
            deltas=[100*(values[(s,'N_clean',setting)][key]-values[(s,reference,setting)][key]) for s in seeds]
            conditions['mean_guard_'+reference+'_'+setting+'_'+key] = float(np.mean(deltas)) >= bound
            # Avoid accepting large individual losses hidden by a positive mean.
            conditions['each_seed_guard_'+reference+'_'+setting+'_'+key] = all(d>=bound for d in deltas)
    target=work/'reports'/stage
    inputs=summary_inputs(work,stage)
    decision={'stage':stage,'proceed':all(conditions.values()),'conditions':conditions,
        'protocol_sha256':sha(protocol_path(work)),'inputs_sha256':inputs,
        'seeds':list(seeds),'threshold_pp':threshold,'trainable_params_each':capacity,
        'old_research_gate_passed':False,'interpretation':'Exploratory operational screen, not statistical proof',
        'next':'same-budget other seeds' if stage=='pilot' else 'conditional ablations and knowledge corruption diagnostics'}
    if (target/'decision.json').exists():
        require(read_json(target/'decision.json')==decision,'New report inputs need a new version')
    else:
        target.mkdir(parents=True,exist_ok=True)
        csv_write(target/'runs.csv',rows); csv_write(target/'paired_deltas.csv',paired)
        write_json(target/'decision.json',decision)
    print(json.dumps(decision,indent=2),flush=True)
    if not decision['proceed']:
        print('STOP THIS EXPLORATORY BUDGET; preserve all candidate results, including negative seeds.',flush=True)


def mechanisms(args):
    value=validate_protocol(args.work,full=True)
    work=Path(value['work'])
    decision=read_json(work/'reports/confirm/decision.json')
    require(decision['proceed'] and decision['inputs_sha256']==summary_inputs(work,'confirm'),
            'Mechanism stage requires the complete confirmed exploration')
    rows,inputs=[],{}
    for seed in SEEDS:
        clean=work/'experiments/name_ekr/runs'/rid('N_clean',seed)
        for variant in ('N_noobj','N_nonull'):
            run=work/'experiments/name_ekr/runs'/rid(variant,seed)
            require(read_json(run/'status.json')['status']=='train_and_val_complete','Ablation incomplete')
            validate_config(read_cfg(run/'effective_config.yml'),require_replay=True)
            complete=read_json(run/'train_complete.json')
            require(complete['base_state_start']==complete['base_state_end'],'Ablation anchor changed')
            require(complete['checkpoint_sha256']==sha(run/'checkpoints/best_branch.pth'),'Ablation checkpoint changed')
            require(read_json(run/'meta/code_manifest.json')==value['code_manifest'],'Ablation code differs')
            for setting in ('cw','ow'):
                p=run/('eval_val_'+setting)/'metrics.json'
                cp=clean/('eval_val_'+setting)/'metrics.json'
                inputs[str(p)],inputs[str(cp)]=sha(p),sha(cp)
                m,c=read_json(p),read_json(cp)
                require(m['checkpoint_sha256']==complete['checkpoint_sha256'] and
                        m['anchor_sha256']==c['anchor_sha256'] and m['variant']==variant and
                        m['seed']==seed and m['setting']==setting,'Ablation evaluation identity differs')
                rows.append({'seed':seed,'comparison':'N_clean-'+variant,'setting':setting,
                    'delta_auc_pp':100*(c['metrics_raw']['AUC']-m['metrics_raw']['AUC']),
                    'delta_hm_pp':100*(c['metrics_raw']['best_hm']-m['metrics_raw']['best_hm']),
                    'delta_unseen_pp':100*(c['metrics_raw']['best_unseen']-m['metrics_raw']['best_unseen'])})
    out=work/'reports/mechanisms'
    if out.exists():
        require(read_json(out/'inputs.json')==inputs,'Changed ablation inputs need a new report version')
    else:
        out.mkdir(parents=True)
        csv_write(out/'paired_deltas.csv',rows); write_json(out/'inputs.json',inputs)
    print('MECHANISM REPORT',out,flush=True)
    print('Inspect whether removing object context/null consistently removes gain. No automatic causal or novelty claim.',flush=True)


def noise(args):
    value=validate_protocol(args.work,full=True)
    work=Path(value['work'])
    decision=read_json(work/'reports/confirm/decision.json')
    require(decision['proceed'] and decision['inputs_sha256']==summary_inputs(work,'confirm'),
            'Corruption diagnosis requires confirmation first')
    from .name_runtime import load_branch
    rows,inputs=[],{}
    for seed in SEEDS:
        for variant in ('N_name','N_clean'):
            run=work/'experiments/name_ekr/runs'/rid(variant,seed)
            cfg=read_cfg(run/'effective_config.yml'); validate_config(cfg,require_replay=True)
            checkpoint=run/'checkpoints/best_branch.pth'
            require(sha(checkpoint)==read_json(run/'train_complete.json')['checkpoint_sha256'],'Noise checkpoint changed')
            model,_=load_branch(cfg,trainset(cfg),checkpoint)
            model.evaluation_checkpoint_sha256=sha(checkpoint)
            for ratio in (0.1,0.3):
                for noise_seed in (11,22,33):
                    for setting in ('cw','ow'):
                        out=run/'noise'/('p{}_n{}_{}'.format(ratio,noise_seed,setting))
                        path=out/'metrics.json'
                        if path.exists():
                            record=read_json(path)
                            require(record['checkpoint_sha256']==sha(checkpoint) and
                                record['noise_ratio']==ratio and record['noise_seed']==noise_seed,
                                'Existing corruption evaluation identity differs')
                        else:
                            with model.corrupt(ratio,noise_seed):
                                record=evaluate(model,cfg,'val',setting,out)
                                record.update(noise_ratio=ratio,noise_seed=noise_seed,
                                    corruption=model.last_corruption)
                                write_json(path,record)
                        inputs[str(path)]=sha(path)
                        clean_path=run/('eval_val_'+setting)/'metrics.json'
                        inputs[str(clean_path)]=sha(clean_path)
                        clean=read_json(clean_path)['metrics_raw']
                        rows.append({'seed':seed,'variant':variant,'ratio':ratio,'noise_seed':noise_seed,
                            'setting':setting,'clean_auc_pct':100*clean['AUC'],
                            'noise_auc_pct':100*record['metrics_raw']['AUC'],
                            'drop_auc_pp':100*(clean['AUC']-record['metrics_raw']['AUC']),
                            'actual_fraction':record['corruption']['actual_fraction']})
            del model
            torch.cuda.empty_cache()
    out=work/'reports/noise'
    if out.exists():
        require(read_json(out/'inputs.json')==inputs,'New corruption inputs require a new report version')
    else:
        out.mkdir(parents=True)
        csv_write(out/'raw_evaluations.csv',rows); write_json(out/'inputs.json',inputs)
        means=[]
        for seed in SEEDS:
            for variant in ('N_name','N_clean'):
                for ratio in (0.1,0.3):
                    for setting in ('cw','ow'):
                        group=[r for r in rows if (r['seed'],r['variant'],r['ratio'],r['setting'])==(seed,variant,ratio,setting)]
                        means.append({'seed':seed,'variant':variant,'ratio':ratio,'setting':setting,
                            'n_noise_seeds':3,'drop_auc_pp_mean':float(np.mean([r['drop_auc_pp'] for r in group]))})
        csv_write(out/'per_model_noise_means.csv',means)
    print('NOISE DIAGNOSIS COMPLETE; NO ADDITIONAL TRAINING',out,flush=True)


def pack(args):
    work=Path(args.work).resolve()
    validate_protocol(work)
    output=Path(args.out).resolve()
    require(not output.exists(),'Refuse to overwrite a result archive')
    paths=[]
    for folder in ('protocol','reports','smoke','diagnostics'):
        paths += [p for p in (work/folder).rglob('*') if p.is_file()]
    for run in (work/'experiments/name_ekr/runs').glob('*'):
        paths += [p for p in run.rglob('*') if p.is_file() and p.suffix not in {'.pth','.pt'}]
    require(paths,'No artifacts to pack')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        digests={}
        for path in sorted(set(paths)):
            name=str(path.relative_to(work)).replace('\\','/')
            z.write(path,name); digests[name]=sha(path)
        z.writestr('export_manifest.json',json.dumps({'files_sha256':digests,'weights_included':False,
            'version':'name_anchor_ekr_exploration_20261007'},indent=2))
    print('REVIEW ZIP',output,'SHA256',sha(output),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--work',required=True); q.add_argument('--previous',required=True)
    for command in ('replay','audit-scores'):
        q=sub.add_parser(command); q.add_argument('--work',required=True)
    q=sub.add_parser('run'); q.add_argument('--work',required=True)
    q.add_argument('--seeds',nargs='+',type=int,required=True)
    q.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(PRIMARY))
    q=sub.add_parser('summary'); q.add_argument('--work',required=True); q.add_argument('--stage',choices=['pilot','confirm'],required=True)
    for command in ('mechanisms','noise'):
        q=sub.add_parser(command); q.add_argument('--work',required=True)
    q=sub.add_parser('pack'); q.add_argument('--work',required=True); q.add_argument('--out',required=True)
    a=p.parse_args()
    {'prepare':prepare,'replay':replay,'audit-scores':audit_scores,'run':run_plan,
        'summary':summary,'pack':pack,'mechanisms':mechanisms,'noise':noise}[a.command](a)


if __name__=='__main__':
    main()
