"""Audit formula-r2 seed evidence and report observed metrics and epoch curves."""
import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

sys.dont_write_bytecode = True
# Preserve the logical checkout root: curated directories can be junctions.
ROOT = Path(__file__).absolute().parents[4]
STAGE = ROOT / '0601方案/02-公式对齐版'
ORDER = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J0', 'Jtext',
         'H_no_pair', 'H_shuffle', 'H_random_prior', 'H_uniform_prior']
FIELDS = ['AUC', 'best_hm', 'best_seen', 'best_unseen', 'hm_seen', 'hm_unseen']


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=STAGE / 'results/seed3351-30epoch-20261012')
    parser.add_argument('--recheck', action='store_true', help='Verify existing raw bytes without overwriting them; regenerate this reviewer outputs')
    args = parser.parse_args()
    archive, out = args.archive.absolute(), args.out.absolute()
    if out.exists() and any(out.iterdir()) and not args.recheck:
        raise FileExistsError('Preserve previous review: ' + str(out))
    torch.set_num_threads(2)
    checks, errors = 0, []

    def require(value, text):
        nonlocal checks
        checks += 1
        if not value:
            errors.append(text)

    sidecar = archive.with_suffix('.zip.sha256')
    require(sha(archive) == sidecar.read_text(encoding='utf-8-sig').split()[0], 'ZIP sidecar mismatch')
    out.mkdir(parents=True, exist_ok=True)
    raw = out / 'raw_export'
    canonical = {}
    for record in read(ROOT / 'provenance/layout.json')['files']:
        if record['current_path']:
            canonical.setdefault(record['sha256'], []).append(record['current_path'])
    source_links, mapped = {}, {}
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP names')
        require(handle.testzip() is None, 'ZIP CRC failed')
        manifest = json.loads(handle.read('export_manifest.json'))
        require(set(names) == set(manifest['files_sha256']) | {'export_manifest.json'}, 'Export manifest member set mismatch')
        release_archive = ROOT / 'output/0601_formula_release_20261011_r2.zip'
        require(sha(release_archive) == '1516d14011c015754adfd54b421999ff3e7d05ee0a257f186e13e29a660c8c49', 'Original r2 release differs')
        with zipfile.ZipFile(release_archive) as release:
            released = {n.split('/source/', 1)[1]: hashlib.sha256(release.read(n)).hexdigest()
                        for n in release.namelist() if '/source/' in n and not n.endswith('/')}
        executed = {}
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or relative.drive or '..' in relative.parts:
                raise ValueError('Unsafe ZIP member: ' + name)
            data = handle.read(name)
            digest = hashlib.sha256(data).hexdigest()
            if name != 'export_manifest.json':
                require(digest == manifest['files_sha256'][name], 'ZIP SHA mismatch: ' + name)
            if name.startswith('source/'):
                executed[name[len('source/'):]] = digest
                candidates = sorted(canonical.get(digest, []), key=lambda p: (
                    not p.startswith('0601方案/02-公式对齐版/'),
                    not p.endswith(name[len('source/'):]), len(p)))
                if candidates:
                    target = ROOT / candidates[0]
                    require(target.is_file() and sha(target) == digest, 'Canonical source changed: ' + candidates[0])
                    source_links[name] = candidates[0]
                    mapped[name] = candidates[0]
                    continue
            target = raw / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not args.recheck or sha(target) != digest:
                    raise ValueError('Preserve changed extracted evidence: ' + name)
            else:
                target.write_bytes(data)
            mapped[name] = target.relative_to(ROOT).as_posix()
        require(executed == released, 'Executed source differs from delivered formula r2')
    protocol = read(raw / 'protocol.json')
    summary = read(raw / 'summary/summary.json')
    jobs = [j for j in protocol['jobs'] if j['seed'] == 3351]
    require(protocol['epochs'] == 30 and protocol['seeds'] == [3351, 3352, 3353], 'Frozen protocol budget/seeds differ')
    require({j['base_variant'] for j in jobs} == set(ORDER) and len(jobs) == 14, 'Single-seed matrix incomplete')
    require(len(protocol['jobs']) == 42, 'Expected 42 planned jobs')
    require(summary['protocol_sha256'] == sha(raw / 'protocol.json'), 'Summary protocol hash differs')
    outside = [s for s in summary['states'] if s['seed'] != 3351]
    require(all(s['state'] == 'not_started' for s in outside) and len(outside) == 28, 'Unexpected other-seed execution')
    require(not summary['core_complete'] and not summary['all_planned_complete'], 'Whole42-job completion must remain false')
    work_prefix = '/var/tmp/czsl_0601_formula_20261011/'
    frozen_available = 0
    for name, digest in protocol['frozen_inputs'].items():
        if name.startswith(work_prefix):
            member = name[len(work_prefix):]
            require(member in mapped and sha(ROOT / mapped[member]) == digest, 'Frozen work input changed: ' + member)
            frozen_available += 1
    split_dir = ROOT / '前期实验/01-baseline/results/data/ut-zap50k/compositional-split-natural'
    splits = {}
    for phase in ('train', 'val', 'test'):
        path = split_dir / (phase + '_pairs.txt')
        original = '/data/xhw/czsl_repro/data/ut-zap50k/compositional-split-natural/' + path.name
        require(sha(path) == protocol['frozen_inputs'][original], 'Real split fingerprint mismatch: ' + phase)
        splits[phase] = [tuple(s.split()) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    helper = module(ROOT / '0601方案/01-残差版/code/review_0601_20261011/audit.py', 'native_replay_helpers')
    if errors:
        write(out / 'audit.json', {'passed': False, 'checks': checks, 'errors': errors})
        raise ValueError('Identity checks failed; no archived evaluator will be executed')
    evaluator = module(ROOT / mapped['source/evaluator_ge.py'], 'formula_seed_evaluator')
    require(evaluator.device == 'cpu', 'Replay must be CPU')
    labels = read(raw / 'runs/A_s3351/meta/labels.json')
    replay, rows, epoch_rows, truth_baselines = [], [], [], {}
    metadata_lookup = {(r['variant'], r['phase'], r['setting']): r for r in summary['rows']}
    for name in ORDER:
        job = next(j for j in jobs if j['base_variant'] == name)
        variant = job['variant']
        run = raw / 'runs' / variant
        config = yaml.safe_load((run / 'config_requested.yml').read_text(encoding='utf-8'))
        config_file = raw / 'configs' / (variant + '.yml')
        require(sha(config_file) == job['cfg_sha256'], 'Job config changed: ' + variant)
        require(config_file.read_bytes() == (run / 'config_requested.yml').read_bytes(), 'Requested/run config mismatch: ' + variant)
        require(config['TRAIN']['max_epoch'] == config['TRAIN']['final_max_epoch'] == 30 and config['TRAIN']['seed'] == 3351,
                'Run budget/seed mismatch: ' + variant)
        status = read(run / 'status.json')
        trained = read(run / 'train_complete.json')
        checkpoint = read(run / 'meta/checkpoint.json')
        require(status['state'] == 'complete' and status['epochs'] == 30, 'Incomplete run: ' + variant)
        require(trained['epochs'] == list(range(1, 31)), 'Incomplete training ledger: ' + variant)
        require(trained['best_sha256'] == status['checkpoint_sha256'] == checkpoint['sha256'], 'Checkpoint identities differ: ' + variant)
        log = (run / 'train.log').read_text(encoding='utf-8', errors='replace')
        finished = [(int(e), float(t)) for e, t in re.findall(r'FULL_EPOCH_FINISHED epoch=(\d+) seconds=([\d.]+)', log)]
        require([e for e, t in finished] == list(range(1, 31)), 'Missing/duplicate log epochs: ' + variant)
        curve = [{'epoch': int(e), 'val_cw_auc_pct': 100 * float(auc), 'val_cw_hm_pct': 100 * float(hm)}
                 for e, auc, hm in re.findall(r'Val/Test Epoch (\d+): AUC=([\d.eE+-]+) \| best_hm=([\d.eE+-]+)', log)]
        require(curve == summary['val_curves'][variant] and len(curve) == 30, 'VAL curve differs: ' + variant)
        save_events = [(int(e), float(a)) for e, a in re.findall(r'Saved new best checkpoint: epoch=(\d+), AUC=([\d.eE+-]+)', log)]
        loaded = [int(e) for e in re.findall(r'Loading validation-best checkpoint from epoch (\d+)', log)]
        require(bool(save_events) and len(loaded) == 1 and save_events[-1][0] == loaded[0], 'Selected epoch event mismatch: ' + variant)
        require(read(run / 'meta/labels.json') == labels, 'Primitive ordering differs: ' + variant)
        model = config['MODEL']
        if model['kcc_feature'] or model['kcc_logic'] or model['formula_joint']:
            smoke = read(run / 'smoke.json')
            require(all(smoke[k] is True for k in ('passed', 'same_native_initialization', 'native_cpu_cuda_rng_preserved',
                                                  'joint_formula_exact')), 'Smoke failed: ' + variant)
            require(smoke['train_infer_formula_identity'] == bool(model['formula_joint']),
                    'Smoke train/infer applicability mismatch: ' + variant)
            require(smoke['optimizer_steps'] == 0 and smoke['test_images_read'] == 0, 'Smoke budget/TEST access changed: ' + variant)
            require(all(np.isfinite(v) and v > 0 for v in smoke['gradients'].values()), 'Gradient smoke missing: ' + variant)
        if model['kcc_feature']:
            require('FULL_GRADIENT epoch=30' in log, 'Final head gradient missing: ' + variant)
        if model['kcc_logic']:
            require('BF_LOGIC epoch=30' in log, 'Final logic evidence missing: ' + variant)
        group_metrics = {}
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                folder = run / ('eval_' + phase + '_' + setting)
                meta = read(folder / 'metrics.json')
                require(meta['best_epoch'] == loaded[0] and meta['checkpoint_sha256'] == checkpoint['sha256'], 'Evaluation checkpoint mismatch: ' + variant)
                require(meta['config_sha256'] == sha(folder / 'config_eval_effective.yml'), 'Eval config SHA mismatch: ' + variant)
                ds = helper.dataset(labels, splits, phase, setting)
                require(read(folder / 'candidates.json') == [list(p) for p in ds.pairs], 'Candidate ordering mismatch: ' + variant)
                shape = (3214 if phase == 'val' else 2914, len(ds.pairs))
                require(meta['score_shape'] == list(shape) and (folder / 'scores.dat').stat().st_size == int(np.prod(shape)) * 4,
                        'Score shape/size mismatch: ' + variant)
                truth = np.load(folder / 'labels.npz', allow_pickle=False)
                current_truth = {key: truth[key] for key in ('attr', 'obj', 'pair')}
                truth_key = phase + '_' + setting
                images = read(folder / 'image_ids.json')
                if truth_key not in truth_baselines:
                    truth_baselines[truth_key] = (current_truth, images)
                else:
                    require(all(np.array_equal(value, truth_baselines[truth_key][0][key]) for key, value in current_truth.items())
                            and images == truth_baselines[truth_key][1], 'Ground truth/image order mismatch: ' + variant)
                matrix = np.memmap(folder / 'scores.dat', mode='r', dtype=np.float32, shape=shape)
                require(np.isfinite(matrix).all(), 'Nonfinite scores: ' + variant)
                recalculated = helper.evaluate(matrix, truth, ds, evaluator.Evaluator)
                difference = max(abs(recalculated[key] - float(meta['metrics_raw'][key])) for key in FIELDS)
                require(difference < 2e-6, 'Metric replay mismatch: ' + variant + '/' + truth_key)
                require(all(abs(metadata_lookup[(variant, phase, setting)][key + '_pct'] - 100 * recalculated[key]) < 0.0002
                            for key in FIELDS[:4]), 'Summary metric mismatch: ' + variant)
                replay.append({'variant': variant, 'phase': phase, 'setting': setting, 'max_abs_difference': difference})
                record = {'variant': name, 'phase': phase, 'setting': setting, 'best_epoch': loaded[0],
                          **{key + '_pct': 100 * value for key, value in recalculated.items()}}
                rows.append(record)
                group_metrics[phase + '_' + setting] = record
                truth.close()
        upto10 = [p for p in curve if p['epoch'] <= 10]
        after10 = [p for p in curve if p['epoch'] > 10]
        best10 = max(p['val_cw_auc_pct'] for p in upto10)
        best30 = max(p['val_cw_auc_pct'] for p in curve)
        last_train = [(int(e), float(loss), float(pair)) for e, loss, pair in re.findall(
            r'Epoch: (\d+) Iter: \d+/\d+, Loss: ([\d.eE+-]+), Acc_Obj: [\d.eE+-]+, Acc_Attr: [\d.eE+-]+, Acc_Pair: ([\d.eE+-]+)', log)]
        require(loaded[0] <= 10 and save_events[-1][0] <= 10 and best10 == best30, 'First10-window selected identity differs: ' + variant)
        epoch_rows.append({'variant': name, 'best_epoch': loaded[0], 'best10_val_cw_auc_pct_logged': best10,
                           'best30_val_cw_auc_pct_logged': best30, 'extra20_best_gain_pp': best30 - best10,
                           'max_epochs11_30_val_cw_auc_pct_logged': max(p['val_cw_auc_pct'] for p in after10),
                           'epoch10_val_cw_auc_pct_logged': upto10[-1]['val_cw_auc_pct'],
                           'epoch30_val_cw_auc_pct_logged': curve[-1]['val_cw_auc_pct'],
                           'mean_last5_val_cw_auc_pct_logged': statistics.mean(p['val_cw_auc_pct'] for p in curve[-5:]),
                           'training_minutes': sum(t for e, t in finished) / 60,
                           'first10_minutes': sum(t for e, t in finished if e <= 10) / 60,
                           'extra20_minutes': sum(t for e, t in finished if e > 10) / 60,
                           'median_min_per_epoch_last5': statistics.median(t for e, t in finished[-5:]) / 60,
                           'selected_checkpoint_sha256': checkpoint['sha256'],
                           'best_within10_is_final_saved_best': True,
                           'train_last_interval_loss': last_train[-1][1] if last_train else None,
                           'train_last_interval_pair_accuracy': last_train[-1][2] if last_train else None,
                           'train_logging_notice': 'Interval mean, not full-epoch mean'})
        print('AUDITED_AND_REPLAYED', variant, 'best_epoch', loaded[0], flush=True)
    if errors:
        write(out / 'audit.json', {'passed': False, 'checks': checks, 'errors': errors})
        raise ValueError('Review failed; preserve evidence and do not upload as verified')
    saved_minutes = sum(r['extra20_minutes'] for r in epoch_rows)
    result = {'passed': True, 'date': '2026-10-12', 'checks': checks, 'errors': errors,
              'source_zip_sha256': sha(archive), 'export_members_verified': len(manifest['files_sha256']),
              'source_files_matched_r2': len(executed), 'frozen_work_inputs_verified': frozen_available,
              'completed_seed3351_jobs': 14, 'completed_epochs': 420, 'evaluations_replayed': 56,
              'max_metric_replay_abs_difference': max(r['max_abs_difference'] for r in replay),
              'native_metric_replays': replay, 'metrics': rows, 'epoch_review': epoch_rows,
              'all14_selected_best_epochs_at_most6': all(r['best_epoch'] <= 6 for r in epoch_rows),
              'extra20_epochs_gpu_hours': saved_minutes / 60,
              'scope': {'dataset': 'ut-zappos', 'seed': 3351, 'epochs_per_run': 30,
                        'evaluation_settings': ['VAL CW', 'VAL OW', 'TEST CW', 'TEST OW'],
                        'curve_precision': 'Four decimals in original logs',
                        'checkpoint_record': 'Identity hashes in original export'}}
    write(out / 'audit.json', result)
    write(out / 'export_member_map.json', {'source_zip_sha256': sha(archive), 'members': mapped})
    write(out / 'source_links.json', source_links)
    shutil.copyfile(sidecar, out / 'original_zip.sha256')
    write(out / 'upload_scope.json', {'original_zip_not_duplicated': True,
                                      'raw_export_members_preserved_or_canonical_linked': len(mapped),
                                      'binary_scores_and_knowledge_use_git_lfs': True,
                                      'model_weights_dataset_images_credentials_included': False})
    lookup = {(r['variant'], r['phase'], r['setting']): r for r in rows}
    lines = ['# formula-aligned-0601 r2：seed 3351 结果', '', '记录日期：2026-10-12。', '',
             'UT-Zappos；14组各30 epoch；首次严格最大VAL CW AUC选模；同一checkpoint评价VAL/TEST CW/OW。', '',
             '## 训练曲线', '', '14组最佳VAL CW checkpoint位于epoch 2–6。曲线读数来自日志四位小数打印，单位为百分比。', '',
             '| 组 | 最佳轮次 | 前10轮最佳VAL CW AUC | 11–30轮最高VAL CW AUC | 第30轮VAL CW AUC | 延长后最佳增益 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in epoch_rows:
        lines.append('| {variant} | {best_epoch} | {best10_val_cw_auc_pct_logged:.2f} | {max_epochs11_30_val_cw_auc_pct_logged:.2f} | {epoch30_val_cw_auc_pct_logged:.2f} | {extra20_best_gain_pp:.2f} |'.format(**r))
    lines += ['', '## 选中 checkpoint 的指标', '', 'AUC单位为百分比；每行的四项指标来自同一选中checkpoint。', '',
              '| 组 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |', '|---|---:|---:|---:|---:|---:|']
    for name in ORDER:
        values = [lookup[(name, phase, setting)]['AUC_pct'] for phase, setting in [('val', 'cw'), ('val', 'ow'), ('test', 'cw'), ('test', 'ow')]]
        lines.append('| {} | {} | {} |'.format(name, lookup[(name, 'val', 'cw')]['best_epoch'], ' | '.join('{:.4f}'.format(v) for v in values)))
    lines += ['', '## 配对差分', '', '单位为百分点，范围为seed 3351的本批结果。', '',
              '| 比较 | VAL CW | VAL OW | TEST CW | TEST OW |', '|---|---:|---:|---:|---:|']
    for left, right in [('H', 'A'), ('G', 'E'), ('H', 'F'), ('H', 'G'), ('B', 'A'), ('D', 'J0'), ('E', 'Jtext'), ('H', 'H_shuffle'), ('H', 'H_random_prior'), ('H', 'H_uniform_prior')]:
        delta = [lookup[(left, p, s)]['AUC_pct'] - lookup[(right, p, s)]['AUC_pct'] for p, s in [('val', 'cw'), ('val', 'ow'), ('test', 'cw'), ('test', 'ow')]]
        lines.append('| {}−{} | {} |'.format(left, right, ' | '.join('{:+.4f}'.format(v) for v in delta)))
    lines += ['', '## 校验与文件', '',
              '{}项检查通过；{}个导出成员完成校验；{}个执行源文件与r2对应。'.format(checks, len(manifest['files_sha256']), len(executed)),
              '56份评价从保存分数复算，原始0–1单位的最大数值差为{}。'.format(result['max_metric_replay_abs_difference']), '',
              '- [audit.json](audit.json)：指标核验与曲线统计。',
              '- [raw_export/summary/summary.json](raw_export/summary/summary.json)：原始指标汇总。',
              '- [raw_export/protocol.json](raw_export/protocol.json)：实际训练协议。',
              '- [export_member_map.json](export_member_map.json)：导出成员索引。', '']
    (out / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'checks', 'export_members_verified', 'completed_seed3351_jobs',
                      'evaluations_replayed', 'max_metric_replay_abs_difference', 'extra20_epochs_gpu_hours')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
