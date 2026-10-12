"""Audit a downloaded formula-r2 seed and review epochs using VAL only.

No model training, no TEST hyperparameter grid, and no historical result edits.
Generated evidence retains original bytes; source duplication is replaced by
verified links to existing canonical files in the curated Git repository.
"""
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
STAGE = ROOT / 'experiments/17-formula-aligned-0601'
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
                    not p.startswith('experiments/17-formula-aligned-0601/'),
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
    split_dir = ROOT / 'experiments/01-baseline/results/data/ut-zap50k/compositional-split-natural'
    splits = {}
    for phase in ('train', 'val', 'test'):
        path = split_dir / (phase + '_pairs.txt')
        original = '/data/xhw/czsl_repro/data/ut-zap50k/compositional-split-natural/' + path.name
        require(sha(path) == protocol['frozen_inputs'][original], 'Real split fingerprint mismatch: ' + phase)
        splits[phase] = [tuple(s.split()) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    helper = module(ROOT / 'experiments/16-residual-0601/code/review_0601_20261011/audit.py', 'native_replay_helpers')
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
              'other_seeds_not_started': 28, 'all14_selected_best_epochs_at_most6': all(r['best_epoch'] <= 6 for r in epoch_rows),
              'extra20_epochs_gpu_hours': saved_minutes / 60,
              'next_two_seeds_28jobs_extra20_gpu_hours_estimated': 2 * saved_minutes / 60,
              'decision': {'recommended_epochs': 10, 'basis': 'VAL CW selection events/curves only; not TEST-tuned',
                           'new_protocol_required': True, 'mutate_existing30_protocol': False,
                           'continue_old30_seed3352_seed3353_automatically': False,
                           'not_a_convergence_proof': True,
                           'seed3351_readout': 'Retrospective first10-window selection inside observed30epoch runs, same actually saved early checkpoint; not an independent10epoch execution'},
              'limitations': ['Single seed; no population significance or general convergence claim',
                              'Training VAL curves rounded to four decimals in raw log; exact selected metrics replayed from saved scores',
                              'Model checkpoint bytes absent from downloaded ZIP; checkpoint identity metadata verified, not actual weights',
                              'Prior TEST already observed; no new TEST budget/parameter tuning performed',
                              'External original dataset images/GloVe/server tensors not all available locally',
                              'No training launched, resumed, interrupted or reconfigured during this review']}
    write(out / 'audit.json', result)
    write(out / 'export_member_map.json', {'source_zip_sha256': sha(archive), 'members': mapped})
    write(out / 'source_links.json', source_links)
    shutil.copyfile(sidecar, out / 'original_zip.sha256')
    write(out / 'upload_scope.json', {'original_zip_not_duplicated': True,
                                      'raw_export_members_preserved_or_canonical_linked': len(mapped),
                                      'binary_scores_and_knowledge_use_git_lfs': True,
                                      'model_weights_dataset_images_credentials_included': False})
    lookup = {(r['variant'], r['phase'], r['setting']): r for r in rows}
    lines = ['# 单 seed3351：公式 r2 结果核验与后续 epoch 评估', '', '日期：2026-10-12。', '',
             '## 结论', '',
             '**建议后续主复刻实验统一10 epoch，不再直接续原30轮协议的另外两个seed。**', '',
             '14组×30轮全部完成；12份真实TRAIN/CUDA公式与梯度smoke通过；56份VAL/TEST CW/OW评价已从原始scores独立CPU复算。',
             '14组的最佳VAL CW checkpoint均在第2–6轮，第11–30轮没有更新任何一组最佳checkpoint。延长预算没有改善选模结果；这不是统计显著性或收敛证明。', '',
             '10轮也是CPF原论文§3.3的训练预算，见[官方论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_A_Conditional_Probability_Framework_for_Compositional_Zero-shot_Learning_ICCV_2025_paper.pdf)。本次建议不是因deadline砍轮数，也不按单seed各组的2–6轮峰值制定不同预算。LogiCZSL本身UT/C-GQA是15轮，不能把全部论文说成10轮；当前选择以CPF底座的统一预算为主。', '',
             '## 预算证据（只依据VAL决策）', '',
             '| 组 | 最佳轮次 | 前10轮最佳VAL CW AUC | 11–30轮最高VAL CW AUC | 第30轮VAL CW AUC | 延长后最佳增益 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in epoch_rows:
        lines.append('| {variant} | {best_epoch} | {best10_val_cw_auc_pct_logged:.2f} | {max_epochs11_30_val_cw_auc_pct_logged:.2f} | {epoch30_val_cw_auc_pct_logged:.2f} | {extra20_best_gain_pp:.2f} |'.format(**r))
    lines += ['', '曲线数字来自日志四位小数打印，转换为百分比；精确最佳指标见下表和各组metrics.json。后期验证下降、训练损失继续下降与过拟合/泛化退化相容，不能据此断言唯一原因。', '',
              '额外20轮累计消耗约 **{:.2f} GPU小时**；若再给两个seed的28组同样延长，预计额外约 **{:.2f} GPU小时**（按本次实测外推，不含等待/初始化/最终评价）。'.format(saved_minutes / 60, 2 * saved_minutes / 60), '',
              '## 精确结果（AUC%，同一VAL CW选中的checkpoint）', '',
              '| 组 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |', '|---|---:|---:|---:|---:|---:|']
    for name in ORDER:
        values = [lookup[(name, phase, setting)]['AUC_pct'] for phase, setting in [('val', 'cw'), ('val', 'ow'), ('test', 'cw'), ('test', 'ow')]]
        lines.append('| {} | {} | {} |'.format(name, lookup[(name, 'val', 'cw')]['best_epoch'], ' | '.join('{:.4f}'.format(v) for v in values)))
    lines += ['', '单seed观察，不报告虚假的跨seed标准差或显著性。G/H没有优于A/E/F；这批负结果完整保留，不能靠继续加epoch掩盖。', '',
              '## 配对差分（百分点，仅描述既有结果，不用于预算选择）', '',
              '| 比较 | VAL CW | VAL OW | TEST CW | TEST OW |', '|---|---:|---:|---:|---:|']
    for left, right in [('H', 'A'), ('G', 'E'), ('H', 'F'), ('H', 'G'), ('B', 'A'), ('D', 'J0'), ('E', 'Jtext'), ('H', 'H_shuffle'), ('H', 'H_random_prior'), ('H', 'H_uniform_prior')]:
        delta = [lookup[(left, p, s)]['AUC_pct'] - lookup[(right, p, s)]['AUC_pct'] for p, s in [('val', 'cw'), ('val', 'ow'), ('test', 'cw'), ('test', 'ow')]]
        lines.append('| {}−{} | {} |'.format(left, right, ' | '.join('{:+.4f}'.format(v) for v in delta)))
    lines += ['', '## 已有结果能否用于10轮？', '',
              '本次条件已经满足：最后保存的最佳模型全部来自前10轮，之后未出现更高VAL CW AUC。因此“仅在前10轮范围选模”的checkpoint与原30轮选模完全相同，可以复用其已保存的四项评价作为**前10轮窗口回溯读数**。',
              '仍然必须标注实际执行30轮，不改原protocol/日志，不冒称已经独立执行了14次10轮训练。下载包没有模型权重，以上权重身份来自checkpoint指纹与事件记录；独立四项指标从真实scores复算。',
              '已有代码的max_epoch仅控制训练循环/预算守卫，固定milestone23/30，前10轮无LR衰减；没有依赖总预算的cosine schedule。因而同seed同输入的前10轮轨迹可按同一实现解释，但跨机器/重跑仍不能保证逐bit恒等。', '',
              '## 后续执行原则', '',
              '- 新建10-epoch协议与work目录，明确冻结轮数；不得编辑现有30轮protocol，也不对旧work使用--resume --seeds all。',
              '- 以seed3352/3353各14组×10轮补重复。若希望三个seed都是独立执行10轮，seed3351也在新协议重跑；这是严格执行身份选择，不是当前有证据必须再训练它。',
              '- 若以窗口回溯方式复用3351，主表逐seed明确3351实际30轮/前10窗口、另两seed实际10轮；不要掩盖不同执行预算。',
              '- 三seed都坚持A–H、J0/Jtext和四项H对照、同VAL选模与四项最终评价；不因没有涨点删除负结果。',
              '- 不把10轮叫作所有新seed必然收敛；若新seed末尾仍持续改善，先依据VAL统一新协议复核预算，而不是按TEST补训。',
              '- 原方案全部备选/CLIP描述编码/其他数据集等仍未覆盖；本报告只评估当前UT主公式接口版本。', '',
              '## 完整性与文件', '',
              '- 原下载ZIP SHA256：`{}`。'.format(sha(archive)),
              '- {}项检查通过，{}个export成员校验，{}个执行source文件与r2一致。'.format(checks, len(manifest['files_sha256']), len(executed)),
              '- 56次指标复算，最大绝对差 `{}`（原始0–1单位）。'.format(result['max_metric_replay_abs_difference']),
              '- [audit.json](audit.json)：核验细节、VAL预算分析、实际指标与局限。',
              '- [raw_export/summary/summary.json](raw_export/summary/summary.json)：服务器原始汇总。',
              '- [raw_export/protocol.json](raw_export/protocol.json)：原30轮协议，不被本次结论改写。',
              '- [export_member_map.json](export_member_map.json)：每个原ZIP成员对应的仓库位置；source去重指向已校验canonical文件，其余原始字节保留。',
              '- 本次不上传大模型权重、数据集图片或凭据，也不重复上传ZIP容器；scores.dat/labels.npz/知识.pt通过Git LFS保存。',
              '- 另外两个seed28次任务仍not_started；“14组已完成”不等于原42次计划全部完成。', '']
    (out / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'checks', 'export_members_verified', 'completed_seed3351_jobs',
                      'evaluations_replayed', 'max_metric_replay_abs_difference', 'extra20_epochs_gpu_hours')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
