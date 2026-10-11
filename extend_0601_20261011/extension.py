"""Append only three frozen-description noise runs; never rewrite the completed main suite."""
import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_PROTOCOL_SHA = 'd7a9b1249f7de7bc6d296e01cb9c15006bead1139486084ef397f13ba3b2a6cd'
R2_MANIFEST_SHA = 'eadae0796a7ef52d1342616c4373f9f0f523380f3e806ceb500de8131134fadd'
MAIN = ['G', 'H', 'C', 'D', 'A', 'B', 'E', 'F', 'H_no_pair', 'H_shuffle', 'H_random_prior', 'H_uniform_prior']
NOISE = ['H_noise10', 'H_noise30', 'H_noise50']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_delivery():
    for name, expected in read(HERE / 'release_manifest.json').items():
        if sha(HERE / name) != expected:
            raise ValueError('Noise extension release changed: ' + name)


def parent_modules(release):
    release = Path(release).resolve()
    if sha(release / 'release_manifest.json') != R2_MANIFEST_SHA:
        raise ValueError('Must use the already delivered 0601 r2 release, not another source version')
    sys.path.insert(0, str(release))
    common = load('extension_parent_common', release / 'common.py')
    common.verify_release(release)
    preparation = load('extension_parent_prepare', release / 'prepare.py')
    queue = load('extension_parent_queue', release / 'run_suite.py')
    return common, preparation, queue


def verify_parent(parent, queue):
    parent = Path(parent).resolve()
    if sha(parent / 'protocol.json') != PARENT_PROTOCOL_SHA:
        raise ValueError('Parent protocol differs from the downloaded and reviewed 12-run batch')
    protocol = queue.verify_protocol(parent)
    if [j['variant'] for j in protocol['jobs']] != MAIN:
        raise ValueError('Unexpected parent jobs')
    summary = read(parent / 'summary/summary.json')
    if not summary['core_complete'] or not summary['all_planned_complete']:
        raise ValueError('Finish parent main suite first')
    frozen = {str(parent / 'protocol.json'): sha(parent / 'protocol.json'),
              str(parent / 'summary/summary.json'): sha(parent / 'summary/summary.json')}
    for job in protocol['jobs']:
        folder = Path(job['run'])
        state, trained, ckpt = read(folder / 'status.json'), read(folder / 'train_complete.json'), read(folder / 'meta/checkpoint.json')
        if state['state'] != 'complete' or trained['epochs'] != list(range(1, 11)):
            raise ValueError('Parent training incomplete: ' + job['variant'])
        if not (ckpt['sha256'] == trained['best_sha256'] == state['checkpoint_sha256']):
            raise ValueError('Parent checkpoint identity differs: ' + job['variant'])
        for relative in ('status.json', 'train_complete.json', 'meta/checkpoint.json', 'meta/labels.json'):
            p = folder / relative
            frozen[str(p)] = sha(p)
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                p = folder / ('eval_' + phase + '_' + setting) / 'metrics.json'
                if read(p)['checkpoint_sha256'] != ckpt['sha256']:
                    raise ValueError('Parent eval identity differs: ' + str(p))
                frozen[str(p)] = sha(p)
    return protocol, frozen


def changed_config(parent_config, work, variant):
    result = copy.deepcopy(parent_config)
    run = Path(work) / 'runs' / variant
    result['MODEL']['kcc_cache'] = str(Path(work) / ('cache_' + variant[2:]))
    result['MODEL']['bf_logic_bank'] = str(Path(work) / 'logic_bank')
    result['TRAIN'].update(kcc_stage='0601_noise_extension_20261011', kcc_variant=variant,
                           train_name='0601_' + variant, kcc_run_dir=str(run),
                           log_dir=str(run / 'logs'), checkpoint_dir=str(run / 'checkpoints'))
    return result


def prepare(args):
    import yaml
    common, preparation, queue = parent_modules(args.release)
    parent = args.parent.resolve()
    old, parent_evidence = verify_parent(parent, queue)
    work, root = args.work.resolve(), Path(old['root']).resolve()
    for protected in (parent, root, args.release.resolve()):
        if work == protected or work in protected.parents or protected in work.parents:
            raise ValueError('New work must be separate from all existing project/main/release directories')
    if work.exists():
        raise FileExistsError('Preserve existing work; use explicit run --resume after inspection: ' + str(work))
    source = Path(old['source'])
    expected_source = {n[7:]: h for n, h in read(args.release / 'release_manifest.json').items() if n.startswith('source/')}
    actual_source = {str(p.relative_to(source)).replace('\\', '/'): sha(p)
                     for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if expected_source != actual_source:
        raise ValueError('Parent executed source differs from sealed r2 training source')
    cache = root / 'knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k'
    preparation.verify_cache(cache)
    work.mkdir(parents=True)
    shutil.copytree(source, work / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(parent / 'logic_bank', work / 'logic_bank')
    shutil.copytree(HERE, work / 'extension_code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    builder = load('extension_cache_builder', work / 'source/tools/build_knowledge.py')
    frozen = dict(old['frozen_inputs'])
    frozen.update(parent_evidence)
    base_config = yaml.safe_load((parent / 'configs/H.yml').read_text(encoding='utf-8'))
    jobs = []
    for variant in NOISE:
        kind = variant[2:]
        frozen.update(preparation.altered_cache(cache, work / ('cache_' + kind), kind, builder, root / 'data/glove/glove.6B.300d.txt'))
        config = changed_config(base_config, work, variant)
        path = work / 'configs' / (variant + '.yml')
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding='utf-8')
        jobs.append({'variant': variant, 'tier': 2, 'cfg': str(path), 'cfg_sha256': sha(path), 'run': str(work / 'runs' / variant)})
    for directory in ('source', 'logic_bank', 'extension_code', 'cache_noise10', 'cache_noise30', 'cache_noise50'):
        for path in (work / directory).rglob('*'):
            if path.is_file():
                frozen[str(path)] = sha(path)
    protocol = copy.deepcopy(old)
    protocol.update(release='extend_0601_20261011', jobs=jobs, frozen_inputs=frozen,
                    source=str(work / 'source'), tier=2, parent_work=str(parent),
                    parent_protocol_sha256=PARENT_PROTOCOL_SHA, parent_release=str(args.release.resolve()),
                    parent_release_manifest_sha256=R2_MANIFEST_SHA,
                    historical_results='12 completed parent runs reused as read-only evidence; only three new noise runs',
                    extension_scope='Nested deletion of Text/Feature primitive OE/AE descriptions only; no prior/logic/OI/pair change')
    protocol['not_covered'] = [x for x in old['not_covered'] if x != 'knowledge noise sensitivity (tier2 not scheduled)']
    protocol['not_covered'].append('noise in OI/logic/prior/pair knowledge (this extension changes Text/Feature descriptions only)')
    write(work / 'protocol.json', protocol)
    print('NOISE_EXTENSION_PREPARED new_full_training_jobs=3 reused_completed_parent_jobs=12 work=' + str(work), flush=True)


def summarize(work):
    """Do not relabel a three-job extension as a freshly executed A-H suite."""
    import numpy as np
    work = Path(work)
    protocol = read(work / 'protocol.json')
    parent = Path(protocol['parent_work'])
    if sha(parent / 'protocol.json') != protocol['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    original = read(parent / 'summary/summary.json')
    rows = copy.deepcopy(original['rows'])
    states = []
    for job in protocol['jobs']:
        run = Path(job['run'])
        state = read(run / 'status.json') if (run / 'status.json').exists() else {'state': 'not_started'}
        states.append({'variant': job['variant'], **state})
        if state['state'] != 'complete':
            continue
        for phase in ('val', 'test'):
            for setting in ('cw', 'ow'):
                meta = read(run / ('eval_' + phase + '_' + setting) / 'metrics.json')
                row = {'variant': job['variant'], 'phase': phase, 'setting': setting, 'best_epoch': meta['best_epoch'],
                       'checkpoint_sha256': meta['checkpoint_sha256']}
                for key in ('AUC', 'best_hm', 'best_seen', 'best_unseen'):
                    value = float(meta['metrics_raw'][key])
                    if not np.isfinite(value):
                        raise ValueError('Nonfinite extension metrics')
                    row[key + '_pct'] = value * 100
                rows.append(row)
    complete = all(s['state'] == 'complete' for s in states)
    dest = work / 'summary'
    result = {'core_complete': original['core_complete'], 'all_planned_complete': complete,
              'extension_complete': complete, 'new_training_jobs': 3, 'reused_parent_runs': 12,
              'parent_protocol_sha256': protocol['parent_protocol_sha256'], 'protocol_sha256': sha(work / 'protocol.json'),
              'states': states, 'rows': rows, 'not_covered': protocol['not_covered']}
    write(dest / 'summary.json', result)
    lines = ['# 0601：噪声扩展与累计主表', '', '仅新增三次完整训练；原 12 组引用只读父批次，不重训、不改写。', '',
             '噪声扩展全部完成：`' + str(complete) + '`。', '',
             '删除只影响 Text/Feature 的 OE/AE 描述，不改变 prior、logic bank、OI 或 pair 模板。', '',
             '| 组 | VAL CW AUC | VAL OW AUC | TEST CW AUC | TEST OW AUC |', '|---|---:|---:|---:|---:|']
    for variant in MAIN + NOISE:
        selected = {(r['phase'], r['setting']): r['AUC_pct'] for r in rows if r['variant'] == variant}
        values = ['{:.4f}'.format(selected[k]) if k in selected else '未完成' for k in [('val', 'cw'), ('val', 'ow'), ('test', 'cw'), ('test', 'ow')]]
        lines.append('| ' + variant + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## 未覆盖', ''] + ['- ' + x for x in protocol['not_covered']]
    (dest / '0601_噪声扩展与总汇总.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return original['core_complete'], complete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--parent', type=Path, required=True)
    prep.add_argument('--release', type=Path, required=True)
    prep.add_argument('--work', type=Path, required=True)
    for mode in ('run', 'summary'):
        task = sub.add_parser(mode)
        task.add_argument('--work', type=Path, required=True)
        task.add_argument('--export', type=Path, required=True)
        task.add_argument('--gpus', default='auto')
        task.add_argument('--max-workers', type=int, default=3)
        task.add_argument('--deadline', default='2026-10-11T17:00:00+08:00')
        task.add_argument('--reserve-job-minutes', type=float, default=120)
        task.add_argument('--resume', action='store_true')
    args = p.parse_args()
    verify_delivery()
    if args.mode == 'prepare':
        prepare(args)
        return
    protocol = read(args.work / 'protocol.json')
    _, _, queue = parent_modules(Path(protocol['parent_release']))
    queue.verify_protocol(args.work)
    if [j['variant'] for j in protocol['jobs']] != NOISE:
        raise ValueError('Extension must schedule exactly three noise jobs')
    queue.summarize = summarize
    if args.mode == 'summary':
        summarize(args.work)
        queue.export(args.work, args.export)
    else:
        rc = queue.run(args)
        print('NOISE_EXTENSION_COMPLETE', read(args.work / 'summary/summary.json')['extension_complete'], flush=True)
        raise SystemExit(rc)


if __name__ == '__main__':
    main()
