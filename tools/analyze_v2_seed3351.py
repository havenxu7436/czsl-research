"""Read saved v2 scores; print derived diagnostics without changing the export."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).absolute().parents[1],
                    help='Repository root (defaults to the parent of tools/)')
parser.add_argument('--verify-archived', action='store_true',
                    help='Recompute and compare the published diagnostic JSON; write nothing')
args = parser.parse_args()
ROOT = args.root.absolute()
if not __debug__:
    raise RuntimeError('Run without python -O: validation assertions must remain enabled')
BATCH = ROOT / '0601方案/v2/results/seed3351-10epoch-20261015'
RAW = BATCH / 'raw_export'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


summary = read(RAW / 'summary/summary.json')
splits = read(RAW / 'checks/split_pairs.json')
vocab = read(ROOT / 'knowledge/ut-zap50k/definitions/legacy-v1/labels.json')
attr_id = {name: i for i, name in enumerate(vocab['attrs'])}
obj_id = {name: i for i, name in enumerate(vocab['objs'])}
train = set(map(tuple, splits['train']))


def load(run, phase, setting):
    folder = run / ('eval_' + phase + '_' + setting)
    meta = read(folder / 'metrics.json')
    pairs = list(map(tuple, read(folder / 'candidates.json')))
    shape = tuple(meta['score_shape'])
    assert len(pairs) == shape[1] and len(set(pairs)) == len(pairs)
    assert (folder / 'scores.dat').stat().st_size == int(np.prod(shape)) * 4
    scores = np.array(np.memmap(folder / 'scores.dat', mode='r', dtype=np.float32, shape=shape))
    with np.load(folder / 'labels.npz') as archive:
        labels = {name: archive[name] for name in archive.files}
    pa = np.array([attr_id[p[0]] for p in pairs])
    po = np.array([obj_id[p[1]] for p in pairs])
    truth = labels['pair']
    assert len(truth) == shape[0] == len(read(folder / 'image_ids.json'))
    assert np.array_equal(pa[truth], labels['attr'])
    assert np.array_equal(po[truth], labels['obj'])
    seen = np.array([p in train for p in pairs])
    legal = np.ones(len(pairs), dtype=bool)
    if setting == 'cw':
        allowed = train | set(map(tuple, splits[phase]))
        legal = np.array([p in allowed for p in pairs])
        assert legal[truth].all()
        scores[:, ~legal] = -np.inf
    assert np.isfinite(scores[:, legal]).all()
    seen_ids = np.where(seen & legal)[0]
    unseen_ids = np.where(~seen & legal)[0]
    si = seen_ids[scores[:, seen_ids].argmax(1)]
    ui = unseen_ids[scores[:, unseen_ids].argmax(1)]
    sm = scores[np.arange(shape[0]), si].astype(np.float64)
    um = scores[np.arange(shape[0]), ui].astype(np.float64)
    return dict(meta=meta, pairs=pairs, scores=scores, labels=labels, pa=pa, po=po,
                truth=truth, truth_seen=seen[truth], si=si, ui=ui, sm=sm, um=um)


def at_bias(data, bias):
    shifted = data['um'] + bias
    use_unseen = (shifted > data['sm']) | ((shifted == data['sm']) & (data['ui'] < data['si']))
    pred = np.where(use_unseen, data['ui'], data['si'])
    correct = pred == data['truth']
    mask = data['truth_seen']
    seen = float(correct[mask].mean())
    unseen = float(correct[~mask].mean())
    hm = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
    return dict(bias=float(bias), accuracy=float(correct.mean()), seen=seen, unseen=unseen, hm=hm), pred


def errors(data, pred):
    aok = data['pa'][pred] == data['labels']['attr']
    ook = data['po'][pred] == data['labels']['obj']
    return dict(n=len(pred), both_correct=int((aok & ook).sum()),
                object_only_error=int((aok & ~ook).sum()),
                attribute_only_error=int((~aok & ook).sum()),
                both_error=int((~aok & ~ook).sum()),
                object_accuracy=float(ook.mean()), attribute_accuracy=float(aok.mean()),
                attr_accuracy_when_joint_pred_object_correct=float(aok[ook].mean()) if ook.any() else None)


def lse(value, axis):
    maximum = value.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(value - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


rows = []
for state in summary['states']:
    if state['state'] != 'complete':
        continue
    run = RAW / 'runs' / state['variant']
    hashes = set()
    for setting in ('cw', 'ow'):
        val, test = load(run, 'val', setting), load(run, 'test', setting)
        hashes.update((val['meta']['checkpoint_sha256'], test['meta']['checkpoint_sha256']))
        choices = [at_bias(val, float(bias))[0] for bias in np.linspace(-5, 5, 201)]
        best = max(choices, key=lambda m: (m['hm'], -abs(m['bias']), -m['bias']))
        frozen, pred = at_bias(test, best['bias'])
        zero, zero_pred = at_bias(test, 0.0)
        # Independently verify optimized argmax against direct score addition.
        seen_candidates = np.array([pair in train for pair in test['pairs']])
        direct = (test['scores'] + best['bias'] * (~seen_candidates)[None]).argmax(1)
        assert np.array_equal(pred, direct)
        gap = test['sm'] - test['um']
        item = dict(variant=state['base_variant'], setting=setting, best_epoch=test['meta']['best_epoch'],
                    val_selected=best, test_at_val_bias=frozen, zero_bias_test=zero,
                    error_counts=errors(test, pred), zero_bias_error_counts=errors(test, zero_pred),
                    test_gap={kind: dict(n=int(mask.sum()), mean=float(gap[mask].mean()), median=float(np.median(gap[mask])))
                              for kind, mask in (('truth_seen', test['truth_seen']), ('truth_unseen', ~test['truth_seen']))})
        if setting == 'ow' and state['base_variant'] != 'A_native':
            O, A = len(obj_id), len(attr_id)
            assert len(test['pairs']) == O * A
            grid = np.empty((len(test['truth']), O, A), dtype=np.float32)
            grid[:, test['po'], test['pa']] = test['scores']
            assert np.allclose(np.exp(grid).sum((1, 2)), 1, atol=3e-5)
            obj_pred = lse(grid, 2).argmax(1)
            attr_pred = grid[np.arange(len(grid)), test['labels']['obj']].argmax(1)
            item['object_marginal_accuracy'] = float((obj_pred == test['labels']['obj']).mean())
            item['attribute_given_gt_object_oracle_accuracy'] = float((attr_pred == test['labels']['attr']).mean())
        rows.append(item)
    assert len(hashes) == 1

uniform_checks = []
for phase in ('val', 'test'):
    for setting in ('cw', 'ow'):
        name = 'eval_' + phase + '_' + setting
        uniform = RAW / 'runs/Rfull_uniform_prior_s3351' / name / 'scores.dat'
        off = RAW / 'runs/Rfull_prior_off_s3351' / name / 'scores.dat'
        uniform_checks.append(dict(phase=phase, setting=setting, identical=sha(uniform) == sha(off), sha256=sha(off)))
assert all(check['identical'] for check in uniform_checks)

kept = {'A_native', 'R0', 'Rtext', 'Rprior', 'Rmvp', 'Rexternal', 'Rmvp_logic',
        'Rfeature', 'Rfull', 'Rfull_noise50', 'Rfull_attention_K',
        'Rfull_marginal_attr', 'R0_native_attention'}
result = dict(
    source_summary_sha256=sha(RAW / 'summary/summary.json'),
    bias_policy='grid[-5,5],step0.05; maximize VAL HM; tie=min abs bias then lower bias; TEST applies frozen bias',
    verified_runs=len(rows) // 2,
    rows=[row for row in rows if row['variant'] in kept],
    uniform_prior_off_checks=uniform_checks,
    analysis_date='2026-10-16',
    scope='UT-Zappos seed3351; 32 runs verified; derived rows retained for 13 named models in the report',
    source_batch='raw_export',
    method='Read-only equivalent of saved-score VAL bias selection, CW masking, errors and unbiased OW marginals; no training and no checkpoint reselection',
    not_computed=['same-weight prior on/off diagnostic',
                  'human semantic error taxonomy', 'LLM-only/mixed prior results'])
assert result['verified_runs'] == 32 and len(result['rows']) == 26
assert result['source_summary_sha256'] == 'f7d5b0f8d3ec1fbb750d4636f93dff813496c354852dc144190c7086c837e6f5'


def compare(actual, expected, path='root'):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            compare(actual[key], expected[key], path + '.' + key)
    elif isinstance(actual, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, path + '[' + str(i) + ']')
    elif isinstance(actual, float):
        assert np.isclose(actual, expected, rtol=1e-9, atol=1e-9), path
    else:
        assert actual == expected, path


if args.verify_archived:
    compare(result, read(BATCH / 'v2单seed后分析读数_20261016.json'))
    print(json.dumps(dict(passed=True, verified_runs=32, verified_evaluations=128,
                          derived_rows=26, uniform_score_files_identical=4,
                          raw_bytes_modified=False), ensure_ascii=False))
else:
    print(json.dumps(result, ensure_ascii=False, indent=2))
