"""Read-only analysis of formula-r2 metrics, selection events and bias-zero scores.

Uses the archived checkpoints selected by VAL CW AUC. Never changes scores,
selects by TEST, loads model weights or starts training. Prints JSON to stdout.
"""
import csv
import hashlib
import json
import re
import statistics as stats
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
STAGE = ROOT / '0601方案/02-公式对齐版'
RESULTS = [STAGE / 'results/seed3351-30epoch-20261012/raw_export',
           STAGE / 'results/seeds3352-3353-10epoch-20261012/raw_export']
SEEDS = [3351, 3352, 3353]
ORDER = list('ABCDEFGH') + ['J0', 'Jtext', 'H_no_pair', 'H_shuffle',
                           'H_random_prior', 'H_uniform_prior']
FIELDS = ['AUC', 'best_hm', 'best_seen', 'best_unseen', 'hm_seen', 'hm_unseen']
COMPARISONS = [('H', 'A'), ('H', 'B'), ('H', 'F'), ('G', 'E'), ('H', 'G'),
               ('E', 'A'), ('F', 'A'), ('B', 'A'), ('C', 'A'), ('F', 'E'),
               ('J0', 'A'), ('Jtext', 'C'), ('Jtext', 'J0'), ('D', 'J0'),
               ('E', 'Jtext'), ('H', 'H_no_pair'), ('H', 'H_shuffle'),
               ('H', 'H_random_prior'), ('H', 'H_uniform_prior')]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def describe(values):
    values = [float(v) for v in values]
    return {'values': values, 'mean': stats.mean(values),
            'sample_std': stats.stdev(values), 'positive_seeds': sum(v > 0 for v in values),
            'negative_seeds': sum(v < 0 for v in values)}


def main():
    rows, runs, curves, checksums = {}, {}, [], {}
    for raw in RESULTS:
        path = raw / 'summary/metrics.csv'
        checksums[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8-sig', newline='') as handle:
            csv_rows = list(csv.DictReader(handle))
        for row in csv_rows:
            seed, base = int(row['seed']), row['base_variant']
            key = (seed, base, row['phase'], row['setting'])
            assert key not in rows
            run = raw / 'runs' / row['variant']
            meta = read(run / ('eval_' + row['phase'] + '_' + row['setting']) / 'metrics.json')
            assert meta['best_epoch'] == int(row['best_epoch'])
            assert meta['checkpoint_sha256'] == row['checkpoint_sha256']
            for field in FIELDS[:4]:
                assert abs(float(row[field + '_pct']) - 100 * meta['metrics_raw'][field]) < 1e-8
            rows[key] = {field: 100 * float(meta['metrics_raw'][field]) for field in FIELDS}
            runs[(seed, base)] = run
    assert len(rows) == 168 and len(runs) == 42
    for seed in SEEDS:
        assert {v for s, v in runs if s == seed} == set(ORDER)
        for base in ORDER:
            run = runs[seed, base]
            state, trained = read(run / 'status.json'), read(run / 'train_complete.json')
            expected_epochs = 30 if seed == 3351 else 10
            assert state['state'] == 'complete' and state['epochs'] == expected_epochs
            assert trained['epochs'] == list(range(1, expected_epochs + 1))
            hashes = {read(run / ('eval_' + phase + '_' + setting) / 'metrics.json')['checkpoint_sha256']
                      for phase in ('val', 'test') for setting in ('cw', 'ow')}
            assert hashes == {trained['best_sha256']}
            log = (run / 'train.log').read_text(encoding='utf-8')
            events = [(int(e), float(v)) for e, v in re.findall(
                r'Saved new best checkpoint: epoch=(\d+), AUC=([\d.eE+-]+)', log)]
            assert events and events[-1][0] <= 10
            first10 = [v for e, v in events if e <= 10]
            assert events[-1][1] == max(first10)
            meta = read(run / 'eval_val_cw/metrics.json')
            assert meta['best_epoch'] == events[-1][0]
            logged = [(int(e), float(a) * 100) for e, a in re.findall(
                r'Val/Test Epoch (\d+): AUC=([\d.eE+-]+)', log)]
            assert len(logged) == expected_epochs
            curves.append({'seed': seed, 'variant': base, 'actual_epochs': expected_epochs,
                           'selected_epoch': events[-1][0],
                           'last_val_cw_auc_logged_pct': logged[-1][1],
                           'no_new_best_after10': True})
    means = {v: {phase + '_' + setting: {field: describe([
        rows[s, v, phase, setting][field] for s in SEEDS]) for field in FIELDS}
        for phase in ('val', 'test') for setting in ('cw', 'ow')} for v in ORDER}
    differences = {a + '-' + b: {phase + '_' + setting: {
        field: describe([rows[s, a, phase, setting][field] - rows[s, b, phase, setting][field]
                         for s in SEEDS]) for field in ('AUC', 'best_hm')}
        for phase in ('val', 'test') for setting in ('cw', 'ow')} for a, b in COMPARISONS}
    labels = read(runs[3351, 'A'] / 'meta/labels.json')
    split = ROOT / '前期实验/01-baseline/results/data/ut-zap50k/compositional-split-natural'
    train_pairs = {tuple(x.split()) for x in (split / 'train_pairs.txt').read_text().splitlines() if x.strip()}
    bias0 = {}
    for v in ORDER:
        values = []
        for seed in SEEDS:
            folder = runs[seed, v] / 'eval_test_ow'
            meta = read(folder / 'metrics.json')
            candidates = read(folder / 'candidates.json')
            assert len(candidates) == 192
            attrs = np.array([labels['attrs'].index(a) for a, o in candidates])
            objs = np.array([labels['objs'].index(o) for a, o in candidates])
            seen = np.array([tuple(p) in train_pairs for p in candidates])
            matrix = np.memmap(folder / 'scores.dat', mode='r', dtype=np.float32,
                               shape=tuple(meta['score_shape']))
            assert np.isfinite(matrix).all()
            with np.load(folder / 'labels.npz', allow_pickle=False) as truth:
                ga, go = truth['attr'], truth['obj']
                pred = matrix.argmax(1)
                obj_ok, attr_ok = objs[pred] == go, attrs[pred] == ga
                pair_ok = obj_ok & attr_ok
                is_seen = np.array([(labels['attrs'][a], labels['objs'][o]) in train_pairs for a, o in zip(ga, go)])
                r = {'seed': seed, 'pair_acc': 100 * pair_ok.mean(),
                     'obj_acc_from_pair': 100 * obj_ok.mean(),
                     'attr_acc_from_pair': 100 * attr_ok.mean(),
                     'object_wrong': 100 * (~obj_ok).mean(),
                     'object_right_attribute_wrong': 100 * (obj_ok & ~attr_ok).mean(),
                     'seen_pair_acc': 100 * pair_ok[is_seen].mean(),
                     'unseen_pair_acc': 100 * pair_ok[~is_seen].mean(),
                     'predicted_seen_fraction': 100 * seen[pred].mean()}
                for key, field in [('pair_acc', 'closed_ub_match'), ('obj_acc_from_pair', 'closed_ub_obj_match'),
                                   ('attr_acc_from_pair', 'closed_ub_attr_match')]:
                    assert abs(r[key] - 100 * meta['metrics_raw'][field]) < 0.00002
                assert abs(r['pair_acc'] + r['object_wrong'] + r['object_right_attribute_wrong'] - 100) < 1e-8
                seen_score, unseen_score = matrix[:, seen].max(1), matrix[:, ~seen].max(1)
                r['mean_seen_minus_unseen_score'] = float((seen_score - unseen_score).mean())
                values.append(r)
        bias0[v] = {key: describe([r[key] for r in values]) for key in values[0] if key != 'seed'}
    print(json.dumps({'source_metrics_sha256': checksums, 'runs_verified': len(runs),
                      'evaluations_verified': len(rows), 'curves': curves, 'means': means,
                      'paired_differences': differences, 'test_ow_bias0': bias0,
                      'common10_selection': 'All42 selected checkpoints within first10; no later3351 record beats them. Actual executed budgets remain30/10/10.'},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
