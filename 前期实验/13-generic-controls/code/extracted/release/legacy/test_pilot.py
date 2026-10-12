"""CPU-only synthetic tests; these do not claim a real CUDA training result."""
import argparse
import ast
import copy
import importlib.util
import io
import json
import pickletools
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
import torch
import export_results as exporter
import pilot
import runtime
from common import ROOT, read_json, sha, state_sha, tensor_load, tensor_save, write_json
from head import CAP, VARIANTS, ConditionalHead, corrected_scores, probability_checks


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        cls.ref = read_json(ROOT / 'reference.json')
        cls.controls = cls.ref['control_protocol']
        generator = torch.Generator().manual_seed(941)
        cls.bank = {axis: dict(units=torch.randn(n, 4, 300, generator=generator), mask=torch.ones(n, 4))
                    for axis, n in (('a', 16), ('o', 12))}
        cls.settings = copy.deepcopy(cls.ref['pilot_settings'])
        cls.settings.update(epochs=2, batch_size=16)
        attrs = ['a%02d' % i for i in range(16)]
        objects = ['o%02d' % i for i in range(12)]
        all_pairs = [(a, o) for a in attrs for o in objects]
        split = dict(train=all_pairs[:96], val=all_pairs[96:160], test=all_pairs[160:])
        cls.datasets = {s: runtime.make_dataset(dict(labels=dict(attrs=attrs, objs=objects)), s, split)
                        for s in ('cw', 'ow')}
        def cache(a, o):
            n = len(a)
            scores = torch.rand(n, 12, 16, generator=generator) * .2
            rows = torch.arange(n)
            scores[rows, o, a] += .2
            scores[rows[::3], (o[::3] + 1) % 12, a[::3]] += .4
            return dict(features=torch.randn(n, 300, generator=generator), scores=scores, attrs=a, objects=o)
        a = torch.arange(48).remainder(4).repeat_interleave(2)
        o = torch.arange(48).remainder(12).repeat_interleave(2)
        cls.caches = dict(train=cache(a, o), val=cache(torch.arange(96).remainder(12), torch.arange(96).remainder(12)))
        cls.ids = dict(train=['synthetic_%05d' % i for i in range(96)], val=['val_%05d' % i for i in range(96)])
        cls.initializations = runtime.matched_initializations(cls.bank, cls.controls)
        cls.evaluator = EVALUATOR

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pilot_test_')
        self.work = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def head(self, variant='P_K'):
        return runtime.fresh_head(self.bank, variant, self.controls, 'cpu')

    def run_variant(self, work, variant='P_K'):
        return pilot.formal_variant(work, variant, self.caches, self.bank, self.controls, self.datasets,
                                    self.evaluator, self.settings, 'synthetic_protocol', self.initializations[variant], 'cpu')

    def test_01_python38_syntax(self):
        for path in ROOT.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=path.name, feature_version=(3, 8))

    def test_02_zero_head_and_off_exact(self):
        for variant in VARIANTS:
            head = self.head(variant)
            cache = self.caches['val']
            self.assertTrue(torch.equal(head(cache['features'], cache['scores']), cache['scores']))
            self.assertIs(head(cache['features'], cache['scores'], force_off=True), cache['scores'])

    def test_03_full_grid_identity(self):
        torch.manual_seed(200)
        base = self.caches['val']['scores']
        delta = CAP * torch.tanh(torch.randn_like(base))
        self.assertLess(max(probability_checks(base, delta).values()), 1e-5)
        with self.assertRaises(ValueError):
            corrected_scores(base.flatten(1), delta.flatten(1))

    def test_04_initialization_and_control_counts(self):
        self.assertEqual(self.initializations['P_K']['parameter_count'], 57760)
        self.assertEqual(self.initializations['P_generic']['parameter_count'], 60864)
        self.assertEqual(len({self.initializations[v]['initial_state_sha256'] for v in VARIANTS if v != 'P_generic'}), 1)

    def test_05_permutation_only_descriptions(self):
        correct = self.head()
        for variant in ('P_perm1', 'P_perm2'):
            head = self.head(variant)
            self.assertTrue(torch.equal(head.a_name, correct.a_name))
            self.assertTrue(torch.equal(head.o_name, correct.o_name))
            self.assertTrue(torch.equal(head.a_knowledge, correct.a_knowledge[head.permutations['a']]))
            self.assertTrue(all(i != p for i, p in enumerate(head.permutations['a'])))

    def test_06_fold_identity_and_coverage(self):
        fit, hold, report = runtime.train_fold(self.caches['train'], self.ids['train'])
        repeat = runtime.train_fold(self.caches['train'], self.ids['train'])
        self.assertTrue(torch.equal(fit, repeat[0]) and torch.equal(hold, repeat[1]))
        self.assertFalse(set(fit.tolist()) & set(hold.tolist()))
        self.assertEqual(sorted(fit.tolist() + hold.tolist()), list(range(96)))
        self.assertTrue(all(x['fit'] and x['hold'] for x in report))

    def test_07_singleton_fold_fails(self):
        value = {k: v[:1] for k, v in self.caches['train'].items()}
        with self.assertRaises(ValueError):
            runtime.train_fold(value, ['singleton'])

    def test_08_data_order_independent_of_head_rng(self):
        indices = torch.arange(96)
        expected = runtime.epoch_order(indices, 1)
        self.head('P_generic')
        self.assertTrue(torch.equal(expected, runtime.epoch_order(indices, 1)))
        self.assertFalse(torch.equal(expected, runtime.epoch_order(indices, 2)))

    def test_09_rng_safe_serialization(self):
        runtime.fresh_head(self.bank, 'P_K', self.controls, 'cpu')
        path = self.work / 'rng.pt'
        tensor_save(path, runtime.rng_state())
        expected = (random.random(), float(np.random.rand()), torch.rand(4))
        runtime.restore_rng(tensor_load(path))
        actual = (random.random(), float(np.random.rand()), torch.rand(4))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_10_optimizer_resume_selfcheck_cpu_only(self):
        result = pilot.cuda_resume_check(self.bank, self.controls, self.caches['train'], self.settings, 'cpu')
        self.assertTrue(result['passed'])
        self.assertEqual(result['diagnostic_optimizer_step_calls'], 3)

    def test_11_all_five_heads_mechanics_and_formal(self):
        mechanics = pilot.mechanics(self.work, self.caches, self.ids, self.bank, self.controls, self.settings,
                                    'synthetic_protocol', self.initializations, 'cpu')
        self.assertTrue(mechanics['passed'])
        reused = pilot.mechanics(self.work, self.caches, self.ids, self.bank, self.controls, self.settings,
                                 'synthetic_protocol', self.initializations, 'cpu')
        self.assertEqual(mechanics, reused)
        for variant in VARIANTS:
            result = self.run_variant(self.work, variant)
            self.assertTrue(result['selected_checkpoint_replay_exact'])
            self.assertEqual(result['formal_optimizer_steps'], 12)
            history = read_json(self.work / 'runs' / variant / 'history.json')
            self.assertEqual(len(history['ledger']), 3)
            self.assertEqual(history['ledger'][0]['conditional_diagnostics']['max_abs_delta'], 0)
        histories = [read_json(self.work / 'runs' / v / 'history.json') for v in VARIANTS]
        self.assertEqual(len({h['ledger'][1]['train_epoch']['epoch_order_sha256'] for h in histories}), 1)

    def test_12_mid_epoch_failure_resumes_exact(self):
        expected_work, interrupted = self.work / 'expected', self.work / 'interrupted'
        self.run_variant(expected_work)
        original = pilot.train_epoch
        calls = [0]
        def fail_after_update(*args, **kwargs):
            calls[0] += 1
            result = original(*args, **kwargs)
            if calls[0] == 2:
                raise RuntimeError('Synthetic interruption after optimizer work, before commit')
            return result
        with patch.object(pilot, 'train_epoch', side_effect=fail_after_update):
            with self.assertRaises(RuntimeError):
                self.run_variant(interrupted)
        self.assertEqual(tensor_load(interrupted / 'runs/P_K/last.pt')['completed_epoch'], 1)
        self.run_variant(interrupted)
        left = tensor_load(expected_work / 'runs/P_K/last.pt')
        right = tensor_load(interrupted / 'runs/P_K/last.pt')
        self.assertTrue(all(torch.equal(value, right['head_state'][k]) for k, value in left['head_state'].items()))
        for l, r in zip(left['ledger'], right['ledger']):
            self.assertEqual(l['metrics_raw'], r['metrics_raw'])
            self.assertTrue(np.array_equal(np.load(expected_work / 'runs/P_K' / l['path'] / 'grid_scores.npy'),
                                           np.load(interrupted / 'runs/P_K' / r['path'] / 'grid_scores.npy')))

    def test_13_derived_history_repair(self):
        self.run_variant(self.work)
        history = self.work / 'runs/P_K/history.json'
        write_json(history, dict(synthetic_corrupted_derived_file=True))
        self.run_variant(self.work)
        self.assertEqual(read_json(history)['completed_epoch'], 2)

    def test_14_checkpoint_identity_rejected(self):
        self.run_variant(self.work)
        checkpoint = self.work / 'runs/P_K/last.pt'
        value = tensor_load(checkpoint)
        value['protocol_sha256'] = 'changed'
        tensor_save(checkpoint, value)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.run_variant(self.work)

    def test_15_epoch_artifact_tamper_rejected(self):
        self.run_variant(self.work)
        history = read_json(self.work / 'runs/P_K/history.json')
        path = self.work / 'runs/P_K' / history['ledger'][1]['path'] / 'evaluation.json'
        write_json(path, dict(synthetic_tamper=True))
        with self.assertRaisesRegex(ValueError, 'HASH MISMATCH'):
            self.run_variant(self.work)

    def test_16_owned_path_escape_rejected(self):
        for name in ('../outside.pt', '/outside.pt', 'C:/outside.pt', 'a\\b.pt'):
            with self.assertRaises(ValueError):
                pilot.owned_child(self.work, name)

    def test_17_budget_gate_and_single_seed_limit(self):
        baseline = {s: {k: .4 for k in runtime.FIELDS} for s in ('cw', 'ow')}
        results = {v: dict(selected=dict(metrics_raw=copy.deepcopy(baseline))) for v in VARIANTS}
        self.assertFalse(pilot.decision(results, baseline, self.ref['budget_gate'])['pilot_budget_gate_passed'])
        results['P_K']['selected']['metrics_raw']['cw']['AUC'] += .004
        decision = pilot.decision(results, baseline, self.ref['budget_gate'])
        self.assertTrue(decision['pilot_budget_gate_passed'])
        self.assertTrue(decision['not_statistical_significance'])
        results['P_K']['selected']['metrics_raw']['ow']['AUC'] -= .01
        self.assertFalse(pilot.decision(results, baseline, self.ref['budget_gate'])['pilot_budget_gate_passed'])

    def export_fixture(self, failed=False):
        release, work = self.work / 'release', self.work / 'results'
        release.mkdir()
        work.mkdir()
        write_json(release / 'reference.json', self.ref)
        write_json(release / 'release_manifest.json', {'reference.json': sha(release / 'reference.json')})
        write_json(work / 'protocol.json', dict(schema='conditional_attribute_pilot_v1'))
        write_json(work / 'execution_status.json', dict(state='failed' if failed else 'complete', protocol_sha256='fixture'))
        write_json(work / 'summary.json', dict(passed=True, protocol_sha256='fixture', decision=dict(pilot_budget_gate_passed=False)))
        destination = self.work / 'export.zip'
        with patch.object(exporter, 'ROOT', release):
            manifest = exporter.export_work(work, destination)
        return manifest, work, destination, release

    def test_18_export_and_stale_success_not_claimed(self):
        manifest, _, destination, _ = self.export_fixture(failed=True)
        self.assertFalse(manifest['execution_passed'])
        self.assertIsNone(manifest['pilot_budget_gate_passed'])
        self.assertEqual(manifest, exporter.verify_zip(destination))

    def test_19_export_preserves_existing_zip(self):
        _, work, destination, release = self.export_fixture()
        expected = sha(destination)
        with patch.object(exporter, 'ROOT', release):
            with self.assertRaises(FileExistsError):
                exporter.export_work(work, destination)
        self.assertEqual(expected, sha(destination))

    def test_20_export_refuses_backbone_cache(self):
        self.assertFalse(exporter.allowed_work_file('train.pt'))
        self.assertFalse(exporter.allowed_work_file('runs/P_K/backbone.pt'))
        self.assertTrue(exporter.allowed_work_file('runs/P_K/last.pt'))
        self.assertTrue(exporter.allowed_work_file('runs/P_K/epochs/epoch_0001/attempt_0000/grid_scores.npy'))

    def test_21_zip_tamper_detected(self):
        _, _, destination, _ = self.export_fixture()
        with destination.open('ab') as handle:
            handle.write(b'synthetic_tamper')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            exporter.verify_zip(destination)

    def test_22_trained_best_and_first_tie_selection(self):
        original = pilot.evaluate_epoch
        def controlled_selection(*args, **kwargs):
            report = original(*args, **kwargs)
            epoch = report['epoch']
            # Synthetic selection fixture only; still save/replay actual nonzero head scores.
            report['metrics_raw']['cw']['AUC'] = .40 if epoch == 0 else .41
            report['metrics_raw']['ow']['AUC'] = .40 + .01 * epoch
            return report
        with patch.object(pilot, 'evaluate_epoch', side_effect=controlled_selection):
            result = self.run_variant(self.work)
            self.assertEqual(result['best_epoch'], 1)
            repeated = self.run_variant(self.work)
            self.assertEqual(repeated['best_epoch'], 1)
        selected = tensor_load(self.work / 'runs/P_K/best_auc.pt')
        self.assertGreater(float(selected['head_state']['output.weight'].abs().sum()), 0)
        self.assertTrue(result['selected_checkpoint_replay_exact'])

    def test_23_pairing_truth_mismatch_refused(self):
        _, _, vectors = runtime.scores_and_diagnostics(self.head(), self.caches['val'], 'cpu')
        changed = copy.deepcopy(vectors)
        changed['objects'][0] = (changed['objects'][0] + 1) % 12
        with self.assertRaises(ValueError):
            runtime.paired_predictions(vectors, changed)

    def test_24_torch113_weights_only_avoids_binfloat(self):
        path = self.work / 'compat.pt'
        payload = dict(value=1.25, nested=(2.5, [3.75]), tensor=torch.tensor([4.0]))
        tensor_save(path, payload)
        self.assertEqual(tensor_load(path)['nested'], payload['nested'])
        import zipfile
        with zipfile.ZipFile(path) as archive:
            pickle_name = next(name for name in archive.namelist() if name.endswith('/data.pkl'))
            operations = [op.name for op, _, _ in pickletools.genops(archive.read(pickle_name))]
        self.assertNotIn('BINFLOAT', operations)
        self.assertNotIn('FLOAT', operations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    global EVALUATOR
    if sha(args.evaluator) != '24b48c4d48e3475baec0a656fca194cdbc6d4b4e32e87c0f22cda3daf586d978':
        raise ValueError('Use the verified original evaluator, not a rewritten metric')
    spec = importlib.util.spec_from_file_location('pilot_test_native_evaluator', args.evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    EVALUATOR = module.Evaluator
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PilotTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(args.out, dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                             errors=[(str(t), error) for t, error in result.errors],
                             failures=[(str(t), error) for t, error in result.failures],
                             scope='CPU synthetic mechanics, five 2-epoch heads, exact interrupted resume, Torch 1.13-safe checkpoint format, export guards',
                             real_cuda_pilot_run=False, python=sys.version, torch=str(torch.__version__),
                             native_evaluator_sha256=sha(args.evaluator)))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
