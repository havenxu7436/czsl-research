"""Local CPU synthetic checks, explicitly not real-data CUDA validation."""
import ast
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import ROOT, new_output, read_json, sha, state_sha, tensor_save, verify_files
from export_results import verify_zip
from head import (CAP, TEMPERATURE, VARIANTS, ConditionalHead, conditional_loss,
                  corrected_scores, derangement, inputs, probability_checks)
from probe_bf import geometry


def bank_fixture():
    g = torch.Generator().manual_seed(37)
    return {axis: dict(units=torch.nn.functional.normalize(torch.randn(n, 4, 300, generator=g), dim=-1),
                       mask=torch.ones(n, 4)) for axis, n in (('a', 16), ('o', 12))}


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(49)
        self.bank = bank_fixture()
        self.x = torch.randn(9, 300)
        self.base = torch.randn(9, 12, 16) * 0.3
        self.a = torch.arange(9) % 16
        self.o = torch.arange(9) % 12

    def test_exact_zero_and_off(self):
        for v in VARIANTS:
            model = ConditionalHead(self.bank, v)
            self.assertTrue(torch.equal(model(self.x, self.base), self.base))
            self.assertTrue(model(self.x, self.base, force_off=True) is self.base)

    def test_float64_probability_identity(self):
        for scale in (0.01, 1, 20):
            base = self.base.double() * scale
            delta = torch.randn_like(base) * CAP
            errors = probability_checks(base, delta)
            self.assertLess(max(errors.values()), 2e-12)

    def test_marginal_preserved_and_full_grid_required(self):
        delta = torch.randn_like(self.base) * CAP
        self.assertLess(max(probability_checks(self.base, delta).values()), 3e-6)
        with self.assertRaises(ValueError):
            corrected_scores(self.base[:, 0], delta[:, 0])

    def test_controls_names_fixed_derangements_fixed_rng(self):
        for variant in ('P_perm1', 'P_perm2'):
            torch.manual_seed(456)
            before = torch.get_rng_state().clone()
            values, permutations = inputs(self.bank, variant)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            for axis in ('a', 'o'):
                original = self.bank[axis]['units']
                p = permutations[axis]
                self.assertEqual(sorted(p), list(range(len(p))))
                self.assertTrue(all(i != j for i, j in enumerate(p)))
                self.assertTrue(torch.equal(values[axis + '_name'], original[:, 0]))
                self.assertTrue(torch.equal(values[axis + '_knowledge'], original[:, 1:].mean(1)[p]))
        self.assertNotEqual(inputs(self.bank, 'P_perm1')[1], inputs(self.bank, 'P_perm2')[1])

    def test_matched_common_initialization_and_name_slots(self):
        reference = None
        for variant in VARIANTS:
            torch.manual_seed(0)
            model = ConditionalHead(self.bank, variant)
            current = model.state_dict()
            if reference is None:
                reference = copy.deepcopy(current)
            else:
                for name, value in reference.items():
                    self.assertTrue(torch.equal(current[name], value))
        values, _ = inputs(self.bank, 'P_name')
        for axis in ('a', 'o'):
            self.assertTrue(torch.equal(values[axis + '_knowledge'], values[axis + '_name']))

    def test_nonzero_output_gradient_and_all_parameters_active(self):
        for variant in VARIANTS:
            torch.manual_seed(0)
            model = ConditionalHead(self.bank, variant)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            active = set()
            for i in range(4):
                optimizer.zero_grad(set_to_none=True)
                loss = conditional_loss(self.base, model.evidence(self.x), self.a, self.o)
                loss.backward()
                self.assertGreater(float(model.output.weight.grad.norm()), 0)
                for name, p in model.named_parameters():
                    self.assertTrue(torch.isfinite(p.grad).all())
                    if p.grad.norm() > 0:
                        active.add(name)
                optimizer.step()
            self.assertEqual(active, set(dict(model.named_parameters())))
            delta = model.evidence(self.x)
            self.assertGreater(float(delta.std(1, unbiased=False).mean()), 1e-9)
            self.assertLessEqual(float(delta.abs().max()), CAP)

    def test_resume_optimizer_exact_next_step(self):
        model = ConditionalHead(self.bank, 'P_K')
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        def step(m, o):
            o.zero_grad(set_to_none=True)
            conditional_loss(self.base, m.evidence(self.x), self.a, self.o).backward()
            o.step()
        step(model, opt)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'last.pt'
            tensor_save(path, dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=torch.get_rng_state()))
            saved = torch.load(path, map_location='cpu', weights_only=True)
            step(model, opt)
            expected = state_sha(model)
            resumed = ConditionalHead(self.bank, 'P_K')
            resumed.load_state_dict(saved['model'])
            resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-3, weight_decay=1e-4)
            resumed_opt.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['rng'])
            step(resumed, resumed_opt)
            self.assertEqual(expected, state_sha(resumed))

    def test_conditional_ce_all_attributes(self):
        model = ConditionalHead(self.bank, 'P_K')
        delta = model.evidence(self.x)
        expected = torch.nn.functional.cross_entropy(
            (self.base + delta)[torch.arange(9), self.o] / TEMPERATURE, self.a)
        self.assertTrue(torch.equal(expected, conditional_loss(self.base, delta, self.a, self.o)))

    def test_gradient_geometry(self):
        vector = torch.tensor([3., 4.])
        self.assertAlmostEqual(geometry(vector, -2 * vector)['cosine'], -1)
        self.assertAlmostEqual(geometry(vector, -2 * vector)['right_over_left_norm'], 2)
        self.assertIsNone(geometry(vector, torch.zeros_like(vector))['cosine'])

    def test_output_protection_and_hash_mismatch(self):
        ref = dict(c_source='/var/tmp/old_c/src', bf_source='/var/tmp/old_bf/src')
        for path in ('/var/tmp/old_c/new', '/data/new', str(ROOT / 'new')):
            with self.assertRaises(ValueError):
                new_output(path, ref, free_mib=0)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                new_output(d, ref, free_mib=0)
            path = Path(d) / 'fixture.json'
            path.write_text('{}', encoding='utf-8')
            self.assertEqual(verify_files({str(path): sha(path)}), 1)
            with self.assertRaises(ValueError):
                verify_files({str(path): '0' * 64})

    def test_zip_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'test.zip'
            data = b'{}'
            manifest = dict(files={'evidence.json': dict(size=2, sha256=hashlib.sha256(data).hexdigest())})
            with zipfile.ZipFile(p, 'x') as z:
                z.writestr('evidence.json', data)
                z.writestr('manifest.json', json.dumps(manifest))
            verify_zip(p)
            q = Path(d) / 'bad.zip'
            with zipfile.ZipFile(q, 'x') as z:
                z.writestr('evidence.json', b'[]')
                z.writestr('manifest.json', json.dumps(manifest))
            with self.assertRaises(ValueError):
                verify_zip(q)

    def test_real_head_smoke_on_synthetic_cache(self):
        # Exercise the real checker with CPU aliases; do not claim real CUDA.
        from unittest.mock import patch
        from prepare import head_smoke
        train = dict(features=torch.randn(16, 300), scores=torch.randn(16, 12, 16) * .3,
                     attrs=torch.arange(16), objects=torch.arange(16) % 12)
        with patch.object(torch.Tensor, 'cuda', lambda value, *a, **k: value), \
             patch.object(torch.nn.Module, 'cuda', lambda value, *a, **k: value):
            report = head_smoke(train, self.bank)
        self.assertEqual(set(report), set(VARIANTS))
        self.assertEqual(report['P_generic']['active_parameter_count'] - report['P_K']['active_parameter_count'], 3104)

    def test_python38_syntax(self):
        for path in ROOT.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))

    def test_reference_matches_download_and_no_scores_claim(self):
        ref = read_json(ROOT / 'reference.json')
        self.assertEqual(ref['preflight_zip_sha256'], '8493967eb5479cb7af0917d8ca5f590f932c9a7aed22a0878613f8d44b265107')
        self.assertEqual(ref['c_reference']['best_epoch'], 6)
        report = read_json(ROOT / 'audit_review.json')
        self.assertFalse(report['independent_scores_recomputed'])
        for diffs in report['paired_config_differences'].values():
            self.assertEqual(set(x['field'] for x in diffs), {'MODEL.kcc_logic', 'MODEL.kcc_beta'})

    def test_native_evaluator_cached_grid_mapping(self):
        from unittest.mock import patch
        from prepare import evaluate_cached
        import sys
        source = ROOT.parent / 'output/BF_conditional_preflight_20261009/extracted/conditional/source/evaluator_ge.py'
        if not source.is_file():
            source = Path(read_json(ROOT / 'reference.json')['c_source']) / 'evaluator_ge.py'
        self.assertEqual(sha(source), '24b48c4d48e3475baec0a656fca194cdbc6d4b4e32e87c0f22cda3daf586d978')
        spec = importlib.util.spec_from_file_location('checked_native_evaluator', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        attrs, objs = ['a0', 'a1', 'a2'], ['o0', 'o1']
        pairs = [(a, o) for a in attrs for o in objs]
        ds = SimpleNamespace(attrs=attrs, objs=objs, pairs=pairs, train_pairs=[pairs[0], pairs[3]],
                             val_pairs=[pairs[4], pairs[5]], phase='val', open_world=True,
                             attr2idx={a: i for i, a in enumerate(attrs)}, obj2idx={o: i for i, o in enumerate(objs)},
                             pair2idx={p: i for i, p in enumerate(pairs)})
        a = torch.tensor([0, 1, 2, 2] * 5)
        o = torch.tensor([0, 1, 0, 1] * 5)
        grid = torch.randn(20, 2, 3) * .01
        grid[torch.arange(20), o, a] += 1
        cache = dict(scores=grid, attrs=a, objects=o)
        with patch.dict(sys.modules, {'evaluator_ge': module}):
            metrics, scores, labels = evaluate_cached(cache, ds)
        self.assertEqual(tuple(scores.shape), (20, 6))
        self.assertTrue(torch.equal(scores, grid.permute(0, 2, 1).reshape(20, 6)))
        self.assertEqual(labels.tolist(), [0, 3, 4, 5] * 5)
        self.assertEqual(metrics['best_seen'], 1)
        self.assertEqual(metrics['best_unseen'], 1)

    def test_export_omits_caches_preserves_failure_and_refuses_overwrite(self):
        from unittest.mock import patch
        import sys
        from export_results import main
        from common import write_json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base, probe, logs = root / 'base', root / 'probe', root / 'logs'
            for p in (base, probe, logs):
                p.mkdir()
            write_json(base / 'failure.json', dict(error='synthetic replay mismatch'))
            write_json(probe / 'probe.json', dict(passed=True))
            tensor_save(base / 'train.pt', torch.ones(2))
            tensor_save(base / 'val.pt', torch.ones(2))
            archive = root / 'diagnostic.zip'
            argv = ['export_results.py', '--base', str(base), '--probe', str(probe), '--logs', str(logs), '--out', str(archive)]
            with patch.object(sys, 'argv', argv), patch('export_results.verify_release', lambda: 'fixture'):
                main()
                with self.assertRaises(FileExistsError):
                    main()
            with zipfile.ZipFile(archive) as z:
                self.assertFalse(any(n.endswith('.pt') for n in z.namelist()))
                self.assertIn('stageA/failure.json', z.namelist())
                self.assertNotIn('stageA/prepared.json', z.namelist())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun, torch=torch.__version__,
                  python=__import__('sys').version, scope='CPU synthetic tests only; no real GPU/data/efficacy validation',
                  failures=[name.id() for name, _ in result.failures + result.errors])
    from common import write_json
    write_json(ROOT / 'local_test_report.json', report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
