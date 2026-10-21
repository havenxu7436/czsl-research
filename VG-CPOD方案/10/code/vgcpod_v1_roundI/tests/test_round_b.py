"""Readout/gate interventions, affine gate effect, gradients, training and paired exports."""
import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from vgcpod import engine
from vgcpod.common import ROOT, load_torch, read_json
from vgcpod.model import VGCPOD
from tests.test_math import fixture, synthetic_cache

torch.set_num_threads(1)


def bfixture(case):
    reference, small, x, y, mask = fixture('V1')
    cfg = read_json(ROOT / 'configs/round_b' / (case + '.json'))
    cfg.update(rank=small['rank'], adapter_rank=small['adapter_rank'], workers=0,
               batch_size=8, eval_batch_size=8, epochs=2)
    model = VGCPOD(reference.ea, reference.eo, reference.ec, cfg, 3351).double()
    return model, cfg, tuple(t.double() for t in x), y, mask


class RoundBMathTests(unittest.TestCase):
    def test_zero_start_and_separate_feature_interventions(self):
        for case in ('B1', 'B2', 'B3'):
            model, cfg, (g, fa, fo), y, mask = bfixture(case)
            initial = model(g, fa, fo)
            self.assertTrue(torch.equal(initial['cond'], initial['base']))
            with torch.no_grad():
                model.Wc.normal_(0, .05)
            out = model(g, fa, fo)
            changed = model(g, fa, fo.roll(1, 0))
            self.assertTrue(torch.equal(out['obj'], changed['obj']))
            self.assertTrue(torch.equal(out['base'], changed['base']))
            self.assertFalse(torch.allclose(out['cond'], changed['cond']))
            if case == 'B1':
                self.assertTrue(torch.equal(out['gamma'], torch.ones_like(out['gamma'])))
                self.assertFalse(model.Bvis.requires_grad)
            else:
                # Local gates depend on fo, but not on the global feature used for readout.
                self.assertTrue(torch.equal(out['gamma'], model(g.roll(1, 0), fa, fo)['gamma']))
                self.assertFalse(torch.allclose(out['gamma'], changed['gamma']))
                self.assertTrue(model.Bvis.requires_grad)

    def test_eta_interpolation_and_static_limit(self):
        full, _, x, y, mask = bfixture('B2')
        weak, _, _, _, _ = bfixture('B3')
        static, _, _, _, _ = bfixture('B2')
        with torch.no_grad():
            full.Wc.normal_(0, .1)
        weak.load_state_dict(full.state_dict())
        static.load_state_dict(full.state_dict())
        static.gate_eta = 0
        f, w, s = [m(*x) for m in (full, weak, static)]
        self.assertTrue(torch.allclose(w['cond'], .75 * s['cond'] + .25 * f['cond'], atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.equal(s['gamma'], torch.ones_like(s['gamma'])))
        self.assertTrue(torch.all(w['gamma'] >= .75) and torch.all(w['gamma'] <= 1.25))
        a = torch.autograd.grad(w['cond'][0, 0, 0], weak.Bvis)[0]
        b = torch.autograd.grad(f['cond'][0, 0, 0], full.Bvis)[0]
        self.assertTrue(torch.allclose(a, .25 * b, atol=1e-12, rtol=1e-12))

    def test_operator_rank_and_full_matrix_readout(self):
        for case in ('B1', 'B2', 'B3'):
            model, cfg, (g, fa, fo), y, mask = bfixture(case)
            with torch.no_grad():
                model.Wc.normal_(0, .05)
            out = model(g, fa, fo)
            for b, j, i in [(0, 0, 0), (2, 1, 2), (6, 0, 1)]:
                v = model.Av @ model.eo[j]
                operator = model.Wc @ torch.diag(v * out['gamma'][b, j]) @ model.Au
                h = g[b]
                if case == 'B1':
                    h = .5 * g[b] + .5 * fo[b, j]
                    h = h / h.norm().clamp_min(1e-12)
                direct = cfg['rho'] * cfg['kappa'] * h @ operator @ model.ea[i]
                self.assertTrue(torch.allclose(out['cond'][b, j, i] - out['base'][b, j, i], direct, atol=1e-12))
                self.assertLessEqual(int(torch.linalg.matrix_rank(operator)), cfg['rank'])
            self.assertTrue(torch.allclose(out['joint'].exp().sum((1, 2)), torch.ones(len(g), dtype=g.dtype), atol=1e-12))

    def test_b3_task_gradient_finite_difference_and_optimizer(self):
        model, cfg, x, y, mask = bfixture('B3')
        opt = engine.make_optimizer(model, cfg)
        model.objective(model(*x), *y, mask)['task'].backward()
        self.assertGreater(float(model.Wc.grad.norm()), 0)
        self.assertEqual(float(model.Bvis.grad.norm()), 0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.objective(model(*x), *y, mask)['task'].backward()
        for name in ('Au', 'Av', 'Bvis', 'Wc'):
            self.assertGreater(float(getattr(model, name).grad.norm()), 0, name)
        flat = model.Bvis.view(-1)
        k = int(model.Bvis.grad.abs().argmax())
        analytic = float(model.Bvis.grad.view(-1)[k])
        with torch.no_grad():
            original = flat[k].item()
            flat[k] = original + 1e-5
            plus = float(model.objective(model(*x), *y, mask)['task'])
            flat[k] = original - 1e-5
            minus = float(model.objective(model(*x), *y, mask)['task'])
            flat[k] = original
        self.assertAlmostEqual((plus - minus) / 2e-5, analytic, delta=max(1e-10, abs(analytic) * 1e-4))
        cp = next(group for group in opt.param_groups if group['name'] == 'cp')
        self.assertIn(id(model.Bvis), [id(p) for p in cp['params']])
        engine.set_learning_rates(opt, cfg, 0, 10)
        self.assertAlmostEqual(cp['lr'], 3e-4)
        engine.set_learning_rates(opt, cfg, 9, 10)
        self.assertAlmostEqual(cp['lr'], 3e-6)


class RoundBIntegrationTests(unittest.TestCase):
    def test_three_cases_resume_and_reference_pair_exports(self):
        spec = importlib.util.spec_from_file_location('round_b_launcher', ROOT / 'round_b.py')
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        with tempfile.TemporaryDirectory(prefix='vgcpod_roundb_') as temp:
            root = Path(temp)
            small = synthetic_cache(root / 'cache')
            for case in ('A5', 'A6', 'B1', 'B2', 'B3'):
                config = ROOT / ('configs/round_a/' if case.startswith('A') else 'configs/round_b/') / (case + '.json')
                cfg = read_json(config)
                cfg.update(rank=small['rank'], adapter_rank=small['adapter_rank'], workers=0,
                           batch_size=8, eval_batch_size=8, epochs=2)
                args = argparse.Namespace(cache=str(root / 'cache'), out=str(root / case), seed=3351,
                                          device='cpu', resume=False)
                engine.train(args, cfg)
                run = root / case / 'ut-zap50k' / cfg['variant'] / 'seed3351'
                if case.startswith('B'):
                    self.assertTrue(launch.validate_completed(run, cfg, 3351, root / 'cache'))
                    h = read_json(run / 'history.json')
                    variance = h[-1]['train']['gamma_image_variance']
                    self.assertEqual(variance, 0) if case == 'B1' else self.assertGreater(variance, 0)
                if case == 'B3':
                    state = load_torch(run / 'checkpoints/last.pt')
                    args.out = str(root / 'interrupted')
                    real_predict = engine.predict
                    calls = [0]
                    def interrupted(*a, **kw):
                        calls[0] += 1
                        if calls[0] == 2:
                            raise InterruptedError('Interrupt after a committed epoch')
                        return real_predict(*a, **kw)
                    with mock.patch.object(engine, 'predict', side_effect=interrupted):
                        with self.assertRaises(InterruptedError):
                            engine.train(args, cfg)
                    args.resume = True
                    engine.train(args, cfg)
                    resumed = load_torch(root / 'interrupted/ut-zap50k/V1/seed3351/checkpoints/last.pt')
                    for name in state['model']:
                        self.assertTrue(torch.equal(state['model'][name], resumed['model'][name]), name)
            rows = launch.summarize(root, reference_out=root)
            self.assertEqual(len(rows), 3)
            summary = read_json(root / 'round_b_summary.json')
            self.assertEqual(len(summary['paired_deltas']), 8)
            self.assertGreater(summary['per_pair_rows'], 0)
            self.assertTrue((root / 'per_pair_gains.csv').is_file())
            with np.load(root / 'B3/ut-zap50k/V1/seed3351/predictions/test_ow.npz') as data:
                np.testing.assert_allclose(data['probabilities'].sum(1), 1, atol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
