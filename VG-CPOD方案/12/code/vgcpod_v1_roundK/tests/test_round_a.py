"""Loss derivation, branch schedules, six configurations, and cache-only integration."""
import argparse
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import torch

from vgcpod import engine
from vgcpod.common import ROOT, load_torch, read_json
from tests.test_math import fixture, synthetic_cache

torch.set_num_threads(1)


class RoundAMathTests(unittest.TestCase):
    def test_loss_identity_and_gradients_at_each_beta(self):
        for beta in (.5, .25, .1):
            model, cfg, x, y, mask = fixture('V1')
            model.cfg['task_aux_beta'] = beta
            model.double()
            x = tuple(t.double() for t in x)
            mask[0, 2] = False
            # Keep fixture truths inside the training support.
            mask[1, 2] = False
            attrs, objs = y
            attrs = torch.where(attrs == 2, torch.zeros_like(attrs), attrs)
            with torch.no_grad():
                model.Wc.normal_(0, .04)
            out = model(*x)
            terms = model.objective(out, attrs, objs, mask)
            equivalent = 1.5 * terms['pair'] + (1.5 * beta / (1 + beta)) * terms['support']
            self.assertTrue(torch.allclose(terms['task'], equivalent, atol=1e-12))
            params = [p for p in model.parameters() if p.requires_grad]
            a = torch.autograd.grad(terms['task'], params, retain_graph=True)
            b = torch.autograd.grad(equivalent, params)
            for left, right in zip(a, b):
                self.assertTrue(torch.allclose(left, right, atol=1e-11, rtol=1e-9))
            if beta == .5:
                legacy = terms['pair'] + .5*terms['cond'] + .5*terms['object']
                self.assertTrue(torch.equal(terms['task'], legacy))

    def test_optimizer_parameter_membership_and_cosine_endpoints(self):
        for variant in ('V0', 'V1'):
            for scale in (.1, .3, 1.):
                model, cfg, _, _, _ = fixture(variant)
                cfg['cp_lr_scale'] = scale
                opt = engine.make_optimizer(model, cfg)
                grouped = [id(p) for g in opt.param_groups for p in g['params']]
                expected = [id(p) for p in model.parameters() if p.requires_grad]
                self.assertCountEqual(grouped, expected)
                self.assertEqual(len(set(grouped)), len(grouped))
                for step, expected_base in [(0, cfg['lr']), (9, cfg['min_lr'])]:
                    engine.set_learning_rates(opt, cfg, step, 10)
                    for g in opt.param_groups:
                        self.assertAlmostEqual(g['lr'], expected_base * g['lr_scale'], places=14)
                self.assertEqual(len(opt.param_groups), 1 if variant == 'V0' else 2)

    def test_default_grouping_preserves_legacy_updates(self):
        for variant in ('V0', 'V1'):
            model, cfg, x, y, mask = fixture(variant)
            legacy, _, _, _, _ = fixture(variant)
            grouped = engine.make_optimizer(model, cfg)
            flat = torch.optim.AdamW([p for p in legacy.parameters() if p.requires_grad], lr=cfg['lr'],
                                    betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'],
                                    weight_decay=cfg['weight_decay'])
            for step in range(5):
                eta = engine.set_learning_rates(grouped, cfg, step, 5)
                flat.param_groups[0]['lr'] = eta
                for m, opt in [(model, grouped), (legacy, flat)]:
                    opt.zero_grad(set_to_none=True)
                    loss = m.objective(m(*x), *y, mask)['total']
                    loss.backward()
                    engine.clip_total_norm([p for p in m.parameters() if p.requires_grad], cfg['gradient_clip'])
                    opt.step()
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, legacy.state_dict()[name]), (variant, name))

    def test_diagnostics_measure_full_support_without_calibration(self):
        model, _, x, y, _ = fixture('V1')
        out = model(*x)
        labels = {'attrs': ['a', 'b', 'c'], 'objs': ['x', 'y']}
        pairs = {'train': [['a', 'x'], ['b', 'y']]}
        a, o = y
        d = engine.distribution_diagnostics(out['joint'].detach().numpy(), out['cond'].detach().numpy(),
                                           a.numpy(), o.numpy(), labels, pairs)
        p = out['joint'].exp()
        ref = (p[:, 0, 0] + p[:, 1, 1]).mean().item()
        self.assertAlmostEqual(d['support_mass_all'], ref, places=6)
        self.assertTrue(0 <= d['gt_object_attr_acc_unseen'] <= 100)


class RoundAIntegrationTests(unittest.TestCase):
    def test_six_configs_train_export_resume_and_summarize(self):
        spec = importlib.util.spec_from_file_location('round_a_launcher', ROOT / 'round_a.py')
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        with tempfile.TemporaryDirectory(prefix='vgcpod_rounda_') as t:
            root = Path(t)
            small = synthetic_cache(root / 'cache')
            for case in launch.CASES:
                real = read_json(ROOT / 'configs/round_a' / (case + '.json'))
                cfg = dict(real, rank=small['rank'], adapter_rank=small['adapter_rank'], workers=0,
                           batch_size=8, eval_batch_size=8, epochs=2)
                args = argparse.Namespace(cache=str(root / 'cache'), out=str(root / case), seed=3351,
                                          device='cpu', resume=False)
                engine.train(args, cfg)
                run = root / case / 'ut-zap50k' / cfg['variant'] / 'seed3351'
                h = read_json(run / 'history.json')
                self.assertIn('val_ow', h[-1])
                self.assertIn('support_mass_unseen', h[-1]['val_diagnostics'])
                self.assertTrue(0 <= h[-1]['train']['clipped_step_fraction'] <= 1)
                self.assertAlmostEqual(h[-1]['train']['centered_delta_rms'] ** 2,
                                       h[-1]['train']['centered_delta_mse'], places=12)
                if cfg['variant'] == 'V1':
                    self.assertAlmostEqual(h[-1]['lr_groups']['cp'] / h[-1]['lr_groups']['adapter'],
                                           cfg['cp_lr_scale'], places=12)
                before = load_torch(run / 'checkpoints/last.pt')
                args.resume = True
                engine.train(args, cfg)
                after = load_torch(run / 'checkpoints/last.pt')
                for name in before['model']:
                    self.assertTrue(torch.equal(before['model'][name], after['model'][name]))
                self.assertTrue(launch.validate_completed(run, cfg, 3351, root / 'cache'))
            rows = launch.summarize(root)
            self.assertEqual(len(rows), 6)
            self.assertEqual(next(r for r in rows if r['case'] == 'A6')['variant'], 'V0')
            self.assertTrue((root / 'round_a_summary.csv').is_file())
            jobs = list(launch.ORDER)
            self.assertEqual([jobs[i::3] for i in range(3)], [['A1', 'A2'], ['A3', 'A4'], ['A6', 'A5']])


if __name__ == '__main__':
    unittest.main(verbosity=2)
