"""Mathematical and full CLI-path tests; no downloaded model/data needed."""
import argparse
import csv
import copy
import importlib.util
import tempfile
import unittest
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from vgcpod.common import ROOT, load_torch, read_json, sha, unit, write_json
from vgcpod.features import query_patches, visual_tokens
from vgcpod.data import load_cache
from vgcpod.metrics import evaluate_scores, exact_curve, native_curve
from vgcpod.model import VARIANTS, VGCPOD, js_divergence

torch.set_num_threads(1)


def fixture(variant='V5', d=12, rank=3):
    torch.manual_seed(83)
    cfg = read_json(ROOT / 'configs' / 'default.json')
    cfg.update(variant=variant, rank=rank, adapter_rank=2, workers=0, batch_size=8,
               eval_batch_size=8, epochs=2)
    ea, eo, ec = unit(torch.randn(3, d)), unit(torch.randn(2, d)), unit(torch.randn(2, 3, d))
    model = VGCPOD(ea, eo, ec, cfg, 3351)
    g, fa, fo = unit(torch.randn(8, d)), unit(torch.randn(8, 3, d)), unit(torch.randn(8, 2, d))
    attrs, objs = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]), torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    mask = torch.ones(2, 3, dtype=torch.bool)
    return model, cfg, (g, fa, fo), (attrs, objs), mask


class FormulaTests(unittest.TestCase):
    def test_gradient_clip_matches_equation(self):
        from vgcpod.engine import clip_total_norm
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        q = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        p.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        q.grad = torch.tensor([12.0], dtype=torch.float64)
        norm = clip_total_norm([p, q], 1.0)
        self.assertEqual(float(norm), 13.0)
        expected = torch.tensor([3.0, 4.0, 12.0], dtype=torch.float64) / (13.0 + 1e-12)
        self.assertTrue(torch.equal(torch.cat([p.grad, q.grad]), expected))

    def test_zero_probability_marginal_and_shared_initialization(self):
        reference = fixture('V0')[0].state_dict()
        for variant in VARIANTS:
            model, cfg, x, y, mask = fixture(variant)
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, reference[key]), (variant, key))
            out = model(*x)
            self.assertLess(float((out['joint'].exp().sum((1, 2)) - 1).abs().max()), 1e-5)
            self.assertLess(float((out['joint'].exp().sum(-1) - out['log_o'].exp()).abs().max()), 1e-5)
            self.assertTrue(torch.equal(out['base'], out['cond']))

    def test_explicit_operator_and_primitive(self):
        model, cfg, x, y, mask = fixture()
        model = model.double()
        g, fa, fo = [v.double() for v in x]
        with torch.no_grad():
            model.Wc.normal_(0, 0.1)
            model.Wa.normal_(0, 0.1)
        out = model(g, fa, fo)
        u, v = model.ea @ model.Au.T, model.eo @ model.Av.T
        h = u[None, None] * v[None, :, None] * out['gamma'][:, :, None]
        delta = h @ model.Wc.T
        direct = cfg['rho'] * cfg['kappa'] * (fo[:, :, None] * delta).sum(-1)
        self.assertTrue(torch.allclose(out['cond'] - out['base'], direct, atol=1e-12))
        operator = model.Wc @ torch.diag(v[1] * out['gamma'][0, 1]) @ model.Au
        self.assertTrue(torch.allclose(operator @ model.ea[2], delta[0, 1, 2], atol=1e-12))
        self.assertLessEqual(int(torch.linalg.matrix_rank(operator)), cfg['rank'])
        pi = (torch.einsum('bid,jd->bij', fa, model.eo) / cfg['tau_pi']).softmax(-1)
        primitive_delta = (u[None] * (pi @ v)) @ model.Wa.T
        primitive_ref = cfg['kappa'] * (g @ model.ea.T) + cfg['rho_a'] * cfg['kappa'] * (fa * primitive_delta).sum(-1)
        self.assertTrue(torch.allclose(out['prim'], primitive_ref, atol=1e-12))
        self.assertTrue(torch.allclose(pi.sum(-1), torch.ones_like(pi.sum(-1))))

    def test_static_uniform_mismatch_and_parameter_counts(self):
        d, r, q = 12, 3, 2
        counts = {'V0': 4*d*q, 'V1': 4*d*q+3*d*r, 'V1L': 4*d*q+3*d*r,
                  'V2': 4*d*q+4*d*r, 'V3': 4*d*q+4*d*r, 'V4': 4*d*q+5*d*r,
                  'V5': 4*d*q+5*d*r, 'V6': 4*d*q+4*d*r, 'V7': 4*d*q+5*d*r}
        for variant, count in counts.items():
            model, cfg, x, y, mask = fixture(variant)
            self.assertEqual(model.trainable_count(), count)
            if variant == 'V4':
                self.assertTrue(torch.all(model(*x)['pi'] == 0.5))
            if variant == 'V6':
                g, fa, fo = x
                ref = 2 * torch.sigmoid((fo.roll(1, 0) @ model.Bvis.T) / cfg['tau_gamma'])
                self.assertTrue(torch.equal(model(*x)['gamma'], ref))
        model, cfg, x, y, mask = fixture('V2')
        with torch.no_grad():
            model.Bvis.zero_()
            model.Wc.normal_()
        out = model(*x)
        self.assertTrue(torch.all(out['gamma'] == 1))
        model.variant, model.dynamic = 'V1L', False
        self.assertTrue(torch.equal(out['cond'], model(*x)['cond']))

    def test_full_loss_finite_difference_and_first_two_gradients(self):
        model, cfg, x, y, mask = fixture()
        model = model.double()
        x = tuple(v.double() for v in x)
        with torch.no_grad():
            model.Wc.normal_(0, 0.1)
            model.Wa.normal_(0, 0.1)
            model.Co.normal_(0, 0.03)
            model.Cc.normal_(0, 0.03)
        loss = model.objective(model(*x), *y, mask)['total']
        loss.backward()
        max_relative = 0
        for name in ('Bo', 'Co', 'Bc', 'Cc', 'Au', 'Av', 'Bvis', 'Wc', 'Wa'):
            parameter = getattr(model, name)
            # Check largest component to avoid ratios at exact/near-zero gradients.
            flat_index = int(parameter.grad.abs().argmax())
            analytic = float(parameter.grad.flatten()[flat_index])
            self.assertGreater(abs(analytic), 1e-12, name)
            with torch.no_grad():
                original = float(parameter.flatten()[flat_index])
                parameter.flatten()[flat_index] = original + 1e-5
                plus = float(model.objective(model(*x), *y, mask)['total'])
                parameter.flatten()[flat_index] = original - 1e-5
                minus = float(model.objective(model(*x), *y, mask)['total'])
                parameter.flatten()[flat_index] = original
            numeric = (plus - minus) / 2e-5
            relative = abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-8)
            max_relative = max(max_relative, relative)
            self.assertLess(relative, 1e-4, name)
        model, cfg, x, y, mask = fixture()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.objective(model(*x), *y, mask)['total'].backward()
        self.assertGreater(float(model.Wc.grad.norm()), 0)
        self.assertEqual(float(model.Au.grad.norm()), 0)
        optimizer.step()
        optimizer.zero_grad()
        out = model(*x)
        # Task loss alone excludes any orthogonality/primitive gradient shortcuts.
        terms = model.objective(out, *y, mask)
        (terms['pair'] + 0.5 * terms['cond'] + 0.5 * terms['object']).backward()
        for name in ('Au', 'Av', 'Bvis'):
            self.assertGreater(float(getattr(model, name).grad.norm()), 0)
        print('Full-loss finite-difference max relative error:', max_relative)

    def test_js_and_patch_query(self):
        p = torch.rand(8, 3).softmax(-1)
        q = torch.rand(8, 3).softmax(-1)
        self.assertEqual(float(js_divergence(p, p)), 0)
        self.assertGreaterEqual(float(js_divergence(p, q)), 0)
        self.assertLessEqual(float(js_divergence(p, q)), np.log(2))
        patches, queries = unit(torch.randn(2, 7, 12)), unit(torch.randn(5, 12))
        result = query_patches(patches, queries, 0.07)
        for b in range(2):
            for j in range(5):
                alpha = (patches[b] @ queries[j] / 0.07).softmax(0)
                ref = unit((alpha[:, None] * patches[b]).sum(0))
                self.assertTrue(torch.allclose(result[b, j], ref, atol=1e-6))

    def test_official_clip_visual_cls(self):
        path = ROOT / 'vendor' / 'clip' / 'model.py'
        spec = importlib.util.spec_from_file_location('official_clip_model_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        visual = module.VisionTransformer(28, 7, 64, 1, 1, 12).eval()
        images = torch.randn(2, 3, 28, 28)
        with torch.no_grad():
            g, patches = visual_tokens(visual, images)
            ref = unit(visual(images))
        self.assertTrue(torch.allclose(g, ref, atol=1e-6))
        self.assertEqual(tuple(patches.shape), (2, 16, 12))


class MetricTests(unittest.TestCase):
    def test_native_scan_equals_archived_evaluator(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('archived_native', ROOT / 'vendor/native_evaluator.py')
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        labels = {'attrs': ['a', 'b', 'c'], 'objs': ['x', 'y']}
        pairs = [(a, o) for a in labels['attrs'] for o in labels['objs']]
        dataset = SimpleNamespace(attrs=labels['attrs'], objs=labels['objs'], pairs=pairs,
                   train_pairs=pairs[:3], val_pairs=pairs[3:], test_pairs=pairs[3:],
                   attr2idx={a: i for i, a in enumerate(labels['attrs'])},
                   obj2idx={o: i for i, o in enumerate(labels['objs'])},
                   pair2idx={p: i for i, p in enumerate(pairs)}, phase='val', open_world=True)
        evaluator = native.Evaluator(dataset, None)
        torch.manual_seed(991)
        scores = torch.randn(80, 6)
        truth = torch.arange(80) % 6
        attrs, objs = evaluator.pairs[truth].T
        allpred = {p: scores[:, k] for k, p in enumerate(pairs)}
        preds = evaluator.score_manifold_model(allpred, objs, bias=1000)
        legacy = evaluator.evaluate_predictions(preds, attrs, objs, truth, allpred)
        ours = native_curve(scores.numpy(), truth.numpy(), np.arange(6) < 3)
        self.assertAlmostEqual(ours['AUC'], legacy['AUC'] * 100, places=10)
        self.assertAlmostEqual(ours['best_hm'], legacy['best_hm'] * 100, places=6)
        self.assertAlmostEqual(ours['best_bias'], legacy['biasterm'], places=7)

    def test_exact_curve_against_enumeration(self):
        rng = np.random.RandomState(5)
        scores = rng.randn(30, 6)
        truth = np.arange(30) % 6
        seen = np.array([True, True, True, False, False, False])
        result = exact_curve(scores, truth, seen)
        s, u = [], []
        for bias in result['biases']:
            pred = (scores + bias * (~seen)[None]).argmax(1)
            s.append((pred[seen[truth]] == truth[seen[truth]]).mean() * 100)
            u.append((pred[~seen[truth]] == truth[~seen[truth]]).mean() * 100)
        np.testing.assert_allclose(result['seen'], s)
        np.testing.assert_allclose(result['unseen'], u)
        area = np.trapz(np.array(s) / 100, np.array(u) / 100) * 100
        self.assertAlmostEqual(result['AUC'], area)


def synthetic_cache(folder):
    model, cfg, x, y, mask = fixture()
    labels = {'attrs': ['a', 'b', 'c'], 'objs': ['x', 'y']}
    pairs = {'train': [['a', 'x'], ['b', 'x'], ['b', 'y'], ['c', 'y']],
             'val': [['a', 'x'], ['b', 'x'], ['a', 'y'], ['c', 'x']],
             'test': [['b', 'y'], ['c', 'y'], ['a', 'y'], ['c', 'x']]}
    folder.mkdir()
    np.savez(folder / 'text_features.npz', ea=model.ea.numpy(), eo=model.eo.numpy(), ec=model.ec.numpy())
    counts = {}
    for phase in ('train', 'val', 'test'):
        rows = []
        for k in range(16):
            a, o = pairs[phase][k % 4]
            rows.append({'image': phase + str(k), 'attr': labels['attrs'].index(a), 'object': labels['objs'].index(o)})
        write_json(folder / (phase + '_records.json'), rows)
        sub = folder / phase
        sub.mkdir()
        for key, shape in [('g', (16, 12)), ('fa', (16, 3, 12)), ('fo', (16, 2, 12))]:
            np.save(sub / (key + '.npy'), unit(torch.randn(shape)).numpy().astype(np.float16))
        counts[phase] = len(rows)
    files = list(folder.glob('*records.json')) + list(folder.glob('*/*.npy')) + [folder / 'text_features.npz']
    write_json(folder / 'features_manifest.json', {'complete': True, 'tau_v': 0.07, 'labels': labels,
               'pairs': pairs, 'counts': counts,
               'files_sha256': {p.relative_to(folder).as_posix(): sha(p) for p in files}})
    return cfg


class IntegrationTests(unittest.TestCase):
    def test_every_ablation_trains_and_exports(self):
        from vgcpod import engine
        with tempfile.TemporaryDirectory(prefix='vgcpod_matrix_test_') as temp:
            root = Path(temp)
            cfg = synthetic_cache(root / 'cache')
            for variant in VARIANTS:
                current = dict(cfg, variant=variant, epochs=1)
                args = argparse.Namespace(cache=str(root / 'cache'), out=str(root / 'runs'), seed=3351,
                                          device='cpu', resume=False)
                engine.train(args, current)
                run = root / 'runs/ut-zap50k' / variant / 'seed3351'
                result = read_json(run / 'metrics.json')
                self.assertEqual(result['selected_epoch'], 1)
                self.assertEqual(set(result['metrics']), {'val_cw', 'val_ow', 'test_cw', 'test_ow'})

    def test_prepare_and_resume_with_synthetic_images(self):
        # Real official CLIP tokenizer/preprocess/visual code; tiny frozen visual weights.
        # The dataset index and checkpoint identity are mocked only inside this fixture.
        import sys
        sys.path.insert(0, str(ROOT / 'vendor'))
        import clip
        from PIL import Image
        from vgcpod import features
        class TinyCLIP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = clip.model.VisionTransformer(224, 14, 1024, 0, 16, 768)
            def encode_image(self, x):
                return self.visual(x)
            def encode_text(self, tokens):
                return torch.sin(tokens.float().sum(1, keepdim=True) + torch.arange(768)[None].float())
        with tempfile.TemporaryDirectory(prefix='vgcpod_encode_test_') as temp:
            root = Path(temp)
            data = root / 'data'
            (data / 'images').mkdir(parents=True)
            labels = read_json(ROOT / 'assets/labels.json')
            records, pairs = {}, {}
            for phase in ('train', 'val', 'test'):
                rows = []
                for i in range(2):
                    name = '%s_%d.png' % (phase, i)
                    im = np.random.RandomState(10 + i).randint(0, 255, (240, 250, 3), dtype=np.uint8)
                    Image.fromarray(im).save(data / 'images' / name)
                    rows.append({'image': name, 'raw_image': name, 'attr': i, 'object': i})
                records[phase] = rows
                pairs[phase] = [[labels['attrs'][i], labels['objs'][i]] for i in range(2)]
            checkpoint = root / 'fake_checkpoint_for_fixture.pt'
            checkpoint.write_bytes(b'test fixture, no pretrained weights')
            args = argparse.Namespace(data_root=str(data), cache=str(root / 'cache'), checkpoint=str(checkpoint),
                                      batch_size=2, shard_size=2, workers=0, tau_v=0.07, device='cpu')
            model = TinyCLIP().eval()
            with mock.patch.object(features, 'CLIP_SHA', sha(checkpoint)), \
                 mock.patch.object(features, 'inspect_dataset', return_value=(labels, pairs, records, {})), \
                 mock.patch.object(clip, 'load', return_value=(model, clip.clip._transform(224))):
                features.prepare(args)
                manifest, text = load_cache(args.cache, verify=True)
                before = dict(manifest['files_sha256'])
                features.prepare(args)
                manifest, text = load_cache(args.cache, verify=True)
                self.assertEqual(before, manifest['files_sha256'])
            self.assertEqual(tuple(text['ec'].shape), (12, 16, 768))
            for phase in ('train', 'val', 'test'):
                with open(root / 'cache' / phase / 'fa.npy', 'rb') as f:
                    array = np.load(f)
                self.assertEqual(array.shape, (2, 16, 768))
                np.testing.assert_allclose(np.linalg.norm(array.astype(np.float32), axis=-1), 1, atol=1e-3)
            target = root / 'cache/train/g.npy'
            data_bytes = bytearray(target.read_bytes())
            data_bytes[-1] ^= 1
            target.write_bytes(data_bytes)
            with self.assertRaises(ValueError):
                load_cache(args.cache, verify=True)

    def test_train_resume_prediction_smoke_and_summary(self):
        from vgcpod import engine
        with tempfile.TemporaryDirectory(prefix='vgcpod_test_') as temp:
            root = Path(temp)
            cfg = synthetic_cache(root / 'cache')
            def args(name, resume=False):
                return argparse.Namespace(cache=str(root / 'cache'), out=str(root / name), seed=3351,
                                          device='cpu', resume=resume)
            full = args('full')
            engine.train(full, cfg)
            real_predict = engine.predict
            calls = [0]
            def interrupted(*a, **kw):
                calls[0] += 1
                if calls[0] == 2:
                    raise InterruptedError('simulated interruption after first committed epoch')
                return real_predict(*a, **kw)
            resumed = args('resumed')
            with mock.patch.object(engine, 'predict', side_effect=interrupted):
                with self.assertRaises(InterruptedError):
                    engine.train(resumed, cfg)
            resumed.resume = True
            engine.train(resumed, cfg)
            suffix = Path('ut-zap50k/V5/seed3351')
            a = load_torch(root / 'full' / suffix / 'checkpoints/last.pt')
            b = load_torch(root / 'resumed' / suffix / 'checkpoints/last.pt')
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
            metrics = read_json(root / 'full' / suffix / 'metrics.json')
            self.assertEqual(set(metrics['metrics']), {'val_cw', 'val_ow', 'test_cw', 'test_ow'})
            for world in ('cw', 'ow'):
                self.assertEqual(metrics['metrics']['val_' + world]['bias_native'],
                                 metrics['metrics']['test_' + world]['fixed_val_bias'])
            with np.load(root / 'full' / suffix / 'predictions/test_ow.npz', allow_pickle=False) as z:
                np.testing.assert_allclose(z['probabilities'].sum(1), 1, atol=1e-5)
            engine.smoke(full, cfg)
            engine.baselines(full, cfg)
            engine.summarize(full)
            self.assertTrue((root / 'full' / 'summary.json').exists())
            from vgcpod import diagnostics
            diag_args = argparse.Namespace(cache=full.cache, run=str(root / 'full' / suffix), device='cpu')
            diagnostics.diagnose(diag_args)
            self.assertTrue((root / 'full' / suffix / 'diagnostics/gamma_mean.png').exists())
            # Identical prediction fixtures must produce exactly zero per-pair gains.
            shutil.copytree(root / 'full' / suffix / 'predictions',
                            root / 'full/ut-zap50k/V4/seed3351/predictions')
            full.seeds = [3351]
            diagnostics.compare(full)
            self.assertTrue((root / 'full/per_pair_gains.csv').exists())
            with open(root / 'full/per_pair_gains.csv', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    self.assertEqual(float(row['pair_gain_unbiased_pct']), 0)
                    self.assertEqual(float(row['gt_object_attr_gain_pct']), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
