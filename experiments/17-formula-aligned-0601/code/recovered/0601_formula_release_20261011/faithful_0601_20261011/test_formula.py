"""Synthetic CPU tests are engineering evidence, never benchmark results."""
import ast
import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import read, sha, write

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


pkg = types.ModuleType('formula_fixture')
pkg.__path__ = [str(HERE / 'source/models')]
sys.modules['formula_fixture'] = pkg
cache_module = load('formula_fixture.kcc_cache', HERE / 'source/models/kcc_cache.py')
full = load('formula_fixture.full_feature', HERE / 'source/models/full_feature.py')
load('formula_fixture.bf_logic_loss', HERE / 'source/models/bf_logic_loss.py')
load('formula_fixture.bf_logic', HERE / 'source/models/bf_logic.py')
utils = load('formula_fixture.ddm_utils', HERE / 'source/models/ddm_utils.py')
formula = load('formula_fixture.formula_model', HERE / 'source/models/formula_model.py')


class Classifier(nn.Module):
    temp = 0.05

    def forward(self, x, w, scale=True):
        z = F.normalize(x, dim=-1) @ F.normalize(w, dim=-1).t()
        return z / self.temp if scale else z


class FakeFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 768)

    def forward(self, x):
        c = self.layer(x)
        p = c[:, None].expand(-1, 4, -1)
        return p * .7, p * .8, p * .9, p, c


class FakeBaseline(nn.Module):
    def __init__(self, ds, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_objs, self.num_attrs, self.word_dim = 12, 16, 300
        self.obj_embedder, self.attr_embedder = nn.Embedding(12, 300), nn.Embedding(16, 300)
        self.tot_objs, self.tot_attrs = torch.arange(12), torch.arange(16)
        self.train_objs = torch.tensor([i % 12 for i in range(32)])
        self.train_attrs = torch.tensor([i % 16 for i in range(32)])
        self.val_objs, self.val_attrs = self.train_objs, self.train_attrs
        self.val_pairs = [(ds.attrs[a], ds.objs[o]) for a, o in zip(self.val_attrs, self.val_objs)]
        self.feat_extractor = FakeFeatures()
        self.objc_embedder, self.attrc_embedder = nn.Linear(768, 300), nn.Linear(900, 300)
        self.obj_guide_embedder, self.objc_embedder_2 = nn.Linear(300, 768), nn.Linear(768, 300)
        self.compose, self.projection_1 = nn.Linear(600, 300), nn.Linear(600, 300)
        self.classifier = Classifier()

    def compose_visual(self, obj, attr):
        return self.projection_1(torch.cat((obj, attr), dim=1))


class FakeCache:
    def _setup_kcc(self, ds, cfg):
        self.kcc_text, self.kcc_calib, self.kcc_lambda = cfg.MODEL.kcc_text, cfg.MODEL.kcc_calib, cfg.MODEL.kcc_lambda
        self.register_buffer('W_o_K', F.normalize(self.obj_embedder.weight.detach().clone(), dim=-1), persistent=False)
        self.register_buffer('W_a_K', F.normalize(self.attr_embedder.weight.detach().clone(), dim=-1), persistent=False)
        self.register_buffer('K_oa_prior', formula.centered_prior(torch.arange(192).reshape(12, 16).float()), persistent=False)

    def _kcc_obj_embeddings(self, ids):
        return self.W_o_K[ids] if self.kcc_text else self.obj_embedder(ids)

    def _kcc_attr_embeddings(self, ids):
        return self.W_a_K[ids] if self.kcc_text else self.attr_embedder(ids)

    def _kcc_calibrate(self, score, objs, attrs):
        return score + self.kcc_lambda * self.K_oa_prior[objs, attrs][None] if self.kcc_calib else score


cache_module.KCCCacheMixin = FakeCache
ddm = types.ModuleType('formula_fixture.ddm')
for key, value in dict(torch=torch, nn=nn, F=F, np=np, DDM=FakeBaseline,
                       choose_texts=utils.choose_texts, choose_visual_tokens=utils.choose_visual_tokens,
                       image_soft_weights=utils.image_soft_weights).items():
    setattr(ddm, key, value)
sys.modules['formula_fixture.ddm'] = ddm
legacy = load('formula_fixture.reference', HERE / 'reference_models/kcc_cpf.py')
actual = load('formula_fixture.kcc_cpf', HERE / 'source/models/kcc_cpf.py')


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        bank = HERE / 'assets/logic_bank'
        self.ds = types.SimpleNamespace(**read(bank / 'manifest.json')['labels'])
        self.cache = self.path / 'cache'
        (self.cache / 'W_ao_K').mkdir(parents=True)
        hashes = {}
        for i in range(12):
            p = self.cache / ('W_ao_K/obj_{:05d}.pt'.format(i))
            torch.save(F.normalize(torch.randn(16, 300), dim=-1), p)
            hashes[p.relative_to(self.cache).as_posix()] = sha(p)
        write(self.cache / 'manifest.json', {'files_sha256': hashes})
        self.cfg = types.SimpleNamespace(MODEL=types.SimpleNamespace(
            kcc_feature=True, kcc_text=True, kcc_calib=True, kcc_lambda=.1,
            kcc_logic=True, kcc_beta=.01, bf_logic_bank=str(bank), bf_logic_q=3.,
            bf_allow_reconstructed=True, full_gain=1., full_use_pair=True,
            kcc_cache=str(self.cache), formula_joint=True),
            TRAIN=types.SimpleNamespace(max_epoch=30, aug_epoch=37, seed=3351))
        self.batch = {'img': torch.randn(4, 8), 'attr': torch.tensor([0, 1, 2, 3]),
                      'obj': torch.tensor([0, 1, 2, 3]), 'pair': torch.tensor([0, 1, 2, 3])}

    def tearDown(self):
        self.temp.cleanup()

    def model(self, cfg=None, old=False):
        c = cfg or self.cfg
        torch.manual_seed(c.TRAIN.seed)
        return (legacy.DDM if old else actual.DDM)(self.ds, c).eval()

    def off(self, text=False):
        c = copy.deepcopy(self.cfg)
        c.MODEL.kcc_feature, c.MODEL.kcc_logic, c.MODEL.kcc_beta = False, False, 0.
        c.MODEL.kcc_calib, c.MODEL.kcc_lambda, c.MODEL.formula_joint = False, 0., False
        c.MODEL.kcc_text = text
        return c

    def test_A_C_exact_legacy_regression(self):
        for text in (False, True):
            c = self.off(text)
            old, new = self.model(c, old=True), self.model(c)
            self.assertEqual(list(old.state_dict()), list(new.state_dict()))
            for key, value in old.state_dict().items():
                self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
            self.assertTrue(torch.equal(old.train_forward(self.batch)['loss_total'], new.train_forward(self.batch)['loss_total']))
            self.assertTrue(torch.equal(old.val_forward(self.batch)['pred'], new.val_forward(self.batch)['pred']))

    def test_head_native_init_and_rng_preserved(self):
        c = copy.deepcopy(self.cfg)
        c.MODEL.kcc_feature = False
        old = self.model(c)
        rng = torch.get_rng_state()
        new = self.model()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in old.state_dict().items():
            self.assertTrue(torch.equal(v, new.state_dict()[k]), k)

    def test_seed_changes_head_without_fixed_global_seed(self):
        a = self.model()
        c = copy.deepcopy(self.cfg)
        c.TRAIN.seed = 3352
        b = self.model(c)
        self.assertFalse(torch.equal(a.full_head.attr_vis_proj.weight, b.full_head.attr_vis_proj.weight))

    def test_GH_exact_head_and_joint_formula(self):
        m = self.model()
        out, z, _ = m.formula_train_terms(self.batch, True)
        vo, va, _, _, _ = m.formula_features(self.batch)
        direct = m.full_head(va, vo, m.W_a_K, m.W_o_K, m.full_pair_embeddings)
        self.assertTrue(torch.equal(z['conditional'], direct))
        expected = z['obj'][:, :, None] + direct + .1 * m.K_oa_prior[None]
        self.assertTrue(torch.equal(z['score'], expected))
        self.assertTrue(torch.equal(out['loss_composition'], F.cross_entropy(expected[:, m.train_objs, m.train_attrs], self.batch['pair'])))

    def test_GH_gt_object_CE_not_native_attr_residual(self):
        m = self.model()
        out, z, _ = m.formula_train_terms(self.batch, True)
        gt = z['conditional'][torch.arange(4), self.batch['obj']]
        expected = F.cross_entropy(gt, self.batch['attr'])
        self.assertTrue(torch.equal(out['loss_conditional_attr'], expected))
        self.assertFalse(torch.allclose(expected, F.cross_entropy(gt + z['attr'], self.batch['attr'])))

    def test_train_infer_formula_identity_GH_and_MVP(self):
        for feature in (True, False):
            c = copy.deepcopy(self.cfg)
            c.MODEL.kcc_feature = feature
            m = self.model(c)
            _, z, _ = m.formula_train_terms(self.batch, True)
            self.assertTrue(torch.allclose(m.val_forward(self.batch)['pred'], z['score'][:, m.train_objs, m.train_attrs], atol=2e-6, rtol=1e-5))

    def test_MVP_conditional_all_cartesian_and_GT_slice(self):
        c = copy.deepcopy(self.cfg)
        c.MODEL.kcc_feature = False
        m = self.model(c)
        out, z, _ = m.formula_train_terms(self.batch, True)
        self.assertEqual(z['comp'].shape, (4, 192))
        self.assertTrue(torch.equal(z['conditional'], z['attr'][:, None] + z['comp'].reshape(4, 12, 16)))
        self.assertTrue(torch.equal(out['loss_conditional_attr'], F.cross_entropy(z['conditional'][torch.arange(4), self.batch['obj']], self.batch['attr'])))

    def test_no_native_composition_gradient_GH(self):
        m = self.model()
        m.train_forward(self.batch)['loss_total'].backward()
        self.assertIsNone(m.projection_1.weight.grad)
        self.assertIsNone(m.compose.weight.grad)
        self.assertGreater(float(m.feat_extractor.layer.weight.grad.abs().sum()), 0)

    def test_head_all_active_finite_gradients(self):
        m = self.model()
        out = m.train_forward(self.batch)
        grads = torch.autograd.grad(out['loss_total'], list(m.full_head.parameters()))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))

    def test_H_AE_reaches_head(self):
        m = self.model()
        out = m.train_forward(self.batch)
        grads = torch.autograd.grad(out['loss_ae'], list(m.full_head.parameters()), allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads if g is not None), 0)

    def test_no_pair_consumption(self):
        c = copy.deepcopy(self.cfg)
        c.MODEL.full_use_pair = False
        m = self.model(c)
        m.train_forward(self.batch)['loss_total'].backward()
        self.assertIsNone(m.full_head.pair_bias[0].weight.grad)
        self.assertGreater(float(m.full_head.attr_vis_proj.weight.grad.abs().sum()), 0)

    def test_G_no_logic(self):
        c = copy.deepcopy(self.cfg)
        c.MODEL.kcc_logic, c.MODEL.kcc_beta = False, 0.
        out = self.model(c).train_forward(self.batch)
        self.assertTrue(torch.equal(out['loss_total'], out['loss_cpf']))

    def test_beta_applied_once(self):
        out = self.model().train_forward(self.batch)
        self.assertTrue(torch.equal(out['loss_logic_weighted'], .01 * out['loss_logic_raw']))
        self.assertTrue(torch.equal(out['loss_total'], out['loss_cpf'] + out['loss_logic_weighted']))

    def test_adapter_literal_probabilities(self):
        m = self.model()
        _, z, nodes = m.formula_train_terms(self.batch, True)
        masked = torch.full_like(z['score'], -torch.inf)
        masked[:, m.train_objs, m.train_attrs] = z['score'][:, m.train_objs, m.train_attrs]
        self.assertTrue(torch.equal(nodes['object'], F.softmax(z['obj'], dim=-1)))
        self.assertTrue(torch.equal(nodes['attribute'], F.softmax(torch.logsumexp(masked, dim=1), dim=-1)))
        self.assertTrue(torch.equal(nodes['pair'], torch.sigmoid(masked)))
        oe = z['image_feat'] @ m.bf_logic.oe.reshape(-1, 300).t()
        self.assertTrue(torch.equal(nodes['oe'], torch.sigmoid(oe).reshape(4, 12, 3)))
        self.assertFalse(torch.allclose(nodes['oe'].sum(1), torch.ones(4, 3)))

    def test_B_external_task_losses_and_adapter(self):
        c = self.off()
        c.MODEL.kcc_logic, c.MODEL.kcc_beta = True, .01
        m = self.model(c)
        out, z, nodes = m.formula_train_terms(self.batch, True)
        task = F.cross_entropy(z['comp'], self.batch['pair']) + .4 * F.cross_entropy(z['obj'], self.batch['obj']) + .6 * F.cross_entropy(z['attr'], self.batch['attr'])
        self.assertTrue(torch.allclose(task, out['loss_cpf']))
        self.assertIsNotNone(nodes)

    def test_centered_literal_sample_std_eps(self):
        k = torch.rand(12, 16)
        self.assertTrue(torch.equal(formula.centered_prior(k), (k - k.mean()) / (k.std() + 1e-8)))
        self.assertTrue(torch.equal(formula.centered_prior(torch.ones_like(k)), torch.zeros_like(k)))

    def test_compatibility_literal_minmax_eps(self):
        o, a, pair = F.normalize(torch.randn(12, 300), dim=-1), F.normalize(torch.randn(16, 300), dim=-1), F.normalize(torch.randn(12, 16, 300), dim=-1)
        raw = F.cosine_similarity(pair, F.normalize((o[:, None] + a[None]) / 2, dim=-1), dim=-1)
        expected = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        self.assertTrue(torch.equal(formula.compatibility(o, a, pair), expected))

    def test_actual_cache_loader_accepts_sample_prior_rejects_population_prior(self):
        cm = load('real_formula_cache_loader', HERE / 'source/models/kcc_cache.py')
        class Probe(cm.KCCCacheMixin, nn.Module):
            def __init__(inner, cfg):
                super().__init__()
                inner.word_dim, inner.num_objs, inner.num_attrs = 300, 12, 16
                inner.obj_embedder, inner.attr_embedder = nn.Embedding(12, 300), nn.Embedding(16, 300)
                inner.obj_embedder.weight.data.copy_(raw_o)
                inner.attr_embedder.weight.data.copy_(raw_a)
                inner._setup_kcc(self.ds, cfg)
        raw_o, raw_a = torch.randn(12, 300), torch.randn(16, 300)
        k = torch.rand(12, 16)
        values = {'W_o_name_raw.pt': raw_o, 'W_a_name_raw.pt': raw_a,
                  'W_o_K.pt': F.normalize(raw_o, dim=-1), 'W_a_K.pt': F.normalize(raw_a, dim=-1),
                  'K_oa_text.pt': k, 'K_oa_prior_centered.pt': formula.centered_prior(k)}
        for name, value in values.items():
            torch.save(value, self.cache / name)
        write(self.cache / 'labels.json', {'attrs': self.ds.attrs, 'objs': self.ds.objs})
        m = {'attrs': self.ds.attrs, 'objs': self.ds.objs, 'pair_axis_order': ['object', 'attribute'],
             'dimension': 300, 'dtype': 'float32',
             'files_sha256': {name: sha(self.cache / name) for name in list(values) + ['labels.json']}}
        write(self.cache / 'manifest.json', m)
        Probe(self.cfg)
        torch.save((k - k.mean()) / k.std(unbiased=False), self.cache / 'K_oa_prior_centered.pt')
        m['files_sha256']['K_oa_prior_centered.pt'] = sha(self.cache / 'K_oa_prior_centered.pt')
        write(self.cache / 'manifest.json', m)
        with self.assertRaises(ValueError):
            Probe(self.cfg)

    def test_optimizer_and_durable_state_roundtrip(self):
        m = self.model()
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
        before = m.full_head.pair_bias[0].weight.detach().clone()
        optimizer.zero_grad()
        m.train_forward(self.batch)['loss_total'].backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, m.full_head.pair_bias[0].weight))
        target = self.path / 'resume.pth'
        torch.save({'model': m.state_dict(), 'optimizer': optimizer.state_dict()}, target)
        restored = self.model()
        state = torch.load(target, weights_only=True)
        restored.load_state_dict(state['model'], strict=True)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
        restored_optimizer.load_state_dict(state['optimizer'])
        self.assertTrue(torch.equal(m.val_forward(self.batch)['pred'], restored.val_forward(self.batch)['pred']))

    def test_no_early_object_argmax(self):
        obj = torch.tensor([[3., 2.]])
        cond = torch.tensor([[[0., 0.], [0., 10.]]])
        score = formula.joint_score(obj, cond)
        self.assertEqual(score.flatten(1).argmax(1).item(), 3)

    def test_aug_bypass_rejected_for_every_group(self):
        for c in (self.cfg, self.off()):
            c = copy.deepcopy(c)
            c.TRAIN.max_epoch = 37
            with self.assertRaises(ValueError):
                self.model(c)

    def test_pair_block_corruption_rejected(self):
        with (self.cache / 'W_ao_K/obj_00000.pt').open('ab') as f:
            f.write(b'bad')
        with self.assertRaises(ValueError):
            self.model()

    def test_all_python38_parse(self):
        for p in HERE.rglob('*.py'):
            if '__pycache__' not in p.parts:
                ast.parse(p.read_text(encoding='utf-8'), feature_version=8)

    def test_engine_generalized_epochs_and_no_deadline_queue(self):
        import engine
        output = '0, 2000, 22500, 0\n1, 13, 24500, 0\n2, 10, 24500, 90\n3, 13, 24500, 0\n'
        with mock.patch('subprocess.check_output', return_value=output):
            self.assertEqual(engine.available_gpus({1, 2, 3}, set()), [1, 3])
        text = (HERE / 'engine.py').read_text(encoding='utf-8')
        self.assertNotIn('epoch=10', text)
        self.assertNotIn('range(1, 11)', text)
        queue = (HERE / 'run_suite.py').read_text(encoding='utf-8')
        self.assertNotIn('reserve_job_minutes', queue)
        self.assertNotIn('LAUNCH_CUTOFF_REACHED', queue)

    def test_report_incomplete_and_paired_seeds(self):
        from report import summarize
        work = self.path / 'report'
        jobs = [{'variant': v + '_s' + str(s), 'base_variant': v, 'seed': s, 'tier': 0,
                 'run': str(work / 'runs' / (v + '_s' + str(s)))} for s in (3351, 3352) for v in ('A', 'H')]
        p = {'jobs': jobs, 'epochs': 30, 'seeds': [3351, 3352], 'not_covered': []}
        for k in ('formula', 'temperature', 'mvp_interface', 'native_baseline', 'logic_adapter', 'description_interface', 'logic_operators',
                  'head', 'knowledge', 'controls', 'training_budget', 'augmentation', 'test_policy'):
            p[k] = 'fixture'
        write(work / 'protocol.json', p)
        self.assertEqual(summarize(work), (False, False))
        self.assertEqual(read(work / 'summary/summary.json')['rows'], [])
        for j in jobs:
            run = Path(j['run'])
            write(run / 'status.json', {'state': 'complete', 'epochs': 30})
            (run / 'train.log').write_text('Val/Test Epoch 1: AUC=0.4 | best_hm=0.5\n', encoding='utf-8')
            for phase in ('val', 'test'):
                for setting in ('cw', 'ow'):
                    auc = .5 if j['base_variant'] == 'H' else .4
                    write(run / ('eval_' + phase + '_' + setting) / 'metrics.json', {
                        'best_epoch': 1, 'checkpoint_sha256': 'fixture',
                        'metrics_raw': {'AUC': auc, 'best_hm': .6, 'best_seen': .7, 'best_unseen': .8}})
        self.assertEqual(summarize(work), (True, True))
        result = read(work / 'summary/summary.json')
        self.assertEqual(len(result['rows']), 16)
        delta = next(d for d in result['paired_differences'] if d['comparison'] == 'H-A')
        self.assertAlmostEqual(delta['AUC_pp']['mean'], 10.)
        self.assertEqual(delta['AUC_pp']['n'], 2)

    def test_preparation_42_jobs_and_same_budget_all_controls(self):
        import prepare
        root, work = self.path / 'root', self.path / 'prepared'
        origin = root / 'knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k'
        origin.mkdir(parents=True)
        for name, value in [('W_o_K.pt', F.normalize(torch.randn(12, 300), dim=-1)),
                            ('W_a_K.pt', F.normalize(torch.randn(16, 300), dim=-1)),
                            ('K_oa_text.pt', torch.rand(12, 16)), ('K_oa_prior_centered.pt', torch.randn(12, 16))]:
            torch.save(value, origin / name)
        import shutil
        shutil.copytree(self.cache / 'W_ao_K', origin / 'W_ao_K')
        write(origin / 'manifest.json', {'files_sha256': {p.relative_to(origin).as_posix(): sha(p) for p in origin.rglob('*.pt')}})
        def legacy_prepare_fixture(root_arg, work_arg, tier):
            work_arg.mkdir()
            (work_arg / 'source').mkdir()
            (work_arg / 'configs').mkdir()
            for kind in ('shuffle', 'random', 'uniform'):
                shutil.copytree(origin, work_arg / ('cache_' + kind))
            result = []
            for name in prepare.CORE + prepare.CONTROLS:
                cache = work_arg / ('cache_' + ('shuffle' if name == 'H_shuffle' else 'random' if name == 'H_random_prior' else 'uniform')) if name in ('H_shuffle', 'H_random_prior', 'H_uniform_prior') else origin
                c = {'MODEL': {'kcc_cache': str(cache), 'kcc_text': True}, 'TRAIN': {}}
                path = work_arg / 'configs' / (name + '.yml')
                import yaml
                path.write_text(yaml.safe_dump(c), encoding='utf-8')
                result.append({'variant': name, 'cfg': str(path)})
            return {'jobs': result, 'frozen_inputs': {}}
        with mock.patch.object(prepare, 'verify_release'), mock.patch.object(prepare.legacy_prepare, 'prepare', side_effect=legacy_prepare_fixture):
            p = prepare.prepare(root, work, 30, [3351, 3352, 3353])
        self.assertEqual(len(p['jobs']), 42)
        self.assertEqual(sum(j['tier'] == 1 for j in p['jobs']), 12)
        for j in p['jobs']:
            import yaml
            c = yaml.safe_load(Path(j['cfg']).read_text(encoding='utf-8'))
            self.assertEqual(c['TRAIN']['max_epoch'], 30)
            self.assertEqual(c['TRAIN']['seed'], j['seed'])
            self.assertEqual(c['MODEL']['formula_joint'], j['base_variant'] not in ('A', 'B', 'C'))
        self.assertEqual(read(work / 'checks/prior_formula.json')['constant_centered_max_abs'], 0.)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(HERE / 'local_test.json', {'tests_run': result.testsRun, 'failures': len(result.failures),
                                    'errors': len(result.errors), 'passed': result.wasSuccessful(),
                                    'actual_server_CUDA_training_performed': False,
                                    'scope': 'Synthetic CPU math/gradient/native-regression/queue/report/preparation checks; no research metrics'})
    sys.exit(0 if result.wasSuccessful() else 1)
