"""CPU wiring/regression tests; synthetic fixtures are never research results."""
import ast
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from build_release import OLD, ROOT, HERE, patch_model, patch_cache, patch_train
from common import sha, read, write


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType('full_test')
package.__path__ = [str(HERE)]
sys.modules['full_test'] = package
load('full_test.kcc_cache', OLD / 'models/kcc_cache.py')
full = load('full_test.full_feature', HERE / 'full_feature.py')
load('full_test.bf_logic_loss', OLD / 'models/bf_logic_loss.py')
load('full_test.bf_logic', OLD / 'models/bf_logic.py')
utils = load('full_test.actual_utils', OLD / 'models/ddm_utils.py')


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = 0.05

    def forward(self, x, w, scale=True):
        z = F.normalize(x, dim=-1) @ F.normalize(w, dim=-1).t()
        return z / self.temp if scale else z


class FakeFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 768)

    def forward(self, x):
        c = self.layer(x)
        p = c[:, None, :].expand(-1, 4, -1)
        return p * .7, p * .8, p * .9, p, c


class FakeBaseline(nn.Module):
    def __init__(self, ds, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_objs, self.num_attrs, self.word_dim = len(ds.objs), len(ds.attrs), 300
        self.obj_embedder, self.attr_embedder = nn.Embedding(12, 300), nn.Embedding(16, 300)
        self.tot_objs, self.tot_attrs = torch.arange(12), torch.arange(16)
        self.train_objs, self.train_attrs = torch.tensor([i % 12 for i in range(32)]), torch.tensor([i % 16 for i in range(32)])
        self.val_objs, self.val_attrs = self.train_objs, self.train_attrs
        self.val_pairs = [(ds.attrs[a], ds.objs[o]) for a, o in zip(self.train_attrs, self.train_objs)]
        self.feat_extractor = FakeFeatures()
        self.objc_embedder, self.attrc_embedder = nn.Linear(768, 300), nn.Linear(900, 300)
        self.obj_guide_embedder, self.objc_embedder_2 = nn.Linear(300, 768), nn.Linear(768, 300)
        self.compose, self.projection = nn.Linear(600, 300), nn.Linear(600, 300)
        self.classifier = Classifier()

    def compose_visual(self, obj, attr):
        return self.projection(torch.cat((obj, attr), dim=1))


class FakeCache:
    def _setup_kcc(self, ds, cfg):
        self.kcc_text, self.kcc_calib, self.kcc_lambda = cfg.MODEL.kcc_text, cfg.MODEL.kcc_calib, cfg.MODEL.kcc_lambda
        self.W_o_K = F.normalize(self.obj_embedder.weight.detach().clone(), dim=-1)
        self.W_a_K = F.normalize(self.attr_embedder.weight.detach().clone(), dim=-1)
        self.K_oa_prior = torch.arange(192).reshape(12, 16).float() / 100

    def _kcc_obj_embeddings(self, indices):
        return self.W_o_K[indices] if self.kcc_text else self.obj_embedder(indices)

    def _kcc_attr_embeddings(self, indices):
        return self.W_a_K[indices] if self.kcc_text else self.attr_embedder(indices)

    def _kcc_calibrate(self, scores, objs, attrs):
        return scores + self.kcc_lambda * self.K_oa_prior[objs, attrs][None] if self.kcc_calib else scores


ddm = types.ModuleType('full_test.ddm')
for key, val in dict(torch=torch, nn=nn, F=F, np=np, DDM=FakeBaseline,
                     choose_texts=utils.choose_texts, choose_visual_tokens=utils.choose_visual_tokens,
                     image_soft_weights=utils.image_soft_weights).items():
    setattr(ddm, key, val)
sys.modules['full_test.ddm'] = ddm
sys.modules['full_test.kcc_cache'].KCCCacheMixin = FakeCache
native = types.ModuleType('full_test.native')
native.__package__ = 'full_test'
exec(compile((OLD / 'models/kcc_cpf.py').read_text(encoding='utf-8'), 'audited_bf_model', 'exec'), native.__dict__)
patched = types.ModuleType('full_test.patched')
patched.__package__ = 'full_test'
exec(compile(patch_model((OLD / 'models/kcc_cpf.py').read_text(encoding='utf-8')), 'full_model', 'exec'), patched.__dict__)


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        bank = HERE / 'assets/logic_bank' if (HERE / 'assets/logic_bank').exists() else ROOT / 'output/BF_conditional_preflight_20261009/extracted/bf/logic_bank'
        self.ds = types.SimpleNamespace(**read(bank / 'manifest.json')['labels'])
        self.cache = self.path / 'cache'
        (self.cache / 'W_ao_K').mkdir(parents=True)
        hashes = {}
        for i in range(12):
            path = self.cache / 'W_ao_K' / ('obj_{:05d}.pt'.format(i))
            torch.save(F.normalize(torch.randn(16, 300), dim=-1), path)
            hashes[str(path.relative_to(self.cache)).replace('\\', '/')] = sha(path)
        write(self.cache / 'manifest.json', {'files_sha256': hashes})
        m = dict(kcc_feature=True, kcc_text=True, kcc_calib=True, kcc_lambda=.1,
                 kcc_logic=True, kcc_beta=.01, bf_logic_bank=str(bank), bf_logic_q=3,
                 bf_allow_reconstructed=True, full_gain=1., full_use_pair=True, kcc_cache=str(self.cache))
        self.cfg = types.SimpleNamespace(MODEL=types.SimpleNamespace(**m), TRAIN=types.SimpleNamespace(max_epoch=10, aug_epoch=37))
        self.batch = {'img': torch.randn(4, 8), 'attr': torch.tensor([0, 1, 2, 3]),
                      'obj': torch.tensor([0, 1, 2, 3]), 'pair': torch.tensor([0, 1, 2, 3])}

    def tearDown(self):
        self.temp.cleanup()

    def model(self, cfg=None, old=False):
        torch.manual_seed(3351)
        return (native.DDM if old else patched.DDM)(self.ds, cfg or self.cfg)

    def test_head_shape_and_pair_branch(self):
        h = full.FactorizedKnowledgeConditionedHead(hidden_dim=32)
        va, vo, ea, eo, ep = [torch.randn(*s) for s in [(4,300),(4,300),(16,300),(12,300),(12,16,300)]]
        score = h(va, vo, ea, eo, ep)
        self.assertEqual(tuple(score.shape), (4,12,16))
        self.assertTrue(torch.allclose(score - h(va, vo, ea, eo), h.pair_bias(ep).squeeze(-1)[None], atol=1e-6))

    def test_marginal_actual_candidate_axis(self):
        scores = torch.tensor([[1.,2.,4.,8.]])
        result = full.attribute_marginal(scores, torch.tensor([0,1,0,1]), 2)
        expected = torch.stack([torch.logsumexp(scores[:,[0,2]],1), torch.logsumexp(scores[:,[1,3]],1)],1)
        self.assertTrue(torch.equal(result, expected))

    def test_marginal_missing_class_rejected(self):
        with self.assertRaises(ValueError):
            full.attribute_marginal(torch.zeros(2,2), torch.tensor([0,0]), 2)

    def test_all_A_F_disabled_regression(self):
        for name in 'ABCDEF':
            c = copy.deepcopy(self.cfg)
            c.MODEL.kcc_feature = False
            c.MODEL.kcc_text = name in 'CEF'
            c.MODEL.kcc_calib = name in 'DEF'
            c.MODEL.kcc_lambda = .1 if c.MODEL.kcc_calib else 0.
            c.MODEL.kcc_logic = name in 'BF'
            c.MODEL.kcc_beta = .01 if c.MODEL.kcc_logic else 0.
            old, new = self.model(c, old=True).eval(), self.model(c).eval()
            self.assertEqual(list(old.state_dict()), list(new.state_dict()))
            for key, value in old.state_dict().items():
                self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
            self.assertTrue(torch.equal(old.train_forward(self.batch)['loss_total'], new.train_forward(self.batch)['loss_total']), name)
            self.assertTrue(torch.equal(old.val_forward(self.batch)['pred'], new.val_forward(self.batch)['pred']), name)

    def test_rng_and_native_initialization_unchanged(self):
        c = copy.deepcopy(self.cfg)
        c.MODEL.kcc_feature = False
        off = self.model(c)
        rng_off = torch.get_rng_state()
        on = self.model()
        self.assertTrue(torch.equal(rng_off, torch.get_rng_state()))
        for k, v in off.state_dict().items():
            self.assertTrue(torch.equal(v, on.state_dict()[k]))

    def test_gain0_cpf_task_loss_and_scores_identity(self):
        c = copy.deepcopy(self.cfg)
        c.MODEL.kcc_feature = False
        off = self.model(c).eval()
        on = self.model().eval()
        on.full_gain = 0.
        self.assertTrue(torch.equal(off.train_forward(self.batch)['loss_cpf'], on.train_forward(self.batch)['loss_cpf']))
        self.assertTrue(torch.equal(off.val_forward(self.batch)['pred'], on.val_forward(self.batch)['pred']))

    def test_H_joint_gradients_and_beta_once(self):
        model = self.model().eval()
        out = model.train_forward(self.batch)
        self.assertTrue(torch.allclose(out['loss_total'], out['loss_cpf'] + .01 * out['loss_logic_raw']))
        head = list(model.full_head.parameters())
        grads = torch.autograd.grad(out['loss_total'], head, retain_graph=True)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(all(g.abs().sum() > 0 for g in grads))
        grads_ae = torch.autograd.grad(out['loss_ae'], head, retain_graph=True, allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads_ae if g is not None), 0)
        visual = list(model.objc_embedder.parameters()) + list(model.attrc_embedder.parameters())
        for name in ('loss_oi', 'loss_oe', 'loss_ae'):
            g = torch.autograd.grad(out[name], visual, retain_graph=True, allow_unused=True)
            self.assertGreater(sum(float(v.abs().sum()) for v in g if v is not None), 0)
        out['loss_total'].backward()
        self.assertGreater(float(model.feat_extractor.layer.weight.grad.abs().sum()), 0)

    def test_G_no_logic(self):
        c = copy.deepcopy(self.cfg)
        c.MODEL.kcc_logic, c.MODEL.kcc_beta = False, 0.
        out = self.model(c).train_forward(self.batch)
        self.assertTrue(torch.equal(out['loss_total'], out['loss_cpf']))
        self.assertGreater(float(out['loss_conditional_attr']), 0)

    def test_no_pair_keeps_other_gradients(self):
        c = copy.deepcopy(self.cfg)
        c.MODEL.full_use_pair = False
        model = self.model(c)
        model.train_forward(self.batch)['loss_total'].backward()
        self.assertIsNone(model.full_head.pair_bias[0].weight.grad)
        self.assertGreater(float(model.full_head.attr_vis_proj.weight.grad.abs().sum()), 0)

    def test_pair_hash_mismatch_rejected(self):
        with (self.cache / 'W_ao_K/obj_00000.pt').open('ab') as f:
            f.write(b'changed')
        with self.assertRaises(ValueError):
            self.model()

    def test_aug_budget_rejected(self):
        c = copy.deepcopy(self.cfg)
        c.TRAIN.aug_epoch = 10
        with self.assertRaises(ValueError):
            self.model(c)

    def test_source_patches_parse_python38(self):
        for name, fn in [('models/kcc_cpf.py',patch_model),('models/kcc_cache.py',patch_cache),('train.py',patch_train)]:
            ast.parse(fn((OLD / name).read_text(encoding='utf-8')), feature_version=8)
        for p in HERE.glob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'), feature_version=8)

    def test_derangement_has_no_fixed_points(self):
        from prepare import derangement
        for n in (12,16):
            p = derangement(n, 123)
            self.assertEqual(set(p.tolist()), set(range(n)))
            self.assertTrue(bool((p != torch.arange(n)).all()))

    def test_constant_prior_centered_is_zero(self):
        builder = load('full_test.builder', OLD / 'tools/build_knowledge.py')
        self.assertTrue(torch.equal(builder.centered_prior(torch.ones(12,16)), torch.zeros(12,16)))

    def test_random_prior_unit_population_std(self):
        builder = load('full_test.builder_random', OLD / 'tools/build_knowledge.py')
        x = builder.centered_prior(torch.rand(12,16,generator=torch.Generator().manual_seed(603354)))
        self.assertLess(abs(float(x.mean())), 1e-6)
        self.assertAlmostEqual(float(x.std(unbiased=False)), 1., places=6)

    def test_incomplete_report_never_invents_metrics(self):
        from report import summarize
        work = self.path / 'report'
        work.mkdir()
        write(work / 'protocol.json', {'jobs':[{'variant':'H','tier':0,'run':str(work/'runs/H')}], 'not_covered':['fixture']})
        core, complete = summarize(work)
        self.assertFalse(core)
        self.assertFalse(complete)
        self.assertEqual(read(work/'summary/summary.json')['rows'], [])

    def test_gpu_selection_excludes_busy_and_reserved(self):
        from run_suite import available_gpus
        text = '0, 2000, 22500, 0\n1, 13, 24500, 0\n2, 10, 24500, 90\n3, 13, 24500, 0\n'
        with mock.patch('subprocess.check_output', return_value=text):
            self.assertEqual(available_gpus(None, {3}), [1])
            self.assertEqual(available_gpus({0,3}, set()), [3])

    def test_real_result_report_scale_and_label_order(self):
        from report import summarize
        work = self.path / 'metric_fixture'
        work.mkdir()
        run = work / 'runs/A'
        write(run / 'status.json', {'state':'complete'})
        write(run / 'meta/labels.json', {'attrs':['Z','A'], 'objs':['O2','O1']})
        write(work / 'protocol.json', {'jobs':[{'variant':'A','tier':0,'run':str(run)},
              {'variant':'H','tier':0,'run':str(work/'runs/H')}], 'not_covered':['synthetic_fixture']})
        for phase in ('val','test'):
            for setting in ('cw','ow'):
                folder = run / ('eval_' + phase + '_' + setting)
                folder.mkdir()
                write(folder / 'metrics.json', {'best_epoch':5,'checkpoint_sha256':'fixture',
                      'metrics_raw':{'AUC':.5, 'best_hm':.6, 'best_seen':.7, 'best_unseen':.8}})
                write(folder / 'candidates.json', [['Z','O2'],['A','O1']])
                np.savez(folder/'labels.npz', attr=np.array([0,1]), obj=np.array([0,1]),
                         top5_indices_at_bias0=np.array([[0,1],[0,1]]))
        self.assertEqual(summarize(work), (False,False))
        result = read(work/'summary/summary.json')
        self.assertEqual(len(result['rows']), 4)
        self.assertEqual(result['rows'][0]['AUC_pct'], 50.)
        self.assertEqual(result['error_partition_bias0']['A_val_cw']['pair_correct_count'], 1)

    def test_altered_cache_preserves_names_pair_and_prior(self):
        from prepare import altered_cache, verify_cache
        builder = load('full_test.control_builder', OLD / 'tools/build_knowledge.py')
        cache = self.cache
        glove = self.path / 'glove.txt'
        glove.write_text('word ' + ' '.join(['1']*300) + '\n', encoding='utf-8')
        inputs = {}
        for which, labels in [('oe',self.ds.objs),('ae',self.ds.attrs)]:
            p = self.path / (which + '_dict.json')
            write(p, {label:['word','word word','word word word'] for label in labels})
            inputs[p.as_posix()] = sha(p)
        inputs[str(glove)] = sha(glove)
        for name, tensor in [('W_o_name_raw.pt',torch.randn(12,300)),('W_a_name_raw.pt',torch.randn(16,300)),
                             ('W_o_K.pt',F.normalize(torch.randn(12,300),dim=-1)),('W_a_K.pt',F.normalize(torch.randn(16,300),dim=-1)),
                             ('K_oa_text.pt',torch.rand(12,16)),('K_oa_prior_centered.pt',torch.randn(12,16))]:
            torch.save(tensor, cache / name)
        manifest = read(cache/'manifest.json')
        manifest.update(objs=self.ds.objs, attrs=self.ds.attrs, dimension=300, pair_axis_order=['object','attribute'], input_sha256=inputs)
        manifest['files_sha256'].update({p.name:sha(p) for p in cache.glob('*.pt')})
        write(cache/'manifest.json', manifest)
        write(cache/'cache_manifest.json', manifest)
        for kind in ('shuffle','noise10','noise30','noise50'):
            dest = self.path / kind
            altered_cache(cache, dest, kind, builder, glove)
            result = verify_cache(dest)
            for name in ('W_o_name_raw.pt','W_a_name_raw.pt','K_oa_text.pt','K_oa_prior_centered.pt','W_ao_K/obj_00000.pt'):
                self.assertEqual(sha(cache/name), sha(dest/name))
            self.assertEqual(result['control']['kind'], kind)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write(HERE / 'local_test.json', {'tests_run':result.testsRun, 'failures':len(result.failures),
          'errors':len(result.errors), 'passed':result.wasSuccessful(), 'real_cuda_training_performed':False,
          'scope':'Synthetic CPU wiring/regression tests only; real TRAIN CUDA smoke runs on the server'})
    sys.exit(0 if result.wasSuccessful() else 1)
