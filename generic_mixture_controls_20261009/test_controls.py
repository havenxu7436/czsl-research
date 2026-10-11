"""CPU-only synthetic tests; no claim of a real CUDA experiment."""
import argparse
import ast
import copy
import importlib.util
import math
import pickletools
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
import torch
from bootstrap import ROOT, child, guard_path, read_json, sha, write_json
from common import state_sha, tensor_load, tensor_save
import export_controls
from model import ALPHA, MODES, NEW_VARIANTS, fresh_head, from_grid, initializations, mix_scores, scored
import run_controls
import runtime as legacy_runtime


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        cls.ref = read_json(ROOT/'legacy/reference.json')
        cls.exp = read_json(ROOT/'experiment.json')
        cls.controls = cls.ref['control_protocol']
        gen = torch.Generator().manual_seed(941)
        cls.bank = {axis: dict(units=torch.randn(n, 4, 300, generator=gen), mask=torch.ones(n, 4))
                    for axis, n in (('a',16),('o',12))}
        cls.init = initializations(cls.bank, cls.controls)
        cls.settings = copy.deepcopy(cls.ref['pilot_settings'])
        cls.settings.update(epochs=2, batch_size=16)
        attrs = ['a%02d'%i for i in range(16)]
        objs = ['o%02d'%i for i in range(12)]
        pairs = [(a,o) for a in attrs for o in objs]
        split = dict(train=pairs[:96], val=pairs[96:160], test=pairs[160:])
        cls.datasets = {s: legacy_runtime.make_dataset(dict(labels=dict(attrs=attrs, objs=objs)), s, split) for s in ('cw','ow')}
        def cache(a,o):
            n=len(a); scores=torch.rand(n,12,16,generator=gen)*.2; rows=torch.arange(n)
            scores[rows,o,a]+=.2; scores[rows[::3],(o[::3]+1)%12,a[::3]]+=.4
            return dict(features=torch.randn(n,300,generator=gen), scores=scores, attrs=a, objects=o)
        a=torch.arange(48).remainder(4).repeat_interleave(2)
        o=torch.arange(48).remainder(12).repeat_interleave(2)
        cls.caches=dict(train=cache(a,o), val=cache(torch.arange(96).remainder(12),torch.arange(96).remainder(12)))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='generic_controls_test_')
        self.work=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_01_python38_syntax(self):
        for p in ROOT.glob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'),filename=p.name,feature_version=(3,8))

    def test_02_fixed_alpha(self):
        self.assertEqual(ALPHA,.25)

    def test_03_modes_complete(self):
        self.assertEqual(set(MODES),{'G_K','G_name','G_perm1','G_perm2'})
        self.assertEqual(NEW_VARIANTS,('G_name','G_perm1','G_perm2'))

    def test_04_original_generic_parameter_count(self):
        for v in MODES:
            h=fresh_head(self.bank,v,self.controls,'cpu')
            self.assertEqual(sum(p.numel() for p in h.parameters()),60864)
            self.assertTrue(hasattr(h,'fusion'))

    def test_05_all_initial_parameters_exact(self):
        self.assertEqual(len({x['initial_state_sha256'] for x in self.init.values()}),1)

    def test_06_correct_content_exact(self):
        g=fresh_head(self.bank,'G_K',self.controls,'cpu')
        old=legacy_runtime.fresh_head(self.bank,'P_generic',self.controls,'cpu')
        self.assertTrue(all(torch.equal(v,old.state_dict()[k]) for k,v in g.state_dict().items()))
        self.assertTrue(torch.equal(g.a_knowledge,old.a_knowledge))

    def test_07_name_content_fills_slot(self):
        h=fresh_head(self.bank,'G_name',self.controls,'cpu')
        self.assertTrue(torch.equal(h.a_knowledge,h.a_name))
        self.assertTrue(torch.equal(h.o_knowledge,h.o_name))

    def test_08_permutation_only_descriptions(self):
        correct=fresh_head(self.bank,'G_K',self.controls,'cpu')
        for v in ('G_perm1','G_perm2'):
            h=fresh_head(self.bank,v,self.controls,'cpu')
            self.assertTrue(torch.equal(h.a_name,correct.a_name))
            self.assertTrue(torch.equal(h.o_name,correct.o_name))
            self.assertTrue(torch.equal(h.a_knowledge,correct.a_knowledge[h.permutations['a']]))

    def test_09_zero_head_exact_base(self):
        base=self.caches['val']['scores']; h=fresh_head(self.bank,'G_K',self.controls,'cpu')
        self.assertTrue(torch.equal(h(self.caches['val']['features'],base),base))

    def test_10_mix_same_input_exact(self):
        base=self.caches['val']['scores']
        self.assertTrue(torch.equal(mix_scores(base,base),base))

    def test_11_mixture_joint_identity(self):
        torch.manual_seed(1); b=torch.randn(7,3,5,dtype=torch.float64); t=torch.randn(7,3,5,dtype=torch.float64)
        g=mix_scores(b,t); pb=torch.softmax(b.flatten(1)/.05,1).reshape_as(b)
        expected=.75*pb+.25*pb.sum(-1,keepdim=True)*torch.softmax(t/.05,-1)
        self.assertLess(float((torch.softmax(g.flatten(1)/.05,1).reshape_as(b)-expected).abs().max()),1e-12)

    def test_12_object_marginal_exact(self):
        b=self.caches['val']['scores']; t=b+torch.randn_like(b)*.03; g=mix_scores(b,t)
        pb=torch.softmax(b.flatten(1)/.05,1).reshape_as(b); pg=torch.softmax(g.flatten(1)/.05,1).reshape_as(b)
        self.assertLess(float((pb.sum(-1)-pg.sum(-1)).abs().max()),1e-5)

    def test_13_total_variation_bound(self):
        b=self.caches['val']['scores']; t=b+torch.randn_like(b)*.3; g=mix_scores(b,t)
        pb=torch.softmax(b.flatten(1)/.05,1); pg=torch.softmax(g.flatten(1)/.05,1)
        self.assertLessEqual(float(.5*(pb-pg).abs().sum(1).max()),ALPHA+1e-5)

    def test_14_full_grid_required(self):
        with self.assertRaises(ValueError):
            mix_scores(torch.zeros(2,4),torch.zeros(2,4))

    def test_15_diagnostics_consistent(self):
        c=self.caches['val']; grid,diag,vectors=from_grid(c['scores'],c['scores']+.01,c['attrs'],c['objects'])
        self.assertEqual(diag['n_images'],len(c['attrs']))
        self.assertEqual(len(vectors['correct']),len(c['attrs']))
        self.assertLessEqual(diag['joint_total_variation_max'],ALPHA+1e-5)

    def test_16_training_uses_unmixed_conditional_ce(self):
        h=fresh_head(self.bank,'G_name',self.controls,'cpu'); opt=legacy_runtime.optimizer(h,self.settings)
        report=legacy_runtime.train_epoch(h,opt,self.caches['train'],torch.arange(96),1,'cpu',16)
        self.assertEqual(report['steps'],6)
        self.assertEqual(set(report['nonzero_gradient_parameters']),set(dict(h.named_parameters())))

    def test_17_scored_preserves_raw_diagnostics(self):
        h=fresh_head(self.bank,'G_name',self.controls,'cpu'); grid,diag,vec,raw=scored(h,self.caches['val'],'cpu')
        self.assertTrue(torch.equal(grid,self.caches['val']['scores']))
        self.assertEqual(diag['unmixed_head_diagnostics']['max_abs_delta'],0)

    def test_18_two_epoch_formal_control(self):
        result=run_controls.legacy_pilot.formal_variant(self.work,'G_name',self.caches,self.bank,self.controls,
                self.datasets,EVALUATOR,self.settings,'synthetic_protocol',self.init['G_name'],'cpu')
        self.assertEqual(result['formal_optimizer_steps'],12)
        self.assertTrue(result['selected_checkpoint_replay_exact'])
        self.assertEqual(len(read_json(self.work/'runs/G_name/history.json')['ledger']),3)

    def test_19_formal_resume_is_exact(self):
        first=run_controls.legacy_pilot.formal_variant(self.work,'G_perm1',self.caches,self.bank,self.controls,
                self.datasets,EVALUATOR,self.settings,'synthetic_protocol',self.init['G_perm1'],'cpu')
        before=sha(self.work/'runs/G_perm1/last.pt')
        second=run_controls.legacy_pilot.formal_variant(self.work,'G_perm1',self.caches,self.bank,self.controls,
                self.datasets,EVALUATOR,self.settings,'synthetic_protocol',self.init['G_perm1'],'cpu')
        self.assertEqual(before,sha(self.work/'runs/G_perm1/last.pt'))
        self.assertEqual(first['selected']['metrics_raw'],second['selected']['metrics_raw'])

    def test_20_first_strict_selection(self):
        original=run_controls.evaluate_epoch
        def controlled(*args,**kwargs):
            r=original(*args,**kwargs); r['metrics_raw']['cw']['AUC']=.4 if r['epoch']==0 else .41; return r
        with patch.object(run_controls.legacy_pilot,'evaluate_epoch',side_effect=controlled):
            result=run_controls.legacy_pilot.formal_variant(self.work,'G_perm2',self.caches,self.bank,self.controls,
                    self.datasets,EVALUATOR,self.settings,'tie_protocol',self.init['G_perm2'],'cpu')
        self.assertEqual(result['best_epoch'],1)

    def test_21_checkpoint_avoids_binfloat(self):
        p=self.work/'safe.pt'; tensor_save(p,dict(value=1.25,nested=(2.5,[3.75])))
        self.assertEqual(tensor_load(p)['nested'],(2.5,[3.75]))
        with zipfile.ZipFile(p) as z:
            name=next(x for x in z.namelist() if x.endswith('/data.pkl'))
            ops=[op.name for op,_,_ in pickletools.genops(z.read(name))]
        self.assertNotIn('BINFLOAT',ops)

    def metric_fixture(self):
        base={s:{k:.4 for k in legacy_runtime.FIELDS} for s in ('cw','ow')}
        return base,{v:dict(selected=dict(metrics_raw=copy.deepcopy(base))) for v in MODES}

    def test_22_equal_controls_fail_gate(self):
        base,results=self.metric_fixture()
        self.assertFalse(run_controls.decision(results,base,self.ref['budget_gate'])['generic_budget_gate_passed'])

    def test_23_correct_advantage_passes_gate(self):
        base,results=self.metric_fixture(); results['G_K']['selected']['metrics_raw']['cw']['AUC']+=.004
        self.assertTrue(run_controls.decision(results,base,self.ref['budget_gate'])['generic_budget_gate_passed'])

    def test_24_ow_damage_fails_gate(self):
        base,results=self.metric_fixture(); m=results['G_K']['selected']['metrics_raw']; m['cw']['AUC']+=.004; m['ow']['AUC']-=.01
        self.assertFalse(run_controls.decision(results,base,self.ref['budget_gate'])['generic_budget_gate_passed'])

    def test_25_export_allowlist(self):
        self.assertTrue(export_controls.allowed('runs/G_name/epochs/epoch_0001/attempt_0000/unmixed_grid_scores.npy'))
        self.assertTrue(export_controls.allowed('runs/G_K/reference_replays/attempt_0000/epoch_0010/grid_scores.npy'))
        self.assertFalse(export_controls.allowed('train.pt'))
        self.assertFalse(export_controls.allowed('runs/G_name/backbone.pt'))

    def test_26_child_escape_refused(self):
        for name in ('../x','/x','C:/x','a\\b'):
            with self.assertRaises(ValueError): child(self.work,name)

    def test_27_reference_is_negative_and_fixed(self):
        self.assertFalse(self.exp['source_old_gate'])
        self.assertEqual(self.exp['alpha'],.25)
        self.assertEqual(self.exp['new_formal_variants'],list(NEW_VARIANTS))

    def test_28_source_manifest_matches_vendored_release(self):
        old=read_json(ROOT/'legacy/release_manifest.json')
        self.assertTrue(all(sha(ROOT/'legacy'/name)==digest for name,digest in old.items()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluator',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    global EVALUATOR
    if sha(args.evaluator)!='24b48c4d48e3475baec0a656fca194cdbc6d4b4e32e87c0f22cda3daf586d978':
        raise ValueError('Use the verified native evaluator')
    spec=importlib.util.spec_from_file_location('generic_test_evaluator',args.evaluator)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); EVALUATOR=module.Evaluator
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ControlTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(args.out,dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
              errors=[(str(t),e) for t,e in result.errors],failures=[(str(t),e) for t,e in result.failures],
              scope='CPU synthetic generic content controls, fixed mixture probability identities, exact resume, gates and guards',
              real_cuda_run=False,python=sys.version,torch=str(torch.__version__),native_evaluator_sha256=sha(args.evaluator)))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':
    main()
