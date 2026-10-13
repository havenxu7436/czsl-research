"""CPU algebra, integration, frozen configuration and asset tests; no accuracy claim."""
import importlib.util
import json
import math
import shutil
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
import yaml
from common import read,write,sha,verify_files
from revision_math import *
from build_revision_assets import hierarchy,load_corpus,sample_variant,materialize,external_prior
from prepare_revision import CORE,SMALL,EXTERNAL,variant_options,prepare
from posthoc_revision import remove_prior,choose_bias,evaluate_bias,analyze_run
from report_revision import summarize

HERE=Path(__file__).absolute().parent
torch.set_num_threads(2)
torch.manual_seed(100)


def fixture_encoded():
    labels=read(HERE/'assets/logic_bank/manifest.json')['labels']
    names,ids=hierarchy(load_corpus(),labels['objs'])
    desc={'oe':F.normalize(torch.randn(12,5,300),dim=-1),'ae':F.normalize(torch.randn(16,5,300),dim=-1)}
    pe=torch.zeros(3,8,300);pm=torch.zeros(3,8,dtype=torch.bool)
    for i,level in enumerate(names):
        pe[i,:len(level)]=F.normalize(torch.randn(len(level),300),dim=-1);pm[i,:len(level)]=True
    return desc,pe,ids,pm,dict(labels=labels,parents=names,synthetic_fixture=True)


def make_cache(path):
    path.mkdir(parents=True)
    labels=read(HERE/'assets/logic_bank/manifest.json')['labels']
    write(path/'labels.json',labels)
    for filename,n in (('W_o_name_raw.pt',12),('W_a_name_raw.pt',16)):
        torch.save(torch.randn(n,300),path/filename)
    (path/'W_ao_K').mkdir()
    for i in range(12): torch.save(F.normalize(torch.randn(16,300),dim=-1),path/'W_ao_K'/('obj_{:05d}.pt'.format(i)))
    files={p.relative_to(path).as_posix():sha(p) for p in path.rglob('*') if p.is_file()}
    write(path/'manifest.json',dict(attrs=labels['attrs'],objs=labels['objs'],dimension=300,dtype='float32',pair_axis_order=['object','attribute'],files_sha256=files))
    return labels


def load_model():
    package=types.ModuleType('_revision_models');package.__path__=[str(HERE/'source/models')]
    sys.modules[package.__name__]=package
    for name in ('ddm_utils','bf_logic_loss','formula_model','revision_math','revision_model'):
        filename=HERE/(name+'.py') if name.startswith('revision_') else HERE/'source/models'/(name+'.py')
        spec=importlib.util.spec_from_file_location(package.__name__+'.'+name,filename)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return sys.modules[package.__name__+'.revision_model'].RevisionMixin


class Algebra(unittest.TestCase):
    def setUp(self):
        self.o=torch.randn(5,4,requires_grad=True);self.u=torch.randn(5,4,3,requires_grad=True)
    def test_joint_normalized(self):
        s,lo,lc=joint_log_probability(self.o,self.u)
        self.assertTrue(torch.allclose(s.exp().sum((1,2)),torch.ones(5),atol=1e-6))
        self.assertTrue(torch.allclose(s.exp().sum(2),self.o.softmax(-1),atol=1e-6))
        self.assertTrue(torch.allclose(s.exp().sum(1).sum(1),torch.ones(5),atol=1e-6))
    def test_row_offset_invariance(self):
        offset=torch.randn(5,4,1)*100
        a=joint_log_probability(self.o,self.u)[0];b=joint_log_probability(self.o+10,self.u+offset)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-5))
    def test_prior_row_constant_invariance(self):
        a=joint_log_probability(self.o,self.u)[0]
        b=joint_log_probability(self.o,self.u,torch.randn(4,1).expand(4,3),0.8)[0]
        self.assertTrue(torch.allclose(a,b,atol=2e-6))
    def test_candidate_ce_reference(self):
        s=joint_log_probability(self.o,self.u)[0][:,[0,1,2],[1,2,0]]
        y=torch.tensor([0,1,2,0,1])
        ce=F.cross_entropy(s,y)
        ref=-(s[torch.arange(5),y]-torch.logsumexp(s,1)).mean()
        self.assertTrue(torch.allclose(ce,ref))
    def test_marginal_uses_object_probabilities(self):
        s=joint_log_probability(self.o,self.u)[0]
        expected=torch.einsum('bo,boa->ba',self.o.softmax(-1),self.u.softmax(-1))
        self.assertTrue(torch.allclose(torch.logsumexp(s,1).exp(),expected,atol=1e-6))
    def test_prior_constant_singleton(self):
        for mode in ('centered','clamped_log'):
            self.assertEqual(make_prior(torch.ones(1,1),mode).abs().max().item(),0)
            self.assertEqual(make_prior(torch.ones(4,3),mode).abs().max().item(),0)
    def test_clamped_log_finite_zeros(self):
        p=make_prior(torch.tensor([[0.,1.,0.]]),'clamped_log')
        self.assertTrue(torch.isfinite(p).all());self.assertAlmostEqual(float(p.sum()),0,places=5)
    def test_invalid_prior(self):
        for k in (torch.tensor([[float('nan')]]),torch.ones(2)):
            with self.assertRaises(ValueError): make_prior(k)
        with self.assertRaises(ValueError): make_prior(torch.tensor([[-1.]]),'clamped_log')
    def test_prior_readout_removal(self):
        p=torch.randn(4,3)
        withp=joint_log_probability(self.o,self.u,p,0.1)[0].detach().numpy()
        expected=joint_log_probability(self.o,self.u)[0].detach().numpy()
        self.assertTrue(np.allclose(remove_prior(withp,p.numpy(),0.1),expected,atol=2e-6))
    def test_js_identity_and_gradient(self):
        p=self.o.log_softmax(-1);q=torch.randn(5,4,requires_grad=True).log_softmax(-1)
        self.assertLess(abs(float(js_divergence(p,p))),1e-6)
        loss=js_divergence(p,q);loss.backward()
        self.assertGreater(self.o.grad.abs().sum().item(),0)
        self.assertLessEqual(float(loss),math.log(2)+1e-6)
    def test_descriptors_norm_invariant(self):
        x=torch.randn(4,300);d=torch.randn(3,5,300);mask=torch.ones(3,5,dtype=torch.bool)
        a=descriptor_logprob(x,d,mask);b=descriptor_logprob(x*20,d*17,mask)
        self.assertTrue(torch.allclose(a,b,atol=2e-6))
    def test_mask_removed_slots_have_no_effect(self):
        x=torch.randn(4,300);d=torch.randn(3,5,300);m=torch.ones(3,5,dtype=torch.bool);m[:,4]=False
        a=descriptor_logprob(x,d,m);d[:,4]=torch.randn(3,300)*1000
        self.assertTrue(torch.equal(a,descriptor_logprob(x,d,m)))
    def test_descriptor_empty_rejected(self):
        with self.assertRaises(ValueError): descriptor_logprob(torch.randn(4,300),torch.randn(3,5,300),torch.zeros(3,5,dtype=torch.bool))
    def test_nested_deletion(self):
        a,b,c=[deletion_mask(16,3,f,603) for f in (.1,.3,.5)]
        self.assertTrue((~a<=~b).all());self.assertTrue((~b<=~c).all())
        self.assertEqual((~c).sum().item(),24)
    def test_schedule_absolute_epoch(self):
        self.assertTrue(np.allclose([beta_at(e,.01,2,3) for e in range(1,7)],[0,0,.01/3,.02/3,.01,.01],atol=1e-12))
    def test_learn_lambda_bounded_grad(self):
        w=torch.tensor(math.log(.1/.9),requires_grad=True)
        s=joint_log_probability(self.o,self.u,torch.randn(4,3),w.sigmoid())[0]
        F.nll_loss(torch.logsumexp(s,1),torch.tensor([0,1,2,0,1])).backward()
        self.assertGreater(abs(float(w.grad)),0);self.assertTrue(0<float(w.sigmoid())<1)
    def test_attention_and_mean_finite(self):
        n,d,m=torch.randn(4,300),torch.randn(4,5,300),torch.zeros(4,5,dtype=torch.bool)
        for method in ('mean','attention'):
            x=aggregate_descriptions(n,d,m,method)
            self.assertTrue(torch.allclose(x,F.normalize(n,dim=-1),atol=1e-6))


class Heads(unittest.TestCase):
    def setUp(self):
        self.va=torch.randn(3,300);self.vo=torch.randn(3,300)
        self.ea=torch.randn(5,300);self.eo=torch.randn(4,300);self.ep=torch.randn(4,5,300)
    def test_chunk_forward_backward_identity(self):
        a=CoupledHead(chunk=1);b=CoupledHead(chunk=10);b.load_state_dict(a.state_dict())
        x=a(self.va,self.vo,self.ea,self.eo,self.ep);y=b(self.va,self.vo,self.ea,self.eo,self.ep)
        self.assertTrue(torch.allclose(x,y,atol=3e-6))
        x.sum().backward();y.sum().backward()
        for p,q in zip(a.parameters(),b.parameters()): self.assertTrue(torch.allclose(p.grad,q.grad,atol=2e-5))
    def test_image_object_attribute_interaction(self):
        h=CoupledHead();z=h(self.va,self.vo,self.ea,self.eo,None)
        difference=(z[:,0,0]-z[:,0,1])-(z[:,1,0]-z[:,1,1])
        self.assertGreater(float(difference.std()),1e-5)
    def test_gt_attr_gradient_reaches_object_projection(self):
        h=CoupledHead();z=h(self.va,self.vo,self.ea,self.eo,self.ep)
        F.cross_entropy(z[:,0],torch.tensor([0,1,2])).backward()
        self.assertGreater(float(h.ko.weight.grad.norm()),0)
        self.assertGreater(float(h.q.weight.grad.norm()),0)
    def test_static_image_independence(self):
        h=CoupledHead(static=True)
        a=h(self.va,self.vo,self.ea,self.eo,self.ep);b=h(self.va+2,self.vo-2,self.ea,self.eo,self.ep)
        self.assertTrue(torch.equal(a,b))
    def test_no_pair_unused_pair_projection(self):
        h=CoupledHead();h(self.va,self.vo,self.ea,self.eo).sum().backward()
        self.assertIsNone(h.kp.weight.grad)


class AssetsConfig(unittest.TestCase):
    def test_hierarchy_levels_real_nested_partitions(self):
        e=fixture_encoded();ids=e[2]
        self.assertEqual([len(row.unique()) for row in ids],[8,5,3])
        for low,high in zip(ids,ids[1:]):
            for label in low.unique():self.assertEqual(len(high[low==label].unique()),1)
    def test_five_distinct_sentences(self):
        c=load_corpus()
        for kind in ('oe','ae'):
            for value in c[kind].values(): self.assertEqual(len(set(value)),5)
    def test_core_controls_options(self):
        self.assertEqual(len(CORE),13);self.assertEqual(len(SMALL),19)
        self.assertFalse(variant_options('Rfull_nopair_head')['use_pair'])
        self.assertTrue(variant_options('Rfull_nopair_head')['prior'])
        self.assertFalse(variant_options('Rfull_nopair_all')['prior'])
        for name in CORE[1:]+SMALL+EXTERNAL:self.assertTrue(variant_options(name)['rev_enabled'])
    def test_assets_mask_prior_held_and_checksums(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'original';make_cache(src);e=fixture_encoded()
            k=materialize(src,root/'main',e)
            materialize(src,root/'noise',e,noise=50,fixed_k=k)
            bank=torch.load(root/'noise/revision_bank.pt',weights_only=True)
            self.assertTrue(torch.equal(k,bank['K']))
            self.assertTrue(bank['oe_mask'].any(-1).all());self.assertTrue(bank['ae_mask'].any(-1).all())
            manifest=read(root/'noise/manifest.json')
            verify_files({str(root/'noise'/name):h for name,h in manifest['files_sha256'].items()})
            evidence=read(root/'noise/revision_evidence.json')
            self.assertEqual(len(evidence['deleted_slots']['ae']),24)
    def test_external_no_leakage_and_missing_scores(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'prior.json';labels={'attrs':['a'],'objs':['o']}
            value=dict(labels=labels,score_axes=['object','attribute'],scores=[[.7]],known=[[False]],
                       provenance=dict(generator='fixture',prompt='fixture',generated_at='fixture',used_split_membership=False,used_validation_or_test_results=False))
            write(p,value)
            with self.assertRaises(ValueError): external_prior(p,labels,'mixed',torch.tensor([[.3]]))
            with self.assertRaises(ValueError): external_prior(p,labels,'llm',torch.tensor([[.3]]))
            value['known']=[[True]];write(p,value)
            self.assertAlmostEqual(external_prior(p,labels,'mixed',torch.tensor([[.3]])).item(),.5,places=6)
            value['provenance']['used_split_membership']=True;write(p,value)
            with self.assertRaises(ValueError): external_prior(p,labels,'mixed',torch.tensor([[.3]]))
    def test_bias_ties_deterministic(self):
        scores=np.array([[4.,0.],[0.,4.]])
        best,_=choose_bias(scores,np.array([0,1]),np.array([True,False]),np.array([True,False]))
        self.assertEqual(best['bias'],0);self.assertEqual(best['hm'],1)
    def test_all_prepare_configuration_and_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            root,work=Path(td)/'root',Path(td)/'work'
            original=root/'knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k';make_cache(original)
            data=root/'data/ut-zap50k/compositional-split-natural';data.mkdir(parents=True)
            for part in ('train','val','test'):(data/(part+'_pairs.txt')).write_text('Canvas Boots.Ankle\n')
            glove=root/'data/glove/glove.6B.300d.txt';glove.parent.mkdir();glove.write_text('synthetic fixture')
            e=fixture_encoded();e[-1]['glove_sha256']=sha(glove)
            def scaffold(r,w,tier):
                (w/'configs').mkdir(parents=True);(w/'source').mkdir();(w/'logic_bank').mkdir()
                return dict(frozen_inputs={})
            with mock.patch('prepare_revision.verify_release'),mock.patch('prepare_revision.legacy_prepare.prepare',side_effect=scaffold),mock.patch('prepare_revision.encode_assets',return_value=e):
                p=prepare(root,work,10,[3351,3352,3353],suite='all')
            self.assertEqual(len(p['jobs']),96)
            self.assertEqual(len({j['variant'] for j in p['jobs']}),96)
            for j in p['jobs']:
                c=yaml.safe_load(Path(j['cfg']).read_text())
                self.assertEqual(c['TRAIN']['max_epoch'],10)
                self.assertEqual(c['MODEL']['kcc_feature'],variant_options(j['base_variant'])['feature'])
                self.assertEqual(sha(j['cfg']),j['cfg_sha256'])
            verify_files(p['frozen_inputs'])


class ModelIntegration(unittest.TestCase):
    def test_full_mixin_train_infer_resume_schedule_and_gradients(self):
        # CPU dummy backbone exercises actual RevisionMixin, not a retyped model.
        Mixin=load_model()
        class Classifier(nn.Module):
            temp=.05
            def forward(self,x,y,scale=True):return F.normalize(x,dim=-1)@F.normalize(y,dim=-1).t()/self.temp
        class Dummy(nn.Module):
            def __init__(self,dset,cfg):
                super().__init__();self.cfg=cfg;self.full_enabled=True;self.full_use_pair=cfg.MODEL.full_use_pair
                self.kcc_text=True;self.kcc_calib=True;self.kcc_lambda=.1
                self.bf_logic=types.SimpleNamespace(enabled=True,beta=.01)
                self.classifier=Classifier();self.num_objs=12;self.num_attrs=16
                self.tot_objs=torch.arange(12);self.tot_attrs=torch.arange(16)
                self.train_objs=torch.tensor([0,1,2,3]);self.train_attrs=torch.tensor([0,1,2,3]);self.val_pairs=list(range(4))
                self.val_objs=self.train_objs;self.val_attrs=self.train_attrs
                self.W_o_K=torch.randn(12,300);self.W_a_K=torch.randn(16,300);self.full_pair_embeddings=torch.randn(12,16,300)
                self.compose=nn.Linear(600,300);self.compose_visual=nn.Linear(600,300)
                self.objc_embedder=nn.Linear(300,300);self.attrc_embedder=nn.Linear(300,300)
            def _kcc_obj_embeddings(self,ids):return self.W_o_K[ids]
            def _kcc_attr_embeddings(self,ids):return self.W_a_K[ids]
        class Model(Mixin,Dummy):
            def compose_visual(self,vo,va):return self.visual_comp(torch.cat((vo,va),-1))
            def formula_features(self,batch):
                vo=self.objc_embedder(batch['img']);va=self.attrc_embedder(batch['img'])
                return vo,va,vo,self.W_o_K,self.W_a_K
        # Assigning an nn.Module to the inherited method name shadows dispatch:
        # Dummy constructor is adapted below so compose_visual stays a method.
        original_init=Dummy.__init__
        def init(self,dset,cfg):
            original_init(self,dset,cfg);self.visual_comp=self._modules.pop('compose_visual')
        Dummy.__init__=init
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'original';make_cache(src);cache=Path(td)/'new';materialize(src,cache,fixture_encoded())
            opt=variant_options('Rfull_delayed_beta')
            cm={k:v for k,v in opt.items() if k.startswith('rev_')}
            cm.update(formula_joint=True,kcc_cache=str(cache),full_use_pair=True)
            cfg=types.SimpleNamespace(MODEL=types.SimpleNamespace(**cm),TRAIN=types.SimpleNamespace(max_epoch=10,aug_epoch=37,seed=3351))
            m=Model(None,cfg)
            batch={'img':torch.randn(4,300),'obj':torch.arange(4),'attr':torch.arange(4),'pair':torch.arange(4)}
            out,z,n=m.formula_train_terms(batch,True)
            self.assertEqual(float(out['loss_logic_weighted']),0)
            m.set_epoch(5);out,z,n=m.formula_train_terms(batch,True)
            self.assertGreater(float(out['loss_logic_weighted']),0)
            infer=m.val_forward(batch)['pred'];self.assertTrue(torch.allclose(infer,z['score'][:,m.train_objs,m.train_attrs]))
            for term in ('loss_cpf','loss_oi','loss_oe','loss_ae'):
                g=torch.autograd.grad(out[term],m.objc_embedder.weight,retain_graph=True)[0]
                self.assertGreater(float(g.norm()),0)
            out['loss_total'].backward();self.assertGreater(float(m.full_head.ko.weight.grad.norm()),0)
            m.load_state_dict(m.state_dict(),strict=True)
            self.assertTrue(torch.allclose(n['pair'].sum((1,2)),torch.ones(4),atol=1e-6))


class DeliveryIntegration(unittest.TestCase):
    def test_config_values_merge_actual_yacs_defaults(self):
        spec=importlib.util.spec_from_file_location('_actual_defaults',HERE/'source/config/defaults.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        base=yaml.safe_load((HERE/'assets/A_ref.yml').read_text())
        for name in CORE+SMALL+EXTERNAL:
            c=module._C.clone();c.merge_from_other_cfg(type(c)(base))
            updates={k:v for k,v in variant_options(name).items() if k.startswith('rev_')}
            c.merge_from_other_cfg(type(c)({'MODEL':updates}))
            self.assertEqual(c.MODEL.rev_attr_mode,updates['rev_attr_mode'])
    def test_python38_syntax_and_source_hooks(self):
        import ast
        for filename in ('revision_math.py','revision_model.py','prepare_revision.py','build_revision_assets.py','posthoc_revision.py','report_revision.py','smoke_revision.py','start_revision.sh'):
            if filename.endswith('.py'):ast.parse((HERE/filename).read_text(encoding='utf-8'),feature_version=(3,8))
        text=(HERE/'source/models/kcc_cpf.py').read_text()
        self.assertIn('DDM(RevisionMixin, LegacyDDM)',text)
        train=(HERE/'source/train.py').read_text()
        self.assertEqual(train.count('model.set_epoch(epoch)'),1)
        self.assertIn("'weight_decay': 0.0",train)
    def test_completed_report_is_budget_and_checkpoint_strict(self):
        with tempfile.TemporaryDirectory() as td:
            work=Path(td);run=work/'runs/Rfull_s3351';run.mkdir(parents=True)
            job=dict(variant=run.name,base_variant='Rfull',seed=3351,tier=0,run=str(run))
            protocol=dict(jobs=[job],epochs=10,seeds=[3351],suite='fixture',formula='fixture',logic='fixture',knowledge='fixture',test_policy='fixture',not_covered=[])
            write(work/'protocol.json',protocol);write(run/'status.json',dict(state='complete',epochs=10))
            (run/'train.log').write_text('Val/Test Epoch 1: AUC=0.2 | best_hm=0.3\n')
            for phase in ('val','test'):
                for setting in ('cw','ow'):
                    write(run/('eval_'+phase+'_'+setting)/'metrics.json',dict(checkpoint_sha256='fixture-checkpoint',best_epoch=1,
                        metrics_raw=dict(AUC=.2,best_hm=.3,best_seen=.4,best_unseen=.5)))
            self.assertEqual(summarize(work),(True,True))
            r=read(work/'summary/summary.json');self.assertEqual(len(r['rows']),4)
            self.assertEqual(r['aggregates'][0]['AUC_pct']['sample_std'],None)
            m=read(run/'eval_test_ow/metrics.json');m['checkpoint_sha256']='different';write(run/'eval_test_ow/metrics.json',m)
            with self.assertRaises(ValueError):summarize(work)
    def test_posthoc_full_fixture_and_no_test_bias_selection(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)/'run';run.mkdir();cache=Path(td)/'cache';cache.mkdir()
            labels=dict(attrs=['a','b'],objs=['o','p']);write(cache/'labels.json',labels)
            k=torch.tensor([[.1,.8],[.7,.2]]);torch.save({'K':k},cache/'revision_bank.pt')
            cfg=dict(MODEL=dict(kcc_cache=str(cache),rev_enabled=True,kcc_calib=True,rev_prior_mode='centered',rev_prior_floor=1e-3))
            (run/'config_requested.yml').write_text(yaml.safe_dump(cfg))
            pairs=[['a','o'],['b','o'],['a','p'],['b','p']]
            obj=torch.tensor([[4.,0.],[4.,0.],[0.,4.],[0.,4.]])
            u=torch.tensor([[[4.,0.],[0.,0.]],[[0.,4.],[0.,0.]],[[0.,0.],[4.,0.]],[[0.,0.],[0.,4.]]])
            mat=joint_log_probability(obj,u,make_prior(k),.1)[0].reshape(4,4).numpy()
            for phase in ('val','test'):
                for setting in ('cw','ow'):
                    d=run/('eval_'+phase+'_'+setting);d.mkdir()
                    write(d/'metrics.json',dict(score_shape=[4,4],effective_parameters={'lambda_effective':.1}))
                    write(d/'candidates.json',pairs);write(d/'image_ids.json',['fixture1','fixture2','fixture3','fixture4'])
                    np.savez(str(d/'labels.npz'),pair=np.arange(4),attr=np.array([0,1,0,1]),obj=np.array([0,0,1,1]))
                    mm=np.memmap(str(d/'scores.dat'),mode='w+',dtype=np.float32,shape=(4,4));mm[:]=mat;mm.flush();del mm
            result=analyze_run(run,{},dict(train=[['a','o'],['b','p']]))
            self.assertEqual(result['settings']['ow']['VAL_selected']['bias'],0)
            self.assertEqual(result['settings']['ow']['TEST_at_VAL_bias']['accuracy'],1)
            self.assertEqual(result['settings']['ow']['error_counts']['both_correct'],4)
            self.assertEqual(result['settings']['ow']['prior_readout_diagnostic']['neutral'],4)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if args.out:
        write(args.out,dict(passed=result.wasSuccessful(),tests=result.testsRun,
            failures=len(result.failures),errors=len(result.errors),torch=torch.__version__,scope='CPU algebra + synthetic asset/config/model integration; no CUDA or accuracy result'))
    sys.exit(0 if result.wasSuccessful() else 1)
