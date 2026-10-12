import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from transition.name_router import NameRouter
from diagnose import altered_bank, derangement, point_stats, probe, score_case


def bank():
    generator=torch.Generator().manual_seed(1907)
    result={}
    for axis,n in [('a',16),('o',12)]:
        units=F.normalize(torch.randn(n,4,300,generator=generator),dim=-1)
        mask=torch.ones(n,4,dtype=torch.bool)
        donor=torch.roll(torch.arange(n),1)
        shuffled=units.clone();shuffled[:,1:]=units[donor,1:]
        result[axis]=dict(unit=units[:,0].clone(),units=units,mask=mask,
            shuffle_units=shuffled,shuffle_mask=mask.clone(),correct=F.normalize(units.mean(1),dim=-1))
    return result


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.bank=bank()
        torch.manual_seed(77)
        self.feature=torch.randn(7,300)
        self.ai=torch.arange(16).repeat_interleave(12)
        self.oi=torch.arange(12).repeat(16)

    def router(self,mode,bank_value=None):
        torch.manual_seed(1729)
        router=NameRouter(self.bank if bank_value is None else bank_value,mode=mode)
        with torch.no_grad():router.raw_gain.fill_(-0.65)
        router.eval().requires_grad_(False)
        return router

    def test_probe_reproduces_all_three_trained_modes_and_reordered_candidates(self):
        for mode in ['name_only','ekr','shuffle']:
            router=self.router(mode)
            index=torch.randperm(192)[:123]
            ai,oi=self.ai[index],self.oi[index]
            expected=router(self.feature,ai,oi)
            actual,attention=probe(router,self.feature,ai,oi)
            self.assertTrue(torch.allclose(actual,expected,atol=1e-7,rtol=1e-6))
            self.assertTrue(torch.allclose(attention.sum(-1),torch.ones(7,123),atol=1e-6))

    def test_ae_intervention_is_deranged_and_preserves_own_names_and_input_bank(self):
        before=copy.deepcopy(self.bank)
        altered,mapping=altered_bank(self.bank,'ae_permutation',11)
        self.assertEqual(sorted(mapping),list(range(16)))
        self.assertTrue(all(i!=j for i,j in enumerate(mapping)))
        self.assertTrue(torch.equal(altered['a']['units'][:,0],self.bank['a']['units'][:,0]))
        self.assertTrue(torch.equal(altered['a']['units'][:,1:],self.bank['a']['units'][mapping,1:]))
        self.assertTrue(torch.equal(before['a']['units'],self.bank['a']['units']))

    def test_name_control_is_invariant_but_clean_router_responds_to_ae(self):
        altered,mapping=altered_bank(self.bank,'ae_permutation',11)
        for mode in ['name_only','ekr']:
            original=self.router(mode); changed=self.router(mode,altered)
            changed.load_state_dict(original.state_dict())
            a,_=probe(original,self.feature,self.ai,self.oi)
            b,_=probe(changed,self.feature,self.ai,self.oi)
            if mode=='name_only':self.assertTrue(torch.equal(a,b))
            else:self.assertGreater(float((a-b).abs().max()),1e-5)

    def test_null_removal_is_a_frozen_renormalization(self):
        router=self.router('ekr')
        before={k:v.clone() for k,v in router.state_dict().items()}
        d,a=probe(router,self.feature,self.ai,self.oi)
        changed,weights=probe(router,self.feature,self.ai,self.oi,remove_null=True)
        self.assertTrue(torch.equal(weights[...,0],torch.zeros_like(weights[...,0])))
        self.assertTrue(torch.allclose(changed,d/(1-a[...,0]),atol=1e-7,rtol=1e-5))
        self.assertTrue(all(torch.equal(v,router.state_dict()[k]) for k,v in before.items()))

    def test_margin_uses_new_strongest_competitor(self):
        base=np.array([[.5,.55,.2],[.5,.4,.1]],dtype=np.float32)
        residual=np.array([[.2,0,0],[0,0,.7]],dtype=np.float32)
        attention=np.full((2,3,5),.2,dtype=np.float32)
        labels=np.array([0,0]);allowed=np.array([True,True,True]); attrs=np.arange(3);objs=np.zeros(3,dtype=int)
        stats,p=point_stats(base,residual,attention,labels,allowed,attrs,objs,labels,np.zeros(2,dtype=int))
        self.assertEqual(stats['corrected'],1);self.assertEqual(stats['broken'],1)
        self.assertAlmostEqual(float(p['margin_delta'][0]),.2,places=6)
        self.assertAlmostEqual(float(p['margin_delta'][1]),-.4,places=6)
        self.assertAlmostEqual(float(p['fixed_competitor_residual_contrast'][1]),0,places=6)

    def test_closed_mask_excludes_disallowed_candidates(self):
        base=np.array([[.5,.4,9]],dtype=np.float32);residual=np.array([[0,0,100]],dtype=np.float32)
        attention=np.full((1,3,5),.2,dtype=np.float32)
        stats,p=point_stats(base,residual,attention,np.array([0]),np.array([True,True,False]),np.arange(3),np.zeros(3,dtype=int),np.array([0]),np.array([0]))
        self.assertEqual(stats['pair_pct'],100);self.assertEqual(stats['prediction_changed'],0)
        self.assertAlmostEqual(float(p['margin_delta'][0]),0)
        self.assertTrue(np.isfinite(stats['ce']))

    def test_score_case_persists_matching_scores_attention_and_per_image_margin(self):
        router=self.router('ekr')
        cache=dict(base=np.zeros((7,192),dtype=np.float32),feature=self.feature.numpy(),ai=self.ai,oi=self.oi,
            labels=dict(pair=np.zeros(7,dtype=np.int64),attr=np.zeros(7,dtype=np.int64),obj=np.zeros(7,dtype=np.int64)),
            allowed=np.ones(192,dtype=bool),candidate_attr=self.ai.numpy(),candidate_obj=self.oi.numpy())
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'case'
            scores,residual,attention,stats=score_case(router,cache,folder,False,batch_size=3)
            expected=router(self.feature,self.ai,self.oi).numpy()
            self.assertTrue(np.allclose(scores,expected,atol=1e-7))
            self.assertTrue(np.allclose(np.load(folder/'scores.npy'),scores))
            self.assertEqual(tuple(np.load(folder/'attention.npy').shape),(7,192,5))
            with np.load(folder/'per_image.npz',allow_pickle=False) as p:self.assertEqual(p['pred'].shape,(7,))
            del scores,residual,attention


if __name__=='__main__':unittest.main()
