"""Matched G0/G1 training definitions and interpretable OW ranking diagnostics."""
import unittest

import numpy as np
import torch

from tests.test_math import fixture
from vgcpod.common import ROOT,read_json
from vgcpod.model import VGCPOD
from vgcpod.engine import task_decomposition
from ranking_diagnostics import diagnose

torch.set_num_threads(1)


class RoundGTests(unittest.TestCase):
    def test_matched_configs_change_only_the_planned_fields(self):
        old=read_json(ROOT/'configs/round_f/F1.json')
        dynamic=read_json(ROOT/'configs/round_g/G1.json')
        static=read_json(ROOT/'configs/round_g/G0.json')
        self.assertEqual(set(old),set(dynamic));self.assertEqual(set(dynamic),set(static))
        self.assertEqual({k for k in old if old[k]!=dynamic[k]},{'task_aux_beta','experiment_id'})
        self.assertEqual({k for k in static if static[k]!=dynamic[k]},{'gate_input','experiment_id'})
        self.assertEqual(dynamic['task_aux_beta'],.075);self.assertEqual(dynamic['rho'],.15)
        self.assertEqual(dynamic['gate_eta'],.25);self.assertEqual(static['gate_input'],'none')

    def test_static_limit_and_beta075_task_gradient(self):
        reference,small,x,y,mask=fixture('V1')
        cfg=read_json(ROOT/'configs/round_g/G0.json')
        cfg.update(rank=small['rank'],adapter_rank=small['adapter_rank'])
        static=VGCPOD(reference.ea,reference.eo,reference.ec,cfg,3351).double()
        dynamic=VGCPOD(reference.ea,reference.eo,reference.ec,dict(cfg,gate_input='local'),3351).double()
        with torch.no_grad():static.Wc.normal_(0,.05)
        dynamic.load_state_dict(static.state_dict());dynamic.gate_eta=0
        keep=~((y[0]==2)&(y[1]==1))
        x=tuple(t[keep].double() for t in x);y=tuple(t[keep] for t in y)
        mask[1,2]=False
        out=static(*x);other=dynamic(*x)
        self.assertTrue(torch.equal(out['gamma'],torch.ones_like(out['gamma'])))
        self.assertTrue(torch.equal(out['cond'],other['cond']))
        self.assertFalse(static.Bvis.requires_grad)
        losses=static.objective(out,*y,mask)
        self.assertEqual(losses['orth'].item(),0.)
        self.assertAlmostEqual(losses['total'].item(),losses['task'].item(),places=13)
        expected=1.5*losses['pair']+(9/86)*losses['support']
        self.assertAlmostEqual(losses['task'].item(),expected.item(),places=12)
        grad=torch.autograd.grad(losses['task'],static.Wc,retain_graph=True)[0]
        separate=torch.autograd.grad(expected,static.Wc,retain_graph=True)[0]
        self.assertTrue(torch.allclose(grad,separate,atol=1e-12,rtol=1e-12))
        losses['total'].backward()
        self.assertIsNone(static.Bvis.grad)
        self.assertGreater(static.Wc.grad.norm().item(),0.)
        logs=task_decomposition({k:v.detach().item() for k,v in losses.items()},.075)
        self.assertAlmostEqual(logs['support_weight'],9/86,places=14)

    def test_ranking_diagnostics_separate_attribute_and_extra_candidates(self):
        manifest=dict(labels=dict(attrs=['a','b'],objs=['x','y']),
            pairs=dict(train=[['a','x']],test=[['b','x']]))
        pairs=np.array([['a','x'],['a','y'],['b','x'],['b','y']])
        probabilities=np.array([[.7,.05,.2,.05],[.2,.1,.5,.2],[.22,.43,.3,.05]])
        data=dict(candidate_pairs=pairs,attribute_truth=np.array([0,1,1]),object_truth=np.array([0,0,0]),
            log_probabilities=np.log(probabilities),conditional_logits_gt_object=np.log(probabilities[:,[0,2]]))
        rows,pair_rows=diagnose(data,manifest,'test')
        row=next(r for r in rows if r['group']=='unseen')
        self.assertEqual(row['images'],2);self.assertEqual(row['extra_candidates'],2)
        self.assertEqual(row['conditional_accuracy'],100.);self.assertEqual(row['object_accuracy'],100.)
        self.assertEqual(row['unseen_rank_ceiling'],50.);self.assertEqual(row['extra_dominates_valid'],50.)
        self.assertAlmostEqual(row['extra_support_pct'],39.)
        self.assertAlmostEqual(row['seen_support_pct'],21.)
        self.assertAlmostEqual(row['conditional_nll'],-np.log([.5/.7,.3/.52]).mean())
        self.assertEqual(next(r for r in pair_rows if not r['seen'])['unseen_ceiling_correct'],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
