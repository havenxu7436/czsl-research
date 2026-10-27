import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from test_formulas import fixture
from trustcpf.model import TrustCPF
from trustcpf.response import response_statistics,select_support,measure_responses
from trustcpf.graph import solve_budget_equal
from trustcpf.common import save_torch,write_json
from trustcpf.stages import combine_measurements

class ResponseTests(unittest.TestCase):
    def test_support_composition_weights_and_exclusion(self):
        labels=dict(attrs=['a','b'],objs=['x','y'])
        rows=[dict(attr=0,object=0)]*5+[dict(attr=1,object=0)]*2+[dict(attr=1,object=1)]*4
        ids,w=select_support(rows,[['a','x'],['b','x']],labels,2,3)
        self.assertEqual(len(ids),4);self.assertTrue(all(i<7 for i in ids))
        self.assertAlmostEqual(float(w.sum()),1.)
        self.assertAlmostEqual(float(w[[i<5 for i in ids]].sum()),.5)
        self.assertEqual(ids,select_support(rows,[['a','x'],['b','x']],labels,2,3)[0])

    def test_exact_utility_sign_weights_and_shift_invariance(self):
        torch.manual_seed(2)
        minus=torch.randn(3,5,7,dtype=torch.double);plus=minus+.2*torch.randn_like(minus)
        y=torch.tensor([1,2,0,5,6]);w=torch.tensor([.1,.2,.15,.15,.4]);mask=torch.tensor([1,1,0,0,0]).bool()
        f=response_statistics(minus,plus,y,w,mask)
        d=plus-minus;lp=minus.log_softmax(-1)
        effect=d.gather(-1,y[None,:,None].expand(3,-1,1)).squeeze(-1)-(lp+d).logsumexp(-1)
        expected=(effect*w.double()/w.sum()/3).sum()
        self.assertAlmostEqual(float(f[0]),float(expected),places=7)
        centered=d-d.mean(-1,keepdim=True)
        self.assertAlmostEqual(float(f[2]),float((centered.square().mean(-1)*w.double()/w.sum()/3).sum().sqrt()),places=7)
        shift=torch.randn(3,5,1,dtype=torch.double)
        torch.testing.assert_close(f,response_statistics(minus,plus+shift,y,w,mask),rtol=1e-5,atol=1e-7)
        zero=response_statistics(minus,minus,y,w,mask);self.assertEqual(float(zero.abs().max()),0.)

    def test_equality_kkt_negative_multiplier_and_counts(self):
        r=torch.tensor([-.03,-.02,-.01,.01],dtype=torch.double);ty=torch.tensor([0,0,1,1])
        pi,g=solve_budget_equal(r,ty,.01,{'0':1.5,'1':.8})
        self.assertLess(g['0'],0.)
        for t in [0,1]:
            self.assertAlmostEqual(float(pi[ty==t].sum()),[1.5,.8][t],places=12)
            self.assertLess(float((r[ty==t]-.01*torch.logit(pi[ty==t])-g[str(t)]).abs().max()),1e-12)

    def test_responses_depend_on_current_model_and_batching(self):
        text,bank,cfg=fixture();model=TrustCPF(text,bank,cfg).eval()
        g=torch.randn(8,12);fo=torch.randn(8,4,12);attrs=torch.arange(8)%4;objs=torch.arange(8)%4
        inputs=[[g,g,fo]];truth=objs*4+attrs;w=torch.ones(8)/8
        f,_=measure_responses(model,inputs,truth,w,attrs,objs,torch.arange(16),bank,2,31)
        split=[[g[:3],g[:3],fo[:3]],[g[3:],g[3:],fo[3:]]]
        f2,_=measure_responses(model,split,truth,w,attrs,objs,torch.arange(16),bank,2,31)
        torch.testing.assert_close(f,f2,rtol=2e-4,atol=2e-6)
        with torch.no_grad():model.pv.weight.add_(torch.randn_like(model.pv.weight)*.3)
        f3,_=measure_responses(model,inputs,truth,w,attrs,objs,torch.arange(16),bank,2,31)
        self.assertGreater(float((f3-f).abs().max()),1e-5)

    def test_fold_specific_join_leaves_query_targets_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);source=p/'source';work=p/'new'
            write_json(work/'reuse.json',dict(source_seed=str(source)))
            for j in [1,5]:
                d=dict(x=torch.ones(3,4).half(),y=torch.tensor([.2,-.1,.3]),noise=torch.ones(3)*.001,
                       edge=torch.tensor([2,0,1]),inclusion=torch.ones(3),sampled=3,total=3)
                save_torch(source/'measurements'/('fold%d.pt'%j),d)
                save_torch(work/'responses'/('fold%d.pt'%j),dict(features=torch.ones(3,8)*j))
            joined=combine_measurements(work,[1,5],True)
            torch.testing.assert_close(joined['y'],d['y'].repeat(2))
            self.assertEqual(joined['x'].shape,(6,12))
            self.assertTrue(torch.all(joined['x'][:3,-8:]==1));self.assertTrue(torch.all(joined['x'][3:,-8:]==5))

if __name__=='__main__':unittest.main()
