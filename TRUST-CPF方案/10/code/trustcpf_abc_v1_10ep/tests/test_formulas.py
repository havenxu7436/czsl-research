import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from trustcpf.model import TrustCPF, candidate_logprob, average_graphs, task_loss
from trustcpf.graph import assemble_bank, solve_budget, permute_bank
from trustcpf.utility import paired_utility, utility_nll, aggregate_moments, regression_metrics
from trustcpf.common import unit, write_json, save_torch
from trustcpf.folds import make_folds

torch.set_num_threads(1)


def fixture(dim=12,na=4,no=4):
    torch.manual_seed(12)
    text={'ea':unit(torch.randn(na,dim)),'eo':unit(torch.randn(no,dim)),
          'ec':unit(torch.randn(no,na,dim))}
    rows=[{'type':ty,'target':((i%no)+na if ty<2 else i%na),
           'slot':str(i%2),'owner':str(i),'text':'sample'} for ty in range(3) for i in range(8)]
    bank=assemble_bank(text,unit(torch.randn(len(rows),dim)),rows)
    cfg={'seed':3351,'rank':5,'message_rank':3,'adapter_rank':3,'synthetic':True,'backbone_rank':5,'backbone_adapter_rank':3}
    return text,bank,cfg


class FormulaTests(unittest.TestCase):
    def test_complete_task_logit_gradients(self):
        from trustcpf.model import task_logit_gradients
        obj=torch.randn(3,4,dtype=torch.double,requires_grad=True)
        cond=torch.randn(3,4,4,dtype=torch.double,requires_grad=True)
        a=torch.tensor([0,3,2]);o=torch.tensor([0,1,2]);c=torch.arange(12)
        lo,lc=obj.log_softmax(-1),cond.log_softmax(-1)
        value=task_loss(lo[:,:,None]+lc,lo,lc,a,o,c)
        got=torch.autograd.grad(value,[obj,cond]);expected=task_logit_gradients(obj,cond,a,o,c)
        self.assertTrue(all(torch.allclose(x,y,atol=1e-12) for x,y in zip(got,expected)))
    def test_zero_graph_and_probability(self):
        text,bank,cfg=fixture();model=TrustCPF(text,bank,cfg).double()
        batch=[torch.randn(3,12,dtype=torch.double),torch.randn(3,4,12,dtype=torch.double),torch.randn(3,4,12,dtype=torch.double)]
        base=model.base(*batch); edges=torch.zeros(24,dtype=torch.double)
        j,lo,lc=model.score(base,edges)
        self.assertTrue(torch.allclose(lo,base['obj'].log_softmax(-1),atol=1e-12))
        self.assertTrue(torch.allclose(lc,base['cond'].log_softmax(-1),atol=1e-12))
        self.assertTrue(torch.allclose(j.exp().sum((1,2)),torch.ones(3,dtype=torch.double),atol=1e-12))
        self.assertTrue(torch.allclose(j.exp().sum(2),lo.exp(),atol=1e-12))

    def test_budget_and_derivative(self):
        r=torch.tensor([.1,.2,-.3,.4,.1,.2],dtype=torch.double);ty=torch.tensor([0,0,1,1,2,2])
        pi,g=solve_budget(r,ty,.02,.5)
        for t in range(3):
            mask=ty==t
            self.assertLessEqual(float(pi[mask].sum()),1+1e-10)
            residual=-r[mask]+.02*torch.logit(pi[mask])+g[str(t)]
            self.assertLess(float(residual.abs().max()),1e-6)
        p,_=solve_budget(r,ty,.02,0);self.assertEqual(float(p.sum()),0)

    def test_heteroscedastic_gradient(self):
        m=torch.tensor([.2,-.1],dtype=torch.double,requires_grad=True)
        v=torch.tensor([.4,.2],dtype=torch.double,requires_grad=True)
        y=torch.tensor([.1,.2],dtype=torch.double);nu=torch.tensor([.03,.04],dtype=torch.double)
        loss=utility_nll(m,v,y,nu,torch.ones(2,dtype=torch.double),{'sampled':2,'total':2})
        gm,gv=torch.autograd.grad(loss,[m,v])
        self.assertTrue(torch.allclose(gm,(m-y)/(v+nu)/2))
        self.assertTrue(torch.allclose(gv,(.5/(v+nu)-.5*(m-y)**2/(v+nu)**2)/2))

    def test_pair_variance_and_gradient_identity(self):
        text,bank,cfg=fixture();model=TrustCPF(text,bank,cfg).double().eval()
        batch=[torch.randn(2,12,dtype=torch.double),torch.randn(2,4,12,dtype=torch.double),torch.randn(2,4,12,dtype=torch.double)]
        base=model.base(*batch);candidates=torch.arange(16);y=torch.tensor([1,5]);w=torch.full((2,),.5,dtype=torch.double)
        result,noise,diff=paired_utility(model,base,y,candidates,torch.tensor([0]),w,4,torch.Generator().manual_seed(5))
        self.assertTrue(torch.allclose(result,diff.mean(1)))
        self.assertTrue(torch.allclose(noise,diff.var(1,unbiased=True)/4))
        b=torch.ones(24,dtype=torch.double);b[0]=0
        minus=-candidate_logprob(model.score(base,b)[0],candidates)[torch.arange(2),y].mean()
        b[0]=1
        plus=-candidate_logprob(model.score(base,b)[0],candidates)[torch.arange(2),y].mean()
        p=torch.tensor(.3,dtype=torch.double,requires_grad=True)
        expected=p*plus+(1-p)*minus
        self.assertTrue(torch.allclose(torch.autograd.grad(expected,p)[0],-(minus-plus)))

    def test_average_graphs_and_jensen(self):
        text,bank,cfg=fixture();model=TrustCPF(text,bank,cfg).eval()
        batch=[torch.randn(3,12),torch.randn(3,4,12),torch.randn(3,4,12)]
        lp,var=average_graphs(model,batch,torch.full((24,),.5),torch.arange(16),4,123)
        self.assertTrue(torch.allclose(lp.exp().sum(1),torch.ones(3),atol=1e-6))
        self.assertTrue((var>=0).all())
        lp2,_=average_graphs(model,batch,torch.full((24,),.5),torch.arange(16),4,123)
        self.assertTrue(torch.equal(lp,lp2))

    def test_total_variance_and_metrics(self):
        m=torch.tensor([[1.,2.],[3.,4.]])
        mm,v,e=aggregate_moments(m,torch.ones_like(m)*2,torch.ones_like(m)*.2)
        self.assertTrue(torch.equal(mm,torch.tensor([2.,3.])))
        self.assertTrue(torch.equal(v,torch.tensor([3.,3.])))
        result=regression_metrics([1,-1,2,-2],[.2]*4,[1,-1,2,-2],[.01]*4)
        self.assertEqual(result['spearman'],1);self.assertEqual(result['direction_auroc'],1)

    def test_fold_coverage_and_noise(self):
        pairs=[(str(a),str(o)) for a in range(4) for o in range(4)]
        folds=make_folds(pairs)
        allq=[tuple(p) for q in folds['query'] for p in q]
        self.assertEqual(len(allq),len(set(allq)))
        self.assertEqual(set(allq),set(pairs))
        text,bank,cfg=fixture(); changed=permute_bank(bank,7101,.5,2)
        ids=torch.where(bank['types']==2)[0]
        self.assertEqual(int(((changed['source'][ids]-bank['source'][ids]).norm(dim=1)>0).sum()),4)

    def test_unmeasured_record_filter_and_semantic_features(self):
        from trustcpf.stages import heldout_edges,remove_heldout,feature_mode
        text,bank,cfg=fixture();held=heldout_edges(bank,{'heldout_edge_fraction':.25})
        self.assertEqual(len(held),6)
        data={'edge':torch.arange(24),'y':torch.arange(24.),'sampled':6,'total':24}
        filtered=remove_heldout(data,held)
        self.assertEqual(len(filtered['y']),18)
        self.assertFalse(torch.isin(filtered['edge'],torch.tensor(held)).any())
        x=feature_mode(torch.ones(2,8*12+13),{'utility_input':'semantic'},12)
        self.assertEqual(float(x[:,55:].sum()),0)

    def test_gradients_and_singleton_batch(self):
        text,bank,cfg=fixture();model=TrustCPF(text,bank,cfg)
        batch=[torch.randn(1,12),torch.randn(1,4,12),torch.randn(1,4,12)]
        j,lo,lc=model(*batch,torch.ones(24))
        loss=task_loss(j,lo,lc,torch.tensor([1]),torch.tensor([2]),torch.arange(16))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertGreater(float(model.graph.right.grad.norm()),0)
        self.assertGreater(float(model.pv.weight.grad.norm()),0)

    def test_native_clip_visual_interface(self):
        self.skipTest('This experiment reuses verified G0 features; CLIP extraction is unchanged.')
        import importlib.util
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('clip_model_test',root/'vendor/clip/model.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from trustcpf.features import visual_tokens,query_patches
        visual=module.VisionTransformer(224,14,64,2,4,32).eval()
        image=torch.randn(2,3,224,224)
        with torch.no_grad():
            g,p=visual_tokens(visual,image)
            self.assertTrue(torch.allclose(g,unit(visual(image)),atol=2e-6))
            self.assertEqual(p.shape,(2,256,32))
            f=query_patches(p,unit(torch.randn(7,32)),.07)
            self.assertTrue(torch.allclose(f.norm(dim=-1),torch.ones(2,7),atol=2e-6))

    def test_exact_curve_matches_working_points(self):
        from trustcpf.metrics import exact_curve
        x=np.array([[2.,1.,.3],[1.,2.,.7],[1.5,.2,2.],[2.,.1,1.]])
        y=np.array([0,1,2,2]);seen=np.array([True,True,False])
        curve=exact_curve(x,y,seen)
        for b,s,u in zip(curve['biases'],curve['seen'],curve['unseen']):
            z=x.copy();z[:,~seen]+=b;ok=z.argmax(1)==y
            self.assertAlmostEqual(s,ok[seen[y]].mean()*100)
            self.assertAlmostEqual(u,ok[~seen[y]].mean()*100)


def synthetic_cache(root):
    text,bank,cfg=fixture(dim=12)
    labels={'attrs':[str(i) for i in range(4)],'objs':[str(i) for i in range(4)]}
    train=[(str(a),str(o)) for a in range(4) for o in range(4) if a!=o]
    unseen=[(str(i),str(i)) for i in range(4)]
    pairs={'train':train,'val':train[:4]+unseen,'test':train[4:8]+unseen}
    for phase,ps in pairs.items():
        rows=[]
        for a,o in ps:
            for k in range(3):rows.append({'image':'%s_%s_%s_%s'%(phase,a,o,k),'attr':int(a),'object':int(o)})
        write_json(root/(phase+'_records.json'),rows)
        (root/phase).mkdir()
        n=len(rows)
        for name,shape in [('g',(n,12)),('fa',(n,4,12)),('fo',(n,4,12))]:
            np.save(root/phase/(name+'.npy'),np.random.RandomState(8).normal(size=shape).astype(np.float16))
    np.savez(root/'text_features.npz',**{k:v.numpy() for k,v in text.items()})
    write_json(root/'features_manifest.json',{'complete':True,'labels':labels,'pairs':pairs,'files_sha256':{}})
    save_torch(root/'graph.pt',bank)


if __name__=='__main__':unittest.main(verbosity=2)
