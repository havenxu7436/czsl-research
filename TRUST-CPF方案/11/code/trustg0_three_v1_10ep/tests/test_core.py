import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from trustg0.common import ROOT,read,write,save,load,seed_all,fingerprint
from trustg0.model import TrustG0,unit,select_pairs,parameters_state,restore_parameters
from trustg0.data import teacher_folds,context_groups
from trustg0.response import InterventionBatch,query_partition
from trustg0.evaluate import masks,scores
from trustg0 import utility,train,workflow
from trustg0.cache import FeatureCache


def fixture(root,teachers=1):
    seed_all(91);torch.set_num_threads(1);d=12;na=5;no=4
    cfg=read(ROOT/'configs/experiment.json');cfg.update(datasets=['toy'],seeds=[5351],teacher_initializations=teachers,
        graph_rank=6,message_rank=3,batch_size=64,eval_batch=64,graphs=2,eval_banks=[1701],permutation_seeds=[7101],
        response_backgrounds=2,measure_batch=64,response_pair_limit=256,query_pair_limit=128,workers=0)
    cfg['g0'].update(rank=4,adapter_rank=3)
    text=dict(ea=unit(torch.randn(na,d)),eo=unit(torch.randn(no,d)),ec=unit(torch.randn(no,na,d)))
    target=torch.tensor(list(range(na,na+no))*2+list(range(na)));types=torch.tensor([0]*no+[1]*no+[2]*na)
    bank=dict(anchors=torch.cat((text['ea'],text['eo'])),source=unit(torch.randn(len(target),d)),target=target,types=types,degree=torch.ones(len(target)))
    pairs=dict(train=[p for p in range(20) if p not in (0,5,10,15)],val=[0,5],test=[10,15]);rows=[]
    for phase,ps in [('train',pairs['train']),('val',[1,2,0,5]),('test',[1,2,10,15])]:
        for p in ps:
            for n in range(6 if phase=='train' else 2):rows.append(dict(image=str(len(rows)),attr=p//no,obj=p%no,pair=p,phase=phase))
    index=dict(name='toy',attrs=list(map(str,range(na))),objs=list(map(str,range(no))),pairs=pairs,rows=rows,images={s:sum(r['phase']==s for r in rows) for s in pairs})
    ds=Path(root)/'toy';folder=ds/'cache';folder.mkdir(parents=True)
    for k,shape in [('g',(len(rows),d)),('patch',(len(rows),8,d))]:np.save(folder/(k+'.npy'),unit(torch.randn(*shape)).numpy().astype(np.float16))
    save(folder/'text.pt',text);save(folder/'bank.pt',bank);write(folder/'complete.json',dict(complete=True))
    groups=context_groups(text['eo'].numpy());write(ds/'groups.json',groups);write(ds/'index.json',index)
    write(ds/'seed5351/folds.json',teacher_folds(index,6342));write(Path(root)/'identity.json',dict(config=cfg,code=fingerprint(),datasets=['toy']))
    return cfg,index,bank,groups,FeatureCache(ds,'cpu')


class Core(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.cfg,self.index,self.bank,self.groups,self.cache=fixture(self.temp.name)
        self.addCleanup(lambda:self.cache.patch._mmap.close())
        self.addCleanup(lambda:self.cache.g._mmap.close())

    def test_graph_aggregation_and_gradient(self):
        m=TrustG0(self.cache.text,self.bank,self.cfg,8);edges=torch.rand(len(self.bank['target']))
        m.graph.right.data.normal_(0,.1)
        msg=m.graph.messages();inc=torch.nn.functional.one_hot(self.bank['target'],len(self.bank['anchors'])).T.float()
        diff=m.graph.source-m.graph.anchors[m.graph.target]
        legacy=diff+torch.einsum('edr,er->ed',m.graph.left[m.graph.types],torch.einsum('erd,ed->er',m.graph.right[m.graph.types],diff))
        torch.testing.assert_close(msg,legacy)
        expected=self.bank['anchors']+self.cfg['eta']*(inc@(msg*(edges/self.bank['degree'])[:,None]))
        torch.testing.assert_close(m.graph.raw_nodes(edges),expected)
        grads=torch.autograd.grad(m.graph.raw_nodes(edges).square().sum(),list(m.graph.parameters()),retain_graph=True)
        exact=torch.autograd.grad(expected.square().sum(),list(m.graph.parameters()))
        for a,b in zip(grads,exact):torch.testing.assert_close(a,b)

    def test_zero_graph_recovers_g0_and_pair_axes(self):
        m=TrustG0(self.cache.text,self.bank,self.cfg,8);g,fo=self.cache.batch([0,1,9]);base=m.base(g,fo)
        actual=m.score(base,torch.zeros(len(self.bank['types'])))[0];expected=m.backbone(g,None,None)['joint']
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        p=torch.tensor([0,3,4,15,19]);torch.testing.assert_close(select_pairs(actual,p,4),torch.stack([actual[:,int(x)%4,int(x)//4] for x in p],1))

    def test_fast_interventions_match_brute_force(self):
        m=TrustG0(self.cache.text,self.bank,self.cfg,8);m.graph.right.data.normal_(0,.1)
        g,fo=self.cache.batch([0,1,9,23]);base=m.base(g,fo);pairs=torch.tensor(self.index['pairs']['train']);truth=torch.tensor([self.index['rows'][i]['pair'] for i in [0,1,9,23]])
        lookup={p:i for i,p in enumerate(pairs.tolist())};iy=torch.tensor([lookup[int(p)] for p in truth]);rows=torch.arange(len(g))
        for bg in [torch.zeros(len(self.bank['types'])),torch.ones(len(self.bank['types'])),torch.rand(len(self.bank['types'])).round()]:
            fast=InterventionBatch(m,base,bg,pairs,truth)
            for e in range(len(bg)):
                logs=[];joints=[]
                for v in (0.,1.):
                    edges=bg.clone();edges[e]=v;joint=select_pairs(m.score(base,edges)[0],pairs,m.no).double();joints.append(joint);logs.append(joint.log_softmax(-1))
                diff=joints[1]-joints[0];entropy=[-(l.exp()*l).sum(-1) for l in logs]
                expected=(logs[1][rows,iy]-logs[0][rows,iy],(diff-diff.mean(-1,keepdim=True)).square().mean(-1),entropy[1]-entropy[0])
                for a,b in zip(fast.effect(e),expected):torch.testing.assert_close(a,b,atol=3e-6,rtol=2e-3)

    def test_fold_roles_and_image_exclusion(self):
        folds=teacher_folds(self.index,6342);queries=[]
        for f in folds:
            queries+=f['query_pairs'];train=set(f['train_ids']);q=set(f['query_ids']);seen=set(f['audit_seen_ids'])
            self.assertFalse(train&q or train&seen or q&seen)
            measure,audit=query_partition(self.index,f['query_ids'],77);self.assertFalse(set(measure)&set(audit));self.assertEqual(set(measure)|set(audit),q)
        self.assertEqual(len(queries),len(set(queries)))

    def test_policy_budget_and_crn(self):
        types=self.bank['types'].numpy();mu=np.random.RandomState(71).randn(2,len(types));pi=utility.policy(mu,types)
        for t in range(3):np.testing.assert_allclose(pi[:,types==t].sum(1),sum(types==t)/2,atol=1e-10)
        self.assertTrue(torch.equal(masks(pi,16,1701,'cpu'),masks(pi,16,1701,'cpu')))
        uniform=utility.policy(np.zeros_like(mu),types);torch.testing.assert_close(uniform,torch.full_like(uniform,.5),atol=1e-12,rtol=0)

    def test_epoch_resume_is_exact(self):
        split=workflow.formal_split(self.index);a=Path(self.temp.name)/'full';b=Path(self.temp.name)/'resume';original=train.save
        train.run(self.cache,self.index,self.bank,self.groups,self.cfg,88,split,a,'g0')
        def interrupt(path,obj):
            original(path,obj)
            if Path(path).name=='last.pt' and obj['epoch']==4:raise InterruptedError('test interruption')
        with patch.object(train,'save',interrupt):
            with self.assertRaises(InterruptedError):train.run(self.cache,self.index,self.bank,self.groups,self.cfg,88,split,b,'g0')
        train.run(self.cache,self.index,self.bank,self.groups,self.cfg,88,split,b,'g0')
        for k,v in load(a/'best.pt')['params'].items():self.assertTrue(torch.equal(v,load(b/'best.pt')['params'][k]),k)
        self.assertEqual(read(b/'complete.json')['epochs'],10)

    def test_stochastic_graph_resume_is_exact(self):
        split=workflow.formal_split(self.index);root=Path(self.temp.name);original=train.save
        train.run(self.cache,self.index,self.bank,self.groups,self.cfg,83,split,root/'g0','g0')
        def go(path):train.run(self.cache,self.index,self.bank,self.groups,self.cfg,84,split,path,'uniform',root/'g0/best.pt')
        go(root/'graph_full')
        def interrupt(path,obj):
            original(path,obj)
            if Path(path).name=='last.pt' and obj['epoch']==4:raise InterruptedError('test interruption')
        with patch.object(train,'save',interrupt):
            with self.assertRaises(InterruptedError):go(root/'graph_resume')
        go(root/'graph_resume')
        for k,v in load(root/'graph_full/best.pt')['params'].items():self.assertTrue(torch.equal(v,load(root/'graph_resume/best.pt')['params'][k]),k)


if __name__=='__main__':unittest.main()
