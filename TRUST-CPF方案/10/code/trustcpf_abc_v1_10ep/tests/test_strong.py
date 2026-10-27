import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from trustcpf.common import ROOT,load_torch,read_json,write_json,save_torch
from trustcpf.model import TrustCPF
from trustcpf.engine import new_model
from test_formulas import fixture,synthetic_cache


class StrongBackboneTests(unittest.TestCase):
    def test_actual_three_g0_checkpoints_exact_zero_graph(self):
        bank=load_torch(ROOT/'assets/knowledge.pt')
        for seed in (3351,3352,3353):
            initial=load_torch(ROOT/'initial'/('g0_seed%d.pt'%seed))
            text={k:initial['model'][k] for k in ('ea','eo','ec')}
            cfg=dict(read_json(ROOT/'configs/default.json'),seed=seed)
            model=new_model(text,bank,cfg,'cpu')
            g=torch.nn.functional.normalize(torch.randn(5,768),dim=-1)
            ref=model.backbone(g,g,g)
            actual=model(g,g,g[:,None,:].expand(-1,12,-1),torch.zeros(96))[0]
            torch.testing.assert_close(actual,ref['joint'],rtol=0,atol=0)
            self.assertGreater(float(model.backbone.Wc.norm()),0)
            self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
            for k,v in model.backbone.state_dict().items():torch.testing.assert_close(v,initial['model'][k],rtol=0,atol=0)

    def test_graph_training_keeps_every_backbone_tensor(self):
        text,bank,cfg=fixture();model=TrustCPF(text,bank,cfg)
        before=copy.deepcopy(model.backbone.state_dict())
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        g=torch.randn(3,12);out=model(g,g,g[:,None,:].expand(-1,4,-1),torch.ones(24))[0]
        loss=-out[:,0,0].mean();opt.zero_grad();loss.backward();opt.step()
        for k,v in before.items():torch.testing.assert_close(v,model.backbone.state_dict()[k],rtol=0,atol=0)
        self.assertGreater(float(model.graph.right.detach().norm()),0)

    def test_fold_backbone_only_uses_inner_pairs(self):
        from trustcpf.engine import setup,teachers
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cache=root/'cache';cache.mkdir();synthetic_cache(cache)
            cfg=dict(read_json(ROOT/'configs/default.json'),synthetic=True,backbone_epochs=1,teacher_epochs=1,
                backbone_rank=4,backbone_adapter_rank=2,rank=4,message_rank=2,batch_size=16)
            write_json(root/'cfg.json',cfg)
            args=SimpleNamespace(cache=str(cache),work=str(root/'work'),config=str(root/'cfg.json'),seed=3351,
                variant=None,quick=False,fold='1',device='cpu')
            teachers(args)
            state=load_torch(root/'work/teachers/1/backbone/last.pt')
            folds=read_json(root/'work/folds.json')
            self.assertFalse(set(map(tuple,state['train_pairs']))&set(map(tuple,folds['query'][0])))
            self.assertEqual(state['initialization'],'fresh_G0_architecture')
            actual=load_torch(root/'work/teachers/1/last.pt')['model']
            for k,v in state['model'].items():torch.testing.assert_close(actual['backbone.'+k],v,rtol=0,atol=0)

    def test_evaluation_outputs_are_replayable(self):
        from trustcpf.engine import setup,evaluate_model
        from trustcpf.metrics import native_curve
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cache=root/'cache';cache.mkdir();synthetic_cache(cache)
            cfg=dict(read_json(ROOT/'configs/default.json'),synthetic=True,backbone_rank=4,backbone_adapter_rank=2,
                     rank=4,message_rank=2,graphs=2)
            write_json(root/'cfg.json',cfg)
            args=SimpleNamespace(cache=str(cache),work=str(root/'work'),config=str(root/'cfg.json'),seed=3351,variant=None,quick=False)
            cfg,m,text,bank,_,work=setup(args);model=new_model(text,bank,cfg,'cpu')
            result=evaluate_model(model,cache,cfg,m,torch.ones(24)*.5,'val',save=work)
            with np.load(work/'val_predictions.npz') as z:
                for world in ('cw','ow'):
                    got=native_curve(z[world+'_logprob'],z[world+'_truth'],z[world+'_seen'])
                    self.assertAlmostEqual(got['AUC'],result[world]['AUC_native'])

if __name__=='__main__':unittest.main()
