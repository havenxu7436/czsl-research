import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from ckcpf.common import load_torch, read_json, write_json, sha
from ckcpf.model import ConditionalCPF
from ckcpf import runtime
from ckcpf.evaluation import evaluate
from ckcpf.data import train_mask
from reference_g0 import VGCPOD

torch.set_num_threads(2)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.initial = load_torch(ROOT/'initial/g0_seed3351.pt')
        self.bank = load_torch(ROOT/'assets/knowledge.pt')
        self.cfg = read_json(ROOT/'configs/default.json')
        self.g = torch.nn.functional.normalize(torch.randn(6,768,generator=torch.Generator().manual_seed(51)),dim=-1)

    def model(self, variant):
        return ConditionalCPF(self.initial,self.bank,variant,self.cfg,3351)

    def test_g0_and_zero_alpha_agree_for_all_seeds(self):
        for seed in runtime.SEEDS:
            initial=load_torch(ROOT/f'initial/g0_seed{seed}.pt')
            old=VGCPOD(initial['model']['ea'],initial['model']['eo'],initial['model']['ec'],initial['config'],seed)
            old.load_state_dict(initial['model'])
            expected=old(self.g,None,None)
            for v in runtime.VARIANTS:
                model=ConditionalCPF(initial,self.bank,v,self.cfg,seed)
                actual=model(self.g)
                for key in ['obj','cond','joint']:
                    torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
                a=torch.tensor([0,1,2,3,4,5]);o=torch.tensor([0,1,2,3,4,5])
                mask=train_mask(read_json(ROOT/'assets/expected_cache.json'))
                torch.testing.assert_close(model.loss(actual,a,o,mask),old.objective(expected,a,o,mask)['total'])

    def test_graph_residual_matches_direct_formula_and_frozen_object(self):
        model=self.model('knowledge'); reference=model(self.g)['obj'].detach().clone()
        with torch.no_grad(): model.alpha.fill_(.7);model.right.normal_(0,.001)
        z=model.prototypes();na=len(model.ea)
        g=self.g;u=z[:na]@model.Au.T;v=z[na:]@model.Av.T
        direct=model.rho*model.kappa*torch.einsum('br,ar,or->boa',g@model.Wc,u,v)
        out=model(g);base=self.model('base')(g)
        torch.testing.assert_close(out['cond'],base['cond']+.7*(direct-model.cp(g,model.ea,model.eo)),atol=2e-5,rtol=1e-5)
        mask=train_mask(read_json(ROOT/'assets/expected_cache.json'))
        loss=model.loss(out,torch.arange(6),torch.arange(6),mask);loss.backward()
        self.assertIsNone(model.Bo.grad);self.assertIsNone(model.Co.grad)
        self.assertIsNone(model.Au.grad);self.assertIsNone(model.Av.grad)
        self.assertGreater(float(model.right.grad.abs().sum()),0.)
        model.optimizer().step()
        torch.testing.assert_close(model(g)['obj'],reference,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),110593)
        self.assertEqual(sum(p.numel() for p in self.model('base').parameters() if p.requires_grad),36864)

    def test_shuffle_preserves_multiset_and_targets(self):
        sh=load_torch(ROOT/'assets/shuffle.pt')
        self.assertTrue(torch.equal(self.bank['target'],sh['target']))
        self.assertTrue(torch.equal(self.bank['degree'],sh['degree']))
        for t in range(3):
            ids=self.bank['types']==t
            left=sorted(x.numpy().tobytes() for x in self.bank['source'][ids])
            right=sorted(x.numpy().tobytes() for x in sh['source'][ids])
            self.assertEqual(left,right)

    def test_final_evaluation_uses_supplied_val_calibration(self):
        manifest=read_json(ROOT/'assets/expected_cache.json')
        labels=manifest['labels']; train=set(map(tuple,manifest['pairs']['train']))
        pairs=list(train)[:4]+[tuple(p) for p in manifest['pairs']['test'] if tuple(p) not in train][:4]
        rng=np.random.RandomState(3);cond=rng.normal(size=(len(pairs),12,16)).astype('float32');obj=rng.normal(size=(len(pairs),12)).astype('float32')
        pred=dict(cond=cond,obj=obj,joint=cond,attrs=np.array([labels['attrs'].index(a) for a,o in pairs]),
                  objs=np.array([labels['objs'].index(o) for a,o in pairs]),delta_abs=np.zeros(len(pairs)))
        cal={w:dict(calibrated=dict(bias=1.234,temperature=.73)) for w in ('cw','ow')}
        result,_=evaluate(pred,manifest,'test',cal)
        for w in ('cw','ow'):
            self.assertEqual(result[w]['calibrated']['bias'],1.234)
            self.assertEqual(result[w]['calibrated']['temperature'],.73)

    def test_training_resume_and_complete_report(self):
        # Synthetic features test the pipeline, not CZSL performance.
        with tempfile.TemporaryDirectory(prefix='pipeline_',dir=ROOT/'validation') as folder:
            root=Path(folder).resolve();self.assertEqual(root.parent,(ROOT/'validation').resolve())
            cache=root/'cache';cache.mkdir()
            manifest=copy.deepcopy(read_json(ROOT/'assets/expected_cache.json'))
            labels=manifest['labels'];train=list(map(tuple,manifest['pairs']['train']))
            for phase in ('train','val','test'):
                pairs=train[:8] if phase=='train' else train[:4]+[tuple(p) for p in manifest['pairs'][phase] if tuple(p) not in set(train)][:4]
                rows=[dict(image=f'{phase}/{i}.jpg',attr=labels['attrs'].index(a),object=labels['objs'].index(o)) for i,(a,o) in enumerate(pairs)]
                (cache/phase).mkdir();np.save(cache/phase/'g.npy',self.g.numpy()[np.arange(len(rows))%6])
                write_json(cache/(phase+'_records.json'),rows)
                manifest['files_sha256'][phase+'/g.npy']=sha(cache/phase/'g.npy')
                manifest['files_sha256'][phase+'_records.json']=sha(cache/(phase+'_records.json'))
            cfg=dict(self.cfg,epochs=2,batch_size=4,eval_batch_size=8)
            args=SimpleNamespace(cache=str(cache),work=str(root/'work'),seed=3351,device='cpu')
            original_save=runtime.save_torch
            def stop_after_first(path,value):
                original_save(path,value)
                if Path(path).name=='last.pt' and value['epoch']==1: raise RuntimeError('test interruption')
            with patch.object(runtime,'save_torch',side_effect=stop_after_first):
                with self.assertRaisesRegex(RuntimeError,'test interruption'):
                    runtime.train_one(args,manifest,cfg,'base')
            runtime.train_one(args,manifest,cfg,'base')
            first=load_torch(root/'work/seed3351/base/last.pt')
            args.work=str(root/'uninterrupted')
            runtime.train_one(args,manifest,cfg,'base')
            second=load_torch(root/'uninterrupted/seed3351/base/last.pt')
            for key in first['model']:
                torch.testing.assert_close(first['model'][key],second['model'][key],rtol=0,atol=0)
            args.work=str(root/'work')
            for seed in runtime.SEEDS:
                args.seed=seed
                for variant in runtime.VARIANTS:
                    runtime.train_one(args,manifest,cfg,variant)
            status=runtime.report(args)
            self.assertEqual(status['completed'],9)
            self.assertTrue((root/'work/seed3353/shuffle/test_predictions.npz').exists())
            compare=read_json(root/'work/paired_comparisons.json')
            self.assertEqual(len(compare['knowledge_minus_base']['cw']['differences']),3)


if __name__=='__main__':unittest.main()
