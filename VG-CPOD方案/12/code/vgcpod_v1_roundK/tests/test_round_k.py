"""Dynamic learning rates, unchanged initialization, Adam steps and resume."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from tests.test_round_j import fixture
from tests.test_math import synthetic_cache
from vgcpod import engine
from vgcpod.common import ROOT, load_torch, read_json, sha
from vgcpod.gate_features import prepare_gate_statistics
from readout_diagnostics import checkpoint_readouts


class RoundKTests(unittest.TestCase):
    def test_configs_change_only_wd_learning_rate_and_experiment_id(self):
        reference = read_json(ROOT/'configs/round_j/J1.json')
        for case, multiplier in [('K1',.3),('K2',.1)]:
            cfg = read_json(ROOT/'configs/round_k'/(case+'.json'))
            self.assertEqual(set(cfg)-set(reference), {'dynamic_readout_lr_multiplier'})
            self.assertEqual({k for k in reference if cfg[k]!=reference[k]}, {'experiment_id'})
            self.assertEqual(cfg['dynamic_readout_lr_multiplier'],multiplier)

    def test_groups_partition_parameters_and_cosine_endpoints(self):
        _, model, cfg, _, _, _ = fixture()
        for multiplier in (.3,.1):
            cfg = dict(cfg,dynamic_readout_lr_multiplier=multiplier)
            optimizer = engine.make_optimizer(model,cfg)
            groups = {g['name']:g for g in optimizer.param_groups}
            ids = [id(p) for g in groups.values() for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
            self.assertEqual([id(p) for p in groups['dynamic_cp']['params']],[id(model.Wd)])
            self.assertIn(id(model.Wc),[id(p) for p in groups['cp']['params']])
            for step, base in [(0,cfg['lr']),(100,cfg['min_lr'])]:
                engine.set_learning_rates(optimizer,cfg,step,101)
                self.assertAlmostEqual(groups['adapter']['lr'],base)
                self.assertAlmostEqual(groups['cp']['lr'],base*.3)
                self.assertAlmostEqual(groups['dynamic_cp']['lr'],base*.3*multiplier)

    def test_first_adam_step_scales_wd_not_its_gradient(self):
        _, reference, cfg, x, y, mask = fixture()
        cfg = dict(cfg,weight_decay=0)
        reference.objective(reference(*x),*y,mask)['total'].backward()
        expected_grad = reference.Wd.grad.clone()
        engine.make_optimizer(reference,cfg).step()
        for multiplier in (.3,.1):
            _, model, _, _, _, _ = fixture()
            # The fixture initializes all seeds identically; the new factor is optimizer-only.
            model.objective(model(*x),*y,mask)['total'].backward()
            torch.testing.assert_close(model.Wd.grad,expected_grad,rtol=0,atol=0)
            engine.make_optimizer(model,dict(cfg,dynamic_readout_lr_multiplier=multiplier)).step()
            torch.testing.assert_close(model.Wc,reference.Wc,rtol=0,atol=0)
            torch.testing.assert_close(model.Wd,reference.Wd*multiplier,rtol=1e-12,atol=1e-15)
            for name in ('Au','Av','Bvis','Bo','Co','Bc','Cc'):
                torch.testing.assert_close(getattr(model,name),getattr(reference,name),rtol=0,atol=0)

    def test_multiplier_one_reproduces_legacy_multiple_adam_steps(self):
        _, old, cfg, x, y, mask = fixture();new = copy.deepcopy(old)
        a,b = engine.make_optimizer(old,cfg),engine.make_optimizer(new,dict(cfg,dynamic_readout_lr_multiplier=1.0))
        for step in range(4):
            for model,opt in [(old,a),(new,b)]:
                opt.zero_grad(set_to_none=True)
                model.objective(model(*x),*y,mask)['total'].backward()
                engine.clip_total_norm([p for p in model.parameters() if p.requires_grad],cfg['gradient_clip'])
                engine.set_learning_rates(opt,cfg,step,4);opt.step()
            for name,value in old.state_dict().items():
                torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)

    def test_checkpoint_diagnostic_matches_actual_adam_adaptive_update(self):
        _, model, cfg, x, y, mask = fixture()
        cfg = dict(cfg,weight_decay=0,dynamic_readout_lr_multiplier=.3)
        optimizer = engine.make_optimizer(model,cfg)
        model.objective(model(*x),*y,mask)['total'].backward();optimizer.step()
        rows = checkpoint_readouts({'model':model.state_dict(),'optimizer':optimizer.state_dict()},cfg)
        self.assertEqual({r['parameter'] for r in rows},{'Wc','Wd'})
        for row in rows:
            self.assertEqual(row['optimizer_step'],1)
            self.assertAlmostEqual(row['lr_scaled_adaptive_step_norm'],float(getattr(model,row['parameter']).norm()),places=12)
        self.assertEqual(next(r for r in rows if r['parameter']=='Wd')['optimizer_group'],'dynamic_cp')

    def test_both_k_cases_resume_parameters_adam_rng_and_lr(self):
        with tempfile.TemporaryDirectory(prefix='vgcpod_k_resume_') as folder:
            root = Path(folder);synthetic_cache(root/'cache')
            stats = prepare_gate_statistics(root/'cache',root/'stats')
            for case in ('K1','K2'):
                cfg = read_json(ROOT/'configs/round_k'/(case+'.json'))
                cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3,
                    gate_statistics_path=str(stats),gate_statistics_sha256=sha(stats))
                args = argparse.Namespace(cache=str(root/'cache'),out=str(root/case/'full'),seed=3351,device='cpu',resume=False)
                engine.train(args,cfg);args.out=str(root/case/'resume');real=engine.predict;calls=[0]
                def interrupted(*a,**kw):
                    calls[0]+=1
                    if calls[0]==2:raise InterruptedError('first epoch committed')
                    return real(*a,**kw)
                with mock.patch.object(engine,'predict',side_effect=interrupted):
                    with self.assertRaises(InterruptedError):engine.train(args,cfg)
                args.resume=True;engine.train(args,cfg)
                for name in ('best.pt','best_native.pt','best_exact.pt','last.pt'):
                    sub=Path('ut-zap50k/V1/seed3351/checkpoints')/name
                    a,b=load_torch(root/case/'full'/sub),load_torch(root/case/'resume'/sub)
                    self.assertEqual(a['epoch'],b['epoch']);self.assertEqual(a['step'],b['step'])
                    self.assertTrue(torch.equal(a['rng']['torch'],b['rng']['torch']))
                    self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])
                    for key,value in a['model'].items():self.assertTrue(torch.equal(value,b['model'][key]),key)
                    for index,state in a['optimizer']['state'].items():
                        for key,value in state.items():
                            other=b['optimizer']['state'][index][key]
                            self.assertTrue(torch.equal(value,other) if torch.is_tensor(value) else value==other)
                history=read_json(root/case/'full/ut-zap50k/V1/seed3351/history.json')
                for entry in history:
                    groups=entry['lr_groups']
                    self.assertAlmostEqual(groups['dynamic_cp']/groups['cp'],cfg['dynamic_readout_lr_multiplier'])


if __name__=='__main__':unittest.main(verbosity=2)
