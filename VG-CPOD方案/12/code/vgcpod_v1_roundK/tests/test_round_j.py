"""Separate-readout identities, gradients, TRAIN-mean intervention and resume."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from tests.test_round_i import fixture as i_fixture
from tests.test_math import synthetic_cache
from vgcpod import engine
from vgcpod.common import ROOT, load_torch, read_json, sha
from vgcpod.gate_features import prepare_gate_statistics
from vgcpod.model import VGCPOD
from mean_gate_eval import mean_gate_model

torch.set_num_threads(1)


def fixture():
    old, cfg, x, y, mask = i_fixture('I2')
    cfg = dict(cfg, experiment_id='J1', cp_readout_mode='separate')
    model = VGCPOD(old.ea, old.eo, old.ec, cfg, 3351, gate_mean=old.gate_fo_mean).double()
    return old, model, cfg, x, y, mask


class RoundJTests(unittest.TestCase):
    def test_config_initialization_parameter_count_and_optimizer(self):
        previous = read_json(ROOT/'configs/round_i/I2.json')
        current = read_json(ROOT/'configs/round_j/J1.json')
        self.assertEqual(set(current)-set(previous), {'cp_readout_mode'})
        self.assertEqual({k for k in previous if previous[k] != current[k]}, {'experiment_id'})
        old, model, cfg, x, _, _ = fixture()
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        self.assertTrue(torch.equal(model.Wd, torch.zeros_like(model.Wd)))
        self.assertTrue(torch.equal(model(*x)['cond'], old(*x)['cond']))
        self.assertEqual(model.trainable_count()-old.trainable_count(), model.Wd.numel())
        optimizer = engine.make_optimizer(model, cfg)
        cp = next(group for group in optimizer.param_groups if group['name'] == 'cp')
        self.assertTrue(any(p is model.Wc for p in cp['params']))
        self.assertTrue(any(p is model.Wd for p in cp['params']))
        self.assertEqual(cp['lr'], 3e-4)
        full = VGCPOD(torch.zeros(16,768), torch.zeros(12,768), torch.zeros(12,16,768),
            current, 3351, gate_mean=torch.zeros(12,768))
        self.assertEqual(full.trainable_count(), 147456)

    def test_tied_readouts_restore_i2_outputs_and_loss_gradients(self):
        old, model, cfg, x, y, mask = fixture()
        with torch.no_grad():
            old.Wc.normal_(0, .07)
            model.Wc.copy_(old.Wc);model.Wd.copy_(old.Wc)
        a, b = old(*x), model(*x)
        torch.testing.assert_close(a['cond'], b['cond'], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(a['joint'], b['joint'], rtol=1e-12, atol=1e-12)
        old.objective(a, *y, mask)['total'].backward()
        model.objective(b, *y, mask)['total'].backward()
        torch.testing.assert_close(old.Wc.grad, model.Wc.grad+model.Wd.grad, rtol=1e-10, atol=1e-12)
        for name in ('Au','Av','Bvis','Bo','Co','Bc','Cc'):
            torch.testing.assert_close(getattr(old,name).grad, getattr(model,name).grad, rtol=1e-10, atol=1e-12)

    def test_explicit_low_rank_operator_and_static_special_case(self):
        _, model, cfg, x, _, _ = fixture()
        with torch.no_grad():model.Wc.normal_(0,.07);model.Wd.normal_(0,.11)
        out = model(*x);v=model.eo@model.Av.T;h=out['gamma']-1
        explicit=torch.stack([torch.stack([x[0][b] @ (
            model.Wc@torch.diag(v[j])+model.Wd@torch.diag(v[j]*h[b,j])) @model.Au@model.ea.T
            for j in range(len(v))]) for b in range(len(x[0]))])
        torch.testing.assert_close(out['cond']-out['base'],cfg['rho']*cfg['kappa']*explicit,rtol=1e-12,atol=1e-12)
        with torch.no_grad():model.Wd.zero_()
        out=model(*x)
        self.assertTrue(torch.equal(out['cp_dynamic'],torch.zeros_like(out['cp_dynamic'])))
        torch.testing.assert_close(out['cond'],out['base']+out['cp_static'],rtol=0,atol=0)

    def test_separate_readout_and_gate_loss_finite_differences(self):
        _, model, _, x, y, mask=fixture()
        with torch.no_grad():model.Wc.normal_(0,.05);model.Wd.normal_(0,.13)
        model.objective(model(*x),*y,mask)['total'].backward()
        for name in ('Wc','Wd','Bvis','Au','Av'):
            parameter=getattr(model,name);index=int(parameter.grad.abs().argmax())
            analytic=float(parameter.grad.flatten()[index]);self.assertGreater(abs(analytic),1e-9,name)
            with torch.no_grad():
                original=float(parameter.flatten()[index])
                parameter.flatten()[index]=original+1e-5;plus=float(model.objective(model(*x),*y,mask)['total'])
                parameter.flatten()[index]=original-1e-5;minus=float(model.objective(model(*x),*y,mask)['total'])
                parameter.flatten()[index]=original
            numeric=(plus-minus)/2e-5
            self.assertLess(abs(numeric-analytic)/max(abs(analytic),abs(numeric)),1e-5,name)

    def test_mean_gate_uses_both_distinct_readouts(self):
        _,model,cfg,x,_,_=fixture()
        with torch.no_grad():model.Wc.normal_(0,.05);model.Wd.normal_(0,.15)
        out=model(*x);mean=out['gamma'].detach().mean(0)
        fixed=mean_gate_model(model,mean)(*x)
        u,v=model.ea@model.Au.T,model.eo@model.Av.T
        expected=torch.stack([torch.stack([x[0][b]@model.Wc@(u*v[j]).T
            +x[0][b]@model.Wd@(u*v[j]*(mean[j]-1)).T
            for j in range(len(v))]) for b in range(len(x[0]))])
        torch.testing.assert_close(fixed['cond'],out['base']+cfg['rho']*cfg['kappa']*expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(fixed['joint'].exp().sum((1,2)),torch.ones(len(x[0]),dtype=torch.float64))
        torch.testing.assert_close(fixed['cp_static'],out['cp_static'],rtol=0,atol=0)
        unit=mean_gate_model(model,torch.ones_like(mean))(*x)
        torch.testing.assert_close(unit['cond'],out['base']+out['cp_static'],rtol=0,atol=0)
        self.assertTrue(torch.equal(unit['cp_dynamic'],torch.zeros_like(unit['cp_dynamic'])))

    def test_zero_initialization_learns_both_readouts_and_gate(self):
        _,model,cfg,x,y,mask=fixture();optimizer=engine.make_optimizer(model,cfg)
        model.objective(model(*x),*y,mask)['total'].backward()
        self.assertGreater(float(model.Wc.grad.norm()),0)
        self.assertGreater(float(model.Wd.grad.norm()),0)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        model.objective(model(*x),*y,mask)['task'].backward()
        for name in ('Au','Av','Bvis'):
            self.assertGreater(float(getattr(model,name).grad.norm()),0,name)

    def test_j1_resume_restores_both_readouts_optimizer_and_mean(self):
        with tempfile.TemporaryDirectory(prefix='vgcpod_j_resume_') as folder:
            root=Path(folder);synthetic_cache(root/'cache')
            stats=prepare_gate_statistics(root/'cache',root/'stats')
            cfg=read_json(ROOT/'configs/round_j/J1.json')
            cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3,
                gate_statistics_path=str(stats),gate_statistics_sha256=sha(stats))
            args=argparse.Namespace(cache=str(root/'cache'),out=str(root/'full'),seed=3351,device='cpu',resume=False)
            engine.train(args,cfg);args.out=str(root/'resume');real=engine.predict;calls=[0]
            def interrupted(*a,**kw):
                calls[0]+=1
                if calls[0]==2:raise InterruptedError('first epoch committed')
                return real(*a,**kw)
            with mock.patch.object(engine,'predict',side_effect=interrupted):
                with self.assertRaises(InterruptedError):engine.train(args,cfg)
            args.resume=True;engine.train(args,cfg)
            full=root/'full/ut-zap50k/V1/seed3351';resume=root/'resume/ut-zap50k/V1/seed3351'
            for name in ('best.pt','best_native.pt','best_exact.pt','last.pt'):
                a=load_torch(full/'checkpoints'/name);b=load_torch(resume/'checkpoints'/name)
                self.assertEqual(a['epoch'],b['epoch']);self.assertEqual(a['step'],b['step'])
                self.assertTrue(torch.equal(a['rng']['torch'],b['rng']['torch']))
                self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])
                for key,value in a['model'].items():self.assertTrue(torch.equal(value,b['model'][key]),key)
                for parameter,state in a['optimizer']['state'].items():
                    for key,value in state.items():
                        other=b['optimizer']['state'][parameter][key]
                        self.assertTrue(torch.equal(value,other) if torch.is_tensor(value) else value==other)
            for entry in read_json(full/'history.json'):
                for key in ('cp_static','cp_dynamic','cp_static_centered','cp_dynamic_centered'):
                    self.assertAlmostEqual(entry['train'][key+'_rms']**2,entry['train'][key+'_mse'],places=12)
            stats.unlink()
            engine.evaluate_run(root/'cache',full,'cpu',out=root/'reloaded')
            self.assertEqual(read_json(root/'reloaded/metrics.json')['metrics'],read_json(full/'metrics.json')['metrics'])


if __name__=='__main__':unittest.main(verbosity=2)
