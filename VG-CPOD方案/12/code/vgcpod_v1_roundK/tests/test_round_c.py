"""Static gain equivalence, centered-gate geometry and gradients, training and exports."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import round_c
from tests.test_math import fixture, synthetic_cache
from vgcpod import engine
from vgcpod.common import ROOT, load_torch, read_json
from vgcpod.model import VGCPOD

torch.set_num_threads(1)


def cfixture(case):
    ref, small, x, y, mask = fixture('V1')
    cfg = read_json(ROOT / 'configs/round_c' / (case+'.json'))
    cfg.update(rank=small['rank'], adapter_rank=small['adapter_rank'], workers=0,
               batch_size=8, eval_batch_size=8, epochs=2)
    model = VGCPOD(ref.ea,ref.eo,ref.ec,cfg,3351).double()
    return model,cfg,tuple(t.double() for t in x),y,mask


class RoundCMathTests(unittest.TestCase):
    def test_static_gain_equivalence_in_forward_and_parameter_gradient(self):
        model,cfg,x,y,mask = cfixture('C1')
        baseline = VGCPOD(model.ea,model.eo,model.ec,dict(cfg,rho=.1),3351).double()
        with torch.no_grad():
            model.Wc.normal_(0,.05)
        baseline.load_state_dict(model.state_dict())
        a,b = model(*x),baseline(*x)
        self.assertTrue(torch.equal(a['base'],b['base']))
        self.assertTrue(torch.equal(a['obj'],b['obj']))
        self.assertTrue(torch.allclose(a['cond']-a['base'],1.25*(b['cond']-b['base']),atol=1e-12))
        ga = torch.autograd.grad(a['cond'][0,0,0],model.Wc)[0]
        gb = torch.autograd.grad(b['cond'][0,0,0],baseline.Wc)[0]
        self.assertTrue(torch.allclose(ga,1.25*gb,atol=1e-12))
        self.assertFalse(model.Bvis.requires_grad)

    def test_mode_mean_bounds_zero_start_and_common_mode_removal(self):
        model,cfg,x,y,mask = cfixture('C2')
        out = model(*x)
        self.assertTrue(torch.equal(out['cond'],out['base']))
        self.assertTrue(torch.allclose(out['gamma'].mean(-1),torch.ones_like(out['gamma'][...,0]),atol=1e-14))
        self.assertTrue(torch.all(out['gamma']>.75) and torch.all(out['gamma']<1.25))
        gradient = torch.autograd.grad(out['gamma'].sum(),model.Bvis)[0]
        self.assertLess(float(gradient.abs().max()),1e-14)
        with torch.no_grad():
            model.Bvis.copy_(model.Bvis[0:1].expand_as(model.Bvis).clone())
        same = model(*x)['gamma']
        self.assertTrue(torch.allclose(same,torch.ones_like(same),atol=1e-14))
        with torch.no_grad():
            model.Bvis.normal_(0,1000)
        extreme = model(*x)['gamma']
        self.assertTrue(torch.isfinite(extreme).all())
        self.assertTrue(torch.all(extreme>=.75) and torch.all(extreme<=1.25))
        self.assertTrue(torch.allclose(extreme.mean(-1),torch.ones_like(extreme[...,0]),atol=1e-14))

    def test_centered_operator_and_visual_gradient_finite_difference(self):
        model,cfg,(g,fa,fo),y,mask = cfixture('C2')
        with torch.no_grad():
            model.Wc.normal_(0,.05)
        out = model(g,fa,fo)
        self.assertTrue(torch.equal(out['gamma'],model(g.roll(1,0),fa,fo)['gamma']))
        self.assertFalse(torch.allclose(out['gamma'],model(g,fa,fo.roll(1,0))['gamma']))
        b,j,i = 0,1,2
        operator = model.Wc @ torch.diag((model.Av @ model.eo[j])*out['gamma'][b,j]) @ model.Au
        direct = cfg['rho']*cfg['kappa']*g[b] @ operator @ model.ea[i]
        self.assertTrue(torch.allclose(out['cond'][b,j,i]-out['base'][b,j,i],direct,atol=1e-12))
        self.assertLessEqual(int(torch.linalg.matrix_rank(operator)),cfg['rank'])
        self.assertTrue(torch.allclose(out['joint'].exp().sum((1,2)),torch.ones(len(g),dtype=g.dtype),atol=1e-12))
        model.objective(out,*y,mask)['task'].backward()
        k = int(model.Bvis.grad.abs().argmax())
        analytic = float(model.Bvis.grad.flatten()[k])
        self.assertGreater(abs(analytic),1e-12)
        with torch.no_grad():
            flat = model.Bvis.view(-1)
            original = flat[k].item()
            flat[k] = original+1e-5
            plus = float(model.objective(model(g,fa,fo),*y,mask)['task'])
            flat[k] = original-1e-5
            minus = float(model.objective(model(g,fa,fo),*y,mask)['task'])
            flat[k] = original
        self.assertAlmostEqual((plus-minus)/2e-5,analytic,delta=max(1e-10,abs(analytic)*1e-4))


class RoundCIntegrationTests(unittest.TestCase):
    def test_train_resume_and_matched_reference_summary(self):
        with tempfile.TemporaryDirectory(prefix='vgcpod_roundc_') as temp:
            root = Path(temp)
            small = synthetic_cache(root/'cache')
            for case in ('A5','A6','B3','C1','C2'):
                sub = 'round_a' if case.startswith('A') else ('round_b' if case.startswith('B') else 'round_c')
                cfg = read_json(ROOT/'configs'/sub/(case+'.json'))
                cfg.update(rank=small['rank'],adapter_rank=small['adapter_rank'],workers=0,
                           batch_size=8,eval_batch_size=8,epochs=2)
                args = argparse.Namespace(cache=str(root/'cache'),out=str(root/case),seed=3351,device='cpu',resume=False)
                case_out = root/case
                engine.train(args,cfg)
                run = case_out/'ut-zap50k'/cfg['variant']/'seed3351'
                if case=='C2':
                    history = read_json(run/'history.json')
                    self.assertLess(history[-1]['train']['gamma_mode_mean_deviation'],1e-6)
                    self.assertGreater(history[-1]['train']['gamma_image_variance'],0)
                    original = load_torch(run/'checkpoints/last.pt')
                    args.out = str(root/'resume')
                    real = engine.predict
                    calls = [0]
                    def interrupted(*a,**kw):
                        calls[0] += 1
                        if calls[0]==2:
                            raise InterruptedError('interrupt after epoch commit')
                        return real(*a,**kw)
                    with mock.patch.object(engine,'predict',side_effect=interrupted):
                        with self.assertRaises(InterruptedError):
                            engine.train(args,cfg)
                    args.resume = True
                    engine.train(args,cfg)
                    resumed = load_torch(root/'resume/ut-zap50k/V1/seed3351/checkpoints/last.pt')
                    for key in original['model']:
                        self.assertTrue(torch.equal(original['model'][key],resumed['model'][key]),key)
            rows = round_c.summarize(root,a_out=root,b_out=root)
            self.assertEqual({r['case'] for r in rows},{'C1','C2','B3'})
            summary = read_json(root/'round_c_summary.json')
            comparisons = {r['comparison'] for r in summary['paired_deltas']}
            self.assertTrue({'C1-A5','C1-B3','C2-A5','C2-B3','C2-C1','B3-A5'} <= comparisons)
            self.assertGreater(summary['per_pair_rows'],0)
            self.assertEqual({r['case'] for r in summary['c_val_cw_ranking']},{'C1','C2'})
            with np.load(root/'C2/ut-zap50k/V1/seed3351/predictions/test_ow.npz') as d:
                np.testing.assert_allclose(d['probabilities'].sum(-1),1,atol=1e-5)


if __name__=='__main__':
    unittest.main(verbosity=2)
