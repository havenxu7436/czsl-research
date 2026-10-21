"""TRAIN-only residual statistics, exact gate equations and checkpoint recovery."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from tests.test_math import fixture,synthetic_cache
from vgcpod import engine
from vgcpod.common import ROOT,load_torch,read_json,sha
from vgcpod.gate_features import prepare_gate_statistics
from vgcpod.model import VGCPOD

torch.set_num_threads(1)


def residual_fixture():
    reference,small,x,y,mask=fixture('V1')
    cfg=read_json(ROOT/'configs/round_h/H1.json')
    cfg.update(rank=small['rank'],adapter_rank=small['adapter_rank'])
    mean=x[2].double().mean(0)
    model=VGCPOD(reference.ea,reference.eo,reference.ec,cfg,3351,gate_mean=mean).double()
    return model,cfg,tuple(t.double() for t in x),y,mask


class RoundHTests(unittest.TestCase):
    def test_config_changes_only_gate_representation(self):
        g=read_json(ROOT/'configs/round_g/G1.json')
        h=read_json(ROOT/'configs/round_h/H1.json')
        self.assertEqual(set(h)-set(g),{'gate_residual_eps'})
        self.assertEqual({key for key in g if g[key]!=h[key]},{'experiment_id','gate_input'})
        self.assertEqual(h['gate_residual_eps'],1e-12)
        model,_,_,_,_=residual_fixture()
        self.assertIn('gate_fo_mean',dict(model.named_buffers()))
        self.assertNotIn('gate_fo_mean',dict(model.named_parameters()))
        self.assertFalse(model.gate_fo_mean.requires_grad)

    def test_statistics_use_all_train_images_and_queries_without_labels(self):
        with tempfile.TemporaryDirectory(prefix='vgcpod_h_means_') as folder:
            root=Path(folder);synthetic_cache(root/'cache')
            # The statistics pass has no dependency on labels or validation/test features.
            for phase in ('val','test'):
                for path in (root/'cache'/phase).glob('*.npy'):path.unlink()
            for path in (root/'cache').glob('*_records.json'):path.unlink()
            expected=np.load(root/'cache/train/fo.npy').astype(np.float64).mean(0)
            path=prepare_gate_statistics(root/'cache',root/'stats',chunk_size=3)
            with np.load(path,allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['mean_fo_float64'],expected)
                np.testing.assert_array_equal(saved['mean_fo'],expected.astype(np.float32))
            before=sha(path)
            self.assertEqual(path,prepare_gate_statistics(root/'cache',root/'stats',chunk_size=7))
            self.assertEqual(before,sha(path))
            self.assertEqual(read_json(path.with_suffix('.json'))['train_images'],16)

    def test_gate_operator_and_full_loss_finite_difference(self):
        model,cfg,x,y,mask=residual_fixture()
        with torch.no_grad():model.Wc.normal_(0,.05)
        g,fa,fo=x
        residual=fo-model.gate_fo_mean[None]
        normalized=residual/residual.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        q=2*torch.sigmoid(normalized@model.Bvis.T/.1)-1
        expected=1+.125*(q-q.mean(-1,keepdim=True))
        out=model(*x)
        torch.testing.assert_close(out['gamma'],expected,rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(out['gamma'].mean(-1),torch.ones_like(q[...,0]),rtol=0,atol=1e-15)
        self.assertLessEqual(float(out['gamma'].max()),1.25)
        self.assertGreaterEqual(float(out['gamma'].min()),.75)
        u,v=model.ea@model.Au.T,model.eo@model.Av.T
        correction=torch.stack([torch.stack([
            g[b]@model.Wc@torch.diag(v[j]*expected[b,j])@model.Au@model.ea.T
            for j in range(len(v))]) for b in range(len(g))])
        torch.testing.assert_close(out['cond']-out['base'],cfg['rho']*cfg['kappa']*correction,rtol=1e-12,atol=1e-12)
        loss=model.objective(out,*y,mask)['total'];loss.backward()
        parameter=model.Bvis;index=int(parameter.grad.abs().argmax())
        analytic=float(parameter.grad.flatten()[index]);self.assertGreater(abs(analytic),1e-10)
        with torch.no_grad():
            original=float(parameter.flatten()[index])
            parameter.flatten()[index]=original+1e-5
            plus=float(model.objective(model(*x),*y,mask)['total'])
            parameter.flatten()[index]=original-1e-5
            minus=float(model.objective(model(*x),*y,mask)['total'])
            parameter.flatten()[index]=original
        numeric=(plus-minus)/2e-5
        self.assertLess(abs(numeric-analytic)/max(abs(numeric),abs(analytic)),1e-5)
        # Check the normalization Jacobian separately from trainable gate parameters.
        z=residual[0,0].clone().requires_grad_();w=torch.arange(len(z),dtype=z.dtype)
        gradient=torch.autograd.grad((F.normalize(z,dim=0,eps=1e-12)*w).sum(),z)[0]
        h=z.detach()/z.detach().norm()
        torch.testing.assert_close(gradient,(w-h*(h@w))/z.detach().norm(),rtol=1e-13,atol=1e-13)

    def test_zero_residual_and_first_two_cp_gradients(self):
        model,cfg,x,y,mask=residual_fixture()
        out=model(*x)
        self.assertTrue(torch.equal(out['base'],out['cond']))
        zero_query=model.gate_fo_mean[None].expand_as(x[2])
        self.assertTrue(torch.equal(model(x[0],x[1],zero_query)['gamma'],torch.ones_like(out['gamma'])))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        model.objective(out,*y,mask)['task'].backward()
        self.assertGreater(model.Wc.grad.norm().item(),0)
        self.assertEqual(model.Bvis.grad.norm().item(),0)
        optimizer.step();optimizer.zero_grad()
        model.objective(model(*x),*y,mask)['task'].backward()
        self.assertGreater(model.Bvis.grad.norm().item(),0)
        self.assertIsNone(model.gate_fo_mean.grad)

    def test_resume_and_evaluation_restore_the_fixed_mean_from_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='vgcpod_h_resume_') as folder:
            root=Path(folder);synthetic_cache(root/'cache')
            statistics=prepare_gate_statistics(root/'cache',root/'stats')
            cfg=read_json(ROOT/'configs/round_h/H1.json')
            cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3,
                gate_statistics_path=str(statistics),gate_statistics_sha256=sha(statistics))
            args=argparse.Namespace(cache=str(root/'cache'),out=str(root/'full'),seed=3351,device='cpu',resume=False)
            engine.train(args,cfg)
            args.out=str(root/'resume');real=engine.predict;calls=[0]
            def interrupted(*a,**kw):
                calls[0]+=1
                if calls[0]==2:raise InterruptedError('first epoch checkpoint committed')
                return real(*a,**kw)
            with mock.patch.object(engine,'predict',side_effect=interrupted):
                with self.assertRaises(InterruptedError):engine.train(args,cfg)
            args.resume=True;engine.train(args,cfg)
            full=root/'full/ut-zap50k/V1/seed3351';resumed=root/'resume/ut-zap50k/V1/seed3351'
            for name in ('best.pt','best_native.pt','best_exact.pt','last.pt'):
                left=load_torch(full/'checkpoints'/name);right=load_torch(resumed/'checkpoints'/name)
                self.assertEqual(left['epoch'],right['epoch']);self.assertEqual(left['step'],right['step'])
                for key,value in left['model'].items():self.assertTrue(torch.equal(value,right['model'][key]),key)
                self.assertTrue(torch.equal(left['rng']['torch'],right['rng']['torch']))
                self.assertEqual(left['optimizer']['param_groups'],right['optimizer']['param_groups'])
                for parameter,state in left['optimizer']['state'].items():
                    for key,value in state.items():
                        other=right['optimizer']['state'][parameter][key]
                        self.assertTrue(torch.equal(value,other) if torch.is_tensor(value) else value==other)
            statistics.unlink()
            for checkpoint,sub in [('best.pt','native'),('best_exact.pt','exact')]:
                target=root/('export_'+sub)
                engine.evaluate_run(root/'cache',full,'cpu',checkpoint_name=checkpoint,out=target)
                original=full if sub=='native' else full/'exact_selection'
                self.assertEqual(read_json(original/'metrics.json')['metrics'],read_json(target/'metrics.json')['metrics'])


if __name__=='__main__':
    unittest.main(verbosity=2)
