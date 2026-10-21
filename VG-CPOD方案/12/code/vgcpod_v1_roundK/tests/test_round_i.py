"""Residual magnitude equations, representative-scale identity and deterministic I1 resume."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from tests.test_round_h import residual_fixture
from tests.test_math import synthetic_cache
from vgcpod import engine
from vgcpod.common import ROOT,load_torch,read_json,sha
from vgcpod.gate_features import prepare_gate_statistics
from vgcpod.model import VGCPOD

torch.set_num_threads(1)


def fixture(case):
    ref,small,x,y,mask=residual_fixture()
    cfg=read_json(ROOT/'configs/round_i'/(case+'.json'))
    cfg.update(rank=small['rank'],adapter_rank=small['adapter_rank'])
    model=VGCPOD(ref.ea,ref.eo,ref.ec,cfg,3351,gate_mean=ref.gate_fo_mean).double()
    return model,cfg,x,y,mask


class RoundITests(unittest.TestCase):
    def test_only_planned_config_changes_and_no_new_parameters(self):
        h=read_json(ROOT/'configs/round_h/H1.json')
        a=read_json(ROOT/'configs/round_i/I1.json');b=read_json(ROOT/'configs/round_i/I2.json')
        self.assertEqual(set(h),set(a));self.assertEqual(set(h),set(b))
        self.assertEqual({k for k in h if h[k]!=a[k]},{'experiment_id','gate_input'})
        self.assertEqual({k for k in h if h[k]!=b[k]},{'experiment_id','tau_gamma'})
        self.assertEqual(a['gate_input'],'train_centered');self.assertEqual(b['tau_gamma'],.56)
        old=residual_fixture()[0];left=fixture('I1')[0];right=fixture('I2')[0]
        self.assertEqual(old.trainable_count(),left.trainable_count())
        self.assertEqual(old.trainable_count(),right.trainable_count())
        self.assertFalse(left.gate_fo_mean.requires_grad)

    def test_centered_gate_cp_operator_and_loss_gradient(self):
        model,cfg,x,y,mask=fixture('I1')
        with torch.no_grad():model.Wc.normal_(0,.05)
        residual=x[2]-model.gate_fo_mean[None]
        q=2*torch.sigmoid(residual@model.Bvis.T/.1)-1
        gamma=1+.125*(q-q.mean(-1,keepdim=True))
        out=model(*x)
        torch.testing.assert_close(out['gamma'],gamma,rtol=1e-13,atol=1e-13)
        u,v=model.ea@model.Au.T,model.eo@model.Av.T
        explicit=torch.stack([torch.stack([x[0][i]@model.Wc@torch.diag(v[j]*gamma[i,j])@model.Au@model.ea.T
            for j in range(len(v))]) for i in range(len(x[0]))])
        torch.testing.assert_close(out['cond']-out['base'],cfg['rho']*cfg['kappa']*explicit,rtol=1e-12,atol=1e-12)
        model.objective(out,*y,mask)['total'].backward()
        parameter=model.Bvis;index=int(parameter.grad.abs().argmax());analytic=float(parameter.grad.flatten()[index])
        self.assertGreater(abs(analytic),1e-10)
        with torch.no_grad():
            original=float(parameter.flatten()[index])
            parameter.flatten()[index]=original+1e-5;plus=float(model.objective(model(*x),*y,mask)['total'])
            parameter.flatten()[index]=original-1e-5;minus=float(model.objective(model(*x),*y,mask)['total'])
            parameter.flatten()[index]=original
        numeric=(plus-minus)/2e-5
        self.assertLess(abs(numeric-analytic)/max(abs(analytic),abs(numeric)),1e-5)
        torch.testing.assert_close(gamma.mean(-1),torch.ones_like(gamma[...,0]),rtol=0,atol=1e-15)
        self.assertGreaterEqual(float(gamma.min()),.75);self.assertLessEqual(float(gamma.max()),1.25)

    def test_representative_scale_identity_and_magnitude_information(self):
        left,cfg,x,_,_=fixture('I1');right=fixture('I2')[0]
        with torch.no_grad():left.Wc.normal_(0,.05)
        right.load_state_dict(left.state_dict())
        scale=cfg['tau_gamma']/right.cfg['tau_gamma']
        direction=torch.nn.functional.normalize(x[2],dim=-1)
        query=left.gate_fo_mean[None]+scale*direction
        a=left(x[0],x[1],query);b=right(x[0],x[1],query)
        torch.testing.assert_close(a['gamma'],b['gamma'],rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(a['cond'],b['cond'],rtol=1e-12,atol=1e-12)
        larger=left.gate_fo_mean[None]+2*scale*direction
        a2=left(x[0],x[1],larger);b2=right(x[0],x[1],larger)
        self.assertGreater(float((a2['gamma']-a['gamma']).abs().max()),1e-4)
        torch.testing.assert_close(b['gamma'],b2['gamma'],rtol=1e-12,atol=1e-12)
        zero=left(x[0],x[1],left.gate_fo_mean[None].expand_as(query))
        self.assertTrue(torch.equal(zero['gamma'],torch.ones_like(zero['gamma'])))

    def test_i1_resume_restores_parameters_optimizer_and_fixed_mean(self):
        with tempfile.TemporaryDirectory(prefix='vgcpod_i_resume_') as folder:
            root=Path(folder);synthetic_cache(root/'cache')
            stats=prepare_gate_statistics(root/'cache',root/'stats')
            cfg=read_json(ROOT/'configs/round_i/I1.json')
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
            stats.unlink()
            for name,sub in [('best.pt','native'),('best_exact.pt','exact')]:
                target=root/('export_'+sub);engine.evaluate_run(root/'cache',full,'cpu',checkpoint_name=name,out=target)
                original=full if sub=='native' else full/'exact_selection'
                self.assertEqual(read_json(target/'metrics.json')['metrics'],read_json(original/'metrics.json')['metrics'])


if __name__=='__main__':unittest.main(verbosity=2)
