"""D1 gain and fixed-checkpoint TRAIN-mean intervention equations."""
import unittest

import torch

from mean_gate_eval import mean_gate_model, train_gate_moments
from tests.test_round_c import cfixture
from vgcpod.common import ROOT, read_json
from vgcpod.model import VGCPOD


class RoundDTests(unittest.TestCase):
    def test_d1_changes_only_rho_and_case_name(self):
        c = read_json(ROOT/'configs/round_c/C2.json')
        d = read_json(ROOT/'configs/round_d/D1.json')
        self.assertEqual({key for key in c if c[key]!=d[key]}, {'rho','experiment_id'})
        self.assertEqual(d['rho'],.125)

    def test_d1_gain_and_gradient_exactly_scale_c2_at_fixed_parameters(self):
        c,cfg,x,y,mask = cfixture('C2')
        d = VGCPOD(c.ea,c.eo,c.ec,dict(cfg,rho=.125,experiment_id='D1'),3351).double()
        with torch.no_grad():
            c.Wc.normal_(0,.05)
        d.load_state_dict(c.state_dict())
        oc,od = c(*x),d(*x)
        self.assertTrue(torch.equal(oc['gamma'],od['gamma']))
        self.assertTrue(torch.allclose(od['cond']-od['base'],1.25*(oc['cond']-oc['base']),atol=1e-12))
        for p1,p2 in [(c.Wc,d.Wc),(c.Bvis,d.Bvis)]:
            gc = torch.autograd.grad(oc['cond'][0,1,2],p1,retain_graph=True)[0]
            gd = torch.autograd.grad(od['cond'][0,1,2],p2,retain_graph=True)[0]
            self.assertTrue(torch.allclose(gd,1.25*gc,atol=1e-12))

    def test_moments_weight_images_and_ignore_labels_for_all_object_queries(self):
        model,cfg,(g,fa,fo),y,mask = cfixture('C2')
        batches = [(g[:3],fa[:3],fo[:3],y[0][:3],y[1][:3]),
                   (g[3:],fa[3:],fo[3:],y[0][3:]*0,y[1][3:]*0)]
        mean,var,n = train_gate_moments(model,batches,'cpu')
        gamma = model(g,fa,fo)['gamma'].detach()
        self.assertEqual(n,8)
        self.assertEqual(tuple(mean.shape),(2,3))
        self.assertTrue(torch.allclose(mean,gamma.mean(0),atol=1e-14))
        self.assertTrue(torch.allclose(var,gamma.var(0,unbiased=False),atol=1e-14))
        self.assertTrue(torch.allclose(mean.mean(-1),torch.ones(2,dtype=mean.dtype),atol=1e-14))
        self.assertTrue(torch.all(mean>=.75) and torch.all(mean<=1.25))

    def test_fixed_gate_removes_image_modulation_and_preserves_chain_distribution(self):
        model,cfg,x,y,mask = cfixture('C2')
        with torch.no_grad():
            model.Wc.normal_(0,.05)
        original = {k:v.clone() for k,v in model.state_dict().items()}
        dynamic = model(*x)
        mean = dynamic['gamma'].detach().mean(0)
        wrapped = mean_gate_model(model,mean)
        out = wrapped(*x)
        self.assertTrue(torch.equal(out['obj'],dynamic['obj']))
        self.assertTrue(torch.equal(out['base'],dynamic['base']))
        self.assertTrue(torch.equal(out['gamma'][0],out['gamma'][-1]))
        rotated = wrapped(x[0],x[1],x[2].roll(1,0))
        self.assertTrue(torch.equal(out['cond'],rotated['cond']))
        u,v = model.ea@model.Au.T,model.eo@model.Av.T
        expected = cfg['rho']*cfg['kappa']*torch.einsum('br,bor,or,ir->boi',
            x[0]@model.Wc,dynamic['gamma']-mean[None],v,u)
        self.assertTrue(torch.allclose(dynamic['cond']-out['cond'],expected,atol=1e-12))
        self.assertTrue(torch.allclose(out['joint'].exp().sum((1,2)),torch.ones(8,dtype=mean.dtype),atol=1e-12))
        for key,value in original.items():
            self.assertTrue(torch.equal(value,model.state_dict()[key]),key)


if __name__=='__main__':
    unittest.main(verbosity=2)
