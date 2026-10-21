"""F1's isolated beta change, support-loss derivatives and detached epoch logging."""
import argparse
from pathlib import Path
import tempfile
import unittest

import torch

from tests.test_math import fixture,synthetic_cache
from vgcpod import engine
from vgcpod.common import ROOT,read_json

torch.set_num_threads(1)


class RoundFTests(unittest.TestCase):
    def test_f1_changes_only_beta_and_name(self):
        old = read_json(ROOT/'configs/round_e/E1.json')
        new = read_json(ROOT/'configs/round_f/F1.json')
        self.assertEqual(set(old),set(new))
        self.assertEqual({key for key in old if old[key]!=new[key]}, {'task_aux_beta','experiment_id'})
        self.assertEqual(new['rho'],.15)
        self.assertEqual(new['task_aux_beta'],.05)

    def test_support_decomposition_gradient_and_finite_difference(self):
        model = fixture('V1')[0].double()
        generator = torch.Generator().manual_seed(61)
        a_logits = torch.randn(4,2,3,generator=generator,dtype=torch.float64,requires_grad=True)
        o_logits = torch.randn(4,2,generator=generator,dtype=torch.float64,requires_grad=True)
        attrs,objs = torch.tensor([0,1,0,2]),torch.tensor([0,0,1,1])
        mask = torch.tensor([[True,True,False],[True,False,True]])
        def losses():
            la,lo = a_logits.log_softmax(-1),o_logits.log_softmax(-1)
            return model.objective(dict(log_a=la,log_o=lo,joint=lo[:,:,None]+la),attrs,objs,mask)
        for beta in (.05,.1):
            model.cfg['task_aux_beta'] = beta
            result = losses()
            self.assertTrue(torch.allclose(result['cond']+result['object'],
                result['pair']+result['support'],atol=1e-13,rtol=1e-13))
            decomposed = 1.5*result['pair']+(1.5*beta/(1+beta))*result['support']
            self.assertTrue(torch.allclose(result['task'],decomposed,atol=1e-13,rtol=1e-13))
            direct = torch.autograd.grad(result['task'],(a_logits,o_logits),retain_graph=True)
            separate = torch.autograd.grad(decomposed,(a_logits,o_logits))
            for left,right in zip(direct,separate):
                self.assertTrue(torch.allclose(left,right,atol=1e-13,rtol=1e-13))
            with torch.no_grad():
                original = a_logits[0,0,0].item()
                a_logits[0,0,0] = original+1e-6
                plus = losses()['task'].item()
                a_logits[0,0,0] = original-1e-6
                minus = losses()['task'].item()
                a_logits[0,0,0] = original
            self.assertAlmostEqual((plus-minus)/2e-6,direct[0][0,0,0].item(),places=8)

    def test_weighted_epoch_logs(self):
        with tempfile.TemporaryDirectory(prefix='vgcpod_f_logs_') as folder:
            root = Path(folder);synthetic_cache(root/'cache')
            cfg = read_json(ROOT/'configs/round_f/F1.json')
            cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=1)
            engine.train(argparse.Namespace(cache=str(root/'cache'),out=str(root/'out'),seed=3351,
                device='cpu',resume=False),cfg)
            history = read_json(root/'out/ut-zap50k/V1/seed3351/history.json')
            train = history[0]['train']
            self.assertAlmostEqual(train['support_weight'],1/14,places=14)
            self.assertAlmostEqual(train['pair_weighted'],1.5*train['pair'],places=12)
            self.assertAlmostEqual(train['support_weighted'],train['support']/14,places=12)
            self.assertLess(train['task_reconstruction_error'],1e-5)
            self.assertLess(abs(train['task']-train['pair_weighted']-train['support_weighted']),1e-5)


if __name__=='__main__':
    unittest.main(verbosity=2)
