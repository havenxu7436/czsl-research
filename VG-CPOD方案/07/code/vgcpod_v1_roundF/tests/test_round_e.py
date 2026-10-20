"""Independent strict-max checkpoint selection and deterministic dual-track resume."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from tests.test_math import synthetic_cache
from vgcpod import engine
from vgcpod.common import ROOT,load_torch,read_json,sha

torch.set_num_threads(1)


def tiny_config():
    cfg = read_json(ROOT/'configs/round_e/E1.json')
    cfg.update(rank=3,adapter_rank=2,workers=0,batch_size=8,eval_batch_size=8,epochs=3)
    return cfg


class RoundETests(unittest.TestCase):
    def test_e1_changes_only_rho_and_experiment_name(self):
        d = read_json(ROOT/'configs/round_d/D1.json')
        e = read_json(ROOT/'configs/round_e/E1.json')
        self.assertEqual({k for k in d if d[k]!=e[k]},{'rho','experiment_id'})
        self.assertEqual(e['rho'],.15)

    def test_independent_first_strict_max_and_separate_exports(self):
        with tempfile.TemporaryDirectory(prefix='vgcpod_e_selection_') as temp:
            root = Path(temp);synthetic_cache(root/'cache')
            args = argparse.Namespace(cache=str(root/'cache'),out=str(root/'out'),seed=3351,
                                      device='cpu',resume=False)
            real = engine.evaluate_scores
            calls = [0]
            def controlled(*a,**kw):
                result = real(*a,**kw)
                idx = calls[0];calls[0] += 1
                if idx<6 and idx%2==0:
                    result[0]['AUC_native'] = [1.,2.,2.][idx//2]
                    result[0]['AUC_exact'] = [3.,2.,3.][idx//2]
                return result
            with mock.patch.object(engine,'evaluate_scores',side_effect=controlled):
                engine.train(args,tiny_config())
            run = root/'out/ut-zap50k/V1/seed3351'
            native = load_torch(run/'checkpoints/best_native.pt')
            exact = load_torch(run/'checkpoints/best_exact.pt')
            last = load_torch(run/'checkpoints/last.pt')
            self.assertEqual(native['epoch'],2)
            self.assertEqual(exact['epoch'],1)
            self.assertEqual(last['best_exact_auc'],3.)
            self.assertEqual(sha(run/'checkpoints/best.pt'),sha(run/'checkpoints/best_native.pt'))
            self.assertEqual(read_json(run/'metrics.json')['selected_epoch'],2)
            self.assertEqual(read_json(run/'exact_selection/metrics.json')['selected_epoch'],1)
            self.assertTrue((run/'exact_selection/predictions/test_ow.npz').exists())
            h = read_json(run/'history.json')
            self.assertEqual([r['selected'] for r in h],[True,True,False])
            self.assertEqual([r['selected_exact'] for r in h],[True,False,False])
            self.assertFalse(torch.equal(native['model']['Wc'],exact['model']['Wc']))

    def test_resume_preserves_both_checkpoint_tracks(self):
        with tempfile.TemporaryDirectory(prefix='vgcpod_e_resume_') as temp:
            root = Path(temp);synthetic_cache(root/'cache')
            args = argparse.Namespace(cache=str(root/'cache'),out=str(root/'full'),seed=3351,
                                      device='cpu',resume=False)
            cfg = tiny_config();engine.train(args,cfg)
            args.out = str(root/'resume')
            real = engine.predict;calls = [0]
            def interrupted(*a,**kw):
                calls[0] += 1
                if calls[0]==2:
                    raise InterruptedError('after first epoch checkpoint commit')
                return real(*a,**kw)
            with mock.patch.object(engine,'predict',side_effect=interrupted):
                with self.assertRaises(InterruptedError):
                    engine.train(args,cfg)
            args.resume = True;engine.train(args,cfg)
            for checkpoint in ('best.pt','best_native.pt','best_exact.pt','last.pt'):
                full = load_torch(root/'full/ut-zap50k/V1/seed3351/checkpoints'/checkpoint)
                resumed = load_torch(root/'resume/ut-zap50k/V1/seed3351/checkpoints'/checkpoint)
                self.assertEqual(full['epoch'],resumed['epoch'])
                self.assertEqual(full['step'],resumed['step'])
                self.assertEqual(full['best_exact_auc'],resumed['best_exact_auc'])
                self.assertTrue(torch.equal(full['rng']['torch'],resumed['rng']['torch']))
                for name,value in full['model'].items():
                    self.assertTrue(torch.equal(value,resumed['model'][name]),(checkpoint,name))


if __name__=='__main__':
    unittest.main(verbosity=2)
