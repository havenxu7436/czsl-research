import contextlib
import copy
import io
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from test_formulas import synthetic_cache
from trustcpf import competition as cmp, competition_workflow as workflow, competition_training as training
from trustcpf import compact, align, engine
from trustcpf.common import ROOT, read_json, write_json, load_torch, save_torch, sha
from trustcpf.data import load_cache
from trustcpf.metrics import exact_curve


def tiny_config():
    return dict(read_json(ROOT / 'configs/default.json'), synthetic=True, backbone_epochs=1, teacher_epochs=1,
        adapt_epochs=10, graphs=2, audit_graphs=2, measurement_blocks=2, q5_measurement_blocks=2,
        risk_images_per_group=4, response_backgrounds=2, audit_eval_seeds=[8201, 8202],
        policy_strengths=[0., 1.], permutation_seeds=[7101], eval_graph_seeds=[1701, 2701],
        batch_size=16, eval_batch_size=64, backbone_rank=4, backbone_adapter_rank=2, rank=4, message_rank=2)


def fixture(root, cfg):
    cache = root / 'cache'; cache.mkdir(); synthetic_cache(cache)
    source = root / 'align'; config = root / 'config.json'; write_json(config, cfg)
    for seed in workflow.SEEDS:
        out = source / 'work' / ('seed%d' % seed)
        args = SimpleNamespace(cache=str(cache), work=str(out), config=str(config), seed=seed,
                               variant='uniform', device='cpu', quick=False)
        engine.reference(args)
        save_torch(out / 'inputs/bank.pt', load_torch(cache / 'graph.pt'))
        write_json(out / 'recipe.json', dict(cfg, seed=seed))
        save_torch(out / 'responses/reference.pt',
            dict(features=torch.tensor(np.random.RandomState(seed).normal(size=(24, 8))),
                 checkpoint_sha256=sha(out / 'reference/last.pt')))
    return SimpleNamespace(cache=str(cache), source=str(source), config=str(config), work=str(root / 'work'),
                           seed=3351, device='cpu')


class CompetitionTests(unittest.TestCase):
    def test_area_matches_exact_events_including_ties_and_singleton_groups(self):
        rng = np.random.RandomState(3)
        for c in (2, 7):
            seen = np.arange(c) < max(1, c//2)
            truth = np.array([0, 0, c-1, c-1, c-2 if c>2 else c-1])
            for integer in (False, True):
                z = rng.randn(5, c)
                if integer: z = np.round(z)
                st = cmp.sufficient(torch.tensor(z), torch.tensor(truth), torch.tensor(seen))
                actual = cmp.area(st, torch.tensor(seen[truth]), hard=True)
                self.assertAlmostEqual(float(actual)*100, exact_curve(z, truth, seen)['AUC'], places=12)
        z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.double)
        st = cmp.sufficient(z, torch.tensor([0, 1]), torch.tensor([True, False]))
        self.assertEqual(float(cmp.area(st, torch.tensor([True, False]), hard=True)), .5)

    def test_smooth_area_bias_invariance_chunking_and_probability_normalization(self):
        torch.manual_seed(44)
        z = torch.randn(11, 7, dtype=torch.double)
        y = torch.tensor([0, 1, 2, 0, 1, 3, 4, 5, 6, 4, 3]); seen = torch.arange(7)<3; sm = seen[y]
        s = cmp.sufficient(z, y, seen); r = cmp.risks(s, sm)
        changed = z.clone(); changed[:, ~seen] += 19.2
        t = cmp.sufficient(changed, y, seen)
        self.assertAlmostEqual(float(cmp.area(t, sm)), float(cmp.area(s, sm)), places=13)
        self.assertAlmostEqual(float(cmp.area(s, sm, chunk=1)), float(cmp.area(s, sm, chunk=99)), places=13)
        nll = -z.log_softmax(-1)[torch.arange(len(y)), y]
        self.assertAlmostEqual(float(r[0]), float(nll[~sm].mean()), places=13)
        self.assertAlmostEqual(float(r[1]), float((nll[sm].mean()+nll[~sm].mean())/2), places=13)
        self.assertTrue(0 <= float(r[2]) <= 1)

    def test_paired_effect_equals_bernoulli_risk_derivative(self):
        rng = np.random.RandomState(4); states = np.array(list(itertools.product([0, 1], repeat=2)))
        y = torch.tensor([0, 1, 2, 3]); seen = torch.tensor([True, True, False, False])
        losses = np.array([cmp.risks(cmp.sufficient(torch.tensor(rng.randn(4, 4)), y, seen), seen[y]).numpy() for _ in states])
        p = np.array([.3, .7])
        def expected(q): return np.prod(np.where(states, q, 1-q), axis=1) @ losses
        for edge in range(2):
            effect = np.zeros(3)
            for i, b in enumerate(states):
                if b[edge]: continue
                on = b.copy(); on[edge] = 1; j = np.flatnonzero((states == on).all(1))[0]
                mass = np.prod(np.where(np.delete(b, edge), np.delete(p, edge), 1-np.delete(p, edge)))
                effect += mass*(losses[i]-losses[j])
            hi=p.copy(); lo=p.copy(); hi[edge]+=1e-5; lo[edge]-=1e-5
            np.testing.assert_allclose(effect, -(expected(hi)-expected(lo))/2e-5, atol=2e-11)

    def test_split_partitions_image_pools_and_keeps_every_inner_pair(self):
        cfg=tiny_config(); labels=dict(attrs=['a', 'b', 'c'], objs=['o'])
        rows=[dict(attr=a, object=0, image='%d_%d' % (a, i)) for a,n in enumerate((1, 11, 12)) for i in range(n)]
        split=cmp.split_risk(rows, labels, [['a','o'],['b','o'],['c','o']], [['c','o']], 5, cfg)
        train=set(split['train_ids']); seen=set(split['seen_pool']); unseen=set(split['unseen_pool'])
        self.assertFalse(train&seen or train&unseen or seen&unseen)
        self.assertEqual(train|seen|unseen, set(range(len(rows))))
        self.assertIn(0, train)
        for group in ('seen', 'unseen'):
            self.assertFalse(set(split['measure'][group])&set(split['evaluate'][group]))
        self.assertEqual(split, cmp.split_risk(rows, labels, [['a','o'],['b','o'],['c','o']], [['c','o']], 5, cfg))

    def test_full_three_seed_queue_resume_and_recompute(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root=Path(td); args=fixture(root, tiny_config())
            source_hash={str(p):sha(p) for p in (root/'align').rglob('*') if p.is_file()}
            for seed in workflow.SEEDS:
                args.seed=seed; workflow.run_seed(args)
                work=root/'work'/('seed%d' % seed); reference=load_torch(work/'reference/last.pt')['model']
                for fold in range(1,6):
                    split=read_json(work/'splits'/('fold%d.json' % fold))
                    for file in ('last.pt', 'backbone/last.pt'):
                        state=load_torch(work/'teachers'/str(fold)/file)
                        self.assertEqual(state['train_ids'], split['train_ids'])
                        self.assertEqual(state['train_pairs'], split['inner_pairs'])
                    resp=load_torch(work/'responses'/('fold%d.pt' % fold))
                    self.assertTrue(set(resp['support_indices']).issubset(split['train_ids']))
                    block=load_torch(work/'measurements'/('fold%d' % fold)/'block00.pt')
                    recomputed=torch.stack([cmp.risks(a,block['sample_seen'])-cmp.risks(b,block['sample_seen'])
                                            for a,b in zip(block['off'],block['on'])])
                    torch.testing.assert_close(recomputed, block['effects'], rtol=0, atol=1e-13)
                for variant in cmp.VARIANTS:
                    state=load_torch(work/'runs'/variant/'best.pt')
                    last=load_torch(work/'runs'/variant/'last.pt')
                    self.assertEqual(last['epoch'], 10)
                    self.assertEqual(len(last['history']), 10)
                    self.assertEqual(last['cfg']['adapt_epochs'], 10)
                    self.assertIn(state['epoch'], range(11))
                    self.assertGreater(last['history'][-1]['learning_rate'], 0)
                    self.assertLess(last['history'][-1]['learning_rate'], last['history'][0]['learning_rate'])
                    for key,value in reference.items():
                        if key.startswith('backbone.'):
                            torch.testing.assert_close(value,state['model'][key],rtol=0,atol=0)
                    result=read_json(work/'runs'/variant/'metrics.json')
                    self.assertEqual(result['variant'],variant)
                    for snap,banks in result['results'].items():
                        self.assertEqual(set(banks), {'1701','2701'})
                        for metrics in banks.values():
                            for world in ('cw','ow'):
                                for k in ('bias','temperature'):
                                    self.assertEqual(metrics['val'][world]['calibrated'][k],metrics['test'][world]['calibrated'][k])
                self.assertEqual(len(read_json(work/'decision/audit.json')['rows']),26)
            status=workflow.report(args); self.assertTrue(status['complete'],status)
            self.assertEqual(status['completed_stages'],42);self.assertEqual(status['summary_rows'],240)
            self.assertEqual(status['training_curve_rows'],165)
            before={str(p):p.stat().st_mtime_ns for p in (root/'work').rglob('*.pt')}
            workflow.run_seed(args)
            self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in (root/'work').rglob('*.pt')})
            self.assertEqual(source_hash,{str(p):sha(p) for p in (root/'align').rglob('*') if p.is_file()})
            # Recover interrupted Q5 decision, reusing completed graph banks.
            (root/'work/progress/seed3353/decision.json').unlink()
            blocks={str(p):sha(p) for p in (root/'work/seed3353/decision/blocks').glob('*.pt')}
            workflow.run_seed(args)
            self.assertEqual(blocks,{str(p):sha(p) for p in (root/'work/seed3353/decision/blocks').glob('*.pt')})
            with (root/'work/seed3353/policies.pt').open('ab') as f:f.write(b'changed')
            self.assertFalse(workflow.report(args)['complete'])


if __name__=='__main__': unittest.main()
