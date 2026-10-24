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
from trustcpf import align, align_workflow as workflow, decision_audit, compact, engine
from trustcpf.common import ROOT, read_json, write_json, load_torch, save_torch, sha


def make_sources(root, cfg, seeds=(3351, 3352, 3353)):
    cache = root / 'cache'; cache.mkdir(); synthetic_cache(cache)
    strong, source = root / 'strong', root / 'compact'
    write_json(root / 'config.json', cfg)
    for seed in seeds:
        old = strong / 'work' / ('seed%d' % seed)
        a = SimpleNamespace(cache=str(cache), work=str(old), config=str(root / 'config.json'), seed=seed,
                            variant='robust', quick=False, fold='5', device='cpu')
        engine.teachers(a); engine.reference(a)
        dest = source / 'work' / ('seed%d' % seed)
        import shutil
        shutil.copytree(old / 'reference', dest / 'reference')
        shutil.copy2(old / 'folds.json', dest / 'folds.json')
        write_json(dest / 'recipe.json', dict(cfg, seed=seed))
        bank = load_torch(cache / 'graph.pt'); types = bank['types'].numpy()
        save_torch(dest / 'runs/compact/bank.pt', bank)
        rng = np.random.RandomState(seed)
        x = [rng.randn(len(types), 8) for _ in range(5)]
        y = [rng.randn(len(types)) * .01 for _ in range(5)]
        for final in (False, True):
            n = 5 if final else 3
            model = compact.fit_ridge(x[:n], y[:n], types, 10.)
            model['fit_fold_ids'] = list(range(1, n+1))
            save_torch(dest / ('utility_all' if final else 'utility') / 'compact/model.pt', model)
        write_json(dest / 'utility/selection.json', dict(selected_alpha=10))
        for name, checkpoint in [('reference', old / 'reference/last.pt'), ('fold5', old / 'teachers/5/last.pt')]:
            st = load_torch(checkpoint)
            save_torch(dest / 'responses' / (name+'.pt'), dict(features=torch.tensor(rng.randn(len(types), 8)),
                checkpoint_sha256=sha(checkpoint), train_pairs=st['train_pairs']))
        from trustcpf.data import load_cache
        manifest, text = load_cache(cache)
        model = align.restore(text, bank, dict(cfg, seed=seed), 'cpu', old / 'reference/last.pt')
        val = engine.evaluate_model(model, cache, cfg, manifest, torch.ones(len(types))*.5, 'val')
        test = engine.evaluate_model(model, cache, cfg, manifest, torch.ones(len(types))*.5, 'test', calibration=val)
        metric = dict(val=val, test=test, epoch=0)
        for v in ('compact', 'shuffled_utility', 'drop'):
            write_json(dest / 'runs' / v / 'metrics.json', metric)
        write_json(dest / 'reference_benchmarks.json', dict(capacity=metric, dense=metric, robust=metric))
    return SimpleNamespace(source=str(source), teacher_source=str(strong), cache=str(cache),
        config=str(root / 'config.json'), work=str(root / 'work'), seed=3351, device='cpu')


def tiny_config():
    return dict(read_json(ROOT / 'configs/default.json'), synthetic=True, backbone_epochs=1, teacher_epochs=1,
        adapt_epochs=1, graphs=2, audit_graphs=2, audit_blocks=2, audit_eval_seeds=[8201, 8202],
        policy_strengths=[0., 1.], permutation_seeds=[7101, 7102], eval_graph_seeds=[1701, 2701],
        batch_size=16, eval_batch_size=64, backbone_rank=4, backbone_adapter_rank=2, rank=4, message_rank=2)


class AlignTests(unittest.TestCase):
    def test_policy_equality_kkt_shift_scale_and_local_permutation(self):
        rng = np.random.RandomState(4); types = np.repeat(np.arange(3), [12, 36, 48]); mu = rng.randn(96)
        pi, info = align.policy(mu, types, 2.)
        for t in range(3):
            ix = types == t
            self.assertAlmostEqual(float(pi[ix].sum()), ix.sum()/2, places=10)
            r = (mu[ix]-mu[ix].mean())/mu[ix].std()
            np.testing.assert_allclose(2*r-torch.logit(pi[ix]).numpy(), info[str(t)]['gamma'], atol=1e-11)
        shifted = mu * np.take([.3, 4, 1e-4], types) + np.take([2., -8., .01], types)
        np.testing.assert_allclose(pi, align.policy(shifted, types, 2.)[0], atol=1e-12)
        permutation = align.permutation(types, 7101); shuffled = pi[permutation]
        for t in range(3):
            np.testing.assert_array_equal(np.sort(pi[types==t]), np.sort(shuffled[types==t]))
        small, eta = align.local_policy(shuffled)
        self.assertLessEqual(float((small-.5).abs().max()), .10000000001)
        for t in range(3): self.assertAlmostEqual(float(small[types==t].sum()), (types==t).sum()/2, places=10)
        np.testing.assert_array_equal(align.policy(np.ones(96), types, 2)[0], np.full(96,.5))
        np.testing.assert_array_equal(align.policy(mu, types, 0)[0], np.full(96,.5))

    def test_mixture_slot_identity_against_enumerated_risk_derivative(self):
        # Two edges, two independent slots; enumerate the complete distribution.
        graph_states = np.array(list(itertools.product([0,1], repeat=2)))
        p = np.array([[.12,.55],[.35,.25],[.68,.16],[.42,.75]])
        w = torch.tensor([.3,.7], dtype=torch.double)
        rho = np.array([.41,.62])
        def masses(q): return np.prod(np.where(graph_states,q,1-q),axis=1)
        def risk(q):
            mass=masses(q)
            return sum(mass[a]*mass[b]*float((-(torch.tensor((p[a]+p[b])/2).log())*w).sum())
                       for a in range(4) for b in range(4))
        for e in range(2):
            estimate=0.
            for j,b in enumerate(graph_states):
                if b[e]: continue
                on=b.copy();on[e]=1
                on_index=np.flatnonzero((graph_states==on).all(1))[0]
                other_mass=np.prod(np.where(np.delete(b,e),np.delete(rho,e),1-np.delete(rho,e)))
                for c in range(4):
                    _,dm=decision_audit.block_effect(torch.tensor(p[j]).log(),torch.tensor(p[on_index]).log(),torch.tensor(p[c]).log(),w)
                    estimate+=2*other_mass*masses(rho)[c]*float(dm)
            h=1e-5;plus=rho.copy();minus=rho.copy();plus[e]+=h;minus[e]-=h
            self.assertAlmostEqual(estimate,-(risk(plus)-risk(minus))/(2*h),places=9)
        d=np.arange(24).reshape(3,8)/1000
        result=decision_audit.summarize_effect(d,2*d,16)
        np.testing.assert_allclose(result['mixture_se2'],256*(2*d).var(0,ddof=1)/3)

    def test_independent_images_and_pair_weights(self):
        labels=dict(attrs=['a','b','c'],objs=['o'])
        rows=[dict(attr=a,object=0,image='%s-%s'%(a,j)) for a,n in enumerate((7,3,1)) for j in range(n)]
        split=decision_audit.split_query(rows,labels,[['a','o'],['b','o'],['c','o']],limit=2)
        self.assertEqual(len(split['excluded']),1)
        self.assertFalse(set(r['row_id'] for r in split['measure'])&set(r['row_id'] for r in split['evaluate']))
        for name in ('measure','evaluate'):
            self.assertAlmostEqual(sum(r['weight'] for r in split[name]),1.)
            for pair in split['pairs']:
                self.assertAlmostEqual(sum(r['weight'] for r in split[name] if tuple(r['pair'])==tuple(pair)),.5)

    def test_three_seed_pipeline_replay_resume_and_frozen_backbone(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root=Path(td); cfg=tiny_config(); args=make_sources(root,cfg)
            source_hashes={str(p):sha(p) for d in ('compact','strong') for p in (root/d).rglob('*') if p.is_file()}
            for seed in workflow.SEEDS:
                args.seed=seed; workflow.run_seed(args)
                work=root/'work'/('seed%d'%seed)
                ref=load_torch(work/'reference/last.pt')['model']
                for variant in align.VARIANTS:
                    ck=load_torch(work/'runs'/variant/'best.pt')
                    for key,v in ref.items():
                        if key.startswith('backbone.'): torch.testing.assert_close(v,ck['model'][key],rtol=0,atol=0)
                    metrics=read_json(work/'runs'/variant/'metrics.json')
                    for snapshot in ('epoch0','best'):
                        self.assertEqual(set(metrics['results'][snapshot]),{'1701','2701'})
                audit=read_json(work/'decision/audit.json')
                self.assertEqual(len(audit['policies']),13)
                self.assertEqual(len(audit['rows']),26)
                effects=load_torch(work/'decision/effects.pt')
                np.testing.assert_allclose(effects['mixture'],2*effects['mixture_slot_blocks'].mean(0))
            status=workflow.report(args);self.assertTrue(status['complete'],status)
            self.assertEqual(status['summary_rows'],144)
            before={str(p):p.stat().st_mtime_ns for p in (root/'work').rglob('*.pt')}
            workflow.run_seed(args)
            self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in (root/'work').rglob('*.pt')})
            self.assertEqual(source_hashes,{str(p):sha(p) for d in ('compact','strong') for p in (root/d).rglob('*') if p.is_file()})
            # A partial decision stage resumes already measured blocks unchanged.
            (root/'work/progress/seed3353/decision.json').unlink()
            blocks={str(p):p.stat().st_mtime_ns for p in (root/'work/seed3353/decision/blocks').glob('*.pt')}
            workflow.run_seed(args)
            self.assertEqual(blocks,{str(p):p.stat().st_mtime_ns for p in (root/'work/seed3353/decision/blocks').glob('*.pt')})
            with (root/'work/seed3353/policies.pt').open('ab') as f: f.write(b'changed')
            self.assertFalse(workflow.report(args)['complete'])
            with self.assertRaisesRegex(ValueError,'output changed'): workflow.run_seed(args)

    def test_zero_strength_reuses_identical_adaptation(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);cfg=tiny_config();cfg['policy_strengths']=[0.]
            args=make_sources(root,cfg,seeds=(3351,));workflow.run_seed(args)
            work=root/'work/seed3351/runs'
            for v in ('align','permuted'):
                self.assertEqual(sha(work/v/'best.pt'),sha(work/'uniform/best.pt'))
                self.assertEqual(read_json(work/v/'reuse.json')['from_variant'],'uniform')


if __name__=='__main__': unittest.main()
