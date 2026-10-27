import contextlib
import copy
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from test_competition import fixture,tiny_config
from trustcpf import r5,compact,align,r5_workflow as workflow,r5_training as train,competition_workflow as old_workflow
from trustcpf.common import ROOT,read_json,write_json,load_torch,sha


class R5FormulaTests(unittest.TestCase):
    def data(self):
        rng=np.random.RandomState(90);types=np.repeat(np.arange(3),[5,7,8])
        xs=[rng.normal(size=(len(types),8)) for _ in range(3)]
        ys=[rng.normal(size=len(types)) for _ in range(3)]
        return xs,ys,types

    def test_normal_equations_and_weighted_pairwise_equivalence(self):
        xs,ys,types=self.data();alpha=.7;models=r5.paired_models(xs,ys,types,alpha);m=models['relative']
        z=[];y=[]
        for x,v in zip(xs,ys):
            design=compact.design(x,types,m['center'],m['spread'])
            for t in range(3):
                k=types==t;d=design[k];q=v[k]/m['scales'][t];n=len(q)
                # ||P r||^2 = (1/n) sum_{i<j}(r_i-r_j)^2.
                for i in range(n):
                    for j in range(i+1,n):z.append((d[i]-d[j])/np.sqrt(n));y.append((q[i]-q[j])/np.sqrt(n))
        z=np.array(z);y=np.array(y)
        expected=np.linalg.solve(z.T@z+alpha*np.eye(27),z.T@y)
        np.testing.assert_allclose(expected,m['beta'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(z.T@(z@m['beta']-y)+alpha*m['beta'],0,atol=1e-12)
        np.testing.assert_array_equal(m['beta'][[0,9,18]],np.zeros(3))
        for key in ('center','spread','scales'):np.testing.assert_array_equal(m[key],models['absolute'][key])

    def test_teacher_type_offsets_cancel_with_shared_normalization(self):
        xs,ys,t=self.data();m=r5.paired_models(xs,ys,t,1.)['relative'];rng=np.random.RandomState(18)
        x2=[x+np.stack([rng.normal(size=8) for _ in range(3)])[t] for x in xs]
        y2=[y+rng.normal(size=3)[t]*50 for y in ys]
        changed=r5.fit_relative(x2,y2,t,1.,m)
        np.testing.assert_allclose(changed['beta'],m['beta'],atol=1e-12)

    def test_budget_shift_and_permutation_match(self):
        xs,ys,t=self.data();mu=ys[0];p,_=align.policy(mu,t,1.)
        changed,_=align.policy(mu+np.array([7.,-5.,13.])[t],t,1.)
        torch.testing.assert_close(p,changed,rtol=0,atol=1e-12)
        perm=align.permutation(t,7101)
        for k in range(3):
            self.assertAlmostEqual(float(p[t==k].sum()),(t==k).sum()/2,places=10)
            torch.testing.assert_close(p[t==k].sort().values,p[perm][t==k].sort().values,rtol=0,atol=0)

    def test_source_comparison_accepts_line_endings_only(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'a.py';b=Path(td)/'b.py';a.write_bytes(b'x=1\ny=2\n');b.write_bytes(b'x=1\r\ny=2\r\n')
            self.assertEqual(workflow.code_digest(a),workflow.code_digest(b))
            b.write_bytes(b'x=3\ny=2\n');self.assertNotEqual(workflow.code_digest(a),workflow.code_digest(b))


class R5QueueTests(unittest.TestCase):
    def test_full_three_seed_import_fit_q5_scan_adapt_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);cfg=tiny_config();cfg.update(experiment='r5_relative_unll_10ep',effect_target='unll')
            args=fixture(root,cfg);oldroot=root/'competition';args.work=str(oldroot/'work')
            (oldroot/'trustcpf').mkdir(parents=True)
            for name in workflow.MATCHED_CODE:shutil.copy2(ROOT/'trustcpf'/name,oldroot/'trustcpf'/name)
            for seed in workflow.SEEDS:args.seed=seed;old_workflow.run_seed(args)
            self.assertTrue(old_workflow.report(args)['complete'])
            original_hashes={p.relative_to(oldroot).as_posix():sha(p) for p in oldroot.rglob('*') if p.is_file()}
            args.source=str(oldroot);args.work=str(root/'r5_work')
            for seed in workflow.SEEDS:
                args.seed=seed;workflow.run_seed(args);work=Path(args.work)/('seed%d'%seed)
                self.assertTrue(read_json(work/'utility/absolute_reproduction.json')['passed'])
                types=load_torch(work/'policies.pt')['types'];p=load_torch(work/'policies.pt')['formal'];perm=align.permutation(types,cfg['permutation_seeds'][0])
                for method in r5.METHODS:torch.testing.assert_close(p['permuted_'+method],p[method][perm],rtol=0,atol=0)
                for v in r5.VARIANTS:
                    best=load_torch(work/'runs'/v/'best.pt');last=load_torch(work/'runs'/v/'last.pt')
                    self.assertEqual(last['epoch'],10);self.assertEqual(len(last['history']),10)
                    self.assertIn(best['epoch'],range(11))
                    ref=load_torch(work/'reference/last.pt')['model']
                    for k,x in ref.items():
                        if k.startswith('backbone.'):torch.testing.assert_close(x,best['model'][k],rtol=0,atol=0)
                    result=read_json(work/'runs'/v/'metrics.json');self.assertEqual(result['variant'],v)
                    for banks in result['results'].values():
                        for entry in banks.values():
                            for world in ('cw','ow'):
                                for key in ('bias','temperature'):self.assertEqual(entry['val'][world]['calibrated'][key],entry['test'][world]['calibrated'][key])
                self.assertEqual(sha(work/'runs/uniform/last.pt'),sha(oldroot/'work'/('seed%d'%seed)/'runs/uniform/last.pt'))
                self.assertEqual(sha(work/'runs/absolute/last.pt'),sha(oldroot/'work'/('seed%d'%seed)/'runs/unll/last.pt'))
                audit=read_json(work/'decision/audit.json');self.assertEqual(len(audit['rows']),14)
                for graph_seed in cfg['audit_eval_seeds']:
                    block=load_torch(work/'decision/blocks'/('evaluate%d.pt'%graph_seed));baseline=block['results']['uniform']
                    for method in r5.POLICIES:
                        self.assertIn(method,block['results'])
                    self.assertEqual(block['origins']['absolute'],'source/predicted_unll')
            status=workflow.report(args);self.assertTrue(status['complete'],status)
            self.assertEqual(status['completed_stages'],27);self.assertEqual(status['summary_rows'],240)
            self.assertEqual(status['training_curve_rows'],165);self.assertEqual(status['decision_bank_rows'],42)
            before={str(p):p.stat().st_mtime_ns for p in Path(args.work).rglob('*.pt')}
            workflow.run_seed(args)
            self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in Path(args.work).rglob('*.pt')})
            (Path(args.work)/'progress/seed3353/decision.json').unlink();workflow.run_seed(args)
            self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in Path(args.work).rglob('*.pt')})
            self.assertEqual(original_hashes,{p.relative_to(oldroot).as_posix():sha(p) for p in oldroot.rglob('*') if p.is_file()})
            if os.getenv('R5_SAVE_SMOKE')=='1':
                dest=ROOT/'validation/smoke_work'
                shutil.copytree(Path(args.work),dest,dirs_exist_ok=True)
                write_json(ROOT/'validation/smoke_config.json',cfg)
            with (Path(args.work)/'seed3353/policies.pt').open('ab') as f:f.write(b'changed')
            self.assertFalse(workflow.report(args)['complete'])


if __name__=='__main__':unittest.main()
