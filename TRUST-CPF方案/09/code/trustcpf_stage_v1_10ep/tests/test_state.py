import contextlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from test_competition import tiny_config,fixture
from trustcpf import state,state_workflow as wf,state_compute as compute,state_eval
from trustcpf import compact,align,competition_workflow as old,competition_training as ct,evaluation
from trustcpf.common import ROOT,read_json,write_json,load_torch,save_torch,sha
from trustcpf.data import load_cache

class StateFormulaTests(unittest.TestCase):
    def test_teacher_average_raw_coordinates_and_projection(self):
        rng=np.random.RandomState(16);types=np.repeat(np.arange(3),[5,7,8])
        xs=[rng.randn(len(types),8) for _ in range(9)];ys=[rng.randn(len(types)) for _ in xs]
        for rel in (False,True):
            m=state.fit_one(xs,ys,types,.7,rel);zz=[];yy=[]
            for x,y in zip(xs,ys):
                z=compact.design(x,types,m['center'],m['spread']);q=y/m['scales'][types]
                if rel:
                    for t in range(3):
                        ix=types==t;p=np.eye(ix.sum())-np.ones((ix.sum(),ix.sum()))/ix.sum()
                        z[ix]=p@z[ix];q[ix]=p@q[ix]
                zz.append(z);yy.append(q)
            z=np.concatenate(zz);y=np.concatenate(yy)
            expected=np.linalg.solve(z.T@z/9+.7*np.eye(27),z.T@y/9)
            np.testing.assert_allclose(m['beta'],expected,atol=1e-12)
            expected=m['raw_intercepts'][types]+(m['raw_slopes'][types]*xs[0]).sum(1)
            np.testing.assert_allclose(compact.predict(m,xs[0],types),expected,atol=1e-12)
            self.assertAlmostEqual(m['alpha'],6.3)

    def test_policy_family_budget_and_sets(self):
        rng=np.random.RandomState(8);t=np.repeat(np.arange(3),[12,36,48])
        cfg={'fixed_strength':1.,'permutation_seeds':[7101,7102,7103,7104,7105]}
        values=state.family({m:rng.randn(96) for m in state.METHODS},t,cfg)
        self.assertEqual(len(values),19)
        for name,r in values.items():
            for ty in range(3):self.assertAlmostEqual(float(r['pi'][t==ty].sum()),(t==ty).sum()/2,places=10)
            if r['permutation_seed'] is not None:
                for ty in range(3):
                    np.testing.assert_array_equal(np.sort(r['pi'][t==ty]),np.sort(values[r['method']]['pi'][t==ty]))
        self.assertEqual(len(state.recognition_keys('reference',cfg)),20)
        self.assertEqual(len(state.recognition_keys('uniform_best',cfg)),22)
        self.assertEqual(len(state.q5_keys(cfg)),25)

class StateQueueTests(unittest.TestCase):
    def test_three_seed_10ep_frozen_pipeline_and_resume(self):
        torch.set_num_threads(1)
        stream=io.StringIO()
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(stream):
            root=Path(td);cfg=tiny_config();cfg['teacher_lambdas']=[.1/3,1/3,10/3,100/3]
            args=fixture(root,cfg)
            # The historical tiny fixture used random reference responses; make
            # real legal responses so paired state inputs can be compared.
            manifest,_=load_cache(args.cache)
            for seed in state.SEEDS:
                w=Path(args.source)/'work'/('seed%d'%seed)
                a=SimpleNamespace(**vars(args));a.work=str(w);a.seed=seed
                (w/'responses/reference.pt').unlink()
                compute.extract_response(a,w/'reference/last.pt',{},w/'responses/reference.pt',
                    seed=cfg['response_seed']+6,allowed_pairs=manifest['pairs']['train'])
            source=root/'source';args.work=str(source/'work')
            for seed in state.SEEDS:
                args.seed=seed
                a=SimpleNamespace(**vars(args));a.work=str(source/'work'/('seed%d'%seed))
                old.prepare(a)
                for fold in range(1,6):ct.teacher(a,fold)
            (source/'trustcpf').mkdir()
            for name in wf.CORE:shutil.copy2(ROOT/'trustcpf'/name,source/'trustcpf'/name)
            before={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
            args.source=str(source);args.work=str(root/'work')
            wf.setup(args)
            for seed in state.SEEDS:
                args.seed=seed;wf.run_seed(args)
                w=Path(args.work)/('seed%d'%seed)
                tc=read_json(w/'runs/uniform/training_contract.json')
                self.assertEqual(tc['epochs_completed'],10)
                self.assertIn(tc['selected_epoch'],range(11))
                p=load_torch(w/'policies.pt')
                for name in state.STATES:
                    self.assertEqual(set(p['states'][name]),set(state.recognition_keys(name,cfg)))
                ref=load_torch(w/'responses/reference.pt');cur=load_torch(w/'responses/uniform_best.pt')
                self.assertEqual(ref['support_indices'],cur['support_indices']);self.assertEqual(ref['seed'],cur['seed'])
                self.assertNotEqual(cur['seed'],load_torch(w/'responses/uniform_best_repeat.pt')['seed'])
            status=wf.report(args)
            self.assertTrue(status['complete'],status)
            self.assertEqual(status['training_curve_rows'],33)
            self.assertEqual(status['recognition_rows'],432);self.assertEqual(status['q5_rows'],54)
            w=Path(args.work)/'seed3353'
            model=align.restore(*load_cache(args.cache)[1:2],load_torch(w/'inputs/bank.pt'),
                read_json(w/'recipe.json'),'cpu',w/'runs/uniform/best.pt')
            # Check the new cached evaluator against the original, all metrics.
            pi=load_torch(w/'policies.pt')['states']['uniform_best']['pooled_absolute']['pi'].float()
            recipe=read_json(w/'recipe.json')
            expected=evaluation.evaluate_model(model,args.cache,recipe,manifest,pi,'val')
            got=read_json(w/'recognition/uniform_best/pooled_absolute/bank1701/metrics.json')['val']
            def close(a,b):
                if isinstance(a,dict):
                    self.assertEqual(set(a),set(b))
                    for k in a:close(a[k],b[k])
                elif a is None:self.assertIsNone(b)
                else:self.assertAlmostEqual(a,b,places=9)
            close(expected,got)
            file=w/'recognition/uniform_best/pooled_absolute/bank1701'
            entry=read_json(file/'metrics.json')
            for world in ('cw','ow'):
                for k in ('bias','temperature'):
                    self.assertEqual(entry['val'][world]['calibrated'][k],entry['test'][world]['calibrated'][k])
            for phase in ('val','test'):
                with np.load(file/(phase+'_predictions.npz')) as z:
                    for world in ('cw','ow'):np.testing.assert_allclose(np.exp(z[world+'_logprob']).sum(1),1,atol=2e-6)
            mtimes={str(p):p.stat().st_mtime_ns for p in Path(args.work).rglob('*.pt')}
            wf.setup(args);wf.run_seed(args)
            self.assertEqual(mtimes,{str(p):p.stat().st_mtime_ns for p in Path(args.work).rglob('*.pt')})
            self.assertEqual(before,{p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()})
            # Partial stage restart reuses immutable completed predictions.
            (Path(args.work)/'progress/seed3353/recognition.json').unlink()
            wf.run_seed(args)
            self.assertEqual(mtimes,{str(p):p.stat().st_mtime_ns for p in Path(args.work).rglob('*.pt')})
            write_json(ROOT/'validation/smoke_status.json',status)
            write_json(ROOT/'validation/smoke_training.json',[read_json(Path(args.work)/('seed%d/runs/uniform/training_contract.json'%s)) for s in state.SEEDS])
            write_json(ROOT/'validation/smoke_aggregates.json',read_json(Path(args.work)/'aggregates.json'))
            # Package the actual completed tiny run through the production packer.
            package=root/'pack'/ROOT.name
            (package/'scripts').mkdir(parents=True)
            shutil.copy2(ROOT/'scripts/pack_results.py',package/'scripts/pack_results.py')
            env=os.environ.copy();env['CPF_WORK']=args.work
            proc=subprocess.run([sys.executable,str(package/'scripts/pack_results.py')],
                env=env,capture_output=True,text=True,check=True)
            archive=package.parent/(ROOT.name+'_results.tar.gz')
            expected_hash=Path(str(archive)+'.sha256').read_text().split()[0]
            self.assertEqual(sha(archive),expected_hash)
            with tarfile.open(archive) as tf:
                members=tf.getmembers()
                self.assertTrue(all(m.name.isascii() and '..' not in Path(m.name).parts and not Path(m.name).is_absolute() for m in members))
                restored=root/'restored';tf.extractall(restored)
            restored_work=restored/ROOT.name/'work'
            for p in Path(args.work).rglob('*'):
                if p.is_file() and not p.name.endswith(('.lock','.writing','.pid')):
                    self.assertEqual(sha(p),sha(restored_work/p.relative_to(Path(args.work))))
            write_json(ROOT/'validation/package_roundtrip.json',dict(passed=True,files=len(members),
                ascii_member_paths=True,all_work_files_verified=True,archive_bytes=archive.stat().st_size))
            (ROOT/'validation/smoke.log').write_text(stream.getvalue(),encoding='utf8')

if __name__=='__main__':unittest.main()

