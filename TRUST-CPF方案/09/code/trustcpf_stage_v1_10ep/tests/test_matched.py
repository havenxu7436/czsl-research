import contextlib,copy,io,os,shutil,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from test_competition import fixture,tiny_config
from trustcpf import matched,matched_train as train,matched_workflow as wf
from trustcpf import state,state_workflow as sw,state_compute as sc,competition_workflow as cw,competition_training as ct
from trustcpf.common import ROOT,read_json,write_json,load_torch,sha
from trustcpf.data import load_cache

def source_state(root,cfg):
    args=fixture(root,cfg);manifest,_=load_cache(args.cache)
    for seed in state.SEEDS:
        w=Path(args.source)/'work'/('seed%d'%seed);a=SimpleNamespace(**vars(args));a.work=str(w);a.seed=seed
        (w/'responses/reference.pt').unlink()
        sc.extract_response(a,w/'reference/last.pt',{},w/'responses/reference.pt',seed=cfg['response_seed']+6,allowed_pairs=manifest['pairs']['train'])
    source=root/'r5';args.work=str(source/'work')
    for seed in state.SEEDS:
        a=SimpleNamespace(**vars(args));a.seed=seed;a.work=str(source/'work'/('seed%d'%seed));cw.prepare(a)
        for j in range(1,6):ct.teacher(a,j)
    (source/'trustcpf').mkdir()
    for name in sw.CORE:shutil.copy2(ROOT/'trustcpf'/name,source/'trustcpf'/name)
    args.source=str(source);args.work=str(root/'state/work');sw.setup(args)
    for seed in state.SEEDS:args.seed=seed;sw.run_seed(args)
    assert sw.report(args)['complete']
    source=root/'state';(source/'trustcpf').mkdir()
    for name in wf.CORE:shutil.copy2(ROOT/'trustcpf'/name,source/'trustcpf'/name)
    args.source=str(source);args.work=str(root/'stage/work')
    return args

class MatchedTests(unittest.TestCase):
    def test_stopping_images_and_reproducibility(self):
        cfg={'stop_seed':9401,'stop_images_per_group':2}
        s=dict(fold=5,train_ids=[0,1],seen_pool=list(range(2,16)),unseen_pool=list(range(16,30)),
            measure={'seen':[2,3],'unseen':[16,17]},evaluate={'seen':[4,5],'unseen':[18,19]})
        a=train.add_stopping_set(s,cfg);self.assertEqual(a,train.add_stopping_set(s,cfg))
        used=set(s['train_ids']+s['measure']['seen']+s['measure']['unseen']+s['evaluate']['seen']+s['evaluate']['unseen'])
        self.assertFalse(used&set(a['stop']['seen']+a['stop']['unseen']))
        self.assertNotIn('stop',s)
        with self.assertRaises(ValueError):train.add_stopping_set(s,dict(cfg,stop_images_per_group=100))

    def test_full_second_batch_and_epoch_resume(self):
        torch.set_num_threads(1);stream=io.StringIO()
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(stream):
            root=Path(td);cfg=tiny_config();cfg.update(stop_images_per_group=2,stage_epochs=10,risk_images_per_group=1)
            args=source_state(root,cfg);source=Path(args.source)
            before={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
            wf.setup(args)
            with self.assertRaises(FileNotFoundError):wf.fit(args)
            w=Path(args.work)/'seed3351';a=copy.copy(args);a.seed=3351;a.work=str(w)
            fresh=root/'fresh';shutil.copytree(w,fresh)
            train.train_teacher(a,1,until_epoch=3)
            self.assertEqual(load_torch(w/'teachers/1/last.pt')['epoch'],3)
            (w/'teachers/1/best.pt').unlink()
            train.train_teacher(a,1)
            b=copy.copy(a);b.work=str(fresh);train.train_teacher(b,1)
            resumed=load_torch(w/'teachers/1/last.pt');direct=load_torch(fresh/'teachers/1/last.pt')
            self.assertEqual(resumed['best_epoch'],direct['best_epoch'])
            for k,v in resumed['model'].items():torch.testing.assert_close(v,direct['model'][k],rtol=0,atol=0)
            for seed in state.SEEDS:args.seed=seed;wf.train_seed(args)
            wf.fit(args)
            for seed in state.SEEDS:
                args.seed=seed;wf.evaluate_seed(args);w=Path(args.work)/('seed%d'%seed)
                self.assertEqual(sha(w/'runs/uniform/best.pt'),sha(source/('work/seed%d/runs/uniform/best.pt'%seed)))
                for j in range(1,6):
                    s=read_json(w/('splits/fold%d.json'%j));st=load_torch(w/('teachers/%d/last.pt'%j));best=load_torch(w/('teachers/%d/best.pt'%j))
                    self.assertEqual(st['epoch'],10);self.assertEqual(len(st['history']),11);self.assertEqual(st['train_ids'],s['train_ids'])
                    chosen=max(st['history'],key=lambda r:(r['stop']['AUC_native'],-r['epoch']))['epoch'];self.assertEqual(chosen,best['epoch'])
                    original=load_torch(w/('old/teachers/%d/last.pt'%j))['model']
                    for k,v in original.items():
                        if k.startswith('backbone.'):torch.testing.assert_close(v,best['model'][k],rtol=0,atol=0)
                    for suffix in ('','_repeat'):
                        r=load_torch(w/('responses/fold%d%s.pt'%(j,suffix)))
                        self.assertTrue(set(r['support_indices'])<=set(s['train_ids']))
                        self.assertEqual(r['checkpoint_sha256'],sha(w/('teachers/%d/best.pt'%j)))
                    block=load_torch(w/('measurements/fold%d/block00.pt'%j));sm=block['sample_seen']
                    expected=(block['on'][:,~sm,0]-block['off'][:,~sm,0]).mean(1)[:,None]
                    torch.testing.assert_close(expected,block['effects'],rtol=0,atol=1e-13)
                pol=load_torch(w/'policies.pt');self.assertEqual(set(pol['states']['uniform_best']),set(matched.formal_keys(cfg)))
                self.assertEqual(set(pol['q5']),set(matched.q5_keys(cfg)))
                for name in ('uniform','old_single','old_single_perm7101'):
                    original='uniform' if name=='uniform' else name.replace('old_single','pooled_absolute')
                    got=read_json(w/'recognition/uniform_best'/name/'bank1701/metrics.json')
                    prev=read_json(source/('work/seed%d/recognition/uniform_best'%seed)/original/'bank1701/metrics.json')
                    for phase in ('val','test'):
                        for world in ('cw','ow'):
                            self.assertEqual(got[phase][world],prev[phase][world])
            status=wf.report(args);self.assertTrue(status['complete'],status)
            self.assertEqual(status['completed_stages'],26);self.assertEqual(status['teacher_curve_rows'],165)
            self.assertEqual(status['recognition_rows'],264);self.assertEqual(status['q5_rows'],66)
            mtimes={str(p):p.stat().st_mtime_ns for p in Path(args.work).rglob('*.pt')}
            wf.setup(args);wf.train_seed(args);wf.fit(args);wf.evaluate_seed(args)
            self.assertEqual(mtimes,{str(p):p.stat().st_mtime_ns for p in Path(args.work).rglob('*.pt')})
            (Path(args.work)/'progress/seed3353/recognition.json').unlink();wf.evaluate_seed(args)
            self.assertEqual(mtimes,{str(p):p.stat().st_mtime_ns for p in Path(args.work).rglob('*.pt')})
            self.assertEqual(before,{p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()})
            package=root/'pack'/ROOT.name;(package/'scripts').mkdir(parents=True)
            shutil.copy2(ROOT/'scripts/pack_results.py',package/'scripts/pack_results.py')
            env=os.environ.copy();env['CPF_WORK']=args.work
            subprocess.run([sys.executable,str(package/'scripts/pack_results.py')],env=env,capture_output=True,text=True,check=True)
            archive=package.parent/(ROOT.name+'_results.tar.gz');self.assertEqual(sha(archive),Path(str(archive)+'.sha256').read_text().split()[0])
            with tarfile.open(archive) as tf:
                self.assertTrue(all(m.name.isascii() and '..' not in Path(m.name).parts for m in tf.getmembers()));tf.extractall(root/'unpacked')
            for p in Path(args.work).rglob('*'):
                if p.is_file() and not p.name.endswith(('.lock','.writing','.pid')):
                    self.assertEqual(sha(p),sha(root/'unpacked'/ROOT.name/'work'/p.relative_to(Path(args.work))))
            write_json(ROOT/'validation/matched_smoke_status.json',status)
            write_json(ROOT/'validation/matched_smoke_validation.json',dict(passed=True,teachers=15,epochs_each=10,
                epoch_resume_bitwise_equal=True,source_immutable=True,formal_checkpoints_unchanged=True,
                old_readout_matches_batch_one=True,package_roundtrip=True,completed_stages=26))
            (ROOT/'validation/matched_smoke.log').write_text(stream.getvalue(),encoding='utf8')

if __name__=='__main__':unittest.main()
