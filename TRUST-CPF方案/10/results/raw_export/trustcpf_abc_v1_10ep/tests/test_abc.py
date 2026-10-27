import copy
import itertools
import math
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
from test_competition import fixture, tiny_config
from trustcpf.common import ROOT, read_json, write_json, load_torch, save_torch, sha
from trustcpf import abc, abc_measure, abc_train, abc_formal, abc_workflow as wf
from trustcpf import compact, align, response, state_compute, state_eval, competition_workflow as cw
from trustcpf import competition_training as ct
from trustcpf.data import load_cache


def stage_fixture(root):
    """Small real tensors and training, with Stage's saved-file contract."""
    cfg=dict(tiny_config(), risk_images_per_group=2, q4_images_per_group=1, stop_images_per_group=1)
    a=fixture(root,cfg); _,text=load_cache(a.cache)
    source=Path(a.source)
    for seed in abc.SEEDS:
        w=source/('work/seed%d'%seed)
        q=SimpleNamespace(**dict(vars(a),seed=seed,work=str(w)))
        state_compute.extract_response(q,w/'reference/last.pt',None,w/'responses/reference_real.pt',
            seed=cfg['response_seed']+6,allowed_pairs=load_cache(a.cache)[0]['pairs']['train'])
        shutil.copy2(w/'responses/reference_real.pt',w/'responses/reference.pt')
    stage=root/'stage'; xs=[];ys=[]
    for seed in abc.SEEDS:
        w=stage/('work/seed%d'%seed);q=SimpleNamespace(**dict(vars(a),seed=seed,work=str(w)))
        cw.prepare(q)
        for j in range(1,6):
            split=read_json(w/('splits/fold%d.json'%j)); split['stop']={}
            for group in ('seen','unseen'):
                used=set(split['measure'][group]+split['evaluate'].get(group,[]))
                split['stop'][group]=sorted(set(split[group+'_pool'])-used)[:1]
            write_json(w/('splits/fold%d.json'%j),split)
            ct.teacher(q,j)
            cp=w/('teachers/%d/best.pt'%j);shutil.copy2(cp.with_name('last.pt'),cp)
            r=load_torch(w/('responses/fold%d.pt'%j))
            state_compute.extract_response(q,cp,r,w/('responses/fold%d_repeat.pt'%j),
                seed=r['seed']+cfg['response_repeat_offset'],allowed_ids=split['train_ids'],allowed_pairs=split['inner_pairs'])
            y=load_torch(w/('measurements/fold%d.pt'%j))
            for k in ('target','se2','blocks'):y[k]=y[k][...,0:1]
            y['targets']=['unll'];save_torch(w/('measurements/fold%d.pt'%j),y)
            for p in (w/('measurements/fold%d'%j)).glob('block*.pt'):
                block=load_torch(p);block['checkpoint_sha256']=sha(cp);save_torch(p,block)
            if j<4:xs.append(r['features'].double().numpy());ys.append(y['target'].flatten())
        dest=w/'runs/uniform/best.pt';dest.parent.mkdir(parents=True);shutil.copy2(w/'reference/last.pt',dest)
    types=load_torch(stage/'work/seed3351/inputs/bank.pt')['types'].numpy()
    save_torch(stage/'work/utility/matched.pt',compact.fit_ridge(xs,ys,types,9*cfg['abc_lam']))
    write_json(stage/'work/status.json',dict(complete=True,synthetic_fixture=True))
    write_json(stage/'work/progress/global/identity.json',dict(cache_manifest=sha(Path(a.cache)/'features_manifest.json')))
    (stage/'trustcpf').mkdir()
    for name in wf.CORE:shutil.copy2(ROOT/'trustcpf'/name,stage/'trustcpf'/name)
    return SimpleNamespace(**dict(vars(a),source=str(stage),work=str(root/'abc_work')))


class FormulaTests(unittest.TestCase):
    def test_mixture_counterfactual_equals_risk_gradient(self):
        p0,p1,q=.09,.61,.37
        for m in (1,2,16):
            def risk(p):
                return -sum(math.comb(m,k)*p**k*(1-p)**(m-k)*math.log((k*p1+(m-k)*p0)/m) for k in range(m+1))
            target=0.
            for k in range(m):
                aux=torch.tensor([math.log(p1)]*k+[math.log(p0)]*(m-1-k),dtype=torch.double).reshape(m-1,1)
                effect=abc.mixture_effect(torch.tensor([[math.log(p0)]],dtype=torch.double),torch.tensor([[math.log(p1)]],dtype=torch.double),aux)
                target+=math.comb(m-1,k)*q**k*(1-q)**(m-1-k)*float(effect)
            self.assertAlmostEqual(target,-(risk(q+1e-5)-risk(q-1e-5))/2e-5,places=8)
        extreme=abc.mixture_effect(torch.full((2,3),-900.),torch.full((2,3),-899.),torch.full((15,3),-901.))
        self.assertTrue(torch.isfinite(extreme).all())

    def test_context_nesting_weights_and_budget(self):
        rng=np.random.RandomState(8);types=np.repeat(np.arange(3),8)
        xs=[rng.randn(24,8) for _ in range(9)];ys=[rng.randn(24) for _ in range(9)]
        global_model=compact.fit_ridge(xs,ys,types,30.)
        weights=[np.array([.2,.8]) for _ in xs]
        c=abc.fit_context(xs,[np.stack([x,x]) for x in xs],[np.stack([y,y]) for y in ys],weights,types,global_model,30.)
        for x in xs:
            np.testing.assert_allclose(abc.predict_context(c,x,np.stack([x,x]),types),
                np.broadcast_to(compact.predict(global_model,x,types),(2,24)),atol=1e-12)
        self.assertEqual(float(np.abs(c['beta'].reshape(3,17)[:,9:]).max()),0.)
        effects=torch.tensor(rng.randn(3,24,8));routes=torch.tensor([0,1,0,1,0,0,0,0]);seen=torch.tensor([1,1,1,1,0,0,0,0],dtype=torch.bool)
        y=abc.conditional_targets(effects,routes,seen)
        np.testing.assert_allclose(y['weights'],[1,0]);self.assertTrue(torch.equal(y['context'][1],torch.zeros(24,dtype=torch.double)))
        for eta in (0,.25,.5,1):
            for scale in (None,np.array([1.,.5,2.])):
                pi=abc.score_policy(np.stack(ys[:2]),types,scale,eta)
                for t in range(3):torch.testing.assert_close(pi[:,types==t].sum(-1),torch.full((2,),4.,dtype=torch.double),rtol=0,atol=1e-9)
                self.assertTrue(((pi>=0)&(pi<=1)).all())

    def test_conditional_statistics_and_text_partition(self):
        torch.manual_seed(4);off=torch.randn(4,6,5);on=off+.1*torch.randn_like(off)
        truth=torch.arange(6)%5;w=torch.ones(6)/6;mask=torch.tensor([1,1,1,0,0,0],dtype=torch.bool)
        value=abc_measure.block_statistics(off,on,truth,w,mask)
        ref=response.response_statistics(off[:,mask],on[:,mask],truth[mask],w[mask],torch.ones(3,dtype=torch.bool))
        torch.testing.assert_close(value,ref[:4].double(),rtol=1e-6,atol=1e-8)
        self.assertIsNone(abc_measure.block_statistics(off,on,truth,w,torch.zeros(6,dtype=torch.bool)))
        x=torch.randn(8,5);groups=abc.object_partition(x,4)['groups']
        self.assertEqual(sorted(groups[0]+groups[1]),list(range(4)));self.assertEqual(len(groups[0]),2)
        self.assertEqual(abc.route(torch.zeros(2,4),groups).tolist(),[0,0])


class IntegrationTests(unittest.TestCase):
    def test_complete_three_seed_ten_epoch_pipeline(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as td:
            a=stage_fixture(Path(td));cfg=read_json(a.config);source=Path(a.source)
            source_hash={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
            # Setup writes local environment metadata: restore it after this test.
            generated={p:p.read_bytes() if p.exists() else None for p in (ROOT/'server_env.sh',ROOT/'setup.json')}
            try:
                wf.setup(a)
                for seed in abc.SEEDS:
                    a.seed=seed;wf.prepare_seed(a)
                    w=Path(a.work)/('seed%d'%seed)
                    last=load_torch(w/'runs/uniform/last.pt');self.assertEqual(last['epoch'],10);self.assertEqual(len(last['history']),11)
                    split=read_json(w/'splits/fold4.json')
                    evaluate=set(sum(split['evaluate'].values(),[]))
                    used=set(split['train_ids']+sum(split['measure'].values(),[])+sum(split['stop'].values(),[]))
                    self.assertFalse(evaluate&used)
                    for j in range(1,6):
                        y=load_torch(w/('effects/fold%d.pt'%j));r=load_torch(w/('context/fold%d.pt'%j))
                        self.assertTrue(torch.isfinite(r['context']).all())
                        for g,e,k in torch.nonzero(r['empty_blocks']).tolist():
                            torch.testing.assert_close(r['context'][g,e,4*k:4*k+4],r['features'][e,4*k:4*k+4],rtol=0,atol=0)
                        torch.testing.assert_close((y['single']['context']*torch.tensor(y['single']['weights'])[:,None]).sum(0),y['single']['target'])
                wf.fit(a)
                for seed in abc.SEEDS:a.seed=seed;wf.q4_seed(a)
                wf.select(a)
                for seed in abc.SEEDS:a.seed=seed;wf.evaluate_seed(a)
                status=wf.report(a);self.assertTrue(status['complete'],status)
                self.assertEqual(status['completed_stages'],36);self.assertEqual(status['training_curve_rows'],33)
                self.assertEqual(status['recognition_rows'],3*abc.formal_count(cfg)*2*4)
                self.assertEqual(source_hash,{str(p):sha(p) for p in source.rglob('*') if p.is_file()})
                before={str(p):sha(p) for p in Path(a.work).rglob('*.pt')}
                wf.prepare_seed(a);wf.fit(a);wf.q4_seed(a);wf.select(a);wf.evaluate_seed(a)
                self.assertEqual(before,{str(p):sha(p) for p in Path(a.work).rglob('*.pt')})
                # Formal global policy reproduces the original evaluator, including saved logits.
                b=copy.copy(a);b.work=str(Path(a.work)/'seed3353')
                w,c,m,t,bank=align.context(b);model=align.restore(t,bank,c,b.device,w/'runs/uniform/best.pt')
                pi=torch.full((len(bank['types']),),.5);groups=read_json(Path(a.work)/'groups.json')['groups']
                c=dict(c,readout='probability');prepared=state_eval.prepare_phase(model,b.cache,'val',c)
                old=state_eval.evaluate_cached(model,b.cache,c,m,pi,'val',prepared=prepared,save=Path(td)/'old')
                new=abc_formal.evaluate_routed(model,b.cache,c,m,pi[None],'val',groups,prepared=prepared,save=Path(td)/'new')
                self.assertEqual(old,new)
                with np.load(Path(td)/'old/val_predictions.npz') as x,np.load(Path(td)/'new/val_predictions.npz') as y:
                    for k in ('cw_logprob','ow_logprob'):np.testing.assert_array_equal(x[k],y[k])
                base=prepared[0][0];routes=abc.route(base['obj'],groups)
                masks=torch.bernoulli(torch.full((2,2,len(pi)),.5));joint=abc_formal.routed_stack(model,base,masks,routes)
                for i,g in enumerate(routes):
                    sub={k:v[i:i+1] for k,v in base.items()}
                    for j in range(2):torch.testing.assert_close(joint[j,i],model.score(sub,masks[g,j])[0][0],rtol=1e-5,atol=1e-6)
                # Force a stage rerun: completed probability files are reused, not resampled.
                mark=Path(a.work)/'progress/seed3353/q5.json';mark.unlink();wf.evaluate_seed(a)
                self.assertTrue(wf.report(a)['complete'])
                # Exercise the actual packer against the complete synthetic work tree.
                release=Path(td)/'pack/trustcpf_abc_v1_10ep';(release/'scripts').mkdir(parents=True)
                shutil.copy2(ROOT/'scripts/pack_results.py',release/'scripts/pack_results.py')
                env=dict(os.environ,CPF_WORK=a.work)
                packed=subprocess.run([sys.executable,str(release/'scripts/pack_results.py')],env=env,capture_output=True,text=True)
                self.assertEqual(packed.returncode,0,packed.stdout+packed.stderr)
                archive=release.parent/'trustcpf_abc_v1_10ep_results.tar.gz'
                expected=Path(str(archive)+'.sha256').read_bytes()
                self.assertNotIn(b'\r',expected);self.assertEqual(expected.decode().split()[0],sha(archive))
                with tarfile.open(archive) as tar:
                    members=tar.getmembers()
                    self.assertTrue(all(m.name.isascii() and not m.name.startswith('/') and '..' not in m.name.split('/') for m in members))
                    item=tar.extractfile('trustcpf_abc_v1_10ep/work/status.json').read()
                    self.assertEqual(item,(Path(a.work)/'status.json').read_bytes())
                # Preserve compact machine-readable validation evidence outside the temporary data.
                write_json(ROOT/'validation/synthetic_pipeline.json',dict(status=status,source_unchanged=True,
                    resume_identical=True,formal_global_matches_legacy=True,routed_manual_matches=True,
                    archive_roundtrip=True,archive_members=len(members),
                    synthetic=True,graphs=cfg['graphs'],seeds=list(abc.SEEDS),training_epochs=10))
            finally:
                for p,data in generated.items():
                    if data is None:p.unlink(missing_ok=True)
                    else:p.write_bytes(data)


class CudaTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_ten_epoch_resume_and_frozen_backbone_on_cuda(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as td:
            root=Path(td);a=fixture(root,tiny_config());source=Path(a.source)/'work/seed3351'
            for name in ('full','resume'):
                dest=root/name;shutil.copytree(source,dest)
            full=SimpleNamespace(**dict(vars(a),work=str(root/'full'),device='cuda'))
            resume=SimpleNamespace(**dict(vars(a),work=str(root/'resume'),device='cuda'))
            abc_train.uniform(full);abc_train.uniform(resume,until_epoch=4);abc_train.uniform(resume)
            x=load_torch(root/'full/runs/uniform/last.pt');y=load_torch(root/'resume/runs/uniform/last.pt')
            for kind in ('model','best_model'):
                for key in x[kind]:torch.testing.assert_close(x[kind][key],y[kind][key],rtol=0,atol=0)
            self.assertEqual(x['best_epoch'],y['best_epoch'])
            self.assertEqual([v.get('loss') for v in x['history']],[v.get('loss') for v in y['history']])
            # Production 16-graph route/mixture path, including world normalization and VAL calibration transfer.
            w,c,m,t,bank=align.context(full);c=dict(c,graphs=16,readout='probability')
            model=align.restore(t,bank,c,'cuda',w/'runs/uniform/best.pt')
            groups=abc.object_partition(bank['anchors'],len(m['labels']['attrs']))['groups']
            pi=torch.tensor(np.random.RandomState(9).uniform(.1,.9,(2,len(bank['types']))),dtype=torch.float32)
            val=abc_formal.evaluate_routed(model,full.cache,c,m,pi,'val',groups,save=root/'cuda_val')
            test=abc_formal.evaluate_routed(model,full.cache,c,m,pi,'test',groups,calibration=val,save=root/'cuda_test')
            for world in ('cw','ow'):
                for k in ('bias','temperature'):self.assertEqual(test[world]['calibrated'][k],val[world]['calibrated'][k])
            with np.load(root/'cuda_val/val_predictions.npz') as z:
                self.assertEqual(z['graph_masks'].shape,(2,16,24))
                for world in ('cw','ow'):np.testing.assert_allclose(np.exp(z[world+'_logprob']).sum(-1),1.,rtol=1e-6,atol=1e-6)
            write_json(ROOT/'validation/cuda_resume.json',dict(device=torch.cuda.get_device_name(),torch=torch.__version__,
                uninterrupted_epochs=10,interrupted_at=4,resumed_to=10,weights_identical=True,
                selected_epoch=x['best_epoch'],frozen_backbone_identical=True,context_graphs=16,
                world_probabilities_normalized=True,val_calibration_reused=True,synthetic=True))


if __name__=='__main__':unittest.main(verbosity=2)
