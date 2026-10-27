"""One Uniform trajectory, paired response banks, Q5 and frozen recognition."""
import math
import time
from pathlib import Path
import numpy as np
import torch
from .common import load_torch,save_torch,read_json,write_json,sha
from . import state,align,engine,compact,response,competition as cmp,competition_training as ct
from . import state_eval
from .data import FeatureDataset
from .folds import pair_ids
from .metrics import native_curve,exact_curve

def checkpoint(work,name):
    return Path(work)/('reference/last.pt' if name=='reference' else 'runs/uniform/best.pt')

def train_uniform(args):
    work,cfg,manifest,text,bank=align.context(args)
    cfg=dict(cfg,variant='uniform',readout='probability')
    out=work/'runs/uniform';out.mkdir(parents=True,exist_ok=True)
    pi=torch.full((len(bank['types']),),.5)
    if not (out/'epoch0.json').exists():
        model=align.restore(text,bank,cfg,args.device,work/'reference/last.pt')
        val=state_eval.evaluate_cached(model,args.cache,cfg,manifest,pi,'val')
        write_json(out/'epoch0.json',dict(epoch=0,val=val))
        del model
    if (out/'best.pt').exists() and not (out/'last.pt').exists():
        (out/'best.pt').unlink()
    engine.train_model(args,cfg,text,bank,manifest,manifest['pairs']['train'],out,
                       pi=pi,initial=work/'reference/last.pt',select=True)
    last=load_torch(out/'last.pt');best=load_torch(out/'best.pt')
    original=load_torch(work/'reference/last.pt')['model']
    assert last['epoch']==cfg['adapt_epochs']==10 and len(last['history'])==10
    for name in ('last','best'):
        values=last['model'] if name=='last' else best['model']
        assert all(torch.equal(v,values[k]) for k,v in original.items() if k.startswith('backbone.'))
    curve=[read_json(out/'epoch0.json')]+last['history']
    chosen=max(curve,key=lambda r:(r['val']['cw']['AUC_native'],-r['epoch']))['epoch']
    assert chosen==best['epoch']
    write_json(out/'training_contract.json',dict(epochs_completed=10,selected_epoch=chosen,
        selection='VAL CW native AUC; first maximum among epoch 0..10',
        initial_sha256=sha(work/'reference/last.pt'),best_sha256=sha(out/'best.pt'),
        last_sha256=sha(out/'last.pt'),backbone_identical=True,
        training='new Uniform trajectory; source Uniform outputs are not imported'))
    print('UNIFORM_TRAIN_COMPLETE',args.seed,'last_epoch',last['epoch'],'best_epoch',chosen,flush=True)

@torch.no_grad()
def extract_response(args,checkpoint_path,template,output,seed=None,allowed_ids=None,allowed_pairs=None):
    work,cfg,manifest,text,bank=align.context(args)
    if Path(output).exists():return load_torch(output)
    rows=read_json(Path(args.cache)/'train_records.json')
    if seed is None:
        ids=list(template['support_indices']);weights=template['weights'];seed=int(template['seed'])
    else:
        if allowed_ids is None:allowed_ids=list(range(len(rows)))
        local_ids,weights=response.select_support([rows[i] for i in allowed_ids],allowed_pairs,
            manifest['labels'],cfg['response_images_per_pair'],seed)
        ids=[allowed_ids[i] for i in local_ids]
    ds=FeatureDataset(args.cache,'train');samples=[ds[i] for i in ids]
    attrs=torch.tensor([s[3] for s in samples]);objs=torch.tensor([s[4] for s in samples])
    candidates=torch.tensor(pair_ids(manifest['labels'],manifest['pairs']['train']),device=args.device)
    lookup={int(c):i for i,c in enumerate(candidates.cpu())}
    truth=torch.tensor([lookup[int(o)*len(manifest['labels']['attrs'])+int(a)] for a,o in zip(attrs,objs)])
    size=cfg['eval_batch_size']
    batches=[[torch.stack([v[k] for v in samples[start:start+size]]) for k in range(3)]
             for start in range(0,len(samples),size)]
    model=align.restore(text,bank,cfg,args.device,checkpoint_path)
    x,count=response.measure_responses(model,batches,truth,weights,attrs,objs,candidates,
        bank,cfg['response_backgrounds'],seed)
    value=dict(features=x,names=response.FEATURE_NAMES,support_indices=ids,weights=weights,
        train_pairs=allowed_pairs if allowed_pairs is not None else template['train_pairs'],
        checkpoint_sha256=sha(checkpoint_path),backgrounds=cfg['response_backgrounds'],
        seed=seed,head_forwards=count)
    save_torch(output,value)
    print('STATE_RESPONSE',args.seed,Path(output).name,len(ids),'images',count,'head forwards',flush=True)
    return value

def responses(args):
    work,cfg,manifest,text,bank=align.context(args)
    base=load_torch(work/'responses/reference.pt')
    assert all(k in base for k in ('support_indices','weights','seed','train_pairs'))
    extract_response(args,checkpoint(work,'uniform_best'),base,work/'responses/uniform_best.pt')
    repeat_seed=int(base['seed'])+cfg['response_repeat_offset']
    for name in state.STATES:
        extract_response(args,checkpoint(work,name),base,work/('responses/'+name+'_repeat.pt'),
            seed=repeat_seed,allowed_pairs=manifest['pairs']['train'])
    split=read_json(work/'splits/fold5.json');q5=load_torch(work/'responses/fold5.pt')
    extract_response(args,work/'teachers/5/last.pt',q5,work/'responses/fold5_repeat.pt',
        seed=int(q5['seed'])+cfg['response_repeat_offset'],allowed_ids=split['train_ids'],
        allowed_pairs=split['inner_pairs'])

def make_policies(args):
    work,cfg,manifest,text,bank=align.context(args);root=work.parent;types=bank['types'].numpy()
    pooled=load_torch(root/'utility/predictions.pt')
    h={name:load_torch(work/('responses/'+name+'.pt'))['features'].double().numpy() for name in state.STATES}
    pred={name:{m:compact.predict(state.predictor(root,m,args.seed),h[name],types) for m in state.METHODS}
          for name in state.STATES}
    policies={}
    for name in state.STATES:
        ps=state.family(pred[name],types,cfg)
        other='uniform_best' if name=='reference' else 'reference'
        mu=pred[other]['pooled_absolute'];pi,info=align.policy(mu,types,cfg['fixed_strength'])
        ps['pooled_absolute_cross_response']=dict(pi=pi,mu=mu,method='pooled_absolute',
            permutation_seed=None,response_state=other,scale=info)
        if name=='uniform_best':
            for control,mu in [('historical_edge_mean',pooled['historical_edge_mean']),
                               ('direct_support_nll',h[name][:,0])]:
                pi,info=align.policy(mu,types,cfg['fixed_strength'])
                ps[control]=dict(pi=pi,mu=mu,method=control,permutation_seed=None,scale=info)
        assert set(ps)==set(state.recognition_keys(name,cfg))
        policies[name]=ps
    raw=load_torch(work/'measurements/fold5.pt');y=raw['target'][:,list(raw['targets']).index('unll')]
    qmeans={m:pooled['predictions'][args.seed][m] for m in state.METHODS};qmeans['measured']=y
    qpolicies=state.family(qmeans,types,cfg)
    assert set(qpolicies)==set(state.q5_keys(cfg))
    save_torch(work/'policies.pt',dict(states=policies,q5=qpolicies,types=types,
        expected_edges=[int((types==t).sum())/2 for t in range(3)],strength=cfg['fixed_strength']))
    repeated=[];transfers=[]
    for name,source in [('reference','reference'),('uniform_best','uniform_best'),('q5','fold5')]:
        x0=load_torch(work/('responses/'+source+'.pt'))['features'].double().numpy()
        x1=load_torch(work/('responses/'+source+'_repeat.pt'))['features'].double().numpy()
        for m in state.METHODS:
            model=state.predictor(root,m,args.seed);p0=compact.predict(model,x0,types);p1=compact.predict(model,x1,types)
            avg=compact.predict(model,(x0+x1)/2,types)
            np.testing.assert_allclose(avg,(p0+p1)/2,atol=1e-10)
            pi0,_=align.policy(p0,types,cfg['fixed_strength']);pi1,_=align.policy(p1,types,cfg['fixed_strength'])
            for t in range(3):
                ix=types==t
                d=(p1-p0)[ix];d-=d.mean()
                repeated.append(dict(seed=args.seed,state=name,method=m,type=t,
                    prediction_rmse=float(np.sqrt(np.mean((p1[ix]-p0[ix])**2))),
                    centered_prediction_variance=float(np.dot(d,d)/2),
                    probability_mse=float((pi1[ix]-pi0[ix]).square().mean()),
                    rho=compact.point_metrics(p0[ix],p1[ix])['spearman']))
            if name=='q5':
                for dest in state.STATES:
                    for t in range(3):
                        ix=types==t
                        transfers.append(dict(seed=args.seed,method=m,state=dest,type=t,
                            input_change_rho=compact.point_metrics(p0[ix],pred[dest][m][ix])['spearman']))
    write_json(work/'responses/repeat_diagnostics.json',repeated)
    write_json(work/'responses/transfer.json',transfers)

@torch.no_grad()
def q5_audit(args):
    work,cfg,manifest,text,bank=align.context(args);policies=load_torch(work/'policies.pt')['q5']
    model=align.restore(text,bank,cfg,args.device,work/'teachers/5/last.pt')
    split=read_json(work/'splits/fold5.json')
    base,candidates,truth,seen=ct.prepared(args,cfg,manifest,model,split,'evaluate')
    sm=seen[truth];rows=[]
    for gs in cfg['audit_eval_seeds']:
        gen=torch.Generator().manual_seed(gs)
        uniforms=torch.rand(cfg['audit_graphs'],len(bank['types']),generator=gen).double()
        results={}
        for name,record in policies.items():
            path=work/'decision'/('bank%d'%gs)/(name+'.pt')
            masks=(uniforms<record['pi'][None]).float()
            if path.exists():value=load_torch(path)
            else:
                logs=torch.stack([ct.score(model,base,m.to(args.device),candidates,cfg['eval_batch_size']) for m in masks])
                sufficient=torch.stack([cmp.sufficient(z,truth,seen) for z in logs])
                per_graph=[cmp.diagnostics(s,sm,cfg['competition_temperature'],cfg['risk_pair_chunk']) for s in sufficient]
                mixlogs=logs.logsumexp(0)-math.log(len(logs))
                mix=cmp.diagnostics(cmp.sufficient(mixlogs,truth,seen),sm,cfg['competition_temperature'],cfg['risk_pair_chunk'])
                z=mixlogs.cpu().numpy();y=truth.cpu().numpy();seen_np=seen.cpu().numpy()
                mix['AUC_native']=native_curve(z,y,seen_np)['AUC']
                assert abs(exact_curve(z,y,seen_np)['AUC']-mix['AUC_exact'])<1e-8
                value=dict(mixture=mix,single={k:float(np.mean([r[k] for r in per_graph])) for k in per_graph[0]},
                    mixture_logprobs=mixlogs.cpu(),single_statistics=sufficient.cpu(),masks=masks,
                    truth=truth.cpu(),seen=seen.cpu(),pi=record['pi'],graph_seed=gs,
                    checkpoint_sha256=sha(work/'teachers/5/last.pt'))
                save_torch(path,value)
            results[name]=value
        for name,v in results.items():
            row=dict(seed=args.seed,policy=name,method=policies[name]['method'],
                     permutation_seed=policies[name]['permutation_seed'],bank=gs)
            for mode in ('single','mixture'):
                row.update({mode+'_'+k:x for k,x in v[mode].items()})
                row[mode+'_NLL_gain']=results['uniform'][mode]['unll']-v[mode]['unll']
                row[mode+'_AUC_gain']=v[mode]['AUC_exact']-results['uniform'][mode]['AUC_exact']
            rows.append(row)
        print('Q5_COMPLETE',args.seed,gs,len(results),'policies',flush=True)
    write_json(work/'decision/audit.json',dict(rows=rows,graphs=cfg['audit_graphs'],
        images={k:len(v) for k,v in split['evaluate'].items()},strength=cfg['fixed_strength']))

@torch.no_grad()
def recognition(args):
    work,cfg,manifest,text,bank=align.context(args);data=load_torch(work/'policies.pt')
    base_model=align.restore(text,bank,cfg,args.device,work/'reference/last.pt')
    # CPF is unchanged across the two head states; cache its outputs once.
    prepared={p:state_eval.prepare_phase(base_model,args.cache,p,cfg) for p in ('val','test')}
    del base_model
    for name in state.STATES:
        cp=checkpoint(work,name);model=align.restore(text,bank,cfg,args.device,cp)
        epoch=0 if name=='reference' else load_torch(cp)['epoch']
        for key in state.recognition_keys(name,cfg):
            record=data['states'][name][key];pi=record['pi'].float()
            for gs in cfg['eval_graph_seeds']:
                dest=work/'recognition'/name/key/('bank%d'%gs);file=dest/'metrics.json'
                if file.exists():continue
                evcfg=dict(cfg,graph_seed=gs,variant='uniform',readout='probability')
                align.synchronize(args.device);tic=time.perf_counter()
                val=state_eval.evaluate_cached(model,args.cache,evcfg,manifest,pi,'val',save=dest,prepared=prepared['val'])
                test=state_eval.evaluate_cached(model,args.cache,evcfg,manifest,pi,'test',calibration=val,save=dest,prepared=prepared['test'])
                align.synchronize(args.device)
                deployment=align.deployment_stats(model,pi,bank,evcfg)
                for t,v in deployment.items():
                    types=bank['types'].numpy();target=bank['target'].numpy()
                    v['expected_node_indegree']=[float(pi[(types==int(t))&(target==n)].sum()) for n in v['target_nodes']]
                value=dict(seed=args.seed,state=name,policy=key,method=record['method'],
                    permutation_seed=record['permutation_seed'],response_state=record.get('response_state',name),
                    graph_seed=gs,graphs=cfg['graphs'],epoch=epoch,strength=cfg['fixed_strength'] if key!='uniform' else 0.,
                    checkpoint_sha256=sha(cp),val=val,test=test,deployment=deployment,seconds=time.perf_counter()-tic)
                write_json(file,value)
                print('FROZEN_SCORE',args.seed,name,key,gs,'TEST_CW_AUC',test['cw']['AUC_native'],
                      'TEST_OW_AUC',test['ow']['AUC_native'],flush=True)

