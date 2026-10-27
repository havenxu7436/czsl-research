"""Evaluate all policies on fixed matched Q5 / imported formal recognizers."""
import math,time
from pathlib import Path
import numpy as np
import torch
from .common import read_json,write_json,load_torch,save_torch,sha
from . import align,matched,competition_training as ct,competition as cmp,state_eval
from .state_compute import checkpoint
from .metrics import native_curve,exact_curve

@torch.no_grad()
def q5_audit(args):
    work,cfg,manifest,text,bank=align.context(args);policies=load_torch(work/'policies.pt')['q5']
    model=align.restore(text,bank,cfg,args.device,work/'teachers/5/best.pt')
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
                    checkpoint_sha256=sha(work/'teachers/5/best.pt'))
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
    for name in ('uniform_best',):
        cp=checkpoint(work,name);model=align.restore(text,bank,cfg,args.device,cp)
        epoch=0 if name=='reference' else load_torch(cp)['epoch']
        for key in matched.formal_keys(cfg):
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

