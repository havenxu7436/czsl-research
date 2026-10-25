"""Local CPU analysis of archived U-NLL targets. No recognition re-training."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
ROOT=BASE/'trustcpf_r5_v1_10ep_results_20261024_210943/trustcpf_r5_v1_10ep'
sys.path.insert(0,str(ROOT))
from trustcpf.common import load_torch
from trustcpf import compact,r5,align
torch.set_num_threads(2)
seeds=(3351,3352,3353)
lambdas=np.array([.1,1,10,100])/3
data={}
for seed in seeds:
    w=ROOT/'work'/f'seed{seed}'
    types=np.asarray(load_torch(w/'policies.pt')['types'],dtype=int)
    xs=[];ys=[];hs=[]
    for fold in range(1,6):
        response=load_torch(w/f'responses/fold{fold}.pt')
        raw=load_torch(w/f'measurements/fold{fold}.pt')
        xs.append(response['features'].double().numpy())
        ys.append(np.asarray(raw['target'])[:,list(raw['targets']).index('unll')])
        hs.append(response['checkpoint_sha256'])
    xr=load_torch(w/'responses/reference.pt')['features'].double().numpy()
    data[seed]=dict(x=xs,y=ys,xr=xr,types=types,teacher_sha=hs)
assert all(np.array_equal(data[s]['types'],types) for s in seeds)

def train(xs,ys,lam,kind):
    m=compact.fit_ridge(xs,ys,types,len(xs)*lam)
    return m if kind=='absolute' else r5.fit_relative(xs,ys,types,len(xs)*lam,{k:m[k] for k in ('center','spread','scales')})

def fitset(mode,kind,lam):
    if mode=='pooled':
        model=train([x for s in seeds for x in data[s]['x'][:3]],
                    [y for s in seeds for y in data[s]['y'][:3]],lam,kind)
        return {s:model for s in seeds}
    return {s:train(data[s]['x'][:3],data[s]['y'][:3],lam,kind) for s in seeds}

def score_q4(models):
    e=[]
    for s in seeds:
        p=compact.predict(models[s],data[s]['x'][3],types)
        y=data[s]['y'][3]
        e.append(np.mean([np.var((p[types==t]-y[types==t])/models[s]['scales'][t]) for t in range(3)]))
    return float(np.mean(e)),[float(a) for a in e]

def report_one(m,s):
    p=compact.predict(m,data[s]['x'][4],types);y=data[s]['y'][4]
    pr=compact.predict(m,data[s]['xr'],types)
    pi,_=align.policy(p,types,1)
    pio,_=align.policy(y,types,1)
    proxy=float(np.dot(y,pi.numpy()-.5))
    permproxies=[float(np.dot(y,pi.numpy()[align.permutation(types,k)]-.5)) for k in (7101,7102,7103,7104,7105)]
    return dict(seed=s,alpha=m['alpha'],**compact.point_metrics(p,y),
        centered=r5.centered_metrics(p,y,types),
        centered_mse=float(np.mean(np.concatenate([(p[types==t]-p[types==t].mean())-(y[types==t]-y[types==t].mean()) for t in range(3)])**2)),
        by_type=[dict(type=t,**compact.point_metrics(p[types==t],y[types==t])) for t in range(3)],
        input_change_rho=[compact.point_metrics(p[types==t],pr[types==t])['spearman'] for t in range(3)],
        linear_utility_gain=proxy,permuted_linear_gain=permproxies,
        linear_gain_minus_permutation=proxy-float(np.mean(permproxies)),
        measured_linear_gain=float(np.dot(y,pio.numpy()-.5)))
out=dict(note='Q5 local prediction and first-order utility surrogate only; no new recognition AUC.',
         lambda_grid=lambdas.tolist(),cells={},source_teacher_sha={s:data[s]['teacher_sha'] for s in seeds})
for mode in ('independent','pooled'):
    for kind in ('absolute','relative'):
        name=mode+'_'+kind
        trials=[]
        for lam in lambdas:
            models=fitset(mode,kind,lam)
            q4,per=score_q4(models)
            trials.append(dict(lam=float(lam),mean_q4=q4,q4_by_seed=per))
        best=min(trials,key=lambda a:(a['mean_q4'],-a['lam']))
        models=fitset(mode,kind,best['lam'])
        rows=[report_one(models[s],s) for s in seeds]
        out['cells'][name]=dict(selected_lambda=best['lam'],trials=trials,rows=rows)
        print(name,'lambda',best['lam'],'q4',best['mean_q4'],
              'Q5 rho',np.mean([r['spearman'] for r in rows]),
              'centered rho',np.mean([r['centered']['spearman'] for r in rows]),
              'centered r2',np.mean([r['centered']['r2'] for r in rows]),
              'proxy gains',[round(r['linear_utility_gain'],6) for r in rows],flush=True)
out['original_per_seed_absolute']=[report_one(load_torch(ROOT/'work'/f'seed{s}'/'utility/absolute/model.pt'),s) for s in seeds]
# Simple readouts use exactly the fitting teachers, and no Q5 target to choose policy.
for control in ('historical_edge_mean','direct_support_nll'):
    rows=[]
    for s in seeds:
        y=data[s]['y'][4]
        p=np.mean([data[ss]['y'][j] for ss in seeds for j in range(3)],0) if control=='historical_edge_mean' else data[s]['x'][4][:,0]
        pi,_=align.policy(p,types,1)
        perms=[float(np.dot(y,pi.numpy()[align.permutation(types,k)]-.5)) for k in (7101,7102,7103,7104,7105)]
        rows.append(dict(seed=s,**compact.point_metrics(p,y),centered=r5.centered_metrics(p,y,types),
            linear_utility_gain=float(np.dot(y,pi.numpy()-.5)),
            linear_gain_minus_permutation=float(np.dot(y,pi.numpy()-.5)-np.mean(perms))))
    out[control]=rows
    print(control,'Q5 rho',np.mean([r['spearman'] for r in rows]),'centered rho',np.mean([r['centered']['spearman'] for r in rows]),'proxy gains',[round(r['linear_utility_gain'],6) for r in rows])

def convert(v):
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,np.ndarray): return v.tolist()
    raise TypeError(type(v))
(OUT/'pooled_diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n',encoding='utf8')

