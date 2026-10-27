"""Teacher-averaged U-NLL ridge; a single selection per factorial cell."""
from pathlib import Path
import numpy as np
import torch
from .common import load_torch,save_torch,read_json,write_json
from . import compact,r5,align,response

SEEDS=(3351,3352,3353)
METHODS=('independent_absolute','pooled_absolute','pooled_relative')
CELLS=('independent_absolute','independent_relative','pooled_absolute','pooled_relative')
STATES=('reference','uniform_best')

def fit_one(xs,ys,types,lam,relative=False):
    m=compact.fit_ridge(xs,ys,types,len(xs)*lam)
    if relative:
        m=r5.fit_relative(xs,ys,types,len(xs)*lam,{k:m[k] for k in ('center','spread','scales')})
    beta=np.asarray(m['beta']).reshape(3,9)
    m.update(lam=float(lam),teachers=len(xs),objective_teacher_averaged=True,
        raw_slopes=m['scales'][:,None]*beta[:,1:]/m['spread'],
        raw_intercepts=m['scales']*(beta[:,0]-(beta[:,1:]*m['center']/m['spread']).sum(1)),
        feature_names=response.FEATURE_NAMES,types=np.asarray(types),edge_ids=np.arange(len(types)),
        fit_folds=[1,2,3],selection_fold=4,audit_fold=5)
    return m

def fit_global(root,cfg):
    root=Path(root);data={}
    for seed in SEEDS:
        w=root/('seed%d'%seed)
        ty=load_torch(w/'inputs/bank.pt')['types'].numpy()
        xs=[load_torch(w/('responses/fold%d.pt'%j))['features'].double().numpy() for j in range(1,6)]
        raws=[load_torch(w/('measurements/fold%d.pt'%j)) for j in range(1,6)]
        ys=[np.asarray(r['target'])[:,list(r['targets']).index('unll')] for r in raws]
        data[seed]=(xs,ys,ty)
    types=data[SEEDS[0]][2]
    if not all(np.array_equal(data[s][2],types) for s in SEEDS):
        raise ValueError('Teacher edge order/types differ')
    def models_for(cell,lam):
        pooled=cell.startswith('pooled');relative=cell.endswith('relative')
        if pooled:
            m=fit_one([x for s in SEEDS for x in data[s][0][:3]],
                      [y for s in SEEDS for y in data[s][1][:3]],types,lam,relative)
            m['teacher_ids']=[(s,j) for s in SEEDS for j in (1,2,3)]
            return {s:m for s in SEEDS}
        result={}
        for s in SEEDS:
            result[s]=fit_one(data[s][0][:3],data[s][1][:3],types,lam,relative)
            result[s]['teacher_ids']=[(s,j) for j in (1,2,3)]
        return result
    selections={};predictions={s:{} for s in SEEDS};audit=[]
    for cell in CELLS:
        trials=[]
        for lam in cfg['teacher_lambdas']:
            models=models_for(cell,lam);errs=[]
            for s in SEEDS:
                x,y,_=data[s];m=models[s];p=compact.predict(m,x[3],types)
                errs.append(float(np.mean([np.var((p[types==t]-y[3][types==t])/m['scales'][t]) for t in range(3)])))
            trials.append(dict(lam=float(lam),q4_error=float(np.mean(errs)),by_seed=errs))
        choice=min(trials,key=lambda v:(v['q4_error'],-v['lam']))
        selections[cell]=dict(selected_lambda=choice['lam'],trials=trials,tie_break='larger lambda')
        models=models_for(cell,choice['lam'])
        for s in SEEDS:
            m=models[s];m['cell']=cell
            dest=root/'utility'/cell/('shared.pt' if cell.startswith('pooled') else 'seed%d.pt'%s)
            if not cell.startswith('pooled') or s==SEEDS[0]:save_torch(dest,m)
            x,y,_=data[s];p=compact.predict(m,x[4],types)
            predictions[s][cell]=p
            pi,_=align.policy(p,types,cfg['fixed_strength'])
            proxy=float(y[4]@(pi.numpy()-.5))
            audit.append(dict(seed=s,cell=cell,lam=choice['lam'],alpha=m['alpha'],
                all=compact.point_metrics(p,y[4]),centered=r5.centered_metrics(p,y[4],types),
                by_type={str(t):compact.point_metrics(p[types==t],y[4][types==t]) for t in range(3)},
                linear_gain=proxy))
        print('SHARED_FIT',cell,'lambda',choice['lam'],'Q4',choice['q4_error'],flush=True)
    edge_mean=np.mean([data[s][1][j] for s in SEEDS for j in range(3)],0)
    save_torch(root/'utility/predictions.pt',dict(predictions=predictions,types=types,historical_edge_mean=edge_mean))
    write_json(root/'utility/selection.json',selections)
    write_json(root/'utility/audit.json',audit)

def predictor(root,cell,seed):
    file='shared.pt' if cell.startswith('pooled') else 'seed%d.pt'%seed
    return load_torch(Path(root)/'utility'/cell/file)

def family(means,types,cfg):
    out={'uniform':dict(pi=torch.full((len(types),),.5,dtype=torch.double),method='uniform',permutation_seed=None)}
    for name,mu in means.items():
        p,info=align.policy(mu,types,cfg['fixed_strength'])
        out[name]=dict(pi=p,mu=np.asarray(mu),method=name,permutation_seed=None,scale=info)
        for ps in cfg['permutation_seeds']:
            ix=align.permutation(types,ps)
            out[name+'_perm%d'%ps]=dict(pi=p[ix],mu=np.asarray(mu)[ix],method=name,permutation_seed=int(ps),
                permutation=ix,scale=info)
    return out

def recognition_keys(state,cfg):
    out=['uniform']
    for m in METHODS:out += [m]+[m+'_perm%d'%p for p in cfg['permutation_seeds']]
    out+=['pooled_absolute_cross_response']
    if state=='uniform_best':out+=['historical_edge_mean','direct_support_nll']
    return out

def q5_keys(cfg):
    out=['uniform']
    for m in METHODS+('measured',):out += [m]+[m+'_perm%d'%p for p in cfg['permutation_seeds']]
    return out

