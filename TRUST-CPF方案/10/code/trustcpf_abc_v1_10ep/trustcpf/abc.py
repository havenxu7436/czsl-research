"""Finite-mixture utility, context ridge and matched-budget policies."""
import itertools
from pathlib import Path
import numpy as np
import torch
from .common import load_torch, save_torch, read_json, write_json
from . import compact, align

SEEDS=(3351,3352,3353)
BASE_METHODS=('matched','amplitude','mix16','context')
CURVE_METHODS=('matched','amplitude','mix16','context','measured1','measured16','measured_context')

def object_partition(anchors,na):
    x=np.asarray(anchors[na:].double());x=x/np.linalg.norm(x,axis=1,keepdims=True)
    n=len(x);size=n//2
    if n<2:raise ValueError('Two object contexts require at least two objects')
    best=None
    for rest in itertools.combinations(range(1,n),size-1):
        left=(0,)+rest;right=tuple(i for i in range(n) if i not in left)
        score=sum(float(((x[list(g)]-x[list(g)].mean(0))**2).sum()) for g in (left,right))
        item=(score,left,right)
        if best is None or item<best:best=item
    return dict(groups=[list(best[1]),list(best[2])],objective=best[0],objects=n)

def route(obj_logits,groups):
    p=obj_logits.softmax(-1)
    return torch.stack([p[:,g].sum(-1) for g in groups],-1).argmax(-1)

def mixture_effect(off,on,aux):
    """[E,N] paired log probabilities; [M-1,N] independent auxiliary logs."""
    off,on,aux=[v.double() for v in (off,on,aux)]
    m=len(aux)+1
    if m==1:return on-off
    a=aux.logsumexp(0)
    return m*(torch.logaddexp(a[None],on)-torch.logaddexp(a[None],off))

def conditional_targets(effects,routes,sample_seen):
    """effects: K x E x N; supervision uses pseudo-unseen images only."""
    keep=~sample_seen.bool();n=int(keep.sum());weights=[];targets=[];blocks=[]
    for g in range(2):
        use=keep&(routes==g);weights.append(float(use.sum())/n)
        b=effects[:,:,use].mean(-1) if use.any() else effects.new_zeros(effects.shape[:2])
        blocks.append(b);targets.append(b.mean(0))
    global_blocks=effects[:,:,keep].mean(-1)
    targets=torch.stack(targets);blocks=torch.stack(blocks)
    torch.testing.assert_close((targets*targets.new_tensor(weights)[:,None]).sum(0),global_blocks.mean(0),atol=1e-11,rtol=1e-9)
    return dict(target=global_blocks.mean(0),blocks=global_blocks,
                context=targets,context_blocks=blocks,weights=np.array(weights))

def context_design(h,hg,types,norm,delta_scale):
    """G x E x 51; zero context difference stays zero."""
    h=np.asarray(h,dtype=np.float64);hg=np.asarray(hg,dtype=np.float64)
    global_x=(h-norm['center'])/norm['spread']
    diff=(hg-h[None])/delta_scale
    raw=np.concatenate([np.ones((*diff.shape[:2],1)),np.broadcast_to(global_x,diff.shape),diff],-1)
    return np.concatenate([raw*(types==t)[None,:,None] for t in range(3)],-1)

def fixed_scale(ys,types):
    return np.array([max(np.sqrt(np.mean([np.mean((y[types==t]-y[types==t].mean())**2) for y in ys])),1e-12) for t in range(3)])

def fit_context(xs,xgs,ys,weights,types,norm,alpha,swap=()):
    delta=np.stack([g-x[None] for x,g in zip(xs,xgs)])
    scale=np.sqrt(np.einsum('jg,jgek->k',np.stack(weights),delta**2)/(len(xs)*len(types)))
    scale=np.maximum(scale,1e-6)
    design=[];labels=[];mass=[]
    for j,(x,g,y,w) in enumerate(zip(xs,xgs,ys,weights)):
        z=context_design(x,g,types,norm,scale)
        if j in swap:
            # Only conditional inputs are swapped. Global input is group-invariant.
            z=z[::-1].copy()
        design.append(z.reshape(-1,51))
        labels.append((y/norm['scales'][types]).reshape(-1))
        mass.append(np.repeat(w,len(types)))
    z=np.concatenate(design);y=np.concatenate(labels);w=np.concatenate(mass)
    beta=np.linalg.solve(z.T@(w[:,None]*z)+alpha*np.eye(51),z.T@(w*y))
    return dict(norm,delta_scale=scale,beta=beta,alpha=float(alpha),dimension=51,swapped_teachers=list(swap))

def predict_context(model,h,hg,types):
    return (context_design(h,hg,types,model,model['delta_scale'])@model['beta'])*model['scales'][types]

def fit(root,cfg):
    root=Path(root);records=[];types=load_torch(root/'seed3351/inputs/bank.pt')['types'].numpy()
    for s in SEEDS:
        for j in (1,2,3):
            w=root/('seed%d'%s)
            r=load_torch(w/('context/fold%d.pt'%j));y=load_torch(w/('effects/fold%d.pt'%j))
            h2=load_torch(w/('responses/fold%d_repeat.pt'%j))['features'].double().numpy()
            records.append((r,y,h2))
    old=load_torch(root/'utility/source_matched.pt')
    norm={k:np.asarray(old[k],dtype=np.float64) for k in ('center','spread','scales')}
    xs=[r['features'].double().numpy() for r,y,h2 in records]
    xgs=[r['context'].double().numpy() for r,y,h2 in records]
    ys=[y['single']['target'].numpy() for r,y,h2 in records]
    ym=[y['mixture']['target'].numpy() for r,y,h2 in records]
    yc=[y['single']['context'].numpy() for r,y,h2 in records]
    weights=[y['single']['weights'] for r,y,h2 in records]
    alpha=len(xs)*cfg['abc_lam']
    matched=compact.fit_ridge(xs,ys,types,alpha,norm)
    np.testing.assert_allclose(matched['beta'],old['beta'],rtol=2e-5,atol=2e-8)
    mix=compact.fit_ridge(xs,ym,types,alpha,norm)
    avg=compact.fit_ridge([(x+h2)/2 for x,(_,_,h2) in zip(xs,records)],ys,types,alpha,norm)
    swap=sorted(np.random.RandomState(cfg['context_shuffle_seed']).choice(len(xs),5,replace=False).tolist())
    models=dict(matched=matched,mix16=mix,average_refit=avg,
        context=fit_context(xs,xgs,yc,weights,types,norm,alpha),
        context_shuffle=fit_context(xs,xgs,yc,weights,types,norm,alpha,swap),
        amplitude_scale=fixed_scale(ys,types),types=types,teacher_ids=[(s,j) for s in SEEDS for j in (1,2,3)])
    save_torch(root/'utility/models.pt',models)
    write_json(root/'utility/fit.json',dict(teachers=9,alpha=alpha,lambda_value=cfg['abc_lam'],
        dimensions=dict(matched=27,mix16=27,context=51,context_shuffle=51,average_refit=27),
        swapped_teacher_ids=[models['teacher_ids'][i] for i in swap],
        amplitude_scale=models['amplitude_scale'].tolist(),source_matched_reproduced=True))

def score_policy(mu,types,scale=None,eta=1.):
    mu=np.atleast_2d(np.asarray(mu,dtype=np.float64));out=[]
    for row in mu:
        if scale is None:p,_=align.policy(row,types,1.)
        else:
            p=torch.empty(len(types),dtype=torch.double)
            for t in range(3):
                ix=types==t;r=(row[ix]-row[ix].mean())/scale[t]
                lo,hi=float(r.min()-50),float(r.max()+50)
                for _ in range(100):
                    mid=(lo+hi)/2;q=1/(1+np.exp(np.clip(mid-r,-700,700)))
                    if q.sum()>len(q)/2:lo=mid
                    else:hi=mid
                p[ix]=torch.from_numpy(1/(1+np.exp(np.clip((lo+hi)/2-r,-700,700))))
        out.append(.5+float(eta)*(p-.5))
    return torch.stack(out)

def means(models,response,repeat=None,effects=None):
    t=models['types'];h=response['features'].double().numpy();hg=response['context'].double().numpy()
    out={k:compact.predict(models[k],h,t) for k in ('matched','mix16')}
    out['amplitude']=out['matched']
    for k in ('context','context_shuffle'):out[k]=predict_context(models[k],h,hg,t)
    if repeat is not None:
        avg=(h+repeat['features'].double().numpy())/2
        out['average_readout']=compact.predict(models['matched'],avg,t)
        out['average_refit']=compact.predict(models['average_refit'],avg,t)
    if effects is not None:
        out.update(measured1=effects['single']['target'].numpy(),measured16=effects['mixture']['target'].numpy(),
                   measured_context=effects['single']['context'].numpy())
    return out

def add_family(out,key,method,mu,eta,models,cfg,permutations=True):
    types=models['types'];scale=models['amplitude_scale'] if method=='amplitude' else None
    pi=score_policy(mu,types,scale,eta)
    out[key]=dict(pi=pi,mu=np.atleast_2d(mu),method=method,eta=float(eta),permutation_seed=None)
    if permutations:
        for ps in cfg['permutation_seeds']:
            ix=align.permutation(types,ps)
            out[key+'_perm%d'%ps]=dict(pi=pi[:,ix],mu=np.atleast_2d(mu)[:,ix],
                method=method,eta=float(eta),permutation_seed=ps,permutation=ix)

def eta_key(method,eta):
    return method+'_e%03d'%round(100*eta)

def curves(models,response,repeat,effects,cfg,selected_eta=None):
    values=means(models,response,repeat,effects)
    out={'uniform':dict(pi=torch.full((1,len(models['types'])),.5,dtype=torch.double),
        mu=np.zeros((1,len(models['types']))),method='uniform',eta=0.,permutation_seed=None)}
    for method in CURVE_METHODS:
        for eta in cfg['amplitudes']:
            if eta:add_family(out,eta_key(method,eta),method,values[method],eta,models,cfg)
    for method in ('average_readout','average_refit'):
        add_family(out,method,method,values[method],1.,models,cfg)
    add_family(out,'context_shuffle','context_shuffle',values['context_shuffle'],1.,models,cfg,False)
    if selected_eta is not None:
        add_family(out,'A_selected','amplitude',values['amplitude'],selected_eta,models,cfg)
    return out

def formal(models,response,cfg,eta):
    values=means(models,response)
    out={'uniform':dict(pi=torch.full((1,len(models['types'])),.5,dtype=torch.double),
        mu=np.zeros((1,len(models['types']))),method='uniform',eta=0.,permutation_seed=None)}
    for name in BASE_METHODS:add_family(out,name,name,values[name],eta if name=='amplitude' else 1.,models,cfg)
    add_family(out,'context_shuffle','context_shuffle',values['context_shuffle'],1.,models,cfg,False)
    return out

def curve_count(cfg,selected=False):
    p=1+len(cfg['permutation_seeds'])
    return 2+(len(CURVE_METHODS)*(len(cfg['amplitudes'])-1)+2+int(selected))*p

def formal_count(cfg):return 2+len(BASE_METHODS)*(1+len(cfg['permutation_seeds']))
