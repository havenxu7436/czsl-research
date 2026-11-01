import numpy as np
import torch

def fit(records,types,lam=10/3):
    x=np.stack([r['features'] for r in records]);y=np.stack([r['target'] for r in records]);xg=np.stack([r['context'] for r in records])
    yg=np.stack([r['context_target'] for r in records]);mass=np.stack([r['context_mass'] for r in records]);J,E,F=x.shape
    center=x.mean((0,1));spread=np.maximum(x.std((0,1)),1e-6)
    scales=np.array([max(np.sqrt(np.mean([(row[types==t]-row[types==t].mean())**2 for row in y])),1e-12) for t in range(3)])
    delta=xg-x[:,None];ds=np.maximum(np.sqrt(np.einsum('jg,jgek->k',mass,delta**2)/(J*E)),1e-6)
    norm=dict(center=center,spread=spread,scales=scales,delta_scale=ds,types=types,alpha=J*lam,teachers=J)
    def design(h,hg=None):
        raw=np.concatenate([np.ones((len(h),1)),(h-center)/spread],-1)
        if hg is not None:raw=np.concatenate([raw,(hg-h)/ds],-1)
        return np.concatenate([raw*(types==t)[:,None] for t in range(3)],-1)
    z=np.concatenate([design(h) for h in x]);yy=(y/scales[types]).reshape(-1)
    matched=np.linalg.solve(z.T@z+J*lam*np.eye(27),z.T@yy)
    models=dict(matched=dict(norm,beta=matched))
    # Deterministic swaps of approximately half of fit teachers; labels stay attached.
    swap=set(np.random.RandomState(20261026).choice(J,(J+1)//2,replace=False).tolist())
    for name in ('context','context_shuffle'):
        zz=[];yy=[];ww=[]
        for j in range(J):
            for g in range(2):
                source=1-g if name=='context_shuffle' and j in swap else g
                zz.append(design(x[j],xg[j,source]));yy.append(yg[j,g]/scales[types]);ww.append(np.full(E,mass[j,g]))
        z=np.concatenate(zz);v=np.concatenate(yy);w=np.concatenate(ww)
        beta=np.linalg.solve(z.T@(w[:,None]*z)+J*lam*np.eye(51),z.T@(w*v))
        models[name]=dict(norm,beta=beta,swapped_teachers=sorted(swap) if name.endswith('shuffle') else [])
    return models

def predict(model,response):
    h=response['features'];types=model['types'];base=np.concatenate([np.ones((len(h),1)),(h-model['center'])/model['spread']],-1)
    if len(model['beta'])==27:raw=base[None]
    else:raw=np.concatenate([np.broadcast_to(base,(2,*base.shape)),(response['context']-h[None])/model['delta_scale']],-1)
    z=np.concatenate([raw*(types==t)[None,:,None] for t in range(3)],-1)
    return (z@model['beta'])*model['scales'][types]

def policy(mu,types):
    result=np.zeros_like(mu,dtype=np.float64)
    for g,row in enumerate(mu):
        for t in range(3):
            ix=types==t;v=row[ix];v=(v-v.mean())/max(v.std(),1e-12);lo,hi=v.min()-50,v.max()+50
            for _ in range(80):
                mid=(lo+hi)/2;p=1/(1+np.exp(np.clip(mid-v,-700,700)))
                if p.sum()>len(v)/2:lo=mid
                else:hi=mid
            result[g,ix]=1/(1+np.exp(np.clip((lo+hi)/2-v,-700,700)))
    return torch.from_numpy(result)

def policies(models,response,types,permutation_seeds):
    out={'uniform':torch.full((1,len(types)),.5,dtype=torch.float64)}
    for name in ('matched','context','context_shuffle'):
        pi=policy(predict(models[name],response),types);out[name]=pi
        if name=='context_shuffle':continue
        for ps in permutation_seeds:
            rng=np.random.RandomState(ps);perm=np.arange(len(types))
            for t in range(3):
                ix=np.flatnonzero(types==t);perm[ix]=rng.permutation(ix)
            out[name+'_perm'+str(ps)]=pi[:,perm]
    return out

def rank_corr(x,y):
    # Average ranks handle constant/tied intervention effects.
    def rank(v):
        order=np.argsort(v,kind='stable');u,starts,counts=np.unique(v[order],return_index=True,return_counts=True)
        ranks=np.empty(len(v),float)
        for start,n in zip(starts,counts):ranks[order[start:start+n]]=start+(n-1)/2
        return ranks
    a,b=rank(x),rank(y)
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None

def audit(models,response,types):
    out={}
    for name in models:
        pred=predict(models[name],response);target=response['target'][None] if pred.shape[0]==1 else response['context_target']
        out[name]=dict(Spearman=rank_corr(pred.ravel(),target.ravel()),RMSE=float(np.sqrt(np.mean((pred-target)**2))),
                      direction=float(np.mean((pred>0)==(target>0))))
    return out
