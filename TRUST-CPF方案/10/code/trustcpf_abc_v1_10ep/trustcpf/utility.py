"""Paired labels, inference-time evidence, heteroscedastic utility and decomposition."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import unit
from .model import candidate_logprob


@torch.no_grad()
def paired_utility(model, base, target_local, candidates, edge_ids, weights, backgrounds, generator):
    ecount = len(model.graph.types)
    ys, vs, differences = [], [], []
    for e in edge_ids.tolist():
        diffs = []
        for _ in range(backgrounds):
            b = (torch.rand(ecount,device=weights.device,generator=generator)<.5).float()
            b[e]=0
            minus=candidate_logprob(model.score(base,b)[0],candidates)
            b[e]=1
            plus=candidate_logprob(model.score(base,b)[0],candidates)
            rows=torch.arange(len(weights),device=weights.device)
            diff=((minus[rows,target_local]-plus[rows,target_local]) * -weights).sum()
            diffs.append(diff)
        ds=torch.stack(diffs)
        ys.append(ds.mean()); vs.append(ds.var(unbiased=True)/backgrounds)
        differences.append(ds)
    return torch.stack(ys), torch.stack(vs), torch.stack(differences)


def support_summary(globals_, rows, allowed_ids, na, no, maximum, rng):
    """Per-composition means followed by equal context averaging; cap per primitive."""
    groups={}
    for i in allowed_ids:
        r=rows[i]; groups.setdefault((r['attr'],r['object']),[]).append(i)
    means=torch.zeros(na+no,globals_.shape[1]); counts=torch.zeros(na+no); contexts=torch.zeros(na+no)
    for c in range(na+no):
        keys=[p for p in groups if (p[0]==c if c<na else p[1]==c-na)]
        pools=[list(rng.permutation(groups[p])) for p in keys]
        selected={p:[] for p in keys}
        total=0
        while total<maximum and any(pools):
            for k in rng.permutation(len(keys)):
                if pools[k] and total<maximum:
                    selected[keys[k]].append(pools[k].pop()); total+=1
        vals=[unit(globals_[idx].float()).mean(0) for idx in selected.values() if idx]
        if vals:
            means[c]=torch.stack(vals).mean(0); counts[c]=total; contexts[c]=len(vals)
    return unit(means),counts,contexts


def utility_features(bank, summary, semantic_only=False):
    mean,count,contexts=summary
    src=bank['source']; dst=bank['anchors'][bank['target']]
    semantic=torch.cat([src,dst,src*dst,(src-dst).abs(),
                        F.one_hot(bank['types'],3).float(),bank['metadata']],-1)
    rc=mean[bank['target']]; ru=torch.zeros_like(rc); mu=torch.ones(len(rc))
    for e,row in enumerate(bank['rows']):
        if row['type']==0:
            children=sorted({r['target'] for r in bank['rows'] if r['type']==0 and r['slot']==row['slot']})
            valid=[c for c in children if count[c]>0]
            if valid: ru[e]=unit(mean[valid].mean(0)); mu[e]=0
    scalars=torch.stack([(rc*src).sum(-1),(rc*dst).sum(-1),
        count[bank['target']].log1p(), contexts[bank['target']].log1p(),
        (count[bank['target']]==0).float(),mu],-1)
    visual=torch.cat([rc,ru,src,dst,scalars],-1)
    if semantic_only: visual=torch.zeros_like(visual)
    return torch.cat([semantic,visual],-1)


class UtilityNet(nn.Module):
    def __init__(self, dim, center, spread):
        super().__init__()
        self.register_buffer('center',center)
        self.register_buffer('spread',spread)
        self.net=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,2))

    def forward(self,x):
        v=self.net((x-self.center)/self.spread)
        return v[:,0], F.softplus(v[:,1])+1e-4


def utility_nll(mean, variance, target, noise, inclusion, edges):
    v=variance+noise.detach()
    loss=.5*(v.log()+(mean-target.detach()).square()/v)
    # Records sample uniformly within the fixed-size stratified subsets B_t.
    # Multiplying by |B_t|/|E| implements the full-edge HT objective.
    return (loss/inclusion).mean()*edges['sampled']/edges['total']


@torch.no_grad()
def ensemble_moments(models, x, scales, types):
    pred=[net(x) for net in models]
    ms=torch.stack([p[0]*scales[types] for p in pred])
    vs=torch.stack([p[1]*scales[types].square() for p in pred])
    return ms.mean(0),vs.mean(0),ms.var(0,unbiased=False)


def aggregate_moments(means, variances, ensemble_vars):
    return means.mean(0),variances.mean(0)+means.var(0,unbiased=False),ensemble_vars.mean(0)


def average_rank(x):
    x=np.asarray(x); order=np.argsort(x,kind='stable'); s=x[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]; ends=np.r_[starts[1:],len(x)]
    r=np.empty(len(x),float)
    for a,b in zip(starts,ends): r[order[a:b]]=(a+b-1)/2
    return r


def regression_metrics(mean,var,y,noise):
    m,v,y,nu=[np.asarray(a,dtype=float) for a in (mean,var,y,noise)]
    if not len(y):
        return {'observations':0,'rmse':None,'mae':None,'spearman':None,'pearson':None,
                'direction_balanced_accuracy':None,'reliable_direction_balanced_accuracy':None,
                'reliable_fraction':None,'direction_auroc':None,'coverage90':None,'nll':None}
    def corr(a,b):
        a=a-a.mean(); b=b-b.mean(); den=np.sqrt((a*a).sum()*(b*b).sum())
        return float((a*b).sum()/den) if den>0 else 0.
    def balanced(mask):
        pos=mask&(y>0); neg=mask&(y<0)
        if not pos.any() or not neg.any(): return None
        return float(((m[pos]>0).mean()+(m[neg]<0).mean())/2)
    reliable=np.abs(y)>1.96*np.sqrt(nu)
    pos=y>0; ranks=average_rank(m)
    auc=float((ranks[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*(~pos).sum())) if pos.any() and (~pos).any() else None
    return {'observations':len(y),'rmse':float(np.sqrt(np.mean((m-y)**2))),'mae':float(np.abs(m-y).mean()),
            'spearman':corr(average_rank(m),average_rank(y)), 'pearson':corr(m,y),
            'direction_balanced_accuracy':balanced(np.ones(len(y),bool)),
            'reliable_direction_balanced_accuracy':balanced(reliable),'reliable_fraction':float(reliable.mean()),
            'direction_auroc':auc,'coverage90':float((np.abs(m-y)<=1.645*np.sqrt(v+nu)).mean()),
            'nll':float(.5*(np.log(v+nu)+(m-y)**2/(v+nu)).mean())}
