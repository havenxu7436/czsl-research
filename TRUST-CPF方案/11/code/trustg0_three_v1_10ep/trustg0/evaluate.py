"""Historical probability-mixture readout on natural CW/OW candidates."""
import math
import numpy as np
import torch
from .model import select_pairs
from .metrics import metrics


def candidates(index,phase,world):
    if world=='ow':return list(range(len(index['attrs'])*len(index['objs'])))
    return sorted(set(index['pairs']['train'])|set(index['pairs'][phase]))


def route(obj,groups):
    p=obj.softmax(-1)
    return torch.stack([p[:,g].sum(-1) for g in groups],-1).argmax(-1)


def masks(pi,count,seed,device):
    gen=torch.Generator().manual_seed(seed)
    u=torch.rand(count,pi.shape[-1],generator=gen,dtype=torch.float64)
    return (u[None]<pi.cpu().double()[:,None]).float().to(device)


@torch.no_grad()
def scores(model,base,pairs,mode,pi,cfg,bank_seed,groups):
    if mode=='g0':
        joint=base['obj'].log_softmax(-1)[:,:,None]+base['cond'].log_softmax(-1)
        return select_pairs(joint,pairs,model.no).log_softmax(-1)
    if mode in ('capacity','dense'):
        mask=base['g'].new_full((len(model.graph.types),),float(mode=='dense'))
        return select_pairs(model.score(base,mask,capacity=mode=='capacity')[0],pairs,model.no).log_softmax(-1)
    sampled=masks(pi,cfg['graphs'],bank_seed,base['g'].device)
    routes=route(base['obj'],groups) if len(pi)==2 else torch.zeros(len(base['g']),device=base['g'].device,dtype=torch.long)
    result=base['g'].new_empty((len(routes),len(pairs)))
    for group in range(len(pi)):
        take=routes==group
        if not take.any():continue
        part={k:v[take] for k,v in base.items()};acc=None
        for mask in sampled[group]:
            log=select_pairs(model.score(part,mask)[0],pairs,model.no).log_softmax(-1)
            acc=log if acc is None else torch.logaddexp(acc,log)
        result[take]=acc-math.log(cfg['graphs'])
    return result


def partition_metadata(pairs,train_pairs):
    values=pairs.detach().cpu().tolist();device=pairs.device
    train=torch.tensor([p in train_pairs for p in values],device=device,dtype=torch.bool)
    lookup=torch.full((max(values)+1,),-1,device=device,dtype=torch.long);lookup[pairs]=torch.arange(len(pairs),device=device)
    return dict(train=train,lookup=lookup)


def statistics(logs,pairs,truth,train_pairs,partition=None):
    partition=partition_metadata(pairs,train_pairs) if partition is None else partition
    train=partition['train']
    values={};truth=torch.as_tensor(truth,device=logs.device,dtype=torch.long)
    if not train.any() or train.all():raise ValueError('Seen and unseen candidate partitions required')
    for name,mask in [('seen',train),('unseen',~train)]:
        val,pos=logs[:,mask].max(-1);values[name+'_score']=val;values[name+'_pred']=pairs[mask][pos]
    iy=partition['lookup'][truth]
    if (iy<0).any():raise ValueError('Truth absent from candidate set')
    conf,pred=logs.max(-1);values.update(truth=truth,is_seen=train[iy],
        truth_logprob=logs.gather(1,iy[:,None])[:,0],confidence=conf.exp(),pred=pairs[pred])
    return {k:v.cpu().numpy() for k,v in values.items()}


def summarize(stats,bias=None):
    out=metrics(stats,bias);hit=stats['pred']==stats['truth'];conf=stats['confidence'];ece=0.
    for l,r in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        use=(conf>=l)&((conf<r) if r<1 else conf<=r)
        if use.any():ece+=use.mean()*abs(hit[use].mean()-conf[use].mean())
    out.update(pair_nll=float(-stats['truth_logprob'].mean()),ECE=float(ece),pair_accuracy=float(hit.mean()*100))
    return out


@torch.no_grad()
def evaluate(model,cache,index,cfg,groups,mode='uniform',pi=None,phase='val',world='cw',selected=None,pair_set=None,seen_pairs=None,bank_seed=1701,bias=None):
    model.eval();ids=[i for i,r in enumerate(index['rows']) if r['phase']==phase] if selected is None else selected
    pairs=torch.tensor(candidates(index,phase,world) if pair_set is None else pair_set,device=cache.device)
    seen=set(index['pairs']['train'] if seen_pairs is None else seen_pairs);items=[];partition=partition_metadata(pairs,seen)
    for start in range(0,len(ids),cfg['eval_batch']):
        ix=ids[start:start+cfg['eval_batch']];g,fo=cache.batch(ix,mode!='g0')
        if mode=='g0':
            m=model.backbone if hasattr(model,'backbone') else model;v=m(g,None,None)
            log=select_pairs(v['joint'],pairs,len(index['objs'])).log_softmax(-1)
        else:log=scores(model,model.base(g,fo),pairs,mode,pi,cfg,bank_seed,groups)
        items.append(statistics(log,pairs,[index['rows'][i]['pair'] for i in ix],seen,partition))
    result={k:np.concatenate([v[k] for v in items]) for k in items[0]};result['image_ids']=np.array(ids)
    return summarize(result,bias),result
