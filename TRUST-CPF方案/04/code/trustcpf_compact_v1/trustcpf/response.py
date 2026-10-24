"""Model-conditioned inputs measured only on each teacher's Inner support."""
from pathlib import Path
import numpy as np
import torch
from .common import read_json,write_json,load_torch,save_torch,sha

FEATURE_NAMES=['all_utility_mean','all_utility_sd','all_centered_score_rms','all_entropy_delta',
               'target_utility_mean','target_utility_sd','target_centered_score_rms','target_entropy_delta']


def measurement_file(work,fold):
    reuse=Path(work)/'reuse.json'
    source=Path(read_json(reuse)['source_seed']) if reuse.exists() else Path(work)
    return source/'measurements'/('fold%d.pt'%fold)


def select_support(rows,allowed_pairs,labels,per_pair,seed):
    """Equal mass per composition, equal mass per selected image."""
    ai={a:i for i,a in enumerate(labels['attrs'])};oi={o:i for i,o in enumerate(labels['objs'])}
    groups={(ai[a],oi[o]):[] for a,o in allowed_pairs}
    for i,r in enumerate(rows):
        key=(r['attr'],r['object'])
        if key in groups:groups[key].append(i)
    rng=np.random.RandomState(seed);ids=[];weights=[]
    for key in sorted(groups):
        pool=groups[key]
        if not pool:raise ValueError('Inner composition has no cached image: '+str(key))
        take=rng.choice(pool,min(per_pair,len(pool)),replace=False).tolist()
        ids.extend(take);weights.extend([1/(len(groups)*len(take))]*len(take))
    return ids,torch.tensor(weights,dtype=torch.float64)


def response_statistics(minus,plus,truth,weights,target_mask):
    """minus/plus: [backgrounds, support images, candidates], float64 statistics."""
    minus=minus.double();plus=plus.double();truth=truth.long()
    lm=minus.log_softmax(-1);lp=plus.log_softmax(-1)
    delta=(lp-lm).gather(-1,truth[None,:,None].expand(len(lm),-1,1)).squeeze(-1)
    d=plus-minus;centered=d-d.mean(-1,keepdim=True)
    entropy_delta=-(lp.exp()*lp).sum(-1)+(lm.exp()*lm).sum(-1)
    result=[]
    for mask in (torch.ones_like(target_mask,dtype=torch.bool),target_mask):
        w=weights.double()*mask
        if float(w.sum())<=0:raise ValueError('Support does not cover graph target')
        w=w/w.sum()/len(lm)
        mean=(delta*w).sum()
        variance=((delta-mean).square()*w).sum()
        rms=(centered.square().mean(-1)*w).sum().sqrt()
        ent=(entropy_delta*w).sum()
        result.extend([mean,variance.clamp_min(0).sqrt(),rms,ent])
    return torch.stack(result).float()


@torch.no_grad()
def measure_responses(model,batch,truth,weights,attrs,objs,candidates,bank,count,seed):
    """Every edge/background shares all non-intervened masks; support may be batched."""
    device=next(model.parameters()).device
    gen=torch.Generator(device='cpu').manual_seed(seed)
    backgrounds=(torch.rand(count,len(bank['types']),generator=gen)<.5).float().to(device)
    bases=[model.base(*[v.to(device) for v in chunk]) for chunk in batch]
    result=[]
    for e in range(len(bank['types'])):
        negative=[];positive=[]
        for background in backgrounds:
            mask=background.clone();mask[e]=0
            negative.append(torch.cat([model.score(b,mask)[0].flatten(1)[:,candidates].cpu() for b in bases]))
            mask[e]=1
            positive.append(torch.cat([model.score(b,mask)[0].flatten(1)[:,candidates].cpu() for b in bases]))
        target=int(bank['target'][e]);na=model.na
        target_mask=(attrs==target) if target<na else (objs==target-na)
        result.append(response_statistics(torch.stack(negative),torch.stack(positive),truth,weights,target_mask))
    return torch.stack(result),2*count*len(bank['types'])*len(bases)


def build_responses(args):
    from .engine import setup,new_model
    from .data import FeatureDataset
    from .folds import pair_ids
    cfg,manifest,text,bank,folds,work=setup(args)
    source=Path(read_json(work/'reuse.json')['source_seed'])
    rows=read_json(Path(args.cache)/'train_records.json')
    dataset=FeatureDataset(args.cache,'train')
    candidates=torch.tensor(pair_ids(manifest['labels'],manifest['pairs']['train']),device=args.device)
    index={p:i for i,p in enumerate(candidates.tolist())}
    for j in range(1,7):
        name='fold%d'%j if j<=5 else 'reference'
        checkpoint=source/'teachers'/str(j)/'last.pt' if j<=5 else work/'reference/last.pt'
        out=work/'responses'/(name+'.pt')
        if out.exists():print('REUSE_RESPONSE',name,flush=True);continue
        state=load_torch(checkpoint)
        allowed=state['train_pairs']
        excluded=set(map(tuple,folds['query'][j-1])) if j<=5 else set()
        if set(map(tuple,allowed))&excluded:raise ValueError('Query composition entered response support')
        ids,w=select_support(rows,allowed,manifest['labels'],cfg['response_images_per_pair'],cfg['response_seed']+j)
        samples=[dataset[i] for i in ids]
        attrs=torch.tensor([s[3] for s in samples]);objs=torch.tensor([s[4] for s in samples])
        na=len(manifest['labels']['attrs'])
        truth=torch.tensor([index[int(o)*na+int(a)] for a,o in zip(attrs,objs)])
        batch=[]
        size=cfg.get('eval_batch_size',256)
        for start in range(0,len(ids),size):
            batch.append([torch.stack([s[k] for s in samples[start:start+size]]) for k in range(3)])
        model=new_model(text,bank,cfg,args.device);model.load_state_dict(state['model']);model.eval()
        features,forwards=measure_responses(model,batch,truth,w,attrs,objs,candidates,bank,cfg['response_backgrounds'],cfg['response_seed']+j)
        save_torch(out,dict(features=features,names=FEATURE_NAMES,checkpoint_sha256=sha(checkpoint),
            support_indices=ids,weights=w,train_pairs=allowed,seed=cfg['response_seed']+j,
            backgrounds=cfg['response_backgrounds'],head_forwards=forwards,source_checkpoint=str(checkpoint)))
        write_json(out.with_suffix('.json'),dict(name=name,support_images=len(ids),compositions=len(allowed),
            query_overlap=0,head_forwards=forwards,feature_names=FEATURE_NAMES,
            min=features.min(0).values.tolist(),max=features.max(0).values.tolist(),checkpoint_sha256=sha(checkpoint)))
        print('RESPONSE_COMPLETE',name,tuple(features.shape),'head_forwards',forwards,flush=True)
