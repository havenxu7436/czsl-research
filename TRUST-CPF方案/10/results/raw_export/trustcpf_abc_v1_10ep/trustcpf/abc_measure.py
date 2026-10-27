"""Reuse per-image Stage interventions; measure independent auxiliaries and context inputs."""
from pathlib import Path
import numpy as np
import torch
from .common import load_torch,save_torch,read_json,sha
from . import abc,align,response,competition_training as ct,state_compute
from .data import FeatureDataset
from .folds import pair_ids

def block_statistics(minus,plus,truth,weights,select):
    minus=minus.double();plus=plus.double()
    lm=minus.log_softmax(-1);lp=plus.log_softmax(-1)
    delta=(lp-lm).gather(-1,truth[None,:,None].expand(len(lm),-1,1)).squeeze(-1)
    d=plus-minus;d=d-d.mean(-1,keepdim=True)
    ent=-(lp.exp()*lp).sum(-1)+(lm.exp()*lm).sum(-1)
    w=weights.double()*select
    if w.sum()==0:return None
    w=w/w.sum()/len(lm);mean=(delta*w).sum()
    return torch.stack([mean,((delta-mean).square()*w).sum().clamp_min(0).sqrt(),
        (d.square().mean(-1)*w).sum().clamp_min(0).sqrt(),(ent*w).sum()])

@torch.no_grad()
def context_response(args,cp,template,output):
    if Path(output).exists():return load_torch(output)
    work,cfg,manifest,text,bank=align.context(args);groups=read_json(work.parent/'groups.json')['groups']
    ids=template['support_indices'];weights=template['weights'];seed=template['seed']
    ds=FeatureDataset(args.cache,'train');samples=[ds[i] for i in ids]
    attrs=torch.tensor([v[3] for v in samples]);objs=torch.tensor([v[4] for v in samples])
    candidates=torch.tensor(pair_ids(manifest['labels'],manifest['pairs']['train']),device=args.device)
    ix={int(c):i for i,c in enumerate(candidates.cpu())}
    truth=torch.tensor([ix[int(o)*len(manifest['labels']['attrs'])+int(a)] for a,o in zip(attrs,objs)])
    model=align.restore(text,bank,cfg,args.device,cp);size=cfg['eval_batch_size']
    bases=[model.base(*[torch.stack([s[k] for s in samples[start:start+size]]).to(args.device) for k in range(3)])
           for start in range(0,len(samples),size)]
    routes=abc.route(torch.cat([v['obj'] for v in bases]),groups).cpu()
    generator=torch.Generator().manual_seed(int(seed))
    backgrounds=(torch.rand(cfg['response_backgrounds'],len(bank['types']),generator=generator)<.5).float().to(args.device)
    global_h=[];context_h=[];empty=[]
    for e in range(len(bank['types'])):
        off=[];on=[]
        for bg in backgrounds:
            mask=bg.clone();mask[e]=0
            off.append(torch.cat([model.score(b,mask)[0].flatten(1)[:,candidates].cpu() for b in bases]))
            mask[e]=1
            on.append(torch.cat([model.score(b,mask)[0].flatten(1)[:,candidates].cpu() for b in bases]))
        off,on=torch.stack(off),torch.stack(on)
        target=int(bank['target'][e]);target_mask=(attrs==target) if target<model.na else (objs==target-model.na)
        global_value=response.response_statistics(off,on,truth,weights,target_mask)
        global_h.append(global_value);by_context=[];missing=[]
        for g in range(2):
            blocks=[];flags=[]
            for k,mask in enumerate((torch.ones_like(target_mask),target_mask)):
                value=block_statistics(off,on,truth,weights,mask&(routes==g))
                flags.append(value is None)
                blocks.append(global_value[k*4:k*4+4].double() if value is None else value)
            by_context.append(torch.cat(blocks).float());missing.append(flags)
        context_h.append(torch.stack(by_context));empty.append(missing)
    value=dict(features=torch.stack(global_h),context=torch.stack(context_h,1),routes=routes,
        empty_blocks=torch.tensor(empty).permute(1,0,2),support_indices=ids,weights=weights,
        seed=seed,backgrounds=backgrounds.cpu(),checkpoint_sha256=sha(cp),
        head_forwards=2*len(bank['types'])*len(backgrounds)*len(bases),groups=groups)
    if 'features' in template and template.get('checkpoint_sha256')==sha(cp):
        torch.testing.assert_close(value['features'],template['features'],atol=1e-7,rtol=1e-5)
        # Retain measured context differences in the original global coordinate.
        # Empty or identical-support blocks must stay exactly zero differences,
        # including small CPU/GPU replay roundoff in the original global input.
        delta=value['context'].double()-value['features'].double()[None]
        value['global_replay_max_abs_error']=float((value['features']-template['features']).abs().max())
        value['context']=(template['features'].double()[None]+delta).to(value['context'].dtype)
        value['features']=template['features'].clone()
    save_torch(output,value)
    print('CONTEXT_RESPONSE',args.seed,Path(output).name,'images',len(ids),'route_counts',torch.bincount(routes,minlength=2).tolist(),flush=True)
    return value

@torch.no_grad()
def teacher(args,fold):
    work,cfg,manifest,text,bank=align.context(args)
    cp=work/('teachers/%d/best.pt'%fold)
    template=load_torch(work/('responses/fold%d.pt'%fold))
    context_response(args,cp,template,work/('context/fold%d.pt'%fold))
    output=work/('effects/fold%d.pt'%fold)
    if output.exists():return
    split=read_json(work/('splits/fold%d.json'%fold))
    model=align.restore(text,bank,cfg,args.device,cp)
    base,candidates,truth,seen=ct.prepared(args,cfg,manifest,model,split,'measure')
    groups=read_json(work.parent/'groups.json')['groups']
    routes=abc.route(base['obj'],groups).cpu();sample_seen=seen[truth].cpu()
    n=len(truth);m=cfg['graphs'];k=cfg['q5_measurement_blocks'] if fold==5 else cfg['measurement_blocks']
    single=[];mixture=[]
    for b in range(k):
        raw=load_torch(work/('measurements/fold%d/block%02d.pt'%(fold,b)))
        assert raw['checkpoint_sha256']==sha(cp)
        torch.testing.assert_close(truth.cpu(),raw['truth'],rtol=0,atol=0)
        torch.testing.assert_close(sample_seen,raw['sample_seen'],rtol=0,atol=0)
        dest=work/('auxiliary/fold%d/block%02d.pt'%(fold,b))
        if dest.exists():aux=load_torch(dest)
        else:
            gen=torch.Generator().manual_seed(cfg['aux_seed']+args.seed*10000+fold*100+b)
            masks=(torch.rand(m-1,len(bank['types']),generator=gen)<.5).float()
            lp=torch.stack([ct.score(model,base,mask.to(args.device),candidates,cfg['eval_batch_size'])
                            .gather(1,truth[:,None]).squeeze(1).cpu() for mask in masks])
            aux=dict(logprob=lp,masks=masks,truth=truth.cpu(),routes=routes,checkpoint_sha256=sha(cp),
                     seed=cfg['aux_seed']+args.seed*10000+fold*100+b,independent_of_intervention=True)
            save_torch(dest,aux)
        minus=raw['off'][:,:,0];plus=raw['on'][:,:,0]
        single.append(plus-minus);mixture.append(abc.mixture_effect(minus,plus,aux['logprob']))
        print('MIXTURE_MEASURE',args.seed,fold,b+1,'/',k,flush=True)
    one=torch.stack(single);mix=torch.stack(mixture)
    result=dict(single=abc.conditional_targets(one,routes,sample_seen),
                mixture=abc.conditional_targets(mix,routes,sample_seen),routes=routes,sample_seen=sample_seen,
                single_per_image=one,mixture_per_image=mix,graphs=m,checkpoint_sha256=sha(cp),
                measure_ids=split['measure']['seen']+split['measure']['unseen'])
    old=load_torch(work/('measurements/fold%d.pt'%fold))
    torch.testing.assert_close(result['single']['target'],torch.as_tensor(old['target']).flatten(),atol=1e-12,rtol=1e-8)
    for target in ('single','mixture'):
        blocks=result[target]['blocks'];result[target]['se2']=blocks.var(0,unbiased=True)/k
    save_torch(output,result)

def formal_response(args):
    work,cfg,manifest,text,bank=align.context(args)
    cp=work/'runs/uniform/best.pt';template=load_torch(work/'responses/reference.pt')
    # New 10-epoch weights, identical support IDs and masks to the reference.
    state_compute.extract_response(args,cp,template,work/'responses/uniform_best.pt')
    context_response(args,cp,load_torch(work/'responses/uniform_best.pt'),work/'context/uniform_best.pt')
