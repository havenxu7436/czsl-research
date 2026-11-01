"""Exact local-node intervention updates; no candidate or relation subsampling."""
from pathlib import Path
import time
import numpy as np
import torch
from .common import load,save,read,write,seed_all
from .data import balanced_sample
from .model import unit,restore_parameters
from .train import new_graph
from .evaluate import route

FEATURE_NAMES=['all_utility_mean','all_utility_sd','all_centered_score_rms','all_entropy_delta',
               'target_utility_mean','target_utility_sd','target_centered_score_rms','target_entropy_delta']


def query_partition(index,ids,seed):
    rng=np.random.RandomState(seed);groups={};measure=[];audit=[]
    for i in ids:groups.setdefault(index['rows'][i]['pair'],[]).append(i)
    for p in sorted(groups):
        v=list(map(int,rng.permutation(groups[p])));cut=max(1,len(v)//2) if len(v)>1 else 0
        measure+=v[:cut];audit+=v[cut:]
    return measure,audit


def finish_moments(m):
    result=[]
    for g in range(3):
        row=[]
        for j in range(2):
            mass,mu,m2,rms2,ent=m[g,j]
            if mass==0:mass,mu,m2,rms2,ent=m[0,j]
            if mass==0:raise ValueError('Support misses a relation owner')
            mu/=mass;row.extend([mu,np.sqrt(max(0,m2/mass-mu**2)),np.sqrt(max(0,rms2/mass)),ent/mass])
        result.append(row)
    return np.array(result)


class InterventionBatch:
    def __init__(self,model,base,bg,pairs,truth,object_columns=None):
        self.model=model;self.base=base;self.bg=bg;self.pairs=pairs
        self.v=model.prepared(base,bg);self.messages=model.graph.messages();self.raw=model.graph.raw_nodes(bg,self.messages)
        self.a=pairs//model.no;self.o=pairs%model.no
        self.object_columns=object_columns
        self.all_columns=torch.arange(len(pairs),device=pairs.device)
        self.logzc=self.v['cond'].double().logsumexp(-1)
        logits=self.v['obj'].double()[:,self.o]+self.v['cond'].double()[:,self.o,self.a]-self.logzc[:,self.o]
        self.lp=logits.log_softmax(-1);self.p=self.lp.exp();self.entropy=-(self.p*self.lp).sum(-1)
        lookup=torch.full((model.na*model.no,),-1,device=pairs.device,dtype=torch.long);lookup[pairs]=torch.arange(len(pairs),device=pairs.device)
        self.truth=lookup[truth]
        if (self.truth<0).any():raise ValueError('Measurement candidate set misses labels')

    def delta(self,e,value):
        m=self.model;target=m.edge_targets[e];raw=self.raw[target]+m.graph.eta*(value-self.bg[e])*self.messages[e]/m.graph.degree[e]
        node=unit(raw[None])[0];v=self.v
        if target>=m.na:
            obj=target-m.na;ix=torch.where(self.o==obj)[0] if self.object_columns is None else self.object_columns[obj]
            dq=unit(m.po(node[None]))[0]-v['qo'][obj]
            change=m.cfg['gain']*((v['qx'][:,obj]*dq)@v['qa'].T)
            newz=(v['cond'][:,obj].double()+change.double()).logsumexp(-1)
            do=m.cfg['gain']*(self.base['g']@(node-v['z'][target]))
            delta=do.double()[:,None]+change.double()[:,self.a[ix]]-(newz-self.logzc[:,obj])[:,None]
        else:
            ix=self.all_columns;attr=target
            dq=unit(m.pa(node[None]))[0]-v['qa'][attr]
            change=m.cfg['gain']*((v['qx']*v['qo'][None])@dq)
            prob=(v['cond'][:,:,attr].double()-self.logzc).exp()
            dz=torch.log1p(prob*torch.expm1(change.double()))
            delta=-dz[:,self.o]+(self.a==attr)[None]*change.double()[:,self.o]
        return ix,delta

    def changed_summary(self,ix,delta):
        p=self.p[:,ix];lp=self.lp[:,ix];em1=torch.expm1(delta)
        z=1+(p*em1).sum(-1);lz=z.log()
        entropy=(self.entropy-(p*em1*lp).sum(-1)-(p*delta.exp()*delta).sum(-1))/z+lz
        pos=torch.searchsorted(ix,self.truth).clamp(max=max(len(ix)-1,0))
        true_delta=torch.zeros_like(lz) if not len(ix) else delta.gather(1,pos[:,None])[:,0]*(ix[pos]==self.truth)
        return true_delta-lz,entropy

    def effect(self,e):
        ix,d0=self.delta(e,0.);iy,d1=self.delta(e,1.)
        t0,h0=self.changed_summary(ix,d0);t1,h1=self.changed_summary(ix,d1)
        d=d1-d0;centered=d.square().sum(-1)/len(self.pairs)-(d.sum(-1)/len(self.pairs)).square()
        return t1-t0,centered.clamp_min(0),h1-h0


@torch.no_grad()
def measure(cache,index,bank,groups,cfg,seed,split,g0_path,graph_path,out,formal=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    seed_all(seed);model=new_graph(cache,bank,cfg,seed,g0_path);restore_parameters(model,load(graph_path)['params']);model.eval()
    selected,w=balanced_sample(index,split['train_ids'],cfg['response_images_per_pair'],seed,cfg['response_pair_limit'])
    if formal:query=[];audit=[];qw=np.zeros(0)
    else:
        pool,audit=query_partition(index,split['query_ids'],seed+1)
        query,qw=balanced_sample(index,pool,cfg['query_images_per_pair'],seed+2,cfg['query_pair_limit'])
        if not query or not audit:raise ValueError('Need disjoint query measurement and audit images')
    combined=selected+query;weights=np.r_[w,qw];E=len(bank['types']);B=cfg['response_backgrounds']
    gen=torch.Generator().manual_seed(seed+3);backgrounds=(torch.rand(B,E,generator=gen)<.5).float().to(cache.device)
    pairs=torch.tensor(index['pairs']['train'],device=cache.device);partial=out/'partial.pt';clock=time.monotonic()
    object_columns=[torch.tensor([j for j,p in enumerate(index['pairs']['train']) if p%model.no==o],device=cache.device,dtype=torch.long) for o in range(model.no)]
    if partial.exists():
        st=load(partial);moments=st['moments'];ymass=st['ymass'];ysum=st['ysum'];effects=st['effects'];routes=st['routes'];start=st['next_image']
    else:
        moments=np.zeros((E,3,2,5));ymass=np.zeros((E,3));ysum=np.zeros((E,3));effects=np.zeros((E,B,len(query)),np.float32);routes=np.zeros(len(combined),np.int64);start=0
    for offset in range(start,len(combined),cfg['measure_batch']):
        ix=combined[offset:offset+cfg['measure_batch']];g,fo=cache.batch(ix);base=model.base(g,fo)
        rr=route(base['obj'],groups);routes[offset:offset+len(ix)]=rr.cpu().numpy()
        true=torch.tensor([index['rows'][i]['pair'] for i in ix],device=cache.device)
        ns=max(0,min(offset+len(ix),len(selected))-offset)
        a=np.array([index['rows'][i]['attr'] for i in ix]);o=np.array([index['rows'][i]['obj'] for i in ix])
        for b,bg in enumerate(backgrounds):
            response=InterventionBatch(model,base,bg,pairs,true,object_columns)
            measured=torch.empty((E,len(ix),3),device=cache.device,dtype=torch.float64)
            for e in range(E):
                for j,v in enumerate(response.effect(e)):measured[e,:,j]=v
            measured=measured.cpu().numpy()
            for e in range(E):
                delta,rms,entropy=measured[e].T
                owner=model.edge_targets[e];target=a==owner if owner<model.na else o==owner-model.na
                for c in range(3):
                    in_context=np.ones(len(ix),bool) if c==0 else routes[offset:offset+len(ix)]==c-1
                    for j,in_owner in enumerate((np.ones(len(ix),bool),target)):
                        ww=weights[offset:offset+ns]*in_context[:ns]*in_owner[:ns]
                        moments[e,c,j]+=[ww.sum(),ww@delta[:ns],ww@(delta[:ns]**2),ww@rms[:ns],ww@entropy[:ns]]
                    ww=weights[offset+ns:offset+len(ix)]*in_context[ns:]
                    ymass[e,c]+=ww.sum();ysum[e,c]+=ww@delta[ns:]
                if ns<len(ix):effects[e,b,offset+ns-len(selected):offset+len(ix)-len(selected)]=delta[ns:]
            del response
        end=offset+len(ix)
        save(partial,dict(next_image=end,moments=moments,ymass=ymass,ysum=ysum,effects=effects,routes=routes))
        write(out/'progress.json',dict(stage='relation_measurement',images=end,total=len(combined),edges=E,backgrounds=B,seconds=time.monotonic()-clock))
        print('MEASURE',seed,end,'/',len(combined),'edges',E,flush=True)
    h=np.stack([finish_moments(m) for m in moments],1);y=np.divide(ysum,ymass,out=np.zeros_like(ysum),where=ymass>0).T
    mass=np.array([qw[routes[len(selected):]==c].sum() for c in range(2)]) if query else np.zeros(2)
    save(out/'response.pt',dict(features=h[0],context=h[1:],target=y[0],context_target=y[1:],context_mass=mass,effects=effects,
        feature_names=FEATURE_NAMES,support_ids=selected,support_weights=w,query_ids=query,query_weights=qw,audit_ids=audit,
        routes=routes,inner_pairs=split['inner_pairs'],query_pairs=split.get('query_pairs',[])))
    write(out/'complete.json',dict(edges=E,support_images=len(selected),query_images=len(query),audit_images=len(audit),
        support_query_overlap=len(set(selected)&set(query)),query_audit_overlap=len(set(query)&set(audit)),backgrounds=B))
    partial.unlink()
