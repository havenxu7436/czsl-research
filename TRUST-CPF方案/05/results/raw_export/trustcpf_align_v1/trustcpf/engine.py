import copy
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .common import read_json, write_json, load_torch, save_torch, seed_all, sha
from .data import FeatureDataset, load_cache
from .model import TrustCPF, task_loss, average_graphs
from .folds import pair_ids, make_folds
from .graph import solve_budget, permute_bank
from .metrics import native_curve, exact_curve, hm


def setup(args):
    cfg=read_json(args.config)
    for k in ('seed','variant'):
        if getattr(args,k,None) is not None: cfg[k]=getattr(args,k)
    if getattr(args,'quick',False):
        cfg.update(backbone_epochs=1,teacher_epochs=1,adapt_epochs=1,utility_epochs=2,fit_episodes=4,
                   eval_episodes=3,backgrounds=2,audit_backgrounds=2,utility_models=2,summaries=2,graphs=2)
    manifest,text=load_cache(args.cache)
    from .common import ROOT
    bank=load_torch(Path(args.cache)/'graph.pt' if cfg.get('synthetic') else ROOT/'assets/knowledge.pt')
    if cfg.get('variant')=='shuffled': bank=permute_bank(bank)
    work=Path(args.work); work.mkdir(parents=True,exist_ok=True)
    folds=make_folds(manifest['pairs']['train'])
    write_json(work/'folds.json',folds)
    write_json(work/'config.json',cfg)
    return cfg,manifest,text,bank,folds,work


def new_model(text,bank,cfg,device):
    seed_all(cfg['seed'])
    from .strong import initialize
    model=TrustCPF(text,bank,cfg)
    initialize(model,cfg)
    return model.to(device)


def batches(cache,phase,cfg,ids=None,shuffle=False):
    ds=FeatureDataset(cache,phase)
    if ids is not None: ds=Subset(ds,ids)
    return DataLoader(ds,batch_size=cfg['batch_size'] if shuffle else cfg.get('eval_batch_size',256),shuffle=shuffle,num_workers=cfg['workers'])


def row_ids(cache,pairs,labels):
    wanted=set(pair_ids(labels,pairs))
    rows=read_json(Path(cache)/'train_records.json')
    return [i for i,r in enumerate(rows) if r['object']*len(labels['attrs'])+r['attr'] in wanted]


def optimizer(model,cfg):
    return torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],
        'lr':cfg['graph_lr'],'initial_lr':cfg['graph_lr']}],weight_decay=cfg['weight_decay'])


def train_model(args,cfg,text,bank,manifest,train_pairs,output,pi=None,initial=None,select=False):
    device=args.device
    model=new_model(text,bank,cfg,device)
    if initial:
        model.load_state_dict(load_torch(initial)['model'])
        for key in ('anchors','source','target','types','degree'):
            getattr(model.graph,key).copy_(bank[key].to(device))
    opt=optimizer(model,cfg)
    candidates=torch.tensor(pair_ids(manifest['labels'],train_pairs),device=device)
    ids=row_ids(args.cache,train_pairs,manifest['labels'])
    loader=batches(args.cache,'train',cfg,ids,True)
    epochs=cfg['adapt_epochs'] if select else cfg['teacher_epochs']
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    pi=torch.full((len(bank['types']),),.5,device=device) if pi is None else pi.to(device)
    edge_logits=torch.nn.Parameter(torch.logit(pi.clamp(.001,.999)))
    free=cfg.get('variant')=='free' and select
    if free: opt.add_param_group({'params':[edge_logits],'lr':cfg['graph_lr']})
    history=[]; start=0; best=-float('inf'); baseline=0.
    if (out/'last.pt').exists():
        state=load_torch(out/'last.pt'); model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        start=state['epoch']; history=state['history']; best=state['best']; baseline=state.get('baseline',0.)
        pi=state['pi'].to(device)
        with torch.no_grad(): edge_logits.copy_(torch.logit(pi.clamp(.001,.999)))
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        if torch.cuda.is_available() and state.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    if select and not (out/'best.pt').exists():
        metrics=evaluate_model(model,args.cache,cfg,manifest,pi,'val')
        best=metrics['cw']['AUC_native']
        save_torch(out/'best.pt',{'model':model.state_dict(),'pi':pi.cpu(),'epoch':0,'cfg':cfg,'val':metrics})
    total_steps=max(1,epochs*len(loader)); warmup=max(1,len(loader))
    for epoch in range(start,epochs):
        model.train(); losses=[]; tic=time.time()
        for bi,data in enumerate(loader):
            step=epoch*len(loader)+bi
            factor=min((step+1)/warmup,1.)*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,total_steps-warmup)))
            for group in opt.param_groups: group['lr']=cfg['graph_lr']*factor
            g,fa,fo,a,o=[t.to(device) for t in data]
            prob=edge_logits.sigmoid() if free else pi
            mask=torch.bernoulli(prob.detach())
            if cfg.get('variant') in ('r0','capacity'): mask.zero_()
            joint,lo,lc=model(g,fa,fo,mask,capacity=(cfg.get('variant')=='capacity'))
            loss=task_loss(joint,lo,lc,a,o,candidates)+cfg['lambda_w']*model.graph.regularizer()
            objective=loss
            if free:
                logq=-torch.nn.functional.binary_cross_entropy_with_logits(edge_logits,mask,reduction='sum')
                safe=prob.clamp(1e-7,1-1e-7)
                kl=(safe*torch.log(2*safe)+(1-safe)*torch.log(2*(1-safe))).mean()
                budget=sum(torch.relu(prob[model.graph.types==ty].mean()-cfg['budget']).square() for ty in range(3))
                objective=loss+(loss.detach()-baseline)*logq+.01*kl+budget
                baseline=.95*baseline+.05*float(loss.detach())
            opt.zero_grad(); objective.backward()
            torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group['params']],cfg['gradient_clip'])
            opt.step(); losses.append(float(loss.detach()))
        if free: pi=edge_logits.sigmoid().detach()
        metrics=evaluate_model(model,args.cache,cfg,manifest,pi,'val') if select else None
        if select and metrics['cw']['AUC_native']>best:
            best=metrics['cw']['AUC_native']
            save_torch(out/'best.pt',{'model':model.state_dict(),'pi':pi.cpu(),'epoch':epoch+1,'cfg':cfg,'val':metrics})
        record={'epoch':epoch+1,'loss':float(np.mean(losses)),'seconds':time.time()-tic,'val':metrics}
        history.append(record); write_json(out/'history.json',history)
        save_torch(out/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'pi':pi.cpu(),
            'epoch':epoch+1,'cfg':cfg,'history':history,'best':best,'baseline':baseline,
            'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'train_pairs':train_pairs})
        print('EPOCH',out.name,epoch+1,'loss',record['loss'],'val_auc',None if metrics is None else best,flush=True)
    write_json(out/'training.json',{'pairs':train_pairs,'epochs':epochs,'seed':cfg['seed'],'checkpoint':sha(out/'last.pt')})
    return out/('best.pt' if select else 'last.pt')


def candidate_spec(manifest,phase,world):
    labels=manifest['labels']; train=set(map(tuple,manifest['pairs']['train']))
    candidates=sorted(train|set(map(tuple,manifest['pairs'][phase]))) if world=='cw' else sorted((a,o) for a in labels['attrs'] for o in labels['objs'])
    ids=torch.tensor(pair_ids(labels,candidates))
    seen=np.array([p in train for p in candidates]); index={tuple(p):i for i,p in enumerate(candidates)}
    return candidates,ids,seen,index


def calibration_metrics(logs,truth,seen,bias,temperature):
    z=torch.as_tensor(logs,dtype=torch.float64).clone()
    z[:,~torch.as_tensor(seen)]+=bias
    lp=(z/temperature).log_softmax(-1); p=lp.exp(); conf,pred=p.topk(1,dim=-1)
    conf,pred=conf[:,0],pred[:,0]
    truth=torch.as_tensor(truth); ok=(pred==truth); sm=torch.as_tensor(seen)[truth]
    s=float(ok[sm].double().mean()); u=float(ok[~sm].double().mean())
    ece=0.
    for b in range(15):
        mask=(conf>=b/15)&(conf<(b+1)/15 if b<14 else conf<=1)
        if mask.any(): ece+=float(mask.double().mean()*(ok[mask].double().mean()-conf[mask].mean()).abs())
    nll=-lp[torch.arange(len(truth)),truth]
    return {'temperature':float(temperature),'bias':float(bias),'seen':100*s,'unseen':100*u,'HM':float(hm(s,u)*100),
        'NLL':float(nll.mean()),'NLL_seen':float(nll[sm].mean()),'NLL_unseen':float(nll[~sm].mean()),'ECE':ece*100}


@torch.no_grad()
def evaluate_model(model,cache,cfg,manifest,pi,phase,calibration=None):
    model.eval(); device=next(model.parameters()).device
    outputs={}
    ds=FeatureDataset(cache,phase)
    for world in ('cw','ow'):
        candidates,ids,seen,index=candidate_spec(manifest,phase,world)
        logs,variances,fulls=[],[],[]
        for data in batches(cache,phase,cfg):
            batch=[v.to(device) for v in data[:3]]
            lp,var,full=average_graphs(model,batch,pi.to(device),ids.to(device),cfg['graphs'],cfg['graph_seed'],
                cfg.get('readout','probability'),cfg.get('variant')=='capacity',True)
            logs.append(lp.cpu()); variances.append(var.cpu()); fulls.append(full.cpu())
        logs=torch.cat(logs).numpy()
        labels=manifest['labels']
        truth=np.array([index[(labels['attrs'][r['attr']],labels['objs'][r['object']])] for r in ds.rows])
        attrs=torch.tensor([r['attr'] for r in ds.rows]); objs=torch.tensor([r['object'] for r in ds.rows])
        pred=torch.tensor(logs).topk(1,dim=-1).indices[:,0]
        ca=torch.tensor([labels['attrs'].index(a) for a,o in candidates])[pred]
        co=torch.tensor([labels['objs'].index(o) for a,o in candidates])[pred]
        ac,oc=ca==attrs,co==objs
        cond=torch.cat(fulls)[torch.arange(len(attrs)),objs].log_softmax(-1)
        native=native_curve(logs,truth,seen); exact=exact_curve(logs,truth,seen)
        bias=exact['best_bias'] if calibration is None else calibration[world]['calibrated']['bias']
        if calibration is None:
            temps=np.exp(np.linspace(math.log(.1),math.log(10),41))
            temp=min(temps,key=lambda t:calibration_metrics(logs,truth,seen,bias,t)['NLL'])
        else: temp=calibration[world]['calibrated']['temperature']
        outputs[world]={'AUC_native':native['AUC'],'AUC_exact':exact['AUC'],
            'HM_oracle_native':native['best_hm'],'HM_oracle_exact':exact['best_hm'],
            'unbiased':calibration_metrics(logs,truth,seen,0.,1.),
            'at_val_bias':calibration_metrics(logs,truth,seen,bias,1.),
            'calibrated':calibration_metrics(logs,truth,seen,bias,temp),
            'graph_variance':float(torch.cat(variances).mean()),'candidates':len(candidates),'images':len(truth),
            'gt_object_attribute_accuracy':float((cond.argmax(-1)==attrs).float().mean()*100),
            'gt_object_attribute_nll':float(-cond[torch.arange(len(attrs)),attrs].mean()),
            'errors':{'both_correct':int((ac&oc).sum()),'attribute_wrong_object_correct':int((~ac&oc).sum()),
                      'attribute_correct_object_wrong':int((ac&~oc).sum()),'both_wrong':int((~ac&~oc).sum())}}
    return outputs


def teachers(args):
    cfg,manifest,text,bank,folds,work=setup(args)
    wanted=list(range(5)) if args.fold=='all' else [int(args.fold)-1]
    for j in wanted:
        q=set(map(tuple,folds['query'][j]))
        inner=[p for p in manifest['pairs']['train'] if tuple(p) not in q]
        from .strong import fold_backbone
        inner_cfg=dict(cfg,_cold_backbone=True)
        initial=fold_backbone(args,inner_cfg,text,bank,manifest,inner,work/'teachers'/str(j+1)/'backbone')
        train_model(args,inner_cfg,text,bank,manifest,inner,work/'teachers'/str(j+1),initial=initial)


def reference(args):
    cfg,manifest,text,bank,folds,work=setup(args)
    train_model(args,cfg,text,bank,manifest,manifest['pairs']['train'],work/'reference')

# One prediction pass supplies both candidate-world probability mixtures.
from .evaluation import evaluate_model
