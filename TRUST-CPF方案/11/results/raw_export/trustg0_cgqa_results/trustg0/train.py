"""Ten-epoch G0 and shared Uniform/capacity training with exact epoch resume."""
import math
from pathlib import Path
import time
import numpy as np
import torch
from .common import seed_all,save,load,write,read,rng_state,restore_rng,sha
from .g0 import VGCPOD
from .model import TrustG0,task_loss,parameters_state,restore_parameters
from .evaluate import evaluate


def new_g0(cache,cfg,seed):return VGCPOD(cache.text['ea'],cache.text['eo'],cache.text['ec'],cfg['g0'],seed).to(cache.device)


def new_graph(cache,bank,cfg,seed,g0_path):
    model=TrustG0(cache.text,bank,cfg,seed).to(cache.device)
    restore_parameters(model.backbone,load(g0_path)['params'])
    return model


def run(cache,index,bank,groups,cfg,seed,split,out,kind,g0_path=None,select=False,initial_graph=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    seed_all(seed)
    model=new_g0(cache,cfg,seed) if kind=='g0' else new_graph(cache,bank,cfg,seed,g0_path)
    if initial_graph is not None:restore_parameters(model,load(initial_graph)['params'])
    if kind=='g0':
        gc=cfg['g0'];adapter={'Bo','Co','Bc','Cc'}
        groups_opt=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and ((n in adapter)==is_adapter)],lr=gc['lr']*scale,scale=scale)
                    for is_adapter,scale in [(True,1.),(False,gc['cp_lr_scale'])]]
        opt=torch.optim.AdamW(groups_opt,betas=tuple(gc['adam_betas']),eps=gc['adam_eps'],weight_decay=gc['weight_decay'])
    else:opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['graph_lr'],weight_decay=cfg['graph_weight_decay'])
    ids=split['train_ids'];pairs=torch.tensor(split['inner_pairs'],device=cache.device,dtype=torch.long)
    if not ids or any(index['rows'][i]['phase']!='train' or index['rows'][i]['pair'] not in set(split['inner_pairs']) for i in ids):raise ValueError('Invalid training split')
    mask=torch.zeros(len(index['objs']),len(index['attrs']),device=cache.device,dtype=torch.bool);mask[pairs%len(index['objs']),pairs//len(index['objs'])]=True
    last=out/'last.pt';start=0;history=[];best=-float('inf');best_epoch=0
    pi=torch.full((1,len(bank['types'])),.5,dtype=torch.double)
    def validation():return evaluate(model,cache,index,cfg,groups,kind,pi)[0]
    if last.exists():
        s=load(last);restore_parameters(model,s['params']);opt.load_state_dict(s['optimizer']);restore_rng(s['rng'])
        start=s['epoch'];history=s['history'];best=s['best'];best_epoch=s['best_epoch']
        if s['train_ids']!=ids:raise ValueError('Resume split differs')
        del s;print('RESUME',kind,seed,start,flush=True)
    else:
        val=validation() if select else None
        if select:best=val['AUC_native']
        history=[dict(epoch=0,val=val)];save(out/'best.pt',dict(params=parameters_state(model),epoch=0))
    steps=math.ceil(len(ids)/cfg['batch_size']);total=cfg['epochs']*steps
    for epoch in range(start+1,cfg['epochs']+1):
        model.train();clock=time.monotonic();total_loss=0.;count=0
        order=np.random.RandomState(seed+epoch*1009).permutation(ids)
        for batch,offset in enumerate(range(0,len(order),cfg['batch_size'])):
            ix=order[offset:offset+cfg['batch_size']].tolist();step=(epoch-1)*steps+batch
            if kind=='g0':
                lr=cfg['g0']['min_lr']+.5*(cfg['g0']['lr']-cfg['g0']['min_lr'])*(1+math.cos(math.pi*step/max(total-1,1)))
                for group in opt.param_groups:group['lr']=lr*group['scale']
            else:
                factor=min((step+1)/steps,1.)*.5*(1+math.cos(math.pi*max(0,step-steps)/max(1,total-steps)))
                for group in opt.param_groups:group['lr']=cfg['graph_lr']*factor
            g,fo=cache.batch(ix,kind!='g0');a=torch.tensor([index['rows'][i]['attr'] for i in ix],device=cache.device)
            o=torch.tensor([index['rows'][i]['obj'] for i in ix],device=cache.device)
            if kind=='g0':loss=model.objective(model(g,None,None),a,o,mask)['total']
            else:
                edges=(torch.rand(len(bank['types']),device=cache.device)<.5).float()
                if kind=='capacity':edges.zero_()
                joint,lo,lc=model.score(model.base(g,fo),edges,kind=='capacity')
                loss=task_loss(joint,lo,lc,a,o,pairs)+cfg['lambda_w']*model.graph.regularizer()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step()
            total_loss+=float(loss.detach())*len(ix);count+=len(ix)
            if (batch+1)%50==0 or batch+1==steps:
                write(out/'progress.json',dict(stage=kind,epoch=epoch,epochs=cfg['epochs'],batch=batch+1,batches=steps,seconds=time.monotonic()-clock))
                print('STEP',kind,seed,epoch,batch+1,'/',steps,flush=True)
        val=validation() if select else None
        if not select or val['AUC_native']>best:
            best=val['AUC_native'] if select else 0.;best_epoch=epoch;save(out/'best.pt',dict(params=parameters_state(model),epoch=epoch))
        history.append(dict(epoch=epoch,loss=total_loss/count,val=val,seconds=time.monotonic()-clock,learning_rates=[v['lr'] for v in opt.param_groups]))
        save(last,dict(params=parameters_state(model),optimizer=opt.state_dict(),rng=rng_state(),epoch=epoch,history=history,best=best,best_epoch=best_epoch,train_ids=ids))
        write(out/'history.json',history);print('EPOCH',kind,seed,epoch,'val_auc',None if val is None else val['AUC_native'],'best',best_epoch,flush=True)
    write(out/'complete.json',dict(epochs=cfg['epochs'],selected_epoch=best_epoch,best_sha256=sha(out/'best.pt'),train_images=len(ids),train_pairs=len(pairs),selection='VAL CW AUC' if select else 'fixed final epoch'))
    write(out/'progress.json',dict(stage='training_complete',epoch=cfg['epochs'],selected_epoch=best_epoch))
    if last.exists():last.unlink()
