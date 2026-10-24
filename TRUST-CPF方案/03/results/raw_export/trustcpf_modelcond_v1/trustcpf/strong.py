"""Fold-local G0 training. Full-TRAIN checkpoints are used only after utility audits."""
import math
from pathlib import Path
import torch
import numpy as np
from .common import ROOT,load_torch,save_torch,write_json,read_json,sha,seed_all


def initialize(model,cfg):
    if cfg.get('_cold_backbone') or cfg.get('synthetic'):
        return
    source=ROOT/'initial'/('g0_seed%d.pt'%cfg['seed'])
    state=load_torch(source)
    model.backbone.load_state_dict(state['model'])


def fold_backbone(args,cfg,text,bank,manifest,pairs,out):
    from .engine import new_model,batches,row_ids
    from .data import mask_from_pairs
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'last.pt'
    inner_cfg=dict(cfg,_cold_backbone=True)
    model=new_model(text,bank,inner_cfg,args.device)
    head=model.backbone
    head.requires_grad_(False)
    for n in ('Bo','Co','Bc','Cc','Au','Av','Wc'):getattr(head,n).requires_grad_(True)
    groups=[]
    for names,scale in [(('Bo','Co','Bc','Cc'),1.),(('Au','Av','Wc'),cfg['backbone_cp_scale'])]:
        groups.append(dict(params=[getattr(head,n) for n in names],lr=cfg['backbone_lr']*scale,
                           initial_lr=cfg['backbone_lr']*scale))
    opt=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
    mask=mask_from_pairs(manifest['labels'],pairs).to(args.device)
    loader=batches(args.cache,'train',cfg,row_ids(args.cache,pairs,manifest['labels']),True)
    history=[];start=0
    if checkpoint.exists():
        state=load_torch(checkpoint)
        if state['train_pairs']!=pairs:raise ValueError('Fold pair identity changed')
        head.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        start=state['epoch'];history=state['history']
        torch.set_rng_state(state['torch_rng'])
        if str(args.device).startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start,cfg['backbone_epochs']):
        factor=.01+.99*(1+math.cos(math.pi*epoch/max(1,cfg['backbone_epochs']-1)))/2
        for group in opt.param_groups:group['lr']=group['initial_lr']*factor
        losses=[]
        for g,fa,fo,a,o in loader:
            g,fa,fo,a,o=[x.to(args.device) for x in (g,fa,fo,a,o)]
            loss=head.objective(head(g,fa,fo),a,o,mask)['total']
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),cfg['gradient_clip']);opt.step()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
        save_torch(checkpoint,dict(model=head.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,
            history=history,train_pairs=pairs,initialization='fresh_G0_architecture',torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None))
        write_json(out/'history.json',history)
        print('FOLD_BACKBONE',out.parent.name,epoch+1,history[-1]['loss'],flush=True)
    model.backbone.requires_grad_(False)
    wrapper=out/'trust_initial.pt'
    save_torch(wrapper,dict(model=model.state_dict(),train_pairs=pairs,base_source=sha(checkpoint)))
    return wrapper
