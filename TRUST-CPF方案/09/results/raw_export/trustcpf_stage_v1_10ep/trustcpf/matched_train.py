"""Ten-epoch fold-head adaptation with an image-disjoint stopping set."""
import copy,math,time
from pathlib import Path
import numpy as np
import torch
from .common import read_json,write_json,load_torch,save_torch,sha
from . import align,engine,competition_training as ct
from .folds import pair_ids
from .model import task_loss
from .metrics import native_curve,exact_curve

def add_stopping_set(split,cfg):
    s=copy.deepcopy(split)
    used=set(s['train_ids'])
    for partition in ('measure','evaluate'):
        for values in s[partition].values():used.update(values)
    rng=np.random.RandomState(cfg['stop_seed']+s['fold']);stop={};remaining={}
    for group in ('seen','unseen'):
        pool=sorted(set(s[group+'_pool'])-used);remaining[group]=len(pool)
        if len(pool)<cfg['stop_images_per_group']:
            raise ValueError('Fold %d has only %d unused %s stopping images'%(s['fold'],len(pool),group))
        stop[group]=rng.choice(pool,cfg['stop_images_per_group'],replace=False).tolist()
    assert not set(stop['seen'])&set(stop['unseen'])
    s.update(stop=stop,stop_seed=cfg['stop_seed']+s['fold'],stop_remaining=remaining)
    return s

@torch.no_grad()
def stopping_metrics(model,prepared,cfg,device):
    base,candidates,truth,seen=prepared
    gen=torch.Generator(device=device).manual_seed(cfg['graph_seed'])
    masks=(torch.rand((cfg['graphs'],len(model.graph.types)),generator=gen,device=device)<.5).float()
    lp=torch.stack([ct.score(model,base,m,candidates,cfg['eval_batch_size']) for m in masks])
    z=(lp.logsumexp(0)-math.log(len(lp))).cpu().numpy();y=truth.cpu().numpy();sc=seen.cpu().numpy()
    return dict(AUC_native=native_curve(z,y,sc)['AUC'],AUC_exact=exact_curve(z,y,sc)['AUC'],
        unseen_NLL=float(-z[~sc[y],y[~sc[y]]].mean()),images=len(y),candidates=len(sc))

def export_checkpoint(out,state):
    save_torch(out/'best.pt',dict(model=state['best_model'],pi=state['pi'],epoch=state['best_epoch'],
        cfg=state['cfg'],stop=state['best_metrics'],train_pairs=state['train_pairs'],train_ids=state['train_ids'],
        stop_ids=state['stop_ids'],initial_sha256=state['initial_sha256']))
    write_json(out/'history.json',state['history'])

def train_teacher(args,fold,until_epoch=None):
    """last.pt is the single authoritative resume state, including its best model."""
    work,cfg,manifest,text,bank=align.context(args);cfg=dict(cfg,_cold_backbone=True,variant='uniform')
    if cfg['stage_epochs']!=10:raise ValueError('This release uses 10 additional teacher epochs')
    split=read_json(work/('splits/fold%d.json'%fold));initial=work/('old/teachers/%d/last.pt'%fold)
    out=work/('teachers/%d'%fold);out.mkdir(parents=True,exist_ok=True)
    model=align.restore(text,bank,cfg,args.device,initial);model.backbone.requires_grad_(False)
    original=load_torch(initial)['model'];opt=engine.optimizer(model,cfg)
    prep=ct.prepared(args,cfg,manifest,model,split,'stop')
    candidates=torch.tensor(pair_ids(manifest['labels'],split['inner_pairs']),device=args.device)
    loader=engine.batches(args.cache,'train',cfg,split['train_ids'],True)
    pi=torch.full((len(bank['types']),),.5,device=args.device)
    if (out/'last.pt').exists():
        st=load_torch(out/'last.pt')
        assert st['initial_sha256']==sha(initial) and st['stop_ids']==split['stop']
        assert st['train_ids']==split['train_ids'] and st['train_pairs']==split['inner_pairs']
        model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer'])
        torch.set_rng_state(st['torch_rng']);np.random.set_state(st['numpy_rng'])
        if str(args.device).startswith('cuda'):torch.cuda.set_rng_state_all(st['cuda_rng'])
    else:
        metrics=stopping_metrics(model,prep,cfg,args.device)
        st=dict(epoch=0,model=copy.deepcopy(model.state_dict()),optimizer=opt.state_dict(),pi=pi.cpu(),
            best_model=copy.deepcopy(model.state_dict()),best_epoch=0,best_metrics=metrics,
            history=[dict(epoch=0,stop=metrics)],cfg=cfg,train_pairs=split['inner_pairs'],train_ids=split['train_ids'],
            stop_ids=split['stop'],initial_sha256=sha(initial),torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None)
        save_torch(out/'last.pt',st)
    export_checkpoint(out,st)
    total=max(1,10*len(loader));warmup=max(1,len(loader));limit=10 if until_epoch is None else min(10,until_epoch)
    for epoch in range(st['epoch'],limit):
        model.train();losses=[];tic=time.perf_counter()
        for bi,data in enumerate(loader):
            step=epoch*len(loader)+bi
            factor=min((step+1)/warmup,1.)*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,total-warmup)))
            for g in opt.param_groups:g['lr']=cfg['graph_lr']*factor
            g,fa,fo,a,o=[x.to(args.device) for x in data];mask=torch.bernoulli(pi)
            joint,lo,lc=model(g,fa,fo,mask)
            loss=task_loss(joint,lo,lc,a,o,candidates)+cfg['lambda_w']*model.graph.regularizer()
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group['params']],cfg['gradient_clip'])
            opt.step();losses.append(float(loss.detach()))
        model.eval();metrics=stopping_metrics(model,prep,cfg,args.device)
        if metrics['AUC_native']>st['best_metrics']['AUC_native']:
            st.update(best_model=copy.deepcopy(model.state_dict()),best_epoch=epoch+1,best_metrics=metrics)
        st['history'].append(dict(epoch=epoch+1,loss=float(np.mean(losses)),stop=metrics,
            learning_rate=opt.param_groups[0]['lr'],seconds=time.perf_counter()-tic))
        st.update(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None)
        save_torch(out/'last.pt',st);export_checkpoint(out,st)
        print('TEACHER_EPOCH',args.seed,fold,epoch+1,'loss',st['history'][-1]['loss'],
            'stop_auc',metrics['AUC_native'],'best_epoch',st['best_epoch'],flush=True)
    if st['epoch']==10:
        chosen=max(st['history'],key=lambda h:(h['stop']['AUC_native'],-h['epoch']))['epoch']
        assert chosen==st['best_epoch']
        assert all(torch.equal(v,st['model'][k].detach().cpu()) and torch.equal(v,st['best_model'][k].detach().cpu()) for k,v in original.items() if k.startswith('backbone.'))
        write_json(out/'contract.json',dict(epochs_completed=10,selected_epoch=chosen,initial_sha256=sha(initial),
            best_sha256=sha(out/'best.pt'),last_sha256=sha(out/'last.pt'),backbone_frozen=True,
            stop_images={k:len(v) for k,v in split['stop'].items()},selection='stop native AUC, earliest maximum',
            main_val_test_used=False))
    return out/'best.pt'

@torch.no_grad()
def measure_unll(args,fold,model):
    """Save on/off sufficient statistics; supervise only unseen NLL."""
    from . import competition as cmp
    work,cfg,manifest,text,bank=align.context(args);split=read_json(work/('splits/fold%d.json'%fold))
    cp=work/('teachers/%d/best.pt'%fold);cph=sha(cp)
    base,candidates,truth,seen=ct.prepared(args,cfg,manifest,model,split,'measure');sm=seen[truth]
    k=cfg['q5_measurement_blocks'] if fold==5 else cfg['measurement_blocks'];values=[]
    for block in range(k):
        path=work/('measurements/fold%d/block%02d.pt'%(fold,block))
        if path.exists():
            saved=load_torch(path);assert saved['checkpoint_sha256']==cph
        else:
            gen=torch.Generator().manual_seed(cfg['measurement_seed']+1000*fold+block)
            mask=(torch.rand(len(bank['types']),generator=gen)<.5).float().to(args.device);off=[];on=[]
            for edge in range(len(mask)):
                b=mask.clone();b[edge]=0;off.append(cmp.sufficient(ct.score(model,base,b,candidates,cfg['eval_batch_size']),truth,seen))
                b[edge]=1;on.append(cmp.sufficient(ct.score(model,base,b,candidates,cfg['eval_batch_size']),truth,seen))
            off,on=torch.stack(off),torch.stack(on)
            effects=(on[:,~sm,0]-off[:,~sm,0]).mean(1)[:,None]
            saved=dict(off=off.cpu(),on=on.cpu(),effects=effects.cpu(),mask=mask.cpu(),truth=truth.cpu(),
                seen_candidates=seen.cpu(),sample_seen=sm.cpu(),targets=['unll'],checkpoint_sha256=cph)
            save_torch(path,saved)
        values.append(saved['effects'].double().numpy())
        print('MATCHED_UTILITY',args.seed,fold,block+1,'/',k,flush=True)
    x=np.stack(values)
    save_torch(work/('measurements/fold%d.pt'%fold),dict(target=x.mean(0),se2=x.var(0,ddof=1)/k,blocks=x,
        targets=['unll'],teacher_sha256=cph,split=split))

def teacher(args,fold):
    from .state_compute import extract_response
    cp=train_teacher(args,fold);work,cfg,manifest,text,bank=align.context(args)
    split=read_json(work/('splits/fold%d.json'%fold));template=load_torch(work/('old/responses/fold%d.pt'%fold))
    extract_response(args,cp,template,work/('responses/fold%d.pt'%fold))
    extract_response(args,cp,template,work/('responses/fold%d_repeat.pt'%fold),
        seed=int(template['seed'])+cfg['response_repeat_offset'],allowed_ids=split['train_ids'],allowed_pairs=split['inner_pairs'])
    model=align.restore(text,bank,cfg,args.device,cp);measure_unll(args,fold,model)
