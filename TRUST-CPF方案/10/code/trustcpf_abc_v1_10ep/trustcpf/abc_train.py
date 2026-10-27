"""Ten-epoch Uniform training; last.pt owns optimizer, RNG and selected weights."""
import copy,math,time
import numpy as np
import torch
from .common import load_torch,save_torch,write_json,sha
from . import align,engine,state_eval
from .model import task_loss
from .folds import pair_ids

def export(out,st):
    save_torch(out/'best.pt',dict(model=st['best_model'],pi=st['pi'],epoch=st['best_epoch'],cfg=st['cfg'],val=st['best_metrics']))
    write_json(out/'history.json',st['history'][1:])
    write_json(out/'epoch0.json',st['history'][0])

def uniform(args,until_epoch=None):
    work,cfg,manifest,text,bank=align.context(args);cfg=dict(cfg,variant='uniform',readout='probability')
    if cfg['adapt_epochs']!=10:raise ValueError('This experiment uses 10 complete epochs')
    out=work/'runs/uniform';out.mkdir(parents=True,exist_ok=True)
    cp=work/'reference/last.pt';model=align.restore(text,bank,cfg,args.device,cp);model.backbone.requires_grad_(False)
    ids=engine.row_ids(args.cache,manifest['pairs']['train'],manifest['labels'])
    opt=engine.optimizer(model,cfg);loader=engine.batches(args.cache,'train',cfg,ids,True)
    candidates=torch.tensor(pair_ids(manifest['labels'],manifest['pairs']['train']),device=args.device)
    pi=torch.full((len(bank['types']),),.5,device=args.device)
    prepared=state_eval.prepare_phase(model,args.cache,'val',cfg)
    def validate():
        model.eval()
        return state_eval.evaluate_cached(model,args.cache,cfg,manifest,pi,'val',prepared=prepared)
    if (out/'last.pt').exists():
        st=load_torch(out/'last.pt')
        assert st['initial_sha256']==sha(cp) and st['cfg']==cfg
        model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer'])
        torch.set_rng_state(st['torch_rng']);np.random.set_state(st['numpy_rng'])
        if str(args.device).startswith('cuda'):torch.cuda.set_rng_state_all(st['cuda_rng'])
    else:
        v=validate()
        st=dict(epoch=0,model=copy.deepcopy(model.state_dict()),best_model=copy.deepcopy(model.state_dict()),
            best_epoch=0,best_metrics=v,optimizer=opt.state_dict(),history=[dict(epoch=0,val=v)],
            pi=pi.cpu(),cfg=cfg,initial_sha256=sha(cp),torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None)
        save_torch(out/'last.pt',st)
    export(out,st)
    limit=10 if until_epoch is None else min(10,until_epoch)
    total=max(1,10*len(loader));warmup=max(1,len(loader))
    for epoch in range(st['epoch'],limit):
        model.train();losses=[];tic=time.perf_counter()
        for bi,data in enumerate(loader):
            step=epoch*len(loader)+bi
            factor=min((step+1)/warmup,1.)*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,total-warmup)))
            for g in opt.param_groups:g['lr']=cfg['graph_lr']*factor
            g,fa,fo,a,o=[v.to(args.device) for v in data];mask=torch.bernoulli(pi)
            joint,lo,lc=model(g,fa,fo,mask)
            loss=task_loss(joint,lo,lc,a,o,candidates)+cfg['lambda_w']*model.graph.regularizer()
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_([p for gg in opt.param_groups for p in gg['params']],cfg['gradient_clip'])
            opt.step();losses.append(float(loss.detach()))
        v=validate()
        if v['cw']['AUC_native']>st['best_metrics']['cw']['AUC_native']:
            st.update(best_epoch=epoch+1,best_metrics=v,best_model=copy.deepcopy(model.state_dict()))
        st['history'].append(dict(epoch=epoch+1,loss=float(np.mean(losses)),val=v,
            learning_rate=opt.param_groups[0]['lr'],seconds=time.perf_counter()-tic))
        st.update(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None)
        save_torch(out/'last.pt',st);export(out,st)
        print('UNIFORM_EPOCH',args.seed,epoch+1,'loss',st['history'][-1]['loss'],'val_auc',v['cw']['AUC_native'],'best_epoch',st['best_epoch'],flush=True)
    if st['epoch']==10:
        original=load_torch(cp)['model']
        for key,value in original.items():
            if key.startswith('backbone.'):
                assert torch.equal(value,st['model'][key].detach().cpu())
                assert torch.equal(value,st['best_model'][key].detach().cpu())
        chosen=max(st['history'],key=lambda r:(r['val']['cw']['AUC_native'],-r['epoch']))['epoch']
        assert chosen==st['best_epoch'] and len(st['history'])==11
        write_json(out/'training_contract.json',dict(epochs_completed=10,selected_epoch=chosen,
            best_sha256=sha(out/'best.pt'),last_sha256=sha(out/'last.pt'),initial_sha256=sha(cp),
            backbone_identical=True,selection='VAL CW native AUC, earliest maximum among epoch 0..10',
            training='new Uniform trajectory from reference; common frozen weights for A/B/C'))
