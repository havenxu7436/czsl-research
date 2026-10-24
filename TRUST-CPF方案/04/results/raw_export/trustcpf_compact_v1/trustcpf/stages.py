from pathlib import Path
import copy
import time
import numpy as np
import torch
from .common import read_json, write_json, load_torch, save_torch, seed_all
from .engine import setup, new_model, row_ids, pair_ids, train_model, evaluate_model, batches
from .utility import (support_summary, utility_features, paired_utility, UtilityNet,
                      utility_nll, ensemble_moments, aggregate_moments, regression_metrics)
from .graph import solve_budget, permute_bank
from .model import average_graphs

_SUPPORT_CACHE={}


def globals_and_rows(cache):
    p=Path(cache)
    return torch.from_numpy(np.load(p/'train/g.npy').astype(np.float32)),read_json(p/'train_records.json')


def measure(args):
    cfg,manifest,text,bank,folds,work=setup(args)
    gs,rows=globals_and_rows(args.cache); na=len(manifest['labels']['attrs']); no=len(manifest['labels']['objs'])
    wanted=list(range(5)) if args.fold=='all' else [int(args.fold)-1]
    for j in wanted:
        started=time.perf_counter()
        dest=work/'measurements'/('fold%d.pt'%(j+1))
        if dest.exists(): print('REUSE',dest,flush=True); continue
        model=new_model(text,bank,cfg,args.device)
        model.load_state_dict(load_torch(work/'teachers'/str(j+1)/'last.pt')['model']); model.eval()
        query=folds['query'][j]; qids=set(pair_ids(manifest['labels'],query))
        inner=[i for i,r in enumerate(rows) if r['object']*na+r['attr'] not in qids]
        groups={p:[i for i,r in enumerate(rows) if r['object']*na+r['attr']==p] for p in qids}
        candidates=torch.tensor(pair_ids(manifest['labels'],manifest['pairs']['train']),device=args.device)
        index={int(p):i for i,p in enumerate(candidates.tolist())}
        rng=np.random.RandomState(cfg['seed']+100*(j+1))
        gen=torch.Generator(device=args.device).manual_seed(cfg['seed']+100*(j+1))
        count=cfg['fit_episodes'] if j<3 else cfg['eval_episodes']
        backgrounds=cfg['backgrounds'] if j<4 else cfg['audit_backgrounds']
        from .data import FeatureDataset
        ds=FeatureDataset(args.cache,'train')
        X,Y,V,E,K,T,D,query_log=[],[],[],[],[],[],[],[]
        for t in range(count):
            chosen=rng.choice(sorted(qids),min(cfg['query_pairs'],len(qids)),replace=False)
            indices=[]; weights=[]
            for p in chosen:
                idx=rng.choice(groups[p],min(cfg['query_images'],len(groups[p])),replace=False)
                indices.extend(idx.tolist()); weights.extend([1/(len(chosen)*len(idx))]*len(idx))
            sample=[ds[i] for i in indices]
            batch=[torch.stack([s[k] for s in sample]).to(args.device) for k in range(3)]
            local=torch.tensor([index[rows[i]['object']*na+rows[i]['attr']] for i in indices],device=args.device)
            edge_ids=[]; kappas=[]
            for ty in range(3):
                ids=torch.where(bank['types']==ty)[0].numpy(); size=min(cfg['edges_per_type'],len(ids))
                selected=rng.choice(ids,size,replace=False); edge_ids.extend(selected.tolist()); kappas.extend([size/len(ids)]*size)
            e=torch.tensor(edge_ids,device=args.device)
            with torch.no_grad():
                base=model.base(*batch)
                y,v,d=paired_utility(model,base,local,candidates,e,torch.tensor(weights,device=args.device),backgrounds,gen)
            summary=support_summary(gs,rows,inner,na,no,cfg['support_images'],rng)
            x=utility_features(bank,summary)
            X.append(x[edge_ids].half()); Y.append(y.cpu()); V.append(v.cpu()); D.append(d.cpu())
            E.append(e.cpu()); K.append(torch.tensor(kappas)); T.append(torch.full((len(e),),t,dtype=torch.long))
            query_log.append({'episode':t,'query_indices':indices,'query_pair_ids':chosen.tolist()})
            if (t+1)%32==0: print('MEASURE',j+1,t+1,'/',count,flush=True)
        save_torch(dest,{'x':torch.cat(X),'y':torch.cat(Y),'noise':torch.cat(V), 'differences':torch.cat(D),
            'edge':torch.cat(E),'inclusion':torch.cat(K),'episode':torch.cat(T),'queries':query_log,
            'sampled':len(edge_ids),'total':len(bank['types']),'fold':j+1,'wall_seconds':time.perf_counter()-started,
            'head_forwards':2*count*len(edge_ids)*backgrounds,'teacher':str(work/'teachers'/str(j+1)/'last.pt')})
        print('MEASUREMENT_COMPLETE',dest,flush=True)


def combine_measurements(work,folds,response=False):
    from .response import measurement_file
    chunks=[]
    for j in folds:
        p=load_torch(measurement_file(work,j))
        if response:
            r=load_torch(work/'responses'/('fold%d.pt'%j))
            p['x']=torch.cat([p['x'].float(),r['features'][p['edge']].float()],-1)
        chunks.append(p)
    result={k:torch.cat([p[k] for p in chunks]) for k in ('x','y','noise','edge','inclusion')}
    result.update(sampled=chunks[0]['sampled'],total=chunks[0]['total'])
    return result


def feature_mode(x,cfg,dim):
    x=x.float().clone()
    if cfg.get('utility_input')=='semantic': x[:,4*dim+7:]=0
    return x


def heldout_edges(bank,cfg):
    rng=np.random.RandomState(20261021); held=[]
    for ty in range(3):
        ids=torch.where(bank['types']==ty)[0].numpy()
        held.extend(rng.permutation(ids)[:int(len(ids)*cfg.get('heldout_edge_fraction',0))].tolist())
    return held


def remove_heldout(data,held):
    keep=~torch.isin(data['edge'],torch.tensor(held,dtype=torch.long))
    return {k:(v[keep] if torch.is_tensor(v) and len(v)==len(keep) else v) for k,v in data.items()}


def fit_utility(args):
    cfg,manifest,text,bank,folds,work=setup(args)
    train=combine_measurements(work,[1,2,3,4,5] if args.fit_all else [1,2,3],cfg.get('utility_input')=='model_response')
    held=heldout_edges(bank,cfg); train=remove_heldout(train,held)
    dev=remove_heldout(combine_measurements(work,[4],cfg.get('utility_input')=='model_response'),held)
    dim=bank['anchors'].shape[1]
    train['x']=feature_mode(train['x'],cfg,dim); dev['x']=feature_mode(dev['x'],cfg,dim)
    center=train['x'].mean(0); spread=train['x'].std(0,unbiased=False).clamp_min(.01)
    if cfg.get('utility_input')=='model_response': spread[-8:]=train['x'][:,-8:].std(0,unbiased=False).clamp_min(1e-6)
    types=bank['types']; scales=torch.stack([train['y'][types[train['edge']]==ty].std(unbiased=False).clamp_min(1e-4) for ty in range(3)])
    out=work/('utility_all' if args.fit_all else 'utility'); out.mkdir(exist_ok=True)
    if args.fit_all:
        recipe=read_json(work/'utility/selection.json'); epochs=recipe['epochs']
    else: epochs=[cfg['utility_epochs']]*cfg['utility_models']
    selected=[]; timing=[]
    for m in range(cfg['utility_models']):
        started=time.perf_counter()
        seed_all(cfg['seed']+10000+m)
        net=UtilityNet(train['x'].shape[1],center,spread).to(args.device)
        opt=torch.optim.AdamW(net.parameters(),lr=cfg['utility_lr'],weight_decay=cfg['weight_decay'])
        best=float('inf'); best_state=None; best_epoch=0; history=[]
        for epoch in range(epochs[m]):
            net.train()
            for ids in torch.randperm(len(train['y'])).split(cfg['utility_batch']):
                ty=types[train['edge'][ids]]; s=scales[ty].to(args.device)
                mean,var=net(train['x'][ids].to(args.device))
                loss=utility_nll(mean,var,train['y'][ids].to(args.device)/s,
                    train['noise'][ids].to(args.device)/s.square(),train['inclusion'][ids].to(args.device),train)
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                means=[]; vars_=[]
                for x in dev['x'].split(cfg['utility_batch']):
                    mm,vv=net(x.to(args.device)); means.append(mm.cpu()); vars_.append(vv.cpu())
                s=scales[types[dev['edge']]]
                metrics=regression_metrics(torch.cat(means)*s,torch.cat(vars_)*s.square(),dev['y'],dev['noise'])
            history.append({'epoch':epoch+1,**metrics})
            if (epoch+1)%5==0: print('UTILITY_EPOCH',m,epoch+1,'dev_nll',metrics['nll'],flush=True)
            if args.fit_all or metrics['nll']<best:
                best=metrics['nll']; best_state=copy.deepcopy(net.cpu().state_dict()); net.to(args.device); best_epoch=epoch+1
        save_torch(out/('model%d.pt'%m),{'model':best_state,'scales':scales,'cfg':cfg,'epoch':best_epoch,'dimension':len(center)})
        write_json(out/('history%d.json'%m),history); selected.append(best_epoch)
        timing.append(time.perf_counter()-started)
        print('UTILITY',m,'epoch',best_epoch,'dev_nll',best,flush=True)
    write_json(out/'selection.json',{'epochs':selected,'folds':[1,2,3,4,5] if args.fit_all else [1,2,3],
                                  'heldout_edges':held,'model_seconds':timing})


def load_utilities(folder,device):
    nets=[]
    for file in sorted(Path(folder).glob('model*.pt')):
        state=load_torch(file); s=state['model']
        net=UtilityNet(state['dimension'],s['center'],s['spread']).to(device)
        net.load_state_dict(s); nets.append(net.eval())
    return nets,state['scales'].to(device),state['cfg']


def utility_audit(args):
    cfg,manifest,text,bank,folds,work=setup(args)
    data=combine_measurements(work,[5],cfg.get('utility_input')=='model_response'); nets,scales,recipe=load_utilities(work/'utility',args.device)
    data['x']=feature_mode(data['x'],recipe,bank['anchors'].shape[1]); all_m=[]; all_v=[]
    for start in range(0,len(data['y']),cfg['utility_batch']):
        sl=slice(start,start+cfg['utility_batch']); ty=bank['types'][data['edge'][sl]].to(args.device)
        m,v,e=ensemble_moments(nets,data['x'][sl].to(args.device),scales,ty)
        all_m.append(m.cpu()); all_v.append((v+e).cpu())
    m=torch.cat(all_m); v=torch.cat(all_v)
    result={'all':regression_metrics(m,v,data['y'],data['noise'])}
    for ty,name in enumerate(('OI','OE','AE')):
        mask=bank['types'][data['edge']]==ty
        result[name]=regression_metrics(m[mask],v[mask],data['y'][mask],data['noise'][mask])
    held=heldout_edges(bank,recipe)
    if held:
        mask=torch.isin(data['edge'],torch.tensor(held))
        result['unmeasured_edges']=regression_metrics(m[mask],v[mask],data['y'][mask],data['noise'][mask])
        result['unmeasured_edge_ids']=held
    fit=remove_heldout(combine_measurements(work,[1,2,3]),held)
    type_m=torch.stack([fit['y'][bank['types'][fit['edge']]==ty].mean() for ty in range(3)])
    type_v=torch.stack([(fit['y'][bank['types'][fit['edge']]==ty].var(unbiased=False)
                      -fit['noise'][bank['types'][fit['edge']]==ty].mean()).clamp_min(1e-8) for ty in range(3)])
    ty=bank['types'][data['edge']]
    result['type_constant']=regression_metrics(type_m[ty],type_v[ty],data['y'],data['noise'])
    lookup_m=type_m[bank['types']].clone();lookup_v=type_v[bank['types']].clone()
    for e in range(len(bank['types'])):
        mask=fit['edge']==e;n=int(mask.sum())
        if n:
            lookup_m[e]=(n*fit['y'][mask].mean()+10*type_m[bank['types'][e]])/(n+10)
            lookup_v[e]=(fit['y'][mask].var(unbiased=False)-fit['noise'][mask].mean()).clamp_min(1e-8)
    result['edge_lookup']=regression_metrics(lookup_m[data['edge']],lookup_v[data['edge']],data['y'],data['noise'])
    save_torch(work/'utility/audit_predictions.pt',dict(mean=m,variance=v,target=data['y'],noise=data['noise'],edge=data['edge']))
    write_json(work/'utility/audit.json',result)
    print('UTILITY_AUDIT Spearman',result['all']['spearman'],'direction',result['all']['direction_balanced_accuracy'],flush=True)


def deployment_moments(args,cfg,manifest,bank,work,folder,inner=None):
    na=len(manifest['labels']['attrs']); no=len(manifest['labels']['objs'])
    key=(str(Path(args.cache).resolve()),None if inner is None else tuple(inner),cfg['summaries'],cfg['support_images'])
    if key not in _SUPPORT_CACHE:
        gs,rows=globals_and_rows(args.cache)
        ids=list(range(len(rows))) if inner is None else inner
        rng=np.random.RandomState(20261021)
        _SUPPORT_CACHE[key]=[support_summary(gs,rows,ids,na,no,cfg['support_images'],rng) for _ in range(cfg['summaries'])]
    nets,scales,recipe=load_utilities(folder,args.device)
    means=[]; vs=[]; es=[]
    for summary in _SUPPORT_CACHE[key]:
        x=feature_mode(utility_features(bank,summary),recipe,bank['anchors'].shape[1])
        if recipe.get('utility_input')=='model_response':
            x=torch.cat([x,load_torch(work/'responses/reference.pt')['features'].float()],-1)
        x=x.to(args.device)
        m,v,e=ensemble_moments(nets,x,scales,bank['types'].to(args.device))
        means.append(m); vs.append(v); es.append(e)
    return tuple(v.cpu() for v in aggregate_moments(torch.stack(means),torch.stack(vs),torch.stack(es)))


def make_structure(cfg,bank,moments):
    m,v,e=moments
    lv=0. if cfg['variant']=='mean' else cfg['lambda_var']
    le=0. if cfg['variant']=='mean' else cfg['lambda_ens']
    reward=m-lv*v.sqrt()-le*e.sqrt()
    if 'fixed_type_budgets' in cfg:
        from .graph import solve_budget_equal
        pi,gamma=solve_budget_equal(reward,bank['types'],cfg['tau_absolute'],cfg['fixed_type_budgets'])
    else: pi,gamma=solve_budget(reward,bank['types'],cfg['tau_absolute'],cfg['budget'])
    if 'availability' in bank: pi=pi*bank['availability']
    return {'pi':pi,'mean':m,'task_variance':v,'ensemble_variance':e,'reward':reward,'multipliers':gamma}


@torch.no_grad()
def tune(args):
    cfg,manifest,text,bank,folds,work=setup(args)
    inner_pairs=[p for p in manifest['pairs']['train'] if tuple(p) not in set(map(tuple,folds['query'][3]))]
    inner=row_ids(args.cache,inner_pairs,manifest['labels'])
    moments=deployment_moments(args,cfg,manifest,bank,work,work/'utility',inner)
    fit=remove_heldout(combine_measurements(work,[1,2,3]),heldout_edges(bank,cfg))
    scale=max(float(fit['y'].std(unbiased=False)),1e-4)
    model=new_model(text,bank,cfg,args.device); model.load_state_dict(load_torch(work/'teachers/4/last.pt')['model']); model.eval()
    from .data import FeatureDataset
    from torch.utils.data import Subset,DataLoader
    ds=FeatureDataset(args.cache,'train'); qids=row_ids(args.cache,folds['query'][3],manifest['labels'])
    loader=DataLoader(Subset(ds,qids),batch_size=cfg['batch_size'])
    candidates=torch.tensor(pair_ids(manifest['labels'],manifest['pairs']['train']),device=args.device)
    index={int(p):i for i,p in enumerate(candidates.tolist())}; history=[]
    def objective(c):
        c=dict(c,tau_absolute=c['tau_scale']*scale)
        structure=make_structure(c,bank,moments); values={}
        for data in loader:
            batch=[v.to(args.device) for v in data[:3]]
            lp,_=average_graphs(model,batch,structure['pi'].to(args.device),candidates,min(4,cfg['graphs']),1701)
            targets=(data[4]*len(manifest['labels']['attrs'])+data[3]).tolist()
            for i,p in enumerate(targets): values.setdefault(p,[]).append(float(-lp[i,index[p]]))
        score=float(np.mean([np.mean(v) for v in values.values()]))
        history.append({'lambda_var':c['lambda_var'],'lambda_ens':c['lambda_ens'],'budget':c['budget'],'tau_scale':c['tau_scale'],'query_nll':score})
        return score
    # Coordinate search fixed in the research protocol; Q4 alone supplies selection labels.
    for keys,choices in [ (('lambda_var','lambda_ens'),[(a,b) for a in (0.,.5,1.) for b in (0.,.25,.5)]),
                          (('budget',),[(a,) for a in (.25,.5,.75)]),
                          (('tau_scale',),[(a,) for a in (.25,.5,1.)]) ]:
        trials=[]
        for vals in choices:
            trial=dict(cfg,**dict(zip(keys,vals))); trials.append((objective(trial),vals))
        _,winner=min(trials,key=lambda p:p[0]); cfg.update(dict(zip(keys,winner)))
    cfg['tau_absolute']=cfg['tau_scale']*scale
    write_json(work/'tuning.json',{'selected':{k:cfg[k] for k in ('lambda_var','lambda_ens','budget','tau_scale','tau_absolute')},'trials':history})
    print('TUNED',read_json(work/'tuning.json')['selected'],flush=True)


def structure(args):
    cfg,manifest,text,bank,folds,work=setup(args)
    if (work/'tuning.json').exists(): cfg.update(read_json(work/'tuning.json')['selected'])
    fit=remove_heldout(combine_measurements(work,[1,2,3]),heldout_edges(bank,cfg))
    cfg.setdefault('tau_absolute',cfg['tau_scale']*max(float(fit['y'].std(unbiased=False)),1e-4))
    folder=work/'utility_all' if (work/'utility_all').exists() else work/'utility'
    moments=deployment_moments(args,cfg,manifest,bank,work,folder)
    result=make_structure(cfg,bank,moments)
    save_torch(work/'structure.pt',result)
    write_json(work/'structure.json',{'pi':result['pi'].tolist(),'mean':result['mean'].tolist(),
        'task_variance':result['task_variance'].tolist(),'ensemble_variance':result['ensemble_variance'].tolist(),
        'multipliers':result['multipliers'],'parameters':cfg,
        'expected_edges':{str(t):float(result['pi'][bank['types']==t].sum()) for t in range(3)}})


def adapt(args):
    cfg,manifest,text,bank,folds,work=setup(args)
    if (work/'tuning.json').exists(): cfg.update(read_json(work/'tuning.json')['selected'])
    main=load_torch(work/'structure.pt')
    if cfg['variant']=='mean':
        cfg['tau_absolute']=read_json(work/'structure.json')['parameters']['tau_absolute']
        pi=make_structure(cfg,bank,(main['mean'],main['task_variance'],main['ensemble_variance']))['pi']
    else: pi=main['pi'].clone()
    if cfg['variant'] in ('r0','capacity'): pi.zero_()
    if cfg['variant']=='dense': pi.fill_(1.)
    if cfg['variant'] in ('drop','free'):
        for ty in range(3): pi[bank['types']==ty]=main['pi'][bank['types']==ty].mean()
    out=work/'runs'/cfg['variant']
    if (out/'bank.pt').exists(): bank=load_torch(out/'bank.pt')
    else: save_torch(out/'bank.pt',bank)
    initial=work/'reference/last.pt'
    if cfg['variant'] in ('r0','capacity'):
        initial=train_model(args,cfg,text,bank,manifest,manifest['pairs']['train'],work/('reference_'+cfg['variant']))
    best=train_model(args,cfg,text,bank,manifest,manifest['pairs']['train'],out,pi,initial,True)
    state=load_torch(best); model=new_model(text,bank,state['cfg'],args.device); model.load_state_dict(state['model'])
    val=evaluate_model(model,args.cache,state['cfg'],manifest,state['pi'],'val',save=out)
    test=evaluate_model(model,args.cache,state['cfg'],manifest,state['pi'],'test',val,save=out)
    p=state['pi'].clamp(1e-7,1-1e-7)
    structure_stats={'expected_edges':{str(ty):float(state['pi'][bank['types']==ty].sum()) for ty in range(3)},
                     'entropy':float(-(p*p.log()+(1-p)*(1-p).log()).sum())}
    write_json(out/'metrics.json',{'variant':cfg['variant'],'seed':cfg['seed'],'epoch':state['epoch'],
        'structure':structure_stats,'val':val,'test':test})
    print('ADAPT_COMPLETE',cfg['variant'],'epoch',state['epoch'],
          'TEST_CW_AUC',test['cw']['AUC_native'],'TEST_OW_AUC',test['ow']['AUC_native'],flush=True)


def noise(args):
    cfg,manifest,text,clean,folds,work=setup(args)
    run=work/'runs'/cfg['variant']; state=load_torch(run/'best.pt')
    bank0=load_torch(run/'bank.pt'); cfg=state['cfg']; model=new_model(text,bank0,cfg,args.device)
    model.load_state_dict(state['model']); cal=read_json(run/'metrics.json')['val']
    structure0=load_torch(work/'structure.pt')
    if (work/'tuning.json').exists(): cfg.update(read_json(work/'tuning.json')['selected'])
    cfg.setdefault('tau_absolute',read_json(work/'structure.json')['parameters']['tau_absolute'])
    noise_file=run/('noise_delete.json' if args.deletion else 'noise_mismatch.json')
    rows=read_json(noise_file) if noise_file.exists() else []
    completed={(r['seed'],r['type'],r['rate']) for r in rows}
    for seed in args.noise_seeds:
        for ty in range(3):
            for rate in args.rates:
                if (seed,ty,rate) in completed:continue
                bank=permute_bank(bank0,seed,rate,ty,args.deletion)
                with torch.no_grad(): model.graph.source.copy_(bank['source'].to(args.device))
                if rate==0:
                    pi=state['pi'].clone()
                elif cfg['variant'] in ('robust','mean'):
                    moments=deployment_moments(args,cfg,manifest,bank,work,work/'utility_all')
                    pi=make_structure(cfg,bank,moments)['pi']
                else: pi=state['pi'].clone()
                if 'availability' in bank: pi=pi*bank['availability']
                metrics=read_json(run/'metrics.json')['test'] if rate==0 else evaluate_model(model,args.cache,cfg,manifest,pi,'test',cal)
                changed=(bank['source']-bank0['source']).norm(dim=-1)>1e-6
                actual=float((1-bank['availability'][bank['types']==ty]).mean()) if 'availability' in bank else float(changed[bank['types']==ty].float().mean())
                rows.append({'seed':seed,'type':ty,'rate':rate,'actual_rate':actual,'kind':'deletion' if args.deletion else 'source_mismatch','test':metrics})
                rows[-1]['expected_edges']={str(t):float(pi[bank['types']==t].sum()) for t in range(3)}
                write_json(noise_file,rows)
                print('NOISE',cfg['variant'],seed,ty,rate,metrics['ow']['AUC_native'],flush=True)
