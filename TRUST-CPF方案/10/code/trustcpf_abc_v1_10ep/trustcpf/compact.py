"""27-parameter, type-block ridge prediction of paired edge-mean utility."""
from pathlib import Path
import numpy as np
import torch
from .common import read_json,write_json,load_torch,save_torch
from .graph import solve_budget_equal
from .utility import average_rank

VARIANTS=('compact','shuffled_utility')
TYPE_NAMES=('OI','OE','AE')

def aggregate_measurements(raw,edge_count):
    """The episode is the averaging unit. Noise is NOT added again to SE^2."""
    y=raw['y'].double().numpy();edge=raw['edge'].numpy()
    result={k:[] for k in ('target','count','se2','background_se2')}
    for e in range(edge_count):
        keep=edge==e;v=y[keep];n=len(v)
        if n<2:raise ValueError('Need at least two measured episodes for edge %d'%e)
        result['target'].append(v.mean());result['count'].append(n)
        result['se2'].append(v.var(ddof=1)/n)
        result['background_se2'].append(float(raw['noise'][keep].double().sum())/(n*n))
    return {k:np.asarray(v,dtype=np.int64 if k=='count' else np.float64) for k,v in result.items()}

def shuffled_stats(stats,types,seed,fold):
    rng=np.random.RandomState(seed+1009*fold);permutation=np.arange(len(types))
    for t in range(3):
        ids=np.flatnonzero(types==t);permutation[ids]=rng.permutation(ids)
    return {k:v[permutation].copy() for k,v in stats.items()},permutation

def design(x,types,center,spread):
    z=np.column_stack([np.ones(len(x)),(x-center)/spread])
    return np.concatenate([z*(types==t)[:,None] for t in range(3)],axis=1)

def fit_ridge(xs,ys,types,alpha,normalization=None):
    x=np.concatenate(xs).astype(np.float64);y=np.concatenate(ys).astype(np.float64)
    t=np.tile(types,len(xs))
    if normalization is None:
        normalization=dict(center=x.mean(0),spread=np.maximum(x.std(0),1e-6),
            scales=np.asarray([max(y[t==k].std(),1e-12) for k in range(3)]))
    center,spread,scales=[np.asarray(normalization[k],dtype=np.float64) for k in ('center','spread','scales')]
    z=design(x,t,center,spread)
    beta=np.linalg.solve(z.T@z+float(alpha)*np.eye(z.shape[1]),z.T@(y/scales[t]))
    return dict(beta=beta,center=center,spread=spread,scales=scales,alpha=float(alpha),
        dimension=int(z.shape[1]),fit_folds=len(xs),intercepts_penalized=True)

def predict(model,x,types):
    return (design(np.asarray(x,dtype=np.float64),types,model['center'],model['spread'])@model['beta'])*model['scales'][types]

def point_metrics(pred,target):
    p=np.asarray(pred,dtype=np.float64);y=np.asarray(target,dtype=np.float64)
    def corr(a,b):
        return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None
    return dict(n=len(y),rmse=float(np.sqrt(np.mean((p-y)**2))),mae=float(np.abs(p-y).mean()),
        spearman=corr(average_rank(p),average_rank(y)),pearson=corr(p,y),
        sign_accuracy=float(np.mean((p>0)==(y>0))))

def policy(mean,types,cfg):
    return solve_budget_equal(torch.as_tensor(mean,dtype=torch.float64),torch.as_tensor(types),
        cfg['tau_absolute'],cfg['fixed_type_budgets'])

def objective(pi,target,tau):
    p=np.asarray(pi,dtype=np.float64)
    # 0*log(0)=0, including exact endpoints for empty/full budgets.
    kl=np.zeros_like(p)
    for q in (p,1-p):
        keep=q>0;kl[keep]+=q[keep]*np.log(2*q[keep])
    return float(np.dot(target,p)-tau*kl.sum())

def regret(pred,target,types,cfg):
    p,_=policy(pred,types,cfg);oracle,_=policy(target,types,cfg)
    value=objective(p.numpy(),target,cfg['tau_absolute'])
    optimum=objective(oracle.numpy(),target,cfg['tau_absolute'])
    difference=optimum-value
    if difference < -1e-10:raise ArithmeticError('Oracle regret is negative: %g'%difference)
    return dict(regret=max(0.,difference),regularized_value=value,oracle_value=optimum,
        raw_utility=float(np.dot(target,p.numpy())),pi=p.tolist())

def load_inputs(work):
    work=Path(work);data=load_torch(work/'inputs/edge_statistics.pt')
    types=np.asarray(data['types'],dtype=np.int64)
    stats=data['folds']
    xs=[load_torch(work/'responses'/('fold%d.pt'%j))['features'].double().numpy() for j in range(1,6)]
    return types,stats,xs

def fit(work,all_folds=False):
    work=Path(work);cfg=read_json(work/'recipe.json');types,stats,xs=load_inputs(work)
    ys=[s['target'] for s in stats]
    if all_folds:
        alpha=read_json(work/'utility/selection.json')['selected_alpha'];used=5
    else:
        used=3;trials=[]
        for alpha in cfg['ridge_alphas']:
            model=fit_ridge(xs[:3],ys[:3],types,alpha)
            p4=predict(model,xs[3],types)
            trials.append(dict(alpha=float(alpha),q4=regret(p4,ys[3],types,cfg),
                q4_point=point_metrics(p4,ys[3])))
        selected=min(trials,key=lambda r:(r['q4']['regret'],-r['alpha']))
        alpha=selected['alpha']
        write_json(work/'utility/selection.json',dict(selected_alpha=alpha,objective='Q4 regularized utility regret',
            fit_folds=[1,2,3],selection_fold=4,audit_fold=5,trials=trials,
            tie_break='larger alpha',shuffled_alpha='same as compact'))
    models={};normalization=None;permutations={}
    for variant in VARIANTS:
        if variant=='compact':target=ys
        else:
            shuffled=[shuffled_stats(s,types,cfg['label_shuffle_seed']+cfg['seed'],j+1) for j,s in enumerate(stats)]
            target=[s[0]['target'] for s in shuffled]
            permutations={str(j+1):s[1].tolist() for j,s in enumerate(shuffled)}
        model=fit_ridge(xs[:used],target[:used],types,alpha,normalization)
        if variant=='compact':normalization={k:model[k] for k in ('center','spread','scales')}
        model.update(variant=variant,fit_fold_ids=list(range(1,used+1)),permutations=permutations if variant!='compact' else {})
        dest=work/('utility_all' if all_folds else 'utility')/variant
        save_torch(dest/'model.pt',model);models[variant]=model
    print('RIDGE_FIT_COMPLETE',used,'folds alpha',alpha,flush=True)
    return models

def audit(work):
    work=Path(work);cfg=read_json(work/'recipe.json');types,stats,xs=load_inputs(work)
    y=stats[4]['target'];raw=load_torch(work/'inputs/q5_observations.pt')
    constant=np.asarray([np.concatenate([s['target'][types==t] for s in stats[:3]]).mean() for t in types])
    base=regret(constant,y,types,cfg)
    for variant in VARIANTS:
        dest=work/'utility'/variant;model=load_torch(dest/'model.pt');p=predict(model,xs[4],types)
        dec=regret(p,y,types,cfg)
        result=dict(variant=variant,alpha=model['alpha'],edge_mean=point_metrics(p,y),
            by_type={TYPE_NAMES[t]:point_metrics(p[types==t],y[types==t]) for t in range(3)},
            decision=dec,type_constant=base,regret_reduction=base['regret']-dec['regret'],
            observation=point_metrics(p[raw['edge']],raw['target']),
            label_mean_se=float(np.sqrt(stats[4]['se2']).mean()),
            transfer_residual_mse=float(np.mean((p-y)**2)),
            label_variance_included_in_reward=False)
        write_json(dest/'audit.json',result)
        save_torch(dest/'audit_predictions.pt',dict(mean=p,target=y,se2=stats[4]['se2'],
            background_se2=stats[4]['background_se2'],count=stats[4]['count'],types=types))
        print('UTILITY_AUDIT',variant,'alpha',model['alpha'],'rho',result['edge_mean']['spearman'],
            'regret',dec['regret'],flush=True)

def structure(work):
    work=Path(work);cfg=read_json(work/'recipe.json');types,stats,xs=load_inputs(work)
    x=load_torch(work/'responses/reference.pt')['features'].double().numpy()
    for variant in VARIANTS:
        model=load_torch(work/'utility_all'/variant/'model.pt');mean=predict(model,x,types)
        pi,gamma=policy(mean,types,cfg)
        dest=work/'structures'/variant
        save_torch(dest/'structure.pt',dict(mean=torch.from_numpy(mean),pi=pi.float(),pi_float64=pi,
            types=torch.from_numpy(types),multipliers=gamma,alpha=model['alpha']))
        actual={str(t):float(pi.float()[torch.from_numpy(types)==t].double().sum()) for t in range(3)}
        write_json(dest/'structure.json',dict(variant=variant,mean=mean.tolist(),pi=pi.tolist(),
            alpha=model['alpha'],expected_edges=actual,budgets=cfg['fixed_type_budgets'],
            tau=cfg['tau_absolute'],multipliers=gamma,reward='predicted edge-mean utility',
            float32_budget_max_error=max(abs(actual[k]-cfg['fixed_type_budgets'][k]) for k in actual)))

def adapt(args,variant):
    from .engine import setup,train_model,new_model,evaluate_model
    args.variant=variant
    cfg,manifest,text,bank,folds,work=setup(args)
    state_graph=load_torch(work/'structures'/variant/'structure.pt');pi=state_graph['pi']
    cfg['compact_alpha']=state_graph['alpha']
    out=work/'runs'/variant;save_torch(out/'bank.pt',bank)
    best=train_model(args,cfg,text,bank,manifest,manifest['pairs']['train'],out,pi,work/'reference/last.pt',True)
    state=load_torch(best);model=new_model(text,bank,state['cfg'],args.device);model.load_state_dict(state['model'])
    reference=load_torch(work/'reference/last.pt')['model']
    frozen=all(torch.equal(v,reference[k]) for k,v in state['model'].items() if k.startswith('backbone.'))
    if not frozen:raise AssertionError('CPF backbone changed during relation adaptation')
    val=evaluate_model(model,args.cache,state['cfg'],manifest,state['pi'],'val',save=out)
    test=evaluate_model(model,args.cache,state['cfg'],manifest,state['pi'],'test',val,save=out)
    p=state['pi'].clamp(1e-7,1-1e-7)
    write_json(out/'metrics.json',dict(variant=variant,seed=cfg['seed'],epoch=state['epoch'],backbone_frozen=frozen,
        structure=dict(expected_edges={str(t):float(state['pi'][bank['types']==t].double().sum()) for t in range(3)},
            entropy=float(-(p*p.log()+(1-p)*(1-p).log()).sum())),val=val,test=test))
    print('ADAPT_COMPLETE',variant,'epoch',state['epoch'],'TEST_CW_AUC',test['cw']['AUC_native'],
        'TEST_OW_AUC',test['ow']['AUC_native'],flush=True)
