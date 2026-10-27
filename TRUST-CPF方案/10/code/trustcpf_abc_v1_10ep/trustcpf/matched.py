"""One matched pooled predictor; single and averaged response readouts."""
from pathlib import Path
import numpy as np
import torch
from .common import load_torch,save_torch,read_json,write_json
from . import state,compact,r5,align
SEEDS=state.SEEDS
METHODS=('old_single','matched_single','old_average','matched_average')

def formal_keys(cfg):
    return ['uniform']+[n for m in METHODS for n in [m]+[m+'_perm%d'%p for p in cfg['permutation_seeds']]]+['matched_cross','matched_average_cross']

def q5_keys(cfg):
    return ['uniform']+[n for m in METHODS+('measured',) for n in [m]+[m+'_perm%d'%p for p in cfg['permutation_seeds']]]

def fit_global(root,cfg):
    root=Path(root);data={};changes=[]
    for s in SEEDS:
        w=root/('seed%d'%s);types=load_torch(w/'inputs/bank.pt')['types'].numpy();xs=[];ys=[]
        for j in range(1,6):
            x=load_torch(w/('responses/fold%d.pt'%j))['features'].double().numpy()
            y=load_torch(w/('measurements/fold%d.pt'%j))['target'][:,0];xs.append(x);ys.append(y)
            old=load_torch(w/('old/measurements/fold%d.pt'%j));oy=old['target'][:,list(old['targets']).index('unll')]
            epoch=read_json(w/('teachers/%d/contract.json'%j))['selected_epoch']
            for t in range(3):
                ix=types==t
                changes.append(dict(seed=s,fold=j,type=t,selected_epoch=epoch,
                    old_mean=float(oy[ix].mean()),new_mean=float(y[ix].mean()),
                    old_sd=float(oy[ix].std()),new_sd=float(y[ix].std()),**compact.point_metrics(y[ix],oy[ix])))
        data[s]=(xs,ys,types)
    types=data[SEEDS[0]][2]
    if not all(np.array_equal(data[s][2],types) for s in SEEDS):raise ValueError('Teacher edge types/order differ')
    xs=[x for s in SEEDS for x in data[s][0][:3]];ys=[y for s in SEEDS for y in data[s][1][:3]]
    trials=[]
    for lam in cfg['teacher_lambdas']:
        m=state.fit_one(xs,ys,types,lam);errs=[]
        for s in SEEDS:
            x,y,_=data[s];p=compact.predict(m,x[3],types)
            errs.append(float(np.mean([np.var((p[types==t]-y[3][types==t])/m['scales'][t]) for t in range(3)])))
        trials.append(dict(lam=lam,q4_error=float(np.mean(errs)),by_seed=errs))
    selected=min(trials,key=lambda r:(r['q4_error'],-r['lam']))
    m=state.fit_one(xs,ys,types,selected['lam'])
    m.update(teacher_ids=[(s,j) for s in SEEDS for j in (1,2,3)],target='unll',stage_matched=True,
        fit_input='primary response only; averaging is a readout comparison')
    save_torch(root/'utility/matched.pt',m)
    write_json(root/'utility/selection.json',dict(selected_lambda=selected['lam'],alpha=m['alpha'],
        trials=trials,rule='mean three Q4 centered type-normalized MSE, ties larger lambda',fit_folds=[1,2,3],audit_fold=5))
    audit=[]
    for s in SEEDS:
        w=root/('seed%d'%s);x0=data[s][0][4];x1=load_torch(w/'responses/fold5_repeat.pt')['features'].double().numpy();y=data[s][1][4]
        for name in METHODS:
            model=load_torch(root/('utility/old.pt' if name.startswith('old') else 'utility/matched.pt'))
            x=(x0+x1)/2 if name.endswith('average') else x0;p=compact.predict(model,x,types);pi,_=align.policy(p,types,1.)
            audit.append(dict(seed=s,method=name,all=compact.point_metrics(p,y),centered=r5.centered_metrics(p,y,types),
                by_type={str(t):compact.point_metrics(p[types==t],y[types==t]) for t in range(3)},linear_gain=float(y@(pi.numpy()-.5))))
    write_json(root/'utility/audit.json',audit);write_json(root/'utility/teacher_change.json',changes)
    print('MATCHED_FIT_COMPLETE','lambda',selected['lam'],'alpha',m['alpha'],flush=True)

def make_policies(args):
    work,cfg,manifest,text,bank=align.context(args);types=bank['types'].numpy();root=work.parent
    models={key:load_torch(root/('utility/'+key+'.pt')) for key in ('old','matched')}
    def inputs(name):
        return [load_torch(work/('responses/'+name+suffix+'.pt'))['features'].double().numpy() for suffix in ('','_repeat')]
    def means(name):
        x0,x1=inputs(name);out={}
        for method in METHODS:
            model=models[method.split('_')[0]];h=(x0+x1)/2 if method.endswith('average') else x0
            out[method]=compact.predict(model,h,types)
            if method.endswith('average'):
                np.testing.assert_allclose(out[method],(compact.predict(model,x0,types)+compact.predict(model,x1,types))/2,atol=1e-11)
        return out
    formal=state.family(means('uniform_best'),types,cfg);ref=means('reference')
    for key,method in [('matched_cross','matched_single'),('matched_average_cross','matched_average')]:
        pi,info=align.policy(ref[method],types,1.)
        formal[key]=dict(pi=pi,mu=ref[method],method=key,permutation_seed=None,response_state='reference',scale=info)
    qmeans=means('fold5');qmeans['measured']=load_torch(work/'measurements/fold5.pt')['target'][:,0]
    q5=state.family(qmeans,types,cfg)
    assert set(formal)==set(formal_keys(cfg)) and set(q5)==set(q5_keys(cfg))
    save_torch(work/'policies.pt',dict(states={'uniform_best':formal},q5=q5,types=types,strength=1.,
        expected_edges=[int((types==t).sum())/2 for t in range(3)]))
    repeats=[]
    for name in ('fold5','reference','uniform_best'):
        x0,x1=inputs(name)
        for modelname,model in models.items():
            p0,p1=[compact.predict(model,x,types) for x in (x0,x1)];a,_=align.policy(p0,types,1.);b,_=align.policy(p1,types,1.)
            for t in range(3):
                ix=types==t
                repeats.append(dict(seed=args.seed,state=name,model=modelname,type=t,
                    rho=compact.point_metrics(p0[ix],p1[ix])['spearman'],prediction_rmse=float(np.sqrt(np.mean((p1[ix]-p0[ix])**2))),
                    probability_mse=float((a[ix]-b[ix]).square().mean())))
    write_json(work/'responses/repeat_diagnostics.json',repeats)
