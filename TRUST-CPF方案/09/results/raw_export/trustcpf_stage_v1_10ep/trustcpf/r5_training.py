"""R5 fits, independent Q5 decisions, frozen scans, matched 10-epoch adaptation."""
import math
import shutil
from pathlib import Path
import numpy as np
import torch
from .common import read_json, write_json, load_torch, save_torch, sha
from . import align, compact, competition as cmp, competition_training as ct, engine, r5
from .metrics import native_curve, exact_curve


def source_work(work):
    return Path(read_json(Path(work)/'reuse.json')['source_seed'])


def fit(args):
    work, cfg, manifest, text, bank = align.context(args)
    types = bank['types'].numpy()
    xs = [load_torch(work/'responses'/('fold%d.pt'%j))['features'].double().numpy() for j in range(1,6)]
    raw = [load_torch(work/'measurements'/('fold%d.pt'%j)) for j in range(1,6)]
    ys = [b['target'][:,list(b['targets']).index('unll')] for b in raw]
    xr = load_torch(work/'responses/reference.pt')['features'].double().numpy()
    trials = {name: [] for name in r5.METHODS}
    for alpha in cfg['ridge_alphas']:
        for name, model in r5.paired_models(xs[:3], ys[:3], types, alpha).items():
            p = compact.predict(model, xs[3], types)
            trials[name].append(dict(alpha=alpha, q4_error=cmp.centered_type_error(p,ys[3],types,model['scales'])))
    selections, predictions, audit = {}, {}, {}
    for name in r5.METHODS:
        choice = min(trials[name], key=lambda v:(v['q4_error'],-v['alpha']))
        selections[name] = dict(selected_alpha=choice['alpha'],trials=trials[name],
            rule='Q4 centered type-normalized MSE; ties larger alpha; same normalization and sum weights')
        for n,dest in ((3,'utility'),(5,'utility_all')):
            model = r5.paired_models(xs[:n],ys[:n],types,choice['alpha'])[name]
            model.update(fit_fold_ids=list(range(1,n+1)), target='unll', method=name)
            save_torch(work/dest/name/'model.pt',model)
            if n == 3:
                p5 = compact.predict(model,xs[4],types)
                p3r = compact.predict(model,xr,types)
            else:
                pr = compact.predict(model,xr,types)
        predictions[name] = dict(q5=p5, reference=pr, reference_three_fold=p3r, q5_target=ys[4])
        audit[name] = dict(all=compact.point_metrics(p5,ys[4]),centered=r5.centered_metrics(p5,ys[4],types),
            by_type={str(t):compact.point_metrics(p5[types==t],ys[4][types==t]) for t in range(3)},
            transfer={str(t):dict(
                input_change_rho=compact.point_metrics(p5[types==t],p3r[types==t])['spearman'],
                refit_change_rho=compact.point_metrics(p3r[types==t],pr[types==t])['spearman'],
                combined_rho=compact.point_metrics(p5[types==t],pr[types==t])['spearman']) for t in range(3)})
        print('R5_FIT',args.seed,name,'alpha',choice['alpha'],'Q5_rho',audit[name]['all']['spearman'],flush=True)
    # This is the exact absolute U-NLL experiment already run in Competition.
    original = source_work(work)
    previous = load_torch(original/'utility/predictions.pt')['unll']
    oldalpha = read_json(original/'utility/selection.json')['unll']['selected_alpha']
    if oldalpha != selections['absolute']['selected_alpha']:
        raise ValueError('Absolute alpha differs from source; check the R5 input/configuration')
    for key in ('q5','reference','q5_target'):
        np.testing.assert_allclose(predictions['absolute'][key],previous[key],rtol=1e-10,atol=1e-12)
    write_json(work/'utility/selection.json',selections)
    write_json(work/'utility/audit.json',audit)
    save_torch(work/'utility/predictions.pt',predictions)
    write_json(work/'utility/absolute_reproduction.json',dict(passed=True,source_alpha=oldalpha,
        source_predictions_sha256=sha(original/'utility/predictions.pt')))


def audit_policies(predictions, types, cfg):
    policies={'uniform':torch.full((len(types),),.5,dtype=torch.double)}
    perm=align.permutation(types,cfg['permutation_seeds'][0])
    for name in r5.METHODS:
        p,_=align.policy(predictions[name]['q5'],types,1.)
        policies[name]=p;policies['permuted_'+name]=p[perm]
    p,_=align.policy(predictions['absolute']['q5_target'],types,1.)
    policies['measured']=p;policies['permuted_measured']=p[perm]
    return {name:policies[name] for name in r5.POLICIES}


@torch.no_grad()
def audit(args):
    work,cfg,manifest,text,bank=align.context(args);types=bank['types'].numpy()
    predictions=load_torch(work/'utility/predictions.pt');policies=audit_policies(predictions,types,cfg)
    split=read_json(work/'splits/fold5.json');source=source_work(work)
    mapped={'uniform':'uniform','absolute':'predicted_unll','permuted_absolute':'predicted_unll_permuted',
            'measured':'measured_unll','permuted_measured':'measured_unll_permuted'}
    prepared=None;rows=[]
    for graph_seed in cfg['audit_eval_seeds']:
        file=work/'decision/blocks'/('evaluate%d.pt'%graph_seed)
        if file.exists():
            saved=load_torch(file)
        else:
            old=load_torch(source/'decision/blocks'/('evaluate%d.pt'%graph_seed))
            gen=torch.Generator().manual_seed(graph_seed)
            uniforms=torch.rand(cfg['audit_graphs'],len(types),generator=gen).double()
            results={};origins={}
            for name,p in policies.items():
                masks=(uniforms<p[None,:]).float()
                oldname=mapped.get(name)
                if oldname is not None and torch.equal(masks,old['results'][oldname]['masks']):
                    results[name]=old['results'][oldname];origins[name]='source/'+oldname;continue
                duplicate=next((n for n,v in results.items() if torch.equal(masks,v['masks'])),None)
                if duplicate is not None:
                    results[name]=results[duplicate];origins[name]='same_masks/'+duplicate;continue
                if prepared is None:
                    model=align.restore(text,bank,cfg,args.device,work/'teachers/5/last.pt')
                    base,candidates,truth,seen=ct.prepared(args,cfg,manifest,model,split,'evaluate')
                    prepared=(model,base,candidates,truth,seen)
                model,base,candidates,truth,seen=prepared;sm=seen[truth]
                if not torch.equal(truth.cpu(),old['truth']) or not torch.equal(seen.cpu(),old['seen']):
                    raise ValueError('Q5 candidate/image order differs from source')
                logs=torch.stack([ct.score(model,base,b.to(args.device),candidates,cfg['eval_batch_size']) for b in masks])
                sufficient=torch.stack([cmp.sufficient(z,truth,seen) for z in logs])
                single=[cmp.diagnostics(s,sm,cfg['competition_temperature'],cfg['risk_pair_chunk']) for s in sufficient]
                mixed=logs.logsumexp(0)-math.log(len(logs))
                mix=cmp.diagnostics(cmp.sufficient(mixed,truth,seen),sm,cfg['competition_temperature'],cfg['risk_pair_chunk'])
                mix['AUC_native']=native_curve(mixed.cpu().numpy(),truth.cpu().numpy(),seen.cpu().numpy())['AUC']
                if abs(exact_curve(mixed.cpu().numpy(),truth.cpu().numpy(),seen.cpu().numpy())['AUC']-mix['AUC_exact'])>1e-9:
                    raise ArithmeticError('Exact AUC and hard competition area differ')
                results[name]=dict(single={k:float(np.mean([v[k] for v in single])) for k in single[0]},mixture=mix,
                    single_statistics=sufficient.cpu(),mixture_logprobs=mixed.cpu(),masks=masks)
                origins[name]='evaluated'
            saved=dict(results=results,truth=old['truth'],seen=old['seen'],policies=policies,graph_seed=graph_seed,
                       origins=origins,source_block_sha256=sha(source/'decision/blocks'/file.name))
            save_torch(file,saved)
        baseline=saved['results']['uniform']
        for name,value in saved['results'].items():
            row=dict(policy=name,bank=graph_seed)
            for readout in ('single','mixture'):
                for key,v in value[readout].items():row[readout+'_'+key]=v
                for target in cmp.TARGETS:row[readout+'_'+target+'_gain']=baseline[readout][target]-value[readout][target]
                row[readout+'_AUC_gain']=value[readout]['AUC_exact']-baseline[readout]['AUC_exact']
            rows.append(row)
        print('R5_Q5_BANK',args.seed,graph_seed,flush=True)
    summary={}
    for name in policies:
        rr=[r for r in rows if r['policy']==name];summary[name]={}
        for key in rr[0]:
            if key in ('policy','bank'):continue
            v=np.array([r[key] for r in rr])
            summary[name][key]=dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),mcse=float(v.std(ddof=1)/np.sqrt(len(v))))
    write_json(work/'decision/audit.json',dict(rows=rows,summary=summary,graphs=cfg['audit_graphs'],
        images={k:len(split['evaluate'][k]) for k in ('seen','unseen')},policy_strength=1.,
        predictor_fit_folds=[1,2,3],selection_fold=4,evaluation_fold=5,
        uncertainty='Graph-bank MCSE conditional on teacher and independent risk images'))


@torch.no_grad()
def scan(args):
    work,cfg,manifest,text,bank=align.context(args);types=bank['types'].numpy()
    predictions=load_torch(work/'utility/predictions.pt');source=source_work(work)
    model=None;rows=[];scanned={};selected={}
    for name in r5.METHODS:
        for strength in sorted(cfg['policy_strengths']):
            p,info=align.policy(predictions[name]['reference'],types,strength)
            for ps in [None]+cfg['permutation_seeds']:
                pi=p if ps is None else p[align.permutation(types,ps)]
                key='%s_k%g_%s'%(name,strength,'real' if ps is None else ps)
                cachekey='uniform' if strength==0 else key;file=work/'scan'/(cachekey+'.json')
                if file.exists():metrics=read_json(file)
                elif strength==0 or (name=='absolute' and ps is None):
                    oldkey='uniform' if strength==0 else 'unll_k%g_real'%strength
                    metrics=read_json(source/'scan'/(oldkey+'.json'));write_json(file,metrics)
                else:
                    if model is None:model=align.restore(text,bank,cfg,args.device,work/'reference/last.pt')
                    metrics=engine.evaluate_model(model,args.cache,cfg,manifest,pi.float(),'val');write_json(file,metrics)
                rows.append(dict(target=name,key=key,strength=float(strength),permutation_seed=ps,val=metrics))
                scanned[key]=dict(pi=pi,type_scale=info)
                print('R5_FROZEN_VAL',args.seed,key,metrics['cw']['AUC_native'],flush=True)
        best=max([x for x in rows if x['target']==name and x['permutation_seed'] is None],
                 key=lambda x:(x['val']['cw']['AUC_native'],-x['strength']))
        selected[name]=dict(key=best['key'],strength=best['strength'],val_auc=best['val']['cw']['AUC_native'])
    formal={'uniform':torch.full((len(types),),.5,dtype=torch.double)}
    perm=align.permutation(types,cfg['permutation_seeds'][0])
    for name in r5.METHODS:
        formal[name]=scanned[selected[name]['key']]['pi'];formal['permuted_'+name]=formal[name][perm]
    old=load_torch(source/'policies.pt')['formal']
    if not torch.equal(formal['absolute'].float(),old['unll'].float()):
        raise ValueError('Absolute formal probabilities differ from source')
    optima={}
    for name in r5.METHODS:
        optima[name]={}
        for ps in cfg['permutation_seeds']:
            best=max([x for x in rows if x['target']==name and x['permutation_seed']==ps],key=lambda x:(x['val']['cw']['AUC_native'],-x['strength']))
            optima[name][str(ps)]=dict(key=best['key'],strength=best['strength'],val_auc=best['val']['cw']['AUC_native'])
    save_torch(work/'policies.pt',dict(formal={v:formal[v] for v in r5.VARIANTS},scanned=scanned,types=types))
    write_json(work/'scan/results.json',rows)
    write_json(work/'selection.json',dict(targets=selected,permutation_seed=cfg['permutation_seeds'][0],
        permutation_optima=optima,rule='epoch0 VAL CW native AUC; ties smaller strength',
        permuted='each method own selected probability multiset, type-wise permutation; diagnostic optima not used'))


def adapt(args,variant):
    work=Path(args.work);source=source_work(work)
    if variant not in ('uniform','absolute'):
        return align.adapt_and_evaluate(args,variant)
    original=source/'runs'/('uniform' if variant=='uniform' else 'unll')
    state=load_torch(original/'last.pt');cfg=read_json(work/'recipe.json')
    requested=load_torch(work/'policies.pt')['formal'][variant].float()
    if state['epoch']!=cfg['adapt_epochs'] or not torch.equal(requested,state['pi'].float()):
        raise ValueError('Source adaptation is not the matching 10-epoch control')
    output=work/'runs'/variant;shutil.copytree(original,output,dirs_exist_ok=True)
    result=read_json(output/'metrics.json');result.update(variant=variant,reused_from=str(original))
    write_json(output/'metrics.json',result)
    write_json(output/'reuse.json',dict(source=str(original),requested_variant=variant,
        checkpoint_sha256=sha(original/'last.pt'),reason='same reference, seed, probabilities, 10-epoch schedule and evaluator'))
    print('REUSE_MATCHED_CONTROL',args.seed,variant,flush=True)
