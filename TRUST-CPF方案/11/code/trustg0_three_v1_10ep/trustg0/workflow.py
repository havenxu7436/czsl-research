"""Three datasets, three independent repetitions, complete G0/TRUST experiment."""
import csv
import gc
from pathlib import Path
import shutil
import numpy as np
import torch
from .common import ROOT,read,write,load,save,sha,fingerprint,lock,seed_all
from .data import index_data,verify_split,teacher_folds,context_groups,balanced_sample
from .cache import FeatureCache,CLIP_SHA
from . import cache as encoding,train,response,utility
from .model import restore_parameters
from .evaluate import candidates,scores,statistics,summarize,evaluate,partition_metadata


def expected_readouts(cfg):return 3+(4+2*len(cfg['permutation_seeds']))*len(cfg['eval_banks'])


def setup(args):
    work=Path(args.work).resolve();source=Path(args.source).resolve();old=read(source/'work/identity.json');cfg=read(args.config)
    clip=Path(old.get('clip',old.get('checkpoint','')))
    if sha(clip)!=CLIP_SHA:raise ValueError('Expected official ViT-L/14 weights')
    identity=dict(config=cfg,code=fingerprint(),checkpoint=str(clip),checkpoint_sha256=CLIP_SHA,paths=old['paths'],datasets=cfg['datasets'],source=str(source))
    work.mkdir(parents=True,exist_ok=True)
    with lock(work/'setup.lock'):
        if (work/'identity.json').exists() and read(work/'identity.json')!=identity:raise ValueError('Work directory belongs to another release')
        if (work/'setup_complete.json').exists():print('SETUP_ALREADY_COMPLETE',flush=True);return
        write(work/'identity.json',identity);checks={}
        for ds in cfg['datasets']:
            print('INDEX',ds,flush=True);index=index_data(identity['paths'][ds],ds);verify_split(index)
            previous=read(source/'work'/ds/'index.json')
            if any(index[k]!=previous[k] for k in ('attrs','objs','metadata_sha256','split_sha256','images')):raise ValueError('Natural split differs from verified prior data: '+ds)
            write(work/ds/'index.json',index);checks[ds]={}
            for seed in cfg['seeds']:
                folds=teacher_folds(index,seed+991);write(work/ds/('seed%d'%seed)/'folds.json',folds)
                checks[ds][str(seed)]=[dict(fold=s['fold'],role=s['role'],inner_pairs=len(s['inner_pairs']),query_pairs=len(s['query_pairs']),train_images=len(s['train_ids']),held_seen_images=len(s['audit_seen_ids']),query_images=len(s['query_ids']),common_pairs=len(s['common_pairs']),primitive_coverage=True) for s in folds]
                print('FOLDS_READY',ds,seed,[len(s['query_pairs']) for s in folds],flush=True)
        need=sum(len(read(work/d/'index.json')['rows'])*257*768*2 for d in cfg['datasets'])/2**30+1.5
        write(work/'split_checks.json',checks);write(work/'setup_complete.json',dict(complete=True,cache_estimate_gib=need,
            g0_trainings=144,relation_trainings=306,epochs_per_stage=10,independent_pipelines=9))
        print('SETUP_COMPLETE','cache_GiB',round(need,1),'free_GiB',round(shutil.disk_usage(work).free/2**30,1),flush=True)


def context(args,with_cache=True):
    work=Path(args.work).resolve();identity=read(work/'identity.json')
    if identity['code']!=fingerprint():raise ValueError('Source/config differs from setup')
    cfg=identity['config'];ds=work/args.dataset;index=read(ds/'index.json')
    if args.dataset not in identity['datasets']:raise ValueError('Unknown dataset')
    if not with_cache:return work,identity,cfg,ds,index
    if not (ds/'cache/complete.json').exists():raise ValueError('Feature cache incomplete')
    cache=FeatureCache(ds,args.device);bank=load(ds/'cache/bank.pt');groups=read(ds/'groups.json')
    w=ds/('seed%d'%args.seed)
    if args.seed not in cfg['seeds']:raise ValueError('Unknown repetition seed')
    return work,cfg,ds,index,cache,bank,groups,w


def prepare(args):
    work,identity,cfg,ds,index=context(args,False)
    encoding.prepare(ds,identity['checkpoint'],cfg,args.device)
    bank=load(ds/'cache/bank.pt');groups=context_groups(bank['anchors'][len(index['attrs']):].numpy())
    write(ds/'groups.json',groups);print('CACHE_COMPLETE',args.dataset,flush=True)


def formal_split(index):
    return dict(train_ids=[i for i,r in enumerate(index['rows']) if r['phase']=='train'],inner_pairs=index['pairs']['train'])


def stages(cache,index,bank,groups,cfg,seed,split,out,formal):
    out=Path(out);g0=out/'g0';train.run(cache,index,bank,groups,cfg,seed,split,g0,'g0',select=formal)
    for kind in (('uniform','capacity') if formal else ('uniform',)):
        reference=out/(kind+'_reference');adapt=out/kind
        train.run(cache,index,bank,groups,cfg,seed+701,split,reference,kind,g0/'best.pt',False)
        train.run(cache,index,bank,groups,cfg,seed+1701,split,adapt,kind,g0/'best.pt',formal,reference/'best.pt')


def train_base(args):
    work,cfg,ds,index,cache,bank,groups,w=context(args)
    with lock(w/'formal.lock'):
        stages(cache,index,bank,groups,cfg,args.seed,formal_split(index),w/'formal',True)
        write(w/'base_complete.json',dict(complete=True))


def teacher(args):
    work,cfg,ds,index,cache,bank,groups,w=context(args)
    if not 1<=args.fold<=5 or not 1<=args.teacher<=cfg['teacher_initializations']:raise ValueError('Invalid teacher')
    split=read(w/'folds.json')[args.fold-1];out=w/'teachers'/('t%d_q%d'%(args.teacher,args.fold));seed=args.seed+10000*args.teacher+97*args.fold
    with lock(out/'run.lock'):
        stages(cache,index,bank,groups,cfg,seed,split,out,False)
        response.measure(cache,index,bank,groups,cfg,seed+3101,split,out/'g0/best.pt',out/'uniform/best.pt',out/'measure')
        write(out/'complete.json',dict(complete=True,seed=seed,fold=args.fold,initialization=args.teacher))


def fit(args):
    work,cfg,ds,index,cache,bank,groups,w=context(args);types=bank['types'].numpy()
    records=[load(w/'teachers'/('t%d_q%d'%(t,q))/'measure/response.pt') for t in range(1,cfg['teacher_initializations']+1) for q in (1,2,3)]
    models=utility.fit(records,types,cfg['ridge_lambda']);save(w/'utility.pt',models)
    report={}
    for q in (4,5):
        report['Q%d'%q]={str(t):utility.audit(models,load(w/'teachers'/('t%d_q%d'%(t,q))/'measure/response.pt'),types) for t in range(1,cfg['teacher_initializations']+1)}
    write(w/'utility_audit.json',report);write(w/'fit_complete.json',dict(complete=True,teachers=len(records),fit_folds=[1,2,3],dev_fold=4,audit_fold=5,independent_seed=args.seed))


def atomic_npz(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(tmp,**values);tmp.replace(path)


def recognize(cache,index,cfg,groups,w,model,capacity,policies):
    """Share image features and frozen base logits across all formal comparisons."""
    methods={'g0':None,'capacity':None,'dense':None,**policies}
    runs=[(name,bs,pi) for name,pi in methods.items() for bs in (cfg['eval_banks'] if pi is not None else cfg['eval_banks'][:1])]
    root=w/'recognition';device=cache.device
    for phase in ('val','test'):
        for world in ('cw','ow'):
            unfinished=[r for r in runs if not (root/r[0]/str(r[1])/(phase+'_'+world+'.npz')).exists()]
            if not unfinished:continue
            ids=[i for i,r in enumerate(index['rows']) if r['phase']==phase];pairs=torch.tensor(candidates(index,phase,world),device=device);seen=set(index['pairs']['train'])
            partition=partition_metadata(pairs,seen)
            items={(n,bs):[] for n,bs,pi in unfinished}
            with torch.no_grad():
                for offset in range(0,len(ids),cfg['eval_batch']):
                    ix=ids[offset:offset+cfg['eval_batch']];g,fo=cache.batch(ix);base=model.base(g,fo)
                    truth=[index['rows'][i]['pair'] for i in ix]
                    for name,bs,pi in unfinished:
                        current=capacity if name=='capacity' else model
                        log=scores(current,base,pairs,name,pi,cfg,bs,groups)
                        items[name,bs].append(statistics(log,pairs,truth,seen,partition))
                    write(w/'progress.json',dict(stage='recognition',phase=phase,world=world,images=offset+len(ix),total=len(ids),readouts=len(unfinished)))
                    if offset%(cfg['eval_batch']*20)==0:print('RECOGNITION',phase,world,offset+len(ix),'/',len(ids),flush=True)
            for name,bs,pi in unfinished:
                parts=items[name,bs];stats={k:np.concatenate([v[k] for v in parts]) for k in parts[0]};stats['image_ids']=np.array(ids)
                atomic_npz(root/name/str(bs)/(phase+'_'+world+'.npz'),stats)
    for name,bs,pi in runs:
        out=root/name/str(bs);result={}
        for world in ('cw','ow'):
            with np.load(out/('val_'+world+'.npz')) as z:val=summarize(dict(z))
            with np.load(out/('test_'+world+'.npz')) as z:test=summarize(dict(z),val['best_bias'])
            result['val_'+world]=val;result['test_'+world]=test
        write(out/'metrics.json',result)
    write(root/'complete.json',dict(complete=True,readouts=len(runs),metric_rows=len(runs)*4))


def q5_decision(cache,index,bank,groups,cfg,w,models):
    out=w/'q5_decision';out.mkdir(exist_ok=True)
    folds=read(w/'folds.json');split=folds[4]
    for t in range(1,cfg['teacher_initializations']+1):
        dest=out/('teacher%d.json'%t)
        if dest.exists():continue
        teacher=w/'teachers'/('t%d_q5'%t);r=load(teacher/'measure/response.pt')
        unseen,_=balanced_sample(index,r['audit_ids'],2,5501+t,cfg['query_pair_limit'])
        seen,_=balanced_sample(index,split['audit_seen_ids'],1,5601+t,cfg['query_pair_limit'])
        ids=seen+unseen
        if not seen or not unseen:raise ValueError('Q5 requires held-out seen/unseen images')
        model=train.new_graph(cache,bank,cfg,0,teacher/'g0/best.pt');restore_parameters(model,load(teacher/'uniform/best.pt')['params']);model.eval()
        policies=utility.policies(models,r,bank['types'].numpy(),cfg['permutation_seeds']);result={}
        for name,pi in policies.items():
            metrics,stats=evaluate(model,cache,index,cfg,groups,name,pi,selected=ids,pair_set=index['pairs']['train'],seen_pairs=split['inner_pairs'])
            result[name]=metrics;atomic_npz(out/('teacher%d'%t)/(name+'.npz'),stats)
        write(dest,dict(metrics=result,seen_images=len(seen),unseen_images=len(unseen),teacher=t))
        print('Q5_COMPLETE',t,flush=True)


def finish(args):
    work,cfg,ds,index,cache,bank,groups,w=context(args)
    with lock(w/'finish.lock'):
        if (w/'complete.json').exists():return
        if not (w/'fit_complete.json').exists():raise ValueError('Utility fit incomplete')
        model=train.new_graph(cache,bank,cfg,args.seed,w/'formal/g0/best.pt');restore_parameters(model,load(w/'formal/uniform/best.pt')['params']);model.eval()
        capacity=train.new_graph(cache,bank,cfg,args.seed,w/'formal/g0/best.pt');restore_parameters(capacity,load(w/'formal/capacity/best.pt')['params']);capacity.eval()
        response.measure(cache,index,bank,groups,cfg,args.seed+9101,formal_split(index),w/'formal/g0/best.pt',w/'formal/uniform/best.pt',w/'formal/measure',True)
        models=load(w/'utility.pt');r=load(w/'formal/measure/response.pt');policies=utility.policies(models,r,bank['types'].numpy(),cfg['permutation_seeds']);save(w/'policies.pt',policies)
        recognize(cache,index,cfg,groups,w,model,capacity,policies)
        del model,capacity;gc.collect()
        q5_decision(cache,index,bank,groups,cfg,w,models)
        write(w/'complete.json',dict(complete=True,seed=args.seed,teachers=5*cfg['teacher_initializations'],formal_readouts=expected_readouts(cfg)))


def report(args):
    work=Path(args.work);identity=read(work/'identity.json');cfg=identity['config'];rows=[];status=[]
    for ds in identity['datasets']:
        for seed in cfg['seeds']:
            w=work/ds/('seed%d'%seed);files=list((w/'recognition').glob('*/*/metrics.json'))
            teachers=len(list((w/'teachers').glob('*/complete.json')));audits=len(list((w/'q5_decision').glob('teacher*.json')))
            done=(w/'complete.json').exists() and len(files)==expected_readouts(cfg) and teachers==5*cfg['teacher_initializations'] and audits==cfg['teacher_initializations']
            status.append(dict(dataset=ds,seed=seed,complete=done,teachers=teachers,readouts=len(files),q5_audits=audits))
            for f in files:
                for phase,value in read(f).items():rows.append(dict(dataset=ds,seed=seed,method=f.parent.parent.name,bank=int(f.parent.name),phase=phase,**value))
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with open(work/'summary.csv','w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    comparisons=[];main=[];audit_rows=[]
    for ds in identity['datasets']:
        for phase in ('val_cw','val_ow','test_cw','test_ow'):
            for method in sorted(set(r['method'] for r in rows if r['dataset']==ds)):
                for metric in ('AUC_native','AUC_exact','HM_native','HM_exact','Seen','Unseen','pair_nll','ECE'):
                    per_seed={}
                    for seed in cfg['seeds']:
                        values=[r[metric] for r in rows if r['dataset']==ds and r['seed']==seed and r['phase']==phase and r['method']==method]
                        if values:per_seed[str(seed)]=float(np.mean(values))
                    if per_seed:
                        v=list(per_seed.values());main.append(dict(dataset=ds,phase=phase,method=method,metric=metric,mean=float(np.mean(v)),std=float(np.std(v,ddof=1)) if len(v)>1 else None,n=len(v),per_seed=per_seed))
        for seed in cfg['seeds']:
            w=work/ds/('seed%d'%seed)
            for p in (w/'q5_decision').glob('teacher*.json'):
                a=read(p)
                for name,m in a['metrics'].items():audit_rows.append(dict(dataset=ds,seed=seed,teacher=a['teacher'],method=name,**m))
    for ds in identity['datasets']:
        for phase in ('test_cw','test_ow'):
            for method,reference in [('uniform','g0'),('matched','uniform'),('context','uniform'),('context','matched'),('context','context_shuffle'),('context','capacity'),('context','dense'),('matched','matched_permuted'),('context','context_permuted')]:
                for metric in ('AUC_native','AUC_exact','HM_at_val_bias','pair_nll'):
                    delta=[]
                    for seed in cfg['seeds']:
                        def vals(name):return [r[metric] for r in rows if r['dataset']==ds and r['seed']==seed and r['phase']==phase and (r['method'].startswith(name.replace('_permuted','_perm')) if name.endswith('_permuted') else r['method']==name)]
                        a,b=vals(method),vals(reference)
                        if a and b:delta.append(float(np.mean(a)-np.mean(b)))
                    if delta:comparisons.append(dict(dataset=ds,phase=phase,method=method,reference=reference,metric=metric,n=len(delta),mean=float(np.mean(delta)),std=float(np.std(delta,ddof=1)) if len(delta)>1 else None,per_seed=delta))
    result=dict(complete=len(status)==len(identity['datasets'])*len(cfg['seeds']) and all(s['complete'] for s in status),pipelines=status,metric_rows=len(rows),comparisons=comparisons)
    write(work/'main_mean_std.json',main);write(work/'q5_decision_summary.json',audit_rows)
    write(work/'compute_summary.json',dict(training_stages=sum(1 for p in work.glob('*/seed*/**/history.json')),
        training_seconds=sum(sum(v.get('seconds',0) for v in read(p)) for p in work.glob('*/seed*/**/history.json')),
        cache_bytes=sum(p.stat().st_size for p in work.glob('*/cache/*') if p.is_file())))
    write(work/'report.json',result);print('STATUS',sum(s['complete'] for s in status),'/',len(status),'rows',len(rows),'complete',result['complete'],flush=True)
    return result
