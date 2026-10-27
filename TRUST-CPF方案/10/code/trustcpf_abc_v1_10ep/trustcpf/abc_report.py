"""Complete paired tables: banks averaged within seed, never counted as training repeats."""
from pathlib import Path
import numpy as np
from .common import load_torch,read_json,write_json
from .state_workflow import check_marker,write_csv
from . import abc,compact,abc_workflow as wf

def pair_rows(rows,axes,metrics):
    index={tuple(r[k] for k in axes)+(r['policy'],):r for r in rows};out=[]
    for r in rows:
        if r['policy']=='uniform' or r['permutation_seed'] is not None:continue
        key=tuple(r[k] for k in axes)
        names=[k[-1] for k in index if k[:-1]==key and k[-1].startswith(r['policy']+'_perm')]
        controls={'uniform':['uniform']}
        if names:controls['permutation_mean']=names
        available={k[-1] for k in index if k[:-1]==key}
        matched='matched' if 'matched' in available else abc.eta_key('matched',r.get('eta',1.))
        if matched not in available:matched=abc.eta_key('matched',1.)
        if matched in available and r['policy']!=matched:controls['matched']=[matched]
        if r['policy'] in ('context',abc.eta_key('context',1.)):controls['context_shuffle']=['context_shuffle']
        for label,policy_names in controls.items():
            refs=[index.get(key+(n,)) for n in policy_names]
            if any(v is None for v in refs):continue
            item={k:r[k] for k in axes};item['comparison']=r['policy']+'-minus-'+label
            for m in metrics:item[m]=r[m]-float(np.mean([v[m] for v in refs]))
            out.append(item)
    return out

def aggregate(rows,keys,metrics,banks):
    within={}
    for r in rows:within.setdefault((r['seed'],)+tuple(r[k] for k in keys),[]).append(r)
    groups={}
    for key,rr in within.items():
        if len(rr)!=banks:continue
        mean={m:float(np.mean([r[m] for r in rr])) for m in metrics}
        groups.setdefault(key[1:],[]).append(dict(seed=key[0],**mean))
    out=[]
    for key,rr in groups.items():
        if {r['seed'] for r in rr}!=set(abc.SEEDS):continue
        row=dict(zip(keys,key));row['by_seed']=rr
        for m in metrics:
            row[m+'_mean']=float(np.mean([r[m] for r in rr]));row[m+'_sd']=float(np.std([r[m] for r in rr],ddof=1))
        out.append(row)
    return out

def report(args):
    root=Path(args.work);cfg=read_json(args.config);rows=[];q4=[];q5=[];directional=[];curves=[];costs=[];missing=[];completed=0
    for name in wf.GLOBAL:
        try:
            mark=check_marker(root,root/('progress/global/'+name+'.json'));completed+=1
            costs.append(dict(seed='global',stage=name,seconds=mark['seconds'],peak_gpu_bytes=0))
        except (OSError,ValueError,KeyError) as e:missing.append('global/'+name+': '+str(e))
    for seed in abc.SEEDS:
        work=root/('seed%d'%seed)
        for name in wf.PREP+wf.DEV+wf.EVAL:
            try:
                mark=check_marker(work,root/('progress/seed%d/%s.json'%(seed,name)));completed+=1
                costs.append(dict(seed=seed,stage=name,seconds=mark['seconds'],peak_gpu_bytes=mark.get('peak_gpu_bytes',0)))
            except (OSError,ValueError,KeyError) as e:missing.append('%d/%s: %s'%(seed,name,e))
        if (work/'runs/uniform/training_contract.json').exists():
            tc=read_json(work/'runs/uniform/training_contract.json')
            history=[read_json(work/'runs/uniform/epoch0.json')]+read_json(work/'runs/uniform/history.json')
            for h in history:curves.append(dict(seed=seed,epoch=h['epoch'],selected=h['epoch']==tc['selected_epoch'],
                loss=h.get('loss'),learning_rate=h.get('learning_rate'),seconds=h.get('seconds'),
                VAL_CW_AUC=h['val']['cw']['AUC_native'],VAL_OW_AUC=h['val']['ow']['AUC_native']))
        for fold,table in [(4,q4),(5,q5)]:
            path=work/('decision/fold%d/audit.json'%fold)
            if path.exists():table.extend(read_json(path)['rows'])
            path=work/('decision/fold%d_measure/audit.json'%fold)
            if path.exists():directional.extend(read_json(path)['rows'])
        for p in sorted((work/'recognition').glob('bank*/policies/*.json')):
            r=read_json(p)
            for phase in ('val','test'):
                for world in ('cw','ow'):
                    m=r[phase][world]
                    rows.append(dict(seed=seed,bank=r['bank'],policy=r['policy'],method=r['method'],eta=r['eta'],
                        permutation_seed=r['permutation_seed'],epoch=r['epoch'],phase=phase,world=world,
                        AUC_native=m['AUC_native'],AUC_exact=m['AUC_exact'],HM=m['at_val_bias']['HM'],
                        seen=m['at_val_bias']['seen'],unseen=m['at_val_bias']['unseen'],
                        raw_NLL=m['unbiased']['NLL'],NLL=m['calibrated']['NLL'],ECE=m['calibrated']['ECE'],
                        graph_variance=m['graph_variance'],seconds=r['seconds'],result_directory=r['result_directory']))
    paired=pair_rows(rows,('seed','bank','phase','world'),('AUC_native','AUC_exact','HM','raw_NLL','NLL'))
    q4paired=pair_rows(q4,('seed','bank'),('mixture_AUC_exact','mixture_unll','single_unll'))
    q5paired=pair_rows(q5,('seed','bank'),('mixture_AUC_exact','mixture_unll','single_unll'))
    agg={}
    agg['recognition']=aggregate(rows,('policy','phase','world'),('AUC_native','AUC_exact','HM','NLL'),len(cfg['eval_graph_seeds']))
    agg['paired']=aggregate(paired,('comparison','phase','world'),('AUC_native','AUC_exact','HM','NLL'),len(cfg['eval_graph_seeds']))
    for name,table in [('q4',q4),('q5',q5)]:
        agg[name]=aggregate(table,('policy',),('mixture_AUC_exact','mixture_NLL_gain','mixture_AUC_gain','single_NLL_gain'),len(cfg['audit_eval_seeds']))
    utility=[];labels=[]
    models=load_torch(root/'utility/models.pt') if (root/'utility/models.pt').exists() else None
    if models:
        ty=models['types']
        for seed in abc.SEEDS:
            w=root/('seed%d'%seed)
            for j in range(1,6):
                if not (w/('effects/fold%d.pt'%j)).exists():continue
                e=load_torch(w/('effects/fold%d.pt'%j))
                r=load_torch(w/('context/fold%d.pt'%j));repeat=load_torch(w/('responses/fold%d_repeat.pt'%j))
                predictions=abc.means(models,r,repeat)
                for method,p in predictions.items():
                    target=e['mixture' if method=='mix16' else 'single']
                    y=target['context'].numpy() if np.asarray(p).ndim==2 else target['target'].numpy()
                    for t in range(3):
                        ix=ty==t;pp=np.asarray(p)[...,ix].reshape(-1);yy=y[...,ix].reshape(-1)
                        if np.asarray(p).ndim==2:
                            use=np.repeat(np.asarray(target['weights'])>0,int(ix.sum()));pp=pp[use];yy=yy[use]
                        utility.append(dict(seed=seed,fold=j,partition='fit' if j<4 else 'dev' if j==4 else 'audit',
                            method=method,type=t,**compact.point_metrics(pp,yy)))
                for t in range(3):
                    ix=ty==t;a=e['single']['target'].numpy()[ix];b=e['mixture']['target'].numpy()[ix]
                    c=e['single']['context'].numpy()[:,ix];weights=e['single']['weights']
                    labels.append(dict(seed=seed,fold=j,type=t,unseen_context_counts=np.bincount(e['routes'][~e['sample_seen']].numpy(),minlength=2).tolist(),
                        target_spearman=compact.point_metrics(a,b)['spearman'],target_sign_change=float(np.mean(np.sign(a)!=np.sign(b))),
                        single_rms=float(np.sqrt(np.mean(a*a))),mixture_rms=float(np.sqrt(np.mean(b*b))),
                        context_variance=float(np.mean(np.sum(weights[:,None]*(c-a)**2,axis=0))),
                        context_support_empty_fraction=float(r['empty_blocks'][:,ix].float().mean())))
    for name,table in [('summary.csv',rows),('paired.csv',paired),('q4_curves.csv',q4),('q5_curves.csv',q5),
        ('q4_paired.csv',q4paired),('q5_paired.csv',q5paired),('training_curves.csv',curves),('costs.csv',costs),
        ('utility.csv',utility),('label_diagnostics.csv',labels),('directional_curves.csv',directional)]:
        write_csv(root/name,table)
    write_json(root/'aggregates.json',agg)
    expected_stages=len(wf.GLOBAL)+3*len(wf.PREP+wf.DEV+wf.EVAL)
    expected_formal=3*abc.formal_count(cfg)*len(cfg['eval_graph_seeds'])*4
    expected_q4=3*abc.curve_count(cfg)*len(cfg['audit_eval_seeds'])
    expected_q5=3*abc.curve_count(cfg,True)*len(cfg['audit_eval_seeds'])
    expected_directional=3*2*(1+len(abc.CURVE_METHODS)*(len(cfg['amplitudes'])-1))*len(cfg['audit_eval_seeds'])
    status=dict(completed_stages=completed,expected_stages=expected_stages,recognition_rows=len(rows),
        expected_recognition_rows=expected_formal,q4_rows=len(q4),expected_q4_rows=expected_q4,
        q5_rows=len(q5),expected_q5_rows=expected_q5,training_curve_rows=len(curves),expected_training_curve_rows=33,
        directional_rows=len(directional),expected_directional_rows=expected_directional,
        missing=missing,complete=not missing and completed==expected_stages and len(rows)==expected_formal and
        len(q4)==expected_q4 and len(q5)==expected_q5 and len(curves)==33 and len(directional)==expected_directional)
    write_json(root/'status.json',status)
    print('STATUS',completed,'/',expected_stages,'recognition',len(rows),'/',expected_formal,
        'Q4',len(q4),'/',expected_q4,'Q5',len(q5),'/',expected_q5,'directional',len(directional),'/',expected_directional,
        'curves',len(curves),'/33 complete',status['complete'],flush=True)
    return status
