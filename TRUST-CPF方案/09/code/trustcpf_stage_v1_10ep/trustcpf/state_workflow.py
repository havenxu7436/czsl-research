"""Global import/fit followed by three independent 10-epoch worker queues."""
import copy
import csv
import hashlib
import os
import platform
import shlex
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,read_json,write_json,load_torch,sha,source_hashes
from .runtime import lock
from .data import load_cache
from . import state,align,state_compute as compute

STAGES=('uniform10','responses','policies','q5','recognition')
CORE=('model.py','g0.py','engine.py','evaluation.py','metrics.py','response.py','compact.py','r5.py','align.py')
INPUTS=('reference/last.pt','responses/reference.pt','folds.json','inputs/bank.pt')+tuple(
    '%s/fold%d.%s'%(folder,j,ext) for j in range(1,6)
    for folder,ext in (('responses','pt'),('measurements','pt'),('splits','json')))+tuple(
    'teachers/%d/last.pt'%j for j in range(1,6))

def code_digest(p):
    return hashlib.sha256(Path(p).read_text(encoding='utf-8-sig').encode('utf8')).hexdigest()

def validate_seed(source,seed,cache,cfg):
    w=Path(source)/'work'/('seed%d'%seed)
    old=read_json(w/'recipe.json');manifest,_=load_cache(cache)
    ref=load_torch(w/'reference/last.pt');bank=load_torch(w/'inputs/bank.pt')
    assert cfg['adapt_epochs']==10,'This release trains exactly 10 epochs'
    for name in CORE:
        if code_digest(Path(source)/'trustcpf'/name)!=code_digest(ROOT/'trustcpf'/name):
            raise ValueError('Source classifier/response math differs: '+name)
    for key in ('batch_size','graph_lr','weight_decay','lambda_w','gradient_clip','graphs','graph_seed',
                'rank','message_rank','eta','alpha','tg','object_context','response_images_per_pair','response_backgrounds'):
        if cfg[key]!=old[key]:raise ValueError('Matched source configuration differs: '+key)
    assert set(map(tuple,ref['train_pairs']))==set(map(tuple,manifest['pairs']['train']))
    for key in ('anchors','source','target','types','degree'):
        assert torch.equal(ref['model']['graph.'+key],bank[key]),key
    original=Path(source)/'work/progress'/('seed%d'%seed)/'identity.json'
    if original.exists():
        assert read_json(original)['cache_manifest']==sha(Path(cache)/'features_manifest.json'),'Feature cache identity differs'
    for j in range(1,6):
        cp=w/('teachers/%d/last.pt'%j);sp=read_json(w/('splits/fold%d.json'%j))
        r=load_torch(w/('responses/fold%d.pt'%j));m=load_torch(w/('measurements/fold%d.pt'%j))
        assert r['checkpoint_sha256']==m['teacher_sha256']==sha(cp)
        assert load_torch(cp)['train_ids']==sp['train_ids'] and m['split']==sp
        assert set(r['support_indices'])<=set(sp['train_ids'])
        assert list(r['names'])==list(compute.response.FEATURE_NAMES)
        assert 'unll' in m['targets'] and r['features'].shape==(len(bank['types']),8)
        np.testing.assert_allclose(np.asarray(m['blocks']).mean(0),m['target'],atol=1e-12)
    response=load_torch(w/'responses/reference.pt')
    assert response['checkpoint_sha256']==sha(w/'reference/last.pt')
    assert all(k in response for k in ('support_indices','weights','seed','train_pairs','names'))
    return w

def hashes(root,files):
    return {Path(p).relative_to(root).as_posix():sha(p) for p in files}

def files_under(root,folders):
    return sorted(p for folder in folders for p in (root/folder).rglob('*')
                  if p.is_file() and not p.name.endswith(('.writing','.lock')) and '__pycache__' not in p.parts)

def check_marker(root,path):
    v=read_json(path)
    for rel,h in v['outputs'].items():
        if not (root/rel).is_file() or sha(root/rel)!=h:
            raise ValueError('Completed output changed: '+str(root/rel))
    return v

def setup(args):
    root=Path(args.work).resolve();cfg=read_json(args.config)
    if cfg['adapt_epochs']!=10:raise ValueError('adapt_epochs must be 10')
    if cfg['fixed_strength']!=1.:raise ValueError('This batch fixes kappa=1')
    if not cfg.get('synthetic'):
        from .cache_validation import validate_cache
        print('CHECKING_CACHE',args.cache,flush=True);validate_cache(args.cache)
    with lock(root/'progress/global'):
        allfiles={};sources={}
        for seed in state.SEEDS:
            sources[seed]=validate_seed(args.source,seed,args.cache,cfg)
            allfiles[str(seed)]={p:sha(sources[seed]/p) for p in INPUTS+('recipe.json',)}
            print('SOURCE_READY',seed,'five teachers, paired U-NLL and reference',flush=True)
        ident=dict(config=cfg,source=str(Path(args.source).resolve()),source_files=allfiles,
            implementation=source_hashes(),knowledge=sha(ROOT/'assets/knowledge.pt'),
            initial={str(s):sha(ROOT/'initial'/('g0_seed%d.pt'%s)) for s in state.SEEDS},
            cache_manifest=sha(Path(args.cache)/'features_manifest.json'))
        identity=root/'progress/global/identity.json'
        if identity.exists() and read_json(identity)!=ident:
            raise ValueError('Use a new work directory for changed inputs or configuration')
        write_json(identity,ident)
        for stage in ('import','fit'):
            marker=root/'progress/global'/(stage+'.json')
            if marker.exists():check_marker(root,marker);print('REUSE_GLOBAL',stage,flush=True);continue
            tic=time.perf_counter()
            if stage=='import':
                outputs=[]
                for seed,src in sources.items():
                    work=root/('seed%d'%seed)
                    for name in INPUTS:
                        dest=work/name;dest.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(src/name,dest);outputs.append(dest)
                    recipe=dict(read_json(src/'recipe.json'),**cfg);recipe.update(seed=seed,variant='uniform')
                    write_json(work/'recipe.json',recipe);outputs.append(work/'recipe.json')
                    write_json(work/'reuse.json',dict(source=str(src),source_files=allfiles[str(seed)],
                        uniform='new 10-epoch training from imported reference',
                        teachers='imported paired supervision; no teacher retraining in batch one'))
                    outputs.append(work/'reuse.json')
                banks=[load_torch(root/('seed%d/inputs/bank.pt'%s)) for s in state.SEEDS]
                for b in banks[1:]:
                    for k in ('anchors','source','target','types','degree'):
                        assert torch.equal(b[k],banks[0][k]),'Cross-seed edge identity differs'
                for j in range(1,6):
                    splits=[read_json(root/('seed%d/splits/fold%d.json'%(s,j))) for s in state.SEEDS]
                    assert splits[0]==splits[1]==splits[2],'Cross-seed fold definitions differ'
            else:
                state.fit_global(root,cfg);outputs=files_under(root,['utility'])
            write_json(marker,dict(completed=True,seconds=time.perf_counter()-tic,outputs=hashes(root,outputs)))
    paths=dict(CPF_PY=sys.executable,CPF_SOURCE=str(Path(args.source).resolve()),CPF_CACHE=str(Path(args.cache).resolve()),
        CPF_WORK=str(root),CPF_CONFIG=str(Path(args.config).resolve()))
    (ROOT/'server_env.sh').write_bytes(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in paths.items()).encode('utf8'))
    write_json(ROOT/'setup.json',dict(paths=paths,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))
    print('SETUP_COMPLETE: 4 ridge cells ready; next run three 10-epoch Uniform workers',flush=True)

def stage_outputs(work,stage):
    if stage=='uniform10':return files_under(work,['runs/uniform'])
    if stage=='responses':
        return [work/('responses/'+name) for name in
                ('uniform_best.pt','reference_repeat.pt','uniform_best_repeat.pt','fold5_repeat.pt')]
    if stage=='policies':return [work/'policies.pt',work/'responses/repeat_diagnostics.json',work/'responses/transfer.json']
    return files_under(work,['decision' if stage=='q5' else 'recognition'])

def run_seed(args):
    root=Path(args.work).resolve();work=root/('seed%d'%args.seed);marks=root/'progress'/('seed%d'%args.seed)
    for stage in ('import','fit'):check_marker(root,root/'progress/global'/(stage+'.json'))
    identity=read_json(root/'progress/global/identity.json')
    if identity['implementation']!=source_hashes():raise ValueError('Implementation changed after setup')
    if identity['config']!=read_json(args.config):raise ValueError('Configuration changed after setup')
    if identity['cache_manifest']!=sha(Path(args.cache)/'features_manifest.json'):raise ValueError('Feature cache changed')
    a=copy.copy(args);a.work=str(work)
    functions=dict(uniform10=compute.train_uniform,responses=compute.responses,policies=compute.make_policies,
                   q5=compute.q5_audit,recognition=compute.recognition)
    with lock(marks):
        for stage in STAGES:
            marker=marks/(stage+'.json')
            if marker.exists():check_marker(work,marker);print('REUSE_STAGE',args.seed,stage,flush=True);continue
            align.synchronize(args.device);tic=time.perf_counter()
            if str(args.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
            print('START_STAGE',args.seed,stage,flush=True);functions[stage](a);align.synchronize(args.device)
            outputs=stage_outputs(work,stage)
            if not outputs:raise RuntimeError('No outputs for '+stage)
            write_json(marker,dict(completed=True,seconds=time.perf_counter()-tic,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else 0,
                outputs=hashes(work,outputs)))
            print('END_STAGE',args.seed,stage,flush=True)
        write_json(marks/'DONE.json',dict(seed=args.seed,completed=True,stages=len(STAGES),epochs=10))
    print('SEED_COMPLETE',args.seed,flush=True)

def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def report(args):
    # Manual status checks and the queue's final report share the same outputs.
    with lock(Path(args.work).resolve()/'progress/report'):
        return _report(args)

def _report(args):
    root=Path(args.work).resolve();root.mkdir(parents=True,exist_ok=True);cfg=read_json(args.config)
    rows=[];decision=[];curves=[];costs=[];deployment=[];repeats=[];transfers=[];missing=[];completed=0
    for stage in ('import','fit'):
        marker=root/'progress/global'/(stage+'.json')
        try:
            record=check_marker(root,marker);completed+=1
            costs.append(dict(seed='global',stage=stage,seconds=record['seconds'],peak_gpu_bytes=0))
        except (OSError,ValueError,KeyError) as e:missing.append('global/'+stage+': '+str(e))
    for seed in state.SEEDS:
        work=root/('seed%d'%seed);marks=root/'progress'/('seed%d'%seed)
        for stage in STAGES:
            try:
                v=check_marker(work,marks/(stage+'.json'));completed+=1
                costs.append(dict(seed=seed,stage=stage,seconds=v['seconds'],peak_gpu_bytes=v.get('peak_gpu_bytes',0)))
            except (OSError,ValueError,KeyError) as e:missing.append('%d/%s: %s'%(seed,stage,e))
        contract=work/'runs/uniform/training_contract.json'
        if contract.exists():
            tc=read_json(contract);history=[read_json(work/'runs/uniform/epoch0.json')]+read_json(work/'runs/uniform/history.json')
            for h in history:
                curves.append(dict(seed=seed,epoch=h['epoch'],selected=h['epoch']==tc['selected_epoch'],
                    loss=h.get('loss'),learning_rate=h.get('learning_rate'),
                    VAL_CW_AUC=h['val']['cw']['AUC_native'],VAL_OW_AUC=h['val']['ow']['AUC_native']))
        for name in state.STATES:
            for key in state.recognition_keys(name,cfg):
                for gs in cfg['eval_graph_seeds']:
                    file=work/'recognition'/name/key/('bank%d'%gs)/'metrics.json'
                    if not file.exists():continue
                    v=read_json(file)
                    for phase in ('val','test'):
                        for world in ('cw','ow'):
                            m=v[phase][world]
                            rows.append(dict(seed=seed,state=name,policy=key,method=v['method'],permutation_seed=v['permutation_seed'],
                                bank=gs,phase=phase,world=world,epoch=v['epoch'],
                                AUC_native=m['AUC_native'],AUC_exact=m['AUC_exact'],HM=m['at_val_bias']['HM'],
                                seen=m['at_val_bias']['seen'],unseen=m['at_val_bias']['unseen'],
                                NLL=m['calibrated']['NLL'],raw_NLL=m['unbiased']['NLL'],ECE=m['calibrated']['ECE']))
                    for ty,d in v['deployment'].items():
                        deployment.append(dict(seed=seed,state=name,policy=key,bank=gs,type=int(ty),
                            expected_edges=d['expected_edges'],probability_std=d['probability_std'],kl=d['kl'],
                            expected_covered_nodes=sum(d['expected_node_coverage']),
                            prototype_displacement=d['prototype_displacement_mean'],
                            node_indegrees=d['expected_node_indegree']))
        if (work/'decision/audit.json').exists():decision.extend(read_json(work/'decision/audit.json')['rows'])
        if (work/'responses/repeat_diagnostics.json').exists():repeats.extend(read_json(work/'responses/repeat_diagnostics.json'))
        if (work/'responses/transfer.json').exists():transfers.extend(read_json(work/'responses/transfer.json'))
    index={(r['seed'],r['state'],r['policy'],r['bank'],r['phase'],r['world']):r for r in rows}
    paired=[]
    for r in rows:
        if r['permutation_seed'] is not None or r['policy']=='uniform':continue
        basekey=(r['seed'],r['state'],r['bank'],r['phase'],r['world'])
        comparisons={'uniform':['uniform']}
        if r['policy'] in state.METHODS:
            comparisons['permutation_mean']=[r['policy']+'_perm%d'%p for p in cfg['permutation_seeds']]
        if r['policy']=='pooled_absolute':
            comparisons['independent_absolute']=['independent_absolute']
            comparisons['cross_response']=['pooled_absolute_cross_response']
        if r['policy']=='pooled_relative':comparisons['pooled_absolute']=['pooled_absolute']
        for control,names in comparisons.items():
            refs=[index.get((basekey[0],basekey[1],n,basekey[2],basekey[3],basekey[4])) for n in names]
            if any(v is None for v in refs):continue
            row={k:r[k] for k in ('seed','state','bank','phase','world')}
            row['comparison']=r['policy']+'-minus-'+control
            for k in ('AUC_native','AUC_exact','HM'):
                row[k]=r[k]-float(np.mean([v[k] for v in refs]))
            row['raw_NLL_gain']=float(np.mean([v['raw_NLL'] for v in refs]))-r['raw_NLL']
            paired.append(row)
    aggregates=[]
    for label,table,keys in [('recognition',rows,('state','policy','bank','phase','world')),
                             ('paired',paired,('state','comparison','bank','phase','world'))]:
        groups={}
        for r in table:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
        for key,group in groups.items():
            if len(group)!=3:continue
            item=dict(table=label,**dict(zip(keys,key)),seeds=[r['seed'] for r in group])
            for metric in ('AUC_native','AUC_exact','HM'):
                vals=[r[metric] for r in group]
                item[metric+'_mean']=float(np.mean(vals));item[metric+'_sd']=float(np.std(vals,ddof=1))
            aggregates.append(item)
    # Average graph banks within each seed before estimating across-seed spread.
    for label,table,keys,metrics in [
        ('recognition_bankmean',rows,('state','policy','phase','world'),('AUC_native','AUC_exact','HM')),
        ('paired_bankmean',paired,('state','comparison','phase','world'),('AUC_native','AUC_exact','HM')),
        ('q5_bankmean',decision,('policy',),('mixture_AUC_exact','mixture_AUC_gain','mixture_NLL_gain'))]:
        within={}
        for r in table:within.setdefault((r['seed'],)+tuple(r[k] for k in keys),[]).append(r)
        groups={}
        for key,rr in within.items():
            expected_banks=len(cfg['audit_eval_seeds'] if label=='q5_bankmean' else cfg['eval_graph_seeds'])
            if len(rr)!=expected_banks:continue
            avg={m:float(np.mean([r[m] for r in rr])) for m in metrics}
            groups.setdefault(key[1:],[]).append(dict(seed=key[0],**avg))
        for key,rr in groups.items():
            if len(rr)!=3:continue
            item=dict(table=label,**dict(zip(keys,key)),seeds=[r['seed'] for r in rr])
            for m in metrics:
                item[m+'_mean']=float(np.mean([r[m] for r in rr]))
                item[m+'_sd']=float(np.std([r[m] for r in rr],ddof=1))
            aggregates.append(item)
    qindex={(r['seed'],r['policy'],r['bank']):r for r in decision};qpaired=[]
    for r in decision:
        if r['permutation_seed'] is not None or r['policy']=='uniform':continue
        controls={'uniform':['uniform'],
                  'permutation_mean':[r['policy']+'_perm%d'%p for p in cfg['permutation_seeds']]}
        for control,names in controls.items():
            refs=[qindex.get((r['seed'],n,r['bank'])) for n in names]
            if any(v is None for v in refs):continue
            qpaired.append(dict(seed=r['seed'],bank=r['bank'],comparison=r['policy']+'-minus-'+control,
                AUC_exact=r['mixture_AUC_exact']-float(np.mean([v['mixture_AUC_exact'] for v in refs])),
                AUC_native=r['mixture_AUC_native']-float(np.mean([v['mixture_AUC_native'] for v in refs])),
                NLL_gain=float(np.mean([v['mixture_unll'] for v in refs]))-r['mixture_unll']))
    utility=[]
    if (root/'utility/audit.json').exists():
        for r in read_json(root/'utility/audit.json'):
            for ty,m in [('all',r['all'])]+list(r['by_type'].items()):
                utility.append(dict(seed=r['seed'],cell=r['cell'],type=ty,lam=r['lam'],alpha=r['alpha'],
                    linear_gain=r['linear_gain'],centered_r2=r['centered']['r2'] if ty=='all' else None,**m))
    tables={'summary.csv':rows,'paired.csv':paired,'q5_decisions.csv':decision,'training_curves.csv':curves,
            'costs.csv':costs,'deployment.csv':deployment,'response_repeats.csv':repeats,'response_transfer.csv':transfers,
            'q5_paired.csv':qpaired,'utility.csv':utility}
    for name,values in tables.items():write_csv(root/name,values)
    write_json(root/'aggregates.json',aggregates)
    expected=3*sum(len(state.recognition_keys(s,cfg)) for s in state.STATES)*len(cfg['eval_graph_seeds'])*4
    expected_q5=3*len(state.q5_keys(cfg))*len(cfg['audit_eval_seeds'])
    status=dict(completed_stages=completed,expected_stages=17,recognition_rows=len(rows),expected_recognition_rows=expected,
        q5_rows=len(decision),expected_q5_rows=expected_q5,training_curve_rows=len(curves),expected_training_curve_rows=33,
        missing=missing,complete=not missing and completed==17 and len(rows)==expected and len(decision)==expected_q5 and len(curves)==33)
    write_json(root/'status.json',status)
    print('STATUS',completed,'/17','recognition',len(rows),'/',expected,'Q5',len(decision),'/',expected_q5,
          'curves',len(curves),'/33','complete',status['complete'],flush=True)
    for row in aggregates:
        if row['table']=='recognition' and row['state']=='uniform_best' and row['bank']==cfg['graph_seed'] and row['phase']=='test' and row['policy'] in ('uniform',)+state.METHODS:
            print('MAIN',row['policy'],row['world'],round(row['AUC_native_mean'],6),flush=True)
    return status

