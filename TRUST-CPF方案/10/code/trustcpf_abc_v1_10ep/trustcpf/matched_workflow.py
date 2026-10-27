"""Import -> three teacher queues -> pooled fit barrier -> three scoring queues."""
import copy,os,platform,shlex,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,read_json,write_json,load_torch,sha,source_hashes
from .runtime import lock
from .data import load_cache
from . import matched,matched_train as train,matched_compute as compute,align
from .state_workflow import code_digest,hashes,files_under,check_marker,write_csv

TRAIN_STAGES=tuple('teacher%d'%j for j in range(1,6))
EVAL_STAGES=('policies','q5','recognition')
CORE=('model.py','g0.py','engine.py','evaluation.py','metrics.py','response.py','compact.py','align.py','state_eval.py')
FORMAL=('reference/last.pt','runs/uniform/best.pt','runs/uniform/training_contract.json','runs/uniform/history.json',
        'runs/uniform/epoch0.json','inputs/bank.pt','folds.json')+tuple('responses/'+n+s+'.pt'
        for n in ('reference','uniform_best') for s in ('','_repeat'))

def setup(args):
    source=Path(args.source).resolve();root=Path(args.work).resolve();cfg=read_json(args.config)
    assert cfg['stage_epochs']==cfg['adapt_epochs']==10 and cfg['fixed_strength']==1.
    if not cfg.get('synthetic'):
        from .cache_validation import validate_cache
        print('CHECKING_CACHE',args.cache,flush=True);validate_cache(args.cache)
    assert read_json(source/'work/status.json')['complete'],'Complete State batch one first'
    source_identity=read_json(source/'work/progress/global/identity.json')
    assert source_identity['cache_manifest']==sha(Path(args.cache)/'features_manifest.json')
    for name in CORE:
        if code_digest(source/'trustcpf'/name)!=code_digest(ROOT/'trustcpf'/name):raise ValueError('Source classifier/math differs: '+name)
    pairs=[];fingerprints={};stops=[];bank_digest=None
    def add(src,dest):pairs.append((src,dest));fingerprints[str(dest.relative_to(root))]=sha(src)
    add(source/'work/utility/pooled_absolute/shared.pt',root/'utility/old.pt')
    add(source/'work/utility/selection.json',root/'utility/old_selection.json')
    for seed in matched.SEEDS:
        sw=source/'work'/('seed%d'%seed);w=root/('seed%d'%seed);oldcfg=read_json(sw/'recipe.json')
        for k in ('batch_size','graph_lr','weight_decay','lambda_w','gradient_clip','graphs','graph_seed','rank','message_rank',
                  'eta','alpha','tg','object_context','response_images_per_pair','response_backgrounds','response_repeat_offset'):
            if oldcfg[k]!=cfg[k]:raise ValueError('Source protocol differs: '+k)
        tc=read_json(sw/'runs/uniform/training_contract.json')
        assert tc['epochs_completed']==10 and tc['best_sha256']==sha(sw/'runs/uniform/best.pt')
        ref=load_torch(sw/'reference/last.pt')['model'];formal=load_torch(sw/'runs/uniform/best.pt')['model']
        assert all(torch.equal(v,formal[k]) for k,v in ref.items() if k.startswith('backbone.'))
        current_bank=sha(sw/'inputs/bank.pt')
        if bank_digest is None:bank_digest=current_bank
        assert current_bank==bank_digest,'Source relation bank/order differs between seeds'
        del ref,formal
        for name in FORMAL:add(sw/name,w/name)
        for j in range(1,6):
            s=read_json(sw/('splits/fold%d.json'%j));new=train.add_stopping_set(s,cfg)
            stops.append((seed,j,new))
            teacher=sw/('teachers/%d/last.pt'%j);response=load_torch(sw/('responses/fold%d.pt'%j));m=load_torch(sw/('measurements/fold%d.pt'%j))
            assert response['checkpoint_sha256']==m['teacher_sha256']==sha(teacher)
            assert load_torch(teacher)['train_ids']==s['train_ids']
            for name in ('teachers/%d/last.pt'%j,'responses/fold%d.pt'%j,'measurements/fold%d.pt'%j,'splits/fold%d.json'%j):
                add(sw/name,w/'old'/name)
        print('SOURCE_READY',seed,'formal checkpoint fixed; stopping images selected',flush=True)
    for j in range(1,6):
        splits=[s for _,f,s in stops if f==j];assert splits[0]==splits[1]==splits[2]
    identity=dict(source=str(source),source_files=fingerprints,config=cfg,implementation=source_hashes(),
        cache_manifest=sha(Path(args.cache)/'features_manifest.json'),knowledge=sha(ROOT/'assets/knowledge.pt'))
    with lock(root/'progress/global'):
        ip=root/'progress/global/identity.json'
        if ip.exists() and read_json(ip)!=identity:raise ValueError('Use a new work directory when inputs or implementation change')
        write_json(ip,identity);mark=root/'progress/global/import.json'
        if mark.exists():check_marker(root,mark)
        else:
            tic=time.perf_counter();outputs=[]
            for src,dest in pairs:
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);outputs.append(dest)
            for seed,j,s in stops:
                p=root/('seed%d/splits/fold%d.json'%(seed,j));write_json(p,s);outputs.append(p)
            for seed in matched.SEEDS:
                w=root/('seed%d'%seed);p=w/'recipe.json';write_json(p,dict(cfg,seed=seed));outputs.append(p)
                p=w/'reuse.json';write_json(p,dict(source=str(source/'work'/('seed%d'%seed)),
                    formal_training='none; exact batch-one Uniform-best',teacher_training='10 additional epochs per fold'));outputs.append(p)
            write_json(mark,dict(completed=True,seconds=time.perf_counter()-tic,outputs=hashes(root,outputs)))
    paths=dict(CPF_PY=sys.executable,CPF_SOURCE=str(source),CPF_CACHE=str(Path(args.cache).resolve()),CPF_WORK=str(root),CPF_CONFIG=str(Path(args.config).resolve()))
    (ROOT/'server_env.sh').write_bytes(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in paths.items()).encode('utf8'))
    write_json(ROOT/'setup.json',dict(paths=paths,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))
    print('SETUP_COMPLETE: next run teacher queues, pooled fit, then frozen scoring',flush=True)

def check_inputs(args,need_fit=False):
    root=Path(args.work).resolve();identity=read_json(root/'progress/global/identity.json')
    assert identity['config']==read_json(args.config) and identity['implementation']==source_hashes()
    assert identity['cache_manifest']==sha(Path(args.cache)/'features_manifest.json')
    check_marker(root,root/'progress/global/import.json')
    if need_fit:check_marker(root,root/'progress/global/fit.json')
    return root

def outputs_for(work,stage):
    if stage.startswith('teacher'):
        j=int(stage[-1]);return files_under(work,['teachers/%d'%j,'measurements/fold%d'%j])+[
            work/('responses/fold%d.pt'%j),work/('responses/fold%d_repeat.pt'%j),work/('measurements/fold%d.pt'%j)]
    if stage=='policies':return [work/'policies.pt',work/'responses/repeat_diagnostics.json']
    return files_under(work,['decision' if stage=='q5' else 'recognition'])

def run_stages(args,stages):
    root=check_inputs(args,stages==EVAL_STAGES);w=root/('seed%d'%args.seed);marks=root/'progress'/('seed%d'%args.seed)
    a=copy.copy(args);a.work=str(w)
    with lock(marks):
        for stage in stages:
            marker=marks/(stage+'.json')
            if marker.exists():check_marker(w,marker);print('REUSE_STAGE',args.seed,stage,flush=True);continue
            if str(args.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
            tic=time.perf_counter();print('START_STAGE',args.seed,stage,flush=True)
            if stage.startswith('teacher'):train.teacher(a,int(stage[-1]))
            else:{'policies':matched.make_policies,'q5':compute.q5_audit,'recognition':compute.recognition}[stage](a)
            align.synchronize(args.device);outputs=outputs_for(w,stage)
            write_json(marker,dict(completed=True,seconds=time.perf_counter()-tic,outputs=hashes(w,outputs),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else 0))
            print('END_STAGE',args.seed,stage,flush=True)
    print('SEED_PHASE_COMPLETE',args.seed,'train' if stages==TRAIN_STAGES else 'evaluate',flush=True)

def train_seed(args):return run_stages(args,TRAIN_STAGES)
def evaluate_seed(args):return run_stages(args,EVAL_STAGES)

def fit(args):
    root=check_inputs(args)
    with lock(root/'progress/global'):
        for seed in matched.SEEDS:
            for stage in TRAIN_STAGES:check_marker(root/('seed%d'%seed),root/('progress/seed%d/%s.json'%(seed,stage)))
        marker=root/'progress/global/fit.json'
        if marker.exists():check_marker(root,marker);print('REUSE_GLOBAL_FIT',flush=True);return
        tic=time.perf_counter();matched.fit_global(root,read_json(args.config))
        outputs=[root/('utility/'+n) for n in ('matched.pt','selection.json','audit.json','teacher_change.json')]
        write_json(marker,dict(completed=True,seconds=time.perf_counter()-tic,outputs=hashes(root,outputs)))

def report(args):
    root=Path(args.work).resolve();root.mkdir(parents=True,exist_ok=True)
    with lock(root/'progress/report'):return _report(root,read_json(args.config))

def _report(root,cfg):
    rows=[];qrows=[];curves=[];teachers=[];costs=[];deployment=[];repeats=[];missing=[];complete=0
    for folder,stages in [('global',('import','fit'))]+[('seed%d'%s,TRAIN_STAGES+EVAL_STAGES) for s in matched.SEEDS]:
        w=root if folder=='global' else root/folder
        for stage in stages:
            try:
                v=check_marker(w,root/'progress'/folder/(stage+'.json'));complete+=1
                costs.append(dict(seed=folder,stage=stage,seconds=v['seconds'],peak_gpu_bytes=v.get('peak_gpu_bytes',0)))
            except (OSError,ValueError,KeyError) as e:missing.append(folder+'/'+stage+': '+str(e))
    for seed in matched.SEEDS:
        w=root/('seed%d'%seed)
        for j in range(1,6):
            cp=w/('teachers/%d/contract.json'%j)
            if not cp.exists():continue
            contract=read_json(cp);h=read_json(cp.parent/'history.json');split=read_json(w/('splits/fold%d.json'%j))
            teachers.append(dict(seed=seed,fold=j,**contract))
            for r in h:curves.append(dict(seed=seed,fold=j,epoch=r['epoch'],selected=r['epoch']==contract['selected_epoch'],
                loss=r.get('loss'),learning_rate=r.get('learning_rate'),STOP_AUC_native=r['stop']['AUC_native'],STOP_AUC_exact=r['stop']['AUC_exact']))
        for policy in matched.formal_keys(cfg):
            for bank in cfg['eval_graph_seeds']:
                p=w/'recognition/uniform_best'/policy/('bank%d/metrics.json'%bank)
                if not p.exists():continue
                v=read_json(p)
                for phase in ('val','test'):
                    for world in ('cw','ow'):
                        m=v[phase][world]
                        rows.append(dict(seed=seed,policy=policy,method=v['method'],permutation_seed=v['permutation_seed'],bank=bank,
                            phase=phase,world=world,epoch=v['epoch'],AUC_native=m['AUC_native'],AUC_exact=m['AUC_exact'],HM=m['at_val_bias']['HM'],
                            seen=m['at_val_bias']['seen'],unseen=m['at_val_bias']['unseen'],NLL=m['calibrated']['NLL'],raw_NLL=m['unbiased']['NLL'],ECE=m['calibrated']['ECE']))
                for t,d in v['deployment'].items():deployment.append(dict(seed=seed,policy=policy,bank=bank,type=int(t),
                    expected_edges=d['expected_edges'],probability_std=d['probability_std'],kl=d['kl'],
                    expected_covered_nodes=sum(d['expected_node_coverage']),prototype_displacement=d['prototype_displacement_mean']))
        if (w/'decision/audit.json').exists():qrows+=read_json(w/'decision/audit.json')['rows']
        if (w/'responses/repeat_diagnostics.json').exists():repeats+=read_json(w/'responses/repeat_diagnostics.json')
    index={(r['seed'],r['policy'],r['bank'],r['phase'],r['world']):r for r in rows};paired=[]
    for r in rows:
        if r['permutation_seed'] is not None or r['policy']=='uniform':continue
        name=r['policy'];controls={'uniform':['uniform']}
        if name in matched.METHODS:controls['permutation_mean']=[name+'_perm%d'%p for p in cfg['permutation_seeds']]
        if name=='matched_single':controls.update(old_single=['old_single'],cross_response=['matched_cross'])
        if name=='matched_average':controls.update(old_average=['old_average'],single=['matched_single'],cross_response=['matched_average_cross'])
        if name=='old_average':controls['single']=['old_single']
        for control,names in controls.items():
            refs=[index.get((r['seed'],n,r['bank'],r['phase'],r['world'])) for n in names]
            if any(x is None for x in refs):continue
            a={k:r[k] for k in ('seed','bank','phase','world')};a['comparison']=name+'-minus-'+control
            for metric in ('AUC_native','AUC_exact','HM'):a[metric]=r[metric]-float(np.mean([v[metric] for v in refs]))
            a['raw_NLL_gain']=float(np.mean([v['raw_NLL'] for v in refs]))-r['raw_NLL'];paired.append(a)
    qi={(r['seed'],r['policy'],r['bank']):r for r in qrows};qp=[]
    for r in qrows:
        name=r['policy']
        if r['permutation_seed'] is not None or name=='uniform':continue
        controls={'uniform':['uniform'],'permutation_mean':[name+'_perm%d'%p for p in cfg['permutation_seeds']]}
        if name=='matched_single':controls['old_single']=['old_single']
        if name=='matched_average':controls.update(old_average=['old_average'],single=['matched_single'])
        if name=='old_average':controls['single']=['old_single']
        for control,names in controls.items():
            refs=[qi.get((r['seed'],n,r['bank'])) for n in names]
            if any(v is None for v in refs):continue
            a=dict(seed=r['seed'],bank=r['bank'],comparison=name+'-minus-'+control)
            for metric in ('AUC_native','AUC_exact'):a[metric]=r['mixture_'+metric]-float(np.mean([v['mixture_'+metric] for v in refs]))
            a['NLL_gain']=float(np.mean([v['mixture_unll'] for v in refs]))-r['mixture_unll'];qp.append(a)
    utility=[]
    if (root/'utility/audit.json').exists():
        for r in read_json(root/'utility/audit.json'):
            for t,v in [('all',r['all'])]+list(r['by_type'].items()):utility.append(dict(seed=r['seed'],method=r['method'],type=t,
                linear_gain=r['linear_gain'],centered_r2=r['centered']['r2'] if t=='all' else None,**v))
    tables={'summary.csv':rows,'paired.csv':paired,'q5_decisions.csv':qrows,'q5_paired.csv':qp,'teacher_curves.csv':curves,
        'teachers.csv':teachers,'utility.csv':utility,'response_repeats.csv':repeats,'deployment.csv':deployment,'costs.csv':costs}
    if (root/'utility/teacher_change.json').exists():tables['teacher_change.csv']=read_json(root/'utility/teacher_change.json')
    for name,table in tables.items():write_csv(root/name,table)
    aggregates=[]
    for label,table,keys,metrics in [
        ('recognition',rows,('policy','bank','phase','world'),('AUC_native','AUC_exact','HM')),
        ('recognition_bankmean',rows,('policy','phase','world'),('AUC_native','AUC_exact','HM')),
        ('paired_bankmean',paired,('comparison','phase','world'),('AUC_native','AUC_exact','HM')),
        ('q5_bankmean',qrows,('policy',),('mixture_AUC_exact','mixture_AUC_gain','mixture_NLL_gain')),
        ('q5_paired_bankmean',qp,('comparison',),('AUC_exact','AUC_native','NLL_gain'))]:
        groups={}
        for r in table:groups.setdefault(tuple(r[k] for k in keys),{}).setdefault(r['seed'],[]).append(r)
        nb=1 if label=='recognition' else len(cfg['audit_eval_seeds'] if label.startswith('q5') else cfg['eval_graph_seeds'])
        for key,byseed in groups.items():
            if set(byseed)!=set(matched.SEEDS) or any(len(v)!=nb for v in byseed.values()):continue
            a=dict(table=label,**dict(zip(keys,key)),seeds=list(matched.SEEDS))
            for metric in metrics:
                values=[float(np.mean([v[metric] for v in byseed[s]])) for s in matched.SEEDS]
                a[metric+'_mean']=float(np.mean(values));a[metric+'_sd']=float(np.std(values,ddof=1));a[metric+'_by_seed']=values
            aggregates.append(a)
    write_json(root/'aggregates.json',aggregates)
    expected=3*len(matched.formal_keys(cfg))*len(cfg['eval_graph_seeds'])*4;eq=3*len(matched.q5_keys(cfg))*len(cfg['audit_eval_seeds'])
    status=dict(completed_stages=complete,expected_stages=26,teachers=len(teachers),expected_teachers=15,
        recognition_rows=len(rows),expected_recognition_rows=expected,q5_rows=len(qrows),expected_q5_rows=eq,
        teacher_curve_rows=len(curves),expected_teacher_curve_rows=165,missing=missing,
        complete=complete==26 and len(teachers)==15 and len(rows)==expected and len(qrows)==eq and len(curves)==165)
    write_json(root/'status.json',status)
    print('STATUS',complete,'/26 teachers',len(teachers),'/15 recognition',len(rows),'/',expected,'Q5',len(qrows),'/',eq,
        'curves',len(curves),'/165 complete',status['complete'],flush=True)
    return status
