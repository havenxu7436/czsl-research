import copy,csv,os,platform,shlex,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,read_json,write_json,load_torch,save_torch,sha,source_hashes,digest
from .runtime import lock
from .graph import solve_budget_equal

SEEDS=(3351,3352,3353)
STAGES=('import','responses','fit','audit','fit_all','structure','adapt')
VARIANTS=('drop','original','conditioned')
CONTROL_FILES=('best.pt','last.pt','bank.pt','history.json','training.json','val_predictions.npz','test_predictions.npz',
               'val_records.json','test_records.json','prediction_axes.json')


def source_seed(args):
    return Path(args.source).resolve()/'work'/('seed%d'%args.seed)


def setup(args):
    from .cache_validation import validate_cache
    source=Path(args.source).resolve();cache=Path(args.cache).resolve()
    status=read_json(source/'work/status.json')
    if not status.get('complete'):raise ValueError('Strong v1 source run is incomplete')
    for name in ('model.py','g0.py','engine.py','evaluation.py','metrics.py'):
        if sha(source/'trustcpf'/name)!=sha(ROOT/'trustcpf'/name):
            raise ValueError('Source classifier/evaluator differs: '+name)
    for seed in SEEDS:
        name='g0_seed%d.pt'%seed
        if sha(source/'initial'/name)!=sha(ROOT/'initial'/name):raise ValueError('G0 checkpoint differs')
    if sha(source/'assets/knowledge.pt')!=sha(ROOT/'assets/knowledge.pt'):raise ValueError('Knowledge bank differs')
    if (ROOT/'MANIFEST.json').exists():
        for name,h in read_json(ROOT/'MANIFEST.json').items():
            if sha(ROOT/name)!=h:raise ValueError('Release file differs: '+name)
    print('CHECKING_EXISTING_FEATURE_CACHE',cache,flush=True)
    validate_cache(cache)
    paths=dict(CPF_PY=sys.executable,CPF_SOURCE=str(source),CPF_CACHE=str(cache),CPF_WORK=str(Path(args.work).resolve()))
    (ROOT/'server_env.sh').write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in paths.items()),encoding='utf8')
    write_json(ROOT/'setup.json',dict(paths=paths,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        cuda=torch.cuda.is_available(),gpu_count=torch.cuda.device_count(),source_status=status))
    print('SETUP_COMPLETE',paths,flush=True)
    return paths


def input_identity(args):
    src=source_seed(args)
    files=['reference/last.pt','structure.pt','structure.json','tuning.json','folds.json','utility/audit.json',
           'utility/audit_predictions.pt','utility/selection.json','runs/drop/metrics.json','runs/robust/metrics.json']
    files+=['measurements/fold%d.pt'%j for j in range(1,6)]
    files+=['teachers/%d/last.pt'%j for j in range(1,6)]
    files+=['runs/'+v+'/'+f for v in ('drop','robust') for f in CONTROL_FILES if (src/'runs'/v/f).exists()]
    return dict(seed=args.seed,source_seed=str(src),source_hashes={f:sha(src/f) for f in files},
        implementation=source_hashes(),response_config=read_json(args.config),
        cache_manifest=sha(Path(args.cache)/'features_manifest.json'),
        knowledge=sha(ROOT/'assets/knowledge.pt'),
        initial=sha(ROOT/'initial'/('g0_seed%d.pt'%args.seed)))


def prepare_seed(args):
    src=source_seed(args);work=Path(args.work)
    state=load_torch(src/'reference/last.pt')
    cfg=copy.deepcopy(state['cfg']);new=read_json(args.config)
    cfg.update({k:new[k] for k in ('response_images_per_pair','response_backgrounds','response_seed')})
    cfg.update(seed=args.seed,variant='conditioned',utility_input='model_response')
    cfg.update(read_json(src/'tuning.json')['selected'])
    old=load_torch(src/'structure.pt')
    bank=load_torch(src/'runs/robust/bank.pt');types=bank['types']
    pi=load_torch(src/'runs/robust/best.pt')['pi']
    if not torch.allclose(pi,old['pi'],atol=1e-6,rtol=0):raise ValueError('Source robust pi differs from its structure')
    budgets={str(t):float(pi[types==t].double().sum()) for t in types.unique().tolist()}
    recovered,gamma=solve_budget_equal(old['reward'],types,cfg['tau_absolute'],budgets)
    error=float((recovered-pi).abs().max())
    if error>1e-6:raise ValueError('Original selection not recovered by equal-budget solve')
    drop=load_torch(src/'runs/drop/best.pt')['pi']
    budget_error=max(abs(float(drop[types==t].double().sum())-budgets[str(t)]) for t in types.unique().tolist())
    if budget_error>1e-5:raise ValueError('Archived drop budget differs')
    cfg['fixed_type_budgets']=budgets
    work.mkdir(parents=True,exist_ok=True)
    write_json(work/'recipe.json',cfg)
    write_json(work/'reuse.json',dict(source_seed=str(src),reference_sha256=sha(src/'reference/last.pt'),
        measurement_source=str(src/'measurements'),controls=dict(drop='drop',original='robust')))
    write_json(work/'budget_recovery.json',dict(budgets=budgets,max_pi_error=error,
        drop_max_budget_error=budget_error,multipliers=gamma))
    (work/'reference').mkdir(exist_ok=True)
    shutil.copy2(src/'reference/last.pt',work/'reference/last.pt')
    shutil.copy2(src/'tuning.json',work/'tuning.json')
    for oldname,newname in [('drop','drop'),('robust','original')]:
        olddir=src/'runs'/oldname;dest=work/'runs'/newname;dest.mkdir(parents=True,exist_ok=True)
        for name in CONTROL_FILES:
            if (olddir/name).exists():shutil.copy2(olddir/name,dest/name)
        m=read_json(olddir/'metrics.json')
        m.update(variant=newname,reused=True,source_variant=oldname,source_metrics_sha256=sha(olddir/'metrics.json'))
        write_json(dest/'metrics.json',m)
    dest=work/'utility_original';dest.mkdir(exist_ok=True)
    for name in ('audit.json','audit_predictions.pt','selection.json'):
        shutil.copy2(src/'utility'/name,dest/name)
    write_json(work/'reference_benchmarks.json',{v:read_json(src/'runs'/v/'metrics.json') for v in ('r0','capacity','dense') if (src/'runs'/v/'metrics.json').exists()})


def run_seed(args):
    from . import stages
    from .response import build_responses
    root=Path(args.work).resolve();folder=root/('seed%d'%args.seed);marks=root/'progress'/('seed%d'%args.seed)
    with lock(marks):
        identity=input_identity(args)
        if (marks/'identity.json').exists() and read_json(marks/'identity.json')!=identity:
            raise ValueError('Recipe/source changed: use a new --work directory')
        write_json(marks/'identity.json',identity)
        a=copy.copy(args);a.work=str(folder);a.variant='conditioned';a.quick=False;a.fold='all';a.fit_all=False
        for stage in STAGES:
            marker=marks/(stage+'.json')
            if marker.exists():
                for rel,h in read_json(marker).get('outputs',{}).items():
                    if not (folder/rel).is_file() or sha(folder/rel)!=h:raise ValueError('Completed stage output changed: '+rel)
                print('REUSE_STAGE',args.seed,stage,flush=True);continue
            print('START_STAGE',args.seed,stage,flush=True);tic=time.perf_counter()
            if stage=='import':prepare_seed(a)
            else:
                a.config=str(folder/'recipe.json')
                a.fit_all=stage=='fit_all'
                {'responses':build_responses,'fit':stages.fit_utility,'audit':stages.utility_audit,
                 'fit_all':stages.fit_utility,'structure':stages.structure,'adapt':stages.adapt}[stage](a)
            outputs={
                'import':['recipe.json','reuse.json','budget_recovery.json','reference/last.pt','runs/drop/metrics.json','runs/original/metrics.json'],
                'responses':['responses/'+x+'.pt' for x in ['fold1','fold2','fold3','fold4','fold5','reference']],
                'fit':['utility/selection.json']+['utility/model%d.pt'%i for i in range(read_json(folder/'recipe.json')['utility_models'])],
                'audit':['utility/audit.json','utility/audit_predictions.pt'],
                'fit_all':['utility_all/selection.json']+['utility_all/model%d.pt'%i for i in range(read_json(folder/'recipe.json')['utility_models'])],
                'structure':['structure.json','structure.pt'],
                'adapt':['runs/conditioned/'+x for x in ('metrics.json','best.pt','last.pt','val_predictions.npz','test_predictions.npz')]
            }[stage]
            write_json(marker,dict(stage=stage,seed=args.seed,seconds=time.perf_counter()-tic,
                outputs={p:sha(folder/p) for p in outputs},completed=True))
            print('END_STAGE',args.seed,stage,flush=True)
        write_json(marks/'DONE.json',dict(seed=args.seed,completed=len(STAGES),expected=len(STAGES)))
    print('SEED_COMPLETE',args.seed,flush=True)


def conditional_diagnostics(folder):
    with np.load(Path(folder)/'test_predictions.npz') as a:
        labels=read_json(Path(folder)/'prediction_axes.json');na=len(labels['attrs'])
        row=np.arange(len(a['attrs']));objs=a['objs'];attrs=a['attrs'];unseen=~a['cw_seen'][a['cw_truth']]
        cond=a['joint'][row,objs];pred=cond.argmax(-1)
        ids=a['cw_candidate_ids'][~a['cw_seen']]
        valid=objs[:,None]==(ids//na)[None,:]
        score=a['joint'].reshape(len(row),-1)[:,ids].copy();score[~valid]=-np.inf
        candidate=ids[score.argmax(-1)];multi=unseen&(valid.sum(-1)>1)
        return dict(unseen_images=int(unseen.sum()),unseen_multi_candidate_images=int(multi.sum()),
            gt_object_attr_unseen=float((pred[unseen]==attrs[unseen]).mean()*100),
            gt_object_attr_unseen_multi=float((candidate[multi]==(objs*na+attrs)[multi]).mean()*100) if multi.any() else None)


def report(args):
    root=Path(args.work).resolve();root.mkdir(parents=True,exist_ok=True)
    rows=[];audits=[];checks=[];missing=[];diagnostics=[]
    for s in SEEDS:
        folder=root/('seed%d'%s)
        for stage in STAGES:
            if not (root/'progress'/('seed%d'%s)/(stage+'.json')).exists():missing.append('%d/%s'%(s,stage))
        for v in VARIANTS:
            file=folder/'runs'/v/'metrics.json'
            if not file.exists():continue
            value=read_json(file)
            diagnostics.append(dict(seed=s,variant=v,**conditional_diagnostics(file.parent)))
            for phase in ('val','test'):
                for world in ('cw','ow'):
                    m=value[phase][world]
                    rows.append(dict(seed=s,variant=v,epoch=value['epoch'],phase=phase,world=world,
                        AUC_native=m['AUC_native'],AUC_exact=m['AUC_exact'],HM=m['at_val_bias']['HM'],
                        NLL=m['calibrated']['NLL'],ECE=m['calibrated']['ECE'],gt_attr_unseen=m['gt_object_attr_acc_unseen']))
        for v,path in [('original','utility_original'),('conditioned','utility')]:
            if (folder/path/'audit.json').exists():audits.append(dict(seed=s,variant=v,audit=read_json(folder/path/'audit.json')))
        if (folder/'budget_recovery.json').exists():checks.append(dict(seed=s,**read_json(folder/'budget_recovery.json')))
    if rows:
        with (root/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    paired=[];aggregates={}
    for w in ('cw','ow'):
        for v in VARIANTS:
            r=[x for x in rows if x['phase']=='test' and x['world']==w and x['variant']==v]
            values=[x['AUC_native'] for x in r]
            if values:aggregates[v+'_'+w]=dict(values=values,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
        for control in ('drop','original'):
            delta=[]
            for s in SEEDS:
                rr={r['variant']:r['AUC_native'] for r in rows if r['seed']==s and r['phase']=='test' and r['world']==w}
                if 'conditioned' in rr and control in rr:delta.append(rr['conditioned']-rr[control])
            if delta:paired.append(dict(world=w,control=control,differences=delta,mean=float(np.mean(delta))))
    write_json(root/'aggregates.json',aggregates);write_json(root/'paired_comparisons.json',paired)
    write_json(root/'utility_comparison.json',audits);write_json(root/'budget_checks.json',checks)
    write_json(root/'conditional_diagnostics.json',diagnostics)
    metrics=len(rows)//4;responses=sum(len(list((root/('seed%d'%s)/'responses').glob('*.pt'))) for s in SEEDS)
    complete=not missing and metrics==9 and len(audits)==6 and responses==18
    value=dict(completed_stages=len(SEEDS)*len(STAGES)-len(missing),expected_stages=len(SEEDS)*len(STAGES),
        metrics=metrics,expected_metrics=9,utility_audits=len(audits),expected_utility_audits=6,
        response_files=responses,expected_response_files=18,missing=missing,complete=complete)
    write_json(root/'status.json',value)
    print('STATUS',value,flush=True)
    if complete:print('ALL_EXPERIMENTS_COMPLETE: 9 recognition results, 6 utility audits, 18 response banks',flush=True)
    return value
