"""Reuse existing strong CPF/paired measurements/response banks, then run six adaptations."""
import copy,csv,os,platform,shlex,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,read_json,write_json,load_torch,save_torch,sha,source_hashes
from .runtime import lock
from . import compact

SEEDS=(3351,3352,3353)
STAGES=('import','fit','audit','fit_all','structure','adapt_compact','adapt_shuffled_utility')
VARIANTS=('drop','compact','shuffled_utility')
CONTROL_FILES=('best.pt','last.pt','bank.pt','history.json','training.json','val_predictions.npz','test_predictions.npz',
               'val_records.json','test_records.json','prediction_axes.json')

def source_seed(args):return Path(args.source).resolve()/'work'/('seed%d'%args.seed)
def response_seed(args):return Path(args.response_source).resolve()/'work'/('seed%d'%args.seed)

def setup(args):
    from .cache_validation import validate_cache
    source=Path(args.source).resolve();response=Path(args.response_source).resolve();cache=Path(args.cache).resolve()
    status=read_json(source/'work/status.json');response_status=read_json(response/'work/status.json')
    if not status.get('complete') or not response_status.get('complete'):raise ValueError('Source experiment incomplete')
    for name in ('model.py','g0.py','engine.py','evaluation.py','metrics.py'):
        if any(sha(base/'trustcpf'/name)!=sha(ROOT/'trustcpf'/name) for base in (source,response)):
            raise ValueError('Source classifier/evaluator differs: '+name)
    for seed in SEEDS:
        name='g0_seed%d.pt'%seed
        if sha(source/'initial'/name)!=sha(ROOT/'initial'/name):raise ValueError('G0 checkpoint differs')
    if sha(source/'assets/knowledge.pt')!=sha(ROOT/'assets/knowledge.pt'):raise ValueError('Knowledge bank differs')
    if (ROOT/'MANIFEST.json').exists():
        for name,h in read_json(ROOT/'MANIFEST.json').items():
            if sha(ROOT/name)!=h:raise ValueError('Release file differs: '+name)
    print('CHECKING_EXISTING_FEATURE_CACHE',cache,flush=True);validate_cache(cache)
    for seed in SEEDS:
        a=copy.copy(args);a.seed=seed;validate_sources(a)
    paths=dict(CPF_PY=sys.executable,CPF_SOURCE=str(source),CPF_RESPONSE_SOURCE=str(response),
        CPF_CACHE=str(cache),CPF_WORK=str(Path(args.work).resolve()))
    (ROOT/'server_env.sh').write_bytes(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in paths.items()).encode('utf8'))
    write_json(ROOT/'setup.json',dict(paths=paths,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        cuda=torch.cuda.is_available(),gpu_count=torch.cuda.device_count(),source_status=status,response_status=response_status))
    print('SETUP_COMPLETE',paths,flush=True)
    return paths

def validate_sources(args):
    src=source_seed(args);resp=response_seed(args)
    if sha(src/'reference/last.pt')!=sha(resp/'reference/last.pt'):raise ValueError('Response reference differs')
    folds=read_json(src/'folds.json')['query']
    bank=load_torch(src/'runs/robust/bank.pt')
    for j in range(1,7):
        name='fold%d'%j if j<=5 else 'reference'
        r=load_torch(resp/'responses'/(name+'.pt'))
        checkpoint=src/'teachers'/str(j)/'last.pt' if j<=5 else src/'reference/last.pt'
        if r['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Response teacher differs: '+name)
        if tuple(r['features'].shape)!=(len(bank['types']),8):raise ValueError('Response shape differs')
        if not torch.isfinite(r['features']).all():raise ValueError('Nonfinite response')
        if j<=5 and set(map(tuple,r['train_pairs']))&set(map(tuple,folds[j-1])):
            raise ValueError('Response Inner overlaps Query')
    return bank

def input_identity(args):
    src=source_seed(args);resp=response_seed(args)
    files=['reference/last.pt','structure.pt','tuning.json','folds.json']
    files+=['measurements/fold%d.pt'%j for j in range(1,6)]
    files+=['teachers/%d/last.pt'%j for j in range(1,6)]
    files+=['runs/'+v+'/'+f for v in ('drop','robust','r0','capacity','dense')
        for f in CONTROL_FILES+('metrics.json',) if (src/'runs'/v/f).exists()]
    rf=['responses/'+x+'.pt' for x in ('fold1','fold2','fold3','fold4','fold5','reference')]
    rf+=['recipe.json','reference/last.pt','runs/conditioned/metrics.json']
    return dict(seed=args.seed,source_seed=str(src),response_seed=str(resp),source_hashes={f:sha(src/f) for f in files},
        response_hashes={f:sha(resp/f) for f in rf},implementation=source_hashes(),config=read_json(args.config),
        cache_manifest=sha(Path(args.cache)/'features_manifest.json'),knowledge=sha(ROOT/'assets/knowledge.pt'),
        initial=sha(ROOT/'initial'/('g0_seed%d.pt'%args.seed)))

def prepare_seed(args):
    src=source_seed(args);resp=response_seed(args);work=Path(args.work);bank=validate_sources(args)
    state=load_torch(src/'reference/last.pt');cfg=copy.deepcopy(state['cfg']);new=read_json(args.config)
    cfg.update({k:new[k] for k in ('ridge_alphas','label_shuffle_seed')})
    cfg.update(read_json(src/'tuning.json')['selected'])
    cfg.update(seed=args.seed,variant='compact',utility_input='compact_response',lambda_var=0.,lambda_ens=0.)
    types=bank['types'];pi=load_torch(src/'runs/robust/best.pt')['pi']
    if not torch.allclose(pi,load_torch(src/'structure.pt')['pi'],atol=1e-6,rtol=0):raise ValueError('Source robust graph differs')
    budgets={str(t):float(pi[types==t].double().sum()) for t in range(3)}
    drop=load_torch(src/'runs/drop/best.pt')['pi']
    budget_error=max(abs(float(drop[types==t].double().sum())-budgets[str(t)]) for t in range(3))
    if budget_error>1e-5:raise ValueError('Archived Drop budget differs')
    oldrecipe=read_json(resp/'recipe.json')
    if cfg['tau_absolute']!=oldrecipe['tau_absolute'] or budgets!=oldrecipe['fixed_type_budgets']:
        raise ValueError('Response experiment temperature/budgets differ')
    cfg['fixed_type_budgets']=budgets;work.mkdir(parents=True,exist_ok=True)
    write_json(work/'recipe.json',cfg)
    write_json(work/'reuse.json',dict(source_seed=str(src),response_seed=str(resp),reference_sha256=sha(src/'reference/last.pt'),
        supervision='original paired Query utility, averaged by fold and edge',control='original Drop'))
    write_json(work/'budget_recovery.json',dict(budgets=budgets,drop_max_budget_error=budget_error,tau=cfg['tau_absolute']))
    (work/'reference').mkdir(exist_ok=True);shutil.copy2(src/'reference/last.pt',work/'reference/last.pt')
    shutil.copy2(src/'folds.json',work/'folds.json')
    dest=work/'runs/drop';dest.mkdir(parents=True,exist_ok=True)
    for name in CONTROL_FILES+('metrics.json',):
        if (src/'runs/drop'/name).exists():shutil.copy2(src/'runs/drop'/name,dest/name)
    (work/'responses').mkdir(exist_ok=True)
    for name in ('fold1','fold2','fold3','fold4','fold5','reference'):
        for suffix in ('.pt','.json'):
            file=resp/'responses'/(name+suffix)
            if file.exists():shutil.copy2(file,work/'responses'/file.name)
    stats=[]
    for j in range(1,6):
        raw=load_torch(src/'measurements'/('fold%d.pt'%j))
        stats.append(compact.aggregate_measurements(raw,len(types)))
        if j==5:save_torch(work/'inputs/q5_observations.pt',dict(target=raw['y'].double().numpy(),edge=raw['edge'].numpy()))
    save_torch(work/'inputs/edge_statistics.pt',dict(types=types.numpy(),folds=stats))
    history={v:read_json(src/'runs'/v/'metrics.json') for v in ('r0','capacity','dense','robust') if (src/'runs'/v/'metrics.json').exists()}
    history['conditioned']=read_json(resp/'runs/conditioned/metrics.json')
    write_json(work/'reference_benchmarks.json',history)

def stage_outputs(stage):
    if stage=='import':return ['recipe.json','reuse.json','budget_recovery.json','reference/last.pt','runs/drop/metrics.json',
        'inputs/edge_statistics.pt','inputs/q5_observations.pt','reference_benchmarks.json']+[
        'responses/'+x+'.pt' for x in ('fold1','fold2','fold3','fold4','fold5','reference')]
    if stage=='fit':return ['utility/selection.json']+['utility/'+v+'/model.pt' for v in compact.VARIANTS]
    if stage=='audit':return ['utility/'+v+'/'+f for v in compact.VARIANTS for f in ('audit.json','audit_predictions.pt')]
    if stage=='fit_all':return ['utility_all/'+v+'/model.pt' for v in compact.VARIANTS]
    if stage=='structure':return ['structures/'+v+'/'+f for v in compact.VARIANTS for f in ('structure.json','structure.pt')]
    return ['runs/'+stage[len('adapt_'): ]+'/'+f for f in ('metrics.json','best.pt','last.pt','val_predictions.npz','test_predictions.npz')]

def run_seed(args):
    root=Path(args.work).resolve();folder=root/('seed%d'%args.seed);marks=root/'progress'/('seed%d'%args.seed)
    with lock(marks):
        identity=input_identity(args)
        if (marks/'identity.json').exists() and read_json(marks/'identity.json')!=identity:
            raise ValueError('Recipe/source changed: use a new --work directory')
        write_json(marks/'identity.json',identity)
        a=copy.copy(args);a.work=str(folder);a.variant='compact';a.quick=False
        for stage in STAGES:
            marker=marks/(stage+'.json')
            if marker.exists():
                for rel,h in read_json(marker)['outputs'].items():
                    if not (folder/rel).is_file() or sha(folder/rel)!=h:raise ValueError('Completed stage output changed: '+rel)
                print('REUSE_STAGE',args.seed,stage,flush=True);continue
            print('START_STAGE',args.seed,stage,flush=True);tic=time.perf_counter()
            if stage=='import':prepare_seed(a)
            else:
                a.config=str(folder/'recipe.json')
                if stage=='fit':compact.fit(folder)
                elif stage=='audit':compact.audit(folder)
                elif stage=='fit_all':compact.fit(folder,True)
                elif stage=='structure':compact.structure(folder)
                else:compact.adapt(a,stage[len('adapt_'):])
            write_json(marker,dict(stage=stage,seed=args.seed,seconds=time.perf_counter()-tic,
                outputs={p:sha(folder/p) for p in stage_outputs(stage)},completed=True))
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
    rows=[];audits=[];checks=[];missing=[];diagnostics=[];historical=[];costs=[]
    for seed in SEEDS:
        folder=root/('seed%d'%seed)
        for stage in STAGES:
            marker=root/'progress'/('seed%d'%seed)/(stage+'.json')
            if not marker.exists():missing.append('%d/%s'%(seed,stage))
            else:
                m=read_json(marker)
                costs.append(dict(seed=seed,stage=stage,seconds=m['seconds']))
                for name,h in m['outputs'].items():
                    if not (folder/name).is_file() or sha(folder/name)!=h:missing.append('%d/changed:%s'%(seed,name))
        for variant in VARIANTS:
            file=folder/'runs'/variant/'metrics.json'
            if not file.exists():continue
            value=read_json(file)
            diagnostics.append(dict(seed=seed,variant=variant,**conditional_diagnostics(file.parent)))
            for phase in ('val','test'):
                for world in ('cw','ow'):
                    m=value[phase][world]
                    rows.append(dict(seed=seed,variant=variant,epoch=value['epoch'],phase=phase,world=world,
                        AUC_native=m['AUC_native'],AUC_exact=m['AUC_exact'],HM=m['at_val_bias']['HM'],
                        NLL=m['calibrated']['NLL'],ECE=m['calibrated']['ECE'],gt_attr_unseen=m['gt_object_attr_acc_unseen']))
        for variant in compact.VARIANTS:
            file=folder/'utility'/variant/'audit.json'
            if file.exists():audits.append(dict(seed=seed,variant=variant,audit=read_json(file)))
        if (folder/'budget_recovery.json').exists():checks.append(dict(seed=seed,**read_json(folder/'budget_recovery.json')))
        if (folder/'reference_benchmarks.json').exists():
            for variant,value in read_json(folder/'reference_benchmarks.json').items():
                for world in ('cw','ow'):
                    m=value['test'][world]
                    historical.append(dict(seed=seed,variant=variant,world=world,AUC_native=m['AUC_native'],AUC_exact=m['AUC_exact']))
    for name,table in [('summary',rows),('historical',historical),('costs',costs)]:
        if table:
            with (root/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    paired=[];aggregates={}
    for world in ('cw','ow'):
        for variant in VARIANTS:
            r=[x for x in rows if x['phase']=='test' and x['world']==world and x['variant']==variant]
            values=[x['AUC_native'] for x in r]
            if values:aggregates[variant+'_'+world]=dict(seeds=[x['seed'] for x in r],values=values,
                mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
        for control in ('drop','shuffled_utility'):
            delta=[];seeds=[]
            for seed in SEEDS:
                rr={r['variant']:r['AUC_native'] for r in rows if r['seed']==seed and r['phase']=='test' and r['world']==world}
                if 'compact' in rr and control in rr:delta.append(rr['compact']-rr[control]);seeds.append(seed)
            if delta:paired.append(dict(world=world,method='compact',control=control,seeds=seeds,differences=delta,
                mean=float(np.mean(delta)),sample_sd=float(np.std(delta,ddof=1)) if len(delta)>1 else None,wins=sum(d>0 for d in delta)))
    write_json(root/'aggregates.json',aggregates);write_json(root/'paired_comparisons.json',paired)
    write_json(root/'utility_comparison.json',audits);write_json(root/'budget_checks.json',checks)
    write_json(root/'conditional_diagnostics.json',diagnostics)
    metrics=len(rows)//4;responses=sum(len(list((root/('seed%d'%s)/'responses').glob('*.pt'))) for s in SEEDS)
    structures=sum(len(list((root/('seed%d'%s)/'structures').glob('*/structure.json'))) for s in SEEDS)
    complete=not missing and metrics==9 and len(audits)==6 and responses==18 and structures==6
    value=dict(completed_stages=len(costs),expected_stages=len(SEEDS)*len(STAGES),
        metrics=metrics,expected_metrics=9,utility_audits=len(audits),expected_utility_audits=6,
        response_files=responses,expected_response_files=18,structures=structures,expected_structures=6,missing=missing,complete=complete)
    write_json(root/'status.json',value)
    print('STATUS',value,flush=True)
    if complete:print('ALL_EXPERIMENTS_COMPLETE: 3 reused Drop + 6 new recognition results, 6 utility audits, 18 reused response banks',flush=True)
    return value
