"""Import -> 10ep/common measurements -> pooled fit -> Q4 -> select -> Q5/formal."""
import copy,platform,shlex,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,read_json,write_json,load_torch,save_torch,sha,source_hashes
from .runtime import lock
from .data import load_cache
from . import abc,abc_measure as measure,abc_eval as evaluate,abc_train,align
from .state_workflow import code_digest,hashes,files_under,check_marker,write_csv

PREP=('uniform10',)+tuple('teacher%d'%j for j in range(1,6))+('formal_response',)
DEV=('q4',)
EVAL=('policies','q5','recognition')
GLOBAL=('import','fit','select')
CORE=('model.py','g0.py','engine.py','evaluation.py','metrics.py','response.py','compact.py','align.py','state_eval.py')

def q4_split(split,cfg):
    s=copy.deepcopy(split);used=set(s['train_ids'])
    for key in ('measure','stop','evaluate'):
        for values in s.get(key,{}).values():used.update(values)
    rng=np.random.RandomState(cfg['q4_seed']);out={}
    for group in ('seen','unseen'):
        pool=sorted(set(s[group+'_pool'])-used)
        if len(pool)<cfg['q4_images_per_group']:raise ValueError('Q4 unused '+group+' pool too small')
        out[group]=rng.choice(pool,cfg['q4_images_per_group'],replace=False).tolist()
    s['evaluate']=out;s['q4_evaluate_seed']=cfg['q4_seed']
    return s

def setup(args):
    source=Path(args.source).resolve();root=Path(args.work).resolve();cfg=read_json(args.config)
    if cfg['adapt_epochs']!=10 or cfg['fixed_strength']!=1. or cfg['abc_lam']!=10/3:
        raise ValueError('Use the released 10epoch/kappa1/lambda10over3 protocol')
    if cfg['amplitudes']!=[0,.25,.5,1]:raise ValueError('A uses eta 0, .25, .5, 1')
    if not cfg.get('synthetic'):
        if cfg['graphs']!=16 or cfg['audit_graphs']!=16:raise ValueError('Production mixtures use 16 graphs')
        from .cache_validation import validate_cache
        validate_cache(args.cache)
    assert read_json(source/'work/status.json')['complete'],'The Stage source experiment must be complete'
    source_identity=read_json(source/'work/progress/global/identity.json')
    assert source_identity['cache_manifest']==sha(Path(args.cache)/'features_manifest.json')
    for name in CORE:
        if code_digest(source/'trustcpf'/name)!=code_digest(ROOT/'trustcpf'/name):
            raise ValueError('Source classifier/response math differs: '+name)
    manifest,text=load_cache(args.cache);pairs=[];fingerprints={};splits={};banks=[]
    def add(src,dst):
        pairs.append((src,dst));fingerprints[dst.relative_to(root).as_posix()]=sha(src)
    add(source/'work/utility/matched.pt',root/'utility/source_matched.pt')
    for seed in abc.SEEDS:
        sw=source/'work'/('seed%d'%seed);w=root/('seed%d'%seed);old=read_json(sw/'recipe.json')
        for key in ('batch_size','graph_lr','weight_decay','lambda_w','gradient_clip','graphs','graph_seed',
                    'rank','message_rank','eta','alpha','tg','object_context','response_images_per_pair','response_backgrounds',
                    'measurement_blocks','q5_measurement_blocks','response_repeat_offset','backbone_rank','backbone_adapter_rank'):
            if cfg[key]!=old[key]:raise ValueError('Source protocol differs: '+key)
        for name in ('inputs/bank.pt','reference/last.pt','responses/reference.pt','folds.json'):
            add(sw/name,w/name)
        add(sw/'runs/uniform/best.pt',w/'source_uniform/best.pt')
        banks.append(load_torch(sw/'inputs/bank.pt'))
        for j in range(1,6):
            cp=sw/('teachers/%d/best.pt'%j);r=load_torch(sw/('responses/fold%d.pt'%j))
            y=load_torch(sw/('measurements/fold%d.pt'%j));sp=read_json(sw/('splits/fold%d.json'%j))
            assert r['checkpoint_sha256']==y['teacher_sha256']==sha(cp)
            assert set(r['support_indices'])<=set(sp['train_ids'])
            assert load_torch(cp)['train_ids']==sp['train_ids']
            splits[seed,j]=q4_split(sp,cfg) if j==4 else sp
            for name in ('teachers/%d/best.pt'%j,'responses/fold%d.pt'%j,'responses/fold%d_repeat.pt'%j,'measurements/fold%d.pt'%j):
                add(sw/name,w/name)
            expected=cfg['q5_measurement_blocks'] if j==5 else cfg['measurement_blocks']
            for b in range(expected):
                name='measurements/fold%d/block%02d.pt'%(j,b);add(sw/name,w/name)
        print('SOURCE_READY',seed,'15 Stage teachers reused; new Uniform 10epoch',flush=True)
    for bank in banks[1:]:
        for k in ('anchors','source','target','types','degree'):assert torch.equal(bank[k],banks[0][k])
    groups=abc.object_partition(banks[0]['anchors'],len(manifest['labels']['attrs']))
    groups['names']=[[manifest['labels']['objs'][i] for i in g] for g in groups['groups']]
    ident=dict(source=str(source),source_files=fingerprints,config=cfg,implementation=source_hashes(),
        cache_manifest=sha(Path(args.cache)/'features_manifest.json'),groups=groups)
    with lock(root/'progress/global'):
        ip=root/'progress/global/identity.json'
        if ip.exists() and read_json(ip)!=ident:raise ValueError('Inputs/configuration changed; use a new work directory')
        write_json(ip,ident);marker=root/'progress/global/import.json'
        if marker.exists():check_marker(root,marker)
        else:
            tic=time.perf_counter();outputs=[]
            for src,dst in pairs:
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);outputs.append(dst)
            for (seed,j),sp in splits.items():
                p=root/('seed%d/splits/fold%d.json'%(seed,j));write_json(p,sp);outputs.append(p)
            for seed in abc.SEEDS:
                w=root/('seed%d'%seed);write_json(w/'recipe.json',dict(cfg,seed=seed));outputs.append(w/'recipe.json')
            write_json(root/'groups.json',groups);outputs.append(root/'groups.json')
            write_json(marker,dict(completed=True,seconds=time.perf_counter()-tic,outputs=hashes(root,outputs)))
    env=dict(CPF_PY=sys.executable,CPF_SOURCE=str(source),CPF_CACHE=str(Path(args.cache).resolve()),
             CPF_WORK=str(root),CPF_CONFIG=str(Path(args.config).resolve()))
    (ROOT/'server_env.sh').write_bytes(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items()).encode('utf8'))
    write_json(ROOT/'setup.json',dict(paths=env,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))
    print('SETUP_COMPLETE: run bash scripts/run_all.sh 1 2 3',flush=True)

def inputs(args):
    root=Path(args.work).resolve();ident=read_json(root/'progress/global/identity.json')
    assert ident['config']==read_json(args.config),'Configuration changed since setup'
    assert ident['implementation']==source_hashes(),'Implementation changed since setup'
    assert ident['cache_manifest']==sha(Path(args.cache)/'features_manifest.json')
    check_marker(root,root/'progress/global/import.json')
    return root

def stage_outputs(w,stage):
    if stage=='uniform10':return files_under(w,['runs/uniform'])
    if stage.startswith('teacher'):
        j=int(stage[-1])
        return files_under(w,['auxiliary/fold%d'%j])+[w/('context/fold%d.pt'%j),w/('effects/fold%d.pt'%j)]
    if stage=='formal_response':return [w/'context/uniform_best.pt',w/'responses/uniform_best.pt']
    if stage in ('q4','q5'):return files_under(w,['decision/fold'+stage[-1],'decision/fold'+stage[-1]+'_measure'])
    if stage=='policies':return [w/'policies.pt']
    return files_under(w,['recognition'])

def stages(args,which):
    root=inputs(args);w=root/('seed%d'%args.seed);marks=root/'progress'/('seed%d'%args.seed)
    if which==DEV:check_marker(root,root/'progress/global/fit.json')
    if which==EVAL:check_marker(root,root/'progress/global/select.json')
    a=copy.copy(args);a.work=str(w)
    with lock(marks):
        for stage in which:
            marker=marks/(stage+'.json')
            if marker.exists():check_marker(w,marker);print('REUSE_STAGE',args.seed,stage,flush=True);continue
            tic=time.perf_counter();align.synchronize(args.device)
            if str(args.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
            print('START_STAGE',args.seed,stage,flush=True)
            if stage=='uniform10':abc_train.uniform(a)
            elif stage.startswith('teacher'):measure.teacher(a,int(stage[-1]))
            elif stage=='formal_response':measure.formal_response(a)
            elif stage=='q4':
                evaluate.decision(a,4);evaluate.decision(a,4,partition='measure')
            elif stage=='policies':evaluate.make_policies(a)
            elif stage=='q5':
                evaluate.decision(a,5,True);evaluate.decision(a,5,partition='measure')
            else:evaluate.recognition(a)
            align.synchronize(args.device)
            output=stage_outputs(w,stage)
            if not output:raise RuntimeError('Stage produced no outputs: '+stage)
            write_json(marker,dict(completed=True,seconds=time.perf_counter()-tic,outputs=hashes(w,output),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else 0))
            print('END_STAGE',args.seed,stage,flush=True)

def prepare_seed(args):return stages(args,PREP)
def q4_seed(args):return stages(args,DEV)
def evaluate_seed(args):return stages(args,EVAL)

def fit(args):
    root=inputs(args);cfg=read_json(args.config)
    for seed in abc.SEEDS:
        for s in PREP:check_marker(root/('seed%d'%seed),root/('progress/seed%d/%s.json'%(seed,s)))
    with lock(root/'progress/global'):
        mark=root/'progress/global/fit.json'
        if mark.exists():return check_marker(root,mark)
        tic=time.perf_counter();abc.fit(root,cfg)
        write_json(mark,dict(completed=True,seconds=time.perf_counter()-tic,
            outputs=hashes(root,[root/'utility/models.pt',root/'utility/fit.json'])))
    print('FIT_COMPLETE: Matched, A scale, Mix16, Context and controls',flush=True)

def select(args):
    root=inputs(args);cfg=read_json(args.config);rows=[]
    for seed in abc.SEEDS:
        check_marker(root/('seed%d'%seed),root/('progress/seed%d/q4.json'%seed))
        rows+=read_json(root/('seed%d/decision/fold4/audit.json'%seed))['rows']
    with lock(root/'progress/global'):
        mark=root/'progress/global/select.json'
        if mark.exists():return check_marker(root,mark)
        tic=time.perf_counter();trials=[]
        for eta in cfg['amplitudes']:
            key='uniform' if eta==0 else abc.eta_key('amplitude',eta)
            selected=[r for r in rows if r['policy']==key]
            assert len(selected)==len(abc.SEEDS)*len(cfg['audit_eval_seeds'])
            trials.append(dict(eta=eta,AUC_exact=float(np.mean([r['mixture_AUC_exact'] for r in selected])),
                by_seed={str(s):float(np.mean([r['mixture_AUC_exact'] for r in selected if r['seed']==s])) for s in abc.SEEDS}))
        best=max(r['AUC_exact'] for r in trials)
        choice=min([r for r in trials if r['AUC_exact']>=best-1e-12],key=lambda r:r['eta'])
        path=root/'utility/selection.json'
        write_json(path,dict(eta=choice['eta'],trials=trials,selection='mean Q4 evaluate exact AUC; smaller eta on ties',
            B_C_strength=1.,Q5_TEST_used=False))
        write_json(mark,dict(completed=True,seconds=time.perf_counter()-tic,outputs=hashes(root,[path])))
    print('A_SELECTED_ETA',choice['eta'],'B/C remain kappa=1',flush=True)

def report(args):
    from .abc_report import report as report_impl
    with lock(Path(args.work)/'progress/report'):return report_impl(args)
