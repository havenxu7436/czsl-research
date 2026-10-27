import contextlib,copy,os,platform,shlex,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,read_json,write_json,load_torch,sha,digest,source_hashes

SEEDS=(3351,3352,3353)
VARIANTS=('r0','capacity','dense','drop','free','mean','robust')
NOISE_VARIANTS=('dense','drop','mean','robust')


@contextlib.contextmanager
def lock(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    f=(folder/'.lock').open('a+b')
    try:
        if os.name=='posix':
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        else:
            import msvcrt
            f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        yield
    finally:f.close()


def setup(args):
    from .cache_validation import validate_cache
    candidates=[Path(args.cache)] if args.cache else [Path('/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'),Path.home()/'experiments/trustcpf_clip_v1/cache/ut-zap50k']
    cache=next((p for p in candidates if (p/'features_manifest.json').exists()),None)
    if cache is None:raise FileNotFoundError('Pass --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14')
    validate_cache(cache)
    if (ROOT/'MANIFEST.json').exists():
        for name,h in read_json(ROOT/'MANIFEST.json').items():
            if sha(ROOT/name)!=h:raise ValueError('Release content differs: '+name)
    values=dict(CPF_PY=sys.executable,CPF_CACHE=str(cache.resolve()),CPF_WORK=str(Path(args.work).resolve()))
    (ROOT/'server_env.sh').write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in values.items()),encoding='utf8')
    write_json(ROOT/'setup.json',dict(paths=values,torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
        cuda=torch.cuda.is_available(),gpu_count=torch.cuda.device_count()))
    print('SETUP_COMPLETE',values,flush=True)


def actions(args):
    from . import engine,stages
    root=Path(args.work);seed=args.seed
    result=[]
    def add(name,folder,stage,variant=None,config=None,fit_all=False):
        a=copy.copy(args);a.work=str(folder);a.stage=stage;a.variant=variant;a.fit_all=fit_all
        if config:a.config=str(config)
        functions=dict(teachers=engine.teachers,measure=stages.measure,fit=stages.fit_utility,audit=stages.utility_audit,
            tune=stages.tune,reference=engine.reference,structure=stages.structure,adapt=stages.adapt,noise=stages.noise)
        result.append((name,a,functions.get(stage)))
    for label,variant in [('main',None),('shuffled','shuffled')]:
        folder=root/(('seed' if label=='main' else 'shuffled')+str(seed))
        for stage in ('teachers','measure','fit','audit','tune','fit_all','reference','structure'):
            add(label+'/'+stage,folder,'fit' if stage=='fit_all' else stage,variant,fit_all=stage=='fit_all')
        for v in VARIANTS if label=='main' else ['shuffled']:add(label+'/adapt_'+v,folder,'adapt',v)
    for ablation in ('semantic','unmeasured'):
        for stage in ('fit','audit'):add(ablation+'/'+stage,root/(ablation+str(seed)),stage,config=ROOT/'configs'/(ablation+'.json'))
    for v in NOISE_VARIANTS:
        for deletion in (False,True):
            add('noise/'+v+('/delete' if deletion else '/mismatch'),root/('seed'+str(seed)),'noise',v)
            result[-1][1].deletion=deletion
    for mode in ('mean_graph','mean_logit'):
        add('readout/'+mode,root/('seed'+str(seed)),'readout','robust')
        result[-1][1].readout_mode=mode
    return result


def readout(args):
    from .engine import setup as inputs,new_model,evaluate_model
    cfg,manifest,text,bank,folds,work=inputs(args)
    state=load_torch(work/'runs/robust/best.pt');cfg=dict(state['cfg'],readout=args.readout_mode)
    model=new_model(text,bank,cfg,args.device);model.load_state_dict(state['model'])
    out=work/'runs'/('readout_'+args.readout_mode)
    val=evaluate_model(model,args.cache,cfg,manifest,state['pi'],'val',save=out)
    test=evaluate_model(model,args.cache,cfg,manifest,state['pi'],'test',val,save=out)
    write_json(out/'metrics.json',dict(variant='readout_'+args.readout_mode,seed=args.seed,epoch=state['epoch'],val=val,test=test,
        checkpoint_source=str(work/'runs/robust/best.pt'),protocol='fixed robust checkpoint; readout-only comparison'))


def run_seed(args):
    root=Path(args.work);plan=actions(args);marks=root/'progress'/('seed'+str(args.seed))
    cfg=read_json(args.config)
    identity=dict(config=cfg,seed=args.seed,source=source_hashes(),configs={p.name:sha(p) for p in (ROOT/'configs').glob('*.json')},
        knowledge=sha(ROOT/'assets/knowledge.pt'),cache_manifest=sha(Path(args.cache)/'features_manifest.json'),
        initial=sha(ROOT/'initial'/('g0_seed%d.pt'%args.seed)),noise_seeds=args.noise_seeds,rates=args.rates,quick=args.quick)
    with lock(marks):
        if (marks/'identity.json').exists() and read_json(marks/'identity.json')!=identity:raise ValueError('Recipe changed; use a new work directory')
        write_json(marks/'identity.json',identity)
        for name,a,fn in plan:
            marker=marks/(name.replace('/','__')+'.json')
            if marker.exists():print('REUSE_STAGE',name,flush=True);continue
            if name.startswith(('semantic/','unmeasured/')):
                dst=Path(a.work)/'measurements';dst.mkdir(parents=True,exist_ok=True)
                for f in (root/('seed'+str(args.seed))/'measurements').glob('fold*.pt'):
                    target=dst/f.name
                    if not target.exists():
                        try:os.link(str(f),str(target))
                        except OSError:shutil.copy2(f,target)
                if cfg.get('synthetic'):
                    acfg=dict(cfg,utility_input='semantic' if name.startswith('semantic/') else 'full')
                    if name.startswith('unmeasured/'):acfg['heldout_edge_fraction']=.2
                    a.config=str(Path(a.work)/'test_config.json');write_json(a.config,acfg)
            print('START_STAGE',args.seed,name,flush=True);tic=time.perf_counter()
            if a.stage=='readout':readout(a)
            else:fn(a)
            write_json(marker,dict(stage=name,seed=args.seed,seconds=time.perf_counter()-tic,completed=True))
            print('END_STAGE',args.seed,name,flush=True)
        write_json(marks/'DONE.json',dict(seed=args.seed,completed=len(plan),expected=len(plan)))
    print('SEED_COMPLETE',args.seed,'stages',len(plan),flush=True)


def report(args):
    from .report import report as tables
    root=Path(args.work);root.mkdir(parents=True,exist_ok=True)
    if list(root.rglob('metrics.json')):tables(args)
    missing=[];stage_count=0;done=0;noise_rows=0;metrics_count=0;audits=0
    for s in SEEDS:
        a=copy.copy(args);a.seed=s
        for name,_,_ in actions(a):
            stage_count+=1
            p=root/'progress'/('seed'+str(s))/(name.replace('/','__')+'.json')
            if p.exists():done+=1
            else:missing.append('seed%d/%s'%(s,name))
        for v in list(VARIANTS)+['readout_mean_graph','readout_mean_logit']:
            if (root/('seed'+str(s))/'runs'/v/'metrics.json').exists():metrics_count+=1
        if (root/('shuffled'+str(s))/'runs/shuffled/metrics.json').exists():metrics_count+=1
        for prefix in ('seed','shuffled','semantic','unmeasured'):
            if (root/(prefix+str(s))/'utility/audit.json').exists():audits+=1
        for v in NOISE_VARIANTS:
            for kind in ('delete','mismatch'):
                p=root/('seed'+str(s))/'runs'/v/('noise_'+kind+'.json')
                if p.exists():noise_rows+=len(read_json(p))
    value=dict(completed_stages=done,expected_stages=stage_count,metrics=metrics_count,expected_metrics=30,
        utility_audits=audits,expected_utility_audits=12,noise_rows=noise_rows,expected_noise_rows=1800,missing=missing,
        complete=done==stage_count and metrics_count==30 and audits==12 and noise_rows==1800)
    write_json(root/'status.json',value)
    print('STATUS',done,'/',stage_count,'metrics',metrics_count,'/30','noise',noise_rows,'/1800','complete',value['complete'],flush=True)
    return value
