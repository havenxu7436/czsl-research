"""Import Competition-10ep artifacts and run the five matched R5 methods."""
import copy
import csv
import hashlib
import platform
import shlex
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, read_json, write_json, load_torch, sha, source_hashes
from .runtime import lock
from .data import load_cache
from . import align, r5 as cmp, r5_training as train

SEEDS=(3351,3352,3353)
STAGES=('import','fit','decision','scan')+tuple('adapt_'+v for v in cmp.VARIANTS)
INPUTS=('reference/last.pt','responses/reference.pt','folds.json','inputs/bank.pt','recipe.json')+tuple(
    '%s/fold%d.%s'%(folder,j,ext) for j in range(1,6)
    for folder,ext in (('responses','pt'),('measurements','pt'),('splits','json')))+tuple(
    'teachers/%d/last.pt'%j for j in range(1,6))
MATCHED_CODE=('model.py','g0.py','strong.py','engine.py','evaluation.py','metrics.py','align.py',
              'compact.py','competition.py','competition_training.py','response.py','data.py','graph.py')


def code_digest(path):
    """LF/CRLF are equivalent; all other source content must match."""
    value=Path(path).read_text(encoding='utf-8-sig').replace('\r\n','\n')
    return hashlib.sha256(value.encode('utf8')).hexdigest()


def source_files(args):
    src=Path(args.source).resolve()/'work'/('seed%d'%args.seed)
    names=set(INPUTS)|{'policies.pt','selection.json','utility/selection.json','utility/predictions.pt'}
    cfg=read_json(args.config)
    names.update('decision/blocks/evaluate%d.pt'%v for v in cfg['audit_eval_seeds'])
    names.add('scan/uniform.json')
    names.update('scan/unll_k%g_real.json'%v for v in cfg['policy_strengths'] if v!=0)
    for v in ('uniform','unll'):
        names.update(p.relative_to(src).as_posix() for p in (src/'runs'/v).rglob('*') if p.is_file() and not p.name.endswith('.writing'))
    return src,{name:sha(src/name) for name in sorted(names)}


def validate_source(args):
    source=Path(args.source).resolve();src=source/'work'/('seed%d'%args.seed)
    cfg=read_json(args.config);old=read_json(src/'recipe.json')
    if not read_json(source/'work/status.json')['complete']:
        raise ValueError('Run the Competition-10ep source queue to completion first: '+str(source))
    for name in MATCHED_CODE:
        if code_digest(source/'trustcpf'/name)!=code_digest(ROOT/'trustcpf'/name):
            raise ValueError('Source implementation differs beyond line endings: '+name)
    # Experimental configurations inherit source parameters. Explicit new values
    # must match, except the experiment identity; this preserves control reuse.
    for key,value in cfg.items():
        if key in ('seed','variant','experiment','effect_target'):continue
        if key in old and value!=old[key]:raise ValueError('R5 matching setting differs: '+key)
    if old['seed']!=args.seed:raise ValueError('Source seed identity differs')
    if cfg.get('effect_target')!='unll':raise ValueError('R5 v1 uses the measured U-NLL column')
    if not cfg.get('synthetic') and cfg['adapt_epochs']!=10:raise ValueError('Use the matched 10-epoch schedule')
    if old['graph_seed']!=cfg['eval_graph_seeds'][0]:raise ValueError('Primary graph bank must be first')
    state=load_torch(src/'reference/last.pt');bank=load_torch(src/'inputs/bank.pt')
    for key in ('anchors','source','target','types','degree'):
        if not torch.equal(state['model']['graph.'+key],bank[key]):raise ValueError('Reference graph differs: '+key)
    manifest,_=load_cache(args.cache)
    if set(map(tuple,state['train_pairs']))!=set(map(tuple,manifest['pairs']['train'])):
        raise ValueError('Reference and cache TRAIN compositions differ')
    original_identity=read_json(source/'work/progress'/('seed%d'%args.seed)/'identity.json')
    if original_identity['cache_manifest']!=sha(Path(args.cache)/'features_manifest.json'):
        raise ValueError('Use the same feature cache as Competition-10ep')
    for fold in range(1,6):
        split=read_json(src/('splits/fold%d.json'%fold));teacher=load_torch(src/('teachers/%d/last.pt'%fold))
        response=load_torch(src/('responses/fold%d.pt'%fold));measured=load_torch(src/('measurements/fold%d.pt'%fold))
        digest=sha(src/('teachers/%d/last.pt'%fold))
        if response['checkpoint_sha256']!=digest or measured['teacher_sha256']!=digest:
            raise ValueError('Teacher/response/measurement identity differs in fold '+str(fold))
        if measured['split']!=split or teacher['train_ids']!=split['train_ids']:
            raise ValueError('Teacher image split differs in fold '+str(fold))
        if 'unll' not in measured['targets']:raise ValueError('Missing U-NLL effect labels')
    if load_torch(src/'responses/reference.pt')['checkpoint_sha256']!=sha(src/'reference/last.pt'):
        raise ValueError('Reference response differs from its checkpoint')
    for variant in ('uniform','unll'):
        if load_torch(src/'runs'/variant/'last.pt')['epoch']!=cfg['adapt_epochs']:
            raise ValueError('Source control did not finish 10 epochs: '+variant)
    return src


def setup(args):
    from .cache_validation import validate_cache
    if (ROOT/'MANIFEST.json').exists():
        for name,value in read_json(ROOT/'MANIFEST.json').items():
            if sha(ROOT/name)!=value:raise ValueError('Release file differs: '+name)
    print('CHECKING_FEATURE_CACHE',args.cache,flush=True);validate_cache(args.cache)
    for seed in SEEDS:
        a=copy.copy(args);a.seed=seed;validate_source(a);source_files(a)
        print('R5_SOURCE_READY',seed,'teachers/measurements/responses and 10ep controls',flush=True)
    paths=dict(CPF_PY=sys.executable,CPF_SOURCE=str(Path(args.source).resolve()),CPF_CACHE=str(Path(args.cache).resolve()),
               CPF_WORK=str(Path(args.work).resolve()),CPF_CONFIG=str(Path(args.config).resolve()))
    (ROOT/'server_env.sh').write_bytes(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in paths.items()).encode('utf8'))
    write_json(ROOT/'setup.json',dict(paths=paths,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        cuda=torch.cuda.is_available(),gpu_count=torch.cuda.device_count(),source_line_endings_normalized=True))
    print('SETUP_COMPLETE',paths,flush=True)


def prepare(args):
    src=validate_source(args);work=Path(args.work);cfg=read_json(src/'recipe.json')
    cfg.update(read_json(args.config));cfg.update(seed=args.seed,variant='r5',readout='probability')
    for name in INPUTS:
        if name=='recipe.json':continue
        dest=work/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/name,dest)
    write_json(work/'recipe.json',cfg)
    _,hashes=source_files(args)
    write_json(work/'reuse.json',dict(source_seed=str(src),source_files=hashes,
        reference_sha256=sha(src/'reference/last.pt'),effect_target='unll',
        teachers='reused Competition dual-group teachers; no new training',
        controls='matching Uniform and U-NLL copied at their adaptation stages'))


def identity(args):
    src,files=source_files(args)
    return dict(seed=args.seed,config=read_json(args.config),source=str(src),source_files=files,
        implementation=source_hashes(),initial=sha(ROOT/'initial'/('g0_seed%d.pt'%args.seed)),
        knowledge=sha(ROOT/'assets/knowledge.pt'),cache_manifest=sha(Path(args.cache)/'features_manifest.json'))


def outputs(stage,work):
    if stage=='import':return sorted(set(INPUTS)|{'reuse.json'})
    if stage=='fit':folders,names=['utility','utility_all'],[]
    elif stage=='decision':folders,names=['decision'],[]
    elif stage=='scan':folders,names=['scan'],['selection.json','policies.pt']
    else:folders,names=['runs/'+stage[len('adapt_'):]],[]
    for folder in folders:
        names.extend(p.relative_to(work).as_posix() for p in (work/folder).rglob('*') if p.is_file() and not p.name.endswith('.writing'))
    return sorted(names)


def run_seed(args):
    root=Path(args.work).resolve();work=root/('seed%d'%args.seed);marks=root/'progress'/('seed%d'%args.seed)
    with lock(marks):
        ident=identity(args)
        if (marks/'identity.json').exists() and read_json(marks/'identity.json')!=ident:
            raise ValueError('Experiment inputs changed; use a new work directory')
        write_json(marks/'identity.json',ident);a=copy.copy(args);a.work=str(work)
        for stage in STAGES:
            marker=marks/(stage+'.json')
            if marker.exists():
                for rel,h in read_json(marker)['outputs'].items():
                    if not (work/rel).is_file() or sha(work/rel)!=h:raise ValueError('Completed output changed: '+rel)
                print('REUSE_STAGE',args.seed,stage,flush=True);continue
            align.synchronize(args.device);tic=time.perf_counter()
            if str(args.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
            print('START_STAGE',args.seed,stage,flush=True)
            if stage=='import':prepare(a)
            elif stage=='fit':train.fit(a)
            elif stage=='decision':train.audit(a)
            elif stage=='scan':train.scan(a)
            else:train.adapt(a,stage[len('adapt_'):])
            align.synchronize(args.device)
            write_json(marker,dict(stage=stage,completed=True,seconds=time.perf_counter()-tic,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else 0,
                outputs={p:sha(work/p) for p in outputs(stage,work)}))
            print('END_STAGE',args.seed,stage,flush=True)
        write_json(marks/'DONE.json',dict(seed=args.seed,completed=True,stages=len(STAGES)))
    print('SEED_COMPLETE',args.seed,flush=True)


def write_csv(file,rows):
    if not rows:return
    with Path(file).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(args):
    root = Path(args.work).resolve(); root.mkdir(parents=True, exist_ok=True)
    rows, risks, scans, costs, utility, deployment, paired, adaptation, latency, teachers = ([] for _ in range(10))
    curves = []
    missing = []; stages = metrics = audits = 0
    for seed in SEEDS:
        work = root / ('seed%d' % seed)
        for stage in STAGES:
            marker = root / 'progress' / ('seed%d' % seed) / (stage+'.json')
            if not marker.exists(): missing.append('%d/%s' % (seed, stage)); continue
            record = read_json(marker); stages += 1
            costs.append(dict(seed=seed, stage=stage, seconds=record['seconds'], peak_gpu_bytes=record['peak_gpu_bytes']))
            for rel, h in record['outputs'].items():
                if not (work / rel).is_file() or sha(work / rel) != h: missing.append('%d/changed:%s' % (seed, rel))
        for fold in range(1, 6):
            if (work / 'measurements' / ('fold%d.pt' % fold)).exists():
                split = read_json(work / 'splits' / ('fold%d.json' % fold))
                teachers.append(dict(seed=seed, fold=fold, train_images=len(split['train_ids']),
                    seen_risk=len(split['measure']['seen']), unseen_risk=len(split['measure']['unseen']),
                    seen_eval=len(split['evaluate'].get('seen', [])), unseen_eval=len(split['evaluate'].get('unseen', []))))
        for variant in cmp.VARIANTS:
            file = work / 'runs' / variant / 'metrics.json'
            if not file.exists(): continue
            result = read_json(file); metrics += 1
            history = read_json(work / 'runs' / variant / 'history.json')
            primary = str(read_json(work / 'recipe.json')['graph_seed'])
            initial = result['results']['epoch0'][primary]['val']
            curves.append(dict(seed=seed, variant=variant, epoch=0, loss=None, learning_rate=None,
                VAL_CW_AUC=initial['cw']['AUC_native'], VAL_OW_AUC=initial['ow']['AUC_native'],
                selected=result['epoch'] == 0))
            for record in history:
                curves.append(dict(seed=seed, variant=variant, epoch=record['epoch'], loss=record['loss'],
                    learning_rate=record['learning_rate'], VAL_CW_AUC=record['val']['cw']['AUC_native'],
                    VAL_OW_AUC=record['val']['ow']['AUC_native'], selected=record['epoch'] == result['epoch']))
            speed = result['inference_latency']
            latency.append(dict(seed=seed, variant=variant, milliseconds_per_image=speed['milliseconds_per_image'],
                                graphs=speed['graphs'], batch_images=speed['batch_images']))
            selection = read_json(work / 'selection.json')
            strength = 0. if variant == 'uniform' else selection['targets'][variant.replace('permuted_', '')]['strength']
            for snapshot, banks in result['results'].items():
                for bank_seed, b in banks.items():
                    for phase in ('val', 'test'):
                        for world in ('cw', 'ow'):
                            m = b[phase][world]
                            rows.append(dict(seed=seed, variant=variant, snapshot=snapshot, bank=int(bank_seed),
                                epoch=b['epoch'], strength=strength, phase=phase, world=world,
                                AUC_native=m['AUC_native'], AUC_exact=m['AUC_exact'], HM=m['at_val_bias']['HM'],
                                seen=m['at_val_bias']['seen'], unseen=m['at_val_bias']['unseen'],
                                NLL=m['calibrated']['NLL'], raw_NLL=m['unbiased']['NLL'], ECE=m['calibrated']['ECE'],
                                gt_attr_unseen=m['gt_object_attr_acc_unseen'], graph_variance=m['graph_variance'],
                                reused_from=result.get('reused_from','')))
                    for ty, v in b['deployment'].items():
                        deployment.append(dict(seed=seed, variant=variant, snapshot=snapshot, bank=int(bank_seed),
                            type=int(ty), strength=strength, expected_edges=v['expected_edges'], probability_std=v['probability_std'],
                            probability_min=v['probability_min'], probability_max=v['probability_max'], kl=v['kl'],
                            expected_covered_nodes=sum(v['expected_node_coverage']), prototype_displacement=v['prototype_displacement_mean']))
        if (work / 'decision/audit.json').exists():
            audits += 1; audit = read_json(work / 'decision/audit.json')
            risks.extend(dict(seed=seed, **r) for r in audit['rows'])
        if (work / 'scan/results.json').exists():
            scans.extend(dict(seed=seed, target=r['target'], strength=r['strength'], permutation_seed=r['permutation_seed'],
                              key=r['key'], VAL_CW_AUC=r['val']['cw']['AUC_native'], VAL_OW_AUC=r['val']['ow']['AUC_native'])
                         for r in read_json(work / 'scan/results.json'))
        if (work / 'utility/audit.json').exists():
            for target, v in read_json(work / 'utility/audit.json').items():
                for ty in v['by_type']:
                    utility.append(dict(seed=seed, target=target, type=int(ty), **v['by_type'][ty],
                                        **v['transfer'][ty]))
    index = {(r['seed'], r['variant'], r['snapshot'], r['bank'], r['phase'], r['world']): r for r in rows}
    comparisons = [('relative', 'uniform'), ('relative', 'absolute'), ('relative', 'permuted_relative'),
                   ('absolute', 'uniform'), ('absolute', 'permuted_absolute')]
    for r in rows:
        for real, control in comparisons:
            if r['variant'] != real: continue
            b = index.get((r['seed'], control, r['snapshot'], r['bank'], r['phase'], r['world']))
            if b:
                paired.append(dict(seed=r['seed'], comparison=real+'-minus-'+control, snapshot=r['snapshot'],
                    bank=r['bank'], phase=r['phase'], world=r['world'], AUC_native=r['AUC_native']-b['AUC_native'],
                    AUC_exact=r['AUC_exact']-b['AUC_exact'], HM=r['HM']-b['HM'], raw_NLL_gain=b['raw_NLL']-r['raw_NLL']))
        if r['snapshot'] == 'best':
            b = index[(r['seed'], r['variant'], 'epoch0', r['bank'], r['phase'], r['world'])]
            adaptation.append(dict(seed=r['seed'], variant=r['variant'], bank=r['bank'], phase=r['phase'], world=r['world'],
                                   epoch=r['epoch'], AUC_gain=r['AUC_native']-b['AUC_native'], HM_gain=r['HM']-b['HM']))
    aggregates = []
    for name, table, keys in [('recognition', rows, ('variant', 'snapshot', 'bank', 'phase', 'world')),
                              ('paired', paired, ('comparison', 'snapshot', 'bank', 'phase', 'world'))]:
        groups = {}
        for r in table: groups.setdefault(tuple(r[k] for k in keys), []).append(r)
        for key, items in groups.items():
            values = np.array([r['AUC_native'] for r in items])
            aggregates.append(dict(table=name, group=dict(zip(keys, key)), n_seeds=len(items),
                AUC_mean=float(values.mean()), AUC_sd=float(values.std(ddof=1)) if len(items)>1 else None))
    for name, table in [('summary', rows), ('paired_comparisons', paired), ('decision_risks', risks), ('scan', scans),
                        ('costs', costs), ('utility_correlations', utility), ('deployment', deployment),
                        ('adaptation_gain', adaptation), ('inference_latency', latency), ('teachers', teachers),
                        ('training_curves', curves)]:
        write_csv(root / (name+'.csv'), table)
    write_json(root / 'aggregates.json', aggregates)
    cfg = read_json(args.config)
    expected_rows = len(SEEDS)*len(cmp.VARIANTS)*2*len(cfg['eval_graph_seeds'])*4
    expected_risks = len(SEEDS)*len(cmp.POLICIES)*len(cfg['audit_eval_seeds'])
    expected_scans = len(SEEDS)*len(cmp.METHODS)*len(cfg['policy_strengths'])*(1+len(cfg['permutation_seeds']))
    expected_curves = len(SEEDS)*len(cmp.VARIANTS)*(cfg['adapt_epochs']+1)
    status = dict(completed_stages=stages, expected_stages=len(SEEDS)*len(STAGES), metrics=metrics, expected_metrics=15,
        utility_audits=audits, expected_utility_audits=3, teachers=len(teachers), expected_teachers=15,
        summary_rows=len(rows), expected_summary_rows=expected_rows, scan_rows=len(scans), expected_scan_rows=expected_scans,
        decision_bank_rows=len(risks), expected_decision_bank_rows=expected_risks,
        training_curve_rows=len(curves), expected_training_curve_rows=expected_curves, missing=missing)
    status['complete'] = (stages == status['expected_stages'] and metrics == 15 and audits == 3 and len(teachers) == 15
                          and len(rows) == expected_rows and len(risks) == expected_risks and len(scans) == expected_scans
                          and len(curves) == expected_curves and not missing)
    write_json(root / 'status.json', status)
    print('REPORT', root / 'summary.csv', flush=True)
    print('STATUS', stages, '/', status['expected_stages'], 'models', metrics, '/15', 'audits', audits, '/3',
          'teachers', len(teachers), '/15', 'complete', status['complete'], flush=True)
    return status
