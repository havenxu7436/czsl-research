"""Three E1 seeds on independent GPUs; native/exact selections and rho comparisons."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys

import round_b as shared
import round_c as previous
from round_d import run_process

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = Path('/data/xhw/vgcpod_v1_runs/roundE_20261020')
DEFAULT_A = Path('/data/xhw/vgcpod_v1_runs/roundA_20261019')
DEFAULT_C = Path('/data/xhw/vgcpod_v1_runs/roundC_20261020')
DEFAULT_D = Path('/data/xhw/vgcpod_v1_runs/roundD_20261020')


def references(a_out,c_out,d_out):
    refs = {}
    for record in shared.read(ROOT/'references/pre_e_complete.json')['runs']:
        cfg = record['config']
        refs[(record['case'],record['seed'])] = dict(case=record['case'],seed=record['seed'],
            rho=cfg['rho'],source_group='packaged_reference',
            feature_manifest_sha256=record['feature_manifest_sha256'],selected_epoch=record['selected_epoch'],
            **shared.metric_columns(record['metrics']))
    for root,cases,stage in [(a_out,('A5','A6'),'roundA'),(c_out,('C1','C2'),'roundC'),(d_out,('D1',),'roundD')]:
        _,live = previous.collect(root,cases,stage)
        refs.update(live)
    return refs


def exact_rows(out):
    rows = []
    for path in sorted((Path(out)/'E1').glob('ut-zap50k/V*/seed*/exact_selection/metrics.json')):
        run = path.parent.parent
        protocol,saved = shared.read(run/'protocol.json'),shared.read(path)
        cfg = protocol['config']
        rows.append(dict(case='E1_exact',seed=protocol['seed'],rho=cfg['rho'],
            selected_epoch=saved['selected_epoch'],selection=saved['selection'],run_dir=str(path.parent),
            feature_manifest_sha256=protocol['feature_manifest_sha256'],**shared.metric_columns(saved['metrics'])))
    return rows


def summarize(out,a_out=DEFAULT_A,c_out=DEFAULT_C,d_out=DEFAULT_D):
    out = Path(out)
    rows,records = previous.collect(out,('E1',),'roundE')
    if not rows:
        raise FileNotFoundError('No completed E1 results: '+str(out))
    refs = references(a_out,c_out,d_out)
    deltas,pairs = [],[]
    for row in rows:
        for case in ('A5','A6','C1','C2','D1'):
            other = refs.get((case,row['seed']))
            if other and other['feature_manifest_sha256']==row['feature_manifest_sha256']:
                deltas.append(dict(comparison='E1-'+case,seed=row['seed'],
                    **{key:row[key]-other[key] for key in shared.COLUMNS}))
                pairs.append((row,other))
    secondary = exact_rows(out)
    selection_effect = []
    for row in secondary:
        native = records[('E1',row['seed'])]
        selection_effect.append(dict(comparison='E1_exact-E1',seed=row['seed'],
            **{key:row[key]-native[key] for key in shared.COLUMNS}))
    cache_ids = {r['feature_manifest_sha256'] for r in rows}
    rho_rows = [r for r in refs.values() if r['case'] in ('C2','D1')
                and r['feature_manifest_sha256'] in cache_ids]+rows
    rho_rows.sort(key=lambda r:(r['rho'],r['seed']))
    rho_keys = ('case','seed','rho','source_group','selected_epoch','feature_manifest_sha256')+shared.COLUMNS
    rho_rows = [{key:r[key] for key in rho_keys} for r in rho_rows]
    effects = []
    for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')]:
        for path in sorted((out/folder/'E1').glob('seed*/metrics.json')):
            data = shared.read(path)
            for setting,values in data['dynamic_minus_train_mean'].items():
                effects.append(dict(case='E1',seed=data['seed'],selection=selection,setting=setting,
                    selected_epoch=data['selected_epoch'],train_images=data['train_images'],
                    **{'dynamic_minus_train_mean_'+key:value for key,value in values.items()}))
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('round_e_summary',rows),('round_e_mean_sd',shared.aggregate(rows)),
                      ('round_e_paired_deltas',deltas),('round_e_paired_mean_sd',shared.aggregate(deltas,'comparison')),
                      ('rho_native_summary',rho_rows),('rho_native_mean_sd',shared.aggregate(rho_rows)),
                      ('exact_selection_summary',secondary),
                      ('exact_selection_effects',selection_effect),
                      ('exact_selection_effect_mean_sd',shared.aggregate(selection_effect,'comparison')),
                      ('mean_gate_effects',effects)]:
        shared.csv_write(out/(name+'.csv'),data)
    pair_count = previous.export_pair_gains(out,pairs)
    result = dict(runs=rows,by_case=shared.aggregate(rows),paired_deltas=deltas,
        paired_mean_sd=shared.aggregate(deltas,'comparison'),rho_native_runs=rho_rows,
        rho_native_mean_sd=shared.aggregate(rho_rows),exact_selection_runs=secondary,
        exact_selection_effects=selection_effect,mean_gate_effects=effects,per_pair_rows=pair_count,
        selection_rule='Primary: first strict max VAL CW AUC_native; secondary: first strict max VAL CW AUC_exact',
        comparison_rule='Matching seeds and feature cache; legacy C2/D1 use their archived native selection')
    shared.dump(out/'round_e_summary.json',result)
    print('ROUND_E_SUMMARY',out/'round_e_summary.json',flush=True)
    return result


def completed(run,cfg,seed,cache):
    from vgcpod.common import sha
    if not (run/'metrics.json').is_file():
        return False
    if not shared.validate_completed(run,cfg,seed,cache):
        raise ValueError('Existing E1 identity differs: '+str(run))
    secondary = run/'exact_selection/metrics.json'
    if not secondary.exists():
        return False
    return (sha(run/'checkpoints/best.pt')==sha(run/'checkpoints/best_native.pt') and
            shared.read(secondary)['checkpoint_sha256']==sha(run/'checkpoints/best_exact.pt'))


def worker(gpu,seed,args):
    config = ROOT/'configs/round_e/E1.json'
    cfg = shared.read(config)
    run = args.out/'E1'/cfg['dataset']/cfg['variant']/('seed%d'%seed)
    log = args.out/'logs'/('E1_seed%d.log'%seed)
    if completed(run,cfg,seed,args.cache):
        training = dict(status='already_complete',log=str(log))
        print('ALREADY_COMPLETE','E1',seed,'GPU',gpu,flush=True)
    else:
        command = [sys.executable,'-u',ROOT/'run.py','train','--cache',args.cache,
            '--out',args.out/'E1','--config',config,'--seed',seed,'--device',args.device]
        if (run/'protocol.json').exists():
            command.append('--resume')
        print('START_E1',seed,'GPU',gpu,'log',log,flush=True)
        code = run_process(command,ROOT,log,gpu)
        training = dict(status='complete' if code==0 else 'failed',exit_code=code,log=str(log))
        if code:
            return dict(case='E1',seed=seed,gpu=gpu,status='failed',training=training)
    evaluations = []
    for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')]:
        out = args.out/folder/'E1'/('seed%d'%seed)
        log = args.out/'logs'/('mean_gate_%s_E1_seed%d.log'%(selection,seed))
        command = [sys.executable,'-u',ROOT/'mean_gate_eval.py','--source-root',ROOT,
            '--cache',args.cache,'--run',run,'--out',out,'--selection',selection,'--device',args.device]
        print('START_MEAN_GATE',selection,seed,'GPU',gpu,flush=True)
        code = run_process(command,ROOT,log,gpu)
        evaluations.append(dict(selection=selection,status='complete' if code==0 else 'failed',
            exit_code=code,log=str(log)))
        if code:
            break
    status = 'failed' if any(r['status']=='failed' for r in evaluations) else 'complete'
    return dict(case='E1',seed=seed,gpu=gpu,status=status,training=training,mean_gate=evaluations)


def launch(args):
    for key in ('out','cache','a_out','c_out','d_out'):
        setattr(args,key,getattr(args,key).resolve())
    if not (args.cache/'features_manifest.json').is_file():
        raise FileNotFoundError('Feature cache absent: '+str(args.cache))
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'logs').mkdir(exist_ok=True)
    with (args.out/'.round_e.lock').open('a') as lock:
        if os.name=='posix':
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = [dict(case='E1',seed=seed,gpu=gpu,rho=.15) for gpu,seed in zip(args.gpus,args.seeds)]
        shared.dump(args.out/'round_e_jobs.json',plan)
        events = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(worker,job['gpu'],job['seed'],args):job for job in plan}
            for future in as_completed(futures):
                try:
                    event = future.result()
                except Exception as error:
                    event = dict(futures[future],status='failed',error=str(error))
                events.append(event)
                shared.dump(args.out/'round_e_status.json',events)
                print('JOB_RESULT','E1',event['seed'],event['status'],flush=True)
        if any(r['status']=='failed' for r in events):
            raise SystemExit('ROUND_E_FAILED: see round_e_status.json and logs')
        summarize(args.out,args.a_out,args.c_out,args.d_out)
        print('ROUND_E_COMPLETE',args.out,'training_jobs 3 mean_gate_jobs 6',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('run','summarize'):
        p = sub.add_parser(name)
        p.add_argument('--out',type=Path,default=DEFAULT_OUT)
        p.add_argument('--a-out',type=Path,default=DEFAULT_A)
        p.add_argument('--c-out',type=Path,default=DEFAULT_C)
        p.add_argument('--d-out',type=Path,default=DEFAULT_D)
        if name=='run':
            p.add_argument('--cache',type=Path,default=Path('/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'))
            p.add_argument('--gpus',nargs=3,default=['1','2','3'])
            p.add_argument('--seeds',nargs=3,type=int,default=[3351,3352,3353])
            p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args = parser.parse_args()
    if args.command=='run':
        if len(set(args.gpus))!=3 or len(set(args.seeds))!=3:
            parser.error('GPU IDs and seeds must each be unique')
        launch(args)
    else:
        summarize(args.out,args.a_out,args.c_out,args.d_out)


if __name__=='__main__':
    main()
