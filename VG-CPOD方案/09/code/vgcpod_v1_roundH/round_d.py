"""D1 + original-source C2 repeats, followed by fixed-checkpoint mean-gate evaluation."""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import round_b as shared
import round_c as previous

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = Path('/data/xhw/vgcpod_v1_runs/roundD_20261020')
DEFAULT_C_OUT = Path('/data/xhw/vgcpod_v1_runs/roundC_20261020')
DEFAULT_C_ROOT = Path('/data/xhw/vgcpod_project/vgcpod_v1_roundC')


def summarize(out, c_out, a_out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, records = previous.collect(c_out, ('C1','C2'), 'roundC')
    d_rows, d_records = previous.collect(out, ('D1',), 'roundD')
    rows += d_rows
    refs = previous.reference_records(a_out)
    for record in shared.read(ROOT/'references/round_c_pilot.json')['runs']:
        refs[(record['case'],record['seed'])] = dict(case=record['case'],seed=record['seed'],
            feature_manifest_sha256=record['feature_manifest_sha256'],**shared.metric_columns(record['metrics']))
    refs.update(records)
    refs.update(d_records)
    deltas, pairs = [], []
    for row in rows:
        names = ('A5','A6','C1','C2') if row['case']=='D1' else ('A5','A6','C1') if row['case']=='C2' else ('A5','A6')
        for name in names:
            other = refs.get((name,row['seed']))
            if other and other['feature_manifest_sha256']==row['feature_manifest_sha256']:
                deltas.append(dict(comparison=row['case']+'-'+name,seed=row['seed'],
                    **{key:row[key]-other[key] for key in shared.COLUMNS}))
                pairs.append((row,other))
    by_case = shared.aggregate(rows) if rows else []
    paired = shared.aggregate(deltas,'comparison') if deltas else []
    shared.csv_write(out/'round_d_summary.csv',rows)
    shared.csv_write(out/'round_d_mean_sd.csv',by_case)
    shared.csv_write(out/'round_d_paired_deltas.csv',deltas)
    shared.csv_write(out/'round_d_paired_mean_sd.csv',paired)
    pair_count = previous.export_pair_gains(out,pairs)
    effects = []
    for path in sorted((out/'mean_gate').glob('*/seed*/metrics.json')):
        record = shared.read(path)
        for setting,values in record['dynamic_minus_train_mean'].items():
            effects.append(dict(case=record['case'],seed=record['seed'],setting=setting,
                selected_epoch=record['selected_epoch'],train_images=record['train_images'],
                **{'dynamic_minus_train_mean_'+key:value for key,value in values.items()}))
    shared.csv_write(out/'mean_gate_effects.csv',effects)
    shared.dump(out/'round_d_summary.json',dict(runs=rows,by_case=by_case,paired_deltas=deltas,
        paired_mean_sd=paired,per_pair_rows=pair_count,mean_gate_effects=effects,
        selection_rule='epoch: first strict maximum VAL CW AUC_native; deltas: matching seeds and cache; one-seed SD is null'))
    print('ROUND_D_SUMMARY',out/'round_d_summary.json',flush=True)
    return rows


def run_process(command, source, log, gpu=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:
        f.write('\nROUND_D_COMMAND '+ ' '.join(map(str,command))+'\n')
        f.flush()
        process = subprocess.run(list(map(str,command)),cwd=str(source),env=previous.environment(gpu),
                                 stdout=f,stderr=subprocess.STDOUT)
    return process.returncode


def c_repeats(args):
    log = args.out/'logs/C2_launcher.log'
    command = [sys.executable,'-u',args.c_root/'round_c.py','run','--cache',args.cache,
        '--out',args.c_out,'--a-out',args.a_out,'--b-out',args.b_out,'--skip-b3',
        '--gpus',*args.gpus[:2],'--cases','C2','--seeds',*args.c_seeds,'--device',args.device]
    print('START_C2_ORIGINAL_SOURCE',args.c_root,'seeds',args.c_seeds,'gpus',args.gpus[:2],flush=True)
    code = run_process(command,args.c_root,log)
    if code:
        return [dict(case='C2',status='failed',exit_code=code,log=str(log))]
    selected = {('C2',seed) for seed in args.c_seeds}
    return [dict(event,source_root=str(args.c_root),output_root=str(args.c_out))
            for event in shared.read(args.c_out/'round_c_status.json')
            if (event.get('case'),event.get('seed')) in selected]


def d_train(args):
    events = []
    config = ROOT/'configs/round_d/D1.json'
    cfg = shared.read(config)
    for seed in args.d_seeds:
        run = args.out/'D1'/cfg['dataset']/cfg['variant']/('seed%d'%seed)
        log = args.out/'logs'/('D1_seed%d.log'%seed)
        if (run/'metrics.json').exists():
            if not shared.validate_completed(run,cfg,seed,args.cache):
                raise ValueError('D1 configuration/source differs: '+str(run))
            events.append(dict(case='D1',seed=seed,gpu=args.gpus[2],status='already_complete'))
            continue
        command = [sys.executable,'-u',ROOT/'run.py','train','--cache',args.cache,'--out',args.out/'D1',
            '--config',config,'--seed',seed,'--device',args.device]
        if (run/'protocol.json').exists():
            command.append('--resume')
        print('START_D1',seed,'GPU',args.gpus[2],'log',log,flush=True)
        code = run_process(command,ROOT,log,args.gpus[2])
        events.append(dict(case='D1',seed=seed,gpu=args.gpus[2],status='complete' if code==0 else 'failed',
                           exit_code=code,log=str(log)))
    return events


def gate_queue(gpu, jobs, args):
    events = []
    for case,seed,source,run in jobs:
        out = args.out/'mean_gate'/case/('seed%d'%seed)
        log = args.out/'logs'/('mean_gate_%s_seed%d.log'%(case,seed))
        print('START_MEAN_GATE',case,seed,'GPU',gpu,flush=True)
        code = run_process([sys.executable,'-u',ROOT/'mean_gate_eval.py','--source-root',source,
            '--cache',args.cache,'--run',run,'--out',out,'--device',args.device],source,log,gpu)
        events.append(dict(case=case,seed=seed,gpu=gpu,status='complete' if code==0 else 'failed',
                           exit_code=code,log=str(log)))
    return events


def launch(args):
    for key in ('out','cache','a_out','b_out','c_out','c_root'):
        setattr(args,key,getattr(args,key).resolve())
    if not (args.cache/'features_manifest.json').is_file():
        raise FileNotFoundError('Cache absent: '+str(args.cache))
    if not (args.c_root/'round_c.py').is_file():
        raise FileNotFoundError('Original C source absent: '+str(args.c_root))
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'.round_d.lock').open('a') as lock:
        if os.name=='posix':
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = [dict(case='C2',seed=seed,gpu=args.gpus[i%2],source_root=str(args.c_root))
                for i,seed in enumerate(args.c_seeds)] if not args.skip_c else []
        plan += [dict(case='D1',seed=seed,gpu=args.gpus[2],source_root=str(ROOT)) for seed in args.d_seeds]
        shared.dump(args.out/'round_d_jobs.json',plan)
        events = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(d_train,args):'D1'}
            if not args.skip_c:
                futures[pool.submit(c_repeats,args)] = 'C2'
            for future in as_completed(futures):
                try:
                    events += future.result()
                except Exception as error:
                    events.append(dict(case=futures[future],status='failed',error=str(error)))
                shared.dump(args.out/'round_d_status.json',events)
        if any(event['status']=='failed' for event in events):
            raise SystemExit('ROUND_D_TRAIN_FAILED: see round_d_status.json and logs')
        jobs = []
        for case,source,output,seeds in [('C2',args.c_root,args.c_out,args.gate_c_seeds),
                                        ('D1',ROOT,args.out,args.d_seeds)]:
            for seed in seeds:
                run = output/case/'ut-zap50k/V1'/('seed%d'%seed)
                if not (run/'metrics.json').is_file():
                    raise FileNotFoundError('Mean-gate checkpoint missing: '+str(run))
                jobs.append((case,seed,source,run))
        gate_events = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(gate_queue,gpu,jobs[i::3],args):gpu for i,gpu in enumerate(args.gpus)}
            for future in as_completed(futures):
                try:
                    gate_events += future.result()
                except Exception as error:
                    gate_events.append(dict(gpu=futures[future],status='failed',error=str(error)))
                shared.dump(args.out/'mean_gate_status.json',gate_events)
        summarize(args.out,args.c_out,args.a_out)
        if any(event['status']=='failed' for event in gate_events):
            raise SystemExit('ROUND_D_MEAN_GATE_FAILED: see mean_gate_status.json and logs')
        print('ROUND_D_COMPLETE',args.out,'training_jobs',len(plan),'mean_gate_jobs',len(jobs),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('run','summarize'):
        p = sub.add_parser(name)
        p.add_argument('--out',type=Path,default=DEFAULT_OUT)
        p.add_argument('--c-out',type=Path,default=DEFAULT_C_OUT)
        p.add_argument('--a-out',type=Path,default=previous.DEFAULT_A_OUT)
        if name=='run':
            p.add_argument('--cache',type=Path,default=Path('/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'))
            p.add_argument('--b-out',type=Path,default=previous.DEFAULT_B_OUT)
            p.add_argument('--c-root',type=Path,default=DEFAULT_C_ROOT)
            p.add_argument('--gpus',nargs=3,default=['1','2','3'])
            p.add_argument('--c-seeds',nargs='+',type=int,default=[3352,3353])
            p.add_argument('--gate-c-seeds',nargs='+',type=int,default=[3351,3352,3353])
            p.add_argument('--d-seeds',nargs='+',type=int,default=[3351])
            p.add_argument('--skip-c',action='store_true')
            p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args = parser.parse_args()
    if args.command=='run':
        for key in ('gpus','c_seeds','gate_c_seeds','d_seeds'):
            values = getattr(args,key)
            if len(values)!=len(set(values)):
                parser.error(key+' must be unique')
        launch(args)
    else:
        summarize(args.out,args.c_out,args.a_out)


if __name__=='__main__':
    main()
