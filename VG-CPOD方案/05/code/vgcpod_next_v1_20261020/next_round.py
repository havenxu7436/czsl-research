"""Run D1/3352, D1/3353, C1/3352 using the existing C/D v1 sources."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys


def interaction_summary(d, args):
    rows = d.shared.read(args.d_out/'round_d_summary.json')['runs']
    records = {(r['case'],r['seed']):r for r in rows}
    refs = d.previous.reference_records(args.a_out)
    contrasts = []
    for seed in (3351,3352):
        group = [records.get((name,seed)) for name in ('D1','C2','C1')]+[refs.get(('A5',seed))]
        if any(r is None for r in group):
            raise FileNotFoundError('Four-cell result absent for seed%d'%seed)
        if len({r['feature_manifest_sha256'] for r in group})!=1:
            raise ValueError('Four-cell feature caches differ')
        high,low,static,base = group
        contrasts.append(dict(comparison='centered_gate_x_rho',seed=seed,
            **{key:high[key]-low[key]-static[key]+base[key] for key in d.shared.COLUMNS}))
    d.shared.csv_write(args.out/'interaction_by_seed.csv',contrasts)
    d.shared.csv_write(args.out/'interaction_mean_sd.csv',d.shared.aggregate(contrasts,'comparison'))
    d.shared.dump(args.out/'interaction.json',dict(formula='(D1-C2)-(C1-A5)',
        seeds=[3351,3352],rows=contrasts,selection='Each run uses its first strict max VAL CW AUC_native'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'))
    parser.add_argument('--d-root',type=Path,default=Path('/data/xhw/vgcpod_project/vgcpod_v1_roundD'))
    parser.add_argument('--c-root',type=Path,default=Path('/data/xhw/vgcpod_project/vgcpod_v1_roundC'))
    parser.add_argument('--d-out',type=Path,default=Path('/data/xhw/vgcpod_v1_runs/roundD_20261020'))
    parser.add_argument('--c-out',type=Path,default=Path('/data/xhw/vgcpod_v1_runs/roundC_20261020'))
    parser.add_argument('--a-out',type=Path,default=Path('/data/xhw/vgcpod_v1_runs/roundA_20261019'))
    parser.add_argument('--b-out',type=Path,default=Path('/data/xhw/vgcpod_v1_runs/roundB_20261019'))
    parser.add_argument('--out',type=Path,default=Path('/data/xhw/vgcpod_v1_runs/next_20261020'))
    parser.add_argument('--gpus',nargs=3,default=['1','2','3'])
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args = parser.parse_args()
    if len(set(args.gpus))!=3:
        parser.error('Use three distinct GPU IDs')
    for name in ('cache','d_root','c_root','d_out','c_out','a_out','b_out','out'):
        setattr(args,name,getattr(args,name).resolve())
    sys.path.insert(0,str(args.d_root))
    import round_d as d
    if Path(d.__file__).resolve().parent!=args.d_root:
        raise ValueError('Selected D source differs from the imported source')
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'logs').mkdir(exist_ok=True)
    with (args.out/'.next.lock').open('a') as lock:
        if os.name=='posix':
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = [dict(case='D1',seed=3352,gpu=args.gpus[0]),
                dict(case='D1',seed=3353,gpu=args.gpus[1]),
                dict(case='C1',seed=3352,gpu=args.gpus[2])]
        d.shared.dump(args.out/'jobs.json',plan)

        def dynamic_job(gpu,seed):
            # Original D functions own their original ROOT, configuration and source identity.
            local = argparse.Namespace(out=args.d_out,cache=args.cache,device=args.device,
                d_seeds=[seed],gpus=[gpu,gpu,gpu])
            (args.d_out/'logs').mkdir(parents=True,exist_ok=True)
            trained = d.d_train(local)[0]
            if trained['status']=='failed':
                return dict(trained,phase='training')
            run = args.d_out/'D1/ut-zap50k/V1'/('seed%d'%seed)
            evaluated = d.gate_queue(gpu,[('D1',seed,args.d_root,run)],local)[0]
            return dict(case='D1',seed=seed,gpu=gpu,status=evaluated['status'],
                        training=trained,mean_gate=evaluated)

        def static_job(gpu):
            log = args.out/'logs/C1_seed3352_launcher.log'
            print('START_C1',3352,'GPU',gpu,'source',args.c_root,flush=True)
            command = [sys.executable,'-u',args.c_root/'round_c.py','run','--cache',args.cache,
                '--out',args.c_out,'--a-out',args.a_out,'--b-out',args.b_out,'--skip-b3',
                '--gpus',gpu,'--cases','C1','--seeds',3352,'--device',args.device]
            code = d.run_process(command,args.c_root,log)
            if code:
                return dict(case='C1',seed=3352,gpu=gpu,status='failed',exit_code=code,log=str(log))
            event = next(r for r in d.shared.read(args.c_out/'round_c_status.json')
                         if r.get('case')=='C1' and r.get('seed')==3352)
            return dict(event,source_root=str(args.c_root))

        events = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(dynamic_job,args.gpus[0],3352):plan[0],
                       pool.submit(dynamic_job,args.gpus[1],3353):plan[1],
                       pool.submit(static_job,args.gpus[2]):plan[2]}
            for future in as_completed(futures):
                try:
                    event = future.result()
                except Exception as error:
                    event = dict(futures[future],status='failed',error=str(error))
                events.append(event)
                d.shared.dump(args.out/'status.json',events)
                print('JOB_RESULT',event['case'],event['seed'],event['status'],flush=True)
        if any(r['status']=='failed' for r in events):
            raise SystemExit('NEXT_FAILED: see status.json and job logs')
        d.summarize(args.d_out,args.c_out,args.a_out)
        interaction_summary(d,args)
        d.shared.dump(args.out/'summary_locations.json',dict(
            combined_summary=str(args.d_out/'round_d_summary.json'),
            paired_mean_sd=str(args.d_out/'round_d_paired_mean_sd.csv'),
            mean_gate_effects=str(args.d_out/'mean_gate_effects.csv'),
            interaction=str(args.out/'interaction.json')))
        print('NEXT_COMPLETE',args.out,'training_jobs 3 mean_gate_jobs 2',flush=True)


if __name__=='__main__':
    main()
