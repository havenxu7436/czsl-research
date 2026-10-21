"""Three F1 seeds: beta=.05 at rho=.15, paired E1 comparisons and gate evaluations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys

import round_b as shared
import round_c as previous
from round_d import run_process
from round_e import completed

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = Path('/data/xhw/vgcpod_v1_runs/roundF_20261020')
DEFAULT_A = Path('/data/xhw/vgcpod_v1_runs/roundA_20261019')
DEFAULT_C = Path('/data/xhw/vgcpod_v1_runs/roundC_20261020')
DEFAULT_D = Path('/data/xhw/vgcpod_v1_runs/roundD_20261020')
DEFAULT_E = Path('/data/xhw/vgcpod_v1_runs/roundE_20261020')
REFERENCE_CASES = ('E1','D1','A5','A6','C2','C1')


def references(a_out,c_out,d_out,e_out):
    refs = {}
    for record in shared.read(ROOT/'references/pre_f_complete.json')['runs']:
        cfg = record['config']
        refs[(record['case'],record['seed'])] = dict(case=record['case'],seed=record['seed'],
            rho=cfg['rho'],beta=cfg['task_aux_beta'],source_group='packaged_reference',
            feature_manifest_sha256=record['feature_manifest_sha256'],selected_epoch=record['selected_epoch'],
            **shared.metric_columns(record['metrics']))
    for root,cases,stage in [(a_out,('A5','A6'),'roundA'),(c_out,('C1','C2'),'roundC'),
                             (d_out,('D1',),'roundD'),(e_out,('E1',),'roundE')]:
        _,live = previous.collect(root,cases,stage)
        refs.update(live)
    return refs


def exact_rows(out):
    rows = []
    for path in sorted((Path(out)/'F1').glob('ut-zap50k/V*/seed*/exact_selection/metrics.json')):
        run = path.parent.parent
        protocol,saved = shared.read(run/'protocol.json'),shared.read(path)
        cfg = protocol['config']
        rows.append(dict(case='F1_exact',seed=protocol['seed'],rho=cfg['rho'],beta=cfg['task_aux_beta'],
            selected_epoch=saved['selected_epoch'],selection=saved['selection'],run_dir=str(path.parent),
            feature_manifest_sha256=protocol['feature_manifest_sha256'],**shared.metric_columns(saved['metrics'])))
    return rows


def support_training_rows(rows):
    """Compare selected and final TRAIN losses; snapshots provide selected epochs."""
    from vgcpod.engine import task_decomposition
    snapshots = {(r['case'],r['seed']):r for r in shared.read(ROOT/'references/pre_f_complete.json')['runs']}
    result = []
    for row in sorted(rows,key=lambda r:(r['case'],r['seed'])):
        if 'run_dir' in row:
            history = shared.read(Path(row['run_dir'])/'history.json')
            entries = [('native_selected',history[row['selected_epoch']-1]),('final',history[-1])]
        else:
            snapshot = snapshots[(row['case'],row['seed'])]
            entries = [('native_selected',dict(epoch=row['selected_epoch'],train=snapshot['train']))]
        for stage,entry in entries:
            train = entry['train']
            parts = task_decomposition(train,row['beta'])
            result.append(dict(case=row['case'],seed=row['seed'],stage=stage,epoch=entry['epoch'],
                rho=row['rho'],beta=row['beta'],source_group=row['source_group'],
                pair=train['pair'],support=train['support'],task=train['task'],orth=train['orth'],
                support_weight=parts['support_weight'],pair_weighted=parts['pair_weighted'],
                support_weighted=parts['support_weighted'],
                task_epoch_reconstruction_error=parts['task_reconstruction_error'],
                task_batch_reconstruction_error_mean=train.get('task_reconstruction_error'),
                support_mass=train['support_mass'],clipped_step_fraction=train['clipped_step_fraction'],
                feature_manifest_sha256=row['feature_manifest_sha256']))
    return result


def summarize(out,a_out=DEFAULT_A,c_out=DEFAULT_C,d_out=DEFAULT_D,e_out=DEFAULT_E):
    out = Path(out)
    rows,records = previous.collect(out,('F1',),'roundF')
    if not rows:
        raise FileNotFoundError('No completed F1 results: '+str(out))
    refs = references(a_out,c_out,d_out,e_out)
    deltas,pairs = [],[]
    for row in rows:
        for case in REFERENCE_CASES:
            other = refs.get((case,row['seed']))
            if other and other['feature_manifest_sha256']==row['feature_manifest_sha256']:
                deltas.append(dict(comparison='F1-'+case,seed=row['seed'],
                    **{key:row[key]-other[key] for key in shared.COLUMNS}))
                pairs.append((row,other))
    secondary = exact_rows(out)
    selection_effect = []
    for row in secondary:
        native = records[('F1',row['seed'])]
        selection_effect.append(dict(comparison='F1_exact-F1',seed=row['seed'],
            **{key:row[key]-native[key] for key in shared.COLUMNS}))
    cache_ids = {r['feature_manifest_sha256'] for r in rows}
    beta_rows = [r for r in refs.values() if r['case']=='E1' and r['rho']==.15
                 and r['feature_manifest_sha256'] in cache_ids]+rows
    beta_rows.sort(key=lambda r:(r['case'],r['seed']))
    keys = ('case','seed','rho','beta','source_group','selected_epoch','feature_manifest_sha256')+shared.COLUMNS
    beta_metrics = [{key:r[key] for key in keys} for r in beta_rows]
    support_rows = support_training_rows(beta_rows)
    effects = []
    for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')]:
        for path in sorted((out/folder/'F1').glob('seed*/metrics.json')):
            data = shared.read(path)
            for setting,values in data['dynamic_minus_train_mean'].items():
                effects.append(dict(case='F1',seed=data['seed'],selection=selection,setting=setting,
                    selected_epoch=data['selected_epoch'],train_images=data['train_images'],
                    **{'dynamic_minus_train_mean_'+key:value for key,value in values.items()}))
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('round_f_summary',rows),('round_f_mean_sd',shared.aggregate(rows)),
                      ('round_f_paired_deltas',deltas),('round_f_paired_mean_sd',shared.aggregate(deltas,'comparison')),
                      ('beta_native_summary',beta_metrics),('beta_native_mean_sd',shared.aggregate(beta_metrics)),
                      ('support_training_summary',support_rows),('exact_selection_summary',secondary),
                      ('exact_selection_effects',selection_effect),
                      ('exact_selection_effect_mean_sd',shared.aggregate(selection_effect,'comparison')),
                      ('mean_gate_effects',effects)]:
        shared.csv_write(out/(name+'.csv'),data)
    pair_count = previous.export_pair_gains(out,pairs)
    result = dict(runs=rows,by_case=shared.aggregate(rows),paired_deltas=deltas,
        paired_mean_sd=shared.aggregate(deltas,'comparison'),beta_native_runs=beta_metrics,
        beta_native_mean_sd=shared.aggregate(beta_metrics),support_training_runs=support_rows,
        exact_selection_runs=secondary,exact_selection_effects=selection_effect,
        mean_gate_effects=effects,per_pair_rows=pair_count,
        selection_rule='Primary: first strict max VAL CW AUC_native; secondary: first strict max VAL CW AUC_exact',
        comparison_rule='F1-E1 is the beta comparison at rho=.15; deltas use matching seeds and feature cache',
        support_rule='Weighted TRAIN epoch means; batch residual is mean absolute batch error; epoch residual uses epoch means')
    shared.dump(out/'round_f_summary.json',result)
    print('ROUND_F_SUMMARY',out/'round_f_summary.json',flush=True)
    return result


def worker(gpu,seed,args):
    config = ROOT/'configs/round_f/F1.json'
    cfg = shared.read(config)
    run = args.out/'F1'/cfg['dataset']/cfg['variant']/('seed%d'%seed)
    log = args.out/'logs'/('F1_seed%d.log'%seed)
    if completed(run,cfg,seed,args.cache):
        training = dict(status='already_complete',log=str(log))
        print('ALREADY_COMPLETE','F1',seed,'GPU',gpu,flush=True)
    else:
        command = [sys.executable,'-u',ROOT/'run.py','train','--cache',args.cache,
            '--out',args.out/'F1','--config',config,'--seed',seed,'--device',args.device]
        if (run/'protocol.json').exists():
            command.append('--resume')
        print('START_F1',seed,'GPU',gpu,'log',log,flush=True)
        code = run_process(command,ROOT,log,gpu)
        training = dict(status='complete' if code==0 else 'failed',exit_code=code,log=str(log))
        if code:
            return dict(case='F1',seed=seed,gpu=gpu,status='failed',training=training)
    evaluations = []
    for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')]:
        out = args.out/folder/'F1'/('seed%d'%seed)
        log = args.out/'logs'/('mean_gate_%s_F1_seed%d.log'%(selection,seed))
        command = [sys.executable,'-u',ROOT/'mean_gate_eval.py','--source-root',ROOT,
            '--cache',args.cache,'--run',run,'--out',out,'--selection',selection,'--device',args.device]
        print('START_MEAN_GATE',selection,seed,'GPU',gpu,flush=True)
        code = run_process(command,ROOT,log,gpu)
        evaluations.append(dict(selection=selection,status='complete' if code==0 else 'failed',
            exit_code=code,log=str(log)))
        if code:
            break
    status = 'failed' if any(r['status']=='failed' for r in evaluations) else 'complete'
    return dict(case='F1',seed=seed,gpu=gpu,status=status,training=training,mean_gate=evaluations)


def launch(args):
    for key in ('out','cache','a_out','c_out','d_out','e_out'):
        setattr(args,key,getattr(args,key).resolve())
    if not (args.cache/'features_manifest.json').is_file():
        raise FileNotFoundError('Feature cache absent: '+str(args.cache))
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'logs').mkdir(exist_ok=True)
    with (args.out/'.round_f.lock').open('a') as lock:
        if os.name=='posix':
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = [dict(case='F1',seed=seed,gpu=gpu,rho=.15,beta=.05) for gpu,seed in zip(args.gpus,args.seeds)]
        shared.dump(args.out/'round_f_jobs.json',plan)
        events = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(worker,job['gpu'],job['seed'],args):job for job in plan}
            for future in as_completed(futures):
                try:
                    event = future.result()
                except Exception as error:
                    event = dict(futures[future],status='failed',error=str(error))
                events.append(event)
                shared.dump(args.out/'round_f_status.json',events)
                print('JOB_RESULT','F1',event['seed'],event['status'],flush=True)
        if any(r['status']=='failed' for r in events):
            raise SystemExit('ROUND_F_FAILED: see round_f_status.json and logs')
        summarize(args.out,args.a_out,args.c_out,args.d_out,args.e_out)
        print('ROUND_F_COMPLETE',args.out,'training_jobs 3 mean_gate_jobs 6',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('run','summarize'):
        p = sub.add_parser(name)
        p.add_argument('--out',type=Path,default=DEFAULT_OUT)
        p.add_argument('--a-out',type=Path,default=DEFAULT_A)
        p.add_argument('--c-out',type=Path,default=DEFAULT_C)
        p.add_argument('--d-out',type=Path,default=DEFAULT_D)
        p.add_argument('--e-out',type=Path,default=DEFAULT_E)
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
        summarize(args.out,args.a_out,args.c_out,args.d_out,args.e_out)


if __name__=='__main__':
    main()
