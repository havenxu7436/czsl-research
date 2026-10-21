"""H1: shared TRAIN visual means, three residual-gate seeds and paired G0/G1 comparisons."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
from pathlib import Path
import sys

import round_b as shared
import round_c as previous
from round_d import run_process
from vgcpod.common import sha
from vgcpod.engine import task_decomposition
from ranking_diagnostics import summarize_entries
from vgcpod.gate_features import prepare_gate_statistics
from round_g import completed

ROOT=Path(__file__).resolve().parent
DEFAULT_OUT=Path('/data/xhw/vgcpod_v1_runs/roundH_20261020')
DEFAULTS={stage:Path('/data/xhw/vgcpod_v1_runs/round%s_%s'%(stage,'20261019' if stage=='A' else '20261020'))
          for stage in ('A','C','D','E','F','G')}
CASES=('H1',)


def references(args):
    refs={}
    for record in shared.read(ROOT/'references/pre_h_complete.json')['runs']:
        cfg=record['config']
        refs[(record['case'],record['seed'])]=dict(case=record['case'],seed=record['seed'],
            rho=cfg['rho'],beta=cfg['task_aux_beta'],source_group='packaged_reference',
            selected_epoch=record['selected_epoch'],feature_manifest_sha256=record['feature_manifest_sha256'],
            **shared.metric_columns(record['metrics']))
    for stage,cases in [('A',('A5','A6')),('C',('C1','C2')),('D',('D1',)),('E',('E1',)),('F',('F1',)),('G',('G0','G1'))]:
        _,live=previous.collect(getattr(args,stage.lower()+'_out'),cases,'round'+stage)
        refs.update(live)
    return refs


def support_training_rows(rows):
    snapshots={(r['case'],r['seed']):r for r in shared.read(ROOT/'references/pre_h_complete.json')['runs']}
    result=[]
    for row in sorted(rows,key=lambda r:(r['case'],r['seed'])):
        if 'run_dir' in row:
            history=shared.read(Path(row['run_dir'])/'history.json')
            entries=[('native_selected',history[row['selected_epoch']-1]),('final',history[-1])]
        else:
            entries=[('native_selected',dict(epoch=row['selected_epoch'],train=snapshots[(row['case'],row['seed'])]['train']))]
        for stage,entry in entries:
            train=entry['train'];parts=task_decomposition(train,row['beta'])
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


def historical_exact_rows(args):
    refs={}
    for record in shared.read(ROOT/'references/pre_h_complete.json')['runs']:
        if record['case'] not in ('G0','G1'):
            continue
        saved=record['exact_selection'];cfg=record['config']
        row=dict(case=record['case']+'_exact',seed=record['seed'],rho=cfg['rho'],beta=cfg['task_aux_beta'],
            selected_epoch=saved['selected_epoch'],source_group='packaged_reference',
            feature_manifest_sha256=record['feature_manifest_sha256'],**shared.metric_columns(saved['metrics']))
        run=Path(args.g_out)/record['case']/cfg['dataset']/cfg['variant']/('seed%d'%record['seed'])
        if (run/'exact_selection/metrics.json').exists():
            protocol=shared.read(run/'protocol.json');live=shared.read(run/'exact_selection/metrics.json')
            row.update(selected_epoch=live['selected_epoch'],source_group='roundG_exact',
                feature_manifest_sha256=protocol['feature_manifest_sha256'],
                run_dir=str(run/'exact_selection'),**shared.metric_columns(live['metrics']))
        refs[(row['case'],row['seed'])]=row
    return refs


def summarize(args):
    out=Path(args.out)
    rows,records=previous.collect(out,CASES,'roundH')
    if not rows:
        raise FileNotFoundError('No completed H1 results: '+str(out))
    refs=references(args);refs.update(records)
    deltas,pairs=[],[]
    for row in rows:
        targets=('G1','G0','E1','F1','D1','A5','A6','C2','C1')
        for case in targets:
            other=refs.get((case,row['seed']))
            if other and other['feature_manifest_sha256']==row['feature_manifest_sha256']:
                deltas.append(dict(comparison=row['case']+'-'+case,seed=row['seed'],
                    **{key:row[key]-other[key] for key in shared.COLUMNS}))
                pairs.append((row,other))
    secondary,effects,entries=[],[],[]
    for row in rows:
        run=Path(row['run_dir']);saved=shared.read(run/'exact_selection/metrics.json')
        secondary.append(dict(case=row['case']+'_exact',seed=row['seed'],rho=row['rho'],beta=row['beta'],
            selected_epoch=saved['selected_epoch'],selection=saved['selection'],
            **shared.metric_columns(saved['metrics'])))
        effects.append(dict(comparison=row['case']+'_exact-'+row['case'],seed=row['seed'],
            **{key:secondary[-1][key]-row[key] for key in shared.COLUMNS}))
        condition='dynamic'
        for selection,path,epoch in [('native',run,row['selected_epoch']),
                                     ('exact',run/'exact_selection',saved['selected_epoch'])]:
            entries.append(dict(case=row['case'],seed=row['seed'],selection=selection,condition=condition,
                path=str(path),selected_epoch=epoch,cache=str(args.cache)))
    cache_ids={r['feature_manifest_sha256'] for r in rows}
    exact_refs=historical_exact_rows(args)
    exact_deltas=[]
    for row in secondary:
        for case in ('G0_exact','G1_exact'):
            ref=exact_refs.get((case,row['seed']))
            if ref and ref['feature_manifest_sha256'] in cache_ids:
                exact_deltas.append(dict(comparison='H1_exact-'+case,seed=row['seed'],
                    **{key:row[key]-ref[key] for key in shared.COLUMNS}))
    for ref in exact_refs.values():
        if 'run_dir' in ref and ref['feature_manifest_sha256'] in cache_ids:
            entries.append(dict(case=ref['case'].split('_')[0],seed=ref['seed'],selection='exact',
                condition='static' if ref['case']=='G0_exact' else 'dynamic',path=ref['run_dir'],
                selected_epoch=ref['selected_epoch'],cache=str(args.cache)))
    beta_rows=[r for r in refs.values() if r['case'] in ('G0','G1','H1') and r['rho']==.15 and r['beta']==.075
               and r['feature_manifest_sha256'] in cache_ids]
    beta_rows.sort(key=lambda r:(r['beta'],r['seed']))
    keys=('case','seed','rho','beta','source_group','selected_epoch','feature_manifest_sha256')+shared.COLUMNS
    beta_metrics=[{key:r[key] for key in keys} for r in beta_rows]
    support_rows=support_training_rows([r for r in refs.values() if r['case'] in ('E1','F1','G0','G1','H1')
                                       and r['feature_manifest_sha256'] in cache_ids])
    mean_effects=[]
    for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')]:
        for path in sorted((out/folder/'H1').glob('seed*/metrics.json')):
            saved=shared.read(path)
            for setting,values in saved['dynamic_minus_train_mean'].items():
                mean_effects.append(dict(case='H1',seed=saved['seed'],selection=selection,setting=setting,
                    selected_epoch=saved['selected_epoch'],train_images=saved['train_images'],
                    **{'dynamic_minus_train_mean_'+key:value for key,value in values.items()}))
            entries.append(dict(case='H1',seed=saved['seed'],selection=selection,condition='train_mean',
                path=str(path.parent/'train_mean'),selected_epoch=saved['selected_epoch'],cache=str(args.cache)))
    for row in refs.values():
        if row['case'] not in CASES and 'run_dir' in row and row['feature_manifest_sha256'] in cache_ids:
            entries.append(dict(case=row['case'],seed=row['seed'],selection='native',
                condition='static' if row['case'] in ('A5','A6','G0') else 'dynamic',
                path=row['run_dir'],selected_epoch=row['selected_epoch'],cache=str(args.cache)))
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('round_h_summary',rows),('round_h_mean_sd',shared.aggregate(rows)),
                      ('round_h_paired_deltas',deltas),('round_h_paired_mean_sd',shared.aggregate(deltas,'comparison')),
                      ('matched_native_summary',beta_metrics),('matched_native_mean_sd',shared.aggregate(beta_metrics)),
                      ('support_training_summary',support_rows),('exact_selection_summary',secondary),
                      ('exact_selection_effects',effects),
                      ('exact_selection_effect_mean_sd',shared.aggregate(effects,'comparison')),
                      ('exact_reference_deltas',exact_deltas),
                      ('exact_reference_paired_mean_sd',shared.aggregate(exact_deltas,'comparison')),
                      ('mean_gate_effects',mean_effects)]:
        shared.csv_write(out/(name+'.csv'),data)
    pair_count=previous.export_pair_gains(out,pairs)
    diagnostics=summarize_entries(entries,shared.read(Path(args.cache)/'features_manifest.json'),out)
    result=dict(runs=rows,by_case=shared.aggregate(rows),paired_deltas=deltas,
        paired_mean_sd=shared.aggregate(deltas,'comparison'),matched_native_runs=beta_metrics,
        matched_native_mean_sd=shared.aggregate(beta_metrics),support_training_runs=support_rows,
        exact_selection_runs=secondary,exact_selection_effects=effects,mean_gate_effects=mean_effects,
        exact_reference_deltas=exact_deltas,exact_reference_paired_mean_sd=shared.aggregate(exact_deltas,'comparison'),
        per_pair_rows=pair_count,ranking_diagnostic_rows=len(diagnostics['runs']),
        selection_rule='First strict max VAL CW native/exact independently; primary comparisons use native',
        comparison_rule='H1-G1: gate feature transform; H1-G0: matched static control; same seeds/cache; exact comparisons use exact-selected references')
    shared.dump(out/'round_h_summary.json',result)
    print('ROUND_H_SUMMARY',out/'round_h_summary.json',flush=True)
    return result


def prepare_runtime_config(args):
    statistics=prepare_gate_statistics(args.cache,args.out/'feature_statistics')
    cfg=shared.read(ROOT/'configs/round_h/H1.json')
    cfg.update(gate_statistics_path=str(statistics.resolve()),gate_statistics_sha256=sha(statistics))
    target=args.out/'configs/H1.json'
    shared.dump(target,cfg)
    return target


def worker(gpu,seed,args):
    config=args.runtime_config;cfg=shared.read(config)
    run=args.out/'H1'/cfg['dataset']/cfg['variant']/('seed%d'%seed)
    log=args.out/'logs'/('H1_seed%d.log'%seed)
    if completed(run,cfg,seed,args.cache):
        training=dict(status='already_complete',log=str(log))
        print('ALREADY_COMPLETE','H1',seed,'GPU',gpu,flush=True)
    else:
        command=[sys.executable,'-u',ROOT/'run.py','train','--cache',args.cache,
            '--out',args.out/'H1','--config',config,'--seed',seed,'--device',args.device]
        if (run/'protocol.json').exists():command.append('--resume')
        print('START_H1',seed,'GPU',gpu,'log',log,flush=True)
        code=run_process(command,ROOT,log,gpu)
        training=dict(status='complete' if code==0 else 'failed',exit_code=code,log=str(log))
        if code:return dict(case='H1',seed=seed,gpu=gpu,status='failed',training=training)
    evaluations=[]
    for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')]:
        destination=args.out/folder/'H1'/('seed%d'%seed)
        log=args.out/'logs'/('mean_gate_%s_H1_seed%d.log'%(selection,seed))
        command=[sys.executable,'-u',ROOT/'mean_gate_eval.py','--source-root',ROOT,
            '--cache',args.cache,'--run',run,'--out',destination,'--selection',selection,'--device',args.device]
        print('START_MEAN_GATE',selection,'H1',seed,'GPU',gpu,flush=True)
        code=run_process(command,ROOT,log,gpu)
        evaluations.append(dict(selection=selection,status='complete' if code==0 else 'failed',exit_code=code,log=str(log)))
        if code:break
    return dict(case='H1',seed=seed,gpu=gpu,
        status='failed' if any(r['status']=='failed' for r in evaluations) else 'complete',
        training=training,mean_gate=evaluations)


def launch(args):
    for key in ('out','cache','a_out','c_out','d_out','e_out','f_out','g_out'):
        setattr(args,key,getattr(args,key).resolve())
    if not (args.cache/'features_manifest.json').is_file():
        raise FileNotFoundError('Feature cache absent: '+str(args.cache))
    (args.out/'logs').mkdir(parents=True,exist_ok=True)
    with (args.out/'.round_h.lock').open('a') as lock:
        if os.name=='posix':
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.runtime_config=prepare_runtime_config(args)
        jobs=[dict(case='H1',seed=seed,gpu=gpu,rho=.15,beta=.075) for gpu,seed in zip(args.gpus,args.seeds)]
        shared.dump(args.out/'round_h_jobs.json',jobs)
        events=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(worker,job['gpu'],job['seed'],args):job for job in jobs}
            for future in as_completed(futures):
                try:event=future.result()
                except Exception as error:event=dict(futures[future],status='failed',error=str(error))
                events.append(event);shared.dump(args.out/'round_h_status.json',events)
                print('JOB_RESULT','H1',event['seed'],event['status'],flush=True)
        if any(r['status']=='failed' for r in events):
            raise SystemExit('ROUND_H_FAILED: see round_h_status.json and logs')
        summarize(args)
        print('ROUND_H_COMPLETE',args.out,'training_jobs 3 mean_gate_jobs 6',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run','summarize'):
        p=sub.add_parser(name)
        p.add_argument('--out',type=Path,default=DEFAULT_OUT)
        p.add_argument('--cache',type=Path,default=Path('/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'))
        for stage,folder in DEFAULTS.items():p.add_argument('--'+stage.lower()+'-out',type=Path,default=folder)
        if name=='run':
            p.add_argument('--gpus',nargs=3,default=['1','2','3'])
            p.add_argument('--seeds',nargs=3,type=int,default=[3351,3352,3353])
            p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    if args.command=='run':
        if len(set(args.gpus))!=3 or len(set(args.seeds))!=3:parser.error('GPU IDs and seeds must each be unique')
        launch(args)
    else:summarize(args)


if __name__=='__main__':
    main()

