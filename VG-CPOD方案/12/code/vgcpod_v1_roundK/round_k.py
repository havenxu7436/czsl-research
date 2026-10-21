"""K1/K2 dynamic readout step sizes: six trainings and twelve gate evaluations."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
import json
import shutil
import numpy as np
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
from readout_diagnostics import summarize_readout_optimizer

ROOT=Path(__file__).resolve().parent
DEFAULT_OUT=Path('/data/xhw/vgcpod_v1_runs/roundK_20261021')
DEFAULTS={stage:Path('/data/xhw/vgcpod_v1_runs/round%s_%s'%(stage,'20261019' if stage=='A' else '20261021' if stage in ('I','J') else '20261020'))
          for stage in ('A','C','D','E','F','G','H','I','J')}
CASES=('K1','K2')


def references(args):
    refs={}
    for record in shared.read(ROOT/'references/pre_k_complete.json')['runs']:
        cfg=record['config']
        refs[(record['case'],record['seed'])]=dict(case=record['case'],seed=record['seed'],
            rho=cfg['rho'],beta=cfg['task_aux_beta'],source_group='packaged_reference',
            selected_epoch=record['selected_epoch'],feature_manifest_sha256=record['feature_manifest_sha256'],
            **shared.metric_columns(record['metrics']))
    for stage,cases in [('A',('A5','A6')),('C',('C1','C2')),('D',('D1',)),('E',('E1',)),('F',('F1',)),('G',('G0','G1')),('H',('H1',)),('I',('I1','I2')),('J',('J1',))]:
        _,live=previous.collect(getattr(args,stage.lower()+'_out'),cases,'round'+stage)
        refs.update(live)
    return refs


def support_training_rows(rows):
    snapshots={(r['case'],r['seed']):r for r in shared.read(ROOT/'references/pre_k_complete.json')['runs']}
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
    for record in shared.read(ROOT/'references/pre_k_complete.json')['runs']:
        if record['case'] not in ('G0','G1','H1','I1','I2','J1'):
            continue
        saved=record['exact_selection'];cfg=record['config']
        row=dict(case=record['case']+'_exact',seed=record['seed'],rho=cfg['rho'],beta=cfg['task_aux_beta'],
            selected_epoch=saved['selected_epoch'],source_group='packaged_reference',
            feature_manifest_sha256=record['feature_manifest_sha256'],**shared.metric_columns(saved['metrics']))
        run=Path(getattr(args,record['case'][0].lower()+'_out'))/record['case']/cfg['dataset']/cfg['variant']/('seed%d'%record['seed'])
        if (run/'exact_selection/metrics.json').exists():
            protocol=shared.read(run/'protocol.json');live=shared.read(run/'exact_selection/metrics.json')
            row.update(selected_epoch=live['selected_epoch'],source_group='round'+record['case'][0]+'_exact',
                feature_manifest_sha256=protocol['feature_manifest_sha256'],
                run_dir=str(run/'exact_selection'),**shared.metric_columns(live['metrics']))
        refs[(row['case'],row['seed'])]=row
    return refs


def summarize(args):
    out=Path(args.out)
    rows,records=previous.collect(out,CASES,'roundK')
    if not rows:
        raise FileNotFoundError('No completed K1/K2 results: '+str(out))
    refs=references(args);refs.update(records)
    deltas,pairs=[],[]
    for row in rows:
        targets=('J1','I2','G0','G1','H1','I1','D1','E1','F1','A5','A6','C2','C1')
        if row['case']=='K1':targets+=('K2',)
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
            feature_manifest_sha256=row['feature_manifest_sha256'],
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
    all_exact=dict(exact_refs)
    all_exact.update({(r['case'],r['seed']):r for r in secondary})
    for row in secondary:
        targets=('J1_exact','I2_exact','G0_exact','G1_exact','H1_exact','I1_exact')
        if row['case']=='K1_exact':targets+=('K2_exact',)
        for case in targets:
            ref=all_exact.get((case,row['seed']))
            if ref and ref['feature_manifest_sha256'] in cache_ids:
                exact_deltas.append(dict(comparison=row['case']+'-'+case,seed=row['seed'],
                    **{key:row[key]-ref[key] for key in shared.COLUMNS}))
    for ref in exact_refs.values():
        if 'run_dir' in ref and ref['feature_manifest_sha256'] in cache_ids:
            entries.append(dict(case=ref['case'].split('_')[0],seed=ref['seed'],selection='exact',
                condition='static' if ref['case']=='G0_exact' else 'dynamic',path=ref['run_dir'],
                selected_epoch=ref['selected_epoch'],cache=str(args.cache)))
    beta_rows=[r for r in refs.values() if r['case'] in ('G0','G1','H1','I1','I2','J1','K1','K2') and r['rho']==.15 and r['beta']==.075
               and r['feature_manifest_sha256'] in cache_ids]
    beta_rows.sort(key=lambda r:(r['beta'],r['seed']))
    keys=('case','seed','rho','beta','source_group','selected_epoch','feature_manifest_sha256')+shared.COLUMNS
    beta_metrics=[{key:r[key] for key in keys} for r in beta_rows]
    support_rows=support_training_rows([r for r in refs.values() if r['case'] in ('E1','F1','G0','G1','H1','I1','I2','J1','K1','K2')
                                       and r['feature_manifest_sha256'] in cache_ids])
    readout_rows=[]
    for row in rows:
        run=Path(row['run_dir']);history=shared.read(run/'history.json')
        exact=shared.read(run/'exact_selection/metrics.json')['selected_epoch']
        for stage,epoch in [('native_selected',row['selected_epoch']),('exact_selected',exact),('final',len(history))]:
            train=history[epoch-1]['train']
            readout_rows.append(dict(case=row['case'],seed=row['seed'],stage=stage,epoch=epoch,
                **{key:value for key,value in train.items() if key.startswith(('cp_static','cp_dynamic','centered_delta'))}))
    optimizer_rows=[]
    for row in refs.values():
        if row['case'] in ('J1','K1','K2') and 'run_dir' in row and row['feature_manifest_sha256'] in cache_ids:
            optimizer_rows.extend(summarize_readout_optimizer(row))
    mean_effects=[]
    for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')]:
        for path in sorted((out/folder).glob('K*/seed*/metrics.json')):
            saved=shared.read(path)
            for setting,values in saved['dynamic_minus_train_mean'].items():
                mean_effects.append(dict(case=saved['case'],seed=saved['seed'],selection=selection,setting=setting,
                    selected_epoch=saved['selected_epoch'],train_images=saved['train_images'],
                    **{'dynamic_minus_train_mean_'+key:value for key,value in values.items()}))
            entries.append(dict(case=saved['case'],seed=saved['seed'],selection=selection,condition='train_mean',
                path=str(path.parent/'train_mean'),selected_epoch=saved['selected_epoch'],cache=str(args.cache)))
    for row in refs.values():
        if row['case'] not in CASES and 'run_dir' in row and row['feature_manifest_sha256'] in cache_ids:
            entries.append(dict(case=row['case'],seed=row['seed'],selection='native',
                condition='static' if row['case'] in ('A5','A6','G0') else 'dynamic',
                path=row['run_dir'],selected_epoch=row['selected_epoch'],cache=str(args.cache)))
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('round_k_summary',rows),('round_k_mean_sd',shared.aggregate(rows)),
                      ('round_k_paired_deltas',deltas),('round_k_paired_mean_sd',shared.aggregate(deltas,'comparison')),
                      ('matched_native_summary',beta_metrics),('matched_native_mean_sd',shared.aggregate(beta_metrics)),
                      ('support_training_summary',support_rows),('exact_selection_summary',secondary),
                      ('exact_selection_effects',effects),
                      ('exact_selection_effect_mean_sd',shared.aggregate(effects,'comparison')),
                      ('exact_reference_deltas',exact_deltas),
                      ('exact_reference_paired_mean_sd',shared.aggregate(exact_deltas,'comparison')),
                      ('mean_gate_effects',mean_effects),('readout_training_summary',readout_rows),
                      ('readout_optimizer_summary',optimizer_rows)]:
        shared.csv_write(out/(name+'.csv'),data)
    pair_count=previous.export_pair_gains(out,pairs)
    diagnostics=summarize_entries(entries,shared.read(Path(args.cache)/'features_manifest.json'),out)
    result=dict(runs=rows,by_case=shared.aggregate(rows),paired_deltas=deltas,
        paired_mean_sd=shared.aggregate(deltas,'comparison'),matched_native_runs=beta_metrics,
        matched_native_mean_sd=shared.aggregate(beta_metrics),support_training_runs=support_rows,
        exact_selection_runs=secondary,exact_selection_effects=effects,mean_gate_effects=mean_effects,
        exact_reference_deltas=exact_deltas,exact_reference_paired_mean_sd=shared.aggregate(exact_deltas,'comparison'),
        readout_training_runs=readout_rows,readout_optimizer_runs=optimizer_rows,
        per_pair_rows=pair_count,ranking_diagnostic_rows=len(diagnostics['runs']),
        selection_rule='First strict max VAL CW native/exact independently; primary comparisons use native',
        comparison_rule='K1/K2-J1: Wd learning rate only; K1-K2: dynamic step size; I2/G0/G1: matched controls; D1: performance reference; native/exact references matched by selection')
    shared.dump(out/'round_k_summary.json',result)
    print('ROUND_K_SUMMARY',out/'round_k_summary.json',flush=True)
    return result



def prepare_runtime_configs(args):
    directory=args.out/'feature_statistics'
    target=directory/'train_object_query_mean.npz'
    reference=args.j_out/'feature_statistics/train_object_query_mean.npz'
    if not target.exists() and reference.exists():
        with np.load(reference,allow_pickle=False) as saved:
            identity=json.loads(str(saved['identity_json'].item()))
        if identity['feature_manifest_sha256']==sha(args.cache/'features_manifest.json'):
            directory.mkdir(parents=True,exist_ok=True)
            shutil.copy2(reference,target)
            print('TRAIN_GATE_MEAN_COPIED_FROM_J',reference,flush=True)
    statistics=prepare_gate_statistics(args.cache,directory)
    result={}
    for case in CASES:
        cfg=shared.read(ROOT/'configs/round_k'/(case+'.json'))
        cfg.update(gate_statistics_path=str(statistics.resolve()),gate_statistics_sha256=sha(statistics))
        path=args.out/'configs'/(case+'.json');shared.dump(path,cfg);result[case]=path
    return result


def worker(gpu,seed,args):
    event=dict(seed=seed,gpu=gpu,status='complete',trainings=[],mean_gate=[])
    for case in CASES:
        config=args.runtime_configs[case];cfg=shared.read(config)
        run=args.out/case/cfg['dataset']/cfg['variant']/('seed%d'%seed)
        log=args.out/'logs'/('%s_seed%d.log'%(case,seed))
        if completed(run,cfg,seed,args.cache):
            training=dict(case=case,status='already_complete',log=str(log))
            print('ALREADY_COMPLETE',case,seed,'GPU',gpu,flush=True)
        else:
            command=[sys.executable,'-u',ROOT/'run.py','train','--cache',args.cache,
                '--out',args.out/case,'--config',config,'--seed',seed,'--device',args.device]
            if (run/'protocol.json').exists():command.append('--resume')
            print('START',case,seed,'GPU',gpu,'log',log,flush=True)
            code=run_process(command,ROOT,log,gpu)
            training=dict(case=case,status='complete' if code==0 else 'failed',exit_code=code,log=str(log))
        event['trainings'].append(training)
        if training['status']=='failed':
            event['status']='failed';return event
    for case in CASES:
        cfg=shared.read(args.runtime_configs[case])
        run=args.out/case/cfg['dataset']/cfg['variant']/('seed%d'%seed)
        for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')]:
            destination=args.out/folder/case/('seed%d'%seed)
            log=args.out/'logs'/('mean_gate_%s_%s_seed%d.log'%(selection,case,seed))
            command=[sys.executable,'-u',ROOT/'mean_gate_eval.py','--source-root',ROOT,
                '--cache',args.cache,'--run',run,'--out',destination,'--selection',selection,'--device',args.device]
            print('START_MEAN_GATE',selection,case,seed,'GPU',gpu,flush=True)
            code=run_process(command,ROOT,log,gpu)
            event['mean_gate'].append(dict(case=case,selection=selection,status='complete' if code==0 else 'failed',exit_code=code,log=str(log)))
            if code:
                event['status']='failed';return event
    return event


def launch(args):
    for key in ('out','cache','a_out','c_out','d_out','e_out','f_out','g_out','h_out','i_out','j_out'):
        setattr(args,key,getattr(args,key).resolve())
    if not (args.cache/'features_manifest.json').is_file():
        raise FileNotFoundError('Feature cache absent: '+str(args.cache))
    (args.out/'logs').mkdir(parents=True,exist_ok=True)
    with (args.out/'.round_k.lock').open('a') as lock:
        if os.name=='posix':
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.runtime_configs=prepare_runtime_configs(args)
        jobs=[dict(cases=list(CASES),seed=seed,gpu=gpu,rho=.15,beta=.075) for gpu,seed in zip(args.gpus,args.seeds)]
        shared.dump(args.out/'round_k_jobs.json',jobs)
        events=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(worker,job['gpu'],job['seed'],args):job for job in jobs}
            for future in as_completed(futures):
                try:event=future.result()
                except Exception as error:event=dict(futures[future],status='failed',error=str(error))
                events.append(event);shared.dump(args.out/'round_k_status.json',events)
                print('JOB_RESULT',event['seed'],event['status'],flush=True)
        if any(r['status']=='failed' for r in events):
            raise SystemExit('ROUND_K_FAILED: see round_k_status.json and logs')
        summarize(args)
        print('ROUND_K_COMPLETE',args.out,'training_jobs 6 mean_gate_jobs 12',flush=True)


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


if __name__=='__main__':main()


