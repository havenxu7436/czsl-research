# Numerical calculations unchanged; repository paths adapted.
"""Recompute H1 predictions, gate moments and matched references from downloaded files."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parent.parent
OUT=Path(__file__).resolve().parent/'recomputed'
ARCHIVE=STAGE/'results/raw_export'
H=ARCHIVE/'roundH_20261020'
CACHE=STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14'
sys.path.insert(0,str(STAGE/'code/vgcpod_v1_roundH'))


def source_file(package,member):
    number={'vgcpod_v1_roundD':'05','vgcpod_v1_roundE':'06','vgcpod_v1_roundF':'07',
            'vgcpod_v1_roundG':'08','vgcpod_v1_roundH':'09'}[package]
    folder=STAGE.parent/number
    mapping=json.loads((folder/'package_members.json').read_text(encoding='utf-8'))['members']
    return folder/mapping[member]


def result_base(stage):
    number={'D':'05','E':'06','F':'07','G':'08','H':'09'}[stage]
    return STAGE.parent/number/'results/raw_export'/('round'+stage+'_20261020')

from ranking_diagnostics import diagnose,METRICS
from vgcpod.metrics import native_curve,exact_curve,hm

torch.set_num_threads(1)
SEEDS=(3351,3352,3353)
SETTINGS=('val_cw','val_ow','test_cw','test_ow')
KEYS=('AUC_native','AUC_exact','HM_at_val_bias','gt_object_attr_acc_unseen',
      'support_mass_unseen','pair_nll_unbiased','gt_object_attr_nll')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_write(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/'tables'/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def aggregate(rows,group_keys,value_keys):
    result=[]
    for key in sorted({tuple(r[k] for k in group_keys) for r in rows}):
        selected=sorted([r for r in rows if tuple(r[k] for k in group_keys)==key],key=lambda r:r['seed'])
        row=dict(zip(group_keys,key));row.update(n=len(selected),seeds=[r['seed'] for r in selected])
        for metric in value_keys:
            values=[r[metric] for r in selected]
            row[metric+'_mean']=float(np.mean(values))
            row[metric+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        result.append(row)
    return result


def main():
    (OUT/'tables').mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('h_prediction_helpers',STAGE.parent/'v1/code/analysis_20261018/analyze_results.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    sha,checkpoint=helper.sha,helper.checkpoint
    manifest=read(CACHE/'features_manifest.json');labels=manifest['labels'];train=set(map(tuple,manifest['pairs']['train']))
    cache_sha=sha(CACHE/'features_manifest.json')
    assert (H/'launcher.exitcode').read_text().strip()=='0'
    status=read(H/'round_h_status.json')
    assert {(r['gpu'],r['seed']) for r in status}=={('1',3351),('2',3352),('3',3353)}
    assert all(r['status']=='complete' and r['training']['exit_code']==0 and len(r['mean_gate'])==2
               and all(e['exit_code']==0 for e in r['mean_gate']) for r in status)
    audit=dict(archive_sha256=sha(STAGE/'results/archives/vgcpod_roundH_results_20261020.tar.gz'),cache_sha256=cache_sha,
        training_jobs=3,mean_gate_jobs=6,prediction_files=0,source_files_verified=0,release_files_verified=0,
        exact_prediction_duplicates=0,dynamic_prediction_duplicates=0,ranking_rows_verified=0)
    errors=dict(native_auc=0.,exact_auc=0.,fixed_bias_hm=0.,probability_sum=0.,conditional_acc=0.,
        support_mass=0.,loss_identity=0.,ranking=0.,checkpoint_mean=0.,saved_curve_auc=0.)
    for stage in ('G','H'):
        source=ARCHIVE/('vgcpod_v1_round'+stage)
        for name,fingerprint in read(source_file(source.name,'release_manifest.json'))['files_sha256'].items():
            assert sha(source_file(source.name,name))==fingerprint,name
            audit['release_files_verified']+=1
    stats=H/'feature_statistics/train_object_query_mean.npz';sidecar=read(stats.with_suffix('.json'))
    with np.load(stats,allow_pickle=False) as saved:
        identity=json.loads(str(saved['identity_json'].item()));mu=saved['mean_fo'].copy();mu64=saved['mean_fo_float64'].copy()
    assert identity['feature_manifest_sha256']==cache_sha and identity['train_images']==22998
    assert identity['object_names']==labels['objs'] and identity['train_fo_sha256']==manifest['files_sha256']['train/fo.npy']
    assert sidecar['npz_sha256']==sha(stats) and np.array_equal(mu,mu64.astype(np.float32))
    cfg_h=read(H/'configs/H1.json');cfg_g=read(source_file('vgcpod_v1_roundG','configs/round_g/G1.json'))
    assert cfg_h['gate_statistics_sha256']==sha(stats)
    changed={k for k in cfg_g if cfg_g[k]!=cfg_h[k]}
    assert changed=={'experiment_id','gate_input'}
    audit['config_changes']=sorted(changed|set(cfg_h).difference(cfg_g))
    rows,histories,gates,moments,diagnostics,pairs,parameters,selections=[],[],[],[],[],[],[],[]
    indexed={}

    def analyze_predictions(case,seed,path,metrics,epoch,selection='native'):
        for setting in SETTINGS:
            phase,world=setting.split('_');mm=metrics[setting]
            data=helper.index_predictions(path/'predictions'/(setting+'.npz'),labels,train,train|set(map(tuple,manifest['pairs'][phase])))
            records=read(CACHE/(phase+'_records.json'))
            assert data['image_ids'].tolist()==[r['image'] for r in records]
            assert data['a'].tolist()==[r['attr'] for r in records] and data['o'].tolist()==[r['object'] for r in records]
            native=native_curve(data['log_probabilities'],data['truth'],data['seen_classes'])
            exact=exact_curve(data['log_probabilities'],data['truth'],data['seen_classes'])
            curves=read(path/'curves'/(setting+'.json'))
            for curve_name in ('native','exact'):
                curve=curves[curve_name]
                area=float(getattr(np,'trapezoid',np.trapz)(np.asarray(curve['seen'])/100,np.asarray(curve['unseen'])/100)*100)
                errors['saved_curve_auc']=max(errors['saved_curve_auc'],abs(area-mm['AUC_'+curve_name]))
                assert curve['AUC']==mm['AUC_'+curve_name]
            errors['native_auc']=max(errors['native_auc'],abs(native['AUC']-mm['AUC_native']))
            errors['exact_auc']=max(errors['exact_auc'],abs(exact['AUC']-mm['AUC_exact']))
            errors['probability_sum']=max(errors['probability_sum'],float(abs(data['probabilities'].sum(-1)-1).max()))
            bias=metrics['val_'+world]['bias_native'];assert mm['fixed_val_bias']==bias
            scores=torch.from_numpy(data['log_probabilities'].copy());scores[:,~torch.from_numpy(data['seen_classes'])]+=bias
            calibrated=scores.topk(1,dim=1).indices[:,0].numpy()==data['truth'];sm=data['seen_images']
            fixed_hm=float(hm(calibrated[sm].mean(),calibrated[~sm].mean())*100)
            errors['fixed_bias_hm']=max(errors['fixed_bias_hm'],abs(fixed_hm-mm['HM_at_val_bias']))
            audit['prediction_files']+=1
            rows.append(dict(case=case,seed=seed,selection=selection,setting=setting,selected_epoch=epoch,
                **{k:v for k,v in mm.items() if isinstance(v,(float,int))}))
            if world=='ow':
                cond=helper.softmax(data['conditional_logits_gt_object']);correct=cond.argmax(-1)==data['a']
                support=data['probabilities'][:,data['seen_classes']].astype(np.float64).sum(-1)
                errors['support_mass']=max(errors['support_mass'],abs(support[~sm].mean()-mm['support_mass_unseen']))
                errors['conditional_acc']=max(errors['conditional_acc'],abs(correct[~sm].mean()*100-mm['gt_object_attr_acc_unseen']))
                diag,per_pair=diagnose(data,manifest,phase)
                diagnostics.extend(dict(case=case,seed=seed,selection=selection,phase=phase,**r) for r in diag)
                pairs.extend(dict(case=case,seed=seed,selection=selection,phase=phase,**r) for r in per_pair)
                indexed[(case,seed,phase)]=dict(correct=correct,conditional=cond,seen=sm,
                    logits=data['conditional_logits_gt_object'].astype(np.float64),pairs=data['true_pairs'],
                    calibrated=calibrated,joint_pred=data['pred'])

    for case in ('D1','E1','F1','G0','G1','H1'):
        stage=case[0];base=result_base(stage);source=ARCHIVE/('vgcpod_v1_round'+stage)
        for seed in SEEDS:
            run=base/case/'ut-zap50k/V1'/('seed%d'%seed)
            p,m,history=read(run/'protocol.json'),read(run/'metrics.json'),read(run/'history.json')
            cfg=p['config'];epoch=int(np.argmax([r['val_cw']['AUC_native'] for r in history]))+1
            assert len(history)==10 and history[-1]['step']==3600 and p['feature_manifest_sha256']==cache_sha
            assert m['selected_epoch']==epoch and sha(run/'checkpoints/best.pt')==m['checkpoint_sha256']
            for name,fingerprint in p['source_sha256'].items():
                assert sha(source_file(source.name,name))==fingerprint,name
                audit['source_files_verified']+=1
            analyze_predictions(case,seed,run,m['metrics'],epoch)
            for item in history:
                t=item['train'];beta=cfg['task_aux_beta']
                errors['loss_identity']=max(errors['loss_identity'],abs(t['task']-1.5*t['pair']-1.5*beta/(1+beta)*t['support']))
                histories.append(dict(case=case,seed=seed,epoch=item['epoch'],selected=item['epoch']==epoch,
                    val_cw=item['val_cw']['AUC_native'],val_cw_exact=item['val_cw']['AUC_exact'],val_ow=item['val_ow']['AUC_native'],
                    task=t['task'],pair=t['pair'],support=t['support'],orth=t['orth'],delta_rms=t['centered_delta_rms'],
                    clip_fraction=t['clipped_step_fraction']))
                if case in ('G1','H1'):
                    gates.append(dict(case=case,seed=seed,epoch=item['epoch'],selected=item['epoch']==epoch,
                        variance=t['gamma_variance'],image_variance=t['gamma_image_variance'],
                        mode_mean_error=t['gamma_mode_mean_deviation']))
            if case in ('G0','G1','H1'):
                native=checkpoint(run/'checkpoints/best_native.pt');exact=checkpoint(run/'checkpoints/best_exact.pt')
                last=checkpoint(run/'checkpoints/last.pt')
                assert sha(run/'checkpoints/best.pt')==sha(run/'checkpoints/best_native.pt')
                assert native['epoch']==epoch and last['history']==history
                exact_epoch=int(np.argmax([r['val_cw']['AUC_exact'] for r in history]))+1
                assert exact['epoch']==exact_epoch
                same=all(torch.equal(value,exact['model'][key]) for key,value in native['model'].items())
                selections.append(dict(case=case,seed=seed,native=epoch,exact=exact_epoch,same_parameters=same))
                saved=read(run/'exact_selection/metrics.json')
                assert saved['checkpoint_sha256']==sha(run/'checkpoints/best_exact.pt')
                analyze_predictions(case+'_exact',seed,run/'exact_selection',saved['metrics'],exact_epoch,'exact')
                for setting in SETTINGS:
                    with np.load(run/'predictions'/(setting+'.npz')) as a,np.load(run/'exact_selection/predictions'/(setting+'.npz')) as b:
                        identical=a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
                        assert identical==same
                        audit['exact_prediction_duplicates']+=int(identical)
                state=native['model']
                parameters.append(dict(case=case,seed=seed,epoch=epoch,
                    Bvis_row_norm_mean=float(state['Bvis'].norm(dim=-1).mean()),
                    Wc_norm=float(state['Wc'].norm()),Au_norm=float(state['Au'].norm()),Av_norm=float(state['Av'].norm())))
                if case=='H1':
                    assert p['config']==cfg_h
                    for ck in (native,exact,last):
                        errors['checkpoint_mean']=max(errors['checkpoint_mean'],float(abs(ck['model']['gate_fo_mean'].numpy()-mu).max()))
                        assert ck['config']['gate_statistics_sha256']==sha(stats)
            print('RUN',case,seed,flush=True)

    for case in ('G1','H1'):
        base=result_base(case[0]);source=ARCHIVE/('vgcpod_v1_round'+case[0])
        for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')]:
            suffix='' if selection=='native' else '_exact'
            for seed in SEEDS:
                target=base/folder/case/('seed%d'%seed);saved=read(target/'metrics.json')
                run=base/case/'ut-zap50k/V1'/('seed%d'%seed)
                ref=run if selection=='native' else run/'exact_selection'
                assert saved['train_images']==22998 and saved['optimizer_steps']==0
                assert max(abs(v) for v in saved['dynamic_minus_saved_auc'].values())==0
                assert saved['identity']['checkpoint_sha256']==sha(run/'checkpoints'/('best.pt' if selection=='native' else 'best_exact.pt'))
                assert saved['identity']['evaluator_sha256']==sha(source_file(source.name,'mean_gate_eval.py'))
                with np.load(target/'train_mean_gamma.npz',allow_pickle=False) as data:
                    mean,var=data['mean_gamma'],data['image_variance']
                    assert mean.shape==(12,32) and data['object_names'].tolist()==labels['objs']
                    moments.append(dict(case=case+suffix,seed=seed,selection=selection,
                        image_variance=float(var.mean()),image_std=float(np.sqrt(var.mean())),static_variance=float(mean.var()),
                        image_fraction=float(var.mean()/(var.mean()+mean.var())),
                        mean_abs_deviation=float(abs(mean-1).mean()),mode_mean_error=float(abs(mean.mean(-1)-1).max())))
                for setting in SETTINGS:
                    with np.load(target/'dynamic/predictions'/(setting+'.npz')) as a,np.load(ref/'predictions'/(setting+'.npz')) as b:
                        assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
                        audit['dynamic_prediction_duplicates']+=1
                analyze_predictions(case+suffix+'_mean',seed,target/'train_mean',saved['metrics']['train_mean'],saved['selected_epoch'],selection)

    metrics={(r['case'],r['seed'],r['setting']):r for r in rows}
    comparisons,similarity,logit_effect=[],[],[]
    contrasts=[('H1',r) for r in ('G0','G1','D1','E1','F1','H1_mean')]
    contrasts += [('H1_exact',r) for r in ('G0_exact','G1_exact','H1_exact_mean')]
    contrasts += [('H1_mean','G1_mean'),('H1_mean','G0'),('G1','G1_mean')]
    for high,low in contrasts:
        for seed in SEEDS:
            for setting in SETTINGS:
                a,b=metrics[(high,seed,setting)],metrics[(low,seed,setting)]
                comparisons.append(dict(comparison=high+'-'+low,seed=seed,setting=setting,**{k:a[k]-b[k] for k in KEYS}))
            for phase in ('val','test'):
                a,b=indexed[(high,seed,phase)],indexed[(low,seed,phase)]
                assert a['pairs']==b['pairs'] and np.array_equal(a['seen'],b['seen'])
                for group in ('all','seen','unseen'):
                    mask=np.ones_like(a['seen']) if group=='all' else a['seen'] if group=='seen' else ~a['seen']
                    ac,bc=a['correct'][mask],b['correct'][mask]
                    similarity.append(dict(comparison=high+'-'+low,seed=seed,phase=phase,group=group,images=int(mask.sum()),
                        recovered=int((ac&~bc).sum()),regressed=int((~ac&bc).sum()),net_correct=int(ac.sum())-int(bc.sum()),
                        conditional_agreement=float((a['conditional'][mask].argmax(-1)==b['conditional'][mask].argmax(-1)).mean()*100)))
                    if low==high+'_mean':
                        delta=(a['logits']-b['logits'])[mask];delta=delta-delta.mean(-1,keepdims=True)
                        logit_effect.append(dict(case=high,seed=seed,phase=phase,group=group,
                            centered_conditional_logit_rms=float(np.sqrt(np.mean(delta**2)))))
    diag_index={(r['case'],r['seed'],r['phase'],r['group']):r for r in diagnostics}
    for r in read(H/'ranking_diagnostics.json')['runs']:
        case=r['case']+('_exact' if r['selection']=='exact' else '')+('_mean' if r['condition']=='train_mean' else '')
        if (case,r['seed'],r['phase'],r['group']) not in diag_index:continue
        other=diag_index[(case,r['seed'],r['phase'],r['group'])]
        errors['ranking']=max(errors['ranking'],max(abs(r[k]-other[k]) for k in METRICS))
        audit['ranking_rows_verified']+=1
    server=read(H/'round_h_summary.json')
    for r in server['runs']:
        for setting in SETTINGS:
            for key in KEYS[:5]:
                if setting+'_'+key in r:assert abs(r[setting+'_'+key]-metrics[('H1',r['seed'],setting)][key])<1e-12
    print('RECOMPUTE_ERRORS',errors,flush=True)
    assert errors['saved_curve_auc']<1e-8 and errors['exact_auc']<.001 and errors['fixed_bias_hm']<1e-8
    assert errors['conditional_acc']<1e-8 and errors['support_mass']<1e-6 and errors['loss_identity']<1e-5
    assert errors['ranking']<1e-12 and errors['checkpoint_mean']==0 and errors['probability_sum']<1e-5
    geometry=[dict(object=label,mean_norm=float(norm),residual_rms_unit_approx=float(np.sqrt(1-norm**2)),
                   representative_normalization_factor=float(1/np.sqrt(1-norm**2)))
              for label,norm in zip(labels['objs'],np.linalg.norm(mu64,axis=-1))]
    audit['max_recompute_errors']=errors
    audit['mean_statistics']=dict(shape=list(mu.shape),train_images=22998,sha256=sha(stats),
        identity_matches_cache=True,checkpoint_buffer_identical=True,
        raw_train_features_locally_available=(CACHE/'train/fo.npy').exists())
    data=dict(metrics=rows,means=aggregate(rows,('case','setting'),KEYS),comparisons=comparisons,
        paired_means=aggregate(comparisons,('comparison','setting'),KEYS),history=histories,
        gates=gates,moments=moments,diagnostics=diagnostics,
        diagnostic_means=aggregate(diagnostics,('case','phase','group'),METRICS),
        similarity=similarity,per_pair=pairs,parameters=parameters,selections=selections,
        logit_effect=logit_effect,query_geometry=geometry,audit=audit)
    for name,values in data.items():
        if isinstance(values,list):csv_write(name,values)
    write(OUT/'analysis.json',data);write(OUT/'audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
