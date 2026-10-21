"""Audit J1 predictions and checkpoints, then compare with the audited I-group history."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'server_results_roundJ_20261021'
RUNS=ARCHIVE/'roundJ_20261021'
SOURCE=ARCHIVE/'vgcpod_v1_roundJ'
OLD=ROOT/'server_results_roundI_20261021'
CACHE=ROOT/'server_results_20261018/vgcpod_v1_runs/cache_ut_clip_l14'
sys.path.insert(0,str(SOURCE))
from ranking_diagnostics import diagnose,METRICS
from vgcpod.metrics import native_curve,exact_curve,hm
from vgcpod.common import load_torch,sha,read_json as read
torch.set_num_threads(1)
SEEDS=(3351,3352,3353)
SETTINGS=('val_cw','val_ow','test_cw','test_ow')
KEYS=('AUC_native','AUC_exact','HM_at_val_bias','gt_object_attr_acc_unseen',
      'support_mass_unseen','pair_nll_unbiased','gt_object_attr_nll')

def write(name,value):
    (OUT/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def csv_write(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'tables'/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def aggregate(rows,keys,values):
    result=[]
    for group in sorted({tuple(r[k] for k in keys) for r in rows}):
        selected=sorted([r for r in rows if tuple(r[k] for k in keys)==group],key=lambda r:r['seed'])
        row=dict(zip(keys,group));row.update(n=len(selected),seeds=[r['seed'] for r in selected])
        for key in values:
            a=[r[key] for r in selected];row[key+'_mean']=float(np.mean(a));row[key+'_sd']=float(np.std(a,ddof=1)) if len(a)>1 else None
        result.append(row)
    return result

def main():
    (OUT/'tables').mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('prediction_helpers',ROOT/'analysis_20261018/analyze_results.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    old=read(ROOT/'analysis_roundI_results_20261021/analysis.json')
    manifest=read(CACHE/'features_manifest.json');labels=manifest['labels'];train=set(map(tuple,manifest['pairs']['train']))
    cache_sha=sha(CACHE/'features_manifest.json')
    assert sha(ROOT/'vgcpod_roundI_results_20261021.tar.gz')==old['audit']['archive_sha256']
    assert (RUNS/'launcher.exitcode').read_text().strip()=='0'
    status=read(RUNS/'round_j_status.json')
    assert {(r['gpu'],r['seed']) for r in status}=={('1',3351),('2',3352),('3',3353)}
    assert all(r['status']=='complete' and len(r['trainings'])==1 and r['trainings'][0]['exit_code']==0
        and len(r['mean_gate'])==2 and all(t['exit_code']==0 for t in r['mean_gate']) for r in status)
    audit=dict(archive_sha256=sha(ROOT/'vgcpod_roundJ_results_20261021.tar.gz'),cache_sha256=cache_sha,
        historical_audit_reused=True,training_jobs=3,mean_gate_jobs=6,prediction_files=0,
        source_files_verified=0,release_files_verified=0,exact_prediction_duplicates=0,dynamic_prediction_duplicates=0,ranking_rows_verified=0)
    for name,fingerprint in read(SOURCE/'release_manifest.json')['files'].items():
        assert sha(SOURCE/name)==fingerprint,name;audit['release_files_verified']+=1
    stats=RUNS/'feature_statistics/train_object_query_mean.npz'
    assert sha(stats)==sha(OLD/'roundI_20261021/feature_statistics/train_object_query_mean.npz')
    with np.load(stats) as f:
        identity=json.loads(str(f['identity_json'].item()));mu=f['mean_fo'].copy()
    assert identity['train_images']==22998 and identity['feature_manifest_sha256']==cache_sha
    cfg=read(RUNS/'configs/J1.json');icfg=read(OLD/'roundI_20261021/configs/I2.json')
    assert set(cfg)-set(icfg)=={'cp_readout_mode'} and cfg['cp_readout_mode']=='separate'
    assert {k for k in icfg if cfg[k]!=icfg[k]}=={'experiment_id','gate_statistics_path'}
    audit['config_changes']=['experiment_id','gate_statistics_path','cp_readout_mode']
    errors={k:0. for k in ('native_auc','exact_auc','fixed_bias_hm','probability_sum','conditional_acc','support_mass','loss_identity','ranking','checkpoint_mean','saved_curve_auc','rms')}
    rows=list(old['metrics']);diagnostics=list(old['diagnostics']);per_pair=list(old['per_pair'])
    histories=list(old['history']);gates=list(old['gates']);moments=list(old['moments']);selections=list(old['selections'])
    parameters=list(old['parameters']);indexed={};new_diag=[];readouts=[];new_rows=[]

    def prediction(case,seed,path,metrics,epoch,selection):
        for setting in SETTINGS:
            phase,world=setting.split('_');mm=metrics[setting]
            data=helper.index_predictions(path/'predictions'/(setting+'.npz'),labels,train,train|set(map(tuple,manifest['pairs'][phase])))
            records=read(CACHE/(phase+'_records.json'))
            assert data['image_ids'].tolist()==[r['image'] for r in records]
            assert data['a'].tolist()==[r['attr'] for r in records] and data['o'].tolist()==[r['object'] for r in records]
            curves=read(path/'curves'/(setting+'.json'))
            for name in ('native','exact'):
                curve=curves[name];area=float(np.trapezoid(np.asarray(curve['seen'])/100,np.asarray(curve['unseen'])/100)*100)
                errors['saved_curve_auc']=max(errors['saved_curve_auc'],abs(area-mm['AUC_'+name]));assert curve['AUC']==mm['AUC_'+name]
            native=native_curve(data['log_probabilities'],data['truth'],data['seen_classes'])
            exact=exact_curve(data['log_probabilities'],data['truth'],data['seen_classes'])
            for name,value in [('native_auc',native['AUC']-mm['AUC_native']),('exact_auc',exact['AUC']-mm['AUC_exact']),
                ('probability_sum',float(abs(data['probabilities'].sum(-1)-1).max()))]:errors[name]=max(errors[name],abs(value))
            bias=metrics['val_'+world]['bias_native'];assert mm['fixed_val_bias']==bias
            scores=torch.from_numpy(data['log_probabilities'].copy());scores[:,~torch.from_numpy(data['seen_classes'])]+=bias
            correct=scores.topk(1,dim=1).indices[:,0].numpy()==data['truth'];seen=data['seen_images']
            recompute=float(hm(correct[seen].mean(),correct[~seen].mean())*100)
            errors['fixed_bias_hm']=max(errors['fixed_bias_hm'],abs(recompute-mm['HM_at_val_bias']))
            row=dict(case=case,seed=seed,selection=selection,setting=setting,selected_epoch=epoch,**{k:v for k,v in mm.items() if isinstance(v,(float,int))})
            rows.append(row);new_rows.append(row);audit['prediction_files']+=1
            if world=='ow':
                conditional=helper.softmax(data['conditional_logits_gt_object']);ok=conditional.argmax(-1)==data['a']
                support=data['probabilities'][:,data['seen_classes']].astype(np.float64).sum(-1)
                errors['conditional_acc']=max(errors['conditional_acc'],abs(ok[~seen].mean()*100-mm['gt_object_attr_acc_unseen']))
                errors['support_mass']=max(errors['support_mass'],abs(support[~seen].mean()-mm['support_mass_unseen']))
                diag,pp=diagnose(data,manifest,phase)
                added=[dict(case=case,seed=seed,selection=selection,phase=phase,**r) for r in diag]
                diagnostics.extend(added);new_diag.extend(added)
                per_pair.extend(dict(case=case,seed=seed,selection=selection,phase=phase,**r) for r in pp)
                indexed[(case,seed,phase)]=dict(seen=seen,correct=ok,pred=conditional.argmax(-1),logits=data['conditional_logits_gt_object'].astype(np.float64),data=data)

    for seed in SEEDS:
        run=RUNS/'J1/ut-zap50k/V1'/('seed%d'%seed)
        p,m,history=read(run/'protocol.json'),read(run/'metrics.json'),read(run/'history.json')
        assert p['config']==cfg and p['trainable_parameters']==147456 and p['feature_manifest_sha256']==cache_sha
        assert len(history)==10 and history[-1]['step']==3600
        for name,fingerprint in p['source_sha256'].items():
            assert sha(SOURCE/name)==fingerprint,name;audit['source_files_verified']+=1
        native=load_torch(run/'checkpoints/best_native.pt');exact=load_torch(run/'checkpoints/best_exact.pt');last=load_torch(run/'checkpoints/last.pt')
        ne=int(np.argmax([r['val_cw']['AUC_native'] for r in history]))+1;ee=int(np.argmax([r['val_cw']['AUC_exact'] for r in history]))+1
        assert native['epoch']==ne and exact['epoch']==ee and last['history']==history
        assert sha(run/'checkpoints/best.pt')==sha(run/'checkpoints/best_native.pt')==m['checkpoint_sha256']
        same=all(torch.equal(v,exact['model'][k]) for k,v in native['model'].items());selections.append(dict(case='J1',seed=seed,native=ne,exact=ee,same_parameters=same))
        for ck in (native,exact,last):
            errors['checkpoint_mean']=max(errors['checkpoint_mean'],float(abs(ck['model']['gate_fo_mean'].numpy()-mu).max()))
            assert ck['model']['Wd'].shape==(768,32) and ck['config']['gate_statistics_sha256']==sha(stats)
        for stage,ck in [('selected',native),('final',last)]:
            state=ck['model'];parameters.append(dict(case='J1',seed=seed,stage=stage,epoch=ck['epoch'],
                Wc_norm=float(state['Wc'].norm()),Wd_norm=float(state['Wd'].norm()),
                readout_cosine=float(torch.nn.functional.cosine_similarity(state['Wc'].flatten(),state['Wd'].flatten(),dim=0)),
                Bvis_row_norm_mean=float(state['Bvis'].norm(dim=-1).mean()),Au_norm=float(state['Au'].norm()),Av_norm=float(state['Av'].norm())))
        for item in history:
            t=item['train'];errors['loss_identity']=max(errors['loss_identity'],abs(t['task']-1.5*t['pair']-1.5*.075/1.075*t['support']))
            histories.append(dict(case='J1',seed=seed,epoch=item['epoch'],selected=item['epoch']==ne,val_cw=item['val_cw']['AUC_native'],
                val_cw_exact=item['val_cw']['AUC_exact'],val_ow=item['val_ow']['AUC_native'],task=t['task'],pair=t['pair'],support=t['support'],orth=t['orth'],delta_rms=t['centered_delta_rms'],clip_fraction=t['clipped_step_fraction']))
            gates.append(dict(case='J1',seed=seed,epoch=item['epoch'],selected=item['epoch']==ne,variance=t['gamma_variance'],image_variance=t['gamma_image_variance'],mode_mean_error=t['gamma_mode_mean_deviation']))
            r=dict(seed=seed,epoch=item['epoch'],selected=item['epoch']==ne,**{k:v for k,v in t.items() if k.startswith(('cp_static','cp_dynamic','centered_delta'))})
            r['centered_cross_mean']=(t['centered_delta_mse']-t['cp_static_centered_mse']-t['cp_dynamic_centered_mse'])/2
            r['centered_component_cosine']=r['centered_cross_mean']/(t['cp_static_centered_rms']*t['cp_dynamic_centered_rms'])
            readouts.append(r)
            for name in ('cp_static','cp_dynamic','cp_static_centered','cp_dynamic_centered'):
                errors['rms']=max(errors['rms'],abs(t[name+'_rms']**2-t[name+'_mse']))
        for selection,epoch,path,folder in [('native',ne,run,'mean_gate'),('exact',ee,run/'exact_selection','mean_gate_exact')]:
            suffix='' if selection=='native' else '_exact';case='J1'+suffix
            saved=read(path/'metrics.json');assert saved['selected_epoch']==epoch
            prediction(case,seed,path,saved['metrics'],epoch,selection)
            target=RUNS/folder/'J1'/('seed%d'%seed);mean=read(target/'metrics.json')
            assert mean['optimizer_steps']==0 and mean['train_images']==22998 and mean['selected_epoch']==epoch
            assert max(abs(v) for v in mean['dynamic_minus_saved_auc'].values())==0
            assert mean['identity']['checkpoint_sha256']==saved['checkpoint_sha256']
            assert mean['identity']['evaluator_sha256']==sha(SOURCE/'mean_gate_eval.py')
            with np.load(target/'train_mean_gamma.npz') as data:
                avg,var=data['mean_gamma'],data['image_variance']
                assert avg.shape==(12,32) and data['object_names'].tolist()==labels['objs']
                moments.append(dict(case=case,seed=seed,selection=selection,image_variance=float(var.mean()),image_std=float(np.sqrt(var.mean())),
                    static_variance=float(avg.var()),image_fraction=float(var.mean()/(var.mean()+avg.var())),mean_abs_deviation=float(abs(avg-1).mean()),mode_mean_error=float(abs(avg.mean(-1)-1).max())))
            for setting in SETTINGS:
                with np.load(target/'dynamic/predictions'/(setting+'.npz')) as a,np.load(path/'predictions'/(setting+'.npz')) as b:
                    assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files);audit['dynamic_prediction_duplicates']+=1
                if selection=='exact':
                    with np.load(run/'predictions'/(setting+'.npz')) as a,np.load(path/'predictions'/(setting+'.npz')) as b:
                        identical=a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
                        assert identical==same;audit['exact_prediction_duplicates']+=int(identical)
            prediction(case+'_mean',seed,target/'train_mean',mean['metrics']['train_mean'],epoch,selection)
        print('J1_AUDITED',seed,ne,ee,flush=True)

    metric={(r['case'],r['seed'],r['setting']):r for r in rows}
    contrasts=[('J1',c) for c in ('I2','I1','H1','G0','G1','D1','E1','F1','J1_mean')]
    contrasts += [('J1_exact',c) for c in ('I2_exact','I1_exact','G0_exact','G1_exact','H1_exact','J1_exact_mean','J1')]
    contrasts += [('J1_mean',c) for c in ('I2_mean','G0','G1_mean')]
    comparisons=[]
    for high,low in contrasts:
        for seed in SEEDS:
            for setting in SETTINGS:
                a,b=metric[(high,seed,setting)],metric[(low,seed,setting)]
                comparisons.append(dict(comparison=high+'-'+low,seed=seed,setting=setting,**{k:a[k]-b[k] for k in KEYS}))
    di={(r['case'],r['seed'],r['phase'],r['group']):r for r in diagnostics}
    for r in read(RUNS/'ranking_diagnostics.json')['runs']:
        case=r['case']+('_exact' if r['selection']=='exact' else '')+('_mean' if r['condition']=='train_mean' else '')
        other=di.get((case,r['seed'],r['phase'],r['group']))
        if other:
            errors['ranking']=max(errors['ranking'],max(abs(r[k]-other[k]) for k in METRICS));audit['ranking_rows_verified']+=1
    assert errors['saved_curve_auc']<1e-8 and errors['exact_auc']<.001 and errors['fixed_bias_hm']<1e-8
    assert errors['support_mass']<1e-6 and errors['conditional_acc']<1e-8 and errors['loss_identity']<1e-5
    assert errors['rms']<1e-12 and errors['checkpoint_mean']==0 and errors['ranking']<1e-12
    audit['max_recompute_errors']=errors
    audit['mean_statistics']=dict(sha256=sha(stats),train_images=22998,shape=list(mu.shape),identical_to_I=True)
    data=dict(metrics=rows,means=aggregate(rows,('case','setting'),KEYS),comparisons=comparisons,
        paired_means=aggregate(comparisons,('comparison','setting'),KEYS),history=histories,gates=gates,moments=moments,
        diagnostics=diagnostics,diagnostic_means=aggregate(diagnostics,('case','phase','group'),METRICS),
        per_pair=per_pair,parameters=parameters,selections=selections,readout_history=readouts,audit=audit)
    for name,value in data.items():
        if isinstance(value,list):csv_write(name,value)
    write('analysis',data);write('audit',audit)
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
