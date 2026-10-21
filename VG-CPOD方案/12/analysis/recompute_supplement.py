"""Read-only error overlap, candidate-restricted diagnosis and update scales."""
import json
import sys
from pathlib import Path
import numpy as np

STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parent.parent
OUT=Path(__file__).resolve().parent/'recomputed'
sys.path.insert(0,str(STAGE/'code/vgcpod_v1_roundK'))
from readout_diagnostics import summarize_readout_optimizer
from vgcpod.common import read_json,sha
from recompute_metrics import csv_write,aggregate

CASES=('D1','G0','I2','J1','K1','K2')
SEEDS=(3351,3352,3353)

def run_dir(case,seed):
    stages={'D1':('05','D','20261020'),'G0':('08','G','20261020'),
            'I2':('10','I','20261021'),'J1':('11','J','20261021'),
            'K1':('12','K','20261021'),'K2':('12','K','20261021')}
    number,letter,date=stages[case]
    return STAGE.parent/number/'results/raw_export'/('round%s_%s'%(letter,date))/case/'ut-zap50k/V1'/('seed%d'%seed)


def load(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}

def main():
    manifest=read_json(STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14/features_manifest.json')
    labels=manifest['labels'];train=set(map(tuple,manifest['pairs']['train']))
    rows=[];overlap=[];shifts=[];confusions=[];steps=[];common=[]
    for seed in SEEDS:
        for phase in ('val','test'):
            arrays={};views={}
            for case in CASES:
                run=run_dir(case,seed);data=load(run/'predictions'/(phase+'_ow.npz'));arrays[case]=data
                attrs,objs=data['attribute_truth'],data['object_truth'];pairs=list(map(tuple,data['candidate_pairs']))
                ai=np.array([labels['attrs'].index(a) for a,o in pairs]);oi=np.array([labels['objs'].index(o) for a,o in pairs])
                seen_candidates=np.array([p in train for p in pairs]);valid_candidates=np.array([p in train|set(map(tuple,manifest['pairs'][phase])) for p in pairs])
                pairindex={p:i for i,p in enumerate(pairs)}
                truth=np.array([pairindex[(labels['attrs'][a],labels['objs'][o])] for a,o in zip(attrs,objs)])
                seen=seen_candidates[truth];unseen=~seen;N=int(unseen.sum())
                scores=data['log_probabilities'].astype(np.float64);cond=data['conditional_logits_gt_object'].astype(np.float64)
                cond_pred=cond.argmax(-1);cond_ok=cond_pred==attrs
                bias=read_json(run/'metrics.json')['metrics'][phase+'_ow']['fixed_val_bias']
                adjusted=scores+bias*(~seen_candidates)[None,:]
                pred_cal=adjusted.argmax(-1);cal_ok=pred_cal==truth
                gt_object=oi[None,:]==objs[:,None]
                pred_gt_all=np.where(gt_object,scores,-np.inf).argmax(-1)
                pred_gt_cal=np.where(gt_object,adjusted,-np.inf).argmax(-1)
                # All attrs, unrestricted by seen status: should match saved conditional logits.
                assert np.array_equal(ai[pred_gt_all],cond_pred)
                pred_unseen=np.where(~seen_candidates,scores,-np.inf).argmax(-1)
                unseen_ok=pred_unseen==truth
                for world,worldmask in [('ow',np.ones(len(pairs),dtype=bool)),('cw',valid_candidates)]:
                    pool=(~seen_candidates)&worldmask
                    pred_pool=np.where(pool,scores,-np.inf).argmax(-1)
                    pred_gt_pool=np.where(gt_object&pool[None,:],scores,-np.inf).argmax(-1)
                    rank_ok=pred_pool==truth;oracle_ok=pred_gt_pool==truth
                    assert not ((rank_ok&~oracle_ok)&unseen).any()
                    row=dict(case=case,seed=seed,phase=phase,world=world,unseen_images=N,
                        unseen_pool_accuracy=100*float(rank_ok[unseen].mean()),
                        gt_object_unseen_pool_accuracy=100*float(oracle_ok[unseen].mean()),
                        within_object_ranking_errors=int((~oracle_ok&unseen).sum()),
                        cross_object_competition_errors=int((oracle_ok&~rank_ok&unseen).sum()),
                        unseen_pool_errors=int((~rank_ok&unseen).sum()),
                        gt_object_all_attr_accuracy=100*float(cond_ok[unseen].mean()),
                        gt_object_val_bias_attr_accuracy=100*float((pred_gt_cal[unseen]==truth[unseen]).mean()),
                        gt_object_all_attr_prediction_seen_pct=100*float(seen_candidates[pred_gt_all[unseen]].mean()),
                        full_val_bias_unseen_accuracy=100*float(cal_ok[unseen].mean()))
                    assert row['within_object_ranking_errors']+row['cross_object_competition_errors']==row['unseen_pool_errors']
                    rows.append(row)
                views[case]=dict(cond_ok=cond_ok,cond_pred=cond_pred,unseen_ok=unseen_ok,pred_unseen=pred_unseen,unseen=unseen)
                if phase=='test':
                    for a in range(len(labels['attrs'])):
                        for p in range(len(labels['attrs'])):
                            count=int(((attrs==a)&(cond_pred==p)&unseen).sum())
                            if count:confusions.append(dict(case=case,seed=seed,truth=labels['attrs'][a],prediction=labels['attrs'][p],images=count))
            for case,base in [('J1','G0'),('K1','G0'),('K2','G0'),('K1','J1'),('K2','J1'),('K1','K2')]:
                a,b=views[case],views[base];mask=a['unseen']
                assert np.array_equal(arrays[case]['image_ids'],arrays[base]['image_ids'])
                for metric in ('cond','unseen'):
                    left,right=a[metric+'_ok'],b[metric+'_ok']
                    agreement=a['cond_pred' if metric=='cond' else 'pred_unseen']==b['cond_pred' if metric=='cond' else 'pred_unseen']
                    recovered=int((left&~right&mask).sum());regressed=int((~left&right&mask).sum());both_wrong=int((~left&~right&mask).sum())
                    overlap.append(dict(comparison=case+'-'+base,seed=seed,phase=phase,metric=metric,
                        recovered=recovered,regressed=regressed,net_correct=recovered-regressed,both_wrong=both_wrong,
                        baseline_wrong=int((~right&mask).sum()),baseline_errors_retained_pct=100*both_wrong/int((~right&mask).sum()),
                        prediction_agreement_pct=100*float(agreement[mask].mean())))
            mask=views['G0']['unseen'];common_cases=('G0','I2','J1','K1','K2')
            for metric in ('cond','unseen'):
                all_wrong=mask.copy()
                for case in common_cases:all_wrong&=~views[case][metric+'_ok']
                common.append(dict(seed=seed,phase=phase,metric=metric,cases=list(common_cases),
                    all_wrong=int(all_wrong.sum()),unseen_images=int(mask.sum()),
                    pct_all_unseen=100*float(all_wrong.sum()/mask.sum()),
                    pct_g0_errors=100*float(all_wrong.sum()/(~views['G0'][metric+'_ok']&mask).sum())))
            for case in ('I2','J1','K1','K2'):
                stage=case[0];base=run_dir(case,seed).parents[3]
                mean=load(base/'mean_gate'/case/('seed%d'%seed)/'train_mean/predictions'/(phase+'_ow.npz'))
                a=arrays[case]['conditional_logits_gt_object'].astype(np.float64)
                b=mean['conditional_logits_gt_object'].astype(np.float64)
                delta=a-b;delta-=delta.mean(-1,keepdims=True)
                shifts.append(dict(case=case,seed=seed,phase=phase,
                    dynamic_minus_mean_centered_rms_unseen=float(np.sqrt(np.mean(delta[mask]**2)))))
        for case in ('J1','K1','K2'):
            run=run_dir(case,seed)
            steps.extend(summarize_readout_optimizer(dict(case=case,seed=seed,run_dir=str(run))))
    numeric=[k for k,v in rows[0].items() if isinstance(v,(float,int)) and k not in ('seed','unseen_images')]
    data=dict(candidate_diagnosis=rows,candidate_means=aggregate(rows,('case','phase','world'),numeric),
        overlap=overlap,common_errors=common,confusions=confusions,dynamic_scales=shifts,optimizer=steps)
    for name,values in data.items():csv_write(name,values)
    (OUT/'supplement.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for r in data['candidate_means']:
        if r['phase']=='test' and r['world']=='ow':
            print(r['case'],{k:round(r[k],3) for k in ('unseen_pool_accuracy_mean','gt_object_unseen_pool_accuracy_mean','within_object_ranking_errors_mean','cross_object_competition_errors_mean','gt_object_val_bias_attr_accuracy_mean','gt_object_all_attr_prediction_seen_pct_mean')})
    print('COMMON',json.dumps([r for r in common if r['phase']=='test']))


if __name__=='__main__':main()
