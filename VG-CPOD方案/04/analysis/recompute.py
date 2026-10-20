# Repository paths adapted; archived numerical calculations are unchanged.
"""C-group results, scalar-gain controls, centered-gate statistics, and matched seeds."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

STAGE = Path(__file__).resolve().parent.parent
ROOT = STAGE.parent.parent
OUT = Path(__file__).resolve().parent / 'recomputed'
NEW = STAGE / 'results/raw_export'
BR = STAGE.parent / '03/results/raw_export/roundB_20261019'
AR = STAGE.parent / '02/results/raw_export/roundA_20261019'
OLD = STAGE.parent / 'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs'
C_MAP = json.loads((STAGE / 'package_members.json').read_text(encoding='utf-8'))['members']
B_STAGE = STAGE.parent / '03'
B_MAP = json.loads((B_STAGE / 'package_members.json').read_text(encoding='utf-8'))['members']
ITEMS = [('A5',3351),('A5',3352),('A6',3351),('A6',3352),
         ('B3',3351),('B3',3352),('C1',3351),('C2',3351)]
SETTINGS = ['val_cw','val_ow','test_cw','test_ow']
torch.set_num_threads(1)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def write(p,data):Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_write(name,rows):
    if not rows:return
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def folder(case,seed):
    base=AR if case.startswith('A') else (BR if case=='B3' and seed==3351 else NEW/('roundB_20261019' if case=='B3' else 'roundC_20261020'))
    return next((base/case).glob(f'ut-zap50k/V*/seed{seed}'))


def main():
    (OUT/'tables').mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    helper=module('prediction_helpers',STAGE.parent/'v1/code/analysis_20261018/analyze_results.py')
    evaluator=module('c_evaluator',STAGE/C_MAP['vgcpod/metrics.py'])
    cache=OLD/'cache_ut_clip_l14';manifest=read(cache/'features_manifest.json');labels=manifest['labels']
    train=set(map(tuple,manifest['pairs']['train']));cache_sha=sha(cache/'features_manifest.json')
    audit=dict(archive_sha256=sha(STAGE/'results/archives/vgcpod_roundC_results_20261020.tar.gz'),
               feature_manifest_sha256=cache_sha,models=len(ITEMS),prediction_files=0,source_files_matched=0,errors=[])
    assert (NEW/'roundC_20261020/launcher.exitcode').read_text().strip()=='0'
    status=read(NEW/'roundC_20261020/round_c_status.json')
    assert {(r['case'],r['seed']) for r in status}=={('C1',3351),('C2',3351),('B3',3352)}
    assert all(r['status'] in ('complete','already_complete') and r['exit_code']==0 for r in status)
    assert sha(folder('B3',3351)/'metrics.json')==sha(BR/'B3/ut-zap50k/V1/seed3351/metrics.json')
    errors=dict(native_auc=0.,exact_auc=0.,hm=0.,normalization=0.,support=0.,loss_identity=0.)
    metrics,diagnostics,history,gates,geometry,per_pair=[],[],[],[],[],[]
    predictions={}
    for case,seed in ITEMS:
        run=folder(case,seed);m,p,h=read(run/'metrics.json'),read(run/'protocol.json'),read(run/'history.json')
        cfg=p['config'];epoch=int(np.argmax([r['val_cw']['AUC_native'] for r in h]))+1
        assert m['selected_epoch']==epoch and p['seed']==seed and p['feature_manifest_sha256']==cache_sha
        assert sha(run/'checkpoints/best.pt')==m['checkpoint_sha256']
        assert len(h)==10 and h[-1]['step']==3600
        if not case.startswith('A'):
            source=B_STAGE if case=='B3' else STAGE
            member_map=B_MAP if case=='B3' else C_MAP
            assert cfg==read(source/('configs/round_b' if case=='B3' else 'configs/round_c')/(case+'.json'))
            for name,fingerprint in p['source_sha256'].items():
                assert sha(source/member_map[name])==fingerprint
                audit['source_files_matched']+=1
            best,last=helper.checkpoint(run/'checkpoints/best.pt'),helper.checkpoint(run/'checkpoints/last.pt')
            assert best['epoch']==epoch and last['epoch']==10 and last['history']==h
            for group in last['optimizer']['param_groups']:
                assert abs(group['lr']-cfg['min_lr']*(cfg['cp_lr_scale'] if group['name']=='cp' else 1))<1e-15
            if case in ('B3','C2'):
                b=best['model']['Bvis'].double();gram=b@b.T;norm=b.norm(dim=-1)
                cosine=gram/(norm[:,None]*norm[None,:]);off=~torch.eye(len(b),dtype=torch.bool)
                geometry.append(dict(case=case,seed=seed,row_norm_mean=float(norm.mean()),
                    row_cosine_mean=float(cosine[off].mean()),mean_row_norm=float(b.mean(0).norm()),
                    orth_loss=float((gram-torch.eye(len(b))).square().mean())))
        for entry in h:
            t=entry['train'];beta=cfg['task_aux_beta']
            errors['loss_identity']=max(errors['loss_identity'],abs(t['task']-1.5*t['pair']-1.5*beta/(1+beta)*t['support']))
            history.append(dict(case=case,seed=seed,epoch=entry['epoch'],chosen=entry['epoch']==epoch,
                val_cw=entry['val_cw']['AUC_native'],val_ow=entry['val_ow']['AUC_native'],
                val_unseen_attr=entry['val_diagnostics']['gt_object_attr_acc_unseen'],
                delta_rms=t['centered_delta_rms'],clipped_fraction=t['clipped_step_fraction']))
            if not case.startswith('A'):
                gates.append(dict(case=case,seed=seed,epoch=entry['epoch'],chosen=entry['epoch']==epoch,
                    gamma_mean=t['gamma_mean'],total_variance=t['gamma_variance'],
                    image_variance=t['gamma_image_variance'],image_std=float(np.sqrt(t['gamma_image_variance'])),
                    total_std=float(np.sqrt(t['gamma_variance'])),
                    image_fraction=t['gamma_image_variance']/t['gamma_variance'] if t['gamma_variance'] else 0,
                    abs_deviation=t['gamma_abs_deviation'],mode_mean_deviation=t.get('gamma_mode_mean_deviation'),
                    clipped_fraction=t['clipped_step_fraction'],delta_rms=t['centered_delta_rms']))
        for setting in SETTINGS:
            phase,world=setting.split('_');mm=m['metrics'][setting]
            cw=train|set(map(tuple,manifest['pairs'][phase]))
            d=helper.index_predictions(run/'predictions'/(setting+'.npz'),labels,train,cw)
            audit['prediction_files']+=1
            rec=read(cache/(phase+'_records.json'))
            assert d['image_ids'].tolist()==[r['image'] for r in rec]
            assert d['a'].tolist()==[r['attr'] for r in rec] and d['o'].tolist()==[r['object'] for r in rec]
            native=evaluator.native_curve(d['log_probabilities'],d['truth'],d['seen_classes'])
            exact=evaluator.exact_curve(d['log_probabilities'],d['truth'],d['seen_classes'])
            errors['native_auc']=max(errors['native_auc'],abs(native['AUC']-mm['AUC_native']))
            errors['exact_auc']=max(errors['exact_auc'],abs(exact['AUC']-mm['AUC_exact']))
            errors['normalization']=max(errors['normalization'],float(abs(d['probabilities'].sum(-1)-1).max()))
            bias=m['metrics']['val_'+world]['bias_native'];assert mm['fixed_val_bias']==bias
            logits=torch.from_numpy(d['log_probabilities'].copy());logits[:,~torch.from_numpy(d['seen_classes'])]+=bias
            ok=logits.topk(1,dim=1).indices[:,0].numpy()==d['truth'];sm=d['seen_images']
            hm=float(evaluator.hm(ok[sm].mean(),ok[~sm].mean())*100)
            errors['hm']=max(errors['hm'],abs(hm-mm['HM_at_val_bias']))
            metrics.append(dict(case=case,seed=seed,setting=setting,epoch=epoch,
                **{k:v for k,v in mm.items() if isinstance(v,(int,float))}))
            if world=='ow':
                prob=d['probabilities'].astype(np.float64);cond=helper.softmax(d['conditional_logits_gt_object'])
                co=cond.argmax(-1)==d['a'];support=prob[:,d['seen_classes']].sum(-1)
                op=np.stack([prob[:,d['oi']==j].sum(-1) for j in range(len(labels['objs']))],axis=-1)
                errors['support']=max(errors['support'],abs(support[~sm].mean()-mm['support_mass_unseen']))
                diagnostics.append(dict(case=case,seed=seed,phase=phase,attr_seen=float(co[sm].mean()*100),
                    attr_unseen=float(co[~sm].mean()*100),attr_all=float(co.mean()*100),
                    seen_correct=int(co[sm].sum()),unseen_correct=int(co[~sm].sum()),
                    support_unseen=float(support[~sm].mean()*100),object_acc=float((op.argmax(-1)==d['o']).mean()*100),
                    unseen_limit_acc=exact['unseen'][-1],unseen_cond_nll=float(-np.log(cond[np.arange(len(cond)),d['a']])[~sm].mean())))
                predictions[(case,seed,phase)]=dict(conditional=cond,joint=prob,correct=co,seen=sm,
                    pairs=d['true_pairs'],joint_pred=d['pred'],calibrated=ok)
                for pair in sorted(set(d['true_pairs'])):
                    mask=np.array([p==pair for p in d['true_pairs']])
                    per_pair.append(dict(case=case,seed=seed,phase=phase,attribute=pair[0],object=pair[1],
                        seen=pair in train,images=int(mask.sum()),correct=int(co[mask].sum()),accuracy=float(co[mask].mean()*100)))
        print('ANALYZED',case,seed,epoch,flush=True)
    comparisons,similarity,pair_deltas=[],[],[]
    pairs=[('C1','A5',3351),('C1','B3',3351),('C2','A5',3351),('C2','A6',3351),('C2','C1',3351),('C2','B3',3351),('B3','A5',3351),('B3','A5',3352)]
    for high,low,seed in pairs:
        for setting in SETTINGS:
            left=next(r for r in metrics if r['case']==high and r['seed']==seed and r['setting']==setting)
            right=next(r for r in metrics if r['case']==low and r['seed']==seed and r['setting']==setting)
            comparisons.append(dict(comparison=high+'-'+low,seed=seed,setting=setting,
                **{key:left[key]-right[key] for key in ['AUC_native','AUC_exact','HM_at_val_bias','gt_object_attr_acc_unseen','pair_nll_unbiased']}))
        for phase in ('val','test'):
            left,right=predictions[(high,seed,phase)],predictions[(low,seed,phase)]
            for group in ('all','seen','unseen'):
                mask=np.ones_like(left['seen']) if group=='all' else (left['seen'] if group=='seen' else ~left['seen'])
                lc,rc=left['correct'][mask],right['correct'][mask]
                similarity.append(dict(comparison=high+'-'+low,seed=seed,phase=phase,group=group,
                    recovered=int((lc&~rc).sum()),regressed=int((~lc&rc).sum()),net_correct=int(lc.sum())-int(rc.sum()),
                    conditional_top1_agreement=float((left['conditional'][mask].argmax(-1)==right['conditional'][mask].argmax(-1)).mean()*100),
                    joint_top1_agreement=float((left['joint_pred'][mask]==right['joint_pred'][mask]).mean()*100),
                    conditional_total_variation=float((abs(left['conditional'][mask]-right['conditional'][mask]).sum(-1)/2).mean()),
                    joint_total_variation=float((abs(left['joint'][mask]-right['joint'][mask]).sum(-1)/2).mean())))
            for pair in sorted(set(left['pairs'])):
                a=next(r for r in per_pair if r['case']==high and r['seed']==seed and r['phase']==phase and (r['attribute'],r['object'])==pair)
                b=next(r for r in per_pair if r['case']==low and r['seed']==seed and r['phase']==phase and (r['attribute'],r['object'])==pair)
                pair_deltas.append(dict(comparison=high+'-'+low,seed=seed,phase=phase,attribute=pair[0],object=pair[1],
                    seen=pair in train,images=a['images'],left_accuracy=a['accuracy'],right_accuracy=b['accuracy'],correct_delta=a['correct']-b['correct']))
    b3_summary=[]
    for setting in SETTINGS:
        group=[r for r in comparisons if r['comparison']=='B3-A5' and r['setting']==setting]
        row=dict(comparison='B3-A5',setting=setting,seeds=[r['seed'] for r in group])
        for k in ['AUC_native','HM_at_val_bias','gt_object_attr_acc_unseen']:
            v=[r[k] for r in group];row[k+'_mean']=float(np.mean(v));row[k+'_sd']=float(np.std(v,ddof=1))
        b3_summary.append(row)
    assert errors['native_auc']<.03 and errors['exact_auc']<.001 and errors['hm']<1e-8
    assert errors['normalization']<1e-5 and errors['support']<1e-6 and errors['loss_identity']<1e-5
    for row in read(NEW/'roundC_20261020/round_c_summary.json')['runs']:
        for setting in SETTINGS:
            selected=next(r for r in metrics if r['case']==row['case'] and r['seed']==row['seed'] and r['setting']==setting)
            assert row[setting+'_AUC_native']==selected['AUC_native']
    audit['max_recompute_errors']=errors
    for name,data in [('all_metrics',metrics),('diagnostics',diagnostics),('history',history),('gates',gates),
                      ('geometry',geometry),('per_pair',per_pair),('comparisons',comparisons),('similarity',similarity),
                      ('per_pair_deltas',pair_deltas),('b3_two_seed_deltas',b3_summary)]:csv_write(name+'.csv',data)
    result=dict(metrics=metrics,diagnostics=diagnostics,history=history,gates=gates,geometry=geometry,
                comparisons=comparisons,similarity=similarity,pair_deltas=pair_deltas,b3_summary=b3_summary,audit=audit)
    write(OUT/'analysis.json',result);write(OUT/'audit.json',audit)
    plots(result)
    print(json.dumps(dict(audit=audit,b3_summary=b3_summary,
        chosen_gates=[r for r in gates if r['chosen']],test_similarity=[r for r in similarity if r['phase']=='test' and r['group']=='all']),indent=2))


def plots(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    cases=['A5','A6','B3','C1','C2'];colors=['#5d9b91','#8983a8','#8b9ba9','#367fa2','#145b7c']
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,setting,title in zip(axes,['val_cw','test_cw','test_ow'],['VAL CW','TEST CW','TEST OW']):
        values=[next(r['AUC_native'] for r in data['metrics'] if r['case']==c and r['seed']==3351 and r['setting']==setting) for c in cases]
        ax.bar(cases,values,color=colors);ax.set_ylim(0,50 if setting=='val_cw' else 40);ax.set_title(title);ax.set_ylabel('AUC_native (area x 100)')
        for i,v in enumerate(values):ax.text(i,v+.4,f'{v:.2f}',ha='center',fontsize=9)
    fig.suptitle('Seed 3351: static gain and centered local gate');fig.tight_layout();fig.savefig(OUT/'figures/01_results.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for c,color in [('B3','#8b9ba9'),('C1','#367fa2'),('C2','#145b7c')]:
        g=[r for r in data['gates'] if r['case']==c and r['seed']==3351]
        for ax,key,title in zip(axes,['gamma_mean','image_std','abs_deviation'],['Training gate mean','Cross-image gate variation','Mean absolute gate deviation']):
            ax.plot([r['epoch'] for r in g],[r[key] for r in g],label=c,color=color);ax.set_title(title);ax.set_xlabel('Epoch');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'figures/02_gate_behavior.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,setting in zip(axes,['test_cw','test_ow']):
        q=[r for r in data['comparisons'] if r['comparison']=='B3-A5' and r['setting']==setting]
        ax.bar([str(r['seed']) for r in q],[r['AUC_native'] for r in q],color=['#5d9b91','#b44d5c'])
        ax.axhline(0,color='#333',lw=1);ax.set_title(setting.upper()+': B3 - A5');ax.set_ylabel('Paired AUC difference (points)')
    fig.tight_layout();fig.savefig(OUT/'figures/03_b3_repetition.png');plt.close(fig)


if __name__=='__main__':main()
