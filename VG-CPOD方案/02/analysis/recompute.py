# Repository paths adapted from the original analysis; numerical calculations unchanged.
"""Three-seed audit of A1/A5/A6 and seed-matched original D0/V1."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

STAGE=Path(__file__).resolve().parent.parent
ROOT=STAGE.parent.parent
OUT=Path(__file__).resolve().parent/'recomputed'
RUNS=STAGE/'results/raw_export/roundA_20261019'
OLD=STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs'
PKG=STAGE/'code/vgcpod_v1_roundA'
SEEDS=[3351,3352,3353]
CASES=['D0','V1','A1','A5','A6']
SETTINGS=['val_cw','val_ow','test_cw','test_ow']
PAIRS=[('A1','V1'),('A5','V1'),('A1','D0'),('A5','D0'),('A5','A1'),('A5','A6'),('A6','D0')]
torch.set_num_threads(1)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def write_json(p,data):Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(name,rows):
    if not rows:return
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def main():
    (OUT/'tables').mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    helper=load_module('first_round_helpers',STAGE.parent/'v1/code/analysis_20261018/analyze_results.py')
    evaluator=load_module('confirmed_evaluator',PKG/'vgcpod/metrics.py')
    manifest=read(OLD/'cache_ut_clip_l14/features_manifest.json');labels=manifest['labels']
    train=set(map(tuple,manifest['pairs']['train']));cache_sha=sha(OLD/'cache_ut_clip_l14/features_manifest.json')
    audit=dict(archive_sha256=sha(STAGE/'results/archives/vgcpod_roundA_confirm_20261019.tar.gz'),archive_runs=12,
               core_cases=3,seeds=SEEDS,newly_completed_runs=6,predictions_recomputed=0,
               feature_manifest_sha256=cache_sha,unchanged_seed3351_files=0,errors=[])
    assert (RUNS/'launcher.exitcode').read_text().strip()=='0'
    status=read(RUNS/'round_a_status.json')
    assert len(status)==6 and {(r['case'],r['seed']) for r in status}=={(c,s) for c in ['A1','A5','A6'] for s in [3352,3353]}
    assert all(r['status']=='complete' and r['exit_code']==0 for r in status)
    archive_metrics=list(RUNS.glob('A*/ut-zap50k/V*/seed*/metrics.json'));assert len(archive_metrics)==12
    # Recheck the original 78 seed3351 hashes against the original first-run inventory.
    first_hashes=read(STAGE/'results/initial_seed3351_hashes.json')
    for rel, fingerprint in first_hashes.items():
        assert sha(RUNS/rel)==fingerprint,rel
        audit['unchanged_seed3351_files']+=1
    errors=dict(accuracy=0.,hm=0.,native_auc=0.,exact_auc=0.,support_mass=0.,normalization=0.,loss_identity=0.)
    rows,diag_rows,pair_rows,hist_rows=[],[],[],[]
    for c in CASES:
        for seed in SEEDS:
            folder=(OLD/'main/ut-zap50k'/('V0' if c=='D0' else 'V1')/f'seed{seed}'
                    if c in ['D0','V1'] else next((RUNS/c).glob(f'ut-zap50k/V*/seed{seed}')))
            m,p,h=read(folder/'metrics.json'),read(folder/'protocol.json'),read(folder/'history.json')
            cfg=p['config'];selected=int(np.argmax([r['val_cw']['AUC_native'] for r in h]))+1
            assert p['seed']==seed and p['feature_manifest_sha256']==cache_sha
            assert m['selected_epoch']==selected and sha(folder/'checkpoints/best.pt')==m['checkpoint_sha256']
            assert [r['epoch'] for r in h]==list(range(1,11)) and h[-1]['step']==3600
            if c.startswith('A'):
                assert cfg==read(STAGE/'configs/round_a'/(c+'.json'))
                for name,fingerprint in p['source_sha256'].items():assert sha(STAGE/read(STAGE/'package_members.json')['members'][name])==fingerprint
                best,last=helper.checkpoint(folder/'checkpoints/best.pt'),helper.checkpoint(folder/'checkpoints/last.pt')
                assert best['epoch']==selected and last['epoch']==10 and last['history']==h
                for g in last['optimizer']['param_groups']:
                    scale=cfg['cp_lr_scale'] if g['name']=='cp' else 1.
                    assert abs(g['lr']-scale*cfg['min_lr'])<1e-15
            for r in h:
                t=r['train']
                hist_rows.append(dict(case=c,seed=seed,epoch=r['epoch'],selected=r['epoch']==selected,
                    val_cw=r['val_cw']['AUC_native'],val_ow=r.get('val_ow',{}).get('AUC_native'),
                    support=r.get('val_diagnostics',{}).get('support_mass_unseen'),
                    attr_unseen=r.get('val_diagnostics',{}).get('gt_object_attr_acc_unseen'),
                    delta_rms=t.get('centered_delta_rms'),clip_fraction=t.get('clipped_step_fraction')))
                if c.startswith('A'):
                    beta=cfg['task_aux_beta'];e=abs(t['task']-1.5*t['pair']-1.5*beta/(1+beta)*t['support'])
                    errors['loss_identity']=max(errors['loss_identity'],e)
                    assert e<1e-5
                    if cfg['variant']=='V1':assert abs(r['lr_groups']['cp']/r['lr_groups']['adapter']-cfg['cp_lr_scale'])<1e-12
            for setting in SETTINGS:
                phase,world=setting.split('_');mm=m['metrics'][setting]
                cw=train|set(map(tuple,manifest['pairs'][phase]))
                d=helper.index_predictions(folder/'predictions'/(setting+'.npz'),labels,train,cw)
                audit['predictions_recomputed']+=1
                records=read(OLD/'cache_ut_clip_l14'/(phase+'_records.json'))
                assert d['image_ids'].tolist()==[r['image'] for r in records]
                assert d['a'].tolist()==[r['attr'] for r in records] and d['o'].tolist()==[r['object'] for r in records]
                assert mm['images']==len(d['truth']) and mm['candidates']==len(d['candidate_pairs'])
                lp=d['log_probabilities'];sm=d['seen_images'];y=d['truth']
                n=evaluator.native_curve(lp,y,d['seen_classes']);ex=evaluator.exact_curve(lp,y,d['seen_classes'])
                errors['native_auc']=max(errors['native_auc'],abs(n['AUC']-mm['AUC_native']))
                errors['exact_auc']=max(errors['exact_auc'],abs(ex['AUC']-mm['AUC_exact']))
                errors['accuracy']=max(errors['accuracy'],abs(d['correct'].mean()*100-mm['pair_acc_unbiased']))
                errors['normalization']=max(errors['normalization'],float(abs(d['probabilities'].sum(1)-1).max()))
                bias=m['metrics']['val_'+world]['bias_native'];assert mm['fixed_val_bias']==bias
                logits=torch.from_numpy(lp.copy());logits[:,~torch.from_numpy(d['seen_classes'])]+=bias
                pred=logits.topk(1,dim=1).indices[:,0].numpy();ok=pred==y
                hm=float(evaluator.hm(ok[sm].mean(),ok[~sm].mean())*100)
                errors['hm']=max(errors['hm'],abs(hm-mm['HM_at_val_bias']))
                cond=helper.softmax(d['conditional_logits_gt_object']);ca=cond.argmax(1)==d['a']
                row=dict(case=c,seed=seed,setting=setting,epoch=selected,**{k:v for k,v in mm.items() if isinstance(v,(int,float))})
                # Supplementary diagnosis: use the exact VAL maximum-HM bias on the same TEST weights.
                ebias=m['metrics']['val_'+world]['bias_exact']
                elogits=torch.from_numpy(lp.copy());elogits[:,~torch.from_numpy(d['seen_classes'])]+=ebias
                eok=elogits.topk(1,dim=1).indices[:,0].numpy()==y
                row['HM_at_exact_val_bias']=float(evaluator.hm(eok[sm].mean(),eok[~sm].mean())*100)
                row['oracle_gap_native']=mm['HM_oracle_native']-mm['HM_at_val_bias']
                rows.append(row)
                if world=='ow':
                    probability=d['probabilities'].astype(np.float64)
                    support=probability[:,d['seen_classes']].sum(1)
                    po=np.stack([probability[:,d['oi']==j].sum(1) for j in range(len(labels['objs']))],axis=-1)
                    dr=dict(case=c,seed=seed,phase=phase,attr_seen=float(ca[sm].mean()*100),
                        attr_unseen=float(ca[~sm].mean()*100),attr_all=float(ca.mean()*100),
                        seen_correct=int(ca[sm].sum()),unseen_correct=int(ca[~sm].sum()),
                        support_unseen=float(support[~sm].mean()*100),support_seen=float(support[sm].mean()*100),
                        object_acc=float((po.argmax(1)==d['o']).mean()*100),object_acc_unseen=float((po.argmax(1)[~sm]==d['o'][~sm]).mean()*100),
                        unseen_limit_acc=ex['unseen'][-1],cond_nll_unseen=float(-np.log(cond[np.arange(len(y)),d['a']][~sm]).mean()))
                    diag_rows.append(dr)
                    if c.startswith('A'):
                        errors['support_mass']=max(errors['support_mass'],abs(dr['support_unseen']/100-mm['support_mass_unseen']))
                        assert abs(dr['attr_unseen']-mm['gt_object_attr_acc_unseen'])<1e-8
                if setting=='test_ow':
                    for pair in sorted(set(d['true_pairs'])):
                        mask=np.array([t==pair for t in d['true_pairs']])
                        pair_rows.append(dict(case=c,seed=seed,attribute=pair[0],object=pair[1],seen=pair in train,
                            images=int(mask.sum()),cond_acc=float(ca[mask].mean()*100),pair_acc=float(ok[mask].mean()*100)))
            print('AUDITED',c,seed,flush=True)
    assert errors['accuracy']<1e-8 and errors['hm']<1e-8 and errors['native_auc']<.03 and errors['exact_auc']<.001
    assert errors['normalization']<1e-5 and errors['support_mass']<1e-6
    saved=read(RUNS/'round_a_summary.json')['runs']
    for r in saved:
        if r['case'] in ['A1','A5','A6']:
            for setting in SETTINGS:
                q=next(t for t in rows if t['case']==r['case'] and t['seed']==r['seed'] and t['setting']==setting)
                assert r[setting+'_AUC_native']==q['AUC_native']
    audit.update(max_recompute_errors=errors,saved_summary='matched',source_hashes='matched delivered package',
                 selected_checkpoints='all first strict maximum VAL CW AUC_native')
    keys=['AUC_native','AUC_exact','HM_at_val_bias','gt_object_attr_acc','pair_nll_unbiased','HM_at_exact_val_bias','oracle_gap_native']
    aggregates=[]
    for c in CASES:
        for setting in SETTINGS:
            q=[r for r in rows if r['case']==c and r['setting']==setting];ar=dict(case=c,setting=setting)
            for key in keys:
                v=[r[key] for r in q];ar[key+'_mean']=float(np.mean(v));ar[key+'_sd']=float(np.std(v,ddof=1))
            aggregates.append(ar)
    comparisons=[]
    for hi,lo in PAIRS:
        for setting in SETTINGS:
            for key in keys:
                values=[next(r[key] for r in rows if r['case']==hi and r['seed']==s and r['setting']==setting)
                       -next(r[key] for r in rows if r['case']==lo and r['seed']==s and r['setting']==setting) for s in SEEDS]
                comparisons.append(dict(comparison=hi+'-'+lo,setting=setting,metric=key,
                    **{f'seed{s}':float(v) for s,v in zip(SEEDS,values)},mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),positive=sum(v>0 for v in values)))
    diag_agg=[]
    for c in CASES:
        for phase in ['val','test']:
            q=[r for r in diag_rows if r['case']==c and r['phase']==phase];ar=dict(case=c,phase=phase)
            for key in [k for k in q[0] if k not in ['case','phase','seed']]:
                ar[key+'_mean']=float(np.mean([r[key] for r in q]));ar[key+'_sd']=float(np.std([r[key] for r in q],ddof=1))
            diag_agg.append(ar)
    pair_agg=[]
    for pair in sorted({(r['attribute'],r['object']) for r in pair_rows}):
        q=[r for r in pair_rows if (r['attribute'],r['object'])==pair]
        ar=dict(attribute=pair[0],object=pair[1],seen=q[0]['seen'],images=q[0]['images'])
        for c in CASES:
            ar[c+'_cond_acc']=float(np.mean([r['cond_acc'] for r in q if r['case']==c]))
        for lo in ['D0','V1','A6']:
            ar['A5_minus_'+lo]=ar['A5_cond_acc']-ar[lo+'_cond_acc']
            ar['contribution_vs_'+lo]=ar['images']/2914*ar['A5_minus_'+lo]
        pair_agg.append(ar)
    for name,data in [('all_metrics',rows),('mean_sd',aggregates),('paired_deltas',comparisons),('diagnostics',diag_rows),
                      ('diagnostic_mean_sd',diag_agg),('per_pair',pair_rows),('pair_mean',pair_agg),('history',hist_rows)]:write_csv(name+'.csv',data)
    result=dict(aggregate=aggregates,paired=comparisons,diagnostics=diag_agg,pairs=pair_agg,audit=audit)
    write_json(OUT/'analysis.json',result);write_json(OUT/'audit.json',audit)
    plots(aggregates,comparisons,diag_agg)
    for c in CASES:
        print(c,[(s,next((r['AUC_native_mean'],r['AUC_native_sd']) for r in aggregates if r['case']==c and r['setting']==s)) for s in SETTINGS])
    print(json.dumps(audit,indent=2))


def plots(agg,paired,diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    colors=['#697887','#367fa2','#5d9b91','#145b7c','#827eae']
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,setting,title in zip(axs,['val_cw','test_cw','test_ow'],['VAL CW','TEST CW','TEST OW']):
        q=[next(r for r in agg if r['case']==c and r['setting']==setting) for c in CASES]
        ax.bar(CASES,[r['AUC_native_mean'] for r in q],yerr=[r['AUC_native_sd'] for r in q],color=colors,capsize=4)
        ax.set_title(title);ax.set_ylabel('AUC_native (area x 100)');ax.set_ylim(0,50 if setting=='val_cw' else 40)
    fig.suptitle('Three seeds: mean +/- sample SD');fig.tight_layout();fig.savefig(OUT/'figures/01_main.png');plt.close(fig)
    cc=['A1-V1','A5-V1','A5-D0','A5-A1','A5-A6']
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,setting in zip(axs,['test_cw','test_ow']):
        for i,c in enumerate(cc):
            r=next(r for r in paired if r['comparison']==c and r['setting']==setting and r['metric']=='AUC_native')
            ax.scatter([i-.07,i,i+.07],[r[f'seed{s}'] for s in SEEDS],s=25,color='#6e869a')
            ax.errorbar(i,r['mean'],yerr=r['sd'],fmt='D',color='#b44d5c',capsize=4,markersize=4)
        ax.axhline(0,color='#333',lw=1);ax.set_xticks(range(len(cc)),cc,rotation=20)
        ax.set_title(setting.upper());ax.set_ylabel('Paired AUC difference (points)')
    fig.suptitle('Dots: matched seeds; diamonds: mean +/- SD');fig.tight_layout();fig.savefig(OUT/'figures/02_paired.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for offset,key,label,color in [(-.18,'attr_seen','Seen','#367fa2'),(.18,'attr_unseen','Unseen','#b44d5c')]:
        q=[next(r for r in diag if r['case']==c and r['phase']=='test') for c in CASES]
        axs[0].bar(np.arange(len(CASES))+offset,[r[key+'_mean'] for r in q],yerr=[r[key+'_sd'] for r in q],width=.34,label=label,color=color,capsize=3)
    axs[0].set_xticks(range(len(CASES)),CASES);axs[0].set_title('TEST GT-object attribute accuracy');axs[0].set_ylabel('Accuracy (%)');axs[0].legend()
    q=[next(r for r in diag if r['case']==c and r['phase']=='test') for c in CASES]
    axs[1].bar(CASES,[r['support_unseen_mean'] for r in q],yerr=[r['support_unseen_sd'] for r in q],color=colors,capsize=3)
    axs[1].set_title('TEST unseen images: probability on seen pairs');axs[1].set_ylabel('Probability mass (%)')
    fig.tight_layout();fig.savefig(OUT/'figures/03_transfer.png');plt.close(fig)


if __name__=='__main__':main()
