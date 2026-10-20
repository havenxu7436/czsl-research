"""Recompute E1 rho sweep, dual selection, and dynamic/mean gate exports."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT/'server_results_roundE_20261020'
A_OUT = ROOT/'server_results_roundA_confirm_20261019/roundA_20261019'
C_OUT = ARCHIVE/'roundC_20261020'
D_OUT = ARCHIVE/'roundD_20261020'
E_OUT = ARCHIVE/'roundE_20261020'
CACHE = ROOT/'server_results_20261018/vgcpod_v1_runs/cache_ut_clip_l14'
SETTINGS = ('val_cw','val_ow','test_cw','test_ow')
SEEDS = (3351,3352,3353)
KEYS = ('AUC_native','AUC_exact','HM_at_val_bias','gt_object_attr_acc_unseen',
        'support_mass_unseen','pair_nll_unbiased','gt_object_attr_nll')
torch.set_num_threads(1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_write(name,rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fields)
        writer.writeheader()
        writer.writerows(rows)


def folder(case,seed):
    base = A_OUT if case.startswith('A') else C_OUT if case.startswith('C') else E_OUT if case.startswith('E') else D_OUT
    return next((base/case).glob('ut-zap50k/V*/seed%d'%seed))


def main():
    (OUT/'tables').mkdir(parents=True,exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    helper = load_module('d_prediction_helpers',ROOT/'analysis_20261018/analyze_results.py')
    evaluator = load_module('d_archived_evaluator',ARCHIVE/'vgcpod_v1_roundD/vgcpod/metrics.py')
    manifest = read(CACHE/'features_manifest.json')
    labels,train = manifest['labels'],set(map(tuple,manifest['pairs']['train']))
    cache_sha = helper.sha(CACHE/'features_manifest.json')
    assert (E_OUT/'launcher.exitcode').read_text().strip()=='0'
    events = read(E_OUT/'round_e_status.json')
    assert {(e['case'],e['seed']) for e in events}=={('E1',3351),('E1',3352),('E1',3353)}
    assert all(e['status']=='complete' and e['training']['exit_code']==0 for e in events)
    assert all(len(e['mean_gate'])==2 and all(m['exit_code']==0 for m in e['mean_gate']) for e in events)
    audit = dict(archive_sha256=helper.sha(ROOT/'vgcpod_roundE_results_20261020.tar.gz'),
        feature_manifest_sha256=cache_sha,base_runs=17,interventions=12,prediction_files=0,
        source_files_verified=0,dynamic_duplicate_files_identical=0,exact_selection_duplicate_files=0)
    errors = dict(native_auc=0.,exact_auc=0.,fixed_bias_hm=0.,probability_sum=0.,support_mass=0.,
                  conditional_acc=0.,loss_identity=0.)
    items = [(case,seed) for case in ('A5','A6','C2','D1','E1') for seed in SEEDS]+[('C1',3351),('C1',3352)]
    rows,histories,gates,moments,predictions,diagnostics,per_pair = [],[],[],[],{},[],[]

    def analyze_predictions(case,seed,run,metrics,epoch):
        for setting in SETTINGS:
            phase,world = setting.split('_')
            mm = metrics[setting]
            cw = train|set(map(tuple,manifest['pairs'][phase]))
            d = helper.index_predictions(run/'predictions'/(setting+'.npz'),labels,train,cw)
            audit['prediction_files'] += 1
            records = read(CACHE/(phase+'_records.json'))
            assert d['image_ids'].tolist()==[x['image'] for x in records]
            assert d['a'].tolist()==[x['attr'] for x in records]
            assert d['o'].tolist()==[x['object'] for x in records]
            native = evaluator.native_curve(d['log_probabilities'],d['truth'],d['seen_classes'])
            exact = evaluator.exact_curve(d['log_probabilities'],d['truth'],d['seen_classes'])
            errors['native_auc'] = max(errors['native_auc'],abs(native['AUC']-mm['AUC_native']))
            errors['exact_auc'] = max(errors['exact_auc'],abs(exact['AUC']-mm['AUC_exact']))
            errors['probability_sum'] = max(errors['probability_sum'],float(abs(d['probabilities'].sum(-1)-1).max()))
            bias = metrics['val_'+world]['bias_native']
            assert mm['fixed_val_bias']==bias
            logits = torch.from_numpy(d['log_probabilities'].copy())
            logits[:,~torch.from_numpy(d['seen_classes'])] += bias
            calibrated = logits.topk(1,dim=1).indices[:,0].numpy()==d['truth']
            sm = d['seen_images']
            hm = float(evaluator.hm(calibrated[sm].mean(),calibrated[~sm].mean())*100)
            errors['fixed_bias_hm'] = max(errors['fixed_bias_hm'],abs(hm-mm['HM_at_val_bias']))
            rows.append(dict(case=case,seed=seed,setting=setting,selected_epoch=epoch,
                **{k:v for k,v in mm.items() if isinstance(v,(int,float))}))
            if world=='ow':
                cond = helper.softmax(d['conditional_logits_gt_object'])
                correct = cond.argmax(-1)==d['a']
                prob = d['probabilities'].astype(np.float64)
                support = prob[:,d['seen_classes']].sum(-1)
                obj_prob = np.stack([prob[:,d['oi']==j].sum(-1) for j in range(len(labels['objs']))],axis=-1)
                errors['support_mass'] = max(errors['support_mass'],abs(support[~sm].mean()-mm['support_mass_unseen']))
                errors['conditional_acc'] = max(errors['conditional_acc'],abs(correct[~sm].mean()*100-mm['gt_object_attr_acc_unseen']))
                diagnostics.append(dict(case=case,seed=seed,phase=phase,
                    seen_images=int(sm.sum()),unseen_images=int((~sm).sum()),
                    seen_correct=int(correct[sm].sum()),unseen_correct=int(correct[~sm].sum()),
                    attr_seen=float(correct[sm].mean()*100),attr_unseen=float(correct[~sm].mean()*100),
                    object_acc=float((obj_prob.argmax(-1)==d['o']).mean()*100),
                    support_unseen=float(support[~sm].mean()*100),
                    unseen_joint_ceiling=exact['unseen'][-1],
                    unseen_cond_nll=float(-np.log(cond[np.arange(len(cond)),d['a']])[~sm].mean())))
                predictions[(case,seed,phase)] = dict(prob=prob,conditional=cond,correct=correct,
                    obj_pred=obj_prob.argmax(-1),joint_pred=d['pred'],calibrated=calibrated,
                    seen=sm,pairs=d['true_pairs'])
                for pair in sorted(set(d['true_pairs'])):
                    mask = np.array([p==pair for p in d['true_pairs']])
                    per_pair.append(dict(case=case,seed=seed,phase=phase,attribute=pair[0],object=pair[1],
                        seen=pair in train,images=int(mask.sum()),correct=int(correct[mask].sum()),
                        accuracy=float(correct[mask].mean()*100),pair_correct=int(d['correct'][mask].sum()),
                        calibrated_pair_correct=int(calibrated[mask].sum())))

    for case,seed in items:
        run = folder(case,seed)
        m,p,h = read(run/'metrics.json'),read(run/'protocol.json'),read(run/'history.json')
        cfg = p['config']
        epoch = int(np.argmax([r['val_cw']['AUC_native'] for r in h]))+1
        assert m['selected_epoch']==epoch and len(h)==10 and h[-1]['step']==3600
        assert p['seed']==seed and p['feature_manifest_sha256']==cache_sha
        assert helper.sha(run/'checkpoints/best.pt')==m['checkpoint_sha256']
        if not case.startswith('A'):
            source = ARCHIVE/('vgcpod_v1_roundC' if case.startswith('C') else 'vgcpod_v1_roundE' if case.startswith('E') else 'vgcpod_v1_roundD')
            sub = 'round_c' if case.startswith('C') else 'round_e' if case.startswith('E') else 'round_d'
            assert cfg==read(source/'configs'/sub/(case+'.json'))
            for name,fingerprint in p['source_sha256'].items():
                assert helper.sha(source/name)==fingerprint==helper.sha(ROOT/source.name/name)
                audit['source_files_verified'] += 1
            best = helper.checkpoint(run/'checkpoints/best.pt')
            last = helper.checkpoint(run/'checkpoints/last.pt')
            assert best['epoch']==epoch and last['history']==h
            assert all(abs(g['lr']-cfg['min_lr']*(cfg['cp_lr_scale'] if g['name']=='cp' else 1))<1e-15
                       for g in last['optimizer']['param_groups'])
        for entry in h:
            t,beta = entry['train'],cfg['task_aux_beta']
            error = abs(t['task']-1.5*t['pair']-1.5*beta/(1+beta)*t['support'])
            errors['loss_identity'] = max(errors['loss_identity'],error)
            histories.append(dict(case=case,seed=seed,epoch=entry['epoch'],chosen=entry['epoch']==epoch,
                val_cw=entry['val_cw']['AUC_native'],val_ow=entry['val_ow']['AUC_native'],
                val_cw_exact=entry['val_cw']['AUC_exact'],
                val_unseen_attr=entry['val_diagnostics']['gt_object_attr_acc_unseen'],
                task=t['task'],pair=t['pair'],cond=t['cond'],object_loss=t['object'],support=t['support'],
                orth=t['orth'],delta_rms=t['centered_delta_rms'],clip_fraction=t['clipped_step_fraction']))
            if case in ('C2','D1','E1'):
                gates.append(dict(case=case,seed=seed,epoch=entry['epoch'],chosen=entry['epoch']==epoch,
                    gamma_mean=t['gamma_mean'],total_variance=t['gamma_variance'],
                    image_variance=t['gamma_image_variance'],image_std=float(np.sqrt(t['gamma_image_variance'])),
                    image_fraction=t['gamma_image_variance']/t['gamma_variance'],
                    abs_deviation=t['gamma_abs_deviation'],mode_mean_error=t['gamma_mode_mean_deviation']))
        analyze_predictions(case,seed,run,m['metrics'],epoch)
        print('BASE',case,seed,flush=True)

    for seed in SEEDS:
        run = folder('E1',seed)
        h = read(run/'history.json')
        native = helper.checkpoint(run/'checkpoints/best_native.pt')
        exact = helper.checkpoint(run/'checkpoints/best_exact.pt')
        assert helper.sha(run/'checkpoints/best.pt')==helper.sha(run/'checkpoints/best_native.pt')
        assert native['epoch']==int(np.argmax([v['val_cw']['AUC_native'] for v in h]))+1
        assert exact['epoch']==int(np.argmax([v['val_cw']['AUC_exact'] for v in h]))+1
        assert native['epoch']==exact['epoch']
        assert all(torch.equal(value,exact['model'][key]) for key,value in native['model'].items())
        saved = read(run/'exact_selection/metrics.json')
        assert saved['checkpoint_sha256']==helper.sha(run/'checkpoints/best_exact.pt')
        analyze_predictions('E1_exact',seed,run/'exact_selection',saved['metrics'],saved['selected_epoch'])
        for setting in SETTINGS:
            with np.load(run/'predictions'/(setting+'.npz'),allow_pickle=False) as a, np.load(run/'exact_selection/predictions'/(setting+'.npz'),allow_pickle=False) as b:
                assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
                audit['exact_selection_duplicate_files']+=1
    for case,output,source,selection in [('C2',D_OUT,'vgcpod_v1_roundD','native'),
        ('D1',D_OUT,'vgcpod_v1_roundD','native'),('E1',E_OUT,'vgcpod_v1_roundE','native'),
        ('E1_exact',E_OUT,'vgcpod_v1_roundE','exact')]:
        for seed in SEEDS:
            base_case = 'E1' if case=='E1_exact' else case
            run = folder(base_case,seed)
            condition = 'mean_gate_exact' if selection=='exact' else 'mean_gate'
            root = output/condition/base_case/('seed%d'%seed)
            result = read(root/'metrics.json')
            checkpoint = 'best_exact.pt' if selection=='exact' else 'best.pt'
            reference = run/'exact_selection' if selection=='exact' else run
            assert result['identity']['checkpoint_sha256']==helper.sha(run/'checkpoints'/checkpoint)
            assert result['identity']['evaluator_sha256']==helper.sha(ARCHIVE/source/'mean_gate_eval.py')
            assert result['train_images']==manifest['counts']['train']==22998 and result['optimizer_steps']==0
            assert max(abs(v) for v in result['dynamic_minus_saved_auc'].values())==0
            with np.load(root/'train_mean_gamma.npz',allow_pickle=False) as d:
                mean,var = d['mean_gamma'],d['image_variance']
                assert mean.shape==(len(labels['objs']),32)
                assert int(d['train_images'])==22998 and d['object_names'].tolist()==labels['objs']
                moments.append(dict(case=case,seed=seed,train_images=22998,
                    mode_mean_error=float(abs(mean.mean(-1)-1).max()),min_mean=float(mean.min()),
                    max_mean=float(mean.max()),mean_abs_deviation=float(abs(mean-1).mean()),
                    static_variance=float(mean.var()),image_variance=float(var.mean()),
                    image_fraction=float(var.mean()/(mean.var()+var.mean()))))
            for setting in SETTINGS:
                with np.load(root/'dynamic/predictions'/(setting+'.npz'),allow_pickle=False) as a, np.load(reference/'predictions'/(setting+'.npz'),allow_pickle=False) as b:
                    assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
                    audit['dynamic_duplicate_files_identical']+=1
            analyze_predictions(case+'_mean',seed,root/'train_mean',result['metrics']['train_mean'],result['selected_epoch'])
            for setting in SETTINGS:
                phase = setting.split('_')[0]
                d = helper.index_predictions(root/'dynamic/predictions'/(setting+'.npz'),labels,train,
                    train|set(map(tuple,manifest['pairs'][phase])))
                ex = evaluator.exact_curve(d['log_probabilities'],d['truth'],d['seen_classes'])
                errors['exact_auc'] = max(errors['exact_auc'],abs(ex['AUC']-result['metrics']['dynamic'][setting]['AUC_exact']))
                audit['prediction_files'] += 1
            print('MEAN_GATE',case,seed,flush=True)

    metrics = {(r['case'],r['seed'],r['setting']):r for r in rows}
    contrasts = [(case,ref,seed) for case in ('C2','C2_mean','D1','D1_mean','E1','E1_mean')
                 for ref in ('A5','A6') for seed in SEEDS]
    contrasts += [(case,case+'_mean',seed) for case in ('C2','D1','E1') for seed in SEEDS]
    contrasts += [('D1','C2',seed) for seed in SEEDS]
    contrasts += [('E1',ref,seed) for ref in ('D1','C2') for seed in SEEDS]
    contrasts += [('E1_mean','D1_mean',seed) for seed in SEEDS]
    contrasts += [('E1_exact','E1',seed) for seed in SEEDS]
    contrasts += [('E1_exact_mean','E1_mean',seed) for seed in SEEDS]
    contrasts += [(case,'C1',seed) for case in ('E1','D1','C2') for seed in (3351,3352)]
    contrasts += [('C1','A5',seed) for seed in (3351,3352)]
    comparisons,similarity,pair_deltas = [],[],[]
    for high,low,seed in contrasts:
        for setting in SETTINGS:
            a,b = metrics[(high,seed,setting)],metrics[(low,seed,setting)]
            comparisons.append(dict(comparison=high+'-'+low,seed=seed,setting=setting,
                **{k:a[k]-b[k] for k in KEYS}))
        for phase in ('val','test'):
            a,b = predictions[(high,seed,phase)],predictions[(low,seed,phase)]
            assert a['pairs']==b['pairs'] and np.array_equal(a['seen'],b['seen'])
            for group in ('all','seen','unseen'):
                mask = np.ones_like(a['seen']) if group=='all' else a['seen'] if group=='seen' else ~a['seen']
                ac,bc = a['correct'][mask],b['correct'][mask]
                similarity.append(dict(comparison=high+'-'+low,seed=seed,phase=phase,group=group,
                    images=int(mask.sum()),recovered=int((ac&~bc).sum()),regressed=int((~ac&bc).sum()),
                    net_correct=int(ac.sum())-int(bc.sum()),
                    conditional_agreement=float((a['conditional'][mask].argmax(-1)==b['conditional'][mask].argmax(-1)).mean()*100),
                    joint_agreement=float((a['joint_pred'][mask]==b['joint_pred'][mask]).mean()*100),
                    object_agreement=float((a['obj_pred'][mask]==b['obj_pred'][mask]).mean()*100),
                    conditional_tv=float((abs(a['conditional'][mask]-b['conditional'][mask]).sum(-1)/2).mean()),
                    calibrated_correct_delta=int(a['calibrated'][mask].sum())-int(b['calibrated'][mask].sum())))
            for pair in sorted(set(a['pairs'])):
                left = next(r for r in per_pair if r['case']==high and r['seed']==seed and r['phase']==phase and (r['attribute'],r['object'])==pair)
                right = next(r for r in per_pair if r['case']==low and r['seed']==seed and r['phase']==phase and (r['attribute'],r['object'])==pair)
                pair_deltas.append(dict(comparison=high+'-'+low,seed=seed,phase=phase,
                    attribute=pair[0],object=pair[1],seen=pair in train,images=left['images'],
                    left_accuracy=left['accuracy'],right_accuracy=right['accuracy'],
                    correct_delta=left['correct']-right['correct'],
                    pair_correct_delta=left['pair_correct']-right['pair_correct'],
                    calibrated_pair_delta=left['calibrated_pair_correct']-right['calibrated_pair_correct']))
    means,paired_means = [],[]
    for case in sorted({r['case'] for r in rows}):
        for setting in SETTINGS:
            group = [r for r in rows if r['case']==case and r['setting']==setting]
            row = dict(case=case,setting=setting,n=len(group),seeds=[r['seed'] for r in group])
            for key in KEYS:
                row[key+'_mean'],row[key+'_sd'] = float(np.mean([r[key] for r in group])),float(np.std([r[key] for r in group],ddof=1)) if len(group)>1 else None
            means.append(row)
    for comp in sorted({r['comparison'] for r in comparisons}):
        for setting in SETTINGS:
            group = [r for r in comparisons if r['comparison']==comp and r['setting']==setting]
            row = dict(comparison=comp,setting=setting,n=len(group),seeds=[r['seed'] for r in group])
            for key in KEYS:
                row[key+'_mean'],row[key+'_sd'] = float(np.mean([r[key] for r in group])),float(np.std([r[key] for r in group],ddof=1)) if len(group)>1 else None
            paired_means.append(row)
    assert errors['exact_auc']<.001 and errors['fixed_bias_hm']<1e-8
    assert errors['conditional_acc']<1e-8 and errors['support_mass']<1e-6
    assert errors['probability_sum']<1e-5 and errors['loss_identity']<1e-5
    audit['max_recompute_errors'] = errors
    data = dict(metrics=rows,means=means,paired_means=paired_means,comparisons=comparisons,
        diagnostics=diagnostics,history=histories,gates=gates,moments=moments,
        similarity=similarity,per_pair=per_pair,pair_deltas=pair_deltas,audit=audit)
    for name in data:
        if name!='audit':
            csv_write(name+'.csv',data[name])
    write(OUT/'analysis.json',data)
    write(OUT/'audit.json',audit)
    plots(data)
    print(json.dumps(audit,indent=2))


def plots(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    cases = ['C2','D1','E1']
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    for ax,setting,title in zip(axes,['val_cw','test_cw','test_ow'],['VAL CW','TEST CW','TEST OW']):
        for seed in SEEDS:
            values = [next(r['AUC_native'] for r in data['metrics'] if r['case']==c
                           and r['seed']==seed and r['setting']==setting) for c in cases]
            ax.plot([.1,.125,.15],values,color='#bbc6cd',lw=1,alpha=.8)
        values = [next(r['AUC_native_mean'] for r in data['means'] if r['case']==c and r['setting']==setting) for c in cases]
        ax.plot([.1,.125,.15],values,color='#176f79',marker='o',lw=2.3,label='3-seed mean')
        for x,y in zip([.1,.125,.15],values):ax.annotate('%.3f'%y,(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
        ax.set_title(title);ax.set_xlabel('rho');ax.set_ylabel('AUC_native (area x 100)');ax.set_xticks([.1,.125,.15]);ax.set_xlim(.095,.155);ax.legend()
        low,high=ax.get_ylim();ax.set_ylim(low,high+.2)
    fig.tight_layout();fig.savefig(OUT/'figures/01_rho_sweep.png');plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for ax,setting,title in zip(axes,['test_cw','test_ow'],['TEST CW','TEST OW']):
        vals = [next(r['AUC_native'] for r in data['comparisons'] if r['comparison']=='E1-D1'
                     and r['seed']==s and r['setting']==setting) for s in SEEDS]
        ax.bar(list(map(str,SEEDS)),vals,color=['#889eae' if v<0 else '#176f79' for v in vals])
        for i,y in enumerate(vals):ax.text(i,y+(.02 if y>=0 else -.02),'%+.3f'%y,ha='center',va='bottom' if y>=0 else 'top')
        ax.axhline(0,color='#333',lw=1);ax.set_ylim(min(vals+[0])-.15,max(vals+[0])+.15)
        ax.set_title(title+': E1 - D1');ax.set_ylabel('Paired AUC_native change (points)')
    fig.tight_layout();fig.savefig(OUT/'figures/02_e1_d1_deltas.png');plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for ax,setting,title in zip(axes,['test_cw','test_ow'],['TEST CW','TEST OW']):
        means = [next(r['AUC_native_mean'] for r in data['paired_means'] if r['comparison']==c+'-'+c+'_mean'
                      and r['setting']==setting) for c in cases]
        ax.bar(cases,means,color=['#8b9caf','#176f79','#6a98a0'],alpha=.65)
        points=[]
        for i,c in enumerate(cases):
            vals=[next(r['AUC_native'] for r in data['comparisons'] if r['comparison']==c+'-'+c+'_mean'
                       and r['seed']==seed and r['setting']==setting) for seed in SEEDS]
            points+=vals;ax.scatter([i-.07,i,i+.07],vals,color='#263943',s=24,zorder=3)
            ax.text(i,means[i]+.025,'%+.3f'%means[i],ha='center',fontsize=9)
        ax.axhline(0,color='#333',lw=1);ax.set_ylim(min(points+[0])-.08,max(points+[0])+.08)
        ax.set_title(title+': dynamic minus TRAIN mean');ax.set_ylabel('AUC_native change (points)')
    fig.tight_layout();fig.savefig(OUT/'figures/03_dynamic_effect.png');plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    values1=[];values2=[]
    for seed in SEEDS:
        a=next(r for r in data['diagnostics'] if r['case']=='E1' and r['seed']==seed and r['phase']=='test')
        b=next(r for r in data['diagnostics'] if r['case']=='D1' and r['seed']==seed and r['phase']=='test')
        values1.append(a['support_unseen']-b['support_unseen']);values2.append(a['attr_unseen']-b['attr_unseen'])
    for ax,values,title,ylabel in zip(axes,[values1,values2],
        ['E1 - D1: seen support on unseen images','E1 - D1: unseen conditional attributes'],
        ['Probability mass change (percentage points)','Accuracy change (percentage points)']):
        ax.bar(list(map(str,SEEDS)),values,color='#889eae');ax.axhline(0,color='#333',lw=1)
        for i,y in enumerate(values):ax.text(i,y+(.02 if y>=0 else -.04),'%+.3f'%y,ha='center',va='bottom' if y>=0 else 'top')
        ax.set_ylim(min(values+[0])-.35,max(values+[0])+.15);ax.set_title(title,fontsize=10);ax.set_ylabel(ylabel)
    fig.tight_layout();fig.savefig(OUT/'figures/04_support_attributes.png');plt.close(fig)


if __name__=='__main__':
    main()
