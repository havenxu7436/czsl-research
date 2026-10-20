# Numerical calculations unchanged; repository paths adapted.
"""Audit G0/G1 matched training, beta comparison and separate checkpoint selections."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parent.parent
OUT = Path(__file__).resolve().parent/'recomputed'
ARCHIVE = STAGE/'results/raw_export'
A_OUT = STAGE.parent/'02/results/raw_export/roundA_20261019'
C_OUT = STAGE.parent/'05/results/raw_export/roundC_20261020'
D_OUT = STAGE.parent/'05/results/raw_export/roundD_20261020'
E_OUT = STAGE.parent/'06/results/raw_export/roundE_20261020'
F_OUT = STAGE.parent/'07/results/raw_export/roundF_20261020'
G_OUT = ARCHIVE/'roundG_20261020'
CACHE = STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14'


def source_file(package,member):
    number = {'vgcpod_v1_roundC':'04','vgcpod_v1_roundD':'05','vgcpod_v1_roundE':'06','vgcpod_v1_roundF':'07','vgcpod_v1_roundG':'08'}[package]
    stage = STAGE.parent/number
    mapping = json.loads((stage/'package_members.json').read_text(encoding='utf-8'))['members']
    return stage/mapping[member]


def reference_base(case,seed):
    if case.startswith('A'):return A_OUT
    if case.startswith('C'):
        return STAGE.parent/('04' if seed==3351 else '05')/'results/raw_export/roundC_20261020'
    return {'D':D_OUT,'E':E_OUT,'F':F_OUT,'G':G_OUT}[case[0]]

SETTINGS = ('val_cw','val_ow','test_cw','test_ow')
SEEDS = (3351,3352,3353)
KEYS = ('AUC_native','AUC_exact','HM_at_val_bias','gt_object_attr_acc_unseen',
        'support_mass_unseen','pair_nll_unbiased','gt_object_attr_nll','seen_at_val_bias','unseen_at_val_bias')
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
    base = reference_base(case,seed)
    return next((base/case).glob('ut-zap50k/V*/seed%d'%seed))


def main():
    (OUT/'tables').mkdir(parents=True,exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    helper = load_module('d_prediction_helpers',STAGE.parent/'v1/code/analysis_20261018/analyze_results.py')
    evaluator = load_module('d_archived_evaluator',source_file('vgcpod_v1_roundD','vgcpod/metrics.py'))
    manifest = read(CACHE/'features_manifest.json')
    labels,train = manifest['labels'],set(map(tuple,manifest['pairs']['train']))
    cache_sha = helper.sha(CACHE/'features_manifest.json')
    assert (G_OUT/'launcher.exitcode').read_text().strip()=='0'
    events = read(G_OUT/'round_g_status.json')
    assert {(e['seed'],e['gpu']) for e in events}=={(3351,'1'),(3352,'2'),(3353,'3')}
    assert all(e['status']=='complete' and len(e['trainings'])==2 and all(t['exit_code']==0 for t in e['trainings']) for e in events)
    assert all([t['case'] for t in e['trainings']]==['G0','G1'] for e in events)
    assert all(len(e['mean_gate'])==2 and all(m['exit_code']==0 for m in e['mean_gate']) for e in events)
    audit = dict(archive_sha256=helper.sha(STAGE/'results/archives/vgcpod_roundG_results_20261020.tar.gz'),
        feature_manifest_sha256=cache_sha,base_runs=26,interventions=24,prediction_files=0,
        source_files_verified=0,dynamic_duplicate_files_identical=0,exact_selection_duplicate_files=0)
    errors = dict(native_auc=0.,exact_auc=0.,fixed_bias_hm=0.,probability_sum=0.,support_mass=0.,
                  conditional_acc=0.,loss_identity=0.)
    items = [(case,seed) for case in ('A5','A6','C2','D1','E1','F1','G0','G1') for seed in SEEDS]+[('C1',3351),('C1',3352)]
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
            source = ARCHIVE/('vgcpod_v1_roundC' if case.startswith('C') else 'vgcpod_v1_roundG' if case.startswith('G') else 'vgcpod_v1_roundF' if case.startswith('F') else 'vgcpod_v1_roundE' if case.startswith('E') else 'vgcpod_v1_roundD')
            sub = 'round_c' if case.startswith('C') else 'round_g' if case.startswith('G') else 'round_f' if case.startswith('F') else 'round_e' if case.startswith('E') else 'round_d'
            assert cfg==read(source_file(source.name,'configs/'+sub+'/'+case+'.json'))
            for name,fingerprint in p['source_sha256'].items():
                assert helper.sha(source_file(source.name,name))==fingerprint
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
                beta=beta,support_weight=1.5*beta/(1+beta),pair_weighted=1.5*t['pair'],
                support_weighted=1.5*beta/(1+beta)*t['support'],support_mass=t['support_mass'],
                batch_loss_residual=t.get('task_reconstruction_error'),
                orth=t['orth'],delta_rms=t['centered_delta_rms'],clip_fraction=t['clipped_step_fraction']))
            if case in ('C2','D1','E1','F1','G1'):
                gates.append(dict(case=case,seed=seed,epoch=entry['epoch'],chosen=entry['epoch']==epoch,
                    gamma_mean=t['gamma_mean'],total_variance=t['gamma_variance'],
                    image_variance=t['gamma_image_variance'],image_std=float(np.sqrt(t['gamma_image_variance'])),
                    image_fraction=t['gamma_image_variance']/t['gamma_variance'],
                    abs_deviation=t['gamma_abs_deviation'],mode_mean_error=t['gamma_mode_mean_deviation']))
        analyze_predictions(case,seed,run,m['metrics'],epoch)
        print('BASE',case,seed,flush=True)

    audit['selection_epochs'] = []
    for case in ('E1','F1','G0','G1'):
        for seed in SEEDS:
            run = folder(case,seed)
            h = read(run/'history.json')
            native = helper.checkpoint(run/'checkpoints/best_native.pt')
            exact = helper.checkpoint(run/'checkpoints/best_exact.pt')
            assert helper.sha(run/'checkpoints/best.pt')==helper.sha(run/'checkpoints/best_native.pt')
            assert native['epoch']==int(np.argmax([v['val_cw']['AUC_native'] for v in h]))+1
            assert exact['epoch']==int(np.argmax([v['val_cw']['AUC_exact'] for v in h]))+1
            same = all(torch.equal(value,exact['model'][key]) for key,value in native['model'].items())
            audit['selection_epochs'].append(dict(case=case,seed=seed,native=native['epoch'],exact=exact['epoch'],same_parameters=same))
            saved = read(run/'exact_selection/metrics.json')
            assert saved['checkpoint_sha256']==helper.sha(run/'checkpoints/best_exact.pt')
            analyze_predictions(case+'_exact',seed,run/'exact_selection',saved['metrics'],saved['selected_epoch'])
            for setting in SETTINGS:
                with np.load(run/'predictions'/(setting+'.npz'),allow_pickle=False) as a, np.load(run/'exact_selection/predictions'/(setting+'.npz'),allow_pickle=False) as b:
                    identical = a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
                    assert identical==same
                    audit['exact_selection_duplicate_files']+=int(identical)
    for case,output,source,selection in [('C2',D_OUT,'vgcpod_v1_roundD','native'),
        ('D1',D_OUT,'vgcpod_v1_roundD','native'),('E1',E_OUT,'vgcpod_v1_roundE','native'),
        ('E1_exact',E_OUT,'vgcpod_v1_roundE','exact'),('F1',F_OUT,'vgcpod_v1_roundF','native'),
        ('F1_exact',F_OUT,'vgcpod_v1_roundF','exact'),('G1',G_OUT,'vgcpod_v1_roundG','native'),
        ('G1_exact',G_OUT,'vgcpod_v1_roundG','exact')]:
        for seed in SEEDS:
            base_case = case.split('_')[0]
            run = folder(base_case,seed)
            condition = 'mean_gate_exact' if selection=='exact' else 'mean_gate'
            root = output/condition/base_case/('seed%d'%seed)
            result = read(root/'metrics.json')
            checkpoint = 'best_exact.pt' if selection=='exact' else 'best.pt'
            reference = run/'exact_selection' if selection=='exact' else run
            assert result['identity']['checkpoint_sha256']==helper.sha(run/'checkpoints'/checkpoint)
            assert result['identity']['evaluator_sha256']==helper.sha(source_file(source,'mean_gate_eval.py'))
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
    contrasts += [('F1',ref,seed) for ref in ('E1','D1','C2','A5','A6','F1_mean','F1_exact') for seed in SEEDS]
    contrasts += [('F1_mean',ref,seed) for ref in ('E1_mean','D1_mean','A5') for seed in SEEDS]
    contrasts += [('F1_exact_mean','F1_mean',seed) for seed in SEEDS]
    contrasts += [('F1','C1',seed) for seed in (3351,3352)]
    contrasts += [('G1',ref,seed) for ref in ('G0','E1','F1','D1','C2','A5','A6','G1_mean') for seed in SEEDS]
    contrasts += [('G0',ref,seed) for ref in ('E1','F1','D1','A5') for seed in SEEDS]
    contrasts += [('G1_mean',ref,seed) for ref in ('G0','E1_mean','F1_mean','A5') for seed in SEEDS]
    contrasts += [('G1_exact',ref,seed) for ref in ('G1','G0_exact','E1','F1','G1_exact_mean') for seed in SEEDS]
    contrasts += [('G0_exact','G0',seed) for seed in SEEDS]
    contrasts += [('G1', 'C1',seed) for seed in (3351,3352)]
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
    print(json.dumps(audit,indent=2))



if __name__=='__main__':
    main()
