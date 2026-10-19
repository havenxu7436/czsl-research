"""Audit and compare the six downloaded round-A experiments with seed-matched references."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'server_results_roundA_20261019/roundA_20261019'
OLD = ROOT / 'server_results_20261018/vgcpod_v1_runs'
PKG = ROOT / 'vgcpod_v1_roundA'
torch.set_num_threads(1)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csvsave(name, rows):
    if not rows:
        return
    with (OUT/'tables'/name).open('w', encoding='utf-8-sig', newline='') as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    (OUT/'tables').mkdir(parents=True, exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    helper = module('old_analysis_helpers', ROOT/'analysis_20261018/analyze_results.py')
    evaluator = module('round_a_saved_evaluator', PKG/'vgcpod/metrics.py')
    manifest = read(OLD/'cache_ut_clip_l14/features_manifest.json')
    labels = manifest['labels']
    train = set(map(tuple, manifest['pairs']['train']))
    audit = {'archive_sha256': sha(ROOT/'vgcpod_roundA_results_20261019.tar.gz'),
             'new_runs': 6, 'prediction_exports_new': 0, 'epochs_per_run': 10,
             'feature_manifest_sha256': sha(OLD/'cache_ut_clip_l14/features_manifest.json'),
             'errors': [], 'all_source_hashes_match_delivered_package': True}
    assert (RUNS/'launcher.exitcode').read_text().strip() == '0'
    statuses = read(RUNS/'round_a_status.json')
    assert len(statuses) == 6 and all(r['status'] == 'complete' and r['exit_code'] == 0 for r in statuses)
    discrepancies = dict(accuracy=0., fixed_bias_hm=0., support_mass=0., native_auc=0., exact_auc=0.,
                         loss_identity=0., distribution_normalization=0.)
    folders = {'D0': OLD/'main/ut-zap50k/V0/seed3351', 'V1': OLD/'main/ut-zap50k/V1/seed3351'}
    for case in ['A1','A2','A3','A4','A5','A6']:
        folders[case] = next((RUNS/case).glob('ut-zap50k/V*/seed3351'))
    metrics, rows, history_rows, distribution, pair_rows, category_rows = {}, [], [], [], [], []
    predictions = {}
    chosen_info = {}
    for case, folder in folders.items():
        m, p, h = read(folder/'metrics.json'), read(folder/'protocol.json'), read(folder/'history.json')
        cfg = p['config']
        assert p['feature_manifest_sha256'] == audit['feature_manifest_sha256']
        selected = int(np.argmax([r['val_cw']['AUC_native'] for r in h]))+1
        assert selected == m['selected_epoch']
        assert sha(folder/'checkpoints/best.pt') == m['checkpoint_sha256']
        if case.startswith('A'):
            assert cfg == read(PKG/'configs/round_a'/(case+'.json'))
            for name, expected in p['source_sha256'].items():
                assert sha(PKG/name) == expected, (case,name)
            assert [r['epoch'] for r in h] == list(range(1,11)) and h[-1]['step'] == 3600
            best, last = helper.checkpoint(folder/'checkpoints/best.pt'), helper.checkpoint(folder/'checkpoints/last.pt')
            assert best['epoch'] == selected and last['epoch'] == 10
            assert last['history'] == h
            for g in last['optimizer']['param_groups']:
                expected = cfg['min_lr'] * (cfg['cp_lr_scale'] if g['name']=='cp' else 1.)
                assert abs(g['lr']-expected) < 1e-15
            beta = cfg['task_aux_beta']
            for r in h:
                t = r['train']
                err = abs(t['task']-(1.5*t['pair']+1.5*beta/(1+beta)*t['support']))
                discrepancies['loss_identity'] = max(discrepancies['loss_identity'], err)
                assert err < 1e-5
                if cfg['variant'] == 'V1':
                    assert abs(r['lr_groups']['cp']/r['lr_groups']['adapter']-cfg['cp_lr_scale']) < 1e-12
        chosen = h[selected-1]
        chosen_info[case] = dict(epoch=selected, beta=cfg.get('task_aux_beta',.5),
            cp_lr=cfg['lr']*cfg.get('cp_lr_scale',1.) if cfg['variant']=='V1' else None,
            delta_rms=chosen['train'].get('centered_delta_rms'), clip_fraction=chosen['train'].get('clipped_step_fraction'),
            val_peak=chosen['val_cw']['AUC_native'], val_last=h[-1]['val_cw']['AUC_native'])
        metrics[case] = m
        for r in h:
            history_rows.append(dict(case=case,epoch=r['epoch'],selected=r['epoch']==selected,
                val_cw=r['val_cw']['AUC_native'],val_ow=r.get('val_ow',{}).get('AUC_native'),
                val_attr_unseen=r.get('val_diagnostics',{}).get('gt_object_attr_acc_unseen'),
                val_support_unseen=r.get('val_diagnostics',{}).get('support_mass_unseen'),
                loss=r['train']['total'], pair=r['train']['pair'], cond=r['train']['cond'],
                delta_rms=r['train'].get('centered_delta_rms'), clip_fraction=r['train'].get('clipped_step_fraction')))
        for setting in ['val_cw','val_ow','test_cw','test_ow']:
            phase,world = setting.split('_')
            cw_pairs = train | set(map(tuple,manifest['pairs'][phase]))
            data = helper.index_predictions(folder/'predictions'/(setting+'.npz'), labels, train, cw_pairs)
            if case.startswith('A'):
                audit['prediction_exports_new'] += 1
            records = read(OLD/'cache_ut_clip_l14'/(phase+'_records.json'))
            assert data['image_ids'].tolist()==[r['image'] for r in records]
            assert data['a'].tolist()==[r['attr'] for r in records] and data['o'].tolist()==[r['object'] for r in records]
            mm = m['metrics'][setting]
            lp = data['log_probabilities']; pred = data['pred']; y=data['truth']; sm=data['seen_images']
            discrepancies['distribution_normalization']=max(discrepancies['distribution_normalization'],float(abs(data['probabilities'].sum(1)-1).max()))
            assert mm['candidates']==len(data['candidate_pairs']) and mm['images']==len(y)
            native=evaluator.native_curve(lp,y,data['seen_classes'])
            exact=evaluator.exact_curve(lp,y,data['seen_classes'])
            discrepancies['native_auc']=max(discrepancies['native_auc'],abs(native['AUC']-mm['AUC_native']))
            discrepancies['exact_auc']=max(discrepancies['exact_auc'],abs(exact['AUC']-mm['AUC_exact']))
            discrepancies['accuracy']=max(discrepancies['accuracy'],abs(data['correct'].mean()*100-mm['pair_acc_unbiased']))
            bias=m['metrics']['val_'+world]['bias_native'];assert mm['fixed_val_bias']==bias
            z=torch.from_numpy(lp.copy());z[:,~torch.from_numpy(data['seen_classes'])]+=bias
            predcal=z.topk(1,dim=1).indices[:,0].numpy();ok=predcal==y
            hm=float(evaluator.hm(ok[sm].mean(),ok[~sm].mean())*100)
            discrepancies['fixed_bias_hm']=max(discrepancies['fixed_bias_hm'],abs(hm-mm['HM_at_val_bias']))
            cp=helper.softmax(data['conditional_logits_gt_object']);condpred=cp.argmax(1)
            if world=='ow':
                support=data['probabilities'][:,data['seen_classes']].sum(1).astype(float)
                objectprob=np.stack([data['probabilities'][:,data['oi']==j].sum(1) for j in range(len(labels['objs']))],axis=-1)
                d=dict(case=case,phase=phase,attr_seen=float((condpred[sm]==data['a'][sm]).mean()*100),
                    attr_unseen=float((condpred[~sm]==data['a'][~sm]).mean()*100),
                    attr_unseen_correct=int((condpred[~sm]==data['a'][~sm]).sum()),
                    attr_seen_correct=int((condpred[sm]==data['a'][sm]).sum()),
                    support_seen=float(support[sm].mean()),support_unseen=float(support[~sm].mean()),
                    object_acc=float((objectprob.argmax(1)==data['o']).mean()*100),
                    object_acc_unseen=float((objectprob.argmax(1)[~sm]==data['o'][~sm]).mean()*100),
                    unseen_limit_acc=exact['unseen'][-1],
                    cond_nll_unseen=float(-np.log(cp[np.arange(len(y)),data['a']][~sm]).mean()))
                distribution.append(d)
                if case.startswith('A'):
                    discrepancies['support_mass']=max(discrepancies['support_mass'],abs(d['support_unseen']-mm['support_mass_unseen']))
                    assert abs(d['attr_unseen']-mm['gt_object_attr_acc_unseen'])<1e-8
            for pair in sorted(set(data['true_pairs'])):
                mask=np.array([t==pair for t in data['true_pairs']])
                pair_rows.append(dict(case=case,setting=setting,attribute=pair[0],object=pair[1],seen=pair in train,
                    images=int(mask.sum()),cond_acc=float((condpred[mask]==data['a'][mask]).mean()*100),
                    pair_acc_unbiased=float(data['correct'][mask].mean()*100),pair_acc_calibrated=float(ok[mask].mean()*100)))
            if setting=='test_ow':
                for ai,name in enumerate(labels['attrs']):
                    mask=data['a']==ai
                    if mask.any():
                        category_rows.append(dict(case=case,attribute=name,images=int(mask.sum()),
                            cond_acc=float((condpred[mask]==ai).mean()*100)))
            rows.append(dict(case=case,setting=setting,epoch=selected,**{k:v for k,v in mm.items() if not isinstance(v,dict)}))
            predictions[case,setting]=data
        print('AUDITED',case,flush=True)
    audit['max_recompute_errors']=discrepancies
    assert discrepancies['accuracy']<1e-8 and discrepancies['fixed_bias_hm']<1e-8
    assert discrepancies['native_auc']<.03 and discrepancies['exact_auc']<.001
    assert discrepancies['distribution_normalization']<1e-5 and discrepancies['support_mass']<1e-6
    summary=read(RUNS/'round_a_summary.json')
    for r in summary['runs']:
        for setting in ['val_cw','val_ow','test_cw','test_ow']:
            assert r[setting+'_AUC_native']==metrics[r['case']]['metrics'][setting]['AUC_native']
    audit['saved_summary_matches']='passed'
    comparisons=[]
    for hi,lo in [('A1','V1'),('A2','V1'),('A3','V1'),('A4','V1'),('A5','V1'),('A5','D0'),
                  ('A5','A1'),('A5','A4'),('A5','A6'),('A6','D0'),('A1','D0')]:
        dr=dict(comparison=hi+'-'+lo)
        for setting in ['val_cw','val_ow','test_cw','test_ow']:
            for key in ['AUC_native','AUC_exact','HM_at_val_bias','gt_object_attr_acc']:
                dr[setting+'_'+key]=metrics[hi]['metrics'][setting][key]-metrics[lo]['metrics'][setting][key]
        comparisons.append(dr)
    interactions={setting:metrics['A5']['metrics'][setting]['AUC_native']-metrics['A1']['metrics'][setting]['AUC_native']
                  -metrics['A4']['metrics'][setting]['AUC_native']+metrics['V1']['metrics'][setting]['AUC_native']
                  for setting in ['val_cw','val_ow','test_cw','test_ow']}
    csvsave('all_metrics.csv',rows);csvsave('history.csv',history_rows);csvsave('distribution.csv',distribution)
    csvsave('per_pair.csv',pair_rows);csvsave('attribute.csv',category_rows);csvsave('comparisons.csv',comparisons)
    save(OUT/'audit.json',audit)
    save(OUT/'analysis.json',dict(metrics=metrics,chosen=chosen_info,distribution=distribution,
                               comparisons=comparisons,interaction=interactions,audit=audit))
    plots(metrics,history_rows,distribution)
    print(json.dumps(dict(chosen=chosen_info,comparisons=comparisons,interaction=interactions,
                         distribution=distribution,audit=audit),ensure_ascii=False,indent=2))


def plots(metrics,history,distribution):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    cases=['D0','V1','A1','A2','A3','A4','A5','A6']
    colors=['#697887','#367fa2','#5d9b91','#b5aaa0','#baa25b','#bc6870','#145b7c','#827eae']
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,setting,title in zip(axs,['val_cw','test_cw','test_ow'],['VAL CW (selection)','TEST CW','TEST OW']):
        vals=[metrics[c]['metrics'][setting]['AUC_native'] for c in cases]
        ax.bar(cases,vals,color=colors)
        ax.axhline(vals[1],ls='--',color='#367fa2',lw=1)
        for i,v in enumerate(vals):ax.text(i,v+.35,f'{v:.2f}',ha='center',fontsize=8)
        ax.set_title(title);ax.set_ylabel('AUC_native (area x 100)');ax.set_ylim(0,max(vals)*1.17)
    fig.suptitle('Round A: seed3351; D0/V1 are the seed-matched original runs')
    fig.tight_layout();fig.savefig(OUT/'figures/01_auc_comparison.png');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for c,col in zip(['V1','A1','A2','A4','A5'],['#367fa2','#5d9b91','#b5aaa0','#bc6870','#145b7c']):
        h=[r for r in history if r['case']==c]
        for ax,key in zip(axs,['val_cw','val_ow','delta_rms']):
            if all(r[key] is not None for r in h):ax.plot([r['epoch'] for r in h],[r[key] for r in h],label=c,color=col)
    for ax,title in zip(axs,['VAL CW AUC','VAL OW AUC','TRAIN centered residual RMS']):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'figures/02_learning_curves.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    cc=['D0','V1','A1','A4','A5','A6']
    for ax,phase in zip(axs,['val','test']):
        for c in cc:
            d=next(r for r in distribution if r['case']==c and r['phase']==phase)
            ax.scatter(d['support_unseen']*100,d['attr_unseen'],s=50)
            ax.annotate(c,(d['support_unseen']*100,d['attr_unseen']),xytext=(4,5),textcoords='offset points')
        ax.set_title(phase.upper()+' unseen images')
        ax.set_xlabel('Probability on seen compositions (%)');ax.set_ylabel('GT-object attribute accuracy (%)')
    fig.tight_layout();fig.savefig(OUT/'figures/03_support_and_transfer.png');plt.close(fig)


if __name__=='__main__':main()
