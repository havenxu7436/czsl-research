"""Describe saved OW probabilities; these diagnostics do not alter predictions."""
from pathlib import Path

import numpy as np

import round_b as shared

METRICS=('object_accuracy','object_nll','conditional_accuracy','conditional_nll',
         'seen_support_pct','extra_support_pct','unseen_rank_ceiling','extra_dominates_valid')


def diagnose(data,manifest,phase):
    labels=manifest['labels'];train=set(map(tuple,manifest['pairs']['train']))
    cw=train|set(map(tuple,manifest['pairs'][phase]))
    pairs=[tuple(p) for p in data['candidate_pairs'].tolist()]
    expected={(a,o) for a in labels['attrs'] for o in labels['objs']}
    if len(pairs)!=len(expected) or set(pairs)!=expected:
        raise ValueError('Ranking diagnostics require all OW candidate pairs')
    ai=np.array([labels['attrs'].index(a) for a,o in pairs])
    oi=np.array([labels['objs'].index(o) for a,o in pairs])
    attrs,objs=data['attribute_truth'],data['object_truth'];index=np.arange(len(attrs))
    truth_pairs=[(labels['attrs'][a],labels['objs'][o]) for a,o in zip(attrs,objs)]
    seen_images=np.array([p in train for p in truth_pairs])
    seen=np.array([p in train for p in pairs])
    valid=np.array([p in cw and p not in train for p in pairs])
    extra=np.array([p not in cw for p in pairs])
    scores=data['log_probabilities'].astype(np.float64)
    joint=np.empty((len(attrs),len(labels['objs']),len(labels['attrs'])),dtype=np.float64)
    joint[:,oi,ai]=scores
    obj=np.logaddexp.reduce(joint,axis=-1)
    conditional=data['conditional_logits_gt_object'].astype(np.float64)
    conditional=conditional-np.logaddexp.reduce(conditional,axis=-1)[:,None]
    unseen=np.flatnonzero(~seen)
    unseen_pred=unseen[scores[:,~seen].argmax(-1)]
    truth_index={p:i for i,p in enumerate(pairs)}
    truth=np.array([truth_index[p] for p in truth_pairs])
    ceiling_correct=unseen_pred==truth
    object_correct=obj.argmax(-1)==objs
    attr_correct=conditional.argmax(-1)==attrs
    support=np.exp(scores[:,seen]).sum(-1)
    extra_support=np.exp(scores[:,extra]).sum(-1)
    dominates=(scores[:,extra].max(-1)>scores[:,valid].max(-1)) if extra.any() and valid.any() else np.zeros(len(attrs),dtype=bool)
    values=dict(object_accuracy=100*object_correct,object_nll=-obj[index,objs],
        conditional_accuracy=100*attr_correct,conditional_nll=-conditional[index,attrs],
        seen_support_pct=100*support,extra_support_pct=100*extra_support,
        unseen_rank_ceiling=100*ceiling_correct,extra_dominates_valid=100*dominates)
    rows=[]
    for group in ('all','seen','unseen'):
        mask=np.ones(len(attrs),dtype=bool) if group=='all' else seen_images if group=='seen' else ~seen_images
        if not mask.any():
            continue
        rows.append(dict(group=group,images=int(mask.sum()),extra_candidates=int(extra.sum()),
            valid_unseen_candidates=int(valid.sum()),**{k:float(v[mask].mean()) for k,v in values.items()}))
    pair_rows=[]
    for pair in sorted(set(truth_pairs)):
        mask=np.array([p==pair for p in truth_pairs])
        pair_rows.append(dict(attribute=pair[0],object=pair[1],seen=pair in train,images=int(mask.sum()),
            conditional_correct=int(attr_correct[mask].sum()),unseen_ceiling_correct=int(ceiling_correct[mask].sum()),
            conditional_nll=float(values['conditional_nll'][mask].mean()),
            seen_support_pct=float(values['seen_support_pct'][mask].mean()),
            extra_support_pct=float(values['extra_support_pct'][mask].mean())))
    return rows,pair_rows


def aggregate(rows,keys=('case','selection','condition','phase','group')):
    import statistics
    result=[]
    for group in sorted({tuple(r[k] for k in keys) for r in rows}):
        selected=sorted([r for r in rows if tuple(r[k] for k in keys)==group],key=lambda r:r['seed'])
        entry=dict(zip(keys,group));entry.update(n_seeds=len(selected),seeds=[r['seed'] for r in selected])
        for key in METRICS:
            values=[r[key] for r in selected]
            entry[key+'_mean']=statistics.mean(values)
            entry[key+'_sd']=statistics.stdev(values) if len(values)>1 else None
        result.append(entry)
    return result


def summarize_entries(entries,manifest,out):
    rows,pairs=[],[]
    for entry in entries:
        for phase in ('val','test'):
            path=Path(entry['path'])/'predictions'/(phase+'_ow.npz')
            if not path.exists():
                continue
            with np.load(path,allow_pickle=False) as archive:
                data={k:archive[k].copy() for k in archive.files}
            expected=shared.read(Path(entry['cache'])/(phase+'_records.json'))
            if data['image_ids'].tolist()!=[r['image'] for r in expected]:
                raise ValueError('Prediction image order differs: '+str(path))
            if data['attribute_truth'].tolist()!=[r['attr'] for r in expected] or data['object_truth'].tolist()!=[r['object'] for r in expected]:
                raise ValueError('Prediction labels differ: '+str(path))
            diag,per_pair=diagnose(data,manifest,phase)
            base={key:entry[key] for key in ('case','seed','selection','condition')}
            base.update(phase=phase,selected_epoch=entry['selected_epoch'])
            rows.extend(dict(base,**r) for r in diag)
            pairs.extend(dict(base,**r) for r in per_pair)
    index={(r['case'],r['seed'],r['selection'],r['condition'],r['phase'],r['group']):r for r in rows}
    deltas=[]
    for left in rows:
        targets=[]
        if left['case'] in ('G1','H1','I1','I2','J1') and left['condition']=='dynamic':
            targets=[('G0','static'),(left['case'],'train_mean')]
            if left['case']=='H1':targets.append(('G1','dynamic'))
            if left['case'] in ('I1','I2'):
                targets+=[('G1','dynamic'),('H1','dynamic')]
                if left['case']=='I1':targets.append(('I2','dynamic'))
            if left['case']=='J1':
                targets+=[('I2','dynamic'),('G1','dynamic'),('H1','dynamic'),('I1','dynamic')]
            if left['selection']=='native':
                targets+=[('E1','dynamic'),('F1','dynamic'),('D1','dynamic'),('A5','static')]
        for case,condition in targets:
            right=index.get((case,left['seed'],left['selection'],condition,left['phase'],left['group']))
            if right:
                comparison=left['case']+'-'+case+('_mean' if case==left['case'] else '')
                deltas.append(dict(comparison=comparison,seed=left['seed'],selection=left['selection'],
                    phase=left['phase'],group=left['group'],**{k:left[k]-right[k] for k in METRICS}))
    paired_mean_sd=aggregate(deltas,('comparison','selection','phase','group'))
    for name,data in [('ranking_diagnostics',rows),('ranking_diagnostics_mean_sd',aggregate(rows)),
                      ('ranking_diagnostic_deltas',deltas),('ranking_diagnostic_paired_mean_sd',paired_mean_sd),
                      ('pair_probability_diagnostics',pairs)]:
        shared.csv_write(Path(out)/(name+'.csv'),data)
    result=dict(runs=rows,mean_sd=aggregate(rows),paired_deltas=deltas,paired_mean_sd=paired_mean_sd,per_pair=pairs,
        rule='Full saved OW distribution; CW-excluded pairs are a split-specific descriptive category')
    shared.dump(Path(out)/'ranking_diagnostics.json',result)
    return result
