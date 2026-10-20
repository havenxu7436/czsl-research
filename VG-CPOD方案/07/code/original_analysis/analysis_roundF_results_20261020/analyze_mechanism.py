"""Locate F1's ranking change using saved full-joint predictions and head swaps."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT/'server_results_roundF_20261020'
CACHE = ROOT/'server_results_20261018/vgcpod_v1_runs/cache_ut_clip_l14'
torch.set_num_threads(1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_csv(name,rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)


def load(case,seed,phase,labels):
    run = ARCHIVE/('round%s_20261020'%case[0])/case/'ut-zap50k/V1'/('seed%d'%seed)
    with np.load(run/'predictions'/('%s_ow.npz'%phase),allow_pickle=False) as data:
        saved = {k:data[k].copy() for k in data.files}
    pairs = [tuple(p) for p in saved['candidate_pairs'].tolist()]
    ai = np.array([labels['attrs'].index(a) for a,o in pairs])
    oi = np.array([labels['objs'].index(o) for a,o in pairs])
    joint = np.empty((len(saved['image_ids']),len(labels['objs']),len(labels['attrs'])),dtype=np.float64)
    joint[:,oi,ai] = saved['log_probabilities']
    obj = np.logaddexp.reduce(joint,axis=-1)
    cond = joint-obj[:,:,None]
    return saved,pairs,ai,oi,joint,obj,cond


def main():
    spec = importlib.util.spec_from_file_location('f_archived_evaluator',ARCHIVE/'vgcpod_v1_roundF/vgcpod/metrics.py')
    evaluator = importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    manifest = read(CACHE/'features_manifest.json');labels = manifest['labels']
    train = set(map(tuple,manifest['pairs']['train']))
    rows,pair_rows,movement,hybrids = [],[],[],[]
    for seed in (3351,3352,3353):
        components = {}
        for phase in ('val','test'):
            components[phase] = {}
            for case in ('E1','F1'):
                saved,pairs,ai,oi,joint,obj,cond = load(case,seed,phase,labels)
                components[phase][case] = (saved,pairs,ai,oi,joint,obj,cond)
                attrs,objs = saved['attribute_truth'],saved['object_truth']
                true = np.arange(len(attrs))
                truth_pairs = [(labels['attrs'][a],labels['objs'][o]) for a,o in zip(attrs,objs)]
                seen_images = np.array([p in train for p in truth_pairs])
                cw = train|set(map(tuple,manifest['pairs'][phase]))
                seen = np.array([p in train for p in pairs])
                valid_unseen = np.array([p in cw and p not in train for p in pairs])
                extra = np.array([p not in cw for p in pairs])
                logp = saved['log_probabilities'].astype(np.float64)
                max_extra = logp[:,extra].max(-1)
                max_valid = logp[:,valid_unseen].max(-1)
                unseen_candidates = np.flatnonzero(~seen)
                pred = unseen_candidates[logp[:,~seen].argmax(-1)]
                ceiling_ok = np.array(pairs)[pred]==np.array(truth_pairs)
                ceiling_ok = ceiling_ok.all(-1)
                pred_extra = extra[pred]
                support = np.exp(logp[:,seen]).sum(-1)
                actual = joint[true,objs,attrs]
                conditional = cond[true,objs]
                object_correct = obj.argmax(-1)==objs
                for group in ('all','seen','unseen'):
                    mask = np.ones(len(attrs),dtype=bool) if group=='all' else seen_images if group=='seen' else ~seen_images
                    rows.append(dict(case=case,seed=seed,phase=phase,group=group,images=int(mask.sum()),
                        object_accuracy=float(100*object_correct[mask].mean()),
                        object_nll=float(-obj[true,objs][mask].mean()),
                        conditional_accuracy=float(100*(conditional.argmax(-1)==attrs)[mask].mean()),
                        conditional_nll=float(-conditional[true,attrs][mask].mean()),
                        unseen_rank_ceiling=float(100*ceiling_ok[mask].mean()),
                        extra_dominates_valid=float(100*(max_extra>max_valid)[mask].mean()),
                        seen_support_pct=float(100*support[mask].mean()),
                        extra_support_pct=float(100*np.exp(logp[:,extra]).sum(-1)[mask].mean()),
                        true_pair_probability_pct=float(100*np.exp(actual[mask]).mean())))
                for pair in sorted(set(truth_pairs)):
                    mask = np.array([p==pair for p in truth_pairs])
                    pair_rows.append(dict(case=case,seed=seed,phase=phase,attribute=pair[0],object=pair[1],
                        seen=pair in train,images=int(mask.sum()),
                        conditional_correct=int((conditional.argmax(-1)==attrs)[mask].sum()),
                        conditional_nll=float(-conditional[true,attrs][mask].mean()),
                        seen_support_pct=float(100*support[mask].mean()),
                        extra_support_pct=float(100*np.exp(logp[:,extra]).sum(-1)[mask].mean()),
                        unseen_ceiling_correct=int(ceiling_ok[mask].sum())))
                components[phase][case] += (ceiling_ok,pred_extra,pred)
            e,f = components[phase]['E1'],components[phase]['F1']
            assert e[0]['image_ids'].tolist()==f[0]['image_ids'].tolist() and e[1]==f[1]
            attrs,objs=e[0]['attribute_truth'],e[0]['object_truth']
            seen_images=np.array([(labels['attrs'][a],labels['objs'][o]) in train for a,o in zip(attrs,objs)])
            gained=(~e[7])&f[7]&~seen_images;lost=e[7]&~f[7]&~seen_images
            movement.append(dict(seed=seed,phase=phase,unseen_images=int((~seen_images).sum()),
                ceiling_gained=int(gained.sum()),ceiling_lost=int(lost.sum()),
                lost_to_extra=int((lost&f[8]).sum()),
                lost_to_valid_unseen=int((lost&~f[8]).sum()),
                extra_to_valid_count=int((e[8]&~f[8]&~seen_images).sum()),
                valid_to_extra_count=int((~e[8]&f[8]&~seen_images).sum())))
        # Swap the two saved probability factors at the same seed and selected epoch.
        # Recalibrate each condition on its own VAL before evaluating TEST HM.
        for name,obj_case,cond_case in [('Eobj_Fcond','E1','F1'),('Fobj_Econd','F1','E1')]:
            biases={}
            for phase in ('val','test'):
                original=components[phase][obj_case]
                other=components[phase][cond_case]
                joint=original[5][:,:,None]+other[6]
                attrs,objs=original[0]['attribute_truth'],original[0]['object_truth']
                for world in ('cw','ow'):
                    metrics,_,_,_=evaluator.evaluate_scores(joint,other[6],attrs,objs,labels,
                        manifest['pairs'],phase,world,fixed_bias=biases.get(world))
                    if phase=='val':biases[world]=metrics['bias_native']
                    hybrids.append(dict(case=name,seed=seed,setting=phase+'_'+world,
                        **{k:v for k,v in metrics.items() if isinstance(v,(float,int))}))
    result=dict(diagnostics=rows,per_pair=pair_rows,ceiling_movements=movement,hybrid_metrics=hybrids,
        hybrid_rule='Saved E1/F1 native checkpoints, same seed; each hybrid selects its own VAL bias; no training')
    (OUT/'mechanism.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,key in [('ranking_diagnostics.csv','diagnostics'),('pair_probability_diagnostics.csv','per_pair'),
                     ('unseen_ceiling_movements.csv','ceiling_movements'),('head_swap_metrics.csv','hybrid_metrics')]:
        save_csv(name,result[key])
    for case in ('E1','F1'):
        group=[r for r in rows if r['case']==case and r['phase']=='test' and r['group']=='unseen']
        print(case,{k:float(np.mean([r[k] for r in group])) for k in ('object_accuracy','object_nll',
            'conditional_accuracy','conditional_nll','unseen_rank_ceiling','extra_dominates_valid','extra_support_pct')})
    for case in ('Eobj_Fcond','Fobj_Econd'):
        print(case,{s:float(np.mean([r['AUC_exact'] for r in hybrids if r['case']==case and r['setting']==s]))
            for s in ('val_cw','val_ow','test_cw','test_ow')})
    print('Ceiling movements',movement)


if __name__=='__main__':
    main()
