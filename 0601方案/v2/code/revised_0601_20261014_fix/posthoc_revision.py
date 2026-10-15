"""No training. Bias frozen on VAL, structural errors and prior readout diagnostics."""
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from common import read, write, sha, verify_files
from revision_math import make_prior


def logsumexp(x, axis):
    m=np.max(x,axis=axis,keepdims=True)
    return (m+np.log(np.exp(x-m).sum(axis=axis,keepdims=True))).squeeze(axis)


def remove_prior(grid, prior, weight):
    # log P(o) is exactly the row marginal of the normalized full OW grid.
    lo=logsumexp(grid,2)
    adjusted=grid-weight*prior[None]
    return lo[:,:,None]+adjusted-logsumexp(adjusted,2)[:,:,None]


def evaluate_bias(scores, truth_pair, seen_candidates, truth_seen, bias):
    pred=np.argmax(scores + bias*(~seen_candidates)[None],axis=1)
    correct=pred==truth_pair
    seen=float(correct[truth_seen].mean()) if truth_seen.any() else None
    unseen=float(correct[~truth_seen].mean()) if (~truth_seen).any() else None
    hm=2*seen*unseen/(seen+unseen) if seen is not None and unseen is not None and seen+unseen>0 else 0.0
    return dict(bias=float(bias),accuracy=float(correct.mean()),seen=seen,unseen=unseen,hm=hm),pred


def choose_bias(scores, truth_pair, seen_candidates, truth_seen):
    # Fixed predeclared grid. Ties prefer smaller magnitude, then lower bias.
    values=[]
    for bias in np.linspace(-5,5,201):
        metric,_=evaluate_bias(scores,truth_pair,seen_candidates,truth_seen,bias)
        values.append(metric)
    best=max(values,key=lambda v:(v['hm'],-abs(v['bias']),-v['bias']))
    return best,values


def load_eval(folder, train_pairs, allowed_pairs=None):
    m=read(folder/'metrics.json')
    pairs=[tuple(p) for p in read(folder/'candidates.json')]
    with np.load(str(folder/'labels.npz')) as archive:
        labels={name:archive[name] for name in archive.files}
    scores=np.array(np.memmap(str(folder/'scores.dat'),mode='r',dtype=np.float32,shape=tuple(m['score_shape'])))
    seen=np.array([p in train_pairs for p in pairs])
    truth=labels['pair']
    if allowed_pairs is not None:
        allowed_pairs={tuple(p) for p in allowed_pairs}
        if not allowed_pairs.issubset(set(pairs)):
            raise ValueError('CW export is missing legal phase candidates')
        legal=np.array([p in allowed_pairs for p in pairs],dtype=bool)
        if not legal.any() or not legal[truth].all():
            raise ValueError('CW truth is outside the legal phase candidates')
        # Preserve saved column/label indices and never mutate the score file.
        scores[:,~legal]=-np.inf
    return m,pairs,labels,scores,seen,seen[truth]


def analyze_run(run, protocol, splits):
    cfg=yaml.safe_load((run/'config_requested.yml').read_text(encoding='utf-8'))
    train_pairs={tuple(p) for p in splits['train']}
    result=dict(policy='Bias grid[-5,5],step0.05; maximize VAL HM; freeze separately CW/OW; TEST never tunes bias',settings={})
    for setting in ('cw','ow'):
        val_allowed=train_pairs|{tuple(p) for p in splits['val']} if setting=='cw' else None
        test_allowed=train_pairs|{tuple(p) for p in splits['test']} if setting=='cw' else None
        val=load_eval(run/('eval_val_'+setting),train_pairs,val_allowed)
        test=load_eval(run/('eval_test_'+setting),train_pairs,test_allowed)
        best,curve=choose_bias(val[3],val[2]['pair'],val[4],val[5])
        metric,pred=evaluate_bias(test[3],test[2]['pair'],test[4],test[5],best['bias'])
        labels=read(Path(cfg['MODEL']['kcc_cache'])/'labels.json')
        pa=np.array([labels['attrs'].index(p[0]) for p in test[1]])
        po=np.array([labels['objs'].index(p[1]) for p in test[1]])
        aok=pa[pred]==test[2]['attr']; ook=po[pred]==test[2]['obj']
        counts={'both_correct':int((aok&ook).sum()),'object_only_error':int((aok&~ook).sum()),
                'attribute_only_error':int((~aok&ook).sum()),'both_error':int((~aok&~ook).sum())}
        item=dict(VAL_selected=best,VAL_bias_curve=curve,TEST_at_VAL_bias=metric,
                  error_counts=counts,zero_bias_TEST=evaluate_bias(test[3],test[2]['pair'],test[4],test[5],0)[0])
        if test[4].any() and (~test[4]).any():
            gap=test[3][:,test[4]].max(1)-test[3][:,~test[4]].max(1)
            item['bias0_seen_minus_unseen_max_score_gap']={kind:dict(n=int(mask.sum()),mean=float(gap[mask].mean()),
                    median=float(np.median(gap[mask]))) for kind,mask in (('truth_seen',test[5]),('truth_unseen',~test[5])) if mask.any()}
        if setting=='ow' and cfg['MODEL'].get('rev_enabled',False):
            O,A=len(labels['objs']),len(labels['attrs'])
            if len(test[1])!=O*A or len(set(test[1]))!=O*A: raise ValueError('Prior removal requires complete cartesian OW')
            grid=np.empty((len(test[3]),O,A),dtype=np.float32)
            grid[:,po,pa]=test[3]
            if not np.allclose(np.exp(grid).sum((1,2)),1,atol=3e-5): raise ValueError('Scores not normalized full joint')
            obj_pred=logsumexp(grid,2).argmax(1)
            row=grid[np.arange(len(grid)),test[2]['obj']]
            item['object_marginal_accuracy']=float((obj_pred==test[2]['obj']).mean())
            item['attribute_accuracy_given_GT_object_ORACLE']=float((row.argmax(1)==test[2]['attr']).mean())
            if cfg['MODEL']['kcc_calib']:
                bank=torch.load(Path(cfg['MODEL']['kcc_cache'])/'revision_bank.pt',weights_only=True)
                prior=make_prior(bank['K'],cfg['MODEL']['rev_prior_mode'],floor=cfg['MODEL']['rev_prior_floor']).numpy()
                weight=test[0]['effective_parameters']['lambda_effective']
                off=remove_prior(grid,prior,weight)[:,po,pa]
                off_metric,off_pred=evaluate_bias(off,test[2]['pair'],test[4],test[5],best['bias'])
                with_ok=pred==test[2]['pair']; off_ok=off_pred==test[2]['pair']
                item['prior_readout_diagnostic']=dict(prior_off_same_bias=off_metric,
                    helped=int((with_ok&~off_ok).sum()),harmed=int((~with_ok&off_ok).sum()),
                    neutral=int((with_ok==off_ok).sum()),
                    caveat='Same trained weights and same VAL-frozen bias; no retraining or causal knowledge-error label')
        images=read(run/('eval_test_'+setting)/'image_ids.json')
        item['failure_review_examples']=[dict(image_id=images[i],truth=list(test[1][int(test[2]['pair'][i])]),
                   predicted=list(test[1][int(pred[i])]),human_semantic_cause='UNREVIEWED')
                   for i in np.where(pred!=test[2]['pair'])[0][:20]]
        result['settings'][setting]=item
    result['run']=run.name
    result['protocol_sha256']=sha(Path(protocol['work'])/'protocol.json') if 'work' in protocol else None
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    a=p.parse_args()
    protocol=read(a.work/'protocol.json')
    verify_files(protocol['frozen_inputs'])
    for job in protocol['jobs']:
        if sha(job['cfg'])!=job['cfg_sha256']:raise ValueError('Frozen config changed: '+job['cfg'])
    splits=read(a.work/'checks/split_pairs.json')
    results=[]
    for job in protocol['jobs']:
        run=Path(job['run'])
        if (run/'status.json').exists() and read(run/'status.json')['state']=='complete':
            result=analyze_run(run,protocol,splits)
            write(run/'posthoc_analysis.json',result)
            results.append(result)
    write(a.work/'summary/posthoc_analysis.json',dict(protocol_sha256=sha(a.work/'protocol.json'),runs=results,
          warning='Statistical error types are not semantic causes; qualitative examples require human inspection'))
    print('POSTHOC_COMPLETE actual_completed_runs='+str(len(results)))
