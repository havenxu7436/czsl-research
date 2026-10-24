"""World-specific probability mixtures, shared prediction pass, and replayable outputs."""
import math
from pathlib import Path
import numpy as np
import torch
from .common import write_json,read_json
from .model import candidate_logprob


@torch.no_grad()
def evaluate_model(model,cache,cfg,manifest,pi,phase,calibration=None,save=None):
    from .engine import batches,candidate_spec,calibration_metrics
    from .metrics import native_curve,exact_curve
    model.eval();device=next(model.parameters()).device
    specs={w:candidate_spec(manifest,phase,w) for w in ('cw','ow')}
    logs={w:[] for w in specs};variances={w:[] for w in specs};fulls=[];objects=[];attrs=[];objs=[]
    mode=cfg.get('readout','probability');pi=pi.to(device)
    gen=torch.Generator(device=device).manual_seed(cfg['graph_seed'])
    masks=[pi if mode=='mean_graph' else (torch.rand(pi.shape,generator=gen,device=device)<pi).float() for _ in range(cfg['graphs'])]
    for data in batches(cache,phase,cfg):
        base=model.base(*[v.to(device) for v in data[:3]])
        stack=torch.stack([model.score(base,e,cfg.get('variant')=='capacity')[0] for e in masks])
        full=stack.mean(0) if mode in ('mean_graph','mean_logit') else stack.logsumexp(0)-math.log(len(masks))
        fulls.append(full.cpu());objects.append(base['obj'].cpu());attrs.append(data[3]);objs.append(data[4])
        for w,(_,ids,_,_) in specs.items():
            ps=torch.stack([candidate_logprob(j,ids.to(device)) for j in stack])
            lp=candidate_logprob(stack.mean(0),ids.to(device)) if mode in ('mean_graph','mean_logit') else ps.logsumexp(0)-math.log(len(masks))
            logs[w].append(lp.cpu());variances[w].append((ps.exp()-ps.exp().mean(0)).square().sum(-1).mean(0).cpu())
    full=torch.cat(fulls);obj=torch.cat(objects);a=torch.cat(attrs);o=torch.cat(objs);row=torch.arange(len(a))
    cond=full[row,o].log_softmax(-1);margin=cond[row,a].clone();wrong=cond.clone();wrong[row,a]=-torch.inf;margin-=wrong.max(-1).values
    labels=manifest['labels'];train=set(map(tuple,manifest['pairs']['train']))
    sm=np.array([(labels['attrs'][aa],labels['objs'][oo]) in train for aa,oo in zip(a,o)])
    outputs={};arrays={'joint':full.numpy(),'obj_base':obj.numpy(),'attrs':a.numpy(),'objs':o.numpy()}
    for w,(candidates,ids,seen,index) in specs.items():
        z=torch.cat(logs[w]).numpy();arrays[w+'_logprob']=z
        truth=np.array([index[(labels['attrs'][aa],labels['objs'][oo])] for aa,oo in zip(a,o)])
        arrays[w+'_candidate_ids']=ids.numpy();arrays[w+'_truth']=truth;arrays[w+'_seen']=seen
        native=native_curve(z,truth,seen);exact=exact_curve(z,truth,seen)
        bias=exact['best_bias'] if calibration is None else calibration[w]['calibrated']['bias']
        temp=min(np.exp(np.linspace(math.log(.1),math.log(10),41)),key=lambda t:calibration_metrics(z,truth,seen,bias,t)['NLL']) if calibration is None else calibration[w]['calibrated']['temperature']
        selected=ids[torch.as_tensor(z).topk(1,dim=1).indices[:,0]];ac=selected%len(labels['attrs'])==a;oc=selected//len(labels['attrs'])==o
        m=dict(AUC_native=native['AUC'],AUC_exact=exact['AUC'],HM_oracle_native=native['best_hm'],HM_oracle_exact=exact['best_hm'],
            unbiased=calibration_metrics(z,truth,seen,0,1),at_val_bias=calibration_metrics(z,truth,seen,bias,1),
            calibrated=calibration_metrics(z,truth,seen,bias,temp),graph_variance=float(torch.cat(variances[w]).mean()),
            candidates=len(candidates),images=len(a),gt_object_attribute_accuracy=float((cond.argmax(-1)==a).float().mean()*100),
            gt_object_attribute_nll=float(-cond[row,a].mean()),object_base_accuracy=float((obj.argmax(-1)==o).float().mean()*100),
            errors=dict(both_correct=int((ac&oc).sum()),attribute_wrong_object_correct=int((~ac&oc).sum()),
                        attribute_correct_object_wrong=int((ac&~oc).sum()),both_wrong=int((~ac&~oc).sum())))
        for name,mask in [('seen',sm),('unseen',~sm)]:
            m['gt_object_attr_acc_'+name]=float((cond.argmax(-1)[mask]==a[mask]).float().mean()*100)
            m['gt_object_margin_'+name]=float(margin[mask].mean())
        outputs[w]=m
    if save is not None:
        save=Path(save);save.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(save/(phase+'_predictions.npz'),**arrays)
        write_json(save/(phase+'_records.json'),read_json(Path(cache)/(phase+'_records.json')))
        write_json(save/'prediction_axes.json',labels)
    return outputs
