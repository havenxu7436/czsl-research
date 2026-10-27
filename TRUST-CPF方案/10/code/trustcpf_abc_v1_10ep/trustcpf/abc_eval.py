"""Q4/Q5 curves and formal scores, with common random masks and exact deduplication."""
import hashlib,math,time
from pathlib import Path
import numpy as np
import torch
from .common import load_torch,save_torch,read_json,write_json,sha
from . import abc,abc_formal,align,compact,competition as cmp,competition_training as ct,state_eval
from .model import candidate_logprob
from .metrics import native_curve,exact_curve

def identity(pi):
    p=pi.double().cpu()
    if len(p)==1:p=p.expand(2,-1)
    return hashlib.sha256(p.numpy().tobytes()).hexdigest()

@torch.no_grad()
def decision(args,fold,selected=False,partition='evaluate'):
    work,cfg,manifest,text,bank=align.context(args);root=work.parent
    models=load_torch(root/'utility/models.pt');groups=read_json(root/'groups.json')['groups']
    effects=load_torch(work/('effects/fold%d.pt'%fold))
    selection=read_json(root/'utility/selection.json')['eta'] if selected else None
    policies=abc.curves(models,load_torch(work/('context/fold%d.pt'%fold)),
        load_torch(work/('responses/fold%d_repeat.pt'%fold)),effects,cfg,selection)
    cp=work/('teachers/%d/best.pt'%fold);model=align.restore(text,bank,cfg,args.device,cp)
    split=read_json(work/('splits/fold%d.json'%fold))
    if partition=='measure':
        policies={k:p for k,p in policies.items() if p['permutation_seed'] is None and
            (k=='uniform' or (p['method'] in abc.CURVE_METHODS and k!='A_selected'))}
    base,candidates,truth,seen=ct.prepared(args,cfg,manifest,model,split,partition)
    routes=abc.route(base['obj'],groups);sample_seen=seen[truth];rows=[]
    dest=work/('decision/fold%d%s'%(fold,'_measure' if partition=='measure' else ''));dest.mkdir(parents=True,exist_ok=True)
    save_torch(dest/'policies.pt',policies)
    for gs in cfg['audit_eval_seeds']:
        gen=torch.Generator().manual_seed(gs)
        uniforms=torch.rand(cfg['graphs'],len(bank['types']),generator=gen).double()
        values={};cache={}
        for name,p in policies.items():
            code=identity(p['pi']);file=dest/('bank%d'%gs)/(code+'.pt')
            if code in cache:v=cache[code]
            elif file.exists():v=load_torch(file)
            else:
                masks=(uniforms[None]<p['pi'][:,None]).float().to(args.device)
                chunks=[]
                for start in range(0,len(truth),cfg['eval_batch_size']):
                    end=start+cfg['eval_batch_size']
                    stack=abc_formal.routed_stack(model,{k:x[start:end] for k,x in base.items()},masks,routes[start:end])
                    chunks.append(torch.stack([z.flatten(1)[:,candidates].double().log_softmax(-1) for z in stack]))
                logs=torch.cat(chunks,1)
                sufficient=torch.stack([cmp.sufficient(z,truth,seen) for z in logs])
                singles=[cmp.diagnostics(st,sample_seen,cfg['competition_temperature'],cfg['risk_pair_chunk']) for st in sufficient]
                mixed=logs.logsumexp(0)-math.log(cfg['graphs'])
                metrics=cmp.diagnostics(cmp.sufficient(mixed,truth,seen),sample_seen,cfg['competition_temperature'],cfg['risk_pair_chunk'])
                metrics['AUC_native']=native_curve(mixed.cpu().numpy(),truth.cpu().numpy(),seen.cpu().numpy())['AUC']
                np.testing.assert_allclose(exact_curve(mixed.cpu().numpy(),truth.cpu().numpy(),seen.cpu().numpy())['AUC'],metrics['AUC_exact'],atol=1e-8)
                v=dict(mixture=metrics,single={k:float(np.mean([d[k] for d in singles])) for k in singles[0]},
                    mixture_logprobs=mixed.cpu(),single_statistics=sufficient.cpu(),masks=masks.cpu(),
                    routes=routes.cpu(),truth=truth.cpu(),seen=seen.cpu(),pi=p['pi'],
                    checkpoint_sha256=sha(cp),sample_partition=partition,
                    image_ids=split[partition]['seen']+split[partition]['unseen'])
                save_torch(file,v)
            cache[code]=v;values[name]=v
            row=dict(seed=args.seed,fold=fold,sample_partition=partition,policy=name,method=p['method'],eta=p['eta'],permutation_seed=p['permutation_seed'],
                     bank=gs,result_file=str(file.relative_to(work)))
            for mode in ('single','mixture'):
                row.update({mode+'_'+k:value for k,value in v[mode].items()})
                row[mode+'_NLL_gain']=values['uniform'][mode]['unll']-v[mode]['unll']
                row[mode+'_AUC_gain']=v[mode]['AUC_exact']-values['uniform'][mode]['AUC_exact']
                targets=effects[mode]
                if len(p['pi'])==1:
                    proxy=targets['target']@(p['pi'][0]-.5)
                else:
                    proxy=(targets['context']*(p['pi']-.5)*torch.as_tensor(targets['weights'])[:,None]).sum()
                row[mode+'_linear_gain']=float(proxy)
                row[mode+'_proxy_minus_actual']=float(proxy)-row[mode+'_NLL_gain']
            rows.append(row)
        print('DECISION_BANK',args.seed,'Q'+str(fold),partition,gs,'policies',len(policies),'unique',len(cache),flush=True)
    write_json(dest/'audit.json',dict(rows=rows,graphs=cfg['graphs'],policies=len(policies),fold=fold,
        sample_partition=partition,images={k:len(v) for k,v in split[partition].items()}))

def make_policies(args):
    work,cfg,manifest,text,bank=align.context(args);root=work.parent
    models=load_torch(root/'utility/models.pt');eta=read_json(root/'utility/selection.json')['eta']
    response=load_torch(work/'context/uniform_best.pt')
    save_torch(work/'policies.pt',abc.formal(models,response,cfg,eta))

@torch.no_grad()
def recognition(args):
    work,cfg,manifest,text,bank=align.context(args);root=work.parent
    cp=work/'runs/uniform/best.pt';model=align.restore(text,bank,cfg,args.device,cp)
    prepared={p:state_eval.prepare_phase(model,args.cache,p,cfg) for p in ('val','test')}
    policies=load_torch(work/'policies.pt');groups=read_json(root/'groups.json')['groups']
    epoch=load_torch(cp)['epoch']
    for gs in cfg['eval_graph_seeds']:
        evcfg=dict(cfg,graph_seed=gs,variant='uniform',readout='probability')
        for name,p in policies.items():
            code=identity(p['pi']);dest=work/'recognition'/('bank%d'%gs)/'objects'/code
            file=dest/'metrics.json'
            if file.exists():metrics=read_json(file)
            else:
                align.synchronize(args.device);tic=time.perf_counter()
                pi=p['pi'].float()
                val=abc_formal.evaluate_routed(model,args.cache,evcfg,manifest,pi,'val',groups,save=dest,prepared=prepared['val'])
                test=abc_formal.evaluate_routed(model,args.cache,evcfg,manifest,pi,'test',groups,calibration=val,save=dest,prepared=prepared['test'])
                align.synchronize(args.device)
                deployment=[]
                for g,pg in enumerate(pi):
                    for t in range(3):
                        q=pg[bank['types']==t].double()
                        deployment.append(dict(context=g,type=t,expected_edges=float(q.sum()),probability_std=float(q.std(unbiased=False)),
                            kl=float((q*torch.log(2*q)+(1-q)*torch.log(2*(1-q))).nan_to_num().sum())))
                metrics=dict(val=val,test=test,epoch=epoch,checkpoint_sha256=sha(cp),
                             deployment=deployment,seconds=time.perf_counter()-tic)
                write_json(file,metrics)
            record=dict(seed=args.seed,policy=name,method=p['method'],permutation_seed=p['permutation_seed'],
                eta=p['eta'],bank=gs,graphs=cfg['graphs'],result_directory=str(dest.relative_to(work)),**metrics)
            write_json(work/'recognition'/('bank%d'%gs)/'policies'/(name+'.json'),record)
            print('FORMAL_SCORE',args.seed,name,gs,'CW',metrics['test']['cw']['AUC_native'],'OW',metrics['test']['ow']['AUC_native'],flush=True)
