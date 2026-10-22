import csv, gc, itertools, json, sys, re
from pathlib import Path
import numpy as np
import torch

BASE=Path(r'C:\Users\llenovo\Documents\ChatGPT\USG')
PROJECT=BASE/'server_results_20261022/trustcpf_clip_v1'
WORK=PROJECT/'work'
OUT=BASE/'analysis_TRUST_CPF_v1_20261022'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(PROJECT))
from trustcpf.utility import regression_metrics, UtilityNet, ensemble_moments
torch.set_num_threads(2)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def stats(x):
    a=np.asarray(x,dtype=float)
    if not len(a):return dict(mean=None,sd=None,values=[])
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0.,values=a.tolist())
def corr(a,b):
    a=np.asarray(a);b=np.asarray(b);return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else 0.
def rank(x):
    from trustcpf.utility import average_rank
    return average_rank(np.asarray(x))
def rho(a,b):return corr(rank(a),rank(b))

def main():
    variants=['r0','capacity','dense','drop','free','mean','robust']
    seeds=[3351,3352,3353]; result={}; metrics={}
    for seed in seeds:
        metrics[seed]={v:read(WORK/f'seed{seed}/runs/{v}/metrics.json') for v in variants}
    result['main']={v:{w:{k:stats([metrics[s][v]['test'][w][k] for s in seeds])
                  for k in ['AUC_native','AUC_exact','HM_oracle_native','gt_object_attribute_accuracy','gt_object_attribute_nll','graph_variance']}
                  for w in ['cw','ow']} for v in variants}
    result['calibration']={v:{phase:{w:{k:stats([metrics[s][v][phase][w][section][key] for s in seeds])
          for k,section,key in [('seen','at_val_bias','seen'),('unseen','at_val_bias','unseen'),('HM','at_val_bias','HM'),('NLL','calibrated','NLL'),('ECE','calibrated','ECE'),('NLL_unseen','calibrated','NLL_unseen')]}
          for w in ['cw','ow']} for phase in ['val','test']} for v in variants}
    result['paired']={v:{w:stats([metrics[s]['robust']['test'][w]['AUC_native']-metrics[s][v]['test'][w]['AUC_native'] for s in seeds]) for w in ['cw','ow']} for v in variants if v!='robust'}
    result['errors']={v:{phase:{w:{'object_accuracy':stats([(metrics[s][v][phase][w]['errors']['both_correct']+metrics[s][v][phase][w]['errors']['attribute_wrong_object_correct'])/metrics[s][v][phase][w]['images']*100 for s in seeds]),
        'pair_accuracy':stats([metrics[s][v][phase][w]['errors']['both_correct']/metrics[s][v][phase][w]['images']*100 for s in seeds]),
        'gt_object_attr_accuracy':stats([metrics[s][v][phase][w]['gt_object_attribute_accuracy'] for s in seeds]),
        'val_auc':stats([metrics[s][v][phase][w]['AUC_native'] for s in seeds])} for w in ['cw','ow']} for phase in ['val','test']} for v in variants}
    result['audit']={str(s):read(WORK/f'seed{s}/utility/audit.json') for s in seeds}
    result['ablation']={n:read(WORK/f'{n}3351/utility/audit.json') for n in ['semantic','unmeasured']}
    result['shuffled']=read(WORK/'shuffled3351/runs/shuffled/metrics.json')
    result['shuffled_audit']=read(WORK/'shuffled3351/utility/audit.json')
    result['structure']={}
    for seed in seeds:
        st=read(WORK/f'seed{seed}/structure.json'); cfg=st['parameters']
        entry={'selected':{k:cfg[k] for k in ['budget','lambda_var','lambda_ens','tau_scale','tau_absolute']},'expected_edges':st['expected_edges'],'pi':st['pi'],
               'pi_min':min(st['pi']),'pi_max':max(st['pi']),'pi_sd':float(np.std(st['pi']))}
        for v in ['drop','free','mean','robust']:
            state=load(WORK/f'seed{seed}/runs/{v}/best.pt')
            p=state['pi'].numpy();entry[v]={'sum':float(p.sum()),'mean':float(p.mean()),'sd':float(p.std()),'min':float(p.min()),'max':float(p.max()),'pi':p.tolist()}
            del state
        entry['free_drop_mean_abs_pi']=float(np.abs(np.array(entry['free']['pi'])-entry['drop']['pi']).mean())
        result['structure'][str(seed)]=entry
    # All 600 evaluations: each noise file is one model, type and seed balanced.
    noise=[];coverage={}
    for v in ['dense','drop','mean','robust']:
        for kind,file in [('deletion','noise_delete.json'),('source_mismatch','noise_mismatch.json')]:
            rows=read(WORK/f'seed3351/runs/{v}/{file}');coverage[f'{v}/{kind}']=len(rows)
            for row in rows:
                for w in ['cw','ow']:
                    noise.append(dict(variant=v,kind=kind,noise_seed=row['seed'],type=row['type'],rate=row['rate'],actual_rate=row['actual_rate'],world=w,auc=row['test'][w]['AUC_native'],auc_exact=row['test'][w]['AUC_exact']))
    result['noise_coverage']=coverage
    log_rows=[]
    for line in (PROJECT/'logs/noise3351.log').read_text(encoding='utf-8').splitlines():
        m=re.match(r'^NOISE (\w+) (\d+) (\d+) ([\d.]+) ([\d.eE+-]+)$',line)
        if m:log_rows.append((m[1],int(m[2]),int(m[3]),float(m[4]),float(m[5])))
    recovered=[];log_checks={}
    for v in ['dense','drop','mean','robust']:
        rr=[r for r in log_rows if r[0]==v]
        assert len(rr)==150,(v,len(rr))
        for kind,segment in [('source_mismatch',rr[:75]),('deletion',rr[75:])]:
            lookup={(seed,ty,rate):auc for _,seed,ty,rate,auc in segment}
            existing=[r for r in noise if r['variant']==v and r['kind']==kind and r['world']=='ow']
            error=max(abs(r['auc']-lookup[(r['noise_seed'],r['type'],r['rate'])]) for r in existing)
            log_checks[f'{v}/{kind}']=error
            assert error<1e-8,(v,kind,error)
            for _,seed,ty,rate,auc in segment:
                recovered.append(dict(variant=v,kind=kind,noise_seed=seed,type=ty,rate=rate,world='ow',auc=auc))
    result['noise_log_recovery']={'records':len(log_rows),'log_vs_json_max_errors':log_checks,'recovered_ow_records':len(recovered)}
    result['recovered_ow_rauc']={}
    for v,kind in itertools.product(['dense','drop','mean','robust'],['source_mismatch','deletion']):
        groups={}
        for r in recovered:
            if r['variant']==v and r['kind']==kind:groups.setdefault((r['noise_seed'],r['type']),[]).append(r)
        all_areas=[];typed={}
        for (seed,ty),curve in groups.items():
            curve=sorted(curve,key=lambda r:r['rate']);area=float(np.trapezoid([r['auc'] for r in curve],[r['rate'] for r in curve])/.5)
            all_areas.append(area);typed.setdefault(str(ty),[]).append(area)
        result['recovered_ow_rauc'][f'{v}/{kind}']={'all':stats(all_areas),'by_type':{t:stats(a) for t,a in typed.items()},
            'curve':{str(rate):stats([r['auc'] for r in recovered if r['variant']==v and r['kind']==kind and r['rate']==rate]) for rate in [0.,.1,.2,.3,.5]}}
    ns={}
    for v,kind,w in itertools.product(['dense','drop','mean','robust'],['deletion','source_mismatch'],['cw','ow']):
        rows=[r for r in noise if (r['variant'],r['kind'],r['world'])==(v,kind,w)]
        groups={}
        for r in rows:groups.setdefault((r['noise_seed'],r['type']),[]).append(r)
        raucs=[];r0=[];r5=[];actual={str(t):{} for t in range(3)};typed={}
        for (seed,ty),curve in groups.items():
            curve=sorted(curve,key=lambda r:r['rate']);x=np.array([r['rate'] for r in curve]);y=np.array([r['auc'] for r in curve])
            area=float(np.trapezoid(y,x)/.5);raucs.append(area);r0.append(y[0]);r5.append(y[-1]);typed.setdefault(str(ty),[]).append(area)
        for ty in range(3):
            for rate in [0,.1,.2,.3,.5]:
                rr=[r for r in rows if r['type']==ty and r['rate']==rate]
                actual[str(ty)][str(rate)]=stats([r['actual_rate'] for r in rr])
        ns[f'{v}/{kind}/{w}']={'complete_grid':len(rows)==75,'rauc':stats(raucs),'r0':stats(r0),'r5':stats(r5),'auc_drop_r5':float(np.mean(r0)-np.mean(r5)),
            'by_type_rauc':{t:stats(a) for t,a in typed.items()},'actual_rates':actual,
            'rate_curve':{str(rate):stats([r['auc'] for r in rows if r['rate']==rate]) for rate in [0,.1,.2,.3,.5]}}
    result['noise']=ns
    result['noise_zero_baseline_max_error']={v:max(abs(r['auc']-metrics[3351][v]['test'][r['world']]['AUC_native']) for r in noise if r['variant']==v and r['rate']==0) for v in ['dense','drop','mean','robust']}
    # Measurement summaries: fold/task effects versus Monte Carlo background noise.
    measure={}; folded_means={}; cost={}
    for s in seeds+[0]:
        name=f'seed{s}' if s else 'shuffled3351'; folded_means[name]=[]; mc=[]; seconds=[];forwards=[]
        for fold in range(1,6):
            p=load(WORK/name/f'measurements/fold{fold}.pt')
            y=p['y'].numpy();nu=p['noise'].numpy();edges=p['edge'].numpy()
            means=np.array([y[edges==e].mean() for e in range(96)])
            folded_means[name].append(means)
            entry={'observations':len(y),'y_mean':float(y.mean()),'y_sd':float(y.std()),'positive_fraction':float((y>0).mean()),
                   'reliable_fraction':float((np.abs(y)>1.96*np.sqrt(nu)).mean()),'noise_to_y_variance':float(nu.mean()/y.var()),
                   'wall_seconds':p['wall_seconds'],'head_forwards':p['head_forwards'],
                   'types':{str(t):{'mean':float(y[(edges>=a)&(edges<b)].mean()),'sd':float(y[(edges>=a)&(edges<b)].std())} for t,a,b in [(0,0,12),(1,12,48),(2,48,96)]}}
            within=float(np.mean([y[edges==e].var() for e in range(96)]))
            entry['within_edge_variance']=within;entry['mc_to_within_edge_variance']=float(nu.mean()/within)
            measure[f'{name}/fold{fold}']=entry;seconds.append(p['wall_seconds']);forwards.append(p['head_forwards'])
            if s==3351 and fold==5: torch.save({k:p[k] for k in ['x','y','noise','edge']},OUT/'q5_compact.pt')
            del p;gc.collect()
        teacher_time=sum(sum(r['seconds'] for r in read(p)) for p in (WORK/name/'teachers').glob('*/history.json'))
        ref_time=sum(sum(r['seconds'] for r in read(p)) for p in (WORK/name).glob('reference*/history.json'))
        adapt_time=sum(sum(r['seconds'] for r in read(p)) for p in (WORK/name/'runs').glob('*/history.json'))
        util_time=sum(sum(read(p).get('model_seconds',[])) for p in (WORK/name).glob('utility*/selection.json'))
        cost[name]={'measure_seconds':sum(seconds),'teacher_seconds':teacher_time,'reference_seconds':ref_time,'adapt_seconds':adapt_time,'utility_seconds':util_time,'recorded_seconds_sum':sum(seconds)+teacher_time+ref_time+adapt_time+util_time,'head_forwards':sum(forwards)}
    result['measurements']=measure;result['cost']=cost
    result['fold_edge_rank_correlation']={name:{f'{i+1}-{j+1}':rho(means[i],means[j]) for i in range(5) for j in range(i+1,5)} for name,means in folded_means.items()}
    result['fit_audit_edge_correlation']={name:{'pearson':corr(np.mean(means[:3],0),means[4]),'spearman':rho(np.mean(means[:3],0),means[4]),
        'sign_agreement':float((np.sign(np.mean(means[:3],0))==np.sign(means[4])).mean())} for name,means in folded_means.items()}
    # Equal Q5 held-out edge subset, including baselines evaluated on those same rows.
    q=load(OUT/'q5_compact.pt'); held=read(WORK/'unmeasured3351/utility/selection.json')['heldout_edges']
    mask=np.isin(q['edge'].numpy(),held); dims=768
    ty=torch.where(q['edge']<12,0,torch.where(q['edge']<48,1,2))
    comparable={}
    for folder in [WORK/'seed3351/utility',WORK/'semantic3351/utility',WORK/'unmeasured3351/utility']:
        nets=[]
        for p in sorted(folder.glob('model*.pt')):
            state=load(p);net=UtilityNet(state['dimension'],state['model']['center'],state['model']['spread']);net.load_state_dict(state['model']);net.eval();nets.append(net)
        x=q['x'].float()
        if state['cfg'].get('utility_input')=='semantic':x[:,4*dims+7:]=0
        mm=[];vv=[]
        with torch.no_grad():
            for start in range(0,len(x),256):
                m,v,e=ensemble_moments(nets,x[start:start+256],state['scales'],ty[start:start+256]);mm.append(m);vv.append(v+e)
        m=torch.cat(mm).numpy();v=torch.cat(vv).numpy()
        comparable[folder.parent.name]={'all':regression_metrics(m,v,q['y'],q['noise']),'held_subset':regression_metrics(m[mask],v[mask],q['y'][mask],q['noise'][mask])}
    for remove in [False,True]:
        yy=[];nn=[];ee=[]
        for fold in [1,2,3]:
            p=load(WORK/f'seed3351/measurements/fold{fold}.pt'); keep=~np.isin(p['edge'].numpy(),held) if remove else np.ones(len(p['y']),bool)
            yy.extend(p['y'].numpy()[keep]);nn.extend(p['noise'].numpy()[keep]);ee.extend(p['edge'].numpy()[keep]);del p
        yy=np.asarray(yy);nn=np.asarray(nn);ee=np.asarray(ee);tt=np.where(ee<12,0,np.where(ee<48,1,2));qt=ty.numpy()
        tm=np.array([yy[tt==t].mean() for t in range(3)]);tv=np.array([max(yy[tt==t].var()-nn[tt==t].mean(),1e-8) for t in range(3)])
        comparable['type_constant_heldout' if remove else 'type_constant_full']={'held_subset':regression_metrics(tm[qt[mask]],tv[qt[mask]],q['y'][mask],q['noise'][mask])}
    result['same_q5_subset_comparison']=comparable
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (OUT/'noise_long.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(noise[0]));writer.writeheader();writer.writerows(noise)
    with (OUT/'noise_ow_recovered.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(recovered[0]));writer.writeheader();writer.writerows(recovered)
    print(json.dumps({'main':{v:[result['main'][v][w]['AUC_native'] for w in ['cw','ow']] for v in variants},'noise_coverage':coverage,
        'noise_rauc':{k:{m:v[m] for m in ['rauc','auc_drop_r5']} for k,v in ns.items()},'audits':{s:{k:v[k] for k in ['all','type_constant','edge_lookup']} for s,v in result['audit'].items()},
        'held_subset':comparable,'shuffled_test':result['shuffled']['test'],'fit_audit_edge_correlation':result['fit_audit_edge_correlation'],'cost':cost},ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()
