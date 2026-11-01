from pathlib import Path
import json,hashlib,csv,sys,statistics
import numpy as np
import torch
import argparse
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--dataset',required=True);a=p.parse_args()
ROOT=a.out.parent;SRC=a.source;OUT=a.out;OUT.mkdir(exist_ok=True,parents=True);DATASET=a.dataset
sys.path.insert(0,str(SRC));from trustg0 import utility

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def meanstd(x):
 x=np.asarray(x);return dict(mean=float(x.mean()),std=float(x.std(ddof=1)) if len(x)>1 else None,values=x.tolist())

def compute(s,bias=None):
 y=s['truth'];seen=s['is_seen'].astype(bool);cs=(s['seen_pred']==y)&seen;cu=(s['unseen_pred']==y)&~seen
 d=s['seen_score'].astype(float)-s['unseen_score'].astype(float)
 unique,inverse=np.unique(d,return_inverse=True);lost=np.bincount(inverse,weights=cs,minlength=len(unique));gained=np.bincount(inverse,weights=cu,minlength=len(unique))
 ss=np.r_[cs.sum(),cs.sum()-lost.cumsum()]/seen.sum();uu=np.r_[0,gained.cumsum()]/(~seen).sum()
 hh=np.divide(2*ss*uu,ss+uu,out=np.zeros_like(ss),where=ss+uu>0)
 trap=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
 bs=np.r_[np.sort(d[cu]-1e-4)[::max(cu.sum()//20,1)],1000.];sn=[];un=[]
 for b in bs:
  hit=np.where(d<b,s['unseen_pred'],s['seen_pred'])==y;sn.append(hit[seen].mean());un.append(hit[~seen].mean())
 sn=np.array(sn);un=np.array(un);hn=np.divide(2*sn*un,sn+un,out=np.zeros_like(sn),where=sn+un>0)
 r=dict(AUC_native=float(trap(sn,un)*100),AUC_exact=float(trap(ss,uu)*100),HM_native=float(hn.max()*100),HM_exact=float(hh.max()*100),Seen=float(ss.max()*100),Unseen=float(uu.max()*100),best_bias=float(bs[hn.argmax()]),pair_nll=float(-s['truth_logprob'].mean()))
 hit=s['pred']==y;conf=s['confidence'];ece=0.
 for i in range(15):
  use=(conf>=i/15)&((conf<(i+1)/15) if i<14 else conf<=1)
  if use.any():ece+=use.mean()*abs(hit[use].mean()-conf[use].mean())
 r.update(ECE=float(ece),pair_accuracy=float(hit.mean()*100))
 if bias is not None:
  hit=np.where(s['unseen_score']+bias>s['seen_score'],s['unseen_pred'],s['seen_pred'])==y;s0=hit[seen].mean();u0=hit[~seen].mean()
  r.update(HM_at_val_bias=float(200*s0*u0/(s0+u0)) if s0+u0 else 0.,seen_at_val_bias=float(s0*100),unseen_at_val_bias=float(u0*100))
 return r

arc=SRC.parent/(SRC.name+'.tar.gz');assert sha(arc)==(SRC.parent/(arc.name+'.sha256')).read_text().split()[0]
manifest=read(SRC/'snapshot_manifest.json');bad=[n for n,h in manifest.items() if sha(SRC/n)!=h];assert not bad,bad
identity=read(SRC/'work/identity.json');cfg=identity['config'];seeds=cfg['seeds'];ds=SRC/'work'/DATASET;index=read(ds/'index.json');bank=load(ds/'cache/bank.pt');types=bank['types'].numpy()
source_bad=[n for n,h in identity['code'].items() if (SRC/n).exists() and sha(SRC/n)!=h];assert not source_bad,source_bad
rows=[];q5=[];errors={};histories=[];stages=[];policy_stats=[];audit=[];surrogate=[];fits=[];checks=[];target_error=0.;frozen_count=0;budget_error=0.;teacher_ids=[]
npzs=0
for seed in seeds:
 w=ds/('seed%d'%seed)
 for f in sorted((w/'recognition').glob('*/*/metrics.json')):
  mm=read(f)
  for phase in ('val_cw','val_ow','test_cw','test_ow'):
   with np.load(f.parent/(phase+'.npz')) as z:s=dict(z)
   actual=compute(s,mm['val_'+phase.split('_')[1]]['best_bias'] if phase.startswith('test') else None)
   npzs+=1
   ids=s['image_ids'];assert np.array_equal(s['truth'],[index['rows'][int(i)]['pair'] for i in ids]);assert all(index['rows'][int(i)]['phase']==phase.split('_')[0] for i in ids)
   for k,v in actual.items():errors[k]=max(errors.get(k,0),abs(v-mm[phase][k]))
   rows.append(dict(seed=seed,method=f.parent.parent.name,bank=int(f.parent.name),phase=phase,**mm[phase]))
 for f in (w/'q5_decision').glob('teacher*.json'):
  mm=read(f);t=mm['teacher']
  for method,m in mm['metrics'].items():
   with np.load(w/'q5_decision'/f'teacher{t}'/(method+'.npz')) as z:s=dict(z)
   actual=compute(s);npzs+=1
   for k,v in actual.items():errors[k]=max(errors.get(k,0),abs(v-m[k]))
   q5.append(dict(seed=seed,teacher=t,method=method,**m))
 for f in w.rglob('history.json'):
  h=read(f);receipt=read(f.parent/'complete.json');assert [r['epoch'] for r in h]==list(range(11));assert receipt['epochs']==10;assert sha(f.parent/'best.pt')==receipt['best_sha256']
  best=load(f.parent/'best.pt');assert best['epoch']==receipt['selected_epoch']
  stages.append(dict(seed=seed,path=f.parent.relative_to(w).as_posix(),selected=best['epoch'],seconds=sum(r.get('seconds',0) for r in h)))
  if 'formal' in f.parts:
   histories.append(dict(seed=seed,kind=f.parent.name,selected=receipt['selected_epoch'],history=h))
  if f.parent.name!='g0':
   g0=load(f.parent.parent/'g0/best.pt')['params'];assert all(torch.equal(v,best['params']['backbone.'+k]) for k,v in g0.items());frozen_count+=1
  if 'teachers' in f.parts:assert all(r['val'] is None for r in h)
 folds=read(w/'folds.json')
 for f in folds:
  train=set(f['train_ids']);held=set(f['audit_seen_ids']);query=set(f['query_ids']);assert not (train&held or train&query or held&query)
  assert set(index['rows'][i]['pair'] for i in train)==set(f['inner_pairs'])
  for t in (1,2,3):
   r=load(w/'teachers'/f't{t}_q{f["fold"]}'/'measure/response.pt');assert set(r['support_ids'])<=train;assert set(r['query_ids'])<=query;assert set(r['audit_ids'])<=query;assert not set(r['query_ids'])&set(r['audit_ids'])
   eff=r['effects'].astype(float);weights=r['query_weights'];pred=np.einsum('ebn,n->e',eff,weights)/eff.shape[1];target_error=max(target_error,float(np.max(abs(pred-r['target']))))
   route=r['routes'][len(r['support_ids']):]
   for group in (0,1):
    ww=weights*(route==group);mass=ww.sum();assert abs(mass-r['context_mass'][group])<1e-12
    yy=np.einsum('ebn,n->e',eff,ww)/eff.shape[1]/mass if mass else np.zeros(len(types));target_error=max(target_error,float(np.max(abs(yy-r['context_target'][group]))))
   checks.append(dict(seed=seed,fold=f['fold'],teacher=t,train=len(train),query_measure=len(r['query_ids']),query_audit=len(r['audit_ids']),seen_audit=len(held)))
   if f['fold']==5:
    with np.load(w/'q5_decision'/f'teacher{t}'/'uniform.npz') as z:aids=set(z['image_ids'].tolist())
    assert aids<=held|set(r['audit_ids']);assert not aids&train;assert not aids&set(r['query_ids'])
 records=[load(w/'teachers'/f't{t}_q{q}'/'measure/response.pt') for t in (1,2,3) for q in (1,2,3)]
 fitted=utility.fit(records,types,cfg['ridge_lambda']);saved=load(w/'utility.pt');fits.append(dict(seed=seed,max_beta_error=max(float(np.max(abs(fitted[k]['beta']-saved[k]['beta']))) for k in fitted)))
 form=load(w/'formal/measure/response.pt');pnew=utility.policies(saved,form,types,cfg['permutation_seeds']);old=load(w/'policies.pt')
 assert set(old)==set(pnew)
 for method,pi in old.items():
  assert torch.allclose(pi,pnew[method],atol=1e-10,rtol=1e-10)
  for g in range(len(pi)):
   for ty in range(3):
    vals=pi[g,types==ty].numpy();budget_error=max(budget_error,abs(vals.sum()-(types==ty).sum()/2))
    if method in ('matched','context','context_shuffle'):policy_stats.append(dict(seed=seed,method=method,context=g,type=ty,min=float(vals.min()),max=float(vals.max()),std=float(vals.std()),budget=float(vals.sum())))
 for q in (4,5):
  for t in (1,2,3):
   r=load(w/'teachers'/f't{t}_q{q}'/'measure/response.pt');recorded=read(w/'utility_audit.json')[f'Q{q}'][str(t)]
   for method,mod in saved.items():
    pred=utility.predict(mod,r);y=r['target'][None] if len(pred)==1 else r['context_target'];m=utility.audit({method:mod},r,types)[method]
    for k,v in m.items():assert abs(v-recorded[method][k])<1e-10
    within=[utility.rank_corr(pred[:,types==ty].ravel(),y[:,types==ty].ravel()) for ty in range(3)]
    audit.append(dict(seed=seed,fold=q,teacher=t,method=method,**m,within_type=within))
    pi=utility.policy(pred,types).numpy();mass=np.array([1.]) if len(pi)==1 else r['context_mass'];surrogate.append(dict(seed=seed,fold=q,teacher=t,method=method,predicted_gain=float(np.sum((pi-.5)*pred*mass[:,None])),measured_linear_gain=float(np.sum((pi-.5)*y*mass[:,None]))))
print('VERIFIED',len(rows),len(q5),'metrics; hashes',len(manifest),'frozen',frozen_count,flush=True)
write('verification.json',dict(archive_bytes=arc.stat().st_size,archive_sha256=sha(arc),manifest_files=len(manifest),formal_rows=len(rows),q5_rows=len(q5),npz_files=npzs,metric_max_errors=errors,training_stages=len(stages),frozen_checkpoint_checks=frozen_count,fold_teacher_checks=len(checks),effect_label_max_error=target_error,ridge_refit=fits,policy_budget_max_error=budget_error,source_code_matches_identity=True))
write('formal_recomputed.json',rows);write('q5_recomputed.json',q5);write('history.json',histories);write('stages.json',stages);write('utility_audits.json',audit);write('policy_statistics.json',policy_stats);write('linear_surrogate.json',surrogate)

def aggregate(data,method,phase,metric):
 return [float(np.mean([r[metric] for r in data if r['seed']==s and (phase is None or r.get('phase')==phase) and (r['method'].startswith(method[:-5]) if method.endswith('_mean') else r['method']==method)])) for s in seeds]
main=[];deltas=[]
methods=['g0','capacity','dense','uniform','matched','context','context_shuffle','matched_perm_mean','context_perm_mean']
for phase in ('val_cw','val_ow','test_cw','test_ow'):
 for method in methods:
  for metric in ('AUC_native','AUC_exact','HM_native','HM_exact','HM_at_val_bias','Seen','Unseen','pair_nll','ECE','pair_accuracy'):
   if metric=='HM_at_val_bias' and phase.startswith('val'):continue
   main.append(dict(phase=phase,method=method,metric=metric,**meanstd(aggregate(rows,method,phase,metric))))
comparisons=[('uniform','g0'),('context','g0'),('matched','uniform'),('context','uniform'),('context','matched'),('context','context_shuffle'),('context','dense'),('context','capacity'),('context','context_perm_mean'),('matched','matched_perm_mean')]
for phase in ('test_cw','test_ow'):
 for a,b in comparisons:
  for metric in ('AUC_native','AUC_exact','HM_at_val_bias','pair_nll','ECE','Seen','Unseen'):
   delta=np.array(aggregate(rows,a,phase,metric))-aggregate(rows,b,phase,metric)
   deltas.append(dict(phase=phase,method=a,reference=b,metric=metric,positive=int((delta>0).sum()),**meanstd(delta)))
qa=[];qd=[]
for method in methods[3:]:
 for metric in ('AUC_native','AUC_exact','HM_exact','pair_nll'):
  qa.append(dict(method=method,metric=metric,**meanstd(aggregate(q5,method,None,metric))))
for a,b in comparisons[2:]:
 if b in ('dense','capacity'):continue
 for metric in ('AUC_native','AUC_exact','pair_nll'):
  dif=[];teacherdif=[]
  for s in seeds:
   vv=[]
   for t in (1,2,3):
    def vals(m):return [r[metric] for r in q5 if r['seed']==s and r['teacher']==t and (r['method'].startswith(m[:-5]) if m.endswith('_mean') else r['method']==m)]
    v=np.mean(vals(a))-np.mean(vals(b));vv.append(v);teacherdif.append(float(v))
   dif.append(float(np.mean(vv)))
  qd.append(dict(method=a,reference=b,metric=metric,positive_teachers=int((np.array(teacherdif)>0).sum()),teacher_differences=teacherdif,**meanstd(dif)))
write('main.json',main);write('paired.json',deltas);write('q5_main.json',qa);write('q5_paired.json',qd)
print('FORMAL paired AUC exact/native')
for r in deltas:
 if r['metric'] in ('AUC_native','AUC_exact') and r['method']=='context':print(r)
print('Q5')
for r in qd:
 if r['metric'] in ('AUC_exact','pair_nll'):print(r)
print('TRAINING')
for h in histories:print(h['seed'],h['kind'],h['selected'],[None if r['val'] is None else round(r['val']['AUC_native'],4) for r in h['history']])
