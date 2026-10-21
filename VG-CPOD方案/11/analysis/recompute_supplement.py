"""Prediction-level differences, conditional correction scales and Adam readout steps."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parent.parent
OUT=Path(__file__).resolve().parent/'recomputed'
NEW=STAGE/'results/raw_export/roundJ_20261021'
OLD=STAGE.parent/'10/results/raw_export/roundI_20261021'
sys.path.insert(0,str(STAGE/'code/vgcpod_v1_roundJ'))
from vgcpod.common import load_torch


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    manifest=read(STAGE.parent/'v1/results/ut-zap50k-3seeds-20261018/raw_export/vgcpod_v1_runs/cache_ut_clip_l14/features_manifest.json')
    train=set(map(tuple,manifest['pairs']['train']));labels=manifest['labels']
    rows=[];objects=[];steps=[]
    for seed in (3351,3352,3353):
        for phase in ('val','test'):
            paths={
                'I2':OLD/('I2/ut-zap50k/V1/seed%d/predictions/%s_ow.npz'%(seed,phase)),
                'I2_mean':OLD/('mean_gate/I2/seed%d/train_mean/predictions/%s_ow.npz'%(seed,phase)),
                'J1':NEW/('J1/ut-zap50k/V1/seed%d/predictions/%s_ow.npz'%(seed,phase)),
                'J1_mean':NEW/('mean_gate/J1/seed%d/train_mean/predictions/%s_ow.npz'%(seed,phase))}
            data={}
            for case,path in paths.items():
                with np.load(path,allow_pickle=False) as archive:data[case]={k:archive[k].copy() for k in archive.files}
            ref=data['I2'];truth=[(labels['attrs'][a],labels['objs'][o]) for a,o in zip(ref['attribute_truth'],ref['object_truth'])]
            seen=np.array([p in train for p in truth])
            for case,baseline in [('J1','I2'),('J1','J1_mean'),('I2','I2_mean')]:
                a,b=data[case],data[baseline]
                assert np.array_equal(a['image_ids'],b['image_ids'])
                logits=a['conditional_logits_gt_object'].astype(np.float64);previous=b['conditional_logits_gt_object'].astype(np.float64)
                ac=logits.argmax(-1)==a['attribute_truth'];bc=previous.argmax(-1)==b['attribute_truth']
                delta=logits-previous;delta-=delta.mean(-1,keepdims=True)
                for group,mask in [('all',np.ones(len(seen),dtype=bool)),('seen',seen),('unseen',~seen)]:
                    rows.append(dict(comparison=case+'-'+baseline,seed=seed,phase=phase,group=group,images=int(mask.sum()),
                        recovered=int((ac&~bc&mask).sum()),regressed=int((~ac&bc&mask).sum()),net_correct=int((ac&mask).sum())-int((bc&mask).sum()),
                        conditional_agreement_pct=float((logits[mask].argmax(-1)==previous[mask].argmax(-1)).mean()*100),
                        centered_conditional_logit_rms=float(np.sqrt(np.mean(delta[mask]**2)))))
                if case=='J1' and baseline=='I2' and phase=='test':
                    for j,name in enumerate(labels['objs']):
                        mask=(~seen)&(a['object_truth']==j)
                        if mask.any():objects.append(dict(seed=seed,object=name,images=int(mask.sum()),
                            net_correct=int((ac&mask).sum())-int((bc&mask).sum()),
                            recovered=int((ac&~bc&mask).sum()),regressed=int((~ac&bc&mask).sum())))
        ck=load_torch(NEW/('J1/ut-zap50k/V1/seed%d/checkpoints/best.pt'%seed))
        optimizer=ck['optimizer'];group=next(g for g in optimizer['param_groups'] if g['name']=='cp')
        assert len(group['params'])==5
        for name,index in zip(('Au','Av','Bvis','Wc','Wd'),group['params']):
            if name not in ('Wc','Wd'):continue
            state=optimizer['state'][index];t=float(state['step']);beta1,beta2=group['betas']
            m=state['exp_avg'].double()/(1-beta1**t);v=state['exp_avg_sq'].double()/(1-beta2**t)
            step=group['lr']*m/(v.sqrt()+group['eps'])
            steps.append(dict(seed=seed,parameter=name,epoch=ck['epoch'],step=int(t),lr=group['lr'],
                parameter_norm=float(ck['model'][name].norm()),adam_adaptive_step_norm=float(step.norm()),
                adam_step_rms=float(step.square().mean().sqrt()),moment_norm=float(m.norm())))
    result=dict(prediction_differences=rows,per_object=objects,adam_steps=steps)
    (OUT/'supplement.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for name,data in result.items():
        with (OUT/'tables'/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps([r for r in rows if r['phase']=='test' and r['group']=='unseen'],indent=2))
    print(json.dumps(steps,indent=2))


if __name__=='__main__':main()
