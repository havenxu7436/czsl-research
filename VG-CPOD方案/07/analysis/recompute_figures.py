"""Standalone F1 experimental figures from verified metric and prediction tables."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent/'recomputed'
D = json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
M = json.loads((OUT/'mechanism.json').read_text(encoding='utf-8'))
SEEDS = (3351,3352,3353)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':160})


def metric(case,seed,setting,key):
    return next(r[key] for r in D['metrics'] if r['case']==case and r['seed']==seed and r['setting']==setting)


def main():
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    cases=['A5','D1','E1','F1']
    for ax,setting,title in zip(axes,['val_cw','test_cw','test_ow'],['VAL CW','TEST CW','TEST OW']):
        for seed in SEEDS:
            values=[metric(c,seed,setting,'AUC_native') for c in cases]
            ax.plot(cases,values,color='#b6c3cb',marker='.',lw=1)
        means=[np.mean([metric(c,s,setting,'AUC_native') for s in SEEDS]) for c in cases]
        ax.plot(cases,means,color='#176f79',marker='o',lw=2.5)
        for i,y in enumerate(means):ax.annotate('%.3f'%y,(i,y),xytext=(0,10),textcoords='offset points',ha='center')
        ax.set_title(title);ax.set_ylabel('AUC_native (area x 100)')
        lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+.35)
    fig.tight_layout();fig.savefig(OUT/'figures/05_f1_comparison.png');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,setting,title in zip(axes,['AUC_native','AUC_native','gt_object_attr_acc_unseen'],
        ['test_cw','test_ow','test_ow'],['TEST CW AUC','TEST OW AUC','Unseen conditional attributes']):
        values=[metric('F1',s,setting,key)-metric('E1',s,setting,key) for s in SEEDS]
        ax.bar(list(map(str,SEEDS)),values,color=['#176f79' if x>0 else '#9c6672' for x in values])
        ax.axhline(0,color='#333',lw=1)
        for i,y in enumerate(values):ax.text(i,y+(.03 if y>0 else -.03),'%+.3f'%y,ha='center',va='bottom' if y>0 else 'top')
        ax.set_ylim(min(values+[0])-.25,max(values+[0])+.25)
        ax.set_title(title);ax.set_ylabel('F1 - E1 (points)')
    fig.tight_layout();fig.savefig(OUT/'figures/06_f1_e1_paired.png');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,['conditional_accuracy','unseen_rank_ceiling','extra_dominates_valid'],
        ['GT-object attribute accuracy','Rank within all unseen candidates','CW-excluded candidate dominates']):
        means=[];maxima=[]
        for i,case in enumerate(['E1','F1']):
            vals=[r[key] for r in M['diagnostics'] if r['case']==case and r['phase']=='test' and r['group']=='unseen']
            means.append(np.mean(vals));maxima.append(max(vals));ax.scatter([i-.06,i,i+.06],vals,color='#263943',zorder=3,s=25)
        ax.bar(['E1','F1'],means,color=['#95a9b7','#176f79'],alpha=.65)
        for i,y in enumerate(means):ax.text(i,maxima[i]+.8,'%.3f'%y,ha='center')
        ax.set_ylim(0,max(means)+8);ax.set_title(title,fontsize=10);ax.set_ylabel('TEST unseen images (%)')
    fig.tight_layout();fig.savefig(OUT/'figures/07_ranking_mechanism.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    cases=['C2','D1','E1','F1']
    for ax,setting in zip(axes,['test_cw','test_ow']):
        means=[];all_values=[]
        for i,c in enumerate(cases):
            vals=[metric(c,s,setting,'AUC_native')-metric(c+'_mean',s,setting,'AUC_native') for s in SEEDS]
            means.append(np.mean(vals));all_values+=vals
            ax.scatter([i-.07,i,i+.07],vals,color='#263943',s=25,zorder=3)
        ax.bar(cases,means,color=['#95a9b7','#95a9b7','#6c99a0','#176f79'],alpha=.65)
        label_y=[.08,.115,.215,.045] if setting=='test_cw' else [-.18,-.06,.12,-.12]
        for i,y in enumerate(means):ax.text(i,label_y[i],'%+.3f'%y,ha='center')
        ax.axhline(0,color='#333',lw=1);ax.set_title(setting.upper()+': dynamic - TRAIN mean')
        ax.set_ylim(min(all_values+[0])-.1,max(all_values+[0])+.1);ax.set_ylabel('AUC_native change (points)')
    fig.tight_layout();fig.savefig(OUT/'figures/08_f1_dynamic_effect.png');plt.close(fig)


if __name__=='__main__':
    main()
