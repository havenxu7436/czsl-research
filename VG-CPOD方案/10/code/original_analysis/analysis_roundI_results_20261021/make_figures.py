"""Scientific figures comparing residual scales and their full training effects."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
D=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
SEEDS=(3351,3352,3353)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'figure.dpi':170})


def values(case,setting,key):
    return np.array([next(r[key] for r in D['metrics'] if r['case']==case and r['seed']==s and r['setting']==setting) for s in SEEDS])


def main():
    cases=['G0','G1','H1','I1','I2'];x=np.arange(len(cases))
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.3))
    for ax,setting,key,title in zip(axes,['test_cw','test_ow','test_ow'],['AUC_native','AUC_native','gt_object_attr_acc_unseen'],
        ['TEST CW AUC','TEST OW AUC','Unseen attributes (GT object, %)']):
        for s in range(3):ax.plot(x,[values(c,setting,key)[s] for c in cases],color='#d4dce0',lw=.9,marker='.')
        native=[values(c,setting,key).mean() for c in cases];exact=[values(c+'_exact',setting,key).mean() for c in cases]
        ax.plot(x,native,color='#176f79',lw=2.2,marker='o',label='Native-selected')
        ax.plot(x,exact,color='#ad6b73',lw=1.6,ls='--',marker='s',label='Exact-selected')
        for i,v in enumerate(native):ax.annotate('%.3f'%v,(i,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(x,cases);ax.set_title(title,fontsize=10);ax.margins(y=.22,x=.12);ax.legend(fontsize=8)
    fig.suptitle('rho=0.15, beta=0.075; AUC evaluated with the native metric',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'figures/01_results_and_selection.png');plt.close(fig)

    cases=['G1','H1','I1','I2'];x=np.arange(4)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.3))
    stds=[np.mean([r['image_std'] for r in D['moments'] if r['case']==c]) for c in cases]
    axes[0].bar(x,stds,color=['#9aaebb','#ad6b73','#7cadb1','#176f79'])
    for i,v in enumerate(stds):axes[0].text(i,v+.0015,'%.4f'%v,ha='center',fontsize=9)
    axes[0].set_xticks(x,cases);axes[0].set_ylim(0,.078);axes[0].set_title('Full-TRAIN image gate std. dev.')
    for ax,setting,title in [(axes[1],'test_cw','TEST CW: dynamic - TRAIN mean'),(axes[2],'test_ow','TEST OW: dynamic - TRAIN mean')]:
        deltas=[values(c,setting,'AUC_native')-values(c+'_mean',setting,'AUC_native') for c in cases]
        avg=[v.mean() for v in deltas];sd=[v.std(ddof=1) for v in deltas]
        ax.bar(x,avg,color=['#9aaebb','#ad6b73','#7cadb1','#176f79'],yerr=sd,capsize=4)
        for i,(v,e) in enumerate(zip(avg,sd)):ax.text(i,v+e+.08,'%+.3f'%v,ha='center',fontsize=9)
        ax.set_xticks(x,cases);ax.axhline(0,color='#444',lw=1);ax.set_title(title,fontsize=10)
        ax.set_ylim(-.5,3.1 if setting=='test_cw' else 2.5);ax.set_ylabel('AUC change (points)')
    fig.tight_layout();fig.savefig(OUT/'figures/02_gate_effects.png');plt.close(fig)

    cases=['H1','I1','I2'];x=np.arange(3)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,setting in zip(axes,['test_cw','test_ow']):
        ref=values('G0',setting,'AUC_native')
        a=[(values(c+'_mean',setting,'AUC_native')-ref).mean() for c in cases]
        b=[(values(c,setting,'AUC_native')-values(c+'_mean',setting,'AUC_native')).mean() for c in cases]
        ax.bar(x-.22,a,width=.22,color='#9aaebb',label='Mean condition - G0')
        ax.bar(x,b,width=.22,color='#176f79',label='Dynamic - mean condition')
        ax.bar(x+.22,np.asarray(a)+b,width=.22,color='#ad6b73',label='Full model - G0')
        ax.axhline(0,color='#444',lw=1);ax.set_xticks(x,cases);ax.set_title(setting.upper())
        ax.set_ylabel('AUC change (points)');ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Intervention and full training effects use distinct comparisons',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'figures/03_static_dynamic_effects.png');plt.close(fig)


if __name__=='__main__':main()
