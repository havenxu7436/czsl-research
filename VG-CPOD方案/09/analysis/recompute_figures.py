"""Scientific comparison plots for H1 and its matched references."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'recomputed'
D=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
SEEDS=(3351,3352,3353)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'figure.dpi':170})


def values(case,setting,key):
    return np.array([next(r[key] for r in D['metrics'] if r['case']==case and r['seed']==s and r['setting']==setting) for s in SEEDS])


def main():
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for ax,setting,key,title in zip(axes,['test_cw','test_ow','test_ow'],
        ['AUC_native','AUC_native','gt_object_attr_acc_unseen'],
        ['TEST CW AUC','TEST OW AUC','Unseen attribute accuracy (%)']):
        for s in range(3):ax.plot([0,1,2],[values(c,setting,key)[s] for c in ['G0','G1','H1']],color='#c4cdd1',lw=1,marker='.')
        means=[values(c,setting,key).mean() for c in ['G0','G1','H1']]
        ax.plot([0,1,2],means,color='#176f79',lw=2.3,marker='o')
        for i,v in enumerate(means):ax.annotate('%.3f'%v,(i,v),xytext=(0,9),textcoords='offset points',ha='center')
        ax.set_xticks([0,1,2],['G0 static','G1 local','H1 residual']);ax.set_title(title,fontsize=10)
        ax.margins(y=.25,x=.18)
    fig.suptitle('Matched rho=0.15, beta=0.075; native-selected checkpoints',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'figures/01_matched_results.png');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13.5,4.3))
    x=np.arange(3)
    for offset,case,color in [(-.18,'G1','#9aaebb'),(.18,'H1','#176f79')]:
        rows=[next(r for r in D['moments'] if r['case']==case and r['seed']==s) for s in SEEDS]
        fraction=[r['image_fraction']*100 for r in rows]
        axes[0].bar(x+offset,fraction,width=.34,color=color,label=case)
        axes[1].bar(x+offset,[r['image_std'] for r in rows],width=.34,color=color,label=case)
        for i,v in enumerate(fraction):axes[0].text(i+offset,v+.9,'%.2f'%v,ha='center',fontsize=8)
    axes[0].set_title('TRAIN image variance fraction (%)');axes[0].set_ylim(0,114)
    axes[1].set_title('TRAIN image gate std. dev.');axes[1].set_ylim(0,.08)
    for ax in axes[:2]:ax.set_xticks(x,list(map(str,SEEDS)));ax.legend(fontsize=8)
    settings=['val_cw','val_ow','test_cw','test_ow'];means=[];sds=[]
    for setting in settings:
        delta=values('H1',setting,'AUC_native')-values('H1_mean',setting,'AUC_native')
        means.append(delta.mean());sds.append(delta.std(ddof=1))
    axes[2].bar(np.arange(4),means,color=['#9aaebb','#9aaebb','#176f79','#176f79'],yerr=sds,capsize=4)
    axes[2].axhline(0,color='#444',lw=1);axes[2].set_xticks(range(4),['VAL\nCW','VAL\nOW','TEST\nCW','TEST\nOW'])
    axes[2].set_title('H1 dynamic - TRAIN mean (AUC)');axes[2].set_ylim(-2.2,3.2)
    for i,v in enumerate(means):axes[2].text(i,v+(.6 if v>=0 else -.55),'%+.3f'%v,ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'figures/02_gate_and_intervention.png');plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(13,6.5),sharex=True)
    for column,seed in enumerate(SEEDS):
        for case,color in [('G1','#9aaebb'),('H1','#176f79')]:
            rows=[r for r in D['history'] if r['case']==case and r['seed']==seed]
            axes[0,column].plot([r['epoch'] for r in rows],[r['val_cw'] for r in rows],marker='.',color=color,label=case)
            selected=next(r for r in rows if r['selected'])
            axes[0,column].scatter(selected['epoch'],selected['val_cw'],s=60,facecolors='none',edgecolors=color,zorder=4)
            axes[1,column].plot([r['epoch'] for r in rows if r['epoch']>1],[r['pair'] for r in rows if r['epoch']>1],color=color,marker='.',label=case)
        axes[0,column].set_title('seed %d'%seed);axes[1,column].set_xlabel('Epoch')
        axes[0,column].legend(fontsize=8)
    axes[0,0].set_ylabel('VAL CW AUC');axes[1,0].set_ylabel('TRAIN pair loss (epochs 2-10)')
    fig.tight_layout();fig.savefig(OUT/'figures/03_history.png');plt.close(fig)


if __name__=='__main__':main()
