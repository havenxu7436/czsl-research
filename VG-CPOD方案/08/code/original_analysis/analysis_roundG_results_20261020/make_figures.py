"""Scientific figures for beta, matched training, gate variance and weight selection."""
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
                     'axes.spines.right':False,'figure.dpi':160})


def metric(case,seed,setting,key):
    return next(r[key] for r in D['metrics'] if r['case']==case and r['seed']==seed and r['setting']==setting)


def main():
    cases=['F1','G1','E1'];betas=[.05,.075,.1]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,setting,key,title in zip(axes,['test_cw','test_ow','test_ow'],
        ['AUC_native','AUC_native','gt_object_attr_acc_unseen'],
        ['TEST CW AUC_native','TEST OW AUC_native','TEST unseen attributes (GT object)']):
        for seed in SEEDS:
            values=[metric(c,seed,setting,key) for c in cases]
            ax.plot(betas,values,color='#bbc8d0',marker='.',lw=1)
        native=[np.mean([metric(c,s,setting,key) for s in SEEDS]) for c in cases]
        exact=[np.mean([metric(c+'_exact',s,setting,key) for s in SEEDS]) for c in cases]
        ax.plot(betas,native,color='#176f79',marker='o',lw=2.3,label='Native-selected mean')
        ax.plot(betas,exact,color='#ad6b73',marker='s',ls='--',lw=1.7,label='Exact-selected mean')
        for i,(x,y) in enumerate(zip(betas,native)):
            ax.annotate('%.3f'%y,(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
        ax.set_title(title,fontsize=10);ax.set_xlabel('beta at rho=0.15');ax.set_xticks(betas)
        ax.set_xlim(.045,.105);ax.legend(fontsize=8)
        lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+.35)
    fig.tight_layout();fig.savefig(OUT/'figures/01_beta_and_selection.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,setting in zip(axes,['test_cw','test_ow']):
        x=np.arange(3);all_values=[]
        for offset,key,label,color in [(-.17,'AUC_native','Native AUC','#176f79'),
                                       (.17,'AUC_exact','Exact AUC','#9aaebb')]:
            vals=[metric('G1',s,setting,key)-metric('G0',s,setting,key) for s in SEEDS]
            all_values+=vals;ax.bar(x+offset,vals,width=.32,color=color,label=label)
            for i,y in enumerate(vals):ax.text(i+offset,y+(.006 if y>=0 else -.006),'%+.3f'%y,
                ha='center',va='bottom' if y>=0 else 'top',fontsize=8)
        ax.axhline(0,color='#333',lw=1);ax.set_xticks(x,list(map(str,SEEDS)))
        ax.set_title(setting.upper()+': G1 - G0');ax.set_ylabel('AUC change (points)')
        ax.set_ylim(min(all_values+[0])-.06,max(all_values+[0])+.06);ax.legend(fontsize=9)
    fig.suptitle('Both metrics use native-selected checkpoints',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'figures/02_matched_static_dynamic.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    fractions=[100*next(r['image_fraction'] for r in D['moments'] if r['case']=='G1' and r['seed']==s) for s in SEEDS]
    static=100-np.array(fractions);x=np.arange(3)
    axes[0].barh(x,static,color='#9aaebb',label='Fixed object/mode variation')
    axes[0].barh(x,fractions,left=static,color='#176f79',label='Image variation')
    for i,value in enumerate(fractions):axes[0].text(101.5,i,'%.2f%% image'%value,va='center',fontsize=9)
    axes[0].set_xlim(0,124);axes[0].set_yticks(x,list(map(str,SEEDS)))
    axes[0].set_xlabel('Fraction of total TRAIN gate variance (%)');axes[0].set_title('G1 native-selected gates')
    axes[0].legend(loc='lower left',fontsize=8,bbox_to_anchor=(0,-.32))
    values=[]
    for offset,setting,label,color in [(-.17,'test_cw','CW','#176f79'),(.17,'test_ow','OW','#9aaebb')]:
        vals=[metric('G1',s,setting,'AUC_native')-metric('G1_mean',s,setting,'AUC_native') for s in SEEDS]
        values+=vals;axes[1].bar(x+offset,vals,width=.32,color=color,label=label)
        for i,y in enumerate(vals):axes[1].text(i+offset,y+(.01 if y>=0 else -.01),'%+.3f'%y,
            ha='center',va='bottom' if y>=0 else 'top',fontsize=8)
    axes[1].set_xticks(x,list(map(str,SEEDS)));axes[1].axhline(0,color='#333',lw=1)
    axes[1].set_ylim(min(values+[0])-.08,max(values+[0])+.08);axes[1].legend()
    axes[1].set_ylabel('TEST AUC_native change (points)');axes[1].set_title('G1 dynamic minus TRAIN mean')
    fig.tight_layout();fig.savefig(OUT/'figures/03_gate_variance_and_effect.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,case in zip(axes,['G0','G1']):
        history=[r for r in D['history'] if r['case']==case and r['seed']==3353 and r['epoch']>=4]
        ax.plot([r['epoch'] for r in history],[r['val_cw'] for r in history],color='#176f79',marker='o',label='Native AUC')
        ax.plot([r['epoch'] for r in history],[r['val_cw_exact'] for r in history],color='#ad6b73',marker='s',label='Exact AUC')
        exact=next(r for r in history if r['epoch']==6);native=next(r for r in history if r['epoch']==9)
        ax.annotate('Exact selects 6',(6,exact['val_cw_exact']),xytext=(7,9),textcoords='offset points',fontsize=9)
        ax.annotate('Native selects 9',(9,native['val_cw']),xytext=(8.05,43.6),textcoords='data',fontsize=9,
            arrowprops=dict(arrowstyle='->',color='#667580',lw=.9))
        ax.set_title(case+' / seed3353: VAL CW');ax.set_xlabel('Epoch (4-10)');ax.set_ylabel('AUC (area x 100)')
        ax.set_xticks(range(4,11));ax.legend(fontsize=9)
        lo,hi=ax.get_ylim();ax.set_ylim(lo-.08,hi+.14)
    fig.tight_layout();fig.savefig(OUT/'figures/04_selection_divergence.png');plt.close(fig)


if __name__=='__main__':
    main()
