"""Scientific figures for J1 results and separate-readout optimization."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
D=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
S=json.loads((OUT/'supplement.json').read_text(encoding='utf-8'))
SEEDS=(3351,3352,3353)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'figure.dpi':170})

def values(case,setting,key='AUC_native'):
    return np.array([next(r[key] for r in D['metrics'] if r['case']==case and r['seed']==s and r['setting']==setting) for s in SEEDS])

def main():
    cases=['D1','G0','G1','I2','J1'];x=np.arange(len(cases))
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.4))
    for ax,setting,key,title in zip(axes,['test_cw','test_ow','test_ow'],['AUC_native','AUC_native','gt_object_attr_acc_unseen'],
        ['TEST CW AUC','TEST OW AUC','Unseen attribute accuracy (GT object, %)']):
        y=np.array([values(c,setting,key) for c in cases])
        for j,s in enumerate(SEEDS):ax.scatter(x+(j-1)*.07,y[:,j],s=23,alpha=.65,label='seed%d'%s)
        means=y.mean(1);ax.plot(x,means,c='#176f79',lw=2,marker='D',label='3-seed mean')
        for i,v in enumerate(means):ax.annotate('%.3f'%v,(i,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(x,cases);ax.set_title(title,fontsize=10);ax.margins(y=.2,x=.1)
    axes[0].legend(fontsize=8)
    fig.suptitle('Native-selected checkpoints; D1 uses beta=0.1, other cases beta=0.075',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'figures/01_results.png');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13.5,4.4))
    for i,(case,color) in enumerate([('I2','#9aaebb'),('J1','#176f79')]):
        vals=[r['image_std'] for r in D['moments'] if r['case']==case]
        axes[0].bar(i,np.mean(vals),color=color)
        axes[0].text(i,np.mean(vals)+.001,'%.4f'%np.mean(vals),ha='center')
        vals=[r['centered_conditional_logit_rms'] for r in S['prediction_differences'] if r['comparison']==case+'-'+case+'_mean' and r['phase']=='test' and r['group']=='unseen']
        axes[1].bar(i,np.mean(vals),color=color)
        axes[1].text(i,np.mean(vals)+.025,'%.4f'%np.mean(vals),ha='center')
    axes[0].set_title('Full-TRAIN image gate std. dev.');axes[0].set_ylim(0,.035)
    axes[1].set_title('TEST unseen dynamic-minus-mean\ncentered conditional-logit RMS');axes[1].set_ylim(0,.78)
    for ax in axes[:2]:ax.set_xticks([0,1],['I2','J1'])
    x=np.arange(3);moment=[];adaptive=[]
    for seed in SEEDS:
        ws=next(r for r in S['adam_steps'] if r['seed']==seed and r['parameter']=='Wc')
        wd=next(r for r in S['adam_steps'] if r['seed']==seed and r['parameter']=='Wd')
        moment.append(wd['moment_norm']/ws['moment_norm']);adaptive.append(wd['adam_adaptive_step_norm']/ws['adam_adaptive_step_norm'])
    axes[2].bar(x-.16,moment,width=.32,color='#9aaebb',label='1st-moment norm ratio')
    axes[2].bar(x+.16,adaptive,width=.32,color='#ad6b73',label='Adaptive-step norm ratio')
    axes[2].set_xticks(x,[str(s) for s in SEEDS]);axes[2].set_ylim(0,1.2)
    axes[2].set_title('J1 selected checkpoint: Wd / Ws');axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'figures/02_optimizer_and_scale.png');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2),sharey=True)
    for ax,seed in zip(axes,SEEDS):
        rows=[r for r in D['readout_history'] if r['seed']==seed]
        epoch=[r['epoch'] for r in rows]
        ax.plot(epoch,[r['cp_static_centered_rms'] for r in rows],c='#176f79',marker='o',label='Static')
        ax.plot(epoch,[r['cp_dynamic_centered_rms'] for r in rows],c='#ad6b73',marker='s',label='Dynamic')
        selected=next(r['epoch'] for r in rows if r['selected']);ax.axvline(selected,c='#777',ls='--',lw=1,label='Selected epoch')
        ax.set_title('seed%d'%seed);ax.set_xlabel('Epoch');ax.set_xticks([1,3,5,7,10]);ax.legend(fontsize=8)
    axes[0].set_ylabel('Attribute-centered training correction RMS')
    fig.tight_layout();fig.savefig(OUT/'figures/03_readout_history.png');plt.close(fig)

if __name__=='__main__':main()
