"""Export scientific figures for the completed K analysis."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'recomputed'
D=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
S=json.loads((OUT/'supplement.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def metric(case,setting,key='AUC_native_mean'):
    return next(r[key] for r in D['means'] if r['case']==case and r['setting']==setting)

def save(fig,name):
    fig.savefig(OUT/'figures'/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'figures'/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

def main():
    cases=['D1','G0','G1','H1','I1','I2','J1','K1','K2']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,world in zip(axes,['cw','ow']):
        setting='test_'+world
        for i,c in enumerate(cases):
            values=[r['AUC_native'] for r in D['metrics'] if r['case']==c and r['setting']==setting]
            ax.scatter(np.full(3,i)+np.array([-.1,0,.1]),values,s=22,alpha=.6,color='#35759f')
        ax.plot(range(len(cases)),[metric(c,setting) for c in cases],color='#153b55',marker='o',label='Three-seed mean')
        ax.axhline(metric('G0',setting),color='#bc671e',ls='--',lw=1,label='Static G0 mean')
        ax.set(xticks=range(len(cases)),xticklabels=cases,title='TEST '+world.upper(),ylabel='AUC (points)')
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False,fontsize=9)
    save(fig,'01_history_plateau')
    cases=['J1','K1','K2'];x=np.arange(3)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    ratio=[np.mean([r['lr_scaled_adaptive_step_norm_over_static'] for r in S['optimizer'] if r['case']==c and r['stage']=='native_selected' and r['parameter']=='Wd']) for c in cases]
    scales=[np.mean([r['dynamic_minus_mean_centered_rms_unseen'] for r in S['dynamic_scales'] if r['case']==c and r['phase']=='test']) for c in cases]
    for ax,y,title in [(axes[0],ratio,'Wd / Ws adaptive update norm'),(axes[1],scales,'Unseen conditional dynamic RMS')]:
        ax.bar(x,y,color=['#265e81','#3f8f96','#96b9bb'])
        for i,v in enumerate(y):ax.text(i,v+max(y)*.025,'%.3f'%v,ha='center')
        ax.set(xticks=x,xticklabels=cases,title=title,ylim=(0,max(y)*1.2))
    axes[2].plot(x,[metric(c,'test_ow') for c in cases],marker='o',color='#265e81',label='Dynamic models')
    axes[2].axhline(metric('G0','test_ow'),ls='--',color='#bc671e',label='Static G0')
    axes[2].set(xticks=x,xticklabels=cases,title='TEST OW AUC',ylim=(24,25.5));axes[2].legend(frameon=False)
    save(fig,'02_update_scale')
    cases=['G0','I2','J1','K1','K2']
    rows=[next(r for r in S['candidate_means'] if r['case']==c and r['phase']=='test' and r['world']=='ow') for c in cases]
    correct=np.array([r['unseen_pool_accuracy_mean'] for r in rows])
    within=np.array([100*r['within_object_ranking_errors_mean']/1891 for r in rows])
    cross=100-correct-within
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.bar(cases,correct,label='Correct among all unseen candidates',color='#327d85')
    ax.bar(cases,within,bottom=correct,label='Wrong attribute within true object',color='#c4804e')
    ax.bar(cases,cross,bottom=correct+within,label='Cross-object competition',color='#697c9d')
    for i in range(len(cases)):
        for bottom,amount in [(0,correct[i]),(correct[i],within[i]),(correct[i]+within[i],cross[i])]:
            ax.text(i,bottom+amount/2,'%.1f%%'%amount,ha='center',va='center',color='white')
    ax.set(ylabel='Share of TEST unseen images (%)',ylim=(0,100),title='Error decomposition in the full OW unseen candidate pool')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=1,frameon=False)
    save(fig,'03_error_decomposition')

if __name__=='__main__':main()
