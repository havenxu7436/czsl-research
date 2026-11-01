import numpy as np

def harmonic(s,u):return 2*s*u/(s+u) if s+u else 0.

def curves(s):
    y=s['truth'];seen=s['is_seen'].astype(bool);cs=(s['seen_pred']==y)&seen;cu=(s['unseen_pred']==y)&~seen
    if not seen.any() or seen.all():raise ValueError('Evaluation requires seen and unseen images')
    d=s['seen_score'].astype(np.float64)-s['unseen_score'].astype(np.float64)
    order=np.argsort(d,kind='stable');t,starts=np.unique(d[order],return_index=True)
    ss=np.r_[cs.sum(),cs.sum()-np.cumsum(np.add.reduceat(cs[order].astype(int),starts))]/seen.sum()
    uu=np.r_[0,np.cumsum(np.add.reduceat(cu[order].astype(int),starts))]/(~seen).sum()
    biases=np.r_[t[0]-1.,t[:-1]+(t[1:]-t[:-1])/2,t[-1]+1.]
    h=np.divide(2*ss*uu,ss+uu,out=np.zeros_like(ss),where=(ss+uu)>0);best=int(h.argmax())
    trapz=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    exact=dict(AUC=float(trapz(ss,uu)*100),HM=float(h[best]*100),bias=float(biases[best]),seen=float(ss.max()*100),unseen=float(uu.max()*100))
    # Legacy finite scan, same -1e-4 and ~20 bins as the existing evaluator.
    diffs=np.sort(d[cu]-1e-4);bs=np.r_[diffs[::max(len(diffs)//20,1)],1000.]
    ns=[];nu=[]
    for b in bs:
        pred=np.where(d<b,s['unseen_pred'],s['seen_pred']);correct=pred==y
        ns.append(correct[seen].mean());nu.append(correct[~seen].mean())
    hh=np.array([harmonic(a,b) for a,b in zip(ns,nu)]);j=int(hh.argmax())
    native=dict(AUC=float(trapz(ns,nu)*100),HM=float(hh[j]*100),bias=float(bs[j]))
    return exact,native

def metrics(stats,bias=None):
    exact,native=curves(stats)
    result=dict(AUC_native=native['AUC'],AUC_exact=exact['AUC'],HM_native=native['HM'],HM_exact=exact['HM'],
                Seen=exact['seen'],Unseen=exact['unseen'],best_bias=native['bias'],images=len(stats['truth']))
    if bias is not None:
        pred=np.where(stats['unseen_score']+bias>stats['seen_score'],stats['unseen_pred'],stats['seen_pred'])
        correct=pred==stats['truth'];seen=stats['is_seen'].astype(bool)
        s,u=correct[seen].mean(),correct[~seen].mean()
        result.update(HM_at_val_bias=100*harmonic(s,u),val_bias=float(bias),seen_at_val_bias=100*float(s),unseen_at_val_bias=100*float(u))
    return result

