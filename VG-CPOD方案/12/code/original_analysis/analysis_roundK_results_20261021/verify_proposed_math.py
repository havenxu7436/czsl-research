"""Algebra checks for the proposed scoring operator; no training or benchmark fitting."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent

def lse(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.exp(z-m).sum(axis=axis, keepdims=True)), axis=axis)

def margin(delta, logw):
    return lse(logw + delta/2) - lse(logw - delta/2)

def main():
    rng = np.random.default_rng(20261021)
    A,N = 7,13
    r = rng.normal(size=(A,N))
    logw = rng.normal(size=N); logw -= lse(logw)
    edges = [(i,j) for i in range(A) for j in range(i+1,A)]
    B = np.zeros((len(edges),A))
    for k,(i,j) in enumerate(edges): B[k,i]=1; B[k,j]=-1
    delta = B@r; m = margin(delta,logw)
    L = B.T@B; D = np.linalg.solve(L+np.ones((A,A))/A,B.T)
    u = D@m
    field = np.zeros((A,A))
    for v,(i,j) in zip(m,edges): field[i,j]=v;field[j,i]=-v
    h = rng.normal(size=N)*10
    shifted = margin(B@(r+h),logw)
    col_bias = rng.normal(size=A)
    bias_u = D@margin(B@(r+col_bias[:,None]),logw)
    eps = 1e-5
    zplus = logw+delta/2; zminus = logw-delta/2
    grad = .5*(np.exp(zplus-lse(zplus)[:,None])+np.exp(zminus-lse(zminus)[:,None]))
    numerical=[]
    for n in range(N):
        offset=np.zeros_like(delta);offset[:,n]=eps
        numerical.append((margin(delta+offset,logw)-margin(delta-offset,logw))/(2*eps))
    linear=r@np.exp(logw)
    ell=lse(r+logw)
    keep=[k for k,(i,j) in enumerate(edges) if j==i+1 or (i,j) in ((0,A-1),(0,3),(2,5))]
    Bs=B[keep];Ls=Bs.T@Bs
    Ds=np.linalg.solve(Ls+np.ones((A,A))/A,Bs.T)
    us=Ds@m[keep]
    toy_r=np.array([[4.,0.,0.],[0.,3.,0.]])
    toy_w=np.full(3,-np.log(3.))
    toy_h=np.array([0.,5.,0.])
    result={
        'purpose':'formula identities only; synthetic arrays, no experimental performance claims',
        'max_antisymmetry_error':float(abs(m+margin(-delta,logw)).max()),
        'common_patch_offset_invariance_error':float(abs(m-shifted).max()),
        'class_bias_equivariance_error':float(abs(bias_u-u-col_bias+col_bias.mean()).max()),
        'gradient_error':float(abs(grad-np.array(numerical).T).max()),
        'normal_equation_error':float(abs(B.T@(B@u-m)).max()),
        'zero_sum_error':float(abs(u.sum())),
        'complete_graph_rowmean_error':float(abs(u-field.mean(axis=1)).max()),
        'ordinary_pool_projection_identity_error':float(abs(D@(B@ell)-(ell-ell.mean())).max()),
        'sparse_graph_normal_equation_error':float(abs(Bs.T@(Bs@us-m[keep])).max()),
        'sparse_graph_ordinary_pool_identity_error':float(abs(Ds@(Bs@ell)-(ell-ell.mean())).max()),
        'small_signal_limit_error':float(abs(D@margin(B@(eps*r),logw)/eps-(linear-linear.mean())).max()),
        'nonlinear_margin_cycle_example':float(field[0,1]+field[1,2]+field[2,0]),
        'projected_score_differs_from_ordinary_pool':float(abs(u-(ell-ell.mean())).max()),
        'toy':{
            'r':toy_r.tolist(),'shared_patch_offset':toy_h.tolist(),
            'ordinary_margin_before':float(np.diff(lse(toy_r+toy_w))[0]*-1),
            'ordinary_margin_after':float(np.diff(lse(toy_r+toy_h+toy_w))[0]*-1),
            'proposed_margin_before':float(margin(toy_r[0]-toy_r[1],toy_w)),
            'proposed_margin_after':float(margin((toy_r+toy_h)[0]-(toy_r+toy_h)[1],toy_w))}
    }
    for k,v in result.items():
        if k.endswith('error'): assert v<1e-8,(k,v)
    assert result['projected_score_differs_from_ordinary_pool']>.01
    (OUT/'proposed_math_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
