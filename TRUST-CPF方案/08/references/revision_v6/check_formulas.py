import json
from pathlib import Path
import numpy as np
rng=np.random.default_rng(20261025)
# Exact Bernoulli expectation and on/off partial derivative, finite graph.
E=5; masks=np.array([[(i>>k)&1 for k in range(E)] for i in range(2**E)])
risk=rng.normal(size=2**E);pi=rng.uniform(.15,.85,E)
def F(p):
    weights=np.prod(np.where(masks,p,1-p),1)
    return weights@risk
grad=[];utility=[]
for e in range(E):
    plus=pi.copy();minus=pi.copy();plus[e]+=1e-6;minus[e]-=1e-6
    grad.append((F(plus)-F(minus))/2e-6)
    z=pi.copy();o=pi.copy();z[e]=0;o[e]=1
    utility.append(F(z)-F(o))
gradient_error=float(np.max(np.abs(np.array(grad)+utility)))
# Averaged normal equations, raw-unit coefficients, relative intercepts.
J=9;T=np.repeat(np.arange(3),(12,36,48));N=len(T);K=8
xs=[rng.normal(size=(N,K)) for _ in range(J)];ys=[rng.normal(size=N) for _ in range(J)]
center=np.concatenate(xs).mean(0);spread=np.concatenate(xs).std(0)
a=np.array([.3,.5,.7]);lam=10/3
def design(x):
    z=np.column_stack((np.ones(N),(x-center)/spread))
    return np.concatenate([z*(T==t)[:,None] for t in range(3)],1)
Z=np.concatenate([design(x) for x in xs]);Y=np.concatenate([y/a[T] for y in ys])
b1=np.linalg.solve(Z.T@Z/J+lam*np.eye(27),Z.T@Y/J)
b2=np.linalg.solve(Z.T@Z+J*lam*np.eye(27),Z.T@Y)
alpha_error=float(np.max(np.abs(b1-b2)))
bet=b1.reshape(3,9);w=a[:,None]*bet[:,1:]/spread
intercept=a*(bet[:,0]-(bet[:,1:]*center/spread).sum(1))
p=(design(xs[0])@b1)*a[T]
raw=intercept[T]+(w[T]*xs[0]).sum(1)
raw_error=float(np.max(np.abs(p-raw)))
mean_error=float(np.max(np.abs((design(np.mean(xs[:2],axis=0))@b1)*a[T]-
 np.mean([(design(x)@b1)*a[T] for x in xs[:2]],axis=0))))
Zc=Z.reshape(J,N,27).copy();Yc=Y.reshape(J,N).copy()
for j in range(J):
    for t in range(3):
        ids=T==t
        Zc[j,ids]-=Zc[j,ids].mean(0)
        Yc[j,ids]-=Yc[j,ids].mean()
pair_error=0.
for t in range(3):
    v=rng.normal(size=(T==t).sum())
    pair_error=max(pair_error,abs(np.sum((v-v.mean())**2)-np.sum((v[:,None]-v[None,:])**2)/(2*len(v))))
Zc=Zc.reshape(-1,27);Yc=Yc.ravel()
bc=np.linalg.solve(Zc.T@Zc+J*lam*np.eye(27),Zc.T@Yc)
intercept_error=float(np.max(np.abs(bc[[0,9,18]])))
# Policy optimality, budget, empirical-error contraction bound.
def sigmoid(z):return 1/(1+np.exp(-z))
errors=[];bound_ratios=[]
for n in (12,36,48):
    z=rng.normal(size=n);z-=z.mean();s2=np.mean(z*z);eps=.7;kappa=1.
    v=z/np.sqrt(s2+eps**2)
    lo,hi=-50.,50.
    for _ in range(180):
        g=(lo+hi)/2;p=sigmoid(kappa*v-g)
        if p.sum()>n/2:lo=g
        else:hi=g
    p=sigmoid(kappa*v-(lo+hi)/2)
    errors.append(abs(p.sum()-n/2))
    lhs=np.linalg.norm(p-.5);rhs=kappa*np.linalg.norm(v)/4
    bound_ratios.append(float(lhs/rhs))
result=dict(bernoulli_derivative_error=gradient_error,alpha_conversion_error=alpha_error,
 raw_coefficient_error=raw_error,repeat_average_error=mean_error,
 pair_difference_identity_error=pair_error,relative_intercept_error=intercept_error,
 probability_budget_error=max(errors),shrink_bound_ratios=bound_ratios)
assert max(gradient_error,alpha_error,raw_error,mean_error,pair_error,intercept_error,max(errors))<1e-8
assert max(bound_ratios)<=1+1e-12
result['passed']=True
Path(__file__).with_name('formula_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))

