import json
from pathlib import Path
import torch

torch.set_default_dtype(torch.float64)
gen=torch.Generator().manual_seed(721)
r=torch.randn(16,11,generator=gen);logw=torch.randn(11,generator=gen).log_softmax(0)

def margins(r,logw):
 delta=r[:,None,:]-r[None,:,:]
 return (logw+delta/2).logsumexp(-1)-(logw-delta/2).logsumexp(-1)

def centered_sep(r,logw):
 z=(r+logw).logsumexp(-1)
 return z-z.mean()

m=margins(r,logw);u=m.mean(1)
h=torch.randn(11,generator=gen)
b=torch.randn(16,generator=gen)
shifted=margins(r+h,logw)
bshift=margins(r+b[:,None],logw)
one=margins(r[:,:1],torch.zeros(1)).mean(1)
normal=torch.stack([sum((u[a]-u[b])-m[a,b] for b in range(16) if b!=a) for a in range(16)])
d=torch.randn(11,generator=gen,requires_grad=True)
f=(logw+d/2).logsumexp(0)-(logw-d/2).logsumexp(0)
gradient=torch.autograd.grad(f,d)[0]
pi_plus=(logw+d.detach()/2).softmax(0);pi_minus=(logw-d.detach()/2).softmax(0)
rr=torch.tensor([[4.,0.,0.],[0.,3.,0.]])
lw=torch.full((3,),-torch.log(torch.tensor(3.)))
hh=torch.tensor([0.,5.,0.])
result=dict(antisymmetry_max_error=float((m+m.T).abs().max()),
 common_patch_response_invariance_max_error=float((shifted-m).abs().max()),
 class_specific_bias_equivariance_max_error=float((bshift-(m+b[:,None]-b[None,:])).abs().max()),
 one_patch_original_margin_max_error=float((one-(r[:,0]-r[:,0].mean())).abs().max()),
 complete_graph_normal_equation_max_error=float(normal.abs().max()),zero_sum_error=float(u.sum().abs()),
 gradient_identity_max_error=float((gradient-(pi_plus+pi_minus)/2).abs().max()),
 example={'separate_before':float((centered_sep(rr,lw)[0]-centered_sep(rr,lw)[1])),
          'separate_after':float((centered_sep(rr+hh,lw)[0]-centered_sep(rr+hh,lw)[1])),
          'compare_before':float(margins(rr,lw)[0,1]),'compare_after':float(margins(rr+hh,lw)[0,1])})
assert all(v<1e-10 for k,v in result.items() if k!='example')
assert abs(result['example']['separate_before']-result['example']['separate_after'])>1.
assert abs(result['example']['compare_before']-result['example']['compare_after'])<1e-10
(Path(__file__).resolve().parent/'recomputed').mkdir(exist_ok=True)
(Path(__file__).resolve().parent/'recomputed/next_operator_math_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
