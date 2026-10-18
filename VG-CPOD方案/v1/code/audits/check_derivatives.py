from pathlib import Path
import json
import sys

import torch

import argparse
parser = argparse.ArgumentParser(description='Verify VG-CPOD formula gradients against autograd')
parser.add_argument('--package', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
BASE = args.package.resolve()
sys.path.insert(0, str(BASE))
from vgcpod.common import read_json, unit
from vgcpod.model import VGCPOD, js_divergence

torch.set_num_threads(1)
torch.manual_seed(1801)
B, N, M, d, r = 4, 2, 3, 12, 3
dtype = torch.float64
a, o = torch.tensor([0, 1, 2, 1]), torch.tensor([0, 1, 0, 1])
bi = torch.arange(B)
ya = torch.nn.functional.one_hot(a, M).to(dtype)
yo = torch.nn.functional.one_hot(o, N).to(dtype)
y = yo[:, :, None] * ya[:, None, :]
mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
O = torch.randn(B, N, dtype=dtype, requires_grad=True)
U = torch.randn(B, N, M, dtype=dtype, requires_grad=True)
R = torch.randn(B, M, dtype=dtype, requires_grad=True)
lo, la = O.log_softmax(-1), U.log_softmax(-1)
S = lo[:, :, None] + la
pair = (-S[bi, o, a] + torch.logsumexp(S[:, mask], -1)).mean()
task = pair - 0.5 * la[bi, o, a].mean() - 0.5 * lo[bi, o].mean()
truth = torch.autograd.grad(task, (U, O), retain_graph=True)
Ps = (S.exp() * mask[None])
Ps = Ps / Ps.sum((1, 2), keepdim=True)
po, rc, qo = lo.exp(), la.exp(), Ps.sum(-1)
manual_U = (Ps - y - rc * (qo - yo)[:, :, None]
            + 0.5 * yo[:, :, None] * (rc - ya[:, None, :])) / B
manual_O = (qo - yo + 0.5 * (po - yo)) / B
errors = {'task_U': float((manual_U - truth[0]).abs().max()),
          'task_O': float((manual_O - truth[1]).abs().max())}
pp, pj = R.softmax(-1), S.exp().sum(1)
mix = (pp + pj) / 2
cent = js_divergence(pp, pj)
truth = torch.autograd.grad(cent, (U, O, R))
zeta, xi = (pj / mix).log(), (pp / mix).log()
zbar = (rc * zeta[:, None]).sum(-1)
manual_U = po[:, :, None] * rc * (zeta[:, None] - zbar[:, :, None]) / (2 * B)
manual_O = po * (zbar - (po * zbar).sum(-1, keepdim=True)) / (2 * B)
manual_R = pp * (xi - (pp * xi).sum(-1, keepdim=True)) / (2 * B)
for name, m, t in zip(('JS_U', 'JS_O', 'JS_R'), (manual_U, manual_O, manual_R), truth):
    errors[name] = float((m - t).abs().max())

cfg = read_json(BASE / 'configs/default.json')
cfg.update(variant='V5', rank=r, adapter_rank=2)
model = VGCPOD(unit(torch.randn(M, d)), unit(torch.randn(N, d)),
                unit(torch.randn(N, M, d)), cfg, 3351).double()
with torch.no_grad():
    for p in model.parameters():
        p.normal_(0, 0.15)
g, fa, fo = unit(torch.randn(B, d, dtype=dtype)), unit(torch.randn(B, M, d, dtype=dtype)), unit(torch.randn(B, N, d, dtype=dtype))
out = model(g, fa, fo)
total = model.objective(out, a, o, mask)['total']
G, Go, H = torch.autograd.grad(total, (out['cond'], out['obj'], out['prim']), retain_graph=True)
u, v = model.ea @ model.Au.T, model.eo @ model.Av.T
gamma = out['gamma']
h = u[None, None] * v[None, :, None] * gamma[:, :, None]
z, za = fo @ model.Wc, fa @ model.Wa
pi, partners = out['pi'], out['pi'] @ v
c, ca = cfg['rho'] * cfg['kappa'], cfg['rho_a'] * cfg['kappa']
du = c * torch.einsum('bji,bjr,jr,bjr->ir', G, z, v, gamma)
du = du + ca * torch.einsum('bi,bir,bir->ir', H, za, partners)
dv = c * torch.einsum('bji,bjr,ir,bjr->jr', G, z, u, gamma)
dv = dv + ca * torch.einsum('bij,bi,bir,ir->jr', pi, H, za, u)
dgamma = c * torch.einsum('bji,bjr,ir,jr->bjr', G, z, u, v)
sig = torch.sigmoid(fo @ model.Bvis.T / cfg['tau_gamma'])
dt = dgamma * 2 * sig * (1 - sig)
manual = {
 'Wc': c * torch.einsum('bji,bjd,bjir->dr', G, fo, h),
 'Wa': ca * torch.einsum('bi,bid,bir->dr', H, fa, u[None] * partners),
 'Au': du.T @ model.ea,
 'Av': dv.T @ model.eo,
 'Bvis': torch.einsum('bjr,bjd->rd', dt, fo) / cfg['tau_gamma']
          + cfg['lambda_orth'] * 4 / r**2 * (model.Bvis @ model.Bvis.T - torch.eye(r, dtype=dtype)) @ model.Bvis,
 'Bo': torch.einsum('bj,bd,jq->dq', Go, g, model.eo @ model.Co),
 'Co': torch.einsum('bj,jd,bq->dq', Go, model.eo, g @ model.Bo),
 'Bc': torch.einsum('bji,bd,jiq->dq', G, g, model.ec @ model.Cc),
 'Cc': torch.einsum('bji,jid,bq->dq', G, model.ec, g @ model.Bc),
}
names = list(manual)
truth = torch.autograd.grad(total, [getattr(model, name) for name in names])
for name, t in zip(names, truth):
    errors[name] = float((manual[name] - t).abs().max())
assert max(errors.values()) < 1e-12, errors
report = {'status': 'passed', 'maximum_absolute_error': max(errors.values()), 'errors': errors}
args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
