"""Equations in mathematical specification §§3,5,6; tensor order [B,O,A]."""
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ('V0', 'V1', 'V1L', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7')


def js_divergence(p, q):
    m = (p + q) * 0.5
    return 0.5 * (p * (p.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())
                  + q * (q.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())).sum(-1).mean()


class VGCPOD(nn.Module):
    def __init__(self, ea, eo, ec, cfg, seed):
        super().__init__()
        self.variant = cfg['variant']
        if self.variant not in VARIANTS:
            raise ValueError('Unknown variant: ' + self.variant)
        self.cfg = dict(cfg)
        self.register_buffer('ea', ea.clone().float())
        self.register_buffer('eo', eo.clone().float())
        self.register_buffer('ec', ec.clone().float())  # [O,A,D]
        m, d = ea.shape
        n = eo.shape[0]
        r, q = cfg['rank'], cfg['adapter_rank']
        if not (1 <= r <= d and 1 <= q <= d) or ec.shape != (n, m, d):
            raise ValueError('Incompatible ranks/text axes')
        # Allocate identical named tensors for every variant; inactive tensors are frozen.
        shapes = {'Bo': (d, q), 'Co': (d, q), 'Bc': (d, q), 'Cc': (d, q),
                  'Au': (r, d), 'Av': (r, d), 'Bvis': (r, d), 'Wc': (d, r), 'Wa': (d, r)}
        generator = torch.Generator().manual_seed(seed)
        for name, shape in shapes.items():
            x = torch.zeros(shape)
            if name in ('Bo', 'Bc'):
                x.normal_(0, 0.02, generator=generator)
            elif name in ('Au', 'Av'):
                x.normal_(0, r ** -0.25, generator=generator)
            elif name == 'Bvis':
                z = torch.randn(d, r, generator=generator)
                basis, upper = torch.linalg.qr(z, mode='reduced')
                signs = upper.diag().sign()
                signs[signs == 0] = 1
                x = (basis * signs).T.contiguous()
            self.register_parameter(name, nn.Parameter(x))
        self.dynamic = self.variant in ('V2', 'V3', 'V4', 'V5', 'V6', 'V7')
        self.primitive = self.variant in ('V4', 'V5', 'V7')
        self.correction = self.variant != 'V0'
        for name in ('Au', 'Av', 'Wc'):
            getattr(self, name).requires_grad_(self.correction)
        self.Bvis.requires_grad_(self.dynamic)
        self.Wa.requires_grad_(self.primitive)

    def forward(self, g, fa, fo):
        c = self.cfg
        kappa, rho = c['kappa'], c['rho']
        obj = kappa * (g @ self.eo.T) + (g @ self.Bo) @ (self.eo @ self.Co).T
        base = kappa * torch.einsum('bd,oid->boi', g, self.ec)
        base = base + torch.einsum('bq,oiq->boi', g @ self.Bc, self.ec @ self.Cc)
        u, v = self.ea @ self.Au.T, self.eo @ self.Av.T
        local = fo if self.variant not in ('V1', 'V3') else g[:, None, :].expand(-1, len(v), -1)
        gamma = torch.ones(g.shape[0], len(v), c['rank'], device=g.device, dtype=g.dtype)
        if self.dynamic:
            gate_input = local.roll(shifts=1, dims=0) if self.variant == 'V6' else local
            gamma = 2 * torch.sigmoid((gate_input @ self.Bvis.T) / c['tau_gamma'])
        cond = base
        if self.correction:
            modes = (local @ self.Wc) * gamma * v[None, :, :]
            cond = base + rho * kappa * torch.einsum('bor,ir->boi', modes, u)
        log_o = obj.log_softmax(-1)
        log_a = cond.log_softmax(-1)
        joint = log_o[:, :, None] + log_a
        result = {'obj': obj, 'base': base, 'cond': cond, 'log_o': log_o,
                  'log_a': log_a, 'joint': joint, 'gamma': gamma}
        if self.primitive:
            if self.variant == 'V4':
                pi = g.new_full((g.shape[0], len(u), len(v)), 1 / len(v))
            else:
                pi = (torch.einsum('bid,jd->bij', fa, self.eo) / c['tau_pi']).softmax(-1)
            partners = pi @ v
            rlogits = kappa * (g @ self.ea.T)
            rlogits = rlogits + c['rho_a'] * kappa * ((fa @ self.Wa) * u[None] * partners).sum(-1)
            result.update(prim=rlogits, pi=pi)
        return result

    def objective(self, out, attrs, objs, train_mask):
        b = torch.arange(len(attrs), device=attrs.device)
        s = out['joint']
        truth = s[b, objs, attrs]
        pair = (-truth + torch.logsumexp(s[:, train_mask], dim=-1)).mean()
        cond = -out['log_a'][b, objs, attrs].mean()
        obj = -out['log_o'][b, objs].mean()
        zero = truth.new_zeros(())
        prim, cent, orth = zero, zero, zero
        if self.primitive:
            prim = F.cross_entropy(out['prim'], attrs)
            cent = js_divergence(out['prim'].softmax(-1), s.exp().sum(1))
        if self.dynamic:
            r = self.cfg['rank']
            orth = (self.Bvis @ self.Bvis.T - torch.eye(r, device=truth.device,
                    dtype=truth.dtype)).square().sum() / (r * r)
        orth_weight = 0.0 if self.variant == 'V7' else self.cfg['lambda_orth']
        total = pair + 0.5 * cond + 0.5 * obj
        total = total + self.cfg['lambda_prim'] * prim + self.cfg['lambda_cent'] * cent + orth_weight * orth
        return {'total': total, 'pair': pair, 'cond': cond, 'object': obj,
                'prim': prim, 'cent': cent, 'orth': orth}

    def trainable_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
