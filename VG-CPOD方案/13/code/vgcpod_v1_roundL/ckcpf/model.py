"""G0 static CP with a conditional-only knowledge prototype residual."""
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalCPF(nn.Module):
    def __init__(self, initial, bank, variant, cfg, seed):
        super().__init__()
        if variant not in ('base', 'knowledge', 'shuffle'):
            raise ValueError(variant)
        self.variant = variant
        self.cfg = dict(cfg)
        state = initial['model']
        self.kappa, self.rho = initial['config']['kappa'], initial['config']['rho']
        for key in ('ea', 'eo', 'ec'):
            self.register_buffer(key, state[key].float().clone())
        for key in ('Bo', 'Co', 'Bc', 'Cc', 'Au', 'Av', 'Wc'):
            # Keep the learned primitive geometry fixed; train the conditional readout.
            self.register_parameter(key, nn.Parameter(state[key].float().clone(), requires_grad=key not in ('Bo', 'Co', 'Au', 'Av')))
        if variant != 'base':
            d = self.ea.shape[1]
            r = cfg['message_rank']
            gen = torch.Generator().manual_seed(seed + 17001)
            self.left = nn.Parameter(torch.randn(3, d, r, generator=gen) * .01)
            self.right = nn.Parameter(torch.zeros(3, r, d))
            self.alpha = nn.Parameter(torch.zeros(()))
            for key in ('source', 'target', 'types', 'degree'):
                self.register_buffer(key, bank[key].clone())
            self.register_buffer('incidence', F.one_hot(bank['target'], len(self.ea) + len(self.eo)).float().T)

    def prototypes(self):
        anchor = torch.cat([self.ea, self.eo])
        diff = self.source - anchor[self.target]
        low = torch.einsum('erd,ed->er', self.right[self.types], diff)
        messages = diff + torch.einsum('edr,er->ed', self.left[self.types], low)
        return F.normalize(anchor + self.cfg['eta'] * (self.incidence @ (messages / self.degree[:, None])), dim=-1, eps=1e-8)

    def cp(self, g, ea, eo):
        # Preserve the original G0 operation order and unnormalized CP factors.
        u, v = ea @ self.Au.T, eo @ self.Av.T
        readout = g[:, None, :].expand(-1, len(v), -1)
        modes = (readout @ self.Wc) * v[None, :, :]
        return self.rho * self.kappa * torch.einsum('bor,ir->boi', modes, u)

    def forward(self, g):
        obj = self.kappa * (g @ self.eo.T) + (g @ self.Bo) @ (self.eo @ self.Co).T
        cond = self.kappa * torch.einsum('bd,oid->boi', g, self.ec)
        cond = cond + torch.einsum('bq,oiq->boi', g @ self.Bc, self.ec @ self.Cc)
        static = self.cp(g, self.ea, self.eo)
        cond = cond + static
        delta = torch.zeros_like(cond)
        if self.variant != 'base':
            z = self.prototypes()
            delta = self.alpha * (self.cp(g, z[:len(self.ea)], z[len(self.ea):]) - static)
            cond = cond + delta
        lo, la = obj.log_softmax(-1), cond.log_softmax(-1)
        return dict(joint=lo[:, :, None] + la, cond=cond, obj=obj, log_o=lo, log_a=la, delta=delta)

    def loss(self, out, a, o, train_mask):
        row = torch.arange(len(a), device=a.device)
        truth = out['joint'][row, o, a]
        pair = (-truth + out['joint'][:, train_mask].logsumexp(-1)).mean()
        cond = -out['log_a'][row, o, a].mean()
        obj = -out['log_o'][row, o].mean()
        beta = self.cfg['beta']
        return 1.5 / (1. + beta) * (pair + beta * (cond + obj))

    def optimizer(self):
        c = self.cfg
        groups = [dict(params=[self.Bc, self.Cc], lr=c['lr_adapter'], initial_lr=c['lr_adapter']),
                  dict(params=[self.Wc], lr=c['lr_cp'], initial_lr=c['lr_cp'])]
        if self.variant != 'base':
            groups += [dict(params=[self.left, self.right], lr=c['lr_message'], initial_lr=c['lr_message']),
                       dict(params=[self.alpha], lr=c['lr_alpha'], initial_lr=c['lr_alpha'], weight_decay=0.)]
        return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, weight_decay=c['weight_decay'])
