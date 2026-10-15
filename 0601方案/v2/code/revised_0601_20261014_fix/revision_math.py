"""Corrected probability algebra; Python 3.8 / torch 1.13 compatible."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def joint_log_probability(obj, conditional, prior=None, weight=0.0):
    if obj.ndim != 2 or conditional.ndim != 3 or conditional.shape[:2] != obj.shape:
        raise ValueError('Expected object BxO and conditional BxOxA')
    obj, conditional = obj.float(), conditional.float()
    if not torch.isfinite(obj).all() or not torch.isfinite(conditional).all():
        raise ValueError('Nonfinite logits')
    if prior is not None:
        if prior.shape != conditional.shape[1:] or not torch.isfinite(prior).all():
            raise ValueError('Invalid prior axes/values')
        conditional = conditional + weight * prior.float()[None]
    lo = F.log_softmax(obj, dim=-1)
    lc = F.log_softmax(conditional, dim=-1)
    return lo[:, :, None] + lc, lo, lc


def make_prior(k, mode='centered', eps=1e-8, floor=1e-3):
    k = k.float()
    if k.ndim != 2 or not torch.isfinite(k).all():
        raise ValueError('Prior requires finite OxA matrix')
    if mode == 'centered':
        # Population std is well-defined for a one-element matrix.
        return (k - k.mean()) / (k.std(unbiased=False) + eps)
    if mode == 'clamped_log':
        if not 0 < floor < 1 or (k < 0).any() or (k > 1).any():
            raise ValueError('clamped_log requires K in [0,1], floor in (0,1)')
        value = k.clamp(min=floor, max=1).log()
        return value - value.mean(dim=-1, keepdim=True)
    raise ValueError('Unknown prior transform: ' + mode)


def js_divergence(logp, logq):
    # Mean over samples, sum over categorical events. No detached branch.
    if logp.shape != logq.shape:
        raise ValueError('JS probability axes differ')
    logm = torch.logaddexp(logp.float(), logq.float()) - math.log(2)
    return (0.5 * (logp.exp() * (logp - logm) +
                   logq.exp() * (logq - logm)).sum(-1)).mean()


def descriptor_logprob(image, descriptions, mask, temperature=0.1):
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError('Invalid description temperature')
    if mask.shape != descriptions.shape[:2] or not mask.any(-1).all():
        raise ValueError('Every category requires at least one active descriptor')
    cosine = torch.einsum('bd,ckd->bck', F.normalize(image.float(), dim=-1),
                          F.normalize(descriptions.float(), dim=-1))
    # Masked mean; removed slots contribute neither score nor denominator.
    z = (cosine * mask[None]).sum(-1) / mask.sum(-1)[None]
    return F.log_softmax(z / temperature, dim=-1)


def aggregate_descriptions(names, descriptions, mask, method='mean', temperature=0.5):
    names = F.normalize(names.float(), dim=-1)
    values = torch.cat((names[:, None], F.normalize(descriptions.float(), dim=-1)), dim=1)
    valid = torch.cat((torch.ones(len(names), 1, dtype=torch.bool, device=mask.device), mask.bool()), dim=1)
    if method == 'mean':
        weights = valid.float() / valid.sum(-1, keepdim=True)
    elif method == 'attention':
        if temperature <= 0:
            raise ValueError('Invalid aggregation temperature')
        z = torch.einsum('cd,ckd->ck', names, values) / temperature
        weights = z.masked_fill(~valid, -torch.inf).softmax(-1)
    else:
        raise ValueError('Unknown aggregation')
    pooled = (values * weights[:, :, None]).sum(1)
    if (pooled.norm(dim=-1) < 1e-8).any():
        raise ValueError('Aggregation collapsed to zero')
    return F.normalize(pooled, dim=-1)


def deletion_mask(n_classes, n_desc, fraction, seed):
    if not 0 <= fraction <= 1:
        raise ValueError('Deletion fraction outside [0,1]')
    order = torch.randperm(n_classes * n_desc, generator=torch.Generator().manual_seed(seed))
    mask = torch.ones(n_classes * n_desc, dtype=torch.bool)
    mask[order[:int(round(n_classes * n_desc * fraction))]] = False
    return mask.reshape(n_classes, n_desc)


def beta_at(epoch, maximum, delay=0, ramp=0):
    if epoch < 1 or maximum < 0 or delay < 0 or ramp < 0:
        raise ValueError('Invalid schedule')
    if epoch <= delay:
        return 0.0
    return maximum * min(1.0, (epoch - delay) / max(1, ramp))


class CoupledHead(nn.Module):
    """Image-dependent object/attribute interaction; no BxOxAxD expansion."""
    def __init__(self, dim=300, rank=128, temperature=0.05, chunk=32, static=False):
        super().__init__()
        if rank < 1 or chunk < 1 or temperature <= 0:
            raise ValueError('Invalid head parameters')
        self.q = nn.Linear(dim * 2, rank, bias=False)
        self.ko = nn.Linear(dim, rank, bias=False)
        self.ka = nn.Linear(dim, rank, bias=False)
        self.qp = nn.Linear(dim * 2, rank, bias=False)
        self.kp = nn.Linear(dim, rank, bias=False)
        self.rank, self.temperature, self.chunk, self.static = rank, temperature, chunk, static

    def forward(self, va, vo, ea, eo, ep=None):
        visual = torch.cat((vo, va), -1)
        q, qp = F.normalize(self.q(visual), dim=-1), F.normalize(self.qp(visual), dim=-1)
        if self.static:
            q, qp = torch.ones_like(q) / math.sqrt(self.rank), torch.ones_like(qp) / math.sqrt(self.rank)
        ko, ka = F.normalize(self.ko(eo), dim=-1), F.normalize(self.ka(ea), dim=-1)
        pieces = []
        for start in range(0, len(eo), self.chunk):
            stop = min(start + self.chunk, len(eo))
            z = math.sqrt(self.rank) * torch.einsum('br,or,ar->boa', q, ko[start:stop], ka)
            if ep is not None:
                kp = F.normalize(self.kp(ep[start:stop]), dim=-1)
                z = z + torch.einsum('br,oar->boa', qp, kp)
            pieces.append(z / self.temperature)
        return torch.cat(pieces, 1)
