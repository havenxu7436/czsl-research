"""CLIP conditional baseline plus typed graph and image-object-attribute residuals."""
import torch
from torch import nn
from torch.nn import functional as F
from .common import unit, ROOT, read_json
from .g0 import VGCPOD


class Adapter(nn.Module):
    def __init__(self, dim, rank=8):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return unit(x + .1 * self.up(F.gelu(self.down(x))))


class TypedGraph(nn.Module):
    def __init__(self, bank, rank=16, eta=.1):
        super().__init__()
        for key in ('anchors', 'source', 'target', 'types', 'degree'):
            self.register_buffer(key, bank[key].clone())
        d = bank['anchors'].shape[1]
        self.left = nn.Parameter(torch.randn(3, d, rank) * .01)
        self.right = nn.Parameter(torch.zeros(3, rank, d))
        self.eta = eta

    def messages(self):
        diff = self.source - self.anchors[self.target]
        low = torch.einsum('erd,ed->er', self.right[self.types], diff)
        return diff + torch.einsum('edr,er->ed', self.left[self.types], low)

    def forward(self, edges):
        msg = self.messages() * (edges / self.degree)[:, None]
        # Dense incidence avoids CUDA scatter nondeterminism for this small graph.
        incidence = F.one_hot(self.target, len(self.anchors)).to(msg.dtype).T
        return unit(self.anchors + self.eta * (incidence @ msg))

    def regularizer(self):
        delta = self.left @ self.right
        return delta.square().sum() / self.anchors.shape[1] ** 2


class TrustCPF(nn.Module):
    def __init__(self, text, bank, cfg):
        super().__init__()
        for k in ('ea', 'eo', 'ec'):
            self.register_buffer(k, text[k].clone())
        d = text['ea'].shape[1]
        self.na, self.no = len(text['ea']), len(text['eo'])
        self.cfg = cfg
        gcfg=read_json(ROOT/'initial/g0_seed3351_protocol.json')['config'].copy()
        gcfg.update(rank=cfg.get('backbone_rank',min(32,d)),adapter_rank=cfg.get('backbone_adapter_rank',min(8,d)))
        self.backbone=VGCPOD(text['ea'],text['eo'],text['ec'],gcfg,cfg.get('seed',3351))
        self.backbone.requires_grad_(False)
        self.graph = TypedGraph(bank, cfg.get('message_rank', 16), cfg.get('eta', .1))
        r = cfg.get('rank', 64)
        self.pv = nn.Linear(d, r, bias=False)
        self.po = nn.Linear(d, r, bias=False)
        self.pa = nn.Linear(d, r, bias=False)

    def base(self, g, fa, fo):
        out=self.backbone(g,fa,fo)
        return {'g':g,'obj':out['obj'],'cond':out['cond'],
                'context':unit(unit(g)[:,None,:]+self.cfg.get('object_context',.5)*unit(fo))}

    def score(self, base, edges, capacity=False):
        anchors = self.graph.anchors
        z = self.graph(edges) if torch.count_nonzero(edges) else anchors
        ao, aa = anchors[self.na:], anchors[:self.na]
        zo, za = z[self.na:], z[:self.na]
        qx = unit(self.pv(base['context']))
        def tri(o, a):
            return torch.einsum('bor,or,ar->boa', qx, unit(self.po(o)), unit(self.pa(a)))
        delta_o = base['g'] @ (zo - ao).T
        delta_u = tri(zo, za) - tri(ao, aa)
        if capacity:
            # Same effective projection matrices; no relation attachment is used.
            o = unit(ao + .1 * ((ao @ self.graph.right[0].T) @ self.graph.left[0].T))
            a = unit(aa + .1 * ((aa @ self.graph.right[1].T) @ self.graph.left[1].T
                            + (aa @ self.graph.right[2].T) @ self.graph.left[2].T))
            delta_o = base['g'] @ (o - ao).T
            delta_u = tri(o, a)  # generic trainable three-way head
        gain = self.cfg.get('alpha', .1) / self.cfg.get('tg', .1)
        obj = base['obj'] + gain * delta_o
        conditional = base['cond'] + gain * delta_u
        lo, lc = obj.log_softmax(-1), conditional.log_softmax(-1)
        return lo[:, :, None] + lc, lo, lc

    def forward(self, g, fa, fo, edges, capacity=False):
        return self.score(self.base(g, fa, fo), edges, capacity)


def task_loss(joint, lo, lc, attrs, objs, candidates):
    flat = joint.flatten(1)
    target = objs * joint.shape[2] + attrs
    lp = -flat.gather(1, target[:, None]).mean() + flat[:, candidates].logsumexp(-1).mean()
    rows = torch.arange(len(attrs), device=attrs.device)
    return 1.5/1.075*(lp+.075*(F.nll_loss(lo,objs)+F.nll_loss(lc[rows,objs],attrs)))


def candidate_logprob(joint, candidates):
    scores = joint.flatten(1)[:, candidates]
    return scores - scores.logsumexp(-1, keepdim=True)


def task_logit_gradients(obj_logits, conditional_logits, attrs, objs, candidates):
    """Analytic task-loss derivatives; used to verify the autograd implementation."""
    po,pc=obj_logits.softmax(-1),conditional_logits.softmax(-1)
    joint=obj_logits.log_softmax(-1)[:,:,None]+conditional_logits.log_softmax(-1)
    h=torch.zeros_like(joint).flatten(1)
    h[:,candidates]=candidate_logprob(joint,candidates).exp()
    row=torch.arange(len(attrs),device=attrs.device)
    h[row,objs*joint.shape[2]+attrs]-=1
    h=h.reshape_as(joint); by_object=h.sum(-1)
    go=by_object+.075*po
    go[row,objs]-=.075
    gu=h-pc*by_object[:,:,None]
    gu[row,objs]+=.075*pc[row,objs]
    gu[row,objs,attrs]-=.075
    return 1.5/1.075*go/len(attrs),1.5/1.075*gu/len(attrs)


def average_graphs(model, batch, pi, candidates, count=16, seed=1701, mode='probability', capacity=False, return_joint=False):
    base = model.base(*batch)
    gen = torch.Generator(device=pi.device).manual_seed(seed)
    outputs = []
    for _ in range(count):
        edges = pi if mode == 'mean_graph' else (torch.rand(pi.shape, generator=gen, device=pi.device) < pi).float()
        outputs.append(model.score(base, edges, capacity)[0])
    stack = torch.stack(outputs)
    if mode in ('mean_graph', 'mean_logit'):
        logs = candidate_logprob(stack.mean(0), candidates)
    else:
        logs = torch.stack([candidate_logprob(j, candidates) for j in outputs]).logsumexp(0) - torch.log(pi.new_tensor(count))
    probs = torch.stack([candidate_logprob(j, candidates).exp() for j in outputs])
    variance = (probs - probs.mean(0)).square().sum(-1).mean(0)
    if return_joint:
        full = stack.mean(0) if mode in ('mean_graph','mean_logit') else stack.logsumexp(0)-torch.log(pi.new_tensor(count))
        return logs, variance, full
    return logs, variance
