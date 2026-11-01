"""Historical G0 (VG-CPOD V1) plus its original relation residual."""
import torch
from torch import nn
from torch.nn import functional as F
from .g0 import VGCPOD


def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-8)


class TypedGraph(nn.Module):
    def __init__(self,bank,rank=16,eta=.1):
        super().__init__()
        for key in ('anchors','source','target','types','degree'):self.register_buffer(key,bank[key].clone())
        d=bank['anchors'].shape[1]
        self.left=nn.Parameter(torch.randn(3,d,rank)*.01)
        self.right=nn.Parameter(torch.zeros(3,rank,d));self.eta=eta
        incoming=[torch.where(bank['target']==j)[0].tolist() for j in range(len(bank['anchors']))]
        width=max(map(len,incoming));pad=len(bank['target'])
        self.register_buffer('incoming',torch.tensor([v+[pad]*(width-len(v)) for v in incoming]))
        order=[]
        for t in range(3):
            ix=torch.where(bank['types']==t)[0];self.register_buffer('type_ids_'+str(t),ix);order.extend(ix.tolist())
        self.register_buffer('restore_edge_order',torch.argsort(torch.tensor(order)))

    def messages(self):
        diff=self.source-self.anchors[self.target]
        parts=[]
        for t in range(3):
            x=diff[getattr(self,'type_ids_'+str(t))]
            parts.append(x+(x@self.right[t].T)@self.left[t].T)
        return torch.cat(parts,0)[self.restore_edge_order]

    def raw_nodes(self,edges,messages=None):
        msg=self.messages() if messages is None else messages
        weighted=msg*(edges/self.degree)[:,None]
        padded=torch.cat((weighted,weighted.new_zeros(1,weighted.shape[-1])),0)
        return self.anchors+self.eta*padded[self.incoming].sum(1)

    def forward(self,edges):return unit(self.raw_nodes(edges))
    def regularizer(self):return (self.left@self.right).square().sum()/self.anchors.shape[1]**2


class TrustG0(nn.Module):
    def __init__(self,text,bank,cfg,seed):
        super().__init__();self.cfg=cfg;self.na=len(text['ea']);self.no=len(text['eo'])
        self.backbone=VGCPOD(text['ea'],text['eo'],text['ec'],cfg['g0'],seed)
        self.backbone.requires_grad_(False)
        self.graph=TypedGraph(bank,cfg['message_rank'],cfg['eta'])
        self.edge_targets=bank['target'].tolist()
        d=text['ea'].shape[1];r=cfg['graph_rank']
        self.pv=nn.Linear(d,r,bias=False);self.po=nn.Linear(d,r,bias=False);self.pa=nn.Linear(d,r,bias=False)

    def base(self,g,fo):
        with torch.no_grad():out=self.backbone(g,None,None)
        context=unit(unit(g)[:,None]+self.cfg['object_context']*unit(fo))
        return dict(g=g,obj=out['obj'],cond=out['cond'],context=context)

    def prepared(self,base,edges,capacity=False):
        anchors=self.graph.anchors;aa,ao=anchors[:self.na],anchors[self.na:]
        z=self.graph(edges);za,zo=z[:self.na],z[self.na:]
        qx=unit(self.pv(base['context']))
        def tri(o,a):return torch.einsum('bor,or,ar->boa',qx,unit(self.po(o)),unit(self.pa(a)))
        do=base['g']@(zo-ao).T;dc=tri(zo,za)-tri(ao,aa)
        if capacity:
            zo=unit(ao+.1*((ao@self.graph.right[0].T)@self.graph.left[0].T))
            za=unit(aa+.1*((aa@self.graph.right[1].T)@self.graph.left[1].T+(aa@self.graph.right[2].T)@self.graph.left[2].T))
            do=base['g']@(zo-ao).T;dc=tri(zo,za)
        obj=base['obj']+self.cfg['gain']*do;cond=base['cond']+self.cfg['gain']*dc
        return dict(obj=obj,cond=cond,qx=qx,qo=unit(self.po(zo)),qa=unit(self.pa(za)),z=z)

    def score(self,base,edges,capacity=False):
        v=self.prepared(base,edges,capacity);lo=v['obj'].log_softmax(-1);lc=v['cond'].log_softmax(-1)
        return lo[:,:,None]+lc,lo,lc


def select_pairs(joint,pairs,no):return joint[:,pairs%no,pairs//no]


def task_loss(joint,lo,lc,attrs,objs,candidates):
    rows=torch.arange(len(attrs),device=attrs.device)
    pair=(-joint[rows,objs,attrs]+select_pairs(joint,candidates,joint.shape[1]).logsumexp(-1)).mean()
    return 1.5/1.075*(pair+.075*(F.nll_loss(lo,objs)+F.nll_loss(lc[rows,objs],attrs)))


def parameters_state(model):return {k:v.detach().cpu().clone() for k,v in model.named_parameters()}


def restore_parameters(model,state):
    current=dict(model.named_parameters())
    if set(current)!=set(state):raise ValueError('Parameter schema changed')
    with torch.no_grad():
        for k,p in current.items():p.copy_(state[k].to(p.device))
