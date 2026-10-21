"""Check a proposed separate readout against actual I2 weights; no training sources are edited."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'server_results_roundI_20261021'
sys.path.insert(0,str(ARCHIVE/'vgcpod_v1_roundI'))
from vgcpod.common import load_torch
from vgcpod.model import VGCPOD

torch.set_num_threads(1)


def main():
    records=[]
    for seed in (3351,3352,3353):
        saved=load_torch(ARCHIVE/'roundI_20261021/I2/ut-zap50k/V1'/('seed%d'%seed)/'checkpoints/best.pt')
        state=saved['model'];cfg=saved['config']
        model=VGCPOD(state['ea'],state['eo'],state['ec'],cfg,seed,gate_mean=state['gate_fo_mean']).double()
        model.load_state_dict({k:v.double() for k,v in state.items()})
        gen=torch.Generator().manual_seed(seed+81)
        g=torch.nn.functional.normalize(torch.randn(7,768,dtype=torch.float64,generator=gen),dim=-1)
        fa=torch.nn.functional.normalize(torch.randn(7,16,768,dtype=torch.float64,generator=gen),dim=-1)
        fo=model.gate_fo_mean[None]+.18*torch.nn.functional.normalize(torch.randn(7,12,768,dtype=torch.float64,generator=gen),dim=-1)
        with torch.no_grad():
            result=model(g,fa,fo);h=result['gamma']-1
            u,v=model.ea@model.Au.T,model.eo@model.Av.T
            def separate(ws,wd):
                static=(g@ws)[:,None,:]*v[None]
                dynamic=(g@wd)[:,None,:]*v[None]*h
                return cfg['rho']*cfg['kappa']*torch.einsum('bor,ir->boi',static+dynamic,u)
            old=result['cond']-result['base']
            tied_error=float((old-separate(model.Wc,model.Wc)).abs().max())
            static_direct=cfg['rho']*cfg['kappa']*torch.einsum('br,jr,ir->bji',g@model.Wc,v,u)
            static_error=float((separate(model.Wc,torch.zeros_like(model.Wc))-static_direct).abs().max())
        # Score derivative identifies how the tied matrix receives both contributions.
        gg=g[0].detach();uv=(u[0]*v[0]).detach();hh=h[0,0].detach();scale=cfg['rho']*cfg['kappa']
        ws=model.Wc.detach().clone().requires_grad_();wd=ws.detach().clone().requires_grad_()
        score=scale*(gg@ws@uv+gg@wd@(uv*hh))
        gs,gd=torch.autograd.grad(score,(ws,wd))
        gs_error=float((gs-scale*gg[:,None]*uv[None]).abs().max())
        gd_error=float((gd-scale*gg[:,None]*(uv*hh)[None]).abs().max())
        tied=model.Wc.detach().clone().requires_grad_()
        tied_score=scale*(gg@tied@(uv*(1+hh)))
        combined=torch.autograd.grad(tied_score,tied)[0]
        gradient_sum_error=float((combined-gs-gd).abs().max())
        assert max(tied_error,static_error,gs_error,gd_error,gradient_sum_error)<1e-12
        records.append(dict(seed=seed,tied_readout_max_error=tied_error,zero_dynamic_max_error=static_error,
            static_score_gradient_error=gs_error,dynamic_score_gradient_error=gd_error,
            tied_gradient_sum_error=gradient_sum_error,current_parameters=model.trainable_count(),
            proposed_parameters=model.trainable_count()+model.Wc.numel()))
    report=dict(verification='CPU float64 algebra on actual I2 checkpoints and synthetic inputs',
        records=records,experiment_training_performed=False,experiment_source_modified=False,
        hypothesis='Separate static/dynamic readout matrices with shared attribute/object factors')
    (OUT/'next_equations_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
