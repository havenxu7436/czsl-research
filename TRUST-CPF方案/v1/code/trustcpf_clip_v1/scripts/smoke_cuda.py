"""768D, 16x12, 96-edge CUDA optimization check on synthetic CLIP feature tensors."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from trustcpf.common import ROOT,read_json,write_json,seed_all,unit
from trustcpf.graph import graph_records,assemble_bank,solve_budget
from trustcpf.model import TrustCPF,task_loss,average_graphs
from trustcpf.utility import UtilityNet,utility_nll


def main():
    seed_all(3351);torch.set_num_threads(4)
    device='cuda'
    labels=read_json(ROOT/'assets/labels.json')
    knowledge={k:read_json(ROOT/'assets/knowledge'/(k+'_dict.json')) for k in ('oi','oe','ae')}
    rows=graph_records(labels,knowledge)
    text={'ea':unit(torch.randn(16,768)),'eo':unit(torch.randn(12,768)),'ec':unit(torch.randn(12,16,768))}
    bank=assemble_bank(text,unit(torch.randn(96,768)),rows)
    cfg=read_json(ROOT/'configs/default.json')
    model=TrustCPF(text,bank,cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    batch=[unit(torch.randn(8,768,device=device)),unit(torch.randn(8,16,768,device=device)),unit(torch.randn(8,12,768,device=device))]
    a=torch.arange(8,device=device);o=torch.arange(8,device=device)
    losses=[];torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        j,lo,lc=model(*batch,torch.bernoulli(torch.full((96,),.5,device=device)))
        loss=task_loss(j,lo,lc,a,o,torch.arange(192,device=device))
        opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        lp,var=average_graphs(model,batch,torch.full((96,),.5,device=device),torch.arange(192,device=device),4)
        error=float((lp.exp().sum(-1)-1).abs().max())
    net=UtilityNet(6157,torch.zeros(6157),torch.ones(6157)).to(device)
    m,v=net(torch.randn(24,6157,device=device))
    uloss=utility_nll(m,v,torch.randn(24,device=device),torch.full((24,),.01,device=device),torch.ones(24,device=device),{'sampled':24,'total':96})
    uloss.backward()
    assert error<1e-5 and all(torch.isfinite(torch.tensor(losses)))
    write_json(ROOT/'validation/smoke_cuda.json',{'device':torch.cuda.get_device_name(),
        'torch':torch.__version__,'shape':[8,12,16],'dimension':768,'edges':96,
        'optimization_steps':3,'losses':losses,'probability_sum_error':error,
        'utility_input_dimension':6157,'utility_loss':float(uloss.detach()),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'kind':'synthetic CLIP-space tensors; CUDA implementation check'})
    print('CUDA_SMOKE_OK',error,flush=True)


if __name__=='__main__':main()
