#!/usr/bin/env python3
import argparse,os
from pathlib import Path

def parser():
    p=argparse.ArgumentParser(description='Model-conditioned TRUST on the completed Strong v1 experiment')
    p.add_argument('stage',choices=['setup','run-seed','report','status'])
    p.add_argument('--source',default=os.getenv('CPF_SOURCE',str(Path.home()/'experiments/trustcpf_strong_v1')))
    p.add_argument('--cache',default=os.getenv('CPF_CACHE','/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'))
    p.add_argument('--work',default=os.getenv('CPF_WORK',str(Path(__file__).parent/'work')))
    p.add_argument('--config',default=str(Path(__file__).parent/'configs/default.json'))
    p.add_argument('--seed',type=int,choices=[3351,3352,3353],default=3351)
    p.add_argument('--device',default='cuda')
    return p

def main(args):
    import torch
    torch.set_num_threads(int(os.getenv('CPF_CPU_THREADS','4')))
    from trustcpf import workflow
    return {'setup':workflow.setup,'run-seed':workflow.run_seed,'status':workflow.report,'report':workflow.report}[args.stage](args)

if __name__=='__main__':main(parser().parse_args())
