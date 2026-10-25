#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

def parser():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description='TRUST-CPF: 15 stage-matched teachers x 10 epochs; frozen recognition')
    p.add_argument('stage',choices=['setup','train-seed','fit','evaluate-seed','report','status'])
    p.add_argument('--source',default=os.getenv('CPF_SOURCE','/data/xhw/experiments/trustcpf_state_v1_10ep'))
    p.add_argument('--cache',default=os.getenv('CPF_CACHE','/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'))
    p.add_argument('--work',default=os.getenv('CPF_WORK',str(root/'work')))
    p.add_argument('--config',default=os.getenv('CPF_CONFIG',str(root/'configs/default.json')))
    p.add_argument('--seed',type=int,choices=[3351,3352,3353],default=3351)
    p.add_argument('--device',default='cuda')
    return p

def main(args):
    import torch
    torch.set_num_threads(int(os.getenv('CPF_CPU_THREADS','4')))
    from trustcpf import matched_workflow as workflow
    return {'setup':workflow.setup,'train-seed':workflow.train_seed,'fit':workflow.fit,
            'evaluate-seed':workflow.evaluate_seed,'report':workflow.report,'status':workflow.report}[args.stage](args)

if __name__=='__main__':
    main(parser().parse_args())
