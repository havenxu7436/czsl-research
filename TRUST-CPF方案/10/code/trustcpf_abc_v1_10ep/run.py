#!/usr/bin/env python3
import argparse,os
from pathlib import Path
def parser():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description='TRUST-CPF A/B/C: 10epoch Uniform and frozen deployment utility')
    p.add_argument('stage',choices=['setup','prepare-seed','fit','q4-seed','select','evaluate-seed','report','status'])
    p.add_argument('--source',default=os.getenv('CPF_SOURCE','/data/xhw/experiments/trustcpf_stage_v1_10ep'))
    p.add_argument('--cache',default=os.getenv('CPF_CACHE','/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14'))
    p.add_argument('--work',default=os.getenv('CPF_WORK',str(root/'work')))
    p.add_argument('--config',default=os.getenv('CPF_CONFIG',str(root/'configs/default.json')))
    p.add_argument('--seed',type=int,choices=[3351,3352,3353],default=3351)
    p.add_argument('--device',default='cuda')
    return p
def main(args):
    import torch
    torch.set_num_threads(int(os.getenv('CPF_CPU_THREADS','4')))
    from trustcpf import abc_workflow as w
    return {'setup':w.setup,'prepare-seed':w.prepare_seed,'fit':w.fit,'q4-seed':w.q4_seed,
            'select':w.select,'evaluate-seed':w.evaluate_seed,'report':w.report,'status':w.report}[args.stage](args)
if __name__=='__main__':main(parser().parse_args())
