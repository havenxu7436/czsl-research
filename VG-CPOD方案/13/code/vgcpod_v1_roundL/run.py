#!/usr/bin/env python3
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
from ckcpf.common import ROOT
from ckcpf import runtime


def main():
    p = argparse.ArgumentParser(description='Conditional knowledge residual on the verified G0 CPF head')
    p.add_argument('command', choices=['setup','train','report','status'])
    p.add_argument('--cache', default=os.getenv('CPF_CACHE'))
    p.add_argument('--work', default=os.getenv('CPF_WORK', str(ROOT/'work')))
    p.add_argument('--config', default=str(ROOT/'configs/default.json'))
    p.add_argument('--seed', type=int, choices=runtime.SEEDS, default=3351)
    p.add_argument('--variant', choices=['all'] + list(runtime.VARIANTS), default='all')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int)
    args = p.parse_args()
    if args.command == 'setup': runtime.setup(args)
    elif args.command in ('report','status'): runtime.report(args)
    else:
        if not args.cache: p.error('Run setup then source server_env.sh, or pass --cache.')
        runtime.train(args)


if __name__ == '__main__': main()
