#!/usr/bin/env python3
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
from pathlib import Path

from vgcpod.common import ROOT, read_json
from vgcpod.model import VARIANTS


def main():
    parser = argparse.ArgumentParser(description='VG-CPOD v1 frozen CLIP experiment')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('doctor', 'prepare'):
        p = commands.add_parser(command)
        p.add_argument('--data-root', required=True)
        p.add_argument('--checkpoint', required=command == 'prepare')
        if command == 'prepare':
            p.add_argument('--cache', required=True)
            p.add_argument('--device', default='cuda')
            p.add_argument('--batch-size', type=int, default=16)
            p.add_argument('--shard-size', type=int, default=512)
            p.add_argument('--workers', type=int, default=4)
            p.add_argument('--tau-v', type=float, default=0.07)
    for command in ('train', 'smoke', 'baselines'):
        p = commands.add_parser(command)
        p.add_argument('--cache', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--config', default=str(ROOT / 'configs' / 'default.json'))
        p.add_argument('--device', default='cuda')
        p.add_argument('--seed', type=int, default=3351)
        p.add_argument('--variant', choices=VARIANTS)
        p.add_argument('--rank', type=int)
        p.add_argument('--tau-pi', type=float)
        p.add_argument('--workers', type=int)
        if command == 'train':
            p.add_argument('--resume', action='store_true')
    p = commands.add_parser('evaluate')
    p.add_argument('--cache', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--device', default='cuda')
    p = commands.add_parser('summarize')
    p.add_argument('--out', required=True)
    p = commands.add_parser('diagnose')
    p.add_argument('--cache', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--device', default='cuda')
    p = commands.add_parser('compare')
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[3351, 3352, 3353])
    args = parser.parse_args()
    if args.command in ('doctor', 'prepare'):
        from vgcpod.features import preflight, prepare
        if args.command == 'doctor':
            preflight(args)
        else:
            if min(args.batch_size, args.shard_size) <= 0 or args.tau_v <= 0:
                parser.error('Encoding batch/shard sizes and tau-v must be positive')
            prepare(args)
        return
    from vgcpod import engine
    if args.command in ('diagnose', 'compare'):
        from vgcpod import diagnostics
        getattr(diagnostics, args.command)(args)
    elif args.command == 'evaluate':
        engine.evaluate_run(args.cache, args.run, args.device)
    elif args.command == 'summarize':
        engine.summarize(args)
    else:
        cfg = read_json(args.config)
        cfg['variant'] = args.variant or cfg['variant']
        if not 0 <= cfg.get('task_aux_beta', 0.5) <= 1:
            parser.error('task_aux_beta must be in [0,1]')
        if not 0 < cfg.get('cp_lr_scale', 1.0) <= 1:
            parser.error('cp_lr_scale must be in (0,1]')
        for name in ('rank', 'tau_pi', 'workers'):
            if getattr(args, name) is not None:
                cfg[name] = getattr(args, name)
        if cfg['dataset'] != 'ut-zap50k':
            parser.error('This v1 data adapter is for UT-Zappos natural split')
        if min(cfg[k] for k in ('batch_size', 'eval_batch_size', 'epochs', 'rank', 'adapter_rank')) <= 0:
            parser.error('Batch sizes, epochs and ranks must be positive')
        if cfg['workers'] < 0 or min(cfg[k] for k in ('tau_v', 'tau_gamma', 'tau_pi')) <= 0:
            parser.error('Workers must be nonnegative; temperatures must be positive')
        getattr(engine, args.command)(args, cfg)


if __name__ == '__main__':
    main()
