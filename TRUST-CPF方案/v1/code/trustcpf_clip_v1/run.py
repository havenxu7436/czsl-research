#!/usr/bin/env python3
"""Server entry point. Every stage is independently runnable and composes into pipeline."""
import argparse
from pathlib import Path


def parser():
    p=argparse.ArgumentParser(description='TRUST-CPF CLIP v1')
    p.add_argument('stage',choices=['preflight','features','graph','teachers','measure','fit','audit','tune',
                                  'reference','structure','adapt','noise','pipeline','report'])
    p.add_argument('--cache',required=True)
    p.add_argument('--work',default='work/seed3351')
    p.add_argument('--config',default=str(Path(__file__).parent/'configs/default.json'))
    p.add_argument('--data-root'); p.add_argument('--checkpoint'); p.add_argument('--knowledge')
    p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int)
    p.add_argument('--variant',choices=['r0','capacity','dense','drop','free','mean','robust','shuffled'])
    p.add_argument('--fold',default='all',choices=['all','1','2','3','4','5'])
    p.add_argument('--fit-all',action='store_true'); p.add_argument('--quick',action='store_true')
    p.add_argument('--batch-size',type=int,default=16); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--shard-size',type=int,default=512); p.add_argument('--tau-v',type=float,default=.07)
    p.add_argument('--noise-seeds',type=int,nargs='+',default=[7101,7102,7103,7104,7105])
    p.add_argument('--rates',type=float,nargs='+',default=[0.,.1,.2,.3,.5]); p.add_argument('--deletion',action='store_true')
    return p


def main(args):
    from trustcpf import features, graph, engine, stages
    dispatch={'preflight':features.preflight,'features':features.prepare,'graph':graph.prepare_bank,
        'teachers':engine.teachers,'reference':engine.reference,'measure':stages.measure,
        'fit':stages.fit_utility,'audit':stages.utility_audit,'tune':stages.tune,
        'structure':stages.structure,'adapt':stages.adapt,'noise':stages.noise}
    if args.stage=='pipeline':
        # Cache preparation is run once before launching a seed matrix.
        engine.teachers(args); stages.measure(args); args.fit_all=False
        stages.fit_utility(args); stages.utility_audit(args); stages.tune(args)
        args.fit_all=True; stages.fit_utility(args); engine.reference(args); stages.structure(args)
        original=args.variant
        for v in (['shuffled'] if original=='shuffled' else ['r0','capacity','dense','drop','free','mean','robust']):
            args.variant=v; stages.adapt(args)
        args.variant=original
    elif args.stage=='report':
        from trustcpf.report import report
        report(args)
    else: dispatch[args.stage](args)


if __name__=='__main__':
    main(parser().parse_args())
