import argparse
from trustg0.common import ROOT
from trustg0 import workflow


def parser():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ('setup','prepare','train-base','teacher','fit','finish','report'):
        s=sub.add_parser(name);s.add_argument('--work',default=str(ROOT/'work'));s.add_argument('--device',default='cuda')
        if name=='setup':
            s.add_argument('--source',default='/data/xhw/experiments/trustcpf_pure_v1_10ep');s.add_argument('--config',default=str(ROOT/'configs/experiment.json'))
        elif name!='report':s.add_argument('--dataset',required=True);s.add_argument('--seed',type=int,default=5351)
        if name=='teacher':s.add_argument('--teacher',type=int,required=True);s.add_argument('--fold',type=int,required=True)
    return p


if __name__=='__main__':
    args=parser().parse_args();getattr(workflow,args.command.replace('-','_'))(args)
