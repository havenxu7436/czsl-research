"""Write an explicitly INVALID-UNTIL-FILLED feasibility input template."""
import argparse
from pathlib import Path
from common import read,write

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve existing feasibility input')
    labels=read(Path(__file__).absolute().parent/'assets/logic_bank/manifest.json')['labels']
    O,A=len(labels['objs']),len(labels['attrs'])
    write(a.out,dict(labels=labels,score_axes=['object','attribute'],scores=[[.5]*A for _ in range(O)],
          known=[[False]*A for _ in range(O)],provenance=dict(generator='',prompt='',generated_at='',
          used_split_membership=False,used_validation_or_test_results=False),
          notice='Fill genuine independent all-cartesian feasibility scores and provenance. known=False is unknown, never a negative. Empty template is rejected. No train/val/test pair membership may be supplied to the generator.'))
