"""Restore stage 11 and recompute all three datasets on CPU."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
from register_trustg0_three import ROOT,STAGE,read,restore_results,restore_archive
def compare(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:compare(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):compare(x,y)
 elif isinstance(a,(float,int)) and not isinstance(a,bool):assert np.isclose(a,b,atol=1e-8,rtol=1e-8),(a,b)
 else:assert a==b,(a,b)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dest',required=True);a=p.parse_args();dest=Path(a.dest).resolve()
 if dest.exists():raise ValueError('Choose a new output directory')
 dest.mkdir(parents=True);restore_results(dest/'results');confirmed=STAGE/'analysis/confirmed';dsmap={'UT':'ut-zap50k','MIT':'mit-states','CGQA':'cgqa'};archives=read(STAGE/'results/archives/index.json')
 for label,ds in dsmap.items():
  archive=archives[ds];source=dest/'results'/archive['snapshot_root'];restore_archive(ds,source.parent/archive['name'])
  for engine in ['recompute.py','diagnostics.py']:
   subprocess.run([sys.executable,str(STAGE/'analysis/engines'/engine),'--source',str(source),'--dataset',ds,'--out',str(dest/'analysis'/label)],check=True)
  for name in ['main.json','paired.json','q5_main.json','q5_paired.json']:compare(read(dest/'analysis'/label/name),read(confirmed/label/name))
  compare(read(dest/'analysis'/label/'nll_subgroups.json')['paired'],read(confirmed/label/'nll_subgroups.json')['paired'])
 result=dict(passed=True,datasets=list(dsmap.values()),recognition_rows=1620,q5_rows=378,prediction_files=1998,training_started=False)
 (dest/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
