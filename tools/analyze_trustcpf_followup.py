"""Recompute archived Align, Competition and R5 predictions and utility audits on CPU."""
import argparse,math,os,shutil,subprocess,sys
from pathlib import Path
from register_trustcpf_followup import ROOT,SPECS,read,write,stage,safe_new,safe_child,restore_archive,materialize,sha
ANALYSIS={'05':'analysis_TRUST_CPF_align_v1_20261024','06':'analysis_TRUST_CPF_competition_10ep_20261024','07':'analysis_TRUST_CPF_r5_10ep_20261024'}
RUNS={'05':['analyze.py'],'06':['verify_results.py','verify_mechanism.py','diagnose_transfer.py','summarize.py'],'07':['verify_results.py','verify_mechanism.py','summarize.py','diagnostics.py']}
COMPARE={'05':['analysis.json','prediction_verification.json','audit_verification.json'],
         '06':['analysis.json','prediction_verification.json','mechanism_verification.json','transfer_diagnostic.json'],
         '07':['analysis.json','prediction_verification.json','mechanism_verification.json','diagnostics.json']}
def same(a,b,path='root'):
 if isinstance(a,dict):
  assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
  for k in a:same(a[k],b[k],path+'.'+k)
 elif isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b),(path,'length')
  for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+f'[{i}]')
 elif isinstance(a,(int,float)) and not isinstance(a,bool):assert isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(path,a,b)
 else:assert a==b,(path,a,b)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=list(SPECS),required=True);ap.add_argument('--dest',required=True);ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
 dest=safe_new(a.dest);dest.mkdir(parents=True);s=stage(a.stage);arc=read(s/'results/archives/archive.json');stem=arc['name'][:-7]
 for row in read(s/'results/raw_export_members.json'):
  src=s/row['path'];out=safe_child(dest/stem,row['runtime_path']);out.parent.mkdir(parents=True,exist_ok=True)
  try:os.link(src,out)
  except OSError:shutil.copy2(src,out)
 restored=restore_archive(a.stage,dest/'restored_archive');shutil.move(restored['archive'],dest/arc['name'])
 pkg=SPECS[a.stage][1];write(dest/(pkg+'_download.json'),dict(name=arc['name'],bytes=arc['bytes'],sha256=arc['sha256']))
 directory=dest/ANALYSIS[a.stage];directory.mkdir()
 for p in (s/'references/original_scripts').glob('*.py.txt'):shutil.copy2(p,directory/p.name[:-4])
 if a.stage in ('06','07'):
  materialize(a.stage,dest/'release_restore');shutil.move(str(dest/'release_restore'/pkg),str(dest/pkg))
 if a.stage=='06':
  previous=dest/ANALYSIS['05'];previous.mkdir();shutil.copy2(stage('05')/'analysis/confirmed/analysis.json',previous/'analysis.json')
 if a.stage=='07':
  previous=stage('06');previous_arc=read(previous/'results/archives/archive.json');previous_stem=previous_arc['name'][:-7]
  for row in read(previous/'results/raw_export_members.json'):
   source=previous/row['path'];out=safe_child(dest/previous_stem,row['runtime_path']);out.parent.mkdir(parents=True,exist_ok=True)
   try:os.link(source,out)
   except OSError:shutil.copy2(source,out)
 runs=[];checks=[]
 if not a.prepare_only:
  for name in RUNS[a.stage]:
   print('RECOMPUTE',a.stage,name,flush=True);log=dest/(name+'.log')
   with log.open('w',encoding='utf8') as f:subprocess.run([sys.executable,str(directory/name)],cwd=directory,stdout=f,stderr=subprocess.STDOUT,check=True)
   runs.append(dict(script=name,source_sha256=sha(directory/name),log=log.name))
  for name in COMPARE[a.stage]:
   same(read(directory/name),read(s/'analysis/confirmed'/name));checks.append(dict(file=name,numerically_equal=True,absolute_tolerance=1e-9,relative_tolerance=1e-9))
 result=dict(stage=a.stage,layout=str(dest),scripts=runs,comparisons=checks,passed=not a.prepare_only,prepared=True,training_started=False)
 write(dest/'analysis_reproduction.json',result);print(result)
if __name__=='__main__':main()
