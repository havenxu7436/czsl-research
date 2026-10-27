"""Recompute all archived ABC predictions, utility coefficients, tables and AUC contributions."""
import argparse,json,math,os,shutil,subprocess,sys
from pathlib import Path
from register_trustcpf_abc import ROOT,read,write,sha,stage,restore_archive,PKG
from register_trustcpf_followup import safe_new,safe_child

def copy(source,target):
    source,target=Path(source),Path(target)
    if source.stat().st_size<256 and source.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise ValueError('Download the required Git LFS input: '+str(source))
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)

def same(a,b,path='root'):
    if path in ('analysis.json.source','audit.json.seconds'):return
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
        for k in a:same(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),(path,'length')
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+f'[{i}]')
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(path,a,b)
    else:assert a==b,(path,a,b)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dest',required=True);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    dest=safe_new(args.dest);dest.mkdir(parents=True);s=stage();arc=read(s/'results/archives/archive.json')
    target=dest/arc['original_parent']/arc['original_extraction']
    for row in read(s/'results/raw_export_members.json'):copy(s/row['path'],safe_child(target,row['runtime_path']))
    # The prior-stage comparison uses only its confirmed summary, not a second training run.
    previous=ROOT/'TRUST-CPF方案/09/results/raw_export/trustcpf_stage_v1_10ep/work/summary.csv'
    copy(previous,dest/'output/stage_results/server_results_20261025_183634/trustcpf_stage_v1_10ep/work/summary.csv')
    restored=restore_archive('10',dest/'restored_archive')
    for suffix in ('','.sha256'):shutil.move(restored['archive']+suffix,str(dest/arc['original_parent']/(arc['name']+suffix)))
    ad=dest/'analysis_TRUST_CPF_abc_v1_20261027';ad.mkdir()
    scripts=('analyze.py','audit.py','supplement.py')
    for name in scripts:copy(s/'references/original_analysis_scripts'/(name+'.txt'),ad/name)
    runs=[];checks=[]
    if not args.prepare_only:
        for name in scripts:
            print('RECOMPUTE',name,flush=True);log=dest/(name+'.log')
            with log.open('w',encoding='utf8') as f:subprocess.run([sys.executable,str(ad/name)],cwd=dest,stdout=f,stderr=subprocess.STDOUT,check=True)
            runs.append(dict(script=name,sha256=sha(ad/name),log=log.name))
        for name in ('analysis.json','audit.json','supplement.json'):
            same(read(ad/name),read(s/'analysis/confirmed'/name),name)
            checks.append(dict(file=name,numerically_equal=True,atol=1e-9,rtol=1e-9))
    result=dict(passed=not args.prepare_only,stage='10',prepared=True,scripts=runs,comparisons=checks,training_started=False)
    write(dest/'analysis_reproduction.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
