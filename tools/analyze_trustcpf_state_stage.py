"""Recompute the archived State/Stage predictions, utility models and tables on CPU."""
import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from register_trustcpf_state_stage import ROOT,SPECS,read,write,sha,stage,restore_archive
from register_trustcpf_followup import safe_new,safe_child

RUNS = ['verify_predictions.py','verify_mechanism.py','summarize.py','check_tables.py']
COMPARE = ['verification.json','prediction_details.json','mechanism_verification.json',
           'mechanism_details.json','summary_data.json','table_verification.json']

def same(a,b,path='root'):
    if path=='summary_data.json.data_root':
        tail=lambda value:tuple(str(value).replace('\\','/').rstrip('/').split('/')[-4:])
        assert tail(a)==tail(b),(path,a,b)
        return
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
        for k in a:
            if k!='seconds':same(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),(path,'length')
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+f'[{i}]')
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(path,a,b)
    else:assert a==b,(path,a,b)

def copy(src,out):
    if src.stat().st_size < 256 and src.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise ValueError('Download the required Git LFS input first: '+str(src))
    out.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(src,out)
    except OSError:shutil.copy2(src,out)

def restore_raw(number,dest):
    s=stage(number);arc=read(s/'results/archives/archive.json')
    prefix=dest/arc['original_parent']/arc['original_extraction']
    for row in read(s/'results/raw_export_members.json'):
        copy(s/row['path'],safe_child(prefix,row['runtime_path']))

def compare_results(number,directory):
    additional=[]
    if number=='09':
        # The confirmed analysis includes this separate post-run checkpoint audit.
        import torch
        torch.set_num_threads(2)
        work=directory.parent/'output/stage_results/server_results_20261025_183634/trustcpf_stage_v1_10ep/work'
        matched=0
        for seed in (3351,3352,3353):
            for fold in range(1,6):
                teacher=work/f'seed{seed}/teachers/{fold}'
                last=torch.load(teacher/'last.pt',map_location='cpu',weights_only=False)
                best=torch.load(teacher/'best.pt',map_location='cpu',weights_only=False)
                assert set(best['model'])==set(last['best_model'])
                assert all(torch.equal(v,last['best_model'][k]) for k,v in best['model'].items())
                matched+=1
        report=read(directory/'verification.json')
        report['selected_checkpoint_weights_match_embedded_best']=matched
        write(directory/'verification.json',report)
        additional.append(dict(check='selected_checkpoint_weights_match_embedded_best',teachers=matched,passed=True))
    checks=[]
    for name in COMPARE:
        same(read(directory/name),read(stage(number)/'analysis/confirmed'/name),name)
        checks.append(dict(file=name,numerically_equal=True,absolute_tolerance=1e-9,relative_tolerance=1e-9,ignored_runtime_field='seconds',
            relocated_path_fields=['data_root'] if number=='08' and name=='summary_data.json' else []))
    return checks,additional

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=list(SPECS),required=True);ap.add_argument('--dest',required=True)
    ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    dest=safe_new(args.dest);dest.mkdir(parents=True);s=stage(args.stage)
    key='state' if args.stage=='08' else 'stage';directory=dest/f'analysis_TRUST_CPF_{key}_v1_20261025';directory.mkdir()
    restore_raw(args.stage,dest)
    arc=read(s/'results/archives/archive.json');restored=restore_archive(args.stage,dest/'restored_archive')
    arcdir=dest/arc['original_parent'];arcdir.mkdir(parents=True,exist_ok=True)
    for suffix in ('','.sha256'):shutil.move(restored['archive']+suffix,str(arcdir/(arc['name']+suffix)))
    if args.stage=='08':
        previous=ROOT/'TRUST-CPF方案/07';a=read(previous/'results/archives/archive.json');stem=a['name'][:-7]
        for seed in (3351,3352,3353):
            names=['runs/uniform/best.pt']+[f'teachers/{fold}/last.pt' for fold in range(1,6)]
            for name in names:
                rel=f'trustcpf_r5_v1_10ep/work/seed{seed}/'+name
                copy(previous/'results/raw_export'/rel,dest/stem/rel)
    else:restore_raw('08',dest)
    for p in (s/'references/original_scripts').glob('*.py.txt'):shutil.copy2(p,directory/p.name[:-4])
    runs=[];checks=[];additional=[]
    if not args.prepare_only:
        for name in RUNS:
            print('RECOMPUTE',args.stage,name,flush=True);log=dest/(name+'.log')
            with log.open('w',encoding='utf8') as f:
                subprocess.run([sys.executable,str(directory/name)],cwd=directory,stdout=f,stderr=subprocess.STDOUT,check=True)
            runs.append(dict(script=name,sha256=sha(directory/name),log=log.name))
        checks,additional=compare_results(args.stage,directory)
    result=dict(stage=args.stage,passed=not args.prepare_only,prepared=True,scripts=runs,comparisons=checks,additional_checks=additional,training_started=False)
    write(dest/'analysis_reproduction.json',result);print(__import__('json').dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
