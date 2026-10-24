"""Restore the original analysis layout and recompute one completed TRUST stage on CPU."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from register_trustcpf_strong_series import ROOT,SPECS,read,write,stage,safe_new,safe_child,restore_archive,sha

SCRIPTS={
 '02':['analyze_strong_v1.py','strong_extra_checks.py'],
 '03':['analyze_modelcond_v1.py','diagnose_modelcond_context.py','diagnose_modelcond_selection.py'],
 '04':['analyze_compact_v1.py','verify_compact_selection.py','diagnose_compact_scaling.py','diagnose_compact_policy.py']}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=list(SPECS),required=True);ap.add_argument('--dest',required=True);ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args();dest=safe_new(a.dest);dest.mkdir(parents=True)
    for n in list(SPECS)[:list(SPECS).index(a.stage)+1]:
        s=stage(n);key=SPECS[n][1];arc=read(s/'results/archives/archive.json');stem=arc['name'][:-7]
        for row in read(s/'results/raw_export_members.json'):
            p=s/row['path'];out=safe_child(dest/stem,row['runtime_path']);out.parent.mkdir(parents=True,exist_ok=True)
            # Analysis reads original inputs; hard links avoid another copy of the large measurements.
            # Confirmed JSON below is copied separately because original analysis rewrites its outputs.
            try:os.link(p,out)
            except OSError:shutil.copy2(p,out)
        confirmed=dest/f'analysis_TRUST_CPF_{key}_v1_20261023';confirmed.mkdir()
        for p in (s/'analysis/confirmed').glob('*.json'):shutil.copy2(p,confirmed/p.name)
        if n==a.stage and n!='02':
            result=restore_archive(n,dest/'archive_restore')
            shutil.move(result['archive'],dest/arc['name'])
            write(dest/f'trustcpf_{key}_v1_download.json',dict(name=arc['name'],bytes=arc['bytes'],sha256=arc['sha256']))
    scripts=dest/'tmp';scripts.mkdir()
    for p in (stage(a.stage)/'references/original_scripts').glob('*.py.txt'):shutil.copy2(p,scripts/p.name[:-4])
    runs=[]
    if not a.prepare_only:
        for name in SCRIPTS[a.stage]:
            print('RECOMPUTE',name,flush=True)
            log=dest/(name+'.log')
            with log.open('w',encoding='utf8') as f:
                subprocess.run([sys.executable,str(scripts/name)],cwd=dest,stdout=f,stderr=subprocess.STDOUT,check=True)
            runs.append(dict(script=name,script_sha256=sha(scripts/name),log=str(log)))
    result=dict(stage=a.stage,layout=str(dest),scripts=runs,training_started=False)
    write(dest/'analysis_reproduction.json',result);print(result)
if __name__=='__main__':main()
