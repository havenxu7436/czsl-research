"""Verify and restore completed TRUST-CPF stage 10 (A/B/C, ten epochs)."""
import argparse,ast,csv,hashlib,json,re,zipfile
from pathlib import Path
from urllib.parse import unquote
from register_trustcpf_followup import ROOT,read,write,sha,files,materialize,doctor,restore_archive,safe_new,safe_child

NUMBER='10';PKG='trustcpf_abc_v1_10ep'
def stage():return ROOT/'TRUST-CPF方案'/NUMBER

def check():
    s=stage();raw=s/'results/raw_export'/PKG;count=0
    for row in read(s/'file_manifest.json'):
        p=s/row['path'];assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path'];count+=1
    records=read(s/'results/raw_export_members.json')
    for row in records:assert sha(s/row['path'])==row['sha256'] and (s/row['path']).stat().st_size==row['bytes'],row['path']
    arc=read(s/'results/archives/archive.json');whole=hashlib.sha256();size=0
    for part in arc['parts']:
        p=s/part['path'];assert p.stat().st_size==part['bytes'] and sha(p)==part['sha256'],p
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):whole.update(b);size+=len(b)
    assert size==arc['bytes'] and whole.hexdigest()==arc['sha256']
    members=read(s/'package_members.json')['members']
    for name,digest in read(s/members['MANIFEST.json']).items():assert sha(s/members[name])==digest,name
    with zipfile.ZipFile(s/'results/deployment'/(PKG+'.zip')) as z:
        assert {n for n in z.namelist() if not n.endswith('/')}=={PKG+'/'+n for n in members}
        for name,destination in members.items():assert z.read(PKG+'/'+name)==(s/destination).read_bytes(),name
    server={row['runtime_path'][len(PKG)+1:]:s/row['path'] for row in records}
    for name,digest in read(raw/'MANIFEST.json').items():assert sha(server[name])==digest,name
    stages=0;outputs=0
    for marker in sorted((raw/'work/progress').glob('*/*.json')):
        m=read(marker)
        if 'outputs' not in m:continue
        base=raw/'work' if marker.parent.name=='global' else raw/'work'/marker.parent.name
        for name,digest in m['outputs'].items():assert sha(base/name)==digest,(marker,name);outputs+=1
        stages+=1
    assert stages==36
    status=read(raw/'work/status.json');assert status['complete'] and not status['missing']
    for key,value in dict(recognition_rows=936,q4_rows=3360,q5_rows=3504,directional_rows=1056,training_curve_rows=33).items():assert status[key]==value,(key,status[key])
    with (raw/'work/summary.csv').open(encoding='utf-8-sig',newline='') as f:summary=list(csv.DictReader(f))
    assert len(summary)==936 and len({tuple(r[k] for k in ('seed','bank','policy','phase','world')) for r in summary})==936
    inv=read(s/'configs/experiment_inventory.json');unique=set()
    for item in inv['recognition']:
        p=s/item['metrics'];assert sha(p)==item['metrics_sha256'] and sha(s/item['shared_checkpoint'])==item['checkpoint_sha256']
        metric=read(p);assert metric['checkpoint_sha256']==item['checkpoint_sha256'] and metric['epoch']==item['selected_epoch']
        for phase in ('val','test'):assert p.with_name(phase+'_predictions.npz').is_file()
        unique.add(str(p))
    assert len(inv['recognition'])==234 and len(unique)==234
    syntax=0;links=0
    for p in (s/'code').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'));syntax+=1
    for p in s.rglob('*.md'):
        text=p.read_text(encoding='utf8');assert text.count('```')%2==0,p
        assert not any(ord(c)<32 and c not in '\r\n\t' for c in text),p
        prose=re.sub(r'```[^\n]*\n[\s\S]*?```','',text)
        for target in re.findall(r'\]\((<[^>]+>|[^)\n]+)\)',prose):
            target=unquote(target.strip('<>')).split('#')[0]
            if not target or re.match(r'^[a-z]+://',target):continue
            assert not re.match(r'^[A-Za-z]:',target) and (p.parent/target).exists(),(p,target);links+=1
    return dict(passed=True,stage=NUMBER,files=count,server_members=len(records),release_members=len(members),
        archive_parts=len(arc['parts']),archive_bytes=size,archive_sha256=arc['sha256'],progress_stages=stages,
        progress_output_hashes=outputs,formal_policy_bank_instances=len(unique),formal_prediction_files=len(unique)*2,
        status=status,python_syntax_files=syntax,local_links=links,scientific_bytes_preserved=True,training_started=False)

def restore_results(destination):
    import os,shutil
    dest=safe_new(destination);dest.mkdir(parents=True);s=stage()
    for item in read(s/'results/raw_export_members.json'):
        source=s/item['path'];assert sha(source)==item['sha256'],source
        p=safe_child(dest,item['runtime_path']);p.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(source,p)
        except OSError:shutil.copy2(source,p)
        if p.suffix=='.sh':p.chmod(p.stat().st_mode|0o111)
    return dict(stage=NUMBER,path=str(dest/PKG),restored=True,training_started=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true')
    p.add_argument('--materialize');p.add_argument('--restore-archive');p.add_argument('--restore-results');a=p.parse_args()
    if a.check:result=check()
    elif a.materialize:result=materialize(NUMBER,a.materialize)
    elif a.restore_archive:result=restore_archive(NUMBER,a.restore_archive)
    elif a.restore_results:result=restore_results(a.restore_results)
    else:p.error('Choose --check, --materialize, --restore-archive or --restore-results')
    print(json.dumps(result,ensure_ascii=False))
