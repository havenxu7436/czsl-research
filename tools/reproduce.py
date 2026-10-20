"""Verify the curated repository and materialize historical layouts without training."""
import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

def read(path): return json.loads(path.read_text(encoding='utf8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()

def safe_child(parent,relative):
    relative=Path(relative)
    if relative.is_absolute() or relative.drive or '..' in relative.parts: raise ValueError('Unsafe relative path')
    p=parent/relative
    # Read-only canonical inputs may live behind directory junctions or symlinks.
    # Newly created runtime targets must remain physically below their destination.
    if parent.resolve()!=ROOT.resolve() and parent.resolve() not in p.resolve().parents: raise ValueError('Path outside target')
    return p

def verify():
    records=read(ROOT/'provenance/layout.json')['files']
    expected={r['current_path']:r['sha256'] for r in records if r['current_path']}
    failures=[]
    for rel,h in expected.items():
        p=safe_child(ROOT,rel)
        if not p.is_file() or sha(p)!=h: failures.append(rel)
    manifest=ROOT/'provenance/current_manifest.json'
    if manifest.exists():
        for r in read(manifest):
            p=safe_child(ROOT,r['path'])
            if not p.is_file() or sha(p)!=r['sha256']: failures.append(r['path'])
    syntax=[]; links=[]; formatting=[]
    docs=[ROOT/'README.md',ROOT/'environment/README.md',ROOT/'knowledge/README.md']
    for line in ('前期实验','0601方案','VG-CPOD方案'):
        docs.extend((ROOT/line).rglob('README.md'))
    imported=ROOT/'provenance/documents'
    if imported.is_dir():
        docs.extend(ROOT/read(p)['document'] for p in imported.glob('*.json'))
    for p in docs:
        text=p.read_text(encoding='utf8')
        if text.count('```')%2: formatting.append(str(p.relative_to(ROOT))+': unclosed fence')
        for raw in re.findall(r'\]\((<[^>]+>|[^)\n]+)\)',text):
            raw=unquote(raw.strip('<>')).split('#')[0]
            if not raw or re.match(r'^[a-z]+://',raw): continue
            if not (p.parent/raw).exists(): links.append({'file':str(p.relative_to(ROOT)),'target':raw})
        for raw in re.findall(r'^\[[^\]]+\]:\s*(<[^>]+>|\S+)',text,re.M):
            raw=unquote(raw.strip('<>')).split('#')[0]
            if not raw or re.match(r'^[a-z]+://',raw): continue
            if not (p.parent/raw).exists(): links.append({'file':str(p.relative_to(ROOT)),'target':raw})
    for rel in expected:
        if rel.endswith('.py'):
            try: ast.parse((ROOT/rel).read_text(encoding='utf-8-sig'))
            except (SyntaxError,UnicodeError) as exc: syntax.append({'file':rel,'error':str(exc)})
    route_check=None
    if (ROOT/'VG-CPOD方案/v1/inputs.json').is_file():
        from register_vgcpod_v1 import check as check_vgcpod
        route_check=check_vgcpod()
    b_check=None
    if (ROOT/'VG-CPOD方案/03/inputs.json').is_file():
        from register_vgcpod_round_b import check as check_vgcpod_b
        b_check=check_vgcpod_b()
    c_check=None
    if (ROOT/'VG-CPOD方案/04/inputs.json').is_file():
        from register_vgcpod_round_c import check as check_vgcpod_c
        c_check=check_vgcpod_c()
    d_check=None
    if (ROOT/'VG-CPOD方案/05/inputs.json').is_file():
        from register_vgcpod_round_d import check as check_vgcpod_d
        d_check=check_vgcpod_d()
    e_check=None
    if (ROOT/'VG-CPOD方案/06/inputs.json').is_file():
        from register_vgcpod_round_e import check as check_vgcpod_e
        e_check=check_vgcpod_e()
    f_check=None
    if (ROOT/'VG-CPOD方案/07/inputs.json').is_file():
        from register_vgcpod_round_f import check as check_vgcpod_f
        f_check=check_vgcpod_f()
    g_check=None
    if (ROOT/'VG-CPOD方案/08/inputs.json').is_file():
        from register_vgcpod_round_g import check as check_vgcpod_g
        g_check=check_vgcpod_g()
    result={'passed':not(failures or links or formatting or syntax),'original_canonical_files':len(expected),
      'changed_or_missing':failures,'broken_links':links,'formatting_errors':formatting,
      'python_syntax_errors':syntax,'vgcpod':route_check,'vgcpod_round_b':b_check,'vgcpod_round_c':c_check,'vgcpod_round_d':d_check,'vgcpod_round_e':e_check,'vgcpod_round_f':f_check,'vgcpod_round_g':g_check,'training_started':False}
    print(json.dumps(result,ensure_ascii=False))
    if not result['passed']: raise SystemExit(1)

def archive_stage(line,number):
    if line is None:
        if number == 'v1': raise ValueError('Use --line VG-CPOD方案 for --stage v1')
        if number == 'v2': raise ValueError('Use --line 0601方案 for --stage v2')
        return number
    matches=[s['id'] for s in read(ROOT/'provenance/stages.json')
      if s.get('line')==line and s.get('number')==number]
    if len(matches)!=1: raise ValueError('Unknown experiment line/number: '+line+'/'+number)
    return matches[0]

def materialize(stage,destination,line=None,number=None):
    if stage=='27':
        from register_vgcpod_round_g import materialize as materialize_vgcpod_g
        print(json.dumps(materialize_vgcpod_g(destination),ensure_ascii=False))
        return
    if stage=='26':
        from register_vgcpod_round_f import materialize as materialize_vgcpod_f
        print(json.dumps(materialize_vgcpod_f(destination),ensure_ascii=False))
        return
    if stage=='25':
        from register_vgcpod_round_e import materialize as materialize_vgcpod_e
        print(json.dumps(materialize_vgcpod_e(destination),ensure_ascii=False))
        return
    if stage=='24':
        from register_vgcpod_round_d import materialize as materialize_vgcpod_d
        print(json.dumps(materialize_vgcpod_d(destination),ensure_ascii=False))
        return
    if stage=='23':
        from register_vgcpod_round_c import materialize as materialize_vgcpod_c
        print(json.dumps(materialize_vgcpod_c(destination),ensure_ascii=False))
        return
    if stage=='22':
        from register_vgcpod_round_b import materialize as materialize_vgcpod_b
        print(json.dumps(materialize_vgcpod_b(destination),ensure_ascii=False))
        return
    if stage=='21':
        from register_vgcpod_round_a import materialize as materialize_vgcpod_a
        print(json.dumps(materialize_vgcpod_a(destination),ensure_ascii=False))
        return
    if stage=='20':
        from register_vgcpod_v1 import materialize as materialize_vgcpod
        print(json.dumps(materialize_vgcpod(destination),ensure_ascii=False))
        return
    layout=read(ROOT/'provenance/layout.json'); destination=Path(destination).expanduser()
    if not destination.is_absolute(): raise ValueError('Choose an absolute destination')
    destination=destination.resolve()
    if destination==ROOT or ROOT in destination.parents:
        raise ValueError('Choose an absolute separate directory outside repository')
    if destination.exists(): raise FileExistsError('Preserve existing directory: '+str(destination))
    subprocess.check_call(['git','rev-parse','--verify',layout['source_tag']+'^{commit}'],cwd=ROOT,stdout=subprocess.DEVNULL)
    selected=[r for r in layout['files'] if int(r['stage'])<=int(stage)]
    expected={r['current_path']:r['sha256'] for r in selected if r['current_path']}
    for relative,digest in expected.items():
        source=safe_child(ROOT,relative)
        if not source.is_file() or sha(source)!=digest:
            raise ValueError('Canonical input changed or missing: '+relative)
    documents=[r for r in selected if r['kind']=='historical_document' and
      not r['original_path'].startswith(('docs/','output/')) and len(Path(r['original_path']).parts)>1]
    # Restore documentation blobs in one Git process, not one process per file.
    queries=[layout['source_tag']+':'+r['original_path'] for r in documents]
    if any('\n' in q or '\r' in q for q in queries): raise ValueError('Unsafe Git blob query')
    blobs={}
    if queries:
        output=subprocess.check_output(['git','cat-file','--batch'],cwd=ROOT,
          input=('\n'.join(queries)+'\n').encode('utf8'))
        offset=0
        for r in documents:
            end=output.index(b'\n',offset); header=output[offset:end].split()
            if len(header)!=3 or header[1]!=b'blob': raise ValueError('Missing historical document')
            size=int(header[2]); offset=end+1; payload=output[offset:offset+size]; offset+=size+1
            if hashlib.sha256(payload).hexdigest()!=r['sha256']: raise ValueError('Historical document changed')
            blobs[r['original_path']]=payload
    destination.mkdir(parents=True)
    receipts=[]
    for r in selected:
        old=r.get('runtime_path',r['original_path'])
        if r['current_path']:
            source=safe_child(ROOT,r['current_path'])
            payload=None
        elif r['kind']=='runtime_support':
            payload=base64.b64decode(r['runtime_support_base64']); source=None
            if hashlib.sha256(payload).hexdigest()!=r['sha256']: raise ValueError('Runtime support changed')
        elif r['kind']=='historical_document':
            # Runtime checksum guards depend on original package README/PLAN files.
            # Restore exact bytes from the immutable Git tag, never from prose rewrites.
            if old.startswith(('docs/','output/')) or len(Path(old).parts)==1: continue
            payload=blobs[old]
            source=None
        else: continue
        paths=[old]
        if old.startswith(layout['project_prefix']): paths.append('project/'+old[len(layout['project_prefix']):])
        for relative in paths:
            target=safe_child(destination,relative); target.parent.mkdir(parents=True,exist_ok=True)
            if payload is None: shutil.copy2(source,target)
            else: target.write_bytes(payload)
            if sha(target)!=r['sha256']: raise ValueError('Runtime copy corrupted')
            if target.suffix=='.sh': target.chmod(target.stat().st_mode|0o111)
        receipts.append({'original_path':old,'sha256':r['sha256']})
    (destination/'materialization.json').write_text(json.dumps({'source_tag':layout['source_tag'],
      'stage':number if line else stage,'line':line,'archive_stage':stage,
      'files':receipts,'original_bytes_modified':False,'training_started':False},ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'stage':number if line else stage,'line':line,'archive_stage':stage,
      'runtime':str(destination),'files_verified':len(receipts),'project':str(destination/'project'),
      'note':'External inputs and frozen absolute paths still require setup; see environment/README.md'},ensure_ascii=False))

def doctor(stage,project,line=None,number=None):
    if stage=='20':
        from register_vgcpod_v1 import doctor as doctor_vgcpod
        result=doctor_vgcpod(project)
        print(json.dumps(result,ensure_ascii=False))
        if not result['ready_for_package_preflight']: raise SystemExit(2)
        return
    project=Path(project).resolve()
    required=[project/'src/KCC0601_v1/train.py',project/'data/glove/glove.6B.300d.txt',
      project/'data/ut-zap50k',project/'knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k/manifest.json']
    issues=[str(p) for p in required if not p.exists()]
    absolute=[]
    for path in project.glob('knowledge/**/manifest.json'):
        record=read(path)
        for old,h in record.get('input_sha256',{}).items():
            if isinstance(h,str) and re.fullmatch('[a-fA-F0-9]{64}',h):
                p=Path(old)
                if not p.is_file(): absolute.append({'path':old,'reason':'frozen path missing','manifest':str(path)})
                elif sha(p)!=h: absolute.append({'path':old,'reason':'input SHA256 mismatch','manifest':str(path)})
    print(json.dumps({'stage':number if line else stage,'line':line,'archive_stage':stage,
      'project':str(project),'missing':issues,'frozen_path_issues':absolute,
      'ready_for_package_preflight':not(issues or absolute),
      'not_checked':['CUDA/GPU permissions','dataset split hashes beyond original package checks',
       'previous-stage training checkpoints','full training/reproduced metrics'],'training_started':False},ensure_ascii=False))
    if issues or absolute: raise SystemExit(2)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    commands.add_parser('verify')
    m=commands.add_parser('materialize'); m.add_argument('--stage',choices=['{:02d}'.format(x) for x in range(1,21)]+['v1','v2'],required=True); m.add_argument('--dest',required=True)
    m.add_argument('--line',choices=['前期实验','0601方案','VG-CPOD方案'])
    d=commands.add_parser('doctor'); d.add_argument('--stage',required=True); d.add_argument('--project',required=True)
    d.add_argument('--line',choices=['前期实验','0601方案','VG-CPOD方案'])
    args=parser.parse_args()
    if args.action=='verify': verify()
    elif args.action=='materialize': materialize(archive_stage(args.line,args.stage),args.dest,args.line,args.stage)
    else: doctor(archive_stage(args.line,args.stage),args.project,args.line,args.stage)
