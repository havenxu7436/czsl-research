"""Verify, restore and register the completed State and Stage batches (08–09)."""
import argparse
import ast
import csv
import hashlib
import re
import subprocess
import zipfile
from urllib.parse import unquote
from register_trustcpf_followup import (
    ROOT, read, write, sha, files, materialize, doctor, restore_archive,
)

SPECS = {
    '08': dict(id='40', package='trustcpf_state_v1_10ep', title='State 第一批：状态与联合效用拟合',
               stages=17, recognition=1512, q5=600, predictions=756, curves=33, uniform_training=3, teacher_training=0),
    '09': dict(id='41', package='trustcpf_stage_v1_10ep', title='Stage 第二批：教师阶段匹配与响应平均',
               stages=26, recognition=972, q5=744, predictions=486, curves=165, uniform_training=0, teacher_training=15),
}

def stage(number): return ROOT/'TRUST-CPF方案'/number

def check(number=None):
    checks = []
    for n in ([number] if number else SPECS):
        spec = SPECS[n]; s = stage(n); pkg = spec['package']; raw = s/'results/raw_export'/pkg
        for row in read(s/'file_manifest.json'):
            p = s/row['path']
            assert p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
        members = read(s/'results/raw_export_members.json')
        for row in members:
            assert sha(s/row['path']) == row['sha256'], row['path']
        a = read(s/'results/archives/archive.json'); h = hashlib.sha256(); size = 0
        for part in a['parts']:
            p = s/part['path']; assert p.stat().st_size == part['bytes'] and sha(p) == part['sha256']
            with p.open('rb') as f:
                for block in iter(lambda: f.read(8*1024**2), b''): h.update(block); size += len(block)
        assert size == a['bytes'] and h.hexdigest() == a['sha256']
        mapping = read(s/'package_members.json')['members']
        for name, digest in read(s/mapping['MANIFEST.json']).items(): assert sha(s/mapping[name]) == digest, name
        with zipfile.ZipFile(s/'results/deployment'/(pkg+'.zip')) as z:
            assert {x for x in z.namelist() if not x.endswith('/')} == {pkg+'/'+x for x in mapping}
            for name, target in mapping.items(): assert z.read(pkg+'/'+name) == (s/target).read_bytes(), name
        # Archived Markdown has a .txt suffix; restore original member names through the mapping.
        server_paths = {x['runtime_path'][len(pkg)+1:]:s/x['path'] for x in members}
        server_manifest = read(raw/'MANIFEST.json')
        for name, digest in server_manifest.items(): assert sha(server_paths[name]) == digest, name
        outputs = 0; progress_count = 0
        for marker in sorted((raw/'work/progress').glob('*/*.json')):
            m = read(marker)
            if 'outputs' not in m: continue
            base = raw/'work' if marker.parent.name == 'global' else raw/'work'/marker.parent.name
            for name, digest in m['outputs'].items(): assert sha(base/name) == digest, (marker, name); outputs += 1
            progress_count += 1
        assert progress_count == spec['stages']
        status = read(raw/'work/status.json')
        assert status['complete'] and not status['missing'] and status['completed_stages'] == spec['stages']
        assert status['recognition_rows'] == spec['recognition'] and status['q5_rows'] == spec['q5']
        inventory = read(s/'configs/experiment_inventory.json')
        for item in inventory['recognition']:
            assert sha(s/item['shared_checkpoint']) == item['checkpoint_sha256']
            for path, digest in item['metrics_sha256'].items():
                assert sha(s/path) == digest
                metric = read(s/path); assert metric['checkpoint_sha256'] == item['checkpoint_sha256']
                for phase in ('val','test'): assert (s/path).with_name(phase+'_predictions.npz').is_file()
        assert len(list((raw/'work').glob('seed*/recognition/*/*/bank*/*_predictions.npz'))) == spec['predictions']
        for p in s.rglob('*.md'):
            t = p.read_text(encoding='utf8'); assert t.count('```') % 2 == 0, p
            assert not any(ord(c)<32 and c not in '\r\n\t' for c in t), p
            prose = re.sub(r'```[^\n]*\n[\s\S]*?```', '', t)
            for link in re.findall(r'\]\((<[^>]+>|[^)\n]+)\)', prose):
                link = unquote(link.strip('<>')).split('#')[0]
                if not link or re.match(r'^[a-z]+://', link): continue
                assert not re.match(r'^[A-Za-z]:',link) and (p.parent/link).exists(), (p,link)
        for p in (s/'code').rglob('*.py'): ast.parse(p.read_text(encoding='utf-8-sig'))
        checks.append(dict(stage=n, passed=True, server_members=len(members), release_members=len(mapping),
            checked_progress_outputs=outputs, status=status, new_uniform_training_runs=spec['uniform_training'],
            new_teacher_training_runs=spec['teacher_training'], recognition_policy_instances=len(inventory['recognition'])))
    return dict(passed=True, stages=checks, scientific_bytes_preserved=True, training_started=False)

def finalize():
    ids = {x['id'] for x in SPECS.values()}
    stages = [x for x in read(ROOT/'provenance/stages.json') if x['id'] not in ids]
    layout = read(ROOT/'provenance/layout.json')
    layout['files'] = [x for x in layout['files'] if x.get('kind') != 'trustcpf_state_stage_server_member']
    index = ROOT/'provenance/metrics_index.csv'
    with index.open(encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f); fields = r.fieldnames; rows = [x for x in r if not x['route'].startswith('TRUST-state-stage-')]
    receipt = []
    for n, spec in SPECS.items():
        s = stage(n); pkg = spec['package']; raw = s/'results/raw_export'/pkg
        status = read(raw/'work/status.json'); inv = read(s/'configs/experiment_inventory.json')
        inputs = dict(line='TRUST-CPF方案',stage=n,archive_stage=spec['id'],title=spec['title'],primary_package=pkg,
            seeds=[3351,3352,3353],new_uniform_training_runs=spec['uniform_training'],new_teacher_training_runs=spec['teacher_training'],
            recognition_policy_instances=len(inv['recognition']),formal_training_results_included=True,
            package_member_map='package_members.json',experiment_inventory='configs/experiment_inventory.json',status=status)
        for key, directory in [('code_files','code'),('configs','configs'),('data_files','results'),('knowledge',f'configs/{pkg}/assets')]:
            inputs[key] = [p.relative_to(s).as_posix() for p in files(s/directory)]
        inputs['documents'] = [p.relative_to(s).as_posix() for p in s.rglob('*.md')]
        write(s/'inputs.json', inputs)
        commands = []
        for name in ('README.md','SERVER_COMMANDS.md','analysis/REPRODUCE.md'):
            for language, text in re.findall(r'```(bash|powershell)\n(.*?)```',(s/name).read_text(encoding='utf8'),re.S):
                commands.append(dict(source=name,language=language,text=text))
        write(s/'runbook.json', dict(line='TRUST-CPF方案',stage=n,primary_package=pkg,commands=commands))
        stages.append(dict(id=spec['id'],line='TRUST-CPF方案',number=n,version='v1',title=n+' · '+spec['title'],
            path='TRUST-CPF方案/'+n,primary_package=pkg,active_package=pkg,status='Completed; source, raw results and analyses archived',
            recognition_rows=spec['recognition'],new_uniform_training_runs=spec['uniform_training'],new_teacher_training_runs=spec['teacher_training']))
        a = read(s/'results/archives/archive.json')
        for f in read(s/'results/raw_export_members.json'):
            layout['files'].append(dict(original_path=a['name']+'::'+f['runtime_path'],runtime_path=f['runtime_path'],
                current_path='TRUST-CPF方案/'+n+'/'+f['path'],sha256=f['sha256'],bytes=f['bytes'],stage=spec['id'],kind='trustcpf_state_stage_server_member'))
        with (raw/'work/summary.csv').open(encoding='utf-8-sig',newline='') as f:
            for item in csv.DictReader(f):
                state = item.get('state','uniform_best'); row = {k:'' for k in fields}
                metric = f'TRUST-CPF方案/{n}/results/raw_export/{pkg}/work/seed{item["seed"]}/recognition/{state}/{item["policy"]}/bank{item["bank"]}/metrics.json'
                row.update(stage=spec['id'],route='TRUST-state-stage-'+n,dataset='ut-zap50k',seed=item['seed'],variant=item['policy'],
                    phase=item['phase'],setting=item['world'],selected_epoch=item['epoch'],auc_pct=item['AUC_native'],hm_pct=item['HM'],source=metric,
                    note='Frozen shared-checkpoint readout; state='+state+'; graph_bank='+item['bank'])
                rows.append(row)
        receipt.append(dict(stage=n,archive=a,status=status,new_uniform_training_runs=spec['uniform_training'],new_teacher_training_runs=spec['teacher_training']))
        write(s/'file_manifest.json',[dict(path=p.relative_to(s).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files(s) if p.name!='file_manifest.json'])
    write(ROOT/'provenance/stages.json',stages); write(ROOT/'provenance/layout.json',layout)
    with index.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = read(ROOT/'provenance/structure_summary.json')
    summary['line_numbering']['TRUST-CPF方案']='01 / v1、02–09'; summary['documentation_updated']='2026-10-25'
    write(ROOT/'provenance/structure_summary.json',summary)
    write(ROOT/'provenance/updates/20261025_trustcpf_state_stage.json',dict(date='2026-10-25',stages=receipt,
        scientific_bytes_preserved=True,training_started=False,reading_pages='08 and 09; original Markdown retained as .md.txt'))
    for p in (ROOT/'provenance/documents').glob('trustcpf_0[89]*.json'):
        doc=read(p); doc['published_sha256']=sha(ROOT/doc['document']); write(p,doc)
    current={x['path']:x for x in read(ROOT/'provenance/current_manifest.json')}
    changed=set(subprocess.check_output(['git','diff','--name-only','-z','HEAD'],cwd=ROOT).decode('utf8').split('\0'))
    changed.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode('utf8').split('\0'))
    changed.update(p.relative_to(ROOT).as_posix() for n in SPECS for p in files(stage(n)))
    for rel in changed:
        p=ROOT/rel
        if rel and rel!='provenance/current_manifest.json' and p.is_file():current[rel]=dict(path=rel,sha256=sha(p),bytes=p.stat().st_size)
    write(ROOT/'provenance/current_manifest.json',[current[k] for k in sorted(current)])

if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=list(SPECS)); ap.add_argument('--check',action='store_true'); ap.add_argument('--finalize',action='store_true')
    ap.add_argument('--materialize'); ap.add_argument('--restore-archive'); args=ap.parse_args()
    if args.finalize: finalize(); result=check(args.stage)
    elif args.materialize or args.restore_archive:
        if not args.stage: ap.error('--stage is required')
        result=materialize(args.stage,args.materialize) if args.materialize else restore_archive(args.stage,args.restore_archive)
    elif args.check: result=check(args.stage)
    else: ap.error('Choose --check, --finalize, --materialize or --restore-archive')
    print(__import__('json').dumps(result,ensure_ascii=False))
