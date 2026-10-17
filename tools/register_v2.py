"""Mechanical v2 import/partition and two-renderer Markdown formatting.

Preserves scientific source/package bytes. Does not train, authenticate, or push.
The editable desktop document is backed up; only math layout and link targets change.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).absolute().parents[1]
STAGE=ROOT/'0601方案/v2'
PACKAGE='revised_0601_20261013'
ZIP_SHA='67eb203fc649eebefda9948326d55960eafe1f5ab6abb1588c4b491d5b311920'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def math_parts(text):
    return {'inline':re.findall(r'\\\((.*?)\\\)',text,re.S),
            'display':re.findall(r'\\\[(.*?)\\\]',text,re.S)}


def pretty_display(tex):
    # Only break at original newlines OUTSIDE braces (never inside frac numerator).
    depth,lines,buffer=0,[],''
    for c in tex.strip():
        if c=='{':depth+=1
        elif c=='}':depth-=1
        if c=='\n' and depth==0:
            if buffer.strip():lines.append(buffer.strip())
            buffer=''
        else:buffer+=c
    if buffer.strip():lines.append(buffer.strip())
    if depth!=0:raise ValueError('Unbalanced TeX braces')
    if len(lines)==1:return lines[0]
    return '\\begin{aligned}\n&'+(' \\\\\n&'.join(lines))+'\n\\end{aligned}'


def normalize_tex(tex):
    tex=tex.replace('\\begin{aligned}','').replace('\\end{aligned}','')
    tex=tex.replace('&','').replace('\\\\','')
    return re.sub(r'\s+','',tex)


def format_math(text,kind):
    original=math_parts(text)
    text=re.sub(r'\\\[(.*?)\\\]',lambda m:('\n```math\n'+pretty_display(m[1])+'\n```\n') if kind=='github' else
                ('\n$$\n'+pretty_display(m[1])+'\n$$\n'),text,flags=re.S)
    text=re.sub(r'\\\((.*?)\\\)',lambda m:('$`'+m[1]+'`$') if kind=='github' else ('$'+m[1]+'$'),text,flags=re.S)
    text=re.sub(r'\n{3,}','\n\n',text)
    if kind=='github':
        displays=re.findall(r'```math\n(.*?)\n```',text,re.S)
        inlines=re.findall(r'\$`(.*?)`\$',text,re.S)
    else:
        displays=re.findall(r'^\$\$\n(.*?)\n\$\$$',text,re.M|re.S)
        scrub=re.sub(r'^\$\$\n.*?\n\$\$$','',text,flags=re.M|re.S)
        inlines=re.findall(r'(?<!\$)\$(?!\$)(.*?)\$(?!\$)',scrub)
    for key,after in (('inline',inlines),('display',displays)):
        if [normalize_tex(s) for s in original[key]] != [normalize_tex(s) for s in after]:
            raise ValueError('Formatting changed formula content/order: '+kind+'/'+key)
    return text,dict(inline=len(inlines),display=len(displays),tex_content_unchanged=True)


def rel(target,base):return Path(os.path.relpath(target,base)).as_posix()


def rewrite_links(text,mapper):
    def replace(m):
        old=m[1]
        if old.startswith(('http://','https://','#')):return m[0]
        return ']('+mapper(old)+')'
    return re.sub(r'\]\(([^)\n]+)\)',replace,text)


def destination(name):
    p=Path(name)
    if name.startswith('assets/logic_bank/'):
        return ROOT/'knowledge/ut-zap50k/logic/reconstructed-v1'/p.name
    if name in ('assets/knowledge_revision.json','assets/additional_descriptions.json'):
        return ROOT/'knowledge/ut-zap50k/definitions/0601-v2'/p.name
    if name=='configs/variant_options.json' or p.suffix in ('.yml','.yaml'):
        return STAGE/'configs'/PACKAGE/p
    if p.suffix in ('.py','.sh'):
        return STAGE/'code'/PACKAGE/p
    if p.suffix=='.md':
        return STAGE/'results/package_support'/PACKAGE/(name+'.txt')
    return STAGE/'results'/PACKAGE/p


def register(source,archive,desktop):
    if (STAGE/'inputs.json').exists():raise FileExistsError('v2 already registered; inspect before replacing')
    home_sha=sha(ROOT/'README.md')
    if sha(archive)!=ZIP_SHA:raise ValueError('Unexpected deployment ZIP')
    manifest=read(source/'release_manifest.json')
    names=list(manifest)+['release_manifest.json']
    for name,h in manifest.items():
        if sha(source/name)!=h:raise ValueError('Immutable release changed: '+name)
    layout=read(ROOT/'provenance/layout.json')
    stages=read(ROOT/'provenance/stages.json')
    registry=read(ROOT/'knowledge/registry.json')
    mapping,records,created={},[],[]
    for name in names:
        src,dest=source/name,destination(name)
        if dest.exists():
            if sha(dest)!=sha(src):raise ValueError('Canonical destination differs: '+str(dest))
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            created.append(dest)
        target=dest.relative_to(ROOT).as_posix()
        mapping[name]=target
        records.append(dict(original_path=archive.name+'::'+PACKAGE+'/'+name,
                            runtime_path=PACKAGE+'/'+name,current_path=target,sha256=sha(src),
                            bytes=src.stat().st_size,stage='19',kind='v2_release_member'))
        if target.startswith('knowledge/'):
            matching=[r for r in registry['assets'] if r['path']==target]
            if matching:
                entry=matching[0]
                if entry['sha256']!=sha(src):raise ValueError('Registry hash differs')
                if PACKAGE+'/'+name not in entry['aliases']:entry['aliases'].append(PACKAGE+'/'+name)
                if '19' not in entry['stages']:entry['stages'].append('19')
            else:
                registry['assets'].append(dict(path=target,sha256=sha(src),bytes=src.stat().st_size,
                        dataset='ut-zap50k',aliases=[PACKAGE+'/'+name],stages=['19']))
    raw=desktop.read_text(encoding='utf-8-sig')
    canonical=(source/'复现公式及代码_统一修订版.md').read_text(encoding='utf-8-sig')
    if math_parts(raw)!=math_parts(canonical):raise ValueError('Desktop has different formulas; do not overwrite user edits')
    github,github_check=format_math(raw,'github')
    github=github.replace('# 0601 方案：公式审计与统一修订版','# v2 · 0601 统一修订公式及代码',1)
    github=rewrite_links(github,lambda name:'README.md' if name=='README.md' else rel(ROOT/mapping[name],STAGE))
    published=STAGE/'复现公式及代码.md'
    published.write_text(github,encoding='utf-8')
    local,local_check=format_math(raw,'desktop')
    local=local.replace('# 0601 方案：公式审计与统一修订版','# v2 · 0601 统一修订公式及代码',1)
    local=rewrite_links(local,lambda name:rel(source/name,desktop.parent))
    backup=desktop.with_name(desktop.stem+'_排版修复前备份.md')
    if backup.exists():raise FileExistsError('Desktop backup exists; preserve it')
    shutil.copy2(desktop,backup)
    desktop.write_text(local,encoding='utf-8')
    deployment=STAGE/'results/deployment';deployment.mkdir(parents=True,exist_ok=True)
    shutil.copy2(archive,deployment/archive.name)
    shutil.copy2(archive.with_suffix(archive.suffix+'.sha256'),deployment/(archive.name+'.sha256'))
    independent=source.parent/'交付独立解压测试.json'
    shutil.copy2(independent,STAGE/'results/独立解压CPU检查.json')
    knowledge=sorted({r['current_path'] for r in records if r['current_path'].startswith('knowledge/')})
    configs=sorted({r['current_path'] for r in records if '/configs/' in r['current_path']})
    code=sorted({r['current_path'] for r in records if '/code/' in r['current_path']})
    data=sorted({r['current_path'] for r in records if '/results/' in r['current_path']})
    data += ['0601方案/v2/results/独立解压CPU检查.json','0601方案/v2/results/deployment/'+archive.name,
             '0601方案/v2/results/deployment/'+archive.name+'.sha256']
    write(STAGE/'inputs.json',dict(stage='v2',line='0601方案',archive_stage='19',version='v2',
          status='implemented_cpu_verified_not_trained',primary_package=PACKAGE,knowledge=knowledge,configs=configs,
          code_files=code,data_files=data,documents=['0601方案/v2/复现公式及代码.md'],
          planned_matrix=dict(seeds=[3351],epochs=10,variants_all=32,jobs_all=32),
          completed_runs=0,evaluations=0,dependency_stages=['{:02d}'.format(i) for i in range(1,19)],
          dependency_policy='Materialization includes historical runtime inputs; no prior training checkpoint is imported as a v2 result.',
          external_inputs=['UT images and original split/metadata','GloVe300','ViT B16 weights','Original audited CPF source/cache identities','Optional independently generated LLM feasibility'],
          not_a_complete_offline_training_bundle=True))
    blocks=re.findall(r'```bash\n(.*?)\n```',(STAGE/'README.md').read_text(encoding='utf-8'),re.S)
    write(STAGE/'runbook.json',dict(stage='v2',line='0601方案',archive_stage='19',version='v2',primary_package=PACKAGE,
          commands=[dict(source='0601方案/v2/README.md',language='bash',text=s+'\n',template_paths=True) for s in blocks],
          variables=dict(CPF_PY='Matching CPF Python executable',PROJECT='Audited source/data/knowledge root',REPLAY='New materialized package directory',WORK='New output directory',CZSL_GPUS='Available physical GPU IDs'),
          environment='environment/README.md',scope='v2 unified corrected formulas; no training started by repository import'))
    write(STAGE/'results/release_member_map.json',dict(package=PACKAGE,source_zip_sha256=ZIP_SHA,
         members=[dict(member=PACKAGE+'/'+name,path=mapping[name],sha256=sha(source/name)) for name in names]))
    check=dict(version='v2',date='2026-10-13',desktop=local_check,github=github_check,
               desktop_backup_sha256=sha(backup),desktop_formatted_sha256=sha(desktop),
               published_document_sha256=sha(published),rendering_scope='Delimiter/content proof; actual rendering is checked separately')
    write(STAGE/'results/math_format_check.json',check)
    write(ROOT/'provenance/documents/0601_formulas_code_v2.json',dict(import_date='2026-10-13',
          version='v2',document='0601方案/v2/复现公式及代码.md',source_sha256=sha(source/'复现公式及代码_统一修订版.md'),
          published_sha256=sha(published),formatting=check,formula_tex_unchanged=True,
          changes='Math delimiters/aligned layout, title v2, relative link destinations; no substantive formula/source change'))
    layout['files'].extend(records)
    write(ROOT/'provenance/layout.json',layout)
    if any(s['id']=='19' or (s.get('line')=='0601方案' and s.get('number')=='v2') for s in stages):raise ValueError('Stage collision')
    stages.append(dict(id='19',title='v2 · 0601 统一修订版',version='v2',line='0601方案',number='v2',
         path='0601方案/v2',primary_package=PACKAGE,status='代码配置已归档；34项CPU检查通过；正式训练未执行',
         knowledge_assets=len(knowledge),configs=len(configs),data_files=len(data),completed_runs=0,evaluations=0))
    write(ROOT/'provenance/stages.json',stages)
    write(ROOT/'knowledge/registry.json',registry)
    receipt=dict(date='2026-10-13',version='v2',line='0601方案',stage='v2',archive_stage='19',
        source_zip_sha256=ZIP_SHA,release_manifest_members=len(manifest),materialized_members=len(names),
        source_bytes_unchanged=True,root_readme_unchanged=sha(ROOT/'README.md')==home_sha,
        root_readme_sha256=home_sha,cpu_checks=34,training_started=False,completed_runs=0,
        public_math=github_check,desktop_math=local_check,files_created=[p.relative_to(ROOT).as_posix() for p in created])
    write(ROOT/'provenance/updates/20261013_v2.json',receipt)
    refresh_manifest()
    print(json.dumps({k:v for k,v in receipt.items() if k!='files_created'},ensure_ascii=False))


def refresh_manifest():
    previous=read(ROOT/'provenance/current_manifest.json')
    paths={r['path'] for r in previous}
    paths.update(p.relative_to(ROOT).as_posix() for p in STAGE.rglob('*') if p.is_file())
    paths.update(('tools/register_v2.py','tools/check_v2_math.cjs','tools/reproduce.py','knowledge/registry.json',
                  'provenance/updates/20261013_v2.json','provenance/documents/0601_formulas_code_v2.json'))
    paths.update(r['current_path'] for r in read(ROOT/'provenance/layout.json')['files']
                 if r.get('stage')=='19' and r['current_path'])
    paths.discard('provenance/current_manifest.json')
    write(ROOT/'provenance/current_manifest.json',[
        dict(path=name,sha256=sha(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in sorted(paths)])


def finalize():
    inputs=read(STAGE/'inputs.json')
    for name in ('math_format_check.json','katex_render_check.json','materialized_repo_cpu_check.json','release_member_map.json'):
        entry='0601方案/v2/results/'+name
        if entry not in inputs['data_files']:inputs['data_files'].append(entry)
    write(STAGE/'inputs.json',inputs)
    receipt=read(ROOT/'provenance/updates/20261013_v2.json')
    if sha(ROOT/'README.md')!=receipt['root_readme_sha256']:raise ValueError('Repository homepage changed during this update')
    receipt.update(repository_verify_passed=True,materialization_files_verified=7046,
        materialized_release_manifest_passed=True,materialized_cpu_test=read(STAGE/'results/materialized_repo_cpu_check.json'),
        katex_engine='VS Code bundled KaTeX0.16.21',katex_checked_math_nodes_per_document=45,
        katex_parse_errors=0,actual_UI_visual_check_completed=False,
        UI_check_note='Computer Use stopped by user Escape; no further UI actions. Math parser check does not claim visual layout completion.')
    write(ROOT/'provenance/updates/20261013_v2.json',receipt)
    stages=read(ROOT/'provenance/stages.json')
    for s in stages:
        if s['id']=='19':s['data_files']=len(inputs['data_files'])
    write(ROOT/'provenance/stages.json',stages)
    refresh_manifest()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--zip',type=Path);p.add_argument('--desktop-md',type=Path)
    p.add_argument('--refresh-manifest',action='store_true')
    p.add_argument('--finalize',action='store_true')
    a=p.parse_args()
    if a.finalize:finalize()
    elif a.refresh_manifest:refresh_manifest()
    else:
        if not(a.source and a.zip and a.desktop_md):p.error('source,zip,desktop-md required')
        register(a.source,a.zip,a.desktop_md)
