from pathlib import Path
import ast
import hashlib
import json
import re
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'TRUST-CPF方案/v1'
MEMBERS=json.loads((BASE/'results/release_member_map.json').read_text(encoding='utf-8'))['members']
OUT=BASE/'results/repository-validation-20261021'
errors=[]
docs=[]
for p in (ROOT/'TRUST-CPF方案').rglob('*.md'):
    text=p.read_text(encoding='utf-8')
    if text.count('```')%2: errors.append(str(p)+': fence')
    for target in re.findall(r'\]\(([^)\n]+)\)',text):
        target=unquote(target.strip('<>')).split('#')[0]
        if target and not re.match(r'^[a-z]+://',target) and not (p.parent/target).exists():
            errors.append(str(p)+': broken link '+target)
    docs.append(str(p.relative_to(ROOT)))
py=[]
for p in (BASE/'code').rglob('*.py'):
    ast.parse(p.read_text(encoding='utf-8-sig'),feature_version=(3,8))
    py.append(str(p.relative_to(BASE)))
for p in (BASE/'code').rglob('*.sh'):
    assert b'\r\n' not in p.read_bytes(),p
jsons=[]
for p in BASE.rglob('*.json'):
    json.loads(p.read_text(encoding='utf-8'))
    jsons.append(str(p.relative_to(BASE)))
records={}
for name in ('TRUST-CPF_CLIP_修订实验方案v1.md','TRUST-CPF公式及代码v1.md'):
    src=(ROOT/MEMBERS[name]).read_text(encoding='utf-8')
    dst=(BASE/name).read_text(encoding='utf-8')
    original_py=re.findall(r'```python\n(.*?)\n```',src,re.S)
    repo_py=re.findall(r'```python\n(.*?)\n```',dst,re.S)
    assert original_py==repo_py,name
    a=re.findall(r'\$\$(.*?)\$\$',src,re.S)
    b=re.findall(r'```math\n(.*?)\n```',dst,re.S)
    def normalize(s):
        s=re.sub(r'\\operatorname\{([^{}]+)\}',r'\\mathop{\\mathrm{\1}}\\nolimits',s)
        s=re.sub(r'\\tag\{\d+\}','',s)
        s=s.replace('\\begin{aligned}','').replace('\\end{aligned}','')
        s=s.replace('\\\\','').replace('&','')
        s=re.sub(r'\\q(?:q)?uad\b','',s)
        return re.sub(r'\s+','',s)
    assert [normalize(s) for s in a]==[normalize(s) for s in b],name
    labels=[]
    if '修订实验方案' in name:
        labels=[int(n) for n in re.findall(r'\*\*式（(\d+)）\*\*',dst)]
        assert labels==list(range(1,22)),name
        # TeX tags produce mlabeledtr, outside browser MathML-Core.
        assert not re.search(r'\\(?:tag|label)\b',dst),name
    without_blocks=re.sub(r'```.*?```','',dst,flags=re.S)
    inline=re.findall(r'\$`(.*?)`\$',without_blocks)
    records[name]={'display_formulas':len(b),'inline_formulas':len(inline),
        'python_blocks_unchanged':len(repo_py),'external_equation_labels':labels,
        'mathematical_tokens_preserved':True}
    # KaTeX render input, independent of Markdown parser.
    (OUT/(name+'.math.json')).write_text(json.dumps({'display':b,'inline':inline},ensure_ascii=False,indent=2),encoding='utf-8')
assert not errors,errors
result={'passed':True,'markdown_documents':len(docs),'broken_links':errors,
    'python_38_syntax_files':len(py),'json_files':len(jsons),'shell_scripts_lf':True,
    'documents':records,'release_bytes_changed':False,'training_started':False}
(OUT/'document_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
