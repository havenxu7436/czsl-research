from pathlib import Path
import json,re,html
import markdown
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
DOC=BASE/'output/TRUST-CPF_公式修订与后续实验方案_v6_20261025.md'
s=DOC.read_text(encoding='utf8')
assert not any(ord(c)<32 and c not in '\n' for c in s)
assert '\ufffd' not in s
blocks=re.findall(r'\$\$(.*?)\$\$',s,re.S)
assert len(blocks)==36,len(blocks)
for b in blocks:
    depth=0
    for c in b:
        if c=='{':depth+=1
        elif c=='}':depth-=1
        assert depth>=0
    assert depth==0,b
tags=re.findall(r'\\tag\{([^}]+)\}',s)
assert len(tags)==len(set(tags))==36
links=re.findall(r'\]\(<?(C:/[^\n>]*?)>?\)',s)
assert all(Path(p).exists() for p in links),[p for p in links if not Path(p).exists()]
tables=0;size=None
for line in s.splitlines()+['']:
    if line.startswith('| '):
        now=line.count('|')-1
        if size is None:size=now;tables+=1
        else:assert size==now,(size,now,line)
    else:size=None
formulas=[]
def stash(m,display):
    token='MATHPLACEHOLDER'+str(len(formulas))+'END'
    formulas.append((token,m[1],display))
    return '\n\n'+token+'\n\n' if display else token
preview=re.sub(r'\$\$(.*?)\$\$',lambda m:stash(m,True),s,flags=re.S)
preview=re.sub(r'(?<!\$)\$([^\n$]+)\$(?!\$)',lambda m:stash(m,False),preview)
body=markdown.markdown(preview,extensions=['tables','fenced_code'])
for token,tex,display in formulas:
    elem=('div','math-block') if display else ('span','math-inline')
    value='<'+elem[0]+' class="'+elem[1]+'">'+html.escape(tex)+'</'+elem[0]+'>'
    body=body.replace('<p>'+token+'</p>',value) if display else body.replace(token,value)
assert 'MATHPLACEHOLDER' not in body
body=re.sub(r'(src|href)="(C:/[^\"]+)"',r'\1="file:///\2"',body)
style='''body{font:16px/1.8 "Microsoft YaHei",sans-serif;max-width:1200px;margin:32px auto;padding:24px;color:#1e293b}h1{font-size:29px}h2{font-size:24px;margin-top:40px}h3{font-size:20px;margin-top:30px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{border:1px solid #d6dce5;padding:8px 10px;text-align:left}th{background:#eff4fa}img{max-width:100%}.math-block{padding:18px 0;overflow-x:auto}math{font-family:"Cambria Math",serif;font-size:20px}a{color:#245ca3}code{background:#f3f4f6}pre{white-space:pre-wrap}blockquote{border-left:4px solid #466890;padding-left:18px}'''
(OUT/'report_preview.html').write_text('<!doctype html><meta charset="utf-8"><style>'+style+'</style>'+body,encoding='utf8')
result=dict(passed=True,tables=tables,formula_blocks=len(blocks),inline_formulas=len(formulas)-len(blocks),
            existing_links=len(links),characters=len(s),no_control_characters=True)
(OUT/'report_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(result)

