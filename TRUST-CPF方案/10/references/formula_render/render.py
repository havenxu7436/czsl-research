from pathlib import Path
import re, json
import markdown

ROOT=Path(__file__).resolve().parent
DOC=ROOT.parent.parent/'trustcpf_abc_v1_10ep/FORMULAS.md'
s=DOC.read_text(encoding='utf-8')
formulas=[]
def capture(m, display):
    token='MATHPLACEHOLDER%04dEND'%len(formulas)
    formulas.append(dict(token=token,tex=m.group(1).strip(),display=display))
    return ('\n\n'+token+'\n\n') if display else token
body=re.sub(r'\$\$([\s\S]*?)\$\$',lambda m:capture(m,True),s)
body=re.sub(r'(?<!\\)\$([^\n$]+?)(?<!\\)\$',lambda m:capture(m,False),body)
assert '$' not in body
body=markdown.markdown(body,extensions=['tables','fenced_code'])
(ROOT/'body.html').write_text(body,encoding='utf-8')
(ROOT/'formulas.json').write_text(json.dumps(formulas,ensure_ascii=False),encoding='utf-8')
errors=[];cols=None;tables=0
for i,line in enumerate(s.splitlines(),1):
    if line.startswith('|'):
        n=line.count('|')-1
        if cols is None:cols=n;tables+=1
        elif cols!=n:errors.append(f'line {i}: table width')
    else:cols=None
for m in re.finditer(r'\]\((<[^>]+>|[^)\n]+)\)',s):
    p=m.group(1).strip('<>')
    if not p.startswith('http') and not Path(p).exists():errors.append('Missing link: '+p)
if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]',s):errors.append('Control/invalid character')
result=dict(passed=not errors,characters=len(s),formulas=len(formulas),display=sum(v['display'] for v in formulas),inline=sum(not v['display'] for v in formulas),tables=tables,errors=errors)
(ROOT/'markdown_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result));assert result['passed']
