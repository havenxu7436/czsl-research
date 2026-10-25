from pathlib import Path
import ast,html,json,re
import markdown
BASE=Path(__file__).resolve().parent.parent
ROOT=BASE/'trustcpf_stage_v1_10ep'
HERE=Path(__file__).resolve().parent
reports={}
for name in ('FORMULAS_AND_CODE.md','SERVER_COMMANDS.md'):
    source=(ROOT/name).read_text(encoding='utf8')
    assert not any(ord(c)<32 and c not in '\n' for c in source)
    protected=[]
    def code(m):
        token='CODEBLOCKPLACEHOLDER'+str(len(protected))+'END'
        protected.append((token,m[0]))
        return token
    fence=re.escape(chr(96)*3)
    s=re.sub(fence+r'.*?'+fence,code,source,flags=re.S)
    formulas=[]
    def math(m,display):
        token='MATHPLACEHOLDER'+str(len(formulas))+'END';formulas.append((token,m[1],display))
        return '\n\n'+token+'\n\n' if display else token
    s=re.sub(r'\$\$(.*?)\$\$',lambda m:math(m,True),s,flags=re.S)
    s=re.sub(r'(?<!\$)\$([^\n$]+)\$(?!\$)',lambda m:math(m,False),s)
    for token,block in protected:s=s.replace(token,block)
    body=markdown.markdown(s,extensions=['tables','fenced_code'])
    for token,tex,display in formulas:
        tag='div' if display else 'span';cls='math-block' if display else 'math-inline'
        val='<'+tag+' class="'+cls+'">'+html.escape(tex)+'</'+tag+'>'
        body=body.replace('<p>'+token+'</p>',val) if display else body.replace(token,val)
    style='''body{font:16px/1.8 "Microsoft YaHei",sans-serif;max-width:1200px;margin:32px auto;padding:24px;color:#1e293b}h1{font-size:29px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d6dce5;padding:8px}th{background:#eef3fa}.math-block{padding:20px 0;overflow-x:auto}math{font-family:"Cambria Math";font-size:20px}pre{font:14px/1.6 Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:16px;border-radius:6px}'''
    (HERE/(Path(name).stem+'.html')).write_bytes(('<!doctype html><meta charset="utf-8"><style>'+style+'</style>'+body).encode())
    reports[name]=dict(display_formulas=sum(x[2] for x in formulas),inline_formulas=sum(not x[2] for x in formulas),code_blocks=len(protected))
python_files=list(ROOT.glob('*.py'))+list((ROOT/'trustcpf').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))
for p in python_files:ast.parse(p.read_text(encoding='utf-8-sig'),feature_version=(3,8))
for p in list((ROOT/'scripts').glob('*.sh'))+list((ROOT/'scripts').glob('*.ps1')):
    assert b'\r' not in p.read_bytes(),p
(ROOT/'validation/document_checks.json').write_bytes((json.dumps(dict(passed=True,documents=reports,python38_syntax_files=len(python_files)),indent=2)+'\n').encode())
print(json.dumps(reports))

