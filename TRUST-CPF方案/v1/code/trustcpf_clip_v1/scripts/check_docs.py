"""Validate formula delimiters, Python excerpts, symbols and relative file links."""
import ast
import re
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    records={}
    for name in ('TRUST-CPF_CLIP_修订实验方案v1.md','TRUST-CPF公式及代码v1.md','服务器实践命令.md','README.md'):
        text=(ROOT/name).read_text(encoding='utf-8')
        blocks=re.findall(r'```python\n(.*?)\n```',text,re.S)
        for code in blocks:
            try: ast.parse(code)
            except IndentationError: ast.parse('class Extracted:\n'+code)
        plain=re.sub(r'```.*?```','',text,flags=re.S)
        assert plain.count('$$')%2==0,name
        assert not re.search(r'(?<!\\)qquad',plain),name
        maths=re.findall(r'\$\$(.*?)\$\$',plain,re.S)
        for math in maths:
            brace=0
            for c in re.sub(r'\\[{}]','',math):
                brace+=int(c=='{')-int(c=='}'); assert brace>=0,(name,math)
            assert brace==0,(name,math)
        tags=[int(x) for x in re.findall(r'\\tag\{(\d+)\}',plain)]
        if tags: assert tags==list(range(1,len(tags)+1))
        for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',text):
            if not target.startswith(('http:', 'https:')): assert (ROOT/target).exists(),(name,target)
        records[name]={'display_equations':len(maths),'numbered_equations':len(tags),
            'python_excerpts':len(blocks),'sha256':hashlib.sha256(text.encode()).hexdigest()}
    (ROOT/'validation/doc_checks.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DOC_CHECKS_OK',len(records),'documents')


if __name__=='__main__':main()
