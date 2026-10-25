"""Check GitHub's rendered Markdown tables and exact mathematical payloads (read-only)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, ProxyHandler, build_opener

ROOT=Path(__file__).resolve().parents[1]

def canonical(text):return re.sub(r'\s+','',text)

def audit(path,ref,out):
    relative=path.relative_to(ROOT).as_posix()
    url='https://github.com/havenxu7436/czsl-research/blob/'+ref+'/'+quote(relative)
    opener=build_opener(ProxyHandler({}))
    for attempt in range(4):
        try:
            with opener.open(Request(url,headers={'User-Agent':'CZSL-Repository-Markdown-QA'}),timeout=35) as response:
                document=response.read().decode('utf8')
            break
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc,HTTPError) and exc.code not in (429,500,502,503,504):raise
            if attempt==3:raise RuntimeError('GitHub page unavailable after retries: '+relative) from exc
            time.sleep(min(5,attempt+1))
    match=re.search(r'<article\b[^>]*class="[^"]*markdown-body[^"]*"[^>]*>[\s\S]*?</article>',document)
    if not match:raise AssertionError('No rendered Markdown article: '+relative)
    article=match[0]
    source=path.read_text(encoding='utf8')
    expected_display=re.findall(r'```math\r?\n([\s\S]*?)```',source)
    expected_inline=re.findall(r'\$`([^\n]*?)`\$',source)
    actual_display=[];actual_inline=[]
    for cls,payload in re.findall(r'<math-renderer\b[^>]*class="([^"]*)"[^>]*>([\s\S]*?)</math-renderer>',article):
        payload=html.unescape(re.sub(r'<[^>]+>','',payload)).strip()
        if 'js-display-math' in cls:actual_display.append(payload[2:-2])
        elif 'js-inline-math' in cls:actual_inline.append(payload[1:-1])
    assert list(map(canonical,expected_display))==list(map(canonical,actual_display)),('Display payload',relative,len(expected_display),len(actual_display))
    assert list(map(canonical,expected_inline))==list(map(canonical,actual_inline)),('Inline payload',relative,len(expected_inline),len(actual_inline))
    expected_tables=len(re.findall(r'(?:^|\n)\|[^\n]+\n\|\s*[-:]',source))
    tables=re.findall(r'<table\b[^>]*>[\s\S]*?</table>',article)
    assert len(tables)==expected_tables,('Tables',relative,expected_tables,len(tables))
    table_columns=[]
    for table in tables:
        rows=re.findall(r'<tr\b[^>]*>([\s\S]*?)</tr>',table)
        counts=[len(re.findall(r'<t[dh]\b',row)) for row in rows]
        assert len(set(counts))==1,('Rendered table columns',relative,counts)
        table_columns.append(counts[0])
    assert not re.search(r'(?:class="[^"]*(?:katex-error|math-error)|Unable to render rich display)',article),relative
    snapshot=out/'html'/relative
    snapshot=snapshot.with_suffix('.html');snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_text(article+'\n',encoding='utf8')
    return dict(file=relative,url=url,github_rendered=True,tables=len(tables),table_columns=table_columns,
        display_formulas=len(actual_display),inline_formulas=len(actual_inline),exact_math_payloads_verified=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--ref',default='main');ap.add_argument('--stages',nargs='+',default=['02','03','04','05','06','07','08','09']);ap.add_argument('--out',type=Path,default=ROOT/'tmp/trustcpf_web_qa')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    docs=[p for n in args.stages for p in (ROOT/'TRUST-CPF方案'/n).rglob('*.md')]
    docs += [ROOT/'README.md',ROOT/'TRUST-CPF方案/README.md']
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda p:audit(p,args.ref,args.out),docs))
    result=dict(passed=True,ref=args.ref,documents=results,document_count=len(results),
        display_formulas=sum(x['display_formulas'] for x in results),inline_formulas=sum(x['inline_formulas'] for x in results),
        tables=sum(x['tables'] for x in results),verification='GitHub HTML rendering, exact math payloads, table structure; local KaTeX validation recorded separately')
    (args.out/'checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='documents'},ensure_ascii=True))

if __name__=='__main__':main()
