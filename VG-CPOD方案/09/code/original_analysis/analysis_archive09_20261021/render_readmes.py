"""Render GitHub math fences as MathML for visual QA of the five READMEs."""
from pathlib import Path
import json
import re
import subprocess

WORK = Path(__file__).resolve().parents[1]
STAGE = WORK.parent/'CZSL/VG-CPOD方案/09'
QA = STAGE/'analysis/repository_qa'
QA.mkdir(parents=True,exist_ok=True)
PANDOC = Path('C:/Users/llenovo/anaconda3/Scripts/pandoc.exe')
CSS = WORK/'analysis_20261018/report.css'
rows = []
for name,relative in [('main','README.md'),('code','code/README.md'),('configs','configs/README.md'),('results','results/README.md'),('analysis','analysis/README.md')]:
    p = STAGE/relative
    text = p.read_text(encoding='utf-8')
    display = text.count('```math\n')
    assert len(re.findall(r'^```math\n(.*?)\n```',text,re.M|re.S))==display
    converted = re.sub(r'^```math\n(.*?)\n```',lambda m:'$$\n'+m[1]+'\n$$',text,flags=re.M|re.S)
    result = subprocess.run([str(PANDOC),'--standalone','--self-contained','--mathml',
        '--from=markdown+tex_math_dollars+pipe_tables','--to=html5','--css='+str(CSS),
        '--metadata=pagetitle:09 '+name,'--output='+str(QA/(name+'.html'))],
        cwd=p.parent,input=converted,text=True,encoding='utf-8',capture_output=True,check=True)
    assert 'Could not convert TeX math' not in result.stderr,result.stderr
    rows.append(dict(document=relative,html=name+'.html',display_formulas=display,stderr=result.stderr))
(QA/'render.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rows,ensure_ascii=False))
