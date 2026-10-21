"""Build the self-contained formula and command guide for the H release."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'vgcpod_v1_roundH'
NAME = 'VG-CPOD_v1_H组实验实施方案_20261020'
PANDOC = Path('C:/Users/llenovo/anaconda3/Scripts/pandoc.exe')


def main():
    doc = PACKAGE/'EXPERIMENT_H_v1.md'
    shutil.copy2(doc,ROOT/(NAME+'.md'))
    command = [str(PANDOC),str(doc),'--standalone','--self-contained','--mathml',
        '--from=markdown+tex_math_dollars+pipe_tables','--to=html5',
        '--css='+str(ROOT/'analysis_20261018/report.css'),
        '--metadata=pagetitle:VG-CPOD v1 H组实验实施方案',
        '--output='+str(ROOT/(NAME+'.html'))]
    subprocess.run(command,check=True)
    sys.path.insert(0,str(ROOT/'tmp/round_a_bash_validation'))
    import bashlex
    text = doc.read_text(encoding='utf-8')
    blocks = re.findall(r'```bash\n(.*?)\n```',text,re.S)
    for block in blocks:
        bashlex.parse(block)
    ps_blocks = re.findall(r'```powershell\n(.*?)\n```',text,re.S)
    qa = ROOT/'analysis_roundH_20261020/qa'
    qa.mkdir(exist_ok=True)
    for i,block in enumerate(ps_blocks):
        (qa/('commands_%d.ps1'%i)).write_text(block+'\n',encoding='utf-8')
    result = dict(bash_blocks_parsed=len(blocks),powershell_blocks=len(ps_blocks),
        expected_display_formulas=text.count('$$')//2)
    (PACKAGE/'validation/document_commands.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
