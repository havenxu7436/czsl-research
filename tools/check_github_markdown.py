"""Reject TeX macros known to fail on the actual GitHub reading pages."""
from pathlib import Path
import json,re,subprocess
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'raw_export','references','code','confirmed','repository_qa'}
def check_text(text):
 fence=chr(96)*3;blocks=[];prose=[];parts=text.split(fence)
 for i,part in enumerate(parts):
  if i%2==0:prose.append(part)
  elif part.startswith('math\n') or part.startswith('math\r\n'):blocks.append(part.split('\n',1)[1])
 prose='\n'.join(prose)
 blocks.extend(re.findall(r'\$\$([\s\S]*?)\$\$',prose))
 blocks.extend(re.findall(r'\$'+chr(96)+r'([^\n]*?)'+chr(96)+r'\$',prose))
 return ['GitHub rejects operatorname; use mathrm or arg/max/min' for b in blocks if re.search(r'\\operatorname\b',b)]
def check():
 names=subprocess.check_output(['git','-c','core.quotepath=false','ls-files','-z','--','*.md'],cwd=ROOT).decode('utf8').split('\0')
 checked=0;failures=[]
 for name in names:
  if not name or EXCLUDED&set(Path(name).parts):continue
  p=ROOT/name
  if not p.is_file():continue
  checked+=1
  for error in check_text(p.read_text(encoding='utf-8-sig')):failures.append(dict(path=name,error=error))
 return dict(passed=not failures,reading_pages=checked,errors=failures)
if __name__=='__main__':
 result=check();print(json.dumps(result,ensure_ascii=False));raise SystemExit(0 if result['passed'] else 1)
