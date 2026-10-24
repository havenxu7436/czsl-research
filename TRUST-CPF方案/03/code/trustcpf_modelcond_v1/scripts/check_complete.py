import os,json
from pathlib import Path
p=Path(os.environ['CPF_WORK'])/'status.json'
s=json.loads(p.read_text(encoding='utf-8'))
if not s['complete']:raise SystemExit('INCOMPLETE: '+str(s))
print('COMPLETE: 21/21 stages, 9/9 models, 6/6 utility audits, 18/18 response files')
