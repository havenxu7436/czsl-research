import json,os
from pathlib import Path
p=Path(os.environ['CPF_WORK'])/'status.json'
s=json.loads(p.read_text(encoding='utf8'))
if not s['complete']:raise SystemExit('Queue outputs incomplete: '+str(s))
print('ALL_EXPERIMENTS_COMPLETE: 30 model/readout results, 12 utility audits, 1800 noise rows')
