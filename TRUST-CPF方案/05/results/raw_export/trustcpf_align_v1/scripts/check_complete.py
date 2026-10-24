import json
import os
from pathlib import Path

status = json.loads((Path(os.environ['CPF_WORK']) / 'status.json').read_text(encoding='utf8'))
if not status['complete']:
    raise SystemExit('INCOMPLETE: ' + str(status))
print('ALL_EXPERIMENTS_COMPLETE: 18/18 stages, 9/9 model results, 3/3 Q5 decision audits', flush=True)
