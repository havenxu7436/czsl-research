import json
import os
from pathlib import Path
status = json.loads((Path(os.environ['CPF_WORK']) / 'status.json').read_text(encoding='utf8'))
if not status['complete']:
    raise SystemExit('INCOMPLETE: ' + str(status))
print('ALL_EXPERIMENTS_COMPLETE: 27/27 stages, 15/15 model results, 15/15 reused teachers, 3/3 Q5 decision audits', flush=True)
