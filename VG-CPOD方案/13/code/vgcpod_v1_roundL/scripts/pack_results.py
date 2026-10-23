import hashlib
import os
import tarfile
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
archive = root.parent / (root.name + '_results_' + stamp + '.tar.gz')
work = Path(os.getenv('CPF_WORK', str(root / 'work'))).resolve()
items = ['run.py','requirements.txt','README.md','SERVER_COMMANDS.md','FORMULAS_AND_CODE.md','REVISED_PLAN.md',
         'ckcpf','configs','assets','initial','scripts','tests','validation','logs','server_env.sh','setup.json','MANIFEST.json']
with tarfile.open(archive, 'w:gz') as tar:
    for name in items:
        p = root / name
        if p.exists():
            tar.add(p, arcname=root.name + '/' + name,
                    filter=lambda t: None if '__pycache__' in t.name or t.name.endswith('.pyc') else t)
    if work.exists(): tar.add(work, arcname=root.name + '/work')
h = hashlib.sha256()
with archive.open('rb') as f:
    for chunk in iter(lambda: f.read(4*1024*1024), b''): h.update(chunk)
archive.with_name(archive.name + '.sha256').write_text(h.hexdigest() + '  ' + archive.name + '\n', encoding='ascii')
(root / 'latest_archive.txt').write_text(str(archive)+'\n',encoding='utf-8')
print('ARCHIVE', archive)
print('SHA256', h.hexdigest())
