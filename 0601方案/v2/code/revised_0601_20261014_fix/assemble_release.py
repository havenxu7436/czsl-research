"""Mechanical source sync, config previews, manifests and immutable ZIP."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path
import yaml
from common import sha,write
from prepare_revision import CORE,SMALL,EXTERNAL,variant_options

HERE=Path(__file__).absolute().parent


def assemble(target):
    for name in ('revision_math.py','revision_model.py'):
        shutil.copyfile(HERE/name,HERE/'source/models'/name)
    shutil.copyfile(HERE/'smoke_revision.py',HERE/'source/tests/smoke_formula.py')
    base=yaml.safe_load((HERE/'assets/A_ref.yml').read_text(encoding='utf-8'))
    options={name:variant_options(name) for name in CORE+SMALL+EXTERNAL}
    write(HERE/'configs/variant_options.json',options)
    for name,opt in options.items():
        c=json.loads(json.dumps(base))
        work=Path('/var/tmp/czsl_0601_revision_20261014_fix')
        signature=(opt['count'],opt['noise'],opt['shuffle'],opt['aggregation'],opt['kind'])
        cache=work/('cache_revision_base' if signature==(3,0,False,'mean','text') else 'cache_'+name)
        c['MODEL'].update(kcc_text=opt['text'],kcc_calib=opt['prior'],kcc_feature=opt['feature'],kcc_logic=opt['logic'],
            kcc_lambda=0.1 if opt['prior'] else 0.0,kcc_beta=0.01 if opt['logic'] else 0.0,
            kcc_cache=cache.as_posix(),bf_logic_bank=(work/'logic_bank').as_posix(),
            formula_joint=opt['rev_enabled'],kcc_score_mode='0601_formula' if opt['rev_enabled'] else 'legacy',
            full_use_pair=opt['use_pair'],full_gain=1.0)
        c['MODEL'].update({k:v for k,v in opt.items() if k.startswith('rev_')})
        run=work/'runs'/(name+'_s3351')
        c['TRAIN'].update(seed=3351,max_epoch=10,final_max_epoch=10,start_epoch=1,kcc_stage='unified_0601_revision_20261013',
              kcc_variant=name+'_s3351',train_name='rev_'+name+'_s3351',kcc_test_at_end=False,
              kcc_run_dir=run.as_posix(),log_dir=(run/'logs').as_posix(),checkpoint_dir=(run/'checkpoints').as_posix())
        (HERE/'configs'/(name+'.yml')).write_text(yaml.safe_dump(c,sort_keys=True),encoding='utf-8')
    files={p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and
           '__pycache__' not in p.parts and p.name!='release_manifest.json' and p.suffix!='.pyc'}
    write(HERE/'release_manifest.json',files)
    if target.exists(): raise FileExistsError('Preserve existing release ZIP')
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files): z.write(HERE/name,'revised_0601_20261013/'+name)
        z.write(HERE/'release_manifest.json','revised_0601_20261013/release_manifest.json')
    target.with_suffix(target.suffix+'.sha256').write_text(sha(target)+'  '+target.name+'\n',encoding='utf-8')
    print(json.dumps(dict(archive=str(target),sha256=sha(target),manifest_files=len(files)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True)
    assemble(p.parse_args().zip)
