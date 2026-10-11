"""Verify the fixed release, unpack safely, and start the authorized tmux run."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import zipfile

EXPECTED_SHA256 = 'e4a0e0e6cdf8fa3bab811bffc52fa7dc51e5012a6b14874e1b7dfb85a9c1611f'
ARCHIVE = Path('/tmp/ekr_v2_budget10_20261008.zip')
TARGET = Path('/tmp/ekr_v2_budget10_20261008')
PY = '/data/xhw/miniconda3/envs/cpf/bin/python'
PRIOR = Path('/var/tmp/czsl_xhw_ekr_v2_align_20261007')
OUT = Path('/var/tmp/czsl_xhw_ekr_v2_align_budget10_20261008')
SESSION = 'ekr_v2_e10'


def require(ok, message):
    if not ok: raise SystemExit(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def unpack():
    require(ARCHIVE.is_file(), 'UPLOAD THE BUDGET10 ZIP TO /tmp FIRST')
    require(sha(ARCHIVE)==EXPECTED_SHA256, 'ZIP SHA256 MISMATCH')
    require(not TARGET.is_symlink(), 'BUNDLE TARGET MUST NOT BE A SYMLINK')
    with zipfile.ZipFile(ARCHIVE) as z:
        names=z.namelist()
        require(len(names)==len(set(names)), 'DUPLICATE ZIP MEMBER')
        require(z.testzip() is None, 'ZIP CRC FAILED')
        for item in z.infolist():
            p=PurePosixPath(item.filename)
            require(not p.is_absolute() and '..' not in p.parts and len(p.parts)==2 and
                    p.parts[0]==TARGET.name and '\\' not in item.filename,
                    'WRONG OR UNSAFE ZIP ROOT')
            require(not stat.S_ISLNK(item.external_attr>>16), 'ZIP LINKS ARE NOT ALLOWED')
        if TARGET.exists():
            require(TARGET.is_dir(), 'BUNDLE TARGET IS NOT A DIRECTORY')
            require({p.name for p in TARGET.iterdir()}=={PurePosixPath(n).name for n in names},
                    'EXISTING BUNDLE INVENTORY DIFFERS; PRESERVE IT')
            for item in z.infolist():
                p=TARGET/PurePosixPath(item.filename).name
                require(not p.is_symlink() and p.is_file() and p.read_bytes()==z.read(item),
                        'EXISTING BUNDLE DIFFERS; PRESERVE IT')
        else:
            z.extractall(str(TARGET.parent))
    print('BUDGET10 ZIP SHA256 AND CRC PASSED',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    unpack()
    subprocess.run([PY,str(TARGET/'verify_release.py')],check=True)
    for variant in ('V_name_align','V_clean_align','V_shuffle_align','V_clean_fixed'):
        require((PRIOR/'runs'/variant/'checkpoints/last_epoch.pth').is_file(),
                'OLD EPOCH-3 CHECKPOINT MISSING: '+variant)
    require(shutil.disk_usage(OUT.parent).free>=5*1024**3, 'NEED 5 GiB FREE ON THE OUTPUT DISK')
    if args.check_only:
        print('BOOTSTRAP CHECK PASSED; NO TRAINING STARTED',flush=True)
        return
    require(shutil.which('tmux') is not None and shutil.which('flock') is not None,
            'tmux AND flock MUST BE AVAILABLE')
    exists=subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode==0
    if exists:
        print('SESSION EXISTS; ATTACH AND CHECK BEFORE RESTARTING',flush=True)
    else:
        command='bash /tmp/ekr_v2_budget10_20261008/run.sh all; task_exit=$?; echo "BUDGET10_EXIT=$task_exit"; exec bash'
        subprocess.run(['tmux','new-session','-d','-s',SESSION,command],check=True)
        print('TMUX SESSION STARTED',SESSION,flush=True)
    print('tmux attach -t '+SESSION,flush=True)
    print('OUTPUT '+str(OUT),flush=True)


if __name__=='__main__':main()
