"""Isolated text dependencies and checksum-verified official model. No CPF upgrades."""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from common import read_json, sha, require, write_json


DEPENDENCIES = ('ftfy==6.1.1', 'wcwidth==0.2.13', 'regex==2024.11.6',
                'packaging==24.2', 'tqdm==4.66.6')


def prepare(work, protocol, supplied=None, offline=False):
    work = Path(work)
    deps = work / 'deps'
    receipt = work / 'dependencies.json'
    if receipt.exists():
        old = read_json(receipt)
        require(old['requirements'] == list(DEPENDENCIES), 'Dependency recipe changed')
        for relative, expected in old['files_sha256'].items():
            require(sha(deps / relative) == expected, 'Isolated dependency changed: ' + relative)
    else:
        require(not deps.exists(), 'Partial dependency directory: preserve it and request diagnosis')
        # --no-deps is essential: torch/torchvision/numpy in cpf are NOT touched.
        require(sys.version_info[:2] == (3, 8), 'Bundled Linux wheels target the existing cpf Python3.8')
        from common import ROOT
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                               '--no-index', '--find-links', str(ROOT / 'vendor/wheels'),
                               '--no-deps', '--only-binary=:all:', '--target', str(deps)] + list(DEPENDENCIES))
        write_json(receipt, {'requirements': list(DEPENDENCIES),
                            'files_sha256': {str(p.relative_to(deps)).replace('\\', '/'): sha(p)
                                             for p in deps.rglob('*') if p.is_file() and
                                             '__pycache__' not in p.parts and p.suffix != '.pyc'}})
    sys.path.insert(0, str(deps))
    expected = protocol['clip_checkpoint_sha256']
    choices = [Path(supplied)] if supplied else [
        work / 'model' / 'ViT-L-14.pt', Path.home() / '.cache' / 'clip' / 'ViT-L-14.pt',
        Path('/data/xhw/.cache/clip/ViT-L-14.pt')]
    for p in choices:
        if p.is_file():
            require(sha(p) == expected, 'CLIP checkpoint SHA mismatch: ' + str(p))
            print('CLIP_CHECKPOINT_VERIFIED', str(p), flush=True)
            return p
    require(not supplied, 'Supplied checkpoint absent')
    require(not offline, 'Offline mode: no verified ViT-L/14 checkpoint found')
    folder = work / 'model'
    folder.mkdir(exist_ok=True)
    target = folder / 'ViT-L-14.pt'
    # Preserve interrupted downloads. A retry uses a distinct new .part, never
    # overwrites a prior partial file or an existing checkpoint.
    n = 0
    while (folder / ('ViT-L-14.pt.part%d' % n)).exists():
        n += 1
    partial = folder / ('ViT-L-14.pt.part%d' % n)
    print('CLIP_DOWNLOAD_BEGIN', protocol['clip_checkpoint_url'], flush=True)
    started = last = time.monotonic()
    size = 0
    with urllib.request.urlopen(protocol['clip_checkpoint_url'], timeout=60) as source, open(partial, 'xb') as out:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            out.write(block)
            size += len(block)
            if time.monotonic() - last >= 10:
                print('CLIP_DOWNLOAD_MB', round(size / 2**20, 1), 'seconds',
                      round(time.monotonic() - started, 1), flush=True)
                last = time.monotonic()
    require(sha(partial) == expected, 'Downloaded model failed SHA verification; partial preserved')
    require(not target.exists(), 'Checkpoint target exists')
    os.replace(str(partial), str(target))
    print('CLIP_DOWNLOAD_COMPLETE', size, flush=True)
    return target
