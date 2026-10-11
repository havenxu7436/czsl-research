#!/usr/bin/env python3
"""Parse documentation Bash blocks; do NOT execute training/upload commands."""
import re
import shutil
import subprocess
from pathlib import Path


def main():
    text=(Path(__file__).resolve().parent/'操作手册.md').read_text(encoding='utf-8')
    commands=re.findall(r'```bash\n(.*?)```',text,re.S)
    bash=shutil.which('bash')
    if shutil.which('wsl.exe'):
        shell=['wsl.exe','-e','bash','-n']
    elif bash:
        shell=[bash,'-n']
    else:
        raise RuntimeError('No bash parser available')
    for i,body in enumerate(commands,1):
        result=subprocess.run(shell,input=body.encode('utf-8'),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if result.returncode:
            raise ValueError(f'Bash block {i}: '+result.stderr.decode('utf-8',errors='replace'))
    print('BASH_COMMAND_BLOCKS_PARSED_NOT_EXECUTED',len(commands))


if __name__=='__main__':
    main()
