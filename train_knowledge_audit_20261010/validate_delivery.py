"""Parse delivered shell blocks without executing their operations."""
import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOC = ROOT.parent / 'output/下一步方案与具体命令_TRAIN知识落地审查_20261010.md'
BASH = Path(r'C:\Users\llenovo\Desktop\Git\bin\bash.exe')


def main():
    document = DOC.read_text(encoding='utf-8')
    blocks = re.findall(r'```(bash|powershell)\n(.*?)\n```', document, re.S)
    counts = {'bash': 0, 'powershell': 0}
    with tempfile.TemporaryDirectory(prefix='train_audit_command_parse_') as temp:
        for index, (language, content) in enumerate(blocks):
            p = Path(temp) / ('block_%02d.%s' % (index, 'sh' if language == 'bash' else 'ps1'))
            p.write_text(content + '\n', encoding='utf-8')
            if language == 'bash':
                result = subprocess.run([str(BASH), '-n', p.as_posix()], capture_output=True, text=True)
            else:
                literal = str(p).replace("'", "''")
                command = "[scriptblock]::Create((Get-Content -LiteralPath '%s' -Raw -Encoding UTF8)) | Out-Null" % literal
                result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                                        capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('%s block %d failed: %s' % (language, index, result.stderr))
            counts[language] += 1
    report = {'passed': True, 'blocks': counts, 'operations_executed': False}
    (ROOT / 'command_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
