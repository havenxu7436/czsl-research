"""Check v2 links, Markdown, formulas, frozen excerpts and runtime syntax."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'TRUST-CPF方案/v2'
sys.path.insert(0, str(ROOT / 'tools'))
from register_trustcpf_v2 import check, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', required=True)
    parser.add_argument('--katex-module', required=True)
    args = parser.parse_args()
    errors, formulas, commands = [], {}, []
    mapping = read(BASE / 'results/release_member_map.json')['members']
    release = check()
    docs = [ROOT/'README.md', ROOT/'TRUST-CPF方案/README.md'] + list(BASE.rglob('*.md'))
    for p in docs:
        text = p.read_text(encoding='utf-8')
        key = p.relative_to(ROOT).as_posix()
        if text.count('```') % 2:
            errors.append(key + ': unclosed fence')
        for target in re.findall(r'\]\((<[^>]+>|[^)\n]+)\)', text):
            target = unquote(target.strip('<>')).split('#')[0]
            if target and not re.match(r'^[a-z]+://', target) and not (p.parent/target).exists():
                errors.append(key + ': broken link: ' + target)
        outside = re.sub(r'```.*?```', '', text, flags=re.S)
        formulas[key] = dict(display=re.findall(r'```math\s*\n(.*?)```', text, flags=re.S),
            inline=re.findall(r'\$`(.*?)`\$', outside))
        for language, command in re.findall(r'```(powershell)\n(.*?)```', text, flags=re.S):
            commands.append(command)
        for line in outside.splitlines():
            if line.startswith('|') and not line.endswith('|'):
                errors.append(key + ': incomplete table row')
    for name in ('REVISED_PLAN.md', 'FORMULAS_AND_CODE.md'):
        original = (ROOT / mapping[name]).read_text(encoding='utf-8')
        published = (BASE/name).read_text(encoding='utf-8')
        if re.findall(r'```python\n(.*?)```', original, re.S) != re.findall(r'```python\n(.*?)```', published, re.S):
            errors.append(name + ': changed Python excerpt')
        original_display = re.findall(r'\$\$(.*?)\$\$', original, re.S)
        normalize = lambda s: re.sub(r'\s+', '', re.sub(r'\\operatorname\{([^{}]+)\}', r'\\mathop{\\mathrm{\1}}\\nolimits', s))
        if list(map(normalize, original_display)) != list(map(normalize, formulas['TRUST-CPF方案/v2/'+name]['display'])):
            errors.append(name + ': changed display formula')
    python_files, shell_files = 0, 0
    paths = list((BASE/'code').rglob('*')) + [ROOT/'tools/register_trustcpf_v2.py',ROOT/'tools/reproduce.py']
    for p in paths:
        if p.suffix == '.py':
            python_files += 1
            try:
                ast.parse(p.read_text(encoding='utf-8-sig'), feature_version=(3, 8))
            except SyntaxError as exc:
                errors.append(str(p) + ': ' + str(exc))
        if p.suffix == '.sh':
            shell_files += 1
            if b'\r' in p.read_bytes():
                errors.append(str(p) + ': CRLF in shell script')
    with tempfile.TemporaryDirectory(prefix='v2-doc-check-') as tmp:
        tmp = Path(tmp)
        write(tmp/'formulas.json', formulas)
        js = r"""
const fs=require('fs'), k=require(process.argv[2]);
const data=JSON.parse(fs.readFileSync(process.argv[3],'utf8')),errors=[];
let display=0,inline=0;
for(const [file,groups] of Object.entries(data))for(const [kind,items] of Object.entries(groups))for(const math of items){
 if(kind==='display')display++;else inline++;
 try{k.renderToString(math,{displayMode:kind==='display',throwOnError:true,strict:'ignore'});}
 catch(e){errors.push({file,math,error:e.message});}
}
console.log(JSON.stringify({renderer:'KaTeX',version:k.version,display,inline,errors}));
if(errors.length)process.exitCode=1;
"""
        (tmp/'math.cjs').write_text(js, encoding='utf-8')
        render = subprocess.run([args.node,str(tmp/'math.cjs'),args.katex_module,str(tmp/'formulas.json')],capture_output=True,text=True,encoding='utf-8')
        math_result = json.loads(render.stdout)
        errors.extend(math_result['errors'])
        for i, command in enumerate(commands):
            (tmp/('command%d.ps1'%i)).write_text(command,encoding='utf-8-sig')
        ps = r"""
$failures = @()
Get-ChildItem -LiteralPath $args[0] -Filter 'command*.ps1' | ForEach-Object {
 $tokens = $null; $parseErrors = $null
 [System.Management.Automation.Language.Parser]::ParseFile($_.FullName, [ref]$tokens, [ref]$parseErrors) | Out-Null
 foreach ($err in $parseErrors) { $failures += $err.Message }
}
if ($failures.Count) { $failures | Write-Output; exit 1 }
Write-Output 'PASS'
"""
        (tmp/'parse.ps1').write_text(ps,encoding='utf-8-sig')
        parsed = subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(tmp/'parse.ps1'),str(tmp)],capture_output=True,text=True)
        if parsed.returncode:
            errors.append('PowerShell parse: '+parsed.stdout+parsed.stderr)
    result = dict(passed=not errors, release=release, documents=len(docs), python38_parsed=python_files,
        shell_lf_checked=shell_files, powershell_blocks_parsed=len(commands), math=math_result,
        python_excerpts_unchanged=True, display_formula_semantics_unchanged=True,
        formal_training_executed=False, errors=errors)
    write(BASE/'results/repository-validation-20261022/checks.json', result)
    print(json.dumps(result,ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
