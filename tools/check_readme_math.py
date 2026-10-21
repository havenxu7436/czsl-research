"""Check README math delimiters and, optionally, GitHub GFM round trips.

    python tools/check_readme_math.py
    python tools/check_readme_math.py --github --report tmp/readme_math.json

The GitHub option calls the render-only Markdown API. MathJax visual checks
are separate from this Markdown/source-integrity check.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
INLINE = re.compile(r'\$`([^`\n]+)`\$|(`+)(.*?)\2|(?<!\\)\$([^$`\n]+)\$')


def extract(text):
    formulas, errors = [], []
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r'^\s*(`{3,}|~{3,})(.*)$', line)
        if match:
            if fence:
                if match[1][0] == fence[0] and len(match[1]) >= len(fence):
                    if language == 'math':
                        formulas.append(dict(line=start, display=True, tex='\n'.join(block)))
                    fence = None
            else:
                fence, language, start, block = match[1], match[2].strip(), number, []
            continue
        if fence:
            block.append(line)
            if language == 'math' and re.search(r'(?<!\\)\\\\\s*$', line):
                errors.append(dict(line=number, error='Bare row break: use an explicit spacing argument, e.g. \\\\[4pt]'))
            continue
        for match in INLINE.finditer(line):
            if match[1] is not None:
                formulas.append(dict(line=number, display=False, tex=match[1]))
            elif match[4] is not None:
                formulas.append(dict(line=number, display=False, tex=match[4]))
                errors.append(dict(line=number, error='Unprotected inline math: use $`...`$'))
        if re.search(r'(?<!\\)\$', INLINE.sub('', line)):
            errors.append(dict(line=number, error='Unmatched math delimiter outside a code span'))
    if fence:
        errors.append(dict(line=start, error='Unclosed code/math fence'))
    for formula in formulas:
        if r'\operatorname' in formula['tex']:
            errors.append(dict(line=formula['line'], error='GitHub rejects operatorname; use a compatible upright operator'))
        if '$$' in formula['tex']:
            errors.append(dict(line=formula['line'], error='Nested display delimiters inside math fence'))
    return formulas, errors


class GitHubMath(HTMLParser):
    def __init__(self):
        super().__init__()
        self.formulas, self.current = [], None

    def handle_starttag(self, tag, attrs):
        if tag == 'math-renderer':
            self.current = dict(display='js-display-math' in dict(attrs).get('class', ''), tex='')
        elif self.current is not None:
            self.current.setdefault('nested_tags', []).append(tag)

    def handle_endtag(self, tag):
        if tag == 'math-renderer' and self.current is not None:
            item = self.current
            delimiter = '$$' if item['display'] else '$'
            tex = item['tex']
            if tex.startswith(delimiter) and tex.endswith(delimiter):
                item['tex'] = tex[len(delimiter):-len(delimiter)]
            self.formulas.append(item)
            self.current = None

    def handle_data(self, data):
        if self.current is not None:
            self.current['tex'] += data


def check(path, args):
    data = (ROOT/path).read_bytes()
    text = data.decode('utf-8-sig')
    formulas, errors = extract(text)
    if not formulas and not errors:
        return None
    row = dict(path=path, sha256=hashlib.sha256(data).hexdigest(),
               display=sum(f['display'] for f in formulas),
               inline=sum(not f['display'] for f in formulas), errors=errors)
    if args.github and formulas:
        request = urllib.request.Request('https://api.github.com/markdown',
            data=json.dumps(dict(text=text, mode='gfm')).encode('utf-8'),
            headers={'Content-Type': 'application/json', 'User-Agent': 'CZSL-README-math-check'})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.direct else urllib.request.build_opener()
        try:
            with opener.open(request, timeout=30) as response:
                rendered = response.read().decode('utf-8')
            parser = GitHubMath()
            parser.feed(rendered)
            expected = [(f['display'], f['tex']) for f in formulas]
            actual = [(f['display'], f['tex']) for f in parser.formulas]
            row['github_source_round_trip'] = expected == actual and not any(f.get('nested_tags') for f in parser.formulas)
            row['github_display'] = sum(f['display'] for f in parser.formulas)
            row['github_inline'] = sum(not f['display'] for f in parser.formulas)
            if not row['github_source_round_trip']:
                errors.append(dict(error='GitHub changed or omitted formula source'))
            if args.html_dir:
                target = args.html_dir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.with_suffix('.html').write_text(rendered, encoding='utf-8')
        except Exception as exc:
            errors.append(dict(error=f'GitHub Markdown check failed: {exc}'))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--github', action='store_true', help='Verify source preservation through GitHub Markdown API')
    parser.add_argument('--direct', action='store_true', help='Use a direct connection for the GitHub check')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--html-dir', type=Path, help='Save rendered GitHub HTML for separate MathJax/layout QA')
    args = parser.parse_args()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    paths = [p for p in tracked if p.endswith('README.md')]
    with ThreadPoolExecutor(max_workers=3 if args.github else 1) as pool:
        rows = [r for r in pool.map(lambda p: check(p, args), paths) if r is not None]
    result = dict(passed=all(not r['errors'] for r in rows), readmes_scanned=len(paths),
                  readmes_with_math=len(rows), display=sum(r['display'] for r in rows),
                  inline=sum(r['inline'] for r in rows), github_checked=args.github, documents=rows)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
