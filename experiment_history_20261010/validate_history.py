"""Read-only validation of the authored retrospective and its source receipt."""
import hashlib
import json
import re
from pathlib import Path
import build_history as history

path = history.OUT
text = path.read_text(encoding='utf-8')
receipt = json.loads(history.RECEIPT.read_text(encoding='utf-8'))
assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
assert receipt['training_started'] is False
for source, identity in receipt['source_files'].items():
    data = Path(source).read_bytes()
    assert len(data) == identity['bytes'], source
    assert hashlib.sha256(data).hexdigest() == identity['sha256'], source
for name, digest in receipt['authored_inputs'].items():
    assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
assert '{{TABLE:' not in text and '{{SOURCE:' not in text
assert text.count('$$') % 2 == 0
leading_dot_decimals = re.findall(r'(?<!\d)[+-]?\.\d+', text)
assert not leading_dot_decimals, ('decimal needs a leading zero', leading_dot_decimals[:10])
display_math = re.findall(r'\$\$([\s\S]*?)\$\$', text)
aligned_math = [block for block in display_math if r'\begin{aligned}' in block]
assert all(block.count(r'\begin{aligned}') == block.count(r'\end{aligned}') for block in aligned_math)
assert not any(re.search(r'&\s*(?:\\\\)?\s*$', block, re.M) for block in aligned_math), 'trailing alignment anchor'
links = re.findall(r'\]\(<(/C:/[^>]+)>\)', text)
assert links and all(Path(link[1:]).is_file() for link in links)
lines = text.splitlines()
tables = 0
table_columns = None
for i, line in enumerate(lines):
    if line.startswith('#'):
        assert i == 0 or not lines[i-1].strip(), ('heading needs blank line',i+1)
        assert i+1 == len(lines) or not lines[i+1].strip(), ('heading needs blank line',i+1)
    if line.startswith('|'):
        # Ignore escaped pipes inside cells.
        count = len(re.findall(r'(?<!\\)\|', line))
        if table_columns is None:
            table_columns = count
            tables += 1
        assert count == table_columns, ('inconsistent table columns',i+1)
    else:
        table_columns = None
for number in range(28):
    assert re.search(r'^## '+str(number)+r'\. ',text,re.M), number
for name in ['A','B','C','D','E','F','G','H']:
    assert '附录 '+name+'：' in text
# Rebuild tables read-only; every generated cell must appear verbatim in report.
built = history.build_tables()
assert len(built) == receipt['table_count']
assert all(value in text for value in built.values())
print(json.dumps({'passed': True, 'source_hashes_checked':len(receipt['source_files']),
                  'source_links_checked':len(links),'generated_tables_checked':len(built),
                  'total_markdown_tables':tables,'main_sections':28,'appendices':8,
                  'display_formulas_checked':len(display_math),'aligned_formulas_checked':len(aligned_math),
                  'missing_decimal_leading_zeros':len(leading_dot_decimals),
                  'lines':len(lines),'bytes':path.stat().st_size,
                  'new_training':False},ensure_ascii=False))
