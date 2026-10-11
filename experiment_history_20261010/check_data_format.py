"""Validate r3 table values against the pre-edit snapshot and original sources."""
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

import build_history as history

ROOT = Path(__file__).resolve().parent
BEFORE = ROOT / 'format_review_r3/before.md'


def tables(text):
    result, pending = [], []
    for line in text.splitlines() + ['']:
        if line.startswith('|'):
            pending.append([cell.strip() for cell in re.split(r'(?<!\\)\|', line)[1:-1]])
        elif pending:
            result.append(pending)
            pending = []
    return result


def normalized(cell):
    return re.sub(r'\s+', '', cell)


def main():
    before = BEFORE.read_text(encoding='utf-8')
    after = history.OUT.read_text(encoding='utf-8')
    old_tables, new_tables = tables(before), tables(after)
    old_assets = next(t for t in old_tables if t[0] == ['版本', 'K均值', 'K总体std', '描述OOV', '整句fallback'])
    new_assets = next(t for t in new_tables if t[0][0] == '版本' and 'K 均值' in t[0])
    for old, new in zip(old_assets[2:], new_assets[2:]):
        expected = list(old)
        expected[3] = old[3].split('%')[0]
        assert expected == new, ('7.2 statistics changed', old, new)
    assert len(old_assets) == len(new_assets)

    # New detail tables are reconstructed from the original machine-readable files.
    built = history.build_tables()
    for key in ['KNOWLEDGE_ASSETS', 'KNOWLEDGE_OOV', 'KNOWLEDGE_ERRORS']:
        assert built[key] in after, key
    v2_bank = history.js('knowledge_v2_bank')
    for axis in ['a', 'o']:
        raw = v2_bank[axis]['residual_sum_max_abs']
        shown = Decimal(format(raw, '.10f'))
        assert shown > 0 and abs(shown - Decimal(str(raw))) <= Decimal('0.00000000005')
    assert '0.0000000745' in after and '0.0000000596' in after
    assert '7.45 × 10⁻⁸' in after and '5.96 × 10⁻⁸' in after

    new_details = [t for t in new_tables if t[0][0] in ['UT v2 描述类别', '数值检查项']]
    assert len(new_details) == 2
    old_rest = [t for t in old_tables if t is not old_assets]
    new_rest = [t for t in new_tables if t is not new_assets and t not in new_details]
    assert len(old_rest) == len(new_rest)
    checked_cells = 0
    for index, (old, new) in enumerate(zip(old_rest, new_rest)):
        if old[0][0] == '本次实际读取的数据/资料':
            # Existing evidence hashes stay identical; r3 adds three source files.
            old_hashes = dict(old[2:])
            new_hashes = dict(new[2:])
            assert old_hashes.items() <= new_hashes.items()
            assert len(new_hashes) == len(old_hashes) + 3
            continue
        assert len(old) == len(new), ('table row count', index)
        for old_row, new_row in zip(old[2:], new[2:]):
            assert len(old_row) == len(new_row)
            for col, (old_cell, new_cell) in enumerate(zip(old_row, new_row)):
                if old_row[0] == '对象上下文错配' and col == 1:
                    assert old_cell == '分数均差3.30e−5／3.83e−5；各1个预测改变'
                    assert new_cell == 'CW 分数均差约 0.0000330；OW 约 0.0000383；各有 1 个预测改变'
                    assert Decimal('3.30e-5') == Decimal('0.0000330')
                    assert Decimal('3.83e-5') == Decimal('0.0000383')
                else:
                    assert normalized(old_cell) == normalized(new_cell), ('table cell changed', index, old_cell, new_cell)
                checked_cells += 1

    # Fail if generated decimal formatting would silently turn a nonzero value into zero.
    tiny_values = []
    original_formatter = history.f
    def checked_formatter(value):
        if isinstance(value, float) and value != 0 and abs(value) < 0.0000005:
            tiny_values.append(value)
        return original_formatter(value)
    history.f = checked_formatter
    try:
        history.build_tables()
    finally:
        history.f = original_formatter
    assert not tiny_values, ('nonzero value rounded to zero', tiny_values)

    report = {
        'passed': True,
        'markdown_sha256': hashlib.sha256(history.OUT.read_bytes()).hexdigest(),
        'baseline_sha256': hashlib.sha256(BEFORE.read_bytes()).hexdigest(),
        'original_tables': len(old_tables), 'revised_tables': len(new_tables),
        'unchanged_or_value_equivalent_cells_checked': checked_cells,
        'section_7_2_source_tables_checked': 3,
        'source_based_detail_tables_added': 2,
        'nonzero_values_rounded_to_zero': len(tiny_values),
        'experimental_values_preserved': True,
    }
    (ROOT / 'format_review_r3/data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
