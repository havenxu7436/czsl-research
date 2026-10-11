"""Predeclared prompt families. Permute only supplemental descriptions."""
from common import require


VARIANTS = ('T_name', 'T_K', 'T_perm1', 'T_perm2',
            'T_AE', 'T_AE_perm1', 'T_AE_perm2')


def make_prompts(protocol, ae, oe):
    labels, names = protocol['labels'], protocol['names']
    permutations = protocol['permutations']
    result = {}
    for variant in VARIANTS:
        perm_key = 'P_perm1' if variant.endswith('perm1') else (
            'P_perm2' if variant.endswith('perm2') else 'P_K')
        perm = permutations[perm_key]
        rows = []
        for oi, obj in enumerate(labels['objs']):
            for ai, attr in enumerate(labels['attrs']):
                base = 'A photo of {} made of {}.'.format(names['o'][oi], names['a'][ai])
                # Nine equal-weight units for ALL variants. Name duplicates and
                # AE duplicates are intentional; encode unique texts once only.
                for aj in range(3):
                    for oj in range(3):
                        prompt = base
                        if variant != 'T_name':
                            attribute_source = labels['attrs'][perm['a'][ai]]
                            prompt += ' It has {}.'.format(ae[attribute_source][aj])
                            if not variant.startswith('T_AE'):
                                object_source = labels['objs'][perm['o'][oi]]
                                prompt += ' Object shape: {}.'.format(oe[object_source][oj])
                        rows.append(prompt)
        require(len(rows) == 12 * 16 * 9, 'Unexpected prompt count')
        result[variant] = rows
    return result


def check_descriptions(protocol, ae, oe):
    for key, table in (('attrs', ae), ('objs', oe)):
        require(set(table) == set(protocol['labels'][key]), 'Description labels differ')
        require(all(isinstance(v, list) and len(v) == 3 and
                    all(isinstance(t, str) and t.strip() for t in v)
                    for v in table.values()), 'Expected three descriptions per class')
    for name, p in protocol['permutations'].items():
        for key, n in (('a', 16), ('o', 12)):
            require(sorted(p[key]) == list(range(n)), 'Not a permutation')
            if name.startswith('P_perm'):
                require(all(i != j for i, j in enumerate(p[key])), 'Not a derangement')
