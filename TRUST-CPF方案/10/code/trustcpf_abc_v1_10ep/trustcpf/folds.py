import numpy as np


def make_folds(pairs, seed=2026, count=5):
    pairs = sorted(map(tuple, pairs))
    primitive = lambda c: {('a',c[0]),('o',c[1])}
    counts = {v:sum(v in primitive(p) for p in pairs) for c in pairs for v in primitive(c)}
    common = [p for p in pairs if any(counts[v]==1 for v in primitive(p))]
    rest = [p for p in pairs if p not in common]
    rng = np.random.RandomState(seed)
    for _ in range(10000):
        perm = rng.permutation(len(rest))
        blocks = [[rest[i] for i in ids] for ids in np.array_split(perm,count)]
        if all(all(primitive(c) <= set.union(*(primitive(p) for p in pairs if p not in q)) for c in q) for q in blocks):
            return {'seed':seed,'common':common,'query':blocks,'roles':['fit','fit','fit','dev','audit']}
    raise ValueError('No coverage-preserving composition partition found')


def pair_ids(labels, pairs):
    return [labels['objs'].index(o)*len(labels['attrs'])+labels['attrs'].index(a) for a,o in pairs]
