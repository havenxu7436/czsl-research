"""CPU checks for sampling reproducibility, split isolation and input preservation."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

from PIL import Image

import export_train_review as exporter


def main():
    workspace = Path(__file__).resolve().parent.parent
    fixture = workspace / 'tmp' / ('ut_train_review_check_' + uuid.uuid4().hex)
    assert workspace.resolve() in fixture.resolve().parents
    data, knowledge = fixture / 'data', fixture / 'knowledge'
    (data / 'images').mkdir(parents=True)
    (data / 'compositional-split-natural').mkdir()
    knowledge.mkdir()
    attrs = {'A' + str(i): ['attr ' + str(i), 'surface ' + str(i), 'finish ' + str(i)] for i in range(16)}
    objs = {'O' + str(i): ['object ' + str(i), 'upper ' + str(i), 'sole ' + str(i)] for i in range(12)}
    (knowledge / 'ae_dict.json').write_text(json.dumps(attrs), encoding='utf-8')
    (knowledge / 'oe_dict.json').write_text(json.dumps(objs), encoding='utf-8')
    metadata = []
    pairs = []
    for a in attrs:
        for o in objs:
            name = a + '_' + o + '.jpg'
            Image.new('RGB', (80, 40), 'white').save(data / 'images' / name)
            metadata.append(dict(image=name, attr=a, obj=o, set='train'))
            pairs.append(a + ' ' + o)
    # These image files deliberately do not exist. Export must never inspect them.
    metadata += [dict(image='missing_val.jpg', attr='A0', obj='O0', set='val'),
                 dict(image='missing_test.jpg', attr='A0', obj='O0', set='test')]
    import torch
    torch.save(metadata, data / 'metadata_compositional-split-natural.t7')
    (data / 'compositional-split-natural/train_pairs.txt').write_text('\n'.join(pairs), encoding='utf-8')
    inputs = {p: exporter.sha(p) for p in [data / 'metadata_compositional-split-natural.t7',
                                         data / 'compositional-split-natural/train_pairs.txt',
                                         knowledge / 'ae_dict.json', knowledge / 'oe_dict.json']}
    def args(name):
        return SimpleNamespace(data=str(data), knowledge=str(knowledge), out=str(fixture / name),
                               split='compositional-split-natural', seed=20261006, max_per_class=4)
    with contextlib.redirect_stdout(io.StringIO()):
        first = exporter.export(args('first'))
        second = exporter.export(args('second'))
    identities = lambda groups: [[r['image'] for r in g['selected']] for g in groups]
    assert identities(first) == identities(second)
    assert len(first) == 28 and sum(len(g['descriptions']) for g in first) == 84
    assert all(r['split'] == 'train' for g in first for r in g['selected'])
    assert all(not r['image'].startswith('missing_') for g in first for r in g['selected'])
    assert all(len(g['selected']) == 4 for g in first)
    assert all(len({r['obj' if g['kind'] == 'a' else 'attr'] for r in g['selected']}) == 4 for g in first)
    assert all(exporter.sha(p) == digest for p, digest in inputs.items())
    try:
        exporter.export(args('first'))
    except FileExistsError:
        pass
    else:
        raise AssertionError('Existing review output was overwritten')
    invalid = args('missing_train_image')
    (data / 'images' / first[0]['selected'][0]['image']).rename(data / 'images' / 'renamed_train_image.jpg')
    try:
        exporter.export(invalid)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Missing TRAIN image was silently ignored')
    assert not Path(invalid.out).exists()
    try:
        exporter.resolve_image(data / 'images', '../metadata_compositional-split-natural.t7')
    except ValueError:
        pass
    else:
        raise AssertionError('Path outside images/ was accepted')
    print('EXPORT CHECKS PASSED: deterministic selection, composition coverage, TRAIN isolation,')
    print('84-sentence preservation, input hashes, overwrite rejection, missing-image rejection, path bounds')
    print('SYNTHETIC FIXTURE OUTPUT:', fixture)


if __name__ == '__main__':
    main()
