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
    raw_root = fixture / 'original_images'
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
            original_name = 'Objects/' + o + '/Brand/' + name
            original = raw_root / original_name
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes((data / 'images' / name).read_bytes())
            metadata.append(dict(image=name, _image=original_name, attr=a, obj=o, set='train'))
            pairs.append(a + ' ' + o)
    # These image files deliberately do not exist. Export must never inspect them.
    metadata += [dict(image='missing_val.jpg', _image='missing_raw_val.jpg', attr='A0', obj='O0', set='val'),
                 dict(image='missing_test.jpg', _image='missing_raw_test.jpg', attr='A0', obj='O0', set='test')]
    import torch
    torch.save(metadata, data / 'metadata_compositional-split-natural.t7')
    (data / 'compositional-split-natural/train_pairs.txt').write_text('\n'.join(pairs), encoding='utf-8')
    inputs = {p: exporter.sha(p) for p in [data / 'metadata_compositional-split-natural.t7',
                                         data / 'compositional-split-natural/train_pairs.txt',
                                         knowledge / 'ae_dict.json', knowledge / 'oe_dict.json']}
    def args(name):
        return SimpleNamespace(data=str(data), knowledge=str(knowledge), out=str(fixture / name),
                               split='compositional-split-natural', seed=20261006, max_per_class=4,
                               image_root=None, image_field='image', check_images_only=False)
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
    # Raw image paths must preserve the CPF IDs and all selected identities.
    original_args = args('original_path_field')
    original_args.image_root = str(raw_root)
    original_args.image_field = '_image'
    with contextlib.redirect_stdout(io.StringIO()):
        original_groups = exporter.export(original_args)
    assert identities(first) == identities(original_groups)
    assert all(row['source_relative_path'] == row['_image'] and
               (raw_root / row['source_relative_path']).is_file()
               for group in original_groups for row in group['selected'])
    original_manifest = json.loads((Path(original_args.out) / 'manifest.json').read_text(encoding='utf-8'))
    assert original_manifest['image_path_field'] == '_image'
    assert original_manifest['sampling_identity_field'] == 'image'
    selected_id = first[0]['selected'][0]['image']
    missing_mapping = [dict(row) for row in metadata]
    for row in missing_mapping:
        if row['image'] == selected_id:
            row.pop('_image')
    missing_args = args('missing_original_mapping')
    missing_args.image_root = str(raw_root)
    missing_args.image_field = '_image'
    try:
        exporter.export(missing_args, metadata_override=missing_mapping)
    except ValueError as error:
        assert 'lacks path field _image' in str(error)
    else:
        raise AssertionError('Missing original field was silently replaced')
    assert not Path(missing_args.out).exists()
    mismatched_mapping = [dict(row) for row in metadata]
    for row in mismatched_mapping:
        if row['image'] == selected_id:
            row['_image'] = 'Objects/O0/Brand/other_identity.jpg'
    mismatch_args = args('wrong_original_identity')
    mismatch_args.image_root = str(raw_root)
    mismatch_args.image_field = '_image'
    try:
        exporter.export(mismatch_args, metadata_override=mismatched_mapping)
    except ValueError as error:
        assert 'basenames differ' in str(error)
    else:
        raise AssertionError('A different original image identity was accepted')
    assert not Path(mismatch_args.out).exists()
    selected_original = raw_root / original_groups[0]['selected'][0]['source_relative_path']
    original_bytes = selected_original.read_bytes()
    renamed_original = selected_original.with_name('temporarily_absent_' + selected_original.name)
    selected_original.rename(renamed_original)
    no_original = args('missing_original_image')
    no_original.image_root = str(raw_root)
    no_original.image_field = '_image'
    try:
        exporter.export(no_original)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Missing original TRAIN image was silently replaced')
    assert not Path(no_original.out).exists()
    renamed_original.rename(selected_original)
    assert selected_original.read_bytes() == original_bytes
    alternate_root = fixture / 'alternate_images'
    (data / 'images').rename(alternate_root)
    explicit = args('explicit_root')
    explicit.image_root = str(alternate_root)
    with contextlib.redirect_stdout(io.StringIO()):
        third = exporter.export(explicit)
    assert identities(first) == identities(third)
    assert json.loads((Path(explicit.out) / 'manifest.json').read_text(encoding='utf-8'))['image_root'] == str(alternate_root.resolve())
    preflight = args('preflight_only')
    preflight.image_root = str(alternate_root)
    preflight.check_images_only = True
    with contextlib.redirect_stdout(io.StringIO()):
        exporter.export(preflight)
    assert not Path(preflight.out).exists() and not Path(preflight.out + '.zip').exists()
    alternate_root.rename(data / 'images')
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
    # Absolute names and links resolving outside the configured tree stay rejected.
    try:
        exporter.resolve_image(data / 'images', '/outside.jpg')
    except ValueError:
        pass
    else:
        raise AssertionError('Absolute image path was accepted')
    external = fixture / 'outside.jpg'
    Image.new('RGB', (80, 40), 'white').save(external)
    link = data / 'images' / 'external_link.jpg'
    import os
    try:
        os.symlink(str(external), str(link))
    except OSError:
        print('SYMLINK CHECK SKIPPED: host does not permit symlink creation')
    else:
        try:
            exporter.resolve_image(data / 'images', 'external_link.jpg')
        except ValueError as error:
            assert 'outside configured image root' in str(error) and 'IS_FILE=True' in str(error)
        else:
            raise AssertionError('External image link was accepted')
    print('EXPORT CHECKS PASSED: deterministic selection, composition coverage, TRAIN isolation,')
    print('84-sentence preservation, input hashes, overwrite rejection, missing-image rejection, path bounds')
    print('EXPLICIT ROOT AND READ-ONLY PREFLIGHT CHECKS PASSED')
    print('ORIGINAL _image MAPPING, UNCHANGED SELECTION, MISSING MAPPING AND IDENTITY CHECKS PASSED')
    print('SYNTHETIC FIXTURE OUTPUT:', fixture)


if __name__ == '__main__':
    main()
