"""Export fixed, composition-balanced TRAIN image contact sheets for UT review.

CPU only. Does not import the training project, load validation/test images,
modify knowledge, create a semantic gate, or start training.
Python 3.8 compatible; uses the existing torch and Pillow installations.
"""
import argparse
import hashlib
import json
import random
import re
import textwrap
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont, ImageOps


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n', encoding='utf-8')


def load_metadata(path):
    import torch
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        # torch 1.13 has no weights_only keyword.
        return torch.load(path, map_location='cpu')


def train_records(metadata, train_pairs):
    records = []
    identities = set()
    image_labels = {}
    original_paths = {}
    for item in metadata:
        if item.get('set') != 'train':
            continue
        pair = (item['attr'], item['obj'])
        if pair not in train_pairs or item['attr'] == 'NA':
            continue
        image = str(item['image']).replace('\\', '/')
        original = item.get('_image')
        if original is not None:
            if not isinstance(original, str) or not original.strip():
                raise ValueError('Expected nonempty original _image path for TRAIN image: ' + image)
            original = original.replace('\\', '/')
        identity = (image,) + pair
        if image in image_labels and image_labels[image] != pair:
            raise ValueError('Conflicting TRAIN labels for image: ' + image)
        image_labels[image] = pair
        if image in original_paths and original_paths[image] != original:
            raise ValueError('Conflicting original paths for TRAIN image: ' + image)
        original_paths[image] = original
        if identity not in identities:
            identities.add(identity)
            record = dict(image=image, attr=pair[0], obj=pair[1], split='train')
            if original is not None:
                record['_image'] = original
            records.append(record)
    return sorted(records, key=lambda row: (row['image'], row['attr'], row['obj']))


def balanced_sample(records, kind, label, maximum, seed):
    field, other = ('attr', 'obj') if kind == 'a' else ('obj', 'attr')
    groups = defaultdict(list)
    for row in records:
        if row[field] == label:
            groups[row[other]].append(row)
    if not groups:
        raise ValueError('No TRAIN images for class: ' + label)
    digest = hashlib.sha256((str(seed) + ':' + kind + ':' + label).encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    selected = []
    while len(selected) < maximum:
        progress = False
        for key in keys:
            if groups[key] and len(selected) < maximum:
                selected.append(dict(groups[key].pop()))
                progress = True
        if not progress:
            break
    return selected


def resolve_image(image_root, relative):
    pure = PurePosixPath(relative)
    if pure.is_absolute() or '..' in pure.parts or ':' in relative:
        raise ValueError('Expected image path relative to configured image root: ' + relative)
    root = Path(image_root).resolve()
    target = root.joinpath(*pure.parts).resolve()
    if root not in target.parents:
        raise ValueError('Selected TRAIN image resolves outside configured image root. '
                         'Check directory symlinks and set --image-root to the actual image tree. '
                         'IMAGE_ROOT=' + str(root) + ' TARGET=' + str(target) +
                         ' IS_FILE=' + str(target.is_file()))
    if not target.is_file():
        raise FileNotFoundError('Selected TRAIN image absent: ' + str(target))
    return target


def font(size):
    for candidate in ('DejaVuSans.ttf', 'arial.ttf',
                      '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_board(group, out, images_root):
    # Preserve the whole product silhouette and aspect ratio; no crop/augmentation.
    width, panel_height, top, columns = 250, 265, 66, 5
    nrows = (len(group['selected']) + columns - 1) // columns
    board = Image.new('RGB', (columns * width, top + panel_height * nrows), 'white')
    draw = ImageDraw.Draw(board)
    title_font, small_font = font(17), font(12)
    draw.text((10, 6), '{}: {}'.format(group['kind'], group['label']), fill='black', font=title_font)
    draw.text((10, 31), 'TRAIN only | balanced across compositions | qualitative review',
              fill='black', font=small_font)
    resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')
    for i, row in enumerate(group['selected']):
        x, y = (i % columns) * width, top + (i // columns) * panel_height
        image_path = resolve_image(images_root, row['source_relative_path'])
        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            row['source_size'] = list(image.size)
            image.thumbnail((230, 185), resample)
            board.paste(image, (x + (width - image.width) // 2, y + (190 - image.height) // 2))
        row['image_sha256'] = sha(image_path)
        panel_lines = [str(i + 1) + '. ' + row['attr']] + textwrap.wrap(row['obj'], 32)
        draw.multiline_text((x + 8, y + 193), '\n'.join(panel_lines[:4]),
                            fill='black', font=small_font, spacing=2)
        draw.rectangle((x, y, x + width - 1, y + panel_height - 1), outline='#dddddd')
    board.save(out / group['board'], 'JPEG', quality=92)


def export(args, metadata_override=None):
    data = Path(args.data).resolve()
    knowledge = Path(args.knowledge).resolve()
    images_root = Path(args.image_root).resolve() if args.image_root else (data / 'images').resolve()
    out = Path(args.out).resolve()
    archive = Path(str(out) + '.zip')
    # Never write inside dataset/source knowledge; archive cannot overwrite either.
    for protected in (data, knowledge, images_root):
        if out == protected or protected in out.parents or out in protected.parents:
            raise ValueError('Review output must be separate from data and knowledge')
    if out.exists() or archive.exists():
        raise FileExistsError('Output exists; choose a new version: ' + str(out))
    if not images_root.is_dir():
        raise FileNotFoundError('Configured image root absent: ' + str(images_root))
    metadata_path = data / ('metadata_' + args.split + '.t7')
    pairs_path = data / args.split / 'train_pairs.txt'
    ae_path, oe_path = knowledge / 'ae_dict.json', knowledge / 'oe_dict.json'
    attrs, objs = read_json(ae_path), read_json(oe_path)
    aliases = read_json(args.path_map) if args.path_map else {}
    if not isinstance(aliases, dict):
        raise ValueError('Path map must be a JSON object')
    for old, new in aliases.items():
        if not isinstance(old, str) or not isinstance(new, str) or not old or not new:
            raise ValueError('Expected nonempty path strings in path map')
        original, mapped = PurePosixPath(old), PurePosixPath(new)
        if original.is_absolute() or mapped.is_absolute() or '..' in original.parts or '..' in mapped.parts or ':' in old or ':' in new:
            raise ValueError('Path map must stay inside the configured image tree')
        if original.name != mapped.name or original.parts[:2] != mapped.parts[:2]:
            raise ValueError('Path map must preserve image basename and object directory')
    audit = read_json(args.audit_report) if args.audit_report else None
    if aliases and audit is None:
        raise ValueError('A path map requires the full fixed-sample audit report')
    if len(attrs) != 16 or len(objs) != 12:
        raise ValueError('Expected UT label counts: 16 attributes, 12 objects')
    if any(not isinstance(v, list) or len(v) != 3 or
           any(not isinstance(t, str) or not t.strip() for t in v)
           for table in (attrs, objs) for v in table.values()):
        raise ValueError('Expected three nonempty descriptions per class')
    pairs = set()
    for line in pairs_path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        pair = tuple(line.split())
        if len(pair) != 2 or pair[0] not in attrs or pair[1] not in objs:
            raise ValueError('TRAIN pair absent from frozen class dictionaries')
        pairs.add(pair)
    metadata = metadata_override if metadata_override is not None else load_metadata(metadata_path)
    records = train_records(metadata, pairs)
    if not records:
        raise ValueError('No usable TRAIN images found')
    inputs = {str(p): sha(p) for p in (metadata_path, pairs_path, ae_path, oe_path)}
    if audit is not None:
        if not audit.get('source_unchanged') or audit.get('seed') != args.seed or audit.get('max_per_class') != args.max_per_class:
            raise ValueError('Audit source status or sampling protocol does not match')
        if Path(audit['original_root']).resolve() != images_root:
            raise ValueError('Audit image root differs from configured root')
        for path, digest in inputs.items():
            if audit['source_sha256'].get(path) != digest:
                raise ValueError('Dataset or knowledge changed since audit: ' + path)
        for old, new in aliases.items():
            matches = [row for row in audit['rows'] if row.get('_image') == old]
            if len(matches) != 1 or matches[0].get('same_basename_candidates') != [new]:
                raise ValueError('Mapped file was not the unique audited candidate: ' + old)
        inputs[str(Path(args.audit_report).resolve())] = sha(args.audit_report)
    if args.path_map:
        inputs[str(Path(args.path_map).resolve())] = sha(args.path_map)
    valid_sources = {row.get(args.image_field) for row in records}
    if any(path not in valid_sources for path in aliases):
        raise ValueError('Path map contains a path outside usable TRAIN records')
    groups = []
    checked_sources, applied_aliases = set(), set()
    for kind, table in (('a', attrs), ('o', objs)):
        for i, (label, descriptions) in enumerate(table.items()):
            slug = re.sub('[^A-Za-z0-9]+', '_', label).strip('_').lower()
            selected = balanced_sample(records, kind, label, args.max_per_class, args.seed)
            for row in selected:
                relative = row.get(args.image_field)
                if not isinstance(relative, str) or not relative.strip():
                    raise ValueError('Selected TRAIN record lacks path field ' + args.image_field + ': ' + row['image'])
                if PurePosixPath(relative).name != PurePosixPath(row['image']).name:
                    raise ValueError('Original and metadata TRAIN basenames differ: ' + row['image'] + ' | ' + relative)
                mapped = aliases.get(relative, relative)
                row['metadata_relative_path'] = relative
                row['source_relative_path'] = mapped
                row['path_alias_applied'] = mapped != relative
                if mapped != relative:
                    applied_aliases.add(relative)
                image_path = resolve_image(images_root, mapped)
                if image_path not in checked_sources:
                    with Image.open(image_path) as image:
                        image.load()
                    checked_sources.add(image_path)
            groups.append(dict(kind=kind, label=label, descriptions=descriptions,
                               board='boards/{}_{:02d}_{}.jpg'.format(kind, i, slug), selected=selected))
    if audit is not None:
        expected = {row['image']: (row['attr'], row['obj'], row.get('_image')) for row in audit['rows']}
        actual = {row['image']: (row['attr'], row['obj'], row.get('_image')) for group in groups for row in group['selected']}
        if expected != actual or audit['panels'] != sum(len(group['selected']) for group in groups):
            raise ValueError('Selected TRAIN identities differ from the audit')
        for row in audit['rows']:
            if row['original']['status'] == 'ok':
                if sha(resolve_image(images_root, row['_image'])) != row['original']['sha256']:
                    raise ValueError('Previously readable TRAIN image changed since audit: ' + row['image'])
    print('IMAGE ROOT:', images_root, flush=True)
    print('IMAGE PATH FIELD:', args.image_field, flush=True)
    print('EXPLICIT PATH ALIASES APPLIED:', len(applied_aliases), flush=True)
    print('READABLE UNIQUE TRAIN IMAGES:', len(checked_sources), flush=True)
    if args.check_images_only:
        print('SELECTED TRAIN IMAGE CHECK PASSED:', sum(len(g['selected']) for g in groups))
        print('NO OUTPUT CREATED; NO TRAINING')
        return groups
    out.mkdir(parents=True)
    (out / 'boards').mkdir()
    for group in groups:
        render_board(group, out, images_root)
        print('BOARD COMPLETE:', group['kind'], group['label'], len(group['selected']), flush=True)
    write_json(out / 'selected_train_images.json', groups)
    write_json(out / 'path_aliases.json', dict(aliases=aliases, applied=sorted(applied_aliases),
               purpose='Exact audited directory aliases only; TRAIN sample identities unchanged'))
    manifest = dict(version='ut_train_review_paths_20261006', seed=args.seed,
                    image_root=str(images_root),
                    image_path_field=args.image_field,
                    sampling_identity_field='image',
                    source_path_aliases=aliases, applied_path_aliases=sorted(applied_aliases),
                    audit_report_sha256=sha(args.audit_report) if args.audit_report else None,
                    max_per_class=args.max_per_class, training_images_total=len(records),
                    train_pairs=len(pairs), attributes=len(attrs), objects=len(objs),
                    panels=sum(len(g['selected']) for g in groups),
                    unique_selected_images=len({r['image'] for g in groups for r in g['selected']}),
                    sample_policy='round-robin over other primitive; deterministic class-local RNG',
                    sample_use='qualitative coverage check, not a prevalence estimate or performance metric',
                    validation_test_images_loaded=False, labels_rewritten=False,
                    script_sha256=sha(Path(__file__)), inputs_sha256=inputs,
                    outputs_sha256={str(p.relative_to(out)).replace('\\', '/'): sha(p)
                                    for p in sorted(out.rglob('*')) if p.is_file()})
    write_json(out / 'manifest.json', manifest)
    review = ['# TRAIN image review template', '',
              'Original sentences are preserved. No approval is inferred.',
              'Read selected_train_images.json to match panel numbers to image paths.',
              'image is the unchanged CPF sample ID; source_relative_path is the file path opened.',
              'Balanced sampling is for qualitative coverage; do not estimate population prevalence.',
              'Repeat appearances across boards are not independent observations.', '',
              'For each sentence record: label meaning / visible cue / coverage / shared cue.',
              'A source, function, or tactile statement may be true without being directly visible.', '']
    for group in groups:
        review += ['## {}: {}'.format(group['kind'], group['label']), '',
                   'Board: ' + group['board'], '',
                   '| Sentence | Label meaning | Visible in sampled images | Coverage notes | Decision |',
                   '| --- | --- | --- | --- | --- |']
        review += ['| {} | pending | pending | pending | pending |'.format(t)
                   for t in group['descriptions']]
        review += ['']
    (out / 'review_template.md').write_text('\n'.join(review), encoding='utf-8')
    if any(sha(p) != digest for p, digest in inputs.items()):
        raise ValueError('Input changed during export; preserve output and investigate')
    # Store the self-contained review folder, with no original images/checkpoints.
    with ZipFile(archive, 'x', ZIP_DEFLATED) as bundle:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                bundle.write(path, out.name + '/' + str(path.relative_to(out)))
    print('EXPORT COMPLETE:', out)
    print('TRAIN IMAGES:', len(records))
    print('REVIEW PANELS:', manifest['panels'])
    print('UNIQUE TRAIN IMAGES:', manifest['unique_selected_images'])
    print('ARCHIVE:', archive)
    print('ARCHIVE_SHA256:', sha(archive))
    print('SOURCE UNCHANGED; NO GPU; NO TRAINING; NO GATE CHANGE')
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--knowledge', required=True)
    parser.add_argument('--image-root', help='Actual tree containing Boots/, Shoes/, Sandals/, etc.; default: DATA/images')
    parser.add_argument('--image-field', choices=('image', '_image'), default='image',
                        help='Metadata path field to open. Use _image for the original UT image tree; sampling still uses image.')
    parser.add_argument('--check-images-only', action='store_true', help='Check every selected TRAIN image without creating output')
    parser.add_argument('--path-map', help='Explicit metadata-relative to real-relative path aliases')
    parser.add_argument('--audit-report', help='Full fixed-sample audit JSON verifying identities and unique candidates')
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', default='compositional-split-natural')
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--max-per-class', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.max_per_class <= 100:
        parser.error('--max-per-class must be between 1 and 100')
    export(args)


if __name__ == '__main__':
    main()
