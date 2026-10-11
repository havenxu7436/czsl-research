"""Read-only diagnosis of every fixed selected UT TRAIN image. Python 3.8."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat

from PIL import Image


def load_exporter(path):
    spec = importlib.util.spec_from_file_location('ut_existing_exporter_v3', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_image(root, relative, exporter):
    result = dict(relative=relative)
    pure = PurePosixPath(relative)
    if pure.is_absolute() or '..' in pure.parts or ':' in relative:
        return dict(result, status='invalid_relative_path')
    target = root.joinpath(*pure.parts).resolve()
    result['resolved'] = str(target)
    if root not in target.parents:
        return dict(result, status='outside_configured_root', file_exists=target.is_file())
    try:
        info = target.stat()
        if not stat.S_ISREG(info.st_mode):
            return dict(result, status='not_regular_file')
        with Image.open(target) as image:
            image.load()
            result['size'] = list(image.size)
            result['format'] = image.format
        return dict(result, status='ok', bytes=info.st_size, sha256=exporter.sha(target))
    except OSError as error:
        status = 'missing' if error.errno in (2, 20) else 'unreadable'
        return dict(result, status=status, error_type=type(error).__name__,
                    errno=error.errno, detail=str(error))
    except Exception as error:
        return dict(result, status='invalid_image', error_type=type(error).__name__, detail=str(error))


def audit(args):
    data, knowledge, raw_root = [Path(value).resolve() for value in
                                 (args.data, args.knowledge, args.image_root)]
    processed_root = (data / 'images').resolve()
    output = Path(args.out).resolve()
    for protected in (data, knowledge, raw_root):
        if output == protected or protected in output.parents:
            raise ValueError('Report must be outside dataset, knowledge and image tree')
    if output.exists():
        raise FileExistsError('Report exists; choose a new filename: ' + str(output))
    if not raw_root.is_dir():
        raise FileNotFoundError('Original image root absent: ' + str(raw_root))
    exporter = load_exporter(Path(args.exporter).resolve())
    metadata_path = data / 'metadata_compositional-split-natural.t7'
    pairs_path = data / 'compositional-split-natural/train_pairs.txt'
    attrs_path, objs_path = knowledge / 'ae_dict.json', knowledge / 'oe_dict.json'
    fingerprints = {str(path): exporter.sha(path) for path in
                    (metadata_path, pairs_path, attrs_path, objs_path)}
    metadata = exporter.load_metadata(metadata_path)
    pairs = {tuple(line.split()) for line in pairs_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()}
    records = exporter.train_records(metadata, pairs)
    attrs, objs = exporter.read_json(attrs_path), exporter.read_json(objs_path)
    selected, panels = {}, 0
    for kind, table in (('a', attrs), ('o', objs)):
        for label in table:
            for row in exporter.balanced_sample(records, kind, label, args.max_per_class, args.seed):
                panels += 1
                if row['image'] not in selected:
                    selected[row['image']] = dict(row, boards=[])
                selected[row['image']]['boards'].append(kind + ':' + label)
    rows = []
    for index, row in enumerate(selected.values(), 1):
        if not row.get('_image'):
            original = dict(status='missing_metadata_field')
        elif PurePosixPath(row['_image']).name != PurePosixPath(row['image']).name:
            original = dict(status='basename_mismatch')
        else:
            original = inspect_image(raw_root, row['_image'], exporter)
        processed = inspect_image(processed_root, row['image'], exporter)
        rows.append(dict(row, original=original, processed=processed))
        if index % 100 == 0:
            print('CHECKED SELECTED TRAIN IMAGES:', index, flush=True)
    missing_names = {PurePosixPath(row['image']).name for row in rows if row['original']['status'] != 'ok'}
    candidates, walk_errors, skipped_links = defaultdict(list), [], []
    files_scanned = 0
    def walk_error(error):
        walk_errors.append(dict(filename=error.filename, errno=error.errno, detail=str(error)))
    if missing_names:
        # Directory inventory only; candidate and other unselected image pixels are not opened.
        # Directory links are not followed, avoiding accidental scans outside this image tree.
        for current, directories, names in os.walk(raw_root, followlinks=False, onerror=walk_error):
            skipped_links.extend(str(Path(current) / name) for name in directories if (Path(current) / name).is_symlink())
            files_scanned += len(names)
            for name in names:
                if name in missing_names:
                    path = Path(current) / name
                    candidates[name].append(str(path.relative_to(raw_root)).replace('\\', '/'))
    for row in rows:
        if row['original']['status'] != 'ok':
            alternatives = sorted(candidates[PurePosixPath(row['image']).name])
            row['same_basename_candidates'] = alternatives
            original_parts = PurePosixPath(row.get('_image', '')).parts
            row['candidates_in_same_object_tree'] = [path for path in alternatives
                    if PurePosixPath(path).parts[:2] == original_parts[:2]]
    failures = [row for row in rows if row['original']['status'] != 'ok']
    counts = dict(Counter(row['original']['status'] for row in rows))
    report = dict(version='ut_train_image_audit_20261006', seed=args.seed,
                  max_per_class=args.max_per_class, train_records=len(records),
                  panels=panels, unique_selected=len(rows), original_root=str(raw_root),
                  processed_root=str(processed_root), original_status_counts=counts,
                  processed_status_counts=dict(Counter(row['processed']['status'] for row in rows)),
                  original_failures=len(failures),
                  failures_with_readable_processed=sum(row['processed']['status'] == 'ok' for row in failures),
                  failures_with_candidates=sum(bool(row.get('same_basename_candidates')) for row in failures),
                  candidate_inventory_files=files_scanned, inventory_errors=walk_errors,
                  skipped_directory_links=skipped_links,
                  audit_script_sha256=exporter.sha(Path(__file__)),
                  source_sha256=fingerprints, source_unchanged=all(exporter.sha(path) == digest for path, digest in fingerprints.items()),
                  sampling_identity_field='image', candidate_images_opened=False,
                  validation_test_images_opened=False, samples_replaced=False, rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('AUDIT COMPLETE')
    print('TRAIN RECORDS:', len(records))
    print('SELECTED PANELS:', panels)
    print('UNIQUE SELECTED TRAIN IMAGES:', len(rows))
    print('ORIGINAL STATUS COUNTS:', json.dumps(counts, sort_keys=True))
    print('ORIGINAL FAILURES:', len(failures))
    print('FAILURES WITH READABLE CPF IMAGE:', report['failures_with_readable_processed'])
    print('FAILURES WITH SAME-NAME CANDIDATES:', report['failures_with_candidates'])
    print('INVENTORY ERRORS:', len(walk_errors))
    print('SKIPPED DIRECTORY LINKS:', len(skipped_links))
    print('SOURCE UNCHANGED:', report['source_unchanged'])
    print('REPORT:', output)
    for row in failures[:8]:
        print('FAILED:', json.dumps(dict(image=row['image'], original=row.get('_image'),
                status=row['original']['status'], cpf_status=row['processed']['status'],
                candidates=row.get('same_basename_candidates', [])), ensure_ascii=True))
    print('NO IMAGES COPIED; NO SAMPLE REPLACEMENT; NO TRAINING')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='/data/xhw/czsl_repro/data/ut-zap50k')
    parser.add_argument('--knowledge', default='/data/xhw/czsl_repro/knowledge/kcc0601_v1/trusted_v1/ut-zap50k')
    parser.add_argument('--image-root', default='/data/xhw/czsl_repro/data/ut-zap50k/_images_extract/ut-zap50k-images-square')
    parser.add_argument('--exporter', default='/tmp/ut_train_review_20261006_v3/export_train_review.py')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--max-per-class', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.max_per_class <= 100:
        parser.error('--max-per-class must be between 1 and 100')
    audit(args)


if __name__ == '__main__':
    main()
