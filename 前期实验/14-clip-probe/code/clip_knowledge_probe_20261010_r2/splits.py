"""Audit generalized VAL: seen pair overlap is valid, image overlap is not."""
from common import require


def audit_pairs(train_pairs, val_pairs, labels, expected):
    for split, pairs in (('train', train_pairs), ('val', val_pairs)):
        require(all(len(p) == 2 and p[0] in labels['attrs'] and p[1] in labels['objs']
                    for p in pairs), 'Unknown primitive in ' + split + ' split')
    seen = train_pairs & val_pairs
    unseen = val_pairs - train_pairs
    counts = {'train_pairs': len(train_pairs), 'val_pairs': len(val_pairs),
              'val_seen_pairs': len(seen), 'val_unseen_pairs': len(unseen)}
    for key, value in counts.items():
        require(value == expected[key], 'Split count mismatch: ' + key)
    return dict(counts, val_seen_pair_labels=[list(p) for p in sorted(seen)],
                val_unseen_pair_labels=[list(p) for p in sorted(unseen)])


def audit_metadata(metadata, ids, labels, train_pairs, val_pairs, expected):
    report = audit_pairs(train_pairs, val_pairs, labels, expected)
    selected = {'train': [], 'val': []}
    for row in metadata:
        phase = row.get('set')
        # Never inspect TEST paths/labels. The legacy metadata file itself
        # contains all splits, but only TRAIN/VAL records are used here.
        if phase not in selected:
            continue
        require(row['attr'] in labels['attrs'] and row['obj'] in labels['objs'],
                'Unexpected TRAIN/VAL primitive')
        pair = (row['attr'], row['obj'])
        allowed = train_pairs if phase == 'train' else val_pairs
        require(pair in allowed, 'Metadata composition outside ' + phase + '_pairs.txt')
        selected[phase].append(row)
    identity_sets = {}
    for phase in ('train', 'val'):
        metadata_ids = [r['image'] for r in selected[phase]]
        require(metadata_ids == ids[phase], 'Metadata/StageA image order differs: ' + phase)
        require(len(metadata_ids) == len(set(metadata_ids)), 'Duplicate ' + phase + ' image identity')
        require(len(metadata_ids) == expected[phase + '_images'], 'Image count mismatch: ' + phase)
        covered = {(r['attr'], r['obj']) for r in selected[phase]}
        require(covered == (train_pairs if phase == 'train' else val_pairs),
                'Metadata split pair coverage differs: ' + phase)
        identity_sets[phase] = set(metadata_ids)
    overlap = len(identity_sets['train'] & identity_sets['val'])
    require(overlap == 0, 'TRAIN/VAL image overlap')
    seen_mask = [(r['attr'], r['obj']) in train_pairs for r in selected['val']]
    seen_images = sum(seen_mask)
    unseen_images = len(seen_mask) - seen_images
    require(seen_images == expected['val_seen_images'], 'VAL seen image count mismatch')
    require(unseen_images == expected['val_unseen_images'], 'VAL unseen image count mismatch')
    report.update(train_images=len(ids['train']), val_images=len(ids['val']),
                  val_seen_images=seen_images, val_unseen_images=unseen_images,
                  train_val_image_id_overlap=overlap, images_decoded=0,
                  test_images_decoded=0, validation_pair_overlap_is_expected=True)
    return report
