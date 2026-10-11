"""Independent NumPy replay. Never executes source from either result ZIP."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RESULT = BASE / 'output/train_knowledge_audit_results_20261010'
ZIP = RESULT / 'czsl_train_knowledge_audit_20261010.zip'
OLD = BASE / 'output/clip_knowledge_probe_results_20261010_r2/czsl_clip_knowledge_probe_20261010_r2.zip'
RELEASE = BASE / 'output/train_knowledge_audit_release_20261010.zip'
VARIANTS = ('T_name', 'T_K', 'T_perm1', 'T_perm2', 'T_AE', 'T_AE_perm1', 'T_AE_perm2')


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    count = 0
    def check(value, title):
        nonlocal count
        count += 1
        if not value:
            raise AssertionError(title)
    def compare(ours, stored, title):
        if isinstance(ours, dict):
            for key, value in ours.items():
                compare(value, stored[key], title + '/' + key)
        elif isinstance(ours, list):
            check(len(ours) == len(stored), title + '/len')
            for index, (a, b) in enumerate(zip(ours, stored)):
                compare(a, b, title + '/' + str(index))
        elif isinstance(ours, float):
            check(abs(ours - stored) < 1e-5, title + '/float')
        else:
            check(ours == stored, title)
    check(digest(ZIP) == Path(str(ZIP) + '.sha256').read_text().split()[0], 'Downloaded SHA')
    check(digest(RELEASE) == 'f1f44b0cbabf965776539a30b430466fa204afe4012348da351c8e85a7cd078b', 'Release SHA')
    check(digest(OLD) == '7fb5d14a81e83b8b0c62fba93ec441bc3f056e0c04cf2749246d31acc6a4bb9f', 'Old r2 identity')
    with zipfile.ZipFile(ZIP) as z, zipfile.ZipFile(OLD) as old, zipfile.ZipFile(RELEASE) as release:
        def js(name):
            return json.loads(z.read(name))
        def ar(archive, name):
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as a:
                return {k: a[k] for k in a.files}
        manifest = js('export_manifest.json')['files_sha256']
        check(z.testzip() is None, 'CRC')
        check(len(z.namelist()) == len(set(z.namelist())), 'Duplicate members')
        check(set(z.namelist()) == set(manifest) | {'export_manifest.json'}, 'Member set')
        for name, expected in manifest.items():
            blob = z.read(name)
            check(hashlib.sha256(blob).hexdigest() == expected, 'Member SHA:' + name)
            extracted = RESULT / 'extracted' / name
            check(extracted.is_file() and digest(extracted) == expected, 'Extracted file differs:' + name)
        frozen = json.loads(release.read('train_knowledge_audit_20261010/release_manifest.json'))
        check(js('source/release_manifest.json') == frozen, 'Frozen manifest')
        for name, expected in frozen['files_sha256'].items():
            check(hashlib.sha256(z.read('source/' + name)).hexdigest() == expected, 'Frozen source:' + name)
        protocol = js('source/protocol.json')
        labels = protocol['labels']
        expected_inputs = js('source/train_input_manifest.json')['files_sha256']
        for name, expected in expected_inputs.items():
            check('val_' not in name and '/val/' not in name and 'test_' not in name, 'TRAIN whitelist:' + name)
            check(hashlib.sha256(old.read('result/' + name)).hexdigest() == expected, 'Old TRAIN SHA:' + name)
        completion, reads = js('result/complete.json'), js('result/input_reads.json')
        check(completion['execution_passed'] and completion['optimizer_steps'] == 0 and
              completion['image_encoder_calls'] == 0 and completion['train_images_decoded'] == 96 and
              completion['val_images_decoded'] == completion['test_images_decoded'] == 0, 'Execution scope')
        check(not any('failure_' in n for n in z.namelist()), 'Failure attempts')
        check(set(reads['r2_files_read']) == set(expected_inputs) and
              reads['val_files_read'] == reads['test_files_read'] == [], 'Read log scope')
        identity = js('result/run_identity.json')
        check(identity['protocol_sha256'] == hashlib.sha256(z.read('source/protocol.json')).hexdigest(), 'Protocol identity')
        check(identity['train_input_manifest_sha256'] == hashlib.sha256(z.read('source/train_input_manifest.json')).hexdigest(), 'Input identity')
        records = json.loads(old.read('result/train_records.json'))
        truth = ar(old, 'result/teacher/train_truth.npz')
        a, o = truth['attrs'], truth['objects']
        check(len(a) == 22998 and np.all(truth['seen']), 'TRAIN dimensions')
        check(np.array_equal(a, [r['attr'] for r in records]) and np.array_equal(o, [r['object'] for r in records]), 'Truth/record order')
        pairs = sorted(set(zip(o.tolist(), a.tolist())))
        check(len(pairs) == 83, 'TRAIN pairs')
        chunks = []
        for start in range(0, len(a), 512):
            end = min(start + 512, len(a))
            stem = 'result/features/train/%06d' % start
            receipt = json.loads(old.read(stem + '.json'))
            shard = ar(old, stem + '.npz')
            h = hashlib.sha256('\n'.join(r['image'] + ' ' + r['image_sha256'] for r in records[start:end]).encode()).hexdigest()
            check(receipt['identity_sha256'] == h == str(shard['identity_sha256'].item()), 'Shard identity')
            check(shard['features'].shape == (end - start, 768), 'Shard shape')
            chunks.append(shard['features'])
        x = np.concatenate(chunks)
        ae, oe = js('source/assets/ae_dict.json'), js('source/assets/oe_dict.json')
        # Independently rebuild the fixed text recipe rather than import released prompts.
        prompts = {}
        for v in VARIANTS:
            pk = 'P_perm1' if v.endswith('perm1') else ('P_perm2' if v.endswith('perm2') else 'P_K')
            p = protocol['permutations'][pk]
            strings = []
            for oi in range(12):
                for ai in range(16):
                    base = 'A photo of %s made of %s.' % (protocol['names']['o'][oi], protocol['names']['a'][ai])
                    for aj in range(3):
                        for oj in range(3):
                            t = base
                            if v != 'T_name':
                                t += ' It has %s.' % ae[labels['attrs'][p['a'][ai]]][aj]
                                if not v.startswith('T_AE'):
                                    t += ' Object shape: %s.' % oe[labels['objs'][p['o'][oi]]][oj]
                            strings.append(t)
            prompts[v] = strings
        check(prompts == json.loads(old.read('result/prompts.json')), 'Exact recipe')
        unique = sorted({t for texts in prompts.values() for t in texts})
        text_receipt = js('result/unit_text_receipt.json')
        check(text_receipt['unique_texts'] == unique and len(unique) == 7041, 'Unit text identity')
        check(text_receipt['sha256'] == manifest['result/unit_text_features.npz'], 'Text receipt SHA')
        vectors = ar(z, 'result/unit_text_features.npz')['features']
        check(vectors.shape == (7041, 768) and vectors.dtype == np.float32 and np.isfinite(vectors).all(), 'Unit text shape/dtype')
        check(np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-3), 'Text normalization')
        lookup = {t: i for i, t in enumerate(unique)}
        old_text = ar(old, 'result/text_features.npz')
        scores, units, replay = {}, {}, {}
        for v in VARIANTS:
            table = vectors[[lookup[t] for t in prompts[v]]].reshape(12, 16, 9, 768)
            means = table.mean(2, dtype=np.float32)
            text_difference = float(np.abs(means - old_text[v]).max())
            check(text_difference == 0, 'Exact original mean text replay:' + v)
            saved = ar(old, 'result/teacher/train_' + v + '.npz')
            scores[v] = saved['cosine'][np.arange(len(a)), o]
            check(np.array_equal(scores[v].argmax(1), saved['q'][np.arange(len(a)), o].argmax(1)), 'Saved top1:' + v)
            raw = np.empty((len(a), 16, 9), dtype=np.float32)
            calculated = np.empty((len(a), 16), dtype=np.float32)
            for oi in range(12):
                ix = np.flatnonzero(o == oi)
                raw[ix] = (x[ix] @ table[oi].reshape(144, 768).T).reshape(-1, 16, 9)
                calculated[ix] = x[ix] @ means[oi].T
            if v != 'T_name':
                units[v] = raw
            diff = float(np.abs(calculated - scores[v]).max())
            mismatch = int((calculated.argmax(1) != scores[v].argmax(1)).sum())
            check(diff < 1e-6 and mismatch == 0, 'Mean cosine/top1 replay:' + v)
            replay[v] = {'text_difference': text_difference, 'cosine_difference': diff, 'top1_mismatches': mismatch}
        def calculate(s, mask=None):
            mask = np.ones(len(a), bool) if mask is None else mask
            aa, oo, ss = a[mask], o[mask], s[mask]
            pred = ss.argmax(1); ok = pred == aa
            gp = sorted(set(zip(oo.tolist(), aa.tolist())))
            micro = 100 * float(ok.mean())
            macro = 100 * float(np.mean([ok[(oo == oi) & (aa == ai)].mean() for oi, ai in gp]))
            other = ss.copy(); other[np.arange(len(aa)), aa] = -np.inf
            margin = ss[np.arange(len(aa)), aa] - other.max(1)
            confusion = np.zeros((16, 16), dtype=np.int64)
            for y, h in zip(aa, pred):
                confusion[y, h] += 1
            return {'images': len(aa), 'pairs': len(gp), 'correct': int(ok.sum()), 'micro_pct': micro,
                    'pair_macro_pct': macro, 'mean_gt_margin': float(margin.mean()), 'confusion': confusion.tolist()}, pred
        name_metrics, name_pred = calculate(scores['T_name'])
        def trans(pred, mask=None):
            mask = np.ones(len(a), bool) if mask is None else mask
            good, reference = pred == a[mask], name_pred[mask] == a[mask]
            repair, damage = int((good & ~reference).sum()), int((~good & reference).sum())
            return {'repair': repair, 'damage': damage, 'net': repair - damage}
        pooled = js('result/pooled_train_report.json')
        for v in VARIANTS:
            m, pred = calculate(scores[v]); m['vs_name'] = trans(pred)
            compare(m, pooled['metrics'][v], 'Pooled:' + v)
            for ai, attr in enumerate(labels['attrs']):
                mask = a == ai
                sub, sub_pred = calculate(scores[v], mask)
                mine = {k: sub[k] for k in ('correct', 'micro_pct', 'pair_macro_pct', 'mean_gt_margin')}
                mine.update(predicted_on_all_train=int((pred == ai).sum()), vs_name=trans(sub_pred, mask))
                compare(mine, pooled['per_attribute'][ai]['variants'][v], 'Attribute:' + v + ':' + attr)
        for row, (oi, ai) in zip(pooled['per_pair'], pairs):
            mask = (o == oi) & (a == ai)
            expected = {'attr': labels['attrs'][ai], 'object': labels['objs'][oi], 'images': int(mask.sum()),
                        'correct': {v: int((scores[v][mask].argmax(1) == ai).sum()) for v in VARIANTS}}
            expected['K_net_vs_name'] = expected['correct']['T_K'] - expected['correct']['T_name']
            compare(expected, row, 'Pair')
        sentences = js('result/sentence_train_report.json')
        views, third_scores, rounding = [], {}, []
        def compare_unit(s, m, pred, stored, title):
            # BLAS accumulation differs across NumPy 1.24/Linux and the local
            # runtime. Bound discrete disagreements to near-tied rows; do not
            # silently call those predictions an exact replay.
            top2 = np.sort(s, axis=1)[:, -2:]
            ambiguous = top2[:, 1] - top2[:, 0] <= 1e-6
            allowance = int(ambiguous.sum())
            macro_bound = 100 * sum(int((ambiguous & (o == oi) & (a == ai)).sum()) /
                                    int(((o == oi) & (a == ai)).sum()) for oi, ai in pairs) / len(pairs)
            for key in ('images', 'pairs'):
                check(m[key] == stored[key], title + '/' + key)
            check(abs(m['correct'] - stored['correct']) <= allowance, title + '/correct near-tie bound')
            check(abs(m['micro_pct'] - stored['micro_pct']) <= 100 * allowance / len(a) + 1e-8, title + '/micro bound')
            check(abs(m['pair_macro_pct'] - stored['pair_macro_pct']) <= macro_bound + 1e-8, title + '/macro bound')
            check(abs(m['mean_gt_margin'] - stored['mean_gt_margin']) < 1e-6, title + '/margin')
            cm, recorded = np.array(m['confusion']), np.array(stored['confusion'])
            check(np.array_equal(cm.sum(1), recorded.sum(1)), title + '/confusion class counts')
            check(int(np.abs(cm - recorded).sum()) <= 2 * allowance, title + '/confusion near-tie bound')
            for ai in range(16):
                check(int(np.abs(cm[ai] - recorded[ai]).sum()) <= 2 * int((ambiguous & (a == ai)).sum()),
                      title + '/class near-tie bound')
            for key in ('repair', 'damage', 'net'):
                check(abs(m['vs_name'][key] - stored['vs_name'][key]) <= allowance, title + '/transition bound')
            check(stored['vs_name']['repair'] - stored['vs_name']['damage'] == stored['vs_name']['net'], title + '/net identity')
            if not np.array_equal(cm, recorded):
                rounding.append({'view': title, 'local_correct': m['correct'], 'reported_correct': stored['correct'],
                                 'confusion_l1_difference': int(np.abs(cm - recorded).sum()),
                                 'near_tie_rows_1e_minus6': allowance,
                                 'micro_difference_pp': m['micro_pct'] - stored['micro_pct'],
                                 'macro_difference_pp': m['pair_macro_pct'] - stored['pair_macro_pct'],
                                 'near_tie_examples': [{'index': int(i), 'image': records[i]['image'],
                                                        'score_gap': float(top2[i, 1] - top2[i, 0])}
                                                       for i in np.flatnonzero(ambiguous)[:12]]})
            return ambiguous, macro_bound
        for family, variants, slots in (('OE_AE', ('T_K', 'T_perm1', 'T_perm2'), range(9)),
                                        ('AE_only', ('T_AE', 'T_AE_perm1', 'T_AE_perm2'), (0, 3, 6))):
            check(len(sentences['families'][family]) == len(slots), 'View count')
            for row, slot in zip(sentences['families'][family], slots):
                check(row['slot'] == slot and row['ae_index'] == slot // 3, 'Sentence position')
                measures = {}
                for v in variants:
                    s = units[v][:, :, slot]
                    m, pred = calculate(s); m['vs_name'] = trans(pred)
                    uncertain, bound = compare_unit(s, m, pred, row['variants'][v],
                                                     'Sentence:' + family + ':' + str(slot) + ':' + v)
                    m['replay_macro_rounding_bound_pp'] = bound
                    measures[v] = m
                    for ai, attr in enumerate(labels['attrs']):
                        mask = a == ai; sub, sub_pred = calculate(s, mask)
                        mine = {'attr': attr, 'images': sub['images'], 'micro_pct': sub['micro_pct'],
                                'mean_gt_margin': sub['mean_gt_margin'], 'vs_name': trans(sub_pred, mask)}
                        stored_attr = row['variants'][v]['per_attribute'][ai]
                        check(mine['attr'] == stored_attr['attr'] and mine['images'] == stored_attr['images'], 'Sentence attr identity')
                        allowance = int((uncertain & mask).sum())
                        check(abs(mine['micro_pct'] - stored_attr['micro_pct']) <= 100 * allowance / mine['images'] + 1e-8,
                              'Sentence attr micro rounding bound')
                        check(abs(mine['mean_gt_margin'] - stored_attr['mean_gt_margin']) < 1e-6, 'Sentence attr margin')
                        for k in ('repair', 'damage', 'net'):
                            check(abs(mine['vs_name'][k] - stored_attr['vs_name'][k]) <= allowance, 'Sentence attr transition rounding bound')
                    if family == 'AE_only' and slot == 6:
                        third_scores[v] = s
                correct, p1, p2 = (measures[v]['pair_macro_pct'] for v in variants)
                deltas = {'name': correct - name_metrics['pair_macro_pct'], 'perm1': correct - p1, 'perm2': correct - p2}
                bounds = {'name': measures[variants[0]]['replay_macro_rounding_bound_pp'],
                          'perm1': measures[variants[0]]['replay_macro_rounding_bound_pp'] + measures[variants[1]]['replay_macro_rounding_bound_pp'],
                          'perm2': measures[variants[0]]['replay_macro_rounding_bound_pp'] + measures[variants[2]]['replay_macro_rounding_bound_pp']}
                for k in deltas:
                    check(abs(deltas[k] - row['correct_macro_deltas_pp'][k]) <= bounds[k] + 1e-8, 'Macro delta rounding bound')
                views.append({'family': family, 'ae_1based': slot // 3 + 1, 'oe_1based': slot % 3 + 1 if family == 'OE_AE' else None,
                              'correct_micro_pct': measures[variants[0]]['micro_pct'], 'correct_macro_pct': correct,
                              'macro_deltas_pp': deltas})
        # Reproduce prediction-independent sample selection from original TRAIN records.
        selected = []
        for ai in range(16):
            groups = {}
            for i, r in enumerate(records):
                if r['attr'] == ai:
                    groups.setdefault(r['object'], []).append(i)
            for group in groups.values():
                group.sort(key=lambda i: hashlib.sha256(('train-audit-1010|' + records[i]['image']).encode()).hexdigest())
            indices = []
            while len(indices) < 6 and any(groups.values()):
                for oi in sorted(groups):
                    if groups[oi] and len(indices) < 6:
                        indices.append(groups[oi].pop(0))
            selected += indices
        sample = js('result/sample_receipt.json')
        check([r['train_index'] for r in sample['samples']] == selected and len(set(selected)) == 96, 'Exact sampling replay')
        for row, index in zip(sample['samples'], selected):
            original = records[index]
            check(row['image'] == original['image'] and row['image_sha256'] == original['image_sha256'] and
                  row['source'] == original['source'] and row['attr'] == labels['attrs'][original['attr']] and
                  row['object'] == labels['objs'][original['object']], 'Sample record')
        # Descriptive concentration diagnostics, not new selection criteria.
        ae3, ae3pred = calculate(third_scores['T_AE'])
        third_classes, leave_out = [], []
        for ai, attr in enumerate(labels['attrs']):
            mask = a == ai
            ours, own_pred = calculate(third_scores['T_AE'], mask)
            baseline, _ = calculate(scores['T_name'], mask)
            cls_pairs = [(oi, aj) for oi, aj in pairs if aj == ai]
            contribution = (ours['pair_macro_pct'] - baseline['pair_macro_pct']) * len(cls_pairs) / len(pairs)
            third_classes.append({'attr': attr, 'images': int(mask.sum()), 'pairs': len(cls_pairs),
                                  'name_correct': baseline['correct'], 'ae3_correct': ours['correct'],
                                  'net': ours['correct'] - baseline['correct'],
                                  'name_class_pair_macro': baseline['pair_macro_pct'],
                                  'ae3_class_pair_macro': ours['pair_macro_pct'],
                                  'contribution_to_total_macro_delta_pp': contribution,
                                  'name_predictions_all_train': int((name_pred == ai).sum()),
                                  'ae3_predictions_all_train': int((ae3pred == ai).sum())})
            outside = a != ai
            base_out, _ = calculate(scores['T_name'], outside)
            correct_out, _ = calculate(third_scores['T_AE'], outside)
            controls = {v: calculate(third_scores[v], outside)[0] for v in ('T_AE_perm1', 'T_AE_perm2')}
            leave_out.append({'excluded_attr': attr, 'remaining_images': int(outside.sum()),
                              'name_macro_pct': base_out['pair_macro_pct'], 'ae3_macro_pct': correct_out['pair_macro_pct'],
                              'delta_macro_name_pp': correct_out['pair_macro_pct'] - base_out['pair_macro_pct'],
                              'delta_macro_controls_pp': {v: correct_out['pair_macro_pct'] - m['pair_macro_pct'] for v, m in controls.items()}})
        env = js('result/text_environment.json')
        check(env['clip_checkpoint_sha256'] == protocol['clip_checkpoint_sha256'] and
              env['clip_source_commit'] == protocol['clip_source_commit'], 'Encoder identity')
        out = {'passed': True, 'checks': count, 'archive_sha256': digest(ZIP), 'members': len(z.namelist()),
               'scientific_source_matches_release': True, 'independent_pooled_replay_exact': True,
               'unit_replay_with_near_tie_tolerance_passed': True, 'unit_views_total': 36,
               'unit_views_confusion_exact': 36 - len(rounding), 'unit_rounding_differences': rounding,
               'mean_text_exact_match_all_seven': True, 'replay': replay,
               'train_images': len(a), 'train_pairs': len(pairs), 'sample_images': len(selected),
               'val_arrays_read': False, 'test_arrays_read': False, 'views': views,
               'ae_third': {'metrics': ae3, 'vs_name': trans(ae3pred), 'per_attribute': third_classes,
                            'leave_one_attribute_out_descriptive_only': leave_out},
               'limits': ['No fresh GPU encoder forward or image-byte reread',
                          'PNG provenance checked; original TRAIN pixels were not independently re-rendered',
                          'All views are same TRAIN combinations; no held-out result or significance claim',
                          'A best sentence position selected after this audit is a post-hoc candidate, not a new validated teacher',
                          'Single-unit cross-runtime discrete differences bounded by 1e-6 near-tied rows; no fresh server raw unit-score replay']}
    target = RESULT / 'independent_audit.json'
    if target.exists():
        raise FileExistsError('Preserve existing independent audit')
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'checks': count, 'members': out['members'], 'archive_sha256': out['archive_sha256'],
                      'views': views, 'ae3_classes': third_classes, 'leave_out_leather': [r for r in leave_out if r['excluded_attr'] == 'Leather']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
