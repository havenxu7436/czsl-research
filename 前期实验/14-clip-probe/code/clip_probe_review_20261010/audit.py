"""Independent, read-only tensor audit. Never executes code from the result ZIP."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
RESULT = WORKSPACE / 'output/clip_knowledge_probe_results_20261010_r2'
ARCHIVE = RESULT / 'czsl_clip_knowledge_probe_20261010_r2.zip'
SOURCE = WORKSPACE / 'clip_knowledge_probe_20261010_r2'
STAGE = WORKSPACE / 'output/conditional_attribute_stageA_results_20261009/extracted/stageA'
VARIANTS = ('T_name', 'T_K', 'T_perm1', 'T_perm2', 'T_AE', 'T_AE_perm1', 'T_AE_perm2')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit():
    checks = 0
    max_nll_difference = max_cos_difference = max_q_difference = 0.0

    def check(test, message):
        nonlocal checks
        checks += 1
        if not test:
            raise AssertionError(message)

    check(digest(ARCHIVE) == Path(str(ARCHIVE) + '.sha256').read_text().split()[0], 'Archive checksum')
    with zipfile.ZipFile(ARCHIVE) as z:
        def js(name):
            return json.loads(z.read(name))

        def arrays(name):
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as a:
                return {k: a[k] for k in a.files}

        names = z.namelist()
        manifest = js('export_manifest.json')['files_sha256']
        check(set(names) == set(manifest) | {'export_manifest.json'}, 'Member set')
        check(len(names) == len(set(names)), 'Duplicate members')
        for name, expected in manifest.items():
            with z.open(name) as f:
                h = hashlib.sha256()
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            check(h.hexdigest() == expected, 'Member checksum: ' + name)
        local_manifest = json.loads((SOURCE / 'release_manifest.json').read_text())
        check(js('source/release_manifest.json') == local_manifest, 'Source release identity')
        for name, expected in local_manifest['files_sha256'].items():
            check(hashlib.sha256(z.read('source/' + name)).hexdigest() == expected, 'Frozen source: ' + name)
        check(js('result/release_identity.json')['release_manifest_sha256'] == digest(SOURCE / 'release_manifest.json'),
              'Runtime source receipt')
        complete = js('result/complete.json')
        check(complete['execution_passed'] and complete['optimizer_steps'] == 0 and complete['test_images_decoded'] == 0,
              'Execution scope')
        check(not any(n.startswith('result/failure_') for n in names), 'Failure records present')
        protocol = js('source/protocol.json')
        check(js('result/release_identity.json')['protocol'] == protocol, 'Runtime protocol identity')
        summary = js('result/summary.json')
        labels = protocol['labels']
        diagnostics = {p: js('result/' + p + '_diagnostics.json') for p in ('train', 'val')}
        # Use original local archived split files and metadata, not exported masks.
        data = Path(r'C:\Users\llenovo\Desktop\CPF_all_results（20261005）\czsl_all_completed_20261005_000428_2772161\files\data\ut-zap50k')
        splits = {}
        for phase in ('train', 'val'):
            p = data / 'compositional-split-natural' / (phase + '_pairs.txt')
            expected = protocol['server_files_sha256'][protocol['data'] + '/compositional-split-natural/' + phase + '_pairs.txt']
            check(digest(p) == expected, 'Original split checksum')
            splits[phase] = {tuple(t.split()) for t in p.read_text().splitlines() if t.strip()}
        check((len(splits['train']), len(splits['val']), len(splits['train'] & splits['val']),
               len(splits['val'] - splits['train'])) == (83, 30, 15, 15), 'Real generalized split')
        metadata_path = data / 'metadata_compositional-split-natural.t7'
        check(digest(metadata_path) == protocol['server_files_sha256'][protocol['data'] + '/' + metadata_path.name], 'Metadata checksum')
        import torch
        metadata = torch.load(str(metadata_path), map_location='cpu', weights_only=False)
        originals = {p: [r for r in metadata if r.get('set') == p] for p in ('train', 'val')}
        records = {p: js('result/' + p + '_records.json') for p in ('train', 'val')}
        image_sets = {p: {r['image'] for r in records[p]} for p in ('train', 'val')}
        check(not image_sets['train'] & image_sets['val'], 'Cross-split image ID overlap')
        pixel_sets = {p: {r['image_sha256'] for r in records[p]} for p in ('train', 'val')}
        pixel_overlap = pixel_sets['train'] & pixel_sets['val']
        text = arrays('result/text_features.npz')
        check(set(text) == set(VARIANTS), 'Text feature variants')
        # Validate prompts with known local source; do not import archive source.
        import sys
        sys.path.insert(0, str(SOURCE))
        from prompts import make_prompts
        expected_prompts = make_prompts(protocol, json.loads((SOURCE / 'assets/ae_dict.json').read_text()),
                                       json.loads((SOURCE / 'assets/oe_dict.json').read_text()))
        check(js('result/prompts.json') == expected_prompts, 'Fixed prompts and permutations')
        recomputed = {v: {} for v in VARIANTS}
        details = {'per_pair': {}, 'class_bias': {}, 'text_norms': {}}
        truths = {}
        all_val_q = {}

        def softmax(x):
            x = x.astype(np.float64) / 0.05
            x -= x.max(-1, keepdims=True)
            e = np.exp(x)
            return e / e.sum(-1, keepdims=True)

        def metrics(q, a, o, mask):
            ix = np.flatnonzero(mask)
            p = q[ix, o[ix]]
            pred = p.argmax(1)
            ok = pred == a[ix]
            groups = sorted(set(zip(o[ix].tolist(), a[ix].tolist())))
            macro = np.mean([ok[(o[ix] == oi) & (a[ix] == ai)].mean() for oi, ai in groups])
            confusion = np.zeros((16, 16), dtype=int)
            for actual, estimated in zip(a[ix], pred):
                confusion[actual, estimated] += 1
            return {'images': len(ix), 'pairs': len(groups), 'accuracy_pct': float(ok.mean() * 100),
                    'pair_macro_accuracy_pct': float(macro * 100),
                    'nll': float(-np.log(p[np.arange(len(ix)), a[ix]].astype(np.float64)).mean()),
                    'confusion': confusion.tolist(), 'correct': int(ok.sum())}

        def compare_metric(actual, stored, title):
            nonlocal max_nll_difference
            for key in ('images', 'pairs', 'confusion'):
                check(actual[key] == stored[key], title + '/' + key)
            for key in ('accuracy_pct', 'pair_macro_accuracy_pct'):
                check(abs(actual[key] - stored[key]) < 1e-10, title + '/' + key)
            difference = abs(actual['nll'] - stored['nll'])
            max_nll_difference = max(max_nll_difference, difference)
            check(difference < 1e-6, title + '/nll')
            c = np.array(actual['confusion'])
            for row in stored['per_attribute']:
                ai = row['index']
                n = int(c[ai].sum())
                check(n == row['images'], title + '/class_count')
                if n:
                    check(abs(100 * c[ai, ai] / n - row['accuracy_pct']) < 1e-10, title + '/class_accuracy')
                else:
                    check(row['accuracy_pct'] is None, title + '/empty_class')

        def transitions(q, ref, a, o, mask):
            ix = np.flatnonzero(mask)
            own = q[ix, o[ix]].argmax(1) == a[ix]
            old = ref[ix, o[ix]].argmax(1) == a[ix]
            repair, damage = int((own & ~old).sum()), int((old & ~own).sum())
            return {'images': len(ix), 'repaired': repair, 'damaged': damage,
                    'net_accuracy_pp': 100 * (repair - damage) / len(ix)}

        def compare_transition(actual, stored, title):
            for k in ('images', 'repaired', 'damaged'):
                check(actual[k] == stored[k], title + '/' + k)
            check(abs(actual['net_accuracy_pp'] - stored['net_accuracy_pp']) < 1e-10, title + '/delta')

        for phase, n in (('train', 22998), ('val', 3214)):
            rows = records[phase]
            check(len(rows) == n, phase + '/records')
            old_ids = json.loads((STAGE / (phase + '_image_ids.json')).read_text())
            check([r['image'] for r in rows] == old_ids == [r['image'] for r in originals[phase]], phase + '/identity order')
            for r, original in zip(rows, originals[phase]):
                check((labels['attrs'][r['attr']], labels['objs'][r['object']]) ==
                      (original['attr'], original['obj']), phase + '/record truth')
                check(len(r['image_sha256']) == 64 and set(r['image_sha256']) <= set('0123456789abcdef'), 'Image checksum syntax')
            truth = arrays('result/teacher/' + phase + '_truth.npz')
            a, o = truth['attrs'], truth['objects']
            seen = np.array([(labels['attrs'][int(ai)], labels['objs'][int(oi)]) in splits['train'] for ai, oi in zip(a, o)])
            check(np.array_equal(a, [r['attr'] for r in rows]) and np.array_equal(o, [r['object'] for r in rows]), phase + '/tensor truth')
            check(np.array_equal(seen, truth['seen']) and np.array_equal(seen, [r['seen'] for r in rows]), phase + '/seen mask')
            if phase == 'val':
                check((int(seen.sum()), int((~seen).sum())) == (877, 2337), 'Native seen/unseen image counts')
            truths[phase] = (a, o, seen)
            chunks = []
            for start in range(0, n, 512):
                end = min(start + 512, n)
                prefix = 'result/features/' + phase + '/%06d' % start
                receipt = js(prefix + '.json')
                check(hashlib.sha256(z.read(prefix + '.npz')).hexdigest() == receipt['sha256'], 'Feature receipt checksum')
                ih = hashlib.sha256('\n'.join(r['image'] + ' ' + r['image_sha256'] for r in rows[start:end]).encode()).hexdigest()
                shard = arrays(prefix + '.npz')
                check(receipt['identity_sha256'] == ih == shard['identity_sha256'].item(), 'Feature identity')
                check((receipt['start'], receipt['end']) == (start, end), 'Feature row boundaries')
                x = shard['features']
                check(x.shape == (end - start, 768) and x.dtype == np.float32 and np.isfinite(x).all(), 'Feature shape and finiteness')
                check(np.allclose(np.linalg.norm(x, axis=1), 1, atol=2e-3), 'Feature normalization')
                chunks.append(x)
            features = np.concatenate(chunks)
            masks = {'all': np.ones(n, dtype=bool)}
            if phase == 'val':
                masks.update(seen=seen, unseen=~seen)
            predictions = {}
            saved_q = {}
            for variant in VARIANTS:
                tf = text[variant]
                check(tf.shape == (12, 16, 768) and np.isfinite(tf).all(), 'Text features')
                norms = np.linalg.norm(tf, axis=-1)
                details['text_norms'][variant] = {'min': float(norms.min()), 'max': float(norms.max()), 'mean': float(norms.mean())}
                saved = arrays('result/teacher/' + phase + '_' + variant + '.npz')
                q, cosine = saved['q'], saved['cosine']
                check(q.shape == cosine.shape == (n, 12, 16), 'Teacher shape')
                check(np.isfinite(q).all() and np.isfinite(cosine).all() and (q > 0).all(), 'Teacher finite values')
                computed_cos = (features @ tf.reshape(192, 768).T).reshape(n, 12, 16)
                cosine_difference = float(np.abs(computed_cos - cosine).max())
                max_cos_difference = max(max_cos_difference, cosine_difference)
                check(cosine_difference < 2e-6, 'Image/text cosine replay')
                expected_q = softmax(cosine).astype(np.float32)
                q_difference = float(np.abs(expected_q - q).max())
                max_q_difference = max(max_q_difference, q_difference)
                check(q_difference < 2e-7 and np.allclose(q.sum(-1), 1, atol=1e-6), 'Probability replay')
                predictions[variant] = q[np.arange(n), o].argmax(1)
                saved_q[variant] = q
                recomputed[variant][phase] = {}
                for subset, mask in masks.items():
                    m = metrics(q, a, o, mask)
                    compare_metric(m, summary['metrics'][variant][phase][subset], variant + '/' + phase + '/' + subset)
                    recomputed[variant][phase][subset] = {k: v for k, v in m.items() if k != 'confusion'}
                counts = np.bincount(predictions[variant], minlength=16)
                details['class_bias'].setdefault(phase, {})[variant] = {labels['attrs'][ai]: int(counts[ai]) for ai in range(16)}
                if phase == 'val':
                    all_val_q[variant] = q
            for variant in VARIANTS:
                for subset, mask in masks.items():
                    actual = transitions(saved_q[variant], saved_q['T_name'], a, o, mask)
                    compare_transition(actual, diagnostics[phase]['transitions_vs_name'][variant][subset], 'Name transitions')
            pair_details = []
            for oi, ai in sorted(set(zip(o.tolist(), a.tolist()))):
                mask = (o == oi) & (a == ai)
                entry = {'attr': labels['attrs'][ai], 'object': labels['objs'][oi],
                         'images': int(mask.sum()), 'seen': (labels['attrs'][ai], labels['objs'][oi]) in splits['train'],
                         'correct': {v: int((predictions[v][mask] == ai).sum()) for v in VARIANTS}}
                entry['K_net_vs_name'] = entry['correct']['T_K'] - entry['correct']['T_name']
                entry['AE_net_vs_name'] = entry['correct']['T_AE'] - entry['correct']['T_name']
                pair_details.append(entry)
            details['per_pair'][phase] = pair_details
            print('INDEPENDENT_TEACHER_REPLAY_PASSED', phase, n, flush=True)
        # Independent VAL CPF reference from archived raw scores, no model execution.
        grid = np.empty((3214, 12, 16), dtype=np.float32)
        reference = json.loads((WORKSPACE / 'generic_mixture_controls_20261009/legacy/reference.json').read_text())
        for name in ('val_scores_ow.npy', 'candidates_ow.json'):
            expected = reference['stageA_files'][reference['stageA_base'] + '/' + name] if name == 'val_scores_ow.npy' else js('result/release_identity.json')['protocol'].get('unused', None)
            if expected is not None:
                check(digest(STAGE / name) == expected, 'Archived CPF input checksum')
        candidates = json.loads((STAGE / 'candidates_ow.json').read_text())
        scores = np.load(STAGE / 'val_scores_ow.npy', allow_pickle=False)
        check(len(candidates) == 192 and len(set(map(tuple, candidates))) == 192, 'CPF OW candidate coverage')
        for i, (attr, obj) in enumerate(candidates):
            grid[:, labels['objs'].index(obj), labels['attrs'].index(attr)] = scores[:, i]
        cpf_q = softmax(grid)
        a, o, seen = truths['val']
        masks = dict(all=np.ones(3214, dtype=bool), seen=seen, unseen=~seen)
        cpf = {}
        for subset, mask in masks.items():
            m = metrics(cpf_q, a, o, mask)
            compare_metric(m, diagnostics['val']['cpf_reference'][subset], 'CPF/' + subset)
            cpf[subset] = {k: v for k, v in m.items() if k != 'confusion'}
            for variant in VARIANTS:
                compare_transition(transitions(all_val_q[variant], cpf_q, a, o, mask),
                                   diagnostics['val']['transitions_vs_C_text'][variant][subset], 'CPF transitions')
        gates = {}
        for key, family, c1, c2 in (('OE_AE_primary', 'T_K', 'T_perm1', 'T_perm2'),
                                    ('AE_only_secondary', 'T_AE', 'T_AE_perm1', 'T_AE_perm2')):
            base = recomputed[family]['val']['unseen']
            diffs = {v: base['pair_macro_accuracy_pct'] - recomputed[v]['val']['unseen']['pair_macro_accuracy_pct']
                     for v in ('T_name', c1, c2)}
            micro_delta = base['accuracy_pct'] - recomputed['T_name']['val']['unseen']['accuracy_pct']
            train_delta = recomputed[family]['train']['all']['pair_macro_accuracy_pct'] - recomputed['T_name']['train']['all']['pair_macro_accuracy_pct']
            passed = min(diffs.values()) >= 1 and micro_delta >= 0 and train_delta >= 0
            check(passed == summary['gates'][key]['passed'] == complete['gates'][key]['passed'], 'Gate replay')
            for v, d in diffs.items():
                check(abs(d - summary['gates'][key]['macro_deltas_pp'][v]) < 1e-10, 'Gate delta')
            gates[key] = dict(passed=passed, macro_deltas_pp=diffs, micro_minus_name_pp=micro_delta, train_macro_minus_name_pp=train_delta)
            mask = ~seen
            ix = np.flatnonzero(mask)
            pair_ids = o[ix] * 16 + a[ix]
            correct1 = all_val_q[family][ix, o[ix]].argmax(1) == a[ix]
            for control in ('T_name', c1, c2):
                correct0 = all_val_q[control][ix, o[ix]].argmax(1) == a[ix]
                delta = correct1.astype(float) - correct0.astype(float)
                means = np.array([delta[pair_ids == p].mean() for p in np.unique(pair_ids)])
                rng = np.random.RandomState(10102026)
                draws = means[rng.randint(0, len(means), size=(2000, len(means)))].mean(1) * 100
                interval = np.percentile(draws, [2.5, 97.5])
                reported = diagnostics['val']['paired_intervals'][family][control]['pair_cluster_bootstrap_95pct_pp']
                check(np.allclose(interval, reported, atol=1e-10, rtol=0), 'Pair bootstrap replay')
        report = {'passed': True, 'checks': checks, 'archive_sha256': digest(ARCHIVE), 'members': len(names),
                  'max_cosine_replay_abs_diff': max_cos_difference, 'max_q_replay_abs_diff': max_q_difference,
                  'max_nll_abs_diff_float64_vs_reported': max_nll_difference,
                  'all_top1_micro_macro_confusion_replays_match': True,
                  'train_val_image_id_overlap': 0, 'train_val_exact_image_byte_hash_overlap': len(pixel_overlap),
                  'unique_image_byte_hashes': {p: len(v) for p, v in pixel_sets.items()},
                  'scientific_source_matches_local_r2': True, 'gates': gates,
                  'metrics': recomputed, 'cpf_val_reference': cpf,
                  'name_transitions_val': diagnostics['val']['transitions_vs_name'],
                  'cpf_transitions_val': diagnostics['val']['transitions_vs_C_text'],
                  'pair_intervals_val': diagnostics['val']['paired_intervals'], 'details': details,
                  'scope_limits': ['No fresh GPU image/text encoder replay',
                                   'Remote current filesystem not inspected',
                                   'TRAIN CPF absolute reference not independently replayed; local TRAIN cache is absent',
                                   'GT-object attribute diagnostics are not deployable CPF CW/OW metrics',
                                   'VAL is development data; no TEST evaluated']}
    output = RESULT / 'independent_audit.json'
    if output.exists():
        raise FileExistsError('Preserve prior independent audit: ' + str(output))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ('metrics', 'cpf_val_reference', 'name_transitions_val', 'cpf_transitions_val', 'pair_intervals_val', 'details')}, ensure_ascii=False))


if __name__ == '__main__':
    audit()
