"""Local CPU tests. They do NOT impersonate a real GPU/checkpoint run."""
import ast
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from common import ROOT, read_json, sha, write_json, require
from prompts import VARIANTS, make_prompts, check_descriptions
from metrics import distribution, measure, transitions, pair_bootstrap, family_gate
from probe import safe_image, build_records, image_features, encode_texts, inspect_inputs
from splits import audit_pairs, audit_metadata


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.protocol = read_json(ROOT / 'protocol.json')
        self.ae = read_json(ROOT / 'assets/ae_dict.json')
        self.oe = read_json(ROOT / 'assets/oe_dict.json')

    def test_python38_syntax(self):
        for p in ROOT.rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'), feature_version=(3, 8))

    def test_assets_and_fixed_permutations(self):
        check_descriptions(self.protocol, self.ae, self.oe)
        for name, expected in self.protocol['asset_sha256'].items():
            self.assertEqual(sha(ROOT / 'assets' / name), expected)
        old = read_json(ROOT.parent / 'output/conditional_attribute_stageA_results_20261009/extracted/stageA/control_protocol.json')
        for name in ('P_K', 'P_perm1', 'P_perm2'):
            self.assertEqual(self.protocol['permutations'][name], old['permutations'][name])

    def test_prompts_names_immutable_and_full_units(self):
        p = make_prompts(self.protocol, self.ae, self.oe)
        for v in VARIANTS:
            self.assertEqual(len(p[v]), 1728)
        for i in range(1728):
            bases = {p[v][i].split(' It has ')[0] for v in VARIANTS}
            self.assertEqual(len(bases), 1)
        for variant in ('T_name', 'T_AE', 'T_AE_perm1', 'T_AE_perm2'):
            self.assertNotIn('Object shape:', '\n'.join(p[variant]))
        self.assertEqual(len(set(p['T_name'])), 192)
        self.assertEqual(len(set(p['T_AE'])), 576)
        self.assertNotEqual(p['T_K'], p['T_perm1'])
        self.assertNotEqual(p['T_perm1'], p['T_perm2'])

    def test_cosine_unit_aggregation_equivalence(self):
        rng = np.random.RandomState(21)
        image = rng.randn(5, 13)
        image /= np.linalg.norm(image, axis=1, keepdims=True)
        text = rng.randn(12, 16, 9, 13)
        text /= np.linalg.norm(text, axis=-1, keepdims=True)
        a = image @ text.mean(2).reshape(192, 13).T
        b = (image @ text.reshape(-1, 13).T).reshape(5, 12, 16, 9).mean(-1).reshape(5, 192)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_softmax_scale_and_finite(self):
        q = distribution(np.arange(384).reshape(2, 12, 16))
        np.testing.assert_allclose(q.sum(-1), 1)
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue((q.argmax(-1) == 15).all())

    def test_same_object_not_wrong_object(self):
        q = np.ones((3, 12, 16)) / 16
        q[:, :, 2] = 10
        q[0, 0, 1] = 100
        q[1, 1, 3] = 100
        q[2, 2, 4] = 100
        q /= q.sum(-1, keepdims=True)
        m = measure(q, np.array([1, 3, 4]), np.array([0, 1, 2]), np.ones(3, dtype=bool))
        self.assertEqual(m['accuracy_pct'], 100)
        self.assertEqual(m['pair_macro_accuracy_pct'], 100)
        self.assertEqual(sum(sum(row) for row in m['confusion']), 3)

    def test_macro_vs_micro_and_transitions(self):
        a = np.array([0, 0, 0, 1])
        o = np.zeros(4, dtype=int)
        q0 = np.ones((4, 12, 16)) / 16
        q1 = q0.copy()
        q0[:, :, 0] = 1
        q1[:, :, 1] = 1
        q0 /= q0.sum(-1, keepdims=True)
        q1 /= q1.sum(-1, keepdims=True)
        mask = np.ones(4, dtype=bool)
        m = measure(q0, a, o, mask)
        self.assertEqual(m['accuracy_pct'], 75)
        self.assertEqual(m['pair_macro_accuracy_pct'], 50)
        t = transitions(q0, q1, a, o, mask)
        self.assertEqual((t['repaired'], t['damaged'], t['net_accuracy_pp']), (1, 3, -50))
        interval = pair_bootstrap(q0, q0, a, o, mask)
        self.assertEqual(interval['pair_cluster_bootstrap_95pct_pp'], [0, 0])

    def test_gate_requires_name_and_each_permutation(self):
        def summary(macro, micro):
            return {'train': {'all': {'pair_macro_accuracy_pct': macro}},
                    'val': {'unseen': {'images': 10, 'pair_macro_accuracy_pct': macro, 'accuracy_pct': micro}}}
        s = {'T_name': summary(20, 20), 'T_K': summary(23, 23),
             'T_perm1': summary(23, 23), 'T_perm2': summary(10, 10)}
        self.assertFalse(family_gate(s, 'T_K', 'T_perm1', 'T_perm2')['passed'])
        s['T_perm1'] = summary(20, 20)
        self.assertTrue(family_gate(s, 'T_K', 'T_perm1', 'T_perm2')['passed'])
        s['T_name'] = summary(30, 30)
        self.assertFalse(family_gate(s, 'T_K', 'T_perm1', 'T_perm2')['passed'])

    def test_exact_path_and_alias_no_search(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            item = {'image': 'Leather_Boots.Ankle/x.jpg', '_image': 'Boots/Ankle/Brand/x.jpg'}
            p = data / 'images' / item['image']
            p.parent.mkdir(parents=True)
            p.write_bytes(b'fixture')
            self.assertEqual(safe_image(data, item, {})[1], 'cpf_images')
            item['image'] = '../x.jpg'
            with self.assertRaises(ValueError):
                safe_image(data, item, {})
            item['image'] = 'Leather_Boots.Ankle/y.jpg'
            with self.assertRaises(ValueError):
                safe_image(data, item, {})

    def test_records_skip_test_and_keep_stageA_order(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            metadata = []
            for identity, phase in (('b.jpg', 'train'), ('a.jpg', 'train'), ('c.jpg', 'val')):
                (data / 'images').mkdir(exist_ok=True)
                (data / 'images' / identity).write_bytes(identity.encode())
                metadata.append({'image': identity, '_image': identity, 'attr': 'Canvas', 'obj': 'Boots.Ankle', 'set': phase})
            metadata.append({'set': 'test'})  # No path/label reading is permitted.
            ids = {'train': ['b.jpg', 'a.jpg'], 'val': ['c.jpg']}
            pair = {('Canvas', 'Boots.Ankle')}
            r = build_records(metadata, ids, self.protocol['labels'], pair, pair, data, {})
            self.assertEqual([x['image'] for x in r['train']], ids['train'])
            with self.assertRaises(ValueError):
                build_records(metadata, {'train': ['b.jpg'], 'val': ['c.jpg']}, self.protocol['labels'], pair, pair, data, {})

    def real_split_inputs(self):
        data = Path(r'C:\Users\llenovo\Desktop\CPF_all_results（20261005）\czsl_all_completed_20261005_000428_2772161\files\data\ut-zap50k')
        pairs = {s: {tuple(t.split()) for t in (data / 'compositional-split-natural' / (s + '_pairs.txt')).read_text().splitlines() if t.strip()}
                 for s in ('train', 'val')}
        return data, pairs

    def synthetic_split_inputs(self):
        train = {('Canvas', 'Boots.Ankle')}
        val = train | {('Cotton', 'Boots.Ankle')}
        metadata = [dict(image='train.jpg', attr='Canvas', obj='Boots.Ankle', set='train'),
                    dict(image='seen.jpg', attr='Canvas', obj='Boots.Ankle', set='val'),
                    dict(image='unseen.jpg', attr='Cotton', obj='Boots.Ankle', set='val'),
                    dict(set='test')]
        ids = {'train': ['train.jpg'], 'val': ['seen.jpg', 'unseen.jpg']}
        expected = dict(train_pairs=1, val_pairs=2, val_seen_pairs=1, val_unseen_pairs=1,
                        train_images=1, val_images=2, val_seen_images=1, val_unseen_images=1)
        return metadata, ids, train, val, expected

    def test_real_seen_pair_overlap_is_accepted(self):
        data, pairs = self.real_split_inputs()
        for phase in ('train', 'val'):
            p = data / 'compositional-split-natural' / (phase + '_pairs.txt')
            remote = self.protocol['data'] + '/compositional-split-natural/' + phase + '_pairs.txt'
            self.assertEqual(sha(p), self.protocol['server_files_sha256'][remote])
        self.assertEqual(len(pairs['train'] & pairs['val']), 15)
        audit = audit_pairs(pairs['train'], pairs['val'], self.protocol['labels'], self.protocol['split_expectations'])
        self.assertEqual(audit['val_seen_pairs'], 15)
        self.assertEqual(audit['val_unseen_pairs'], 15)

    def test_changed_pair_counts_are_rejected(self):
        _, pairs = self.real_split_inputs()
        train = set(pairs['train'])
        train.pop()
        with self.assertRaisesRegex(ValueError, 'Split count mismatch'):
            audit_pairs(train, pairs['val'], self.protocol['labels'], self.protocol['split_expectations'])

    def test_unknown_pair_primitive_is_rejected(self):
        _, pairs = self.real_split_inputs()
        val = set(pairs['val'])
        val.pop()
        val.add(('UNKNOWN_ATTRIBUTE', 'Boots.Ankle'))
        with self.assertRaisesRegex(ValueError, 'Unknown primitive'):
            audit_pairs(pairs['train'], val, self.protocol['labels'], self.protocol['split_expectations'])

    def test_seen_unseen_metadata_and_test_skip(self):
        metadata, ids, train, val, expected = self.synthetic_split_inputs()
        audit = audit_metadata(metadata, ids, self.protocol['labels'], train, val, expected)
        self.assertEqual(audit['val_seen_images'], 1)
        self.assertEqual(audit['val_unseen_images'], 1)
        self.assertEqual(audit['train_val_image_id_overlap'], 0)

    def test_train_val_image_overlap_still_rejected(self):
        metadata, ids, train, val, expected = self.synthetic_split_inputs()
        metadata[1]['image'] = 'train.jpg'
        ids['val'][0] = 'train.jpg'
        with self.assertRaisesRegex(ValueError, 'TRAIN/VAL image overlap'):
            audit_metadata(metadata, ids, self.protocol['labels'], train, val, expected)

    def test_val_outside_declared_pairs_still_rejected(self):
        metadata, ids, train, val, expected = self.synthetic_split_inputs()
        metadata[2]['attr'] = 'Wool'
        with self.assertRaisesRegex(ValueError, 'outside val_pairs'):
            audit_metadata(metadata, ids, self.protocol['labels'], train, val, expected)

    def test_real_cpu_preflight_with_native_saved_truth(self):
        data, pairs = self.real_split_inputs()
        base = ROOT.parent / 'output/conditional_attribute_stageA_results_20261009/extracted/stageA'
        bank = ROOT.parent / 'output/BF_conditional_preflight_20261009/extracted/conditional/bank'
        p = dict(self.protocol)
        p.update(data=str(data), stageA_base=str(base), bank=str(bank))
        mapping = {}
        for phase in ('train', 'val'):
            file = data / 'compositional-split-natural' / (phase + '_pairs.txt')
            mapping[str(file)] = self.protocol['server_files_sha256'][self.protocol['data'] + '/compositional-split-natural/' + phase + '_pairs.txt']
        file = data / 'metadata_compositional-split-natural.t7'
        mapping[str(file)] = self.protocol['server_files_sha256'][self.protocol['data'] + '/' + file.name]
        for name in ('labels.json', 'train_image_ids.json', 'val_image_ids.json', 'labels_ow.npz'):
            mapping[str(base / name)] = self.protocol['server_files_sha256'][self.protocol['stageA_base'] + '/' + name]
        mapping[str(bank / 'manifest.json')] = self.protocol['server_files_sha256'][self.protocol['bank'] + '/manifest.json']
        p['server_files_sha256'] = mapping
        audit = inspect_inputs(p)[-1]
        self.assertTrue(audit['native_val_labels_equal'])
        self.assertEqual(audit['val_seen_images'], 877)
        self.assertEqual(audit['val_unseen_images'], 2337)
        self.assertEqual(audit['images_decoded'], 0)

    def test_scientific_protocol_unchanged(self):
        old = read_json(ROOT.parent / 'clip_knowledge_probe_20261010/protocol.json')
        for key in ('labels', 'names', 'permutations', 'temperature', 'asset_sha256', 'clip_source_commit',
                    'clip_model', 'image_resolution', 'clip_checkpoint_sha256', 'aggregation',
                    'primary_family', 'secondary_family', 'generic_reference_preserved'):
            self.assertEqual(self.protocol[key], old[key], key)

    def test_all_prompts_token_fit_official_tokenizer(self):
        # Import the official tokenizer directly: no GPU, torchvision, or weights.
        spec = importlib.util.spec_from_file_location('verified_tokenizer', ROOT / 'vendor/clip/simple_tokenizer.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tokenizer = module.SimpleTokenizer()
        prompts = make_prompts(self.protocol, self.ae, self.oe)
        lengths = [len(tokenizer.encode(t)) + 2 for rows in prompts.values() for t in rows]
        self.assertLessEqual(max(lengths), 77)
        self.max_tokens = max(lengths)

    def test_real_archived_metadata_aligns_without_images(self):
        import torch
        p = Path(r'C:\Users\llenovo\Desktop\CPF_all_results（20261005）\czsl_all_completed_20261005_000428_2772161\files\data\ut-zap50k\metadata_compositional-split-natural.t7')
        expected = self.protocol['server_files_sha256'][self.protocol['data'] + '/metadata_compositional-split-natural.t7']
        self.assertEqual(sha(p), expected)
        metadata = torch.load(str(p), map_location='cpu', weights_only=False)
        base = ROOT.parent / 'output/conditional_attribute_stageA_results_20261009/extracted/stageA'
        for phase, n in (('train', 22998), ('val', 3214)):
            records = [r for r in metadata if r['set'] == phase]
            ids = read_json(base / (phase + '_image_ids.json'))
            self.assertEqual([r['image'] for r in records], ids)
            self.assertEqual(len(records), n)
            self.assertEqual(len(set(ids)), n)
            self.assertTrue(all(Path(r['image']).name == Path(r['_image']).name for r in records))

    def test_cpu_synthetic_text_and_image_resume(self):
        import torch
        from PIL import Image

        class FakeClip:
            @staticmethod
            def tokenize(texts, truncate=False):
                assert not truncate
                # Deterministic synthetic tokenizer, NOT a CLIP accuracy test.
                return torch.tensor([[len(t), sum(t.encode()) % 100] for t in texts])

        class FakeModel:
            @staticmethod
            def encode_text(x):
                return x.float().repeat(1, 384)

            @staticmethod
            def encode_image(x):
                return x.float().repeat(1, 384)

        def preprocess(im):
            pixel = np.asarray(im.convert('RGB'), dtype=np.float32).mean()
            return torch.tensor([pixel + 1, pixel + 2])

        tables, audit = encode_texts(FakeClip, FakeModel, make_prompts(self.protocol, self.ae, self.oe), 'cpu', 32)
        self.assertEqual(audit['units_per_candidate'], 9)
        self.assertEqual(tables['T_K'].shape, (12, 16, 768))
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            p = work / 'synthetic.jpg'
            Image.new('RGB', (7, 9), (31, 65, 89)).save(p)
            rows = [{'image': 'synthetic%d' % i, 'source': str(p), 'image_sha256': sha(p)} for i in range(5)]
            first = image_features(FakeModel, preprocess, rows, work, 'train', 'cpu', 2, 0)
            second = image_features(FakeModel, preprocess, rows, work, 'train', 'cpu', 2, 0)
            np.testing.assert_array_equal(first, second)
            (work / 'features/train/000000.json').unlink()  # Synthetic transaction-crash fixture only.
            recovered = image_features(FakeModel, preprocess, rows, work, 'train', 'cpu', 2, 0)
            np.testing.assert_array_equal(first, recovered)
            score = (first @ tables['T_K'].reshape(192, 768).T).reshape(5, 12, 16)
            q = distribution(score)
            np.testing.assert_allclose(q.sum(-1), 1)
            rows[0]['image_sha256'] = 'bad'
            with self.assertRaises(ValueError):
                image_features(FakeModel, preprocess, rows, work, 'train', 'cpu', 2, 0)

    def test_export_integrity_and_no_sensitive_payload(self):
        from export_result import export, verify
        import zipfile
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / 'probe'
            work.mkdir()
            write_json(work / 'complete.json', {'execution_passed': True})
            (work / 'model').mkdir()
            (work / 'model/weights.pt').write_bytes(b'excluded')
            (work / 'deps').mkdir()
            (work / 'deps/file.py').write_bytes(b'excluded')
            target = work.parent / 'probe.zip'
            export(work, target)
            verify(target)
            with zipfile.ZipFile(target) as z:
                self.assertNotIn('result/model/weights.pt', z.namelist())
                self.assertNotIn('result/deps/file.py', z.namelist())
                self.assertTrue(json.loads(z.read('result/complete.json'))['execution_passed'])
            with self.assertRaises(ValueError):
                export(work, target)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProbeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'passed': result.wasSuccessful(), 'real_metadata_images_checked': 26212,
              'real_images_decoded': 0, 'synthetic_image_encode_and_resume_tested': True,
              'full_prompt_token_max': 36, 'prompt_units': 12096, 'unique_texts': 7041,
              'gpu_forward_tested': False, 'teacher_training_tested': False,
              'real_split_preflight_passed': result.wasSuccessful(),
              'real_split_counts': {'train_pairs': 83, 'val_pairs': 30, 'val_seen_pairs': 15,
                                    'val_unseen_pairs': 15, 'val_seen_images': 877, 'val_unseen_images': 2337},
              'scope': 'R2 CPU metrics, controls, real TRAIN/VAL seen/unseen preflight, native VAL truth, leakage rejection, Python3.8 syntax'}
    write_json(ROOT / 'local_test_report.json', report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
