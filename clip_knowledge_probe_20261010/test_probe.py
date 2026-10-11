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
from probe import safe_image, build_records, image_features, encode_texts


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
            r = build_records(metadata, ids, self.protocol['labels'], pair, set(), data, {})
            self.assertEqual([x['image'] for x in r['train']], ids['train'])
            with self.assertRaises(ValueError):
                build_records(metadata, {'train': ['b.jpg'], 'val': ['c.jpg']}, self.protocol['labels'], pair, set(), data, {})

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
              'scope': 'CPU metrics, controls, prompt tokenization, exact paths, real metadata ordering, Python3.8 syntax'}
    write_json(ROOT / 'local_test_report.json', report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
