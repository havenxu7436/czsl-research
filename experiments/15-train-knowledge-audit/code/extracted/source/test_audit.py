"""Local CPU tests, including real archived TRAIN identities; no GPU/VAL data."""
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from common import ROOT, Inputs, require, sha, valid_relative, write_json
from analysis import (VARIANTS, metrics, transition, select_samples, pooled_report,
                      sentence_report, make_prompts, unit_scores)
from report import contacts, safe_train_image, human_form, html_report


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT.parent / 'clip_knowledge_probe_20261010_r2/protocol.json').read_text())
        cls.ae = json.loads((ROOT.parent / 'clip_knowledge_probe_20261010_r2/assets/ae_dict.json').read_text())
        cls.oe = json.loads((ROOT.parent / 'clip_knowledge_probe_20261010_r2/assets/oe_dict.json').read_text())
        rng = np.random.RandomState(7)
        cls.a = np.repeat(np.arange(16), 2)
        cls.o = np.tile([0, 1], 16)
        cls.features = rng.randn(32, 768).astype(np.float32)
        cls.features /= np.linalg.norm(cls.features, axis=1, keepdims=True)
        cls.tables = {}
        for v in VARIANTS:
            t = rng.randn(12, 16, 9, 768).astype(np.float32)
            t /= np.linalg.norm(t, axis=-1, keepdims=True)
            cls.tables[v] = t
        cls.means = {v: t.mean(2, dtype=np.float32) for v, t in cls.tables.items()}
        cls.scores = {v: np.stack([cls.features[i] @ cls.means[v][cls.o[i]].T for i in range(32)]) for v in VARIANTS}

    def test_safe_relatives(self):
        valid_relative('features/train/000000.npz')
        for s in ('../foo', '/foo', 'C:/foo', 'foo\\bar'):
            with self.assertRaises(ValueError):
                valid_relative(s)

    def test_closed_input_allowlist(self):
        with self.assertRaises(ValueError):
            Inputs({'teacher/val_T_K.npz': 'x'}, root=ROOT)
        inp = Inputs({}, root=ROOT)
        with self.assertRaises(ValueError):
            inp.blob('teacher/test_T_K.npz')

    def test_input_sha_and_zip_backend(self):
        b = b'{"value":1}'
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, 'w') as z:
            z.writestr('result/prompts.json', b)
        with zipfile.ZipFile(memory) as z:
            inp = Inputs({'prompts.json': hashlib.sha256(b).hexdigest()}, archive=z)
            self.assertEqual(inp.js('prompts.json'), {'value': 1})
            self.assertEqual(inp.reads, {'prompts.json'})
            bad = Inputs({'prompts.json': '0' * 64}, archive=z)
            with self.assertRaises(ValueError):
                bad.js('prompts.json')

    def test_micro_macro_difference(self):
        a, o = np.array([0, 0, 0, 1]), np.zeros(4, dtype=int)
        scores = np.zeros((4, 16)); scores[:, 0] = 1
        m, _, _ = metrics(scores, a, o)
        self.assertEqual(m['micro_pct'], 75)
        self.assertEqual(m['pair_macro_pct'], 50)

    def test_margin_excludes_ground_truth(self):
        scores = np.zeros((1, 16)); scores[0, 3] = .7; scores[0, 5] = .2
        m, _, margin = metrics(scores, np.array([3]), np.array([0]))
        self.assertAlmostEqual(margin[0], .5)
        self.assertAlmostEqual(m['mean_gt_margin'], .5)

    def test_transition(self):
        actual = np.array([0, 1, 2, 3])
        old, new = np.array([0, 0, 2, 0]), np.array([0, 1, 0, 0])
        self.assertEqual(transition(new, old, actual), {'repair': 1, 'damage': 1, 'net': 0})

    def test_prompts_identical_to_original(self):
        spec = importlib.util.spec_from_file_location('known_original_prompts', ROOT.parent / 'clip_knowledge_probe_20261010_r2/prompts.py')
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        ours = make_prompts(self.protocol, self.ae, self.oe)
        self.assertEqual(ours, old.make_prompts(self.protocol, self.ae, self.oe))
        self.assertEqual(len(set(t for values in ours.values() for t in values)), 7041)

    def test_name_preserved_under_permutation(self):
        prompts = make_prompts(self.protocol, self.ae, self.oe)
        for index, base in enumerate(prompts['T_name']):
            for v in VARIANTS:
                self.assertTrue(prompts[v][index].startswith(base))

    def test_unit_scoring_equals_mean(self):
        raw = unit_scores(self.features, self.tables['T_K'], self.o)
        self.assertTrue(np.allclose(raw.mean(2), self.scores['T_K'], atol=2e-7))

    def test_sentence_replay_and_three_nine_positions(self):
        report = sentence_report(self.features, self.tables, self.means, self.scores, self.a, self.o, self.protocol['labels'])
        self.assertEqual(len(report['families']['OE_AE']), 9)
        self.assertEqual(len(report['families']['AE_only']), 3)
        self.assertTrue(all(r['top1_mismatches'] == 0 for r in report['replay'].values()))
        self.assertFalse(report['weights_fitted'])

    def test_bad_unit_shape(self):
        with self.assertRaises(ValueError):
            unit_scores(self.features, self.tables['T_K'][:, :, :8], self.o)

    def test_sentence_mismatch_stops(self):
        changed = dict(self.means); changed['T_K'] = self.means['T_K'] + .01
        with self.assertRaises(ValueError):
            sentence_report(self.features, self.tables, changed, self.scores, self.a, self.o, self.protocol['labels'])

    def test_pooled_attribute_counts(self):
        report, preds = pooled_report(self.scores, self.a, self.o, self.protocol['labels'])
        self.assertEqual(len(report['per_attribute']), 16)
        self.assertEqual(len(report['per_pair']), 32)
        self.assertEqual(sum(r['images'] for r in report['per_attribute']), 32)

    def test_sampling_deterministic_object_round_robin(self):
        records = [{'attr': ai, 'object': oi, 'image': '%d-%d-%d' % (ai, oi, j)}
                   for ai in range(16) for oi in range(3) for j in range(5)]
        selected = select_samples(records)
        self.assertEqual(selected, select_samples(records))
        self.assertEqual(len(selected), 96)
        for ai in range(16):
            ids = [i for i in selected if records[i]['attr'] == ai]
            self.assertEqual(len(ids), 6)
            self.assertEqual({records[i]['object'] for i in ids}, {0, 1, 2})

    def test_real_train_selection_and_existing_metrics(self):
        path = ROOT.parent / 'output/clip_knowledge_probe_results_20261010_r2/czsl_clip_knowledge_probe_20261010_r2.zip'
        with zipfile.ZipFile(path) as z:
            records = json.loads(z.read('result/train_records.json'))
            selected = select_samples(records)
            self.assertEqual(len(selected), 96)
            self.assertEqual({records[i]['attr'] for i in selected}, set(range(16)))
            self.assertTrue(all(records[i]['seen'] for i in selected))
            with np.load(io.BytesIO(z.read('result/teacher/train_truth.npz')), allow_pickle=False) as truth:
                a, o = truth['attrs'], truth['objects']
            score = {}
            for v in VARIANTS:
                with np.load(io.BytesIO(z.read('result/teacher/train_' + v + '.npz')), allow_pickle=False) as saved:
                    score[v] = saved['cosine'][np.arange(len(a)), o]
            report, _ = pooled_report(score, a, o, self.protocol['labels'])
            self.assertAlmostEqual(report['metrics']['T_name']['micro_pct'], 8.792068875554396)
            self.assertAlmostEqual(report['metrics']['T_K']['pair_macro_pct'], 22.485594092269384)
            write_json(ROOT / 'local_train_pooled_report.json', report)
            write_json(ROOT / 'local_selected_train_indices.json', {'train_indices': selected,
                                                                  'selects_on_predictions': False})

    def test_image_path_and_checksum_guard(self):
        with tempfile.TemporaryDirectory(prefix='train_audit_path_') as temp:
            root = Path(temp); path = root / 'image.bin'; path.write_bytes(b'abc')
            row = {'source': str(path), 'image_sha256': sha(path)}
            self.assertEqual(safe_train_image(row, root), path)
            row['image_sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                safe_train_image(row, root)

    def test_contact_sheets_html_and_blank_review(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='train_audit_report_') as temp:
            root = Path(temp); data = root / 'data'; data.mkdir(); out = root / 'report'; out.mkdir()
            records = []
            for ai in range(16):
                for j in range(6):
                    p = data / ('%02d_%d.png' % (ai, j)); Image.new('RGB', (200, 110), (ai * 13, 180, j * 30)).save(p)
                    records.append({'image': p.name, 'source': str(p), 'image_sha256': sha(p), 'attr': ai, 'object': j % 2})
            preds = {v: np.array([r['attr'] for r in records]) for v in VARIANTS}
            protocol = dict(self.protocol, data=str(data))
            self.assertEqual(contacts(records, preds, protocol, out), 96)
            self.assertEqual(len(list((out / 'contact_sheets').glob('*.png'))), 16)
            pooled, _ = pooled_report(self.scores, self.a, self.o, protocol['labels'])
            sentences = sentence_report(self.features, self.tables, self.means, self.scores, self.a, self.o, protocol['labels'])
            human_form(self.ae, self.oe, protocol, out)
            html_report(pooled, sentences, protocol, self.ae, self.oe, out)
            self.assertEqual(len(json.loads((out / 'human_review_blank.json').read_text())['items']), 84)
            self.assertIn('contact_sheets/attr_15.png', (out / 'review.html').read_text(encoding='utf-8'))
            # Retain one tiny fixture preview for visual QA, never real data.
            import shutil
            shutil.copyfile(out / 'contact_sheets/attr_14.png', ROOT / 'local_fixture_preview.png')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(ROOT / 'local_test_report.json', {'passed': result.wasSuccessful(), 'tests': result.testsRun,
                                               'failures': len(result.failures), 'errors': len(result.errors),
                                               'real_train_metric_replay': True, 'gpu_run': False,
                                               'val_arrays_read': False, 'test_arrays_read': False})
    raise SystemExit(0 if result.wasSuccessful() else 1)
