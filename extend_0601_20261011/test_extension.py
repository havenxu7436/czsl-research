"""CPU-only structural tests. These are not real training results."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import extension as app


class ExtensionTests(unittest.TestCase):
    def fixture(self, root):
        jobs, rows = [], []
        for variant in app.MAIN:
            run = root / 'runs' / variant
            app.write(run / 'status.json', {'state': 'complete', 'checkpoint_sha256': 'same'})
            app.write(run / 'train_complete.json', {'epochs': list(range(1, 11)), 'best_sha256': 'same'})
            app.write(run / 'meta/checkpoint.json', {'sha256': 'same'})
            app.write(run / 'meta/labels.json', {'attrs': [], 'objs': []})
            for phase in ('val', 'test'):
                for setting in ('cw', 'ow'):
                    app.write(run / ('eval_' + phase + '_' + setting) / 'metrics.json', {'checkpoint_sha256': 'same'})
                    rows.append({'variant': variant, 'phase': phase, 'setting': setting, 'AUC_pct': 40.0})
            jobs.append({'variant': variant, 'run': str(run)})
        protocol = {'jobs': jobs}
        app.write(root / 'protocol.json', protocol)
        app.write(root / 'summary/summary.json', {'core_complete': True, 'all_planned_complete': True, 'rows': rows})
        return protocol

    def test_exact_three_jobs(self):
        self.assertEqual(app.NOISE, ['H_noise10', 'H_noise30', 'H_noise50'])
        self.assertFalse(set(app.NOISE) & set(app.MAIN))

    def test_parent_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = self.fixture(root)
            before = {str(p): app.sha(p) for p in root.rglob('*') if p.is_file()}
            queue = SimpleNamespace(verify_protocol=lambda p: protocol)
            with mock.patch.object(app, 'PARENT_PROTOCOL_SHA', app.sha(root / 'protocol.json')):
                returned, evidence = app.verify_parent(root, queue)
            self.assertEqual(returned, protocol)
            self.assertIn(str(root / 'summary/summary.json'), evidence)
            self.assertEqual(before, {str(p): app.sha(p) for p in root.rglob('*') if p.is_file()})

    def test_incomplete_parent_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = self.fixture(root)
            app.write(root / 'runs/H/train_complete.json', {'epochs': [1, 2], 'best_sha256': 'same'})
            with mock.patch.object(app, 'PARENT_PROTOCOL_SHA', app.sha(root / 'protocol.json')):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    app.verify_parent(root, SimpleNamespace(verify_protocol=lambda p: protocol))

    def test_stale_eval_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = self.fixture(root)
            app.write(root / 'runs/H/eval_test_ow/metrics.json', {'checkpoint_sha256': 'wrong'})
            with mock.patch.object(app, 'PARENT_PROTOCOL_SHA', app.sha(root / 'protocol.json')):
                with self.assertRaisesRegex(ValueError, 'eval identity'):
                    app.verify_parent(root, SimpleNamespace(verify_protocol=lambda p: protocol))

    def test_configuration_changes_only_asset_and_output_paths(self):
        cfg = {'MODEL': {'kcc_cache': '/old/cache', 'bf_logic_bank': '/old/bank', 'kcc_lambda': .1,
                         'kcc_beta': .01, 'full_gain': 1., 'full_use_pair': True},
               'TRAIN': {'seed': 3351, 'max_epoch': 10, 'lr': .0001, 'finetune_backbone': True,
                         'kcc_stage': 'old', 'kcc_variant': 'H', 'train_name': 'old',
                         'kcc_run_dir': '/old', 'log_dir': '/old/log', 'checkpoint_dir': '/old/ckpt'}}
        original = copy.deepcopy(cfg)
        got = app.changed_config(cfg, Path('/new/work'), 'H_noise30')
        self.assertEqual(cfg, original)
        for key in ('kcc_lambda', 'kcc_beta', 'full_gain', 'full_use_pair'):
            self.assertEqual(got['MODEL'][key], original['MODEL'][key])
        for key in ('seed', 'max_epoch', 'lr', 'finetune_backbone'):
            self.assertEqual(got['TRAIN'][key], original['TRAIN'][key])
        self.assertTrue(got['MODEL']['kcc_cache'].replace('\\', '/').endswith('/cache_noise30'))

    def test_partial_summary_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, work = Path(tmp) / 'parent', Path(tmp) / 'noise'
            self.fixture(parent)
            app.write(work / 'protocol.json', {'parent_work': str(parent), 'parent_protocol_sha256': app.sha(parent / 'protocol.json'),
                                               'jobs': [{'variant': v, 'run': str(work / 'runs' / v)} for v in app.NOISE],
                                               'not_covered': ['description counts']})
            core, complete = app.summarize(work)
            self.assertTrue(core)
            self.assertFalse(complete)
            summary = app.read(work / 'summary/summary.json')
            self.assertEqual(len(summary['rows']), 48)
            self.assertEqual(summary['new_training_jobs'], 3)
            self.assertEqual(summary['reused_parent_runs'], 12)
            self.assertTrue(all(s['state'] == 'not_started' for s in summary['states']))

    def test_full_summary_appends_only_twelve_noise_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, work = Path(tmp) / 'parent', Path(tmp) / 'noise'
            self.fixture(parent)
            app.write(work / 'protocol.json', {'parent_work': str(parent), 'parent_protocol_sha256': app.sha(parent / 'protocol.json'),
                                               'jobs': [{'variant': v, 'run': str(work / 'runs' / v)} for v in app.NOISE], 'not_covered': []})
            for variant in app.NOISE:
                run = work / 'runs' / variant
                app.write(run / 'status.json', {'state': 'complete'})
                for phase in ('val', 'test'):
                    for setting in ('cw', 'ow'):
                        app.write(run / ('eval_' + phase + '_' + setting) / 'metrics.json',
                                  {'best_epoch': 6, 'checkpoint_sha256': variant, 'metrics_raw':
                                   {'AUC': .35, 'best_hm': .5, 'best_seen': .6, 'best_unseen': .7}})
            self.assertEqual(app.summarize(work), (True, True))
            summary = app.read(work / 'summary/summary.json')
            self.assertEqual(len(summary['rows']), 60)
            self.assertEqual([r['AUC_pct'] for r in summary['rows'] if r['variant'] in app.NOISE], [35.] * 12)

    def test_server_python38_syntax(self):
        ast.parse((Path(app.__file__)).read_text(encoding='utf-8'), feature_version=(3, 8))

    def test_prepare_integration_only_three_jobs_and_parent_unchanged(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            parent, work, release, project = [base / p for p in ('parent', 'noise', 'r2', 'project')]
            original = self.fixture(parent)
            source = parent / 'source'
            (source / 'tools').mkdir(parents=True)
            (source / 'tools/build_knowledge.py').write_text('# fixture only\n', encoding='utf-8')
            app.write(release / 'release_manifest.json', {'source/tools/build_knowledge.py': app.sha(source / 'tools/build_knowledge.py')})
            app.write(parent / 'logic_bank/manifest.json', {'fixture': True})
            (parent / 'configs').mkdir()
            config = {'MODEL': {'kcc_cache': '/old/cache', 'bf_logic_bank': '/old/bank', 'kcc_lambda': .1},
                      'TRAIN': {'seed': 3351, 'max_epoch': 10}}
            (parent / 'configs/H.yml').write_text(yaml.safe_dump(config), encoding='utf-8')
            original.update(source=str(source), root=str(project), frozen_inputs={}, not_covered=['knowledge noise sensitivity (tier2 not scheduled)', 'multi-seed'])
            before = {str(p): app.sha(p) for p in parent.rglob('*') if p.is_file()}

            def altered(cache, dest, kind, builder, glove):
                app.write(dest / 'manifest.json', {'fixture_noise': kind})
                return {}

            preparation = SimpleNamespace(verify_cache=lambda p: {}, altered_cache=altered)
            args = SimpleNamespace(parent=parent, work=work, release=release)
            with mock.patch.object(app, 'parent_modules', return_value=(None, preparation, None)), \
                 mock.patch.object(app, 'verify_parent', return_value=(original, {})):
                app.prepare(args)
                with self.assertRaises(FileExistsError):
                    app.prepare(args)
            result = app.read(work / 'protocol.json')
            self.assertEqual([j['variant'] for j in result['jobs']], app.NOISE)
            self.assertTrue(all(j['tier'] == 2 for j in result['jobs']))
            self.assertEqual(before, {str(p): app.sha(p) for p in parent.rglob('*') if p.is_file()})
            for job in result['jobs']:
                self.assertEqual(app.sha(job['cfg']), job['cfg_sha256'])
                cfg = yaml.safe_load(Path(job['cfg']).read_text(encoding='utf-8'))
                self.assertEqual(cfg['MODEL']['kcc_lambda'], .1)
            self.assertNotIn('knowledge noise sensitivity (tier2 not scheduled)', result['not_covered'])


if __name__ == '__main__':
    unittest.main()
