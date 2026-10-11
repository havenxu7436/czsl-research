"""Synthetic filesystem tests, not server training or GPU verification."""
import ast
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import export_preflight as core


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_bytes(core.json_bytes(value))
    elif isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding='utf-8')
    return core.sha(path)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='bf_preflight_test_')
        self.root = Path(self.temp.name)
        self.bf, self.ut = self.root / 'bf', self.root / 'ut'
        self.out = self.root / 'evidence.zip'
        self.old_reference = dict(core.C_REFERENCE)
        self.old_source = dict(core.C_SOURCE_REFERENCE)
        self.make_fixture()

    def tearDown(self):
        core.C_REFERENCE.clear()
        core.C_REFERENCE.update(self.old_reference)
        core.C_SOURCE_REFERENCE.clear()
        core.C_SOURCE_REFERENCE.update(self.old_source)
        self.temp.cleanup()

    def make_fixture(self):
        # These deliberately invalid model weights must never be deserialized.
        native = 'raise RuntimeError("EXPERIMENT SOURCE MUST NEVER EXECUTE IN EXPORTER")\n'
        src = self.bf / 'src'
        code_sha = write(src / 'train.py', native)
        install_sha = write(src / 'bf_install.json', {'output_hashes': {'train.py': code_sha}})
        bank = self.bf / 'logic_bank/ut-zap50k'
        bank_sha = write(bank / 'bank.pt', b'NOT_A_TORCH_PICKLE')
        manifest_sha = write(bank / 'manifest.json', {'bank_sha256': bank_sha})
        write(bank / 'coverage.json', {})
        protocol = dict(seed=3351, epochs=10, beta=0.01, q=3,
                        source=str(src), install_sha256=install_sha,
                        logic_bank_manifest_sha256=manifest_sha, configs={})
        for variant in core.BF_VARIANTS:
            run = self.bf / 'experiments/runs' / variant
            cfg = self.bf / 'experiments/configs' / (variant + '.yml')
            cfg_sha = write(cfg, 'MODEL:\n  kcc_logic: %s\n' % (variant in ('B', 'F')))
            protocol['configs'][variant] = dict(path=str(cfg), run=str(run), sha256=cfg_sha)
            checkpoint = run / 'checkpoints/native/best_auc.pth'
            checkpoint_sha = write(checkpoint, ('weights-' + variant).encode())
            write(run / 'meta/checkpoint.json', dict(path=str(checkpoint), sha256=checkpoint_sha))
            write(run / 'config_requested.yml', cfg.read_bytes())
            for name in ('status.txt', 'start_time.txt', 'end_time.txt', 'eval_val_cw.log', 'eval_val_ow.log'):
                write(run / name, 'train_and_val_complete\n' if name == 'status.txt' else 'fixture\n')
            log = 'fixture\n'
            if variant in ('B', 'F'):
                log += 'BF_LOGIC epoch=10 iter=1 cpf=2.0 oi=0.3 oe=0.2 ae=0.1 weighted=0.006\n'
                write(self.bf / 'experiments/smoke' / (variant + '_s3351.json'),
                      dict(passed=True, config_sha256=cfg_sha, install_sha256=install_sha,
                           logic_manifest_sha256=manifest_sha,
                           individual_visual_gradient_norm=dict(loss_oi=1.0, loss_oe=2.0, loss_ae=3.0)))
            write(run / 'train.log', log)
            for setting in ('cw', 'ow'):
                folder = run / ('eval_val_' + setting)
                write(folder / 'metrics.json', dict(phase='val', seed=3351, variant=variant,
                      setting=setting, dataset='ut-zappos', checkpoint_sha256=checkpoint_sha,
                      metrics_raw=dict(AUC=0.4, best_hm=0.5, best_seen=0.6, best_unseen=0.7)))
                for name in ('config_eval_effective.yml', 'candidates.json', 'image_ids.json', 'labels.npz'):
                    write(folder / name, b'fixture')
        write(self.bf / 'experiments/protocol/seed_3351.json', protocol)
        exp = self.ut / 'experiments/semantic_v2'
        run = exp / 'runs' / core.C_RUN
        cfg = exp / 'configs' / (core.C_RUN + '.yml')
        core.C_REFERENCE['config_sha256'] = write(cfg, 'MODEL:\n  semantic_text: correct\n')
        write(exp / 'config_index_semantic_e10.json', {core.C_RUN: str(cfg)})
        code = {}
        for relative in core.C_SOURCE_REFERENCE:
            code[relative] = write(self.ut / 'src' / relative, native)
            core.C_SOURCE_REFERENCE[relative] = code[relative]
        write(self.ut / 'src/transition_install.json', {})
        write(self.ut / 'src/v2_assets/assets_sha256.json', {})
        write(self.ut / 'prepared.json', {'code_manifest': code})
        bank = self.ut / 'bank/ut-zap50k'
        core.C_REFERENCE['bank_sha256'] = write(bank / 'bank.pt', b'ALSO_NOT_PICKLE')
        core.C_REFERENCE['bank_manifest_sha256'] = write(bank / 'manifest.json', {})
        for name in ('labels.json', 'descriptions.json'):
            write(bank / name, {})
        write(self.ut / 'cache/ut-zap50k/manifest.json', {})
        write(run / 'train.log', 'fixture C_text\n')
        write(run / 'status.json', {})
        write(run / 'meta/provenance.json', {})
        core.C_REFERENCE['checkpoint_sha256'] = write(run / 'checkpoints/best_auc.pth', b'NOT_MODEL')
        for setting in ('cw', 'ow'):
            folder = run / ('eval_val_' + setting)
            write(folder / 'metrics.json', dict(run_id=core.C_RUN, variant='C_text', seed=0, phase='val',
                  setting=setting, best_epoch=6, checkpoint_sha256=core.C_REFERENCE['checkpoint_sha256'],
                  bank_manifest_sha256=core.C_REFERENCE['bank_manifest_sha256'],
                  metrics_raw=dict(AUC=core.C_REFERENCE[setting + '_auc'])))
            for name in ('config_eval_effective.yml', 'candidates.json', 'image_ids.json', 'labels.npz'):
                write(folder / name, b'fixture')
        # Images/test results must not be copied or read by the exporter.
        write(self.ut / 'data/test_images/forbidden.jpg', b'not image')
        write(run / 'eval_test_cw/metrics.json', {'forbidden': True})

    def run_export(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return core.export(self.bf, self.ut, self.out)

    def test_complete_export_read_only_no_weights_or_test(self):
        before = {str(p): core.sha(p) for folder in (self.bf, self.ut)
                  for p in folder.rglob('*') if p.is_file()}
        report = self.run_export()
        self.assertTrue(report['bf_record_checks_passed'])
        self.assertTrue(report['conditional_input_checks_passed'])
        self.assertFalse(report['new_training'])
        self.assertFalse(report['new_evaluation'])
        self.assertFalse(report['conditional']['new_model_implemented'])
        self.assertEqual(report['missing'], [])
        after = {str(p): core.sha(p) for folder in (self.bf, self.ut)
                 for p in folder.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        with zipfile.ZipFile(self.out) as archive:
            names = archive.namelist()
            self.assertFalse(any(name.endswith('.pth') or '/eval_test_' in name for name in names))
            self.assertIn('bf/logic_bank/bank.pt', names)
            self.assertIn('conditional/bank/bank.pt', names)
        self.assertGreater(core.verify_archive(self.out)['files'], 30)

    def test_existing_export_is_never_overwritten(self):
        self.run_export()
        digest = core.sha(self.out)
        with self.assertRaises(FileExistsError):
            self.run_export()
        self.assertEqual(digest, core.sha(self.out))

    def test_missing_log_exports_gap_does_not_claim_pass(self):
        (self.bf / 'experiments/runs/B/train.log').unlink()
        report = self.run_export()
        self.assertFalse(report['bf_record_checks_passed'])
        self.assertTrue(any(item['archive_name'].endswith('B/train.log') for item in report['missing']))
        self.assertTrue(self.out.is_file())

    def test_changed_C_anchor_is_flagged_not_reselected(self):
        write(self.ut / 'experiments/semantic_v2/runs' / core.C_RUN / 'checkpoints/best_auc.pth', b'CHANGED')
        report = self.run_export()
        self.assertFalse(report['conditional_input_checks_passed'])
        self.assertTrue(any(not item['passed'] and item['name'] == 'sha256'
                            for item in report['checks'] if item['section'] == 'conditional'))

    def test_output_inside_source_rejected(self):
        with self.assertRaises(ValueError):
            core.export(self.bf, self.ut, self.bf / 'must_not_create.zip')
        self.assertFalse((self.bf / 'must_not_create.zip').exists())

    def test_escaping_manifest_path_not_read(self):
        outside = self.root / 'outside_secret.txt'
        write(outside, 'NEVER_EXPORT_THIS')
        collector = core.Evidence()
        self.assertFalse(collector.collect(outside, 'bf/escape.txt', self.bf))
        self.assertEqual(collector.files, {})
        self.assertFalse(collector.checks[-1]['passed'])

    def test_corrupted_archive_detected(self):
        self.run_export()
        with zipfile.ZipFile(self.out) as original:
            payloads = {name: original.read(name) for name in original.namelist()}
        payloads['audit.json'] = b'{"tampered":true}'
        damaged = self.root / 'damaged.zip'
        with zipfile.ZipFile(damaged, 'x') as archive:
            for name, data in payloads.items():
                archive.writestr(name, data)
        with self.assertRaises(ValueError):
            core.verify_archive(damaged)

    def test_python38_syntax(self):
        for name in ('export_preflight.py', 'tests.py'):
            ast.parse((Path(__file__).parent / name).read_text(encoding='utf-8'),
                      filename=name, feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main(verbosity=2)
