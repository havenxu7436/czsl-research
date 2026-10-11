"""Build a self-contained, hashed release from the audited BF source snapshot."""
import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = HERE / 'reference_source' if (HERE / 'reference_source').exists() else ROOT / 'output/BF_conditional_preflight_20261009/extracted/bf/source'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Ambiguous patch anchor: ' + old[:100])
    return text.replace(old, new, 1)


def patch_model(text):
    text = once(text, 'from .bf_logic import CPFLogicAdapter',
                'from .bf_logic import CPFLogicAdapter\nfrom .full_feature import FullFeatureMixin')
    text = once(text, 'class DDM(KCCCacheMixin, BaselineDDM):',
                'class DDM(FullFeatureMixin, KCCCacheMixin, BaselineDDM):')
    text = once(text, '        self.bf_logic = CPFLogicAdapter(dset, cfg)',
                '        self.bf_logic = CPFLogicAdapter(dset, cfg)\n        self._setup_full(dset, cfg)')
    start = text.index('    def train_forward(self, batch):')
    end = text.index('    def train_forward_augment(', start)
    block = text[start:end]
    block = once(block, '        comp_pred = self._kcc_calibrate(comp_pred, self.train_objs, self.train_attrs)',
                 "        comp_pred, conditional_attr_pred, full_terms = self._full_train(\n"
                 "            vis_obj_new, attr_emb, obj_pred, attr_pred, comp_pred, obj_labels, attr_labels)\n"
                 "        comp_pred = self._kcc_calibrate(comp_pred, self.train_objs, self.train_attrs)\n"
                 "        logic_attr_pred = self._full_logic_attr(comp_pred, attr_pred)")
    block = once(block, 'self.bf_logic(vis_obj_new, attr_emb, obj_pred, attr_pred, self.classifier)',
                 'self.bf_logic(vis_obj_new, attr_emb, obj_pred, logic_attr_pred, self.classifier)')
    block = once(block, "F.cross_entropy(attr_pred, batch['attr'])", "F.cross_entropy(conditional_attr_pred, batch['attr'])")
    block = once(block, '        attr_pred = torch.max(attr_pred, dim=1)[1]',
                 '        attr_pred = torch.max(conditional_attr_pred, dim=1)[1]')
    block = once(block, '        out.update(logic_terms)', '        out.update(logic_terms)\n        out.update(full_terms)')
    text = text[:start] + block + text[end:]
    text = once(text, '        pred = comp_pred + attr_pred + obj_pred',
                '        pred = comp_pred + attr_pred + obj_pred\n        pred = self._full_eval(pred, vis_obj_new, attr_emb)')
    ast.parse(text)
    return text


def patch_cache(text):
    return once(text, '        for flag in ("kcc_feature",):\n            if bool(getattr(cfg.MODEL, flag, False)):\n                raise NotImplementedError(flag + " is not implemented in the A/C/D/E MVP")\n', '')


def patch_train(text):
    text = once(text, 'import argparse\nimport sys\n',
                 "import argparse\nimport sys\nimport hashlib\n\n"
                 "def checkpoint_sha256(path):\n"
                 "    h = hashlib.sha256()\n"
                 "    with open(path, 'rb') as f:\n"
                 "        for block in iter(lambda: f.read(1024 * 1024), b''):\n"
                 "            h.update(block)\n"
                 "    return h.hexdigest()\n")
    text = once(text, "'loss_logic_raw', 'loss_logic_weighted'\n", "'loss_logic_raw', 'loss_logic_weighted', 'full_delta_rms', 'loss_conditional_attr'\n")
    text = once(text, "        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()",
                 "        optimizer.zero_grad()\n        loss.backward()\n"
                 "        if bool(getattr(cfg.MODEL, 'kcc_feature', False)) and idx == 0:\n"
                 "            head_g = sum(float(p.grad.detach().square().sum()) for p in model.full_head.parameters() if p.grad is not None) ** 0.5\n"
                 "            if not np.isfinite(head_g) or head_g <= 0:\n                raise ValueError('Feature head has no finite gradient')\n"
                 "            print('FULL_GRADIENT epoch={} norm={} conditional_ce={} delta_rms={}'.format(epoch, head_g, float(out['loss_conditional_attr'].detach()), float(out['full_delta_rms'].detach())), flush=True)\n"
                 "        optimizer.step()")
    text = once(text, '        train(epoch, model, optimizer, trainloader, logger, device, cfg)',
                 "        epoch_start_time = time.time()\n        train(epoch, model, optimizer, trainloader, logger, device, cfg)")
    text = once(text, '        epoch += 1\n\n    best_path =',
                 "        print('FULL_EPOCH_FINISHED epoch={} seconds={:.2f}'.format(epoch, time.time()-epoch_start_time), flush=True)\n        epoch += 1\n\n    best_path =")
    # Every finished epoch is durable. Restart explicitly resumes model/optimizer/RNG.
    anchor = '    while epoch <= cfg.TRAIN.max_epoch:'
    resume = """    resume_path = os.environ.get('FULL0601_RESUME')
    if resume_path:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        resume_best = os.path.join(ckpt_dir, 'best_auc.pth')
        if not os.path.isfile(resume_best) or checkpoint_sha256(resume_best) != state['best_checkpoint_sha256']:
            raise ValueError('Best checkpoint changed after the last durable epoch; preserve evidence, do not silently resume inconsistent state')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        best_records, best_epoch = state['best_records'], state['best_epoch']
        epoch = state['epoch'] + 1
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        print('FULL_RESUME completed_epoch={} next_epoch={}'.format(state['epoch'], epoch), flush=True)

"""
    text = once(text, anchor, resume + anchor)
    anchor = "        print('FULL_EPOCH_FINISHED epoch={} seconds={:.2f}'.format(epoch, time.time()-epoch_start_time), flush=True)"
    save = """        last_path = os.path.join(ckpt_dir, 'last_resume.pth')
        tmp_path = last_path + '.pending'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best_records': best_records, 'best_epoch': best_epoch,
                    'best_checkpoint_sha256': checkpoint_sha256(os.path.join(ckpt_dir, 'best_auc.pth')),
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, tmp_path)
        os.replace(tmp_path, last_path)
"""
    text = once(text, anchor, save + anchor)
    ast.parse(text)
    return text


def build():
    destination = HERE / 'source'
    if destination.exists():
        raise FileExistsError('Do not replace an existing built source: ' + str(destination))
    shutil.copytree(OLD, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if not (HERE / 'reference_source').exists():
        shutil.copytree(OLD, HERE / 'reference_source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    old_hashes = {str(p.relative_to(OLD)).replace('\\', '/'): sha(p) for p in OLD.rglob('*') if p.is_file()}
    for name, patch in (('models/kcc_cpf.py', patch_model), ('models/kcc_cache.py', patch_cache), ('train.py', patch_train)):
        target = destination / name
        target.write_text(patch(target.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
    shutil.copyfile(HERE / 'full_feature.py', destination / 'models/full_feature.py')
    shutil.copyfile(HERE / 'smoke_full.py', destination / 'tests/smoke_full.py')
    assets = HERE / 'assets'
    assets.mkdir()
    shutil.copytree(ROOT / 'output/BF_conditional_preflight_20261009/extracted/bf/logic_bank', assets / 'logic_bank')
    for variant in ('A_ref', 'B', 'E_ref', 'F'):
        origin = ROOT / 'output/BF_conditional_preflight_20261009/extracted/bf/runs' / variant / 'config_requested.yml'
        shutil.copyfile(origin, assets / (variant + '.yml'))
    shutil.copyfile(ROOT / 'output/BF_conditional_preflight_20261009/extracted/bf/runs/F/meta/cache_manifest.json', assets / 'legacy_cache_manifest.json')
    (assets / 'source_origin.json').write_text(json.dumps(old_hashes, indent=2), encoding='utf-8')
    expected = {str(p.relative_to(HERE)).replace('\\', '/'): sha(p) for p in sorted(HERE.rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and p.name != 'release_manifest.json'}
    (HERE / 'release_manifest.json').write_text(json.dumps(expected, indent=2), encoding='utf-8')
    archive = ROOT / 'output/0601_priority_release_20261010.zip'
    if archive.exists():
        raise FileExistsError(str(archive))
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(expected):
            z.write(HERE / name, HERE.name + '/' + name)
        z.write(HERE / 'release_manifest.json', HERE.name + '/release_manifest.json')
    archive.with_suffix(archive.suffix + '.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='utf-8')
    print('RELEASE_BUILT', archive, 'sha256', sha(archive))


if __name__ == '__main__':
    build()
