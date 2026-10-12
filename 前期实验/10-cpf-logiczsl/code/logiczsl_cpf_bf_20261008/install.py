#!/usr/bin/env python3
"""Copy the verified original ACDE implementation; patch only the new copy."""
import argparse
import ast
import json
import shutil
from pathlib import Path
from common import sha, write_json

BUNDLE = Path(__file__).resolve().parent
EXPECTED = {
    'models/kcc_cache.py': 'e7999b7e227323713995d4cf44e8041dc9dce77046bc27520525f60a1d0766f8',
    'models/kcc_cpf.py': 'eaee77a0bf06f02dd653e5e60b7a07865c8cd24e6b8d9911aaE696e9d97aab65'.lower(),
    'train.py': 'a3ec0e837ade53cb40777d799387a7f4a8261ce382246d7ef18dab75848592ed',
    'tools/eval_mvp.py': 'db155c499e93e9ec8e37ac6be196e0357431c99fbd7b604ed5ec82089f3a8af6',
    'models/ddm.py': 'd1ebcc4e4a40070ca05ef06520a096c82c0ab02c2388a38698ff4f966d656055',
    'models/ddm_utils.py': 'a9a2501ed80b6799bc4d021f276b7db241e01930785d079814bc6562332cf348',
    'dataset.py': '430ac2b7f388ce28dd84e1218a7e3180b87d4620a86f9ab339240f204948086f',
    'evaluator_ge.py': '24b48c4d48e3475baec0a656fca194cdbc6d4b4e32e87c0f22cda3daf586d978',
    'config/defaults.py': '3838d4525a2d8dbf1714675cdcd141fb554c44a5acafa53f6b015199604f4bb3'
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Patch anchor must occur exactly once: ' + old[:100])
    return text.replace(old, new, 1)


def patch_model(text):
    text = replace_once(text, 'from .kcc_cache import KCCCacheMixin',
                        'from .kcc_cache import KCCCacheMixin\nfrom .bf_logic import CPFLogicAdapter')
    text = replace_once(text, '        self._setup_kcc(dset, cfg)',
                        '        self._setup_kcc(dset, cfg)\n        self.bf_logic = CPFLogicAdapter(dset, cfg)')
    # Scope transformations to training methods; no inference score change.
    start = text.index('    def train_forward(self, batch):')
    aug = text.index('    def train_forward_augment(', start)
    val = text.index('    def val_forward(', aug)
    blocks = [text[start:aug], text[aug:val]]
    changed = []
    for block, visual in zip(blocks, ('vis_obj_new', 'vis_obj_all')):
        block = replace_once(block, '        # obj loss and accuracy',
            '        logic_terms = self.bf_logic(' + visual + ', attr_emb, obj_pred, attr_pred, self.classifier)\n\n'
            '        # obj loss and accuracy')
        block = replace_once(block, "        out = {\n            'loss_total': loss,", 
            "        loss_cpf = loss\n        loss = loss + logic_terms['loss_logic_weighted']\n"
            "        out = {\n            'loss_total': loss,\n            'loss_cpf': loss_cpf,")
        block = replace_once(block, '        return out', '        out.update(logic_terms)\n        return out')
        changed.append(block)
    return text[:start] + ''.join(changed) + text[val:]


def patch_train(text):
    text = replace_once(text, "        'loss_total'\n", 
        "        'loss_total', 'loss_cpf', 'loss_oi', 'loss_oe', 'loss_ae', 'loss_logic_raw', 'loss_logic_weighted'\n")
    text = replace_once(text, "        loss = out['loss_total']\n", 
        "        loss = out['loss_total']\n"
        "        if not torch.isfinite(loss):\n            raise ValueError('Nonfinite training loss')\n"
        "        if bool(getattr(cfg.MODEL, 'kcc_logic', False)) and cfg.MODEL.kcc_beta > 0:\n"
        "            if not out['loss_logic_raw'].requires_grad:\n                raise ValueError('Logic loss disconnected')\n"
        "            if idx == 0 or (idx + 1) % cfg.TRAIN.disp_interval == 0:\n"
        "                print('BF_LOGIC epoch={} iter={} cpf={:.8g} oi={:.8g} oe={:.8g} ae={:.8g} weighted={:.8g}'.format(\n"
        "                    epoch, idx + 1, *(float(out[k].detach()) for k in ('loss_cpf','loss_oi','loss_oe','loss_ae','loss_logic_weighted'))), flush=True)\n")
    text = replace_once(text, '    seed = cfg.TRAIN.seed\n    np.random.seed(seed)',
                        '    seed = cfg.TRAIN.seed\n    random.seed(seed)\n    np.random.seed(seed)')
    text = replace_once(text, '    torch.backends.cudnn.benchmark = True',
                        '    torch.backends.cudnn.benchmark = False  # Same reproducibility rule for A_ref/B/E_ref/F.')
    return text


def install(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Separate, nonnested source/destination required')
    if destination.exists():
        raise FileExistsError('Source copy exists; do not overwrite: ' + str(destination))
    original_hashes = {}
    for name, expected in EXPECTED.items():
        actual = sha(source / name)
        if actual != expected:
            raise ValueError('Source version differs from audited ACDE source: ' + name + ' sha256=' + actual)
        original_hashes[name] = actual
    patches = {
        'models/kcc_cache.py': replace_once((source / 'models/kcc_cache.py').read_text(encoding='utf-8'),
                                          'for flag in ("kcc_logic", "kcc_feature"):', 'for flag in ("kcc_feature",):'),
        'models/kcc_cpf.py': patch_model((source / 'models/kcc_cpf.py').read_text(encoding='utf-8')),
        'train.py': patch_train((source / 'train.py').read_text(encoding='utf-8'))
    }
    for name, text in patches.items():
        ast.parse(text, filename=name)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc',
                      'kcc_mvp_backup*', 'kcc_mvp_step3', '*.zip'))
    for name, text in patches.items():
        with (destination / name).open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
    shutil.copyfile(BUNDLE / 'model/logic_loss.py', destination / 'models/bf_logic_loss.py')
    shutil.copyfile(BUNDLE / 'model/bf_logic.py', destination / 'models/bf_logic.py')
    shutil.copyfile(BUNDLE / 'smoke.py', destination / 'tests/smoke_bf.py')
    # The unchanged eval_mvp.py loads models.kcc_cpf and thus uses the same new copy.
    outputs = {str(p.relative_to(destination)).replace('\\', '/'): sha(p)
               for p in sorted(destination.rglob('*')) if p.is_file()}
    write_json(destination / 'bf_install.json', {
        'release': BUNDLE.name, 'source': str(source), 'original_guarded_hashes': original_hashes,
        'output_hashes': outputs, 'inference_changed': False,
        'notes': ['Logic is training-only; no extra trainable parameters.',
                  'Python RNG seeded and cudnn benchmark disabled for all new paired groups.',
                  'Legacy tools/run_mvp.sh is not used for B/F.']
    })
    print('INSTALLED_SEPARATE_COPY', destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    install(a.source, a.destination)
