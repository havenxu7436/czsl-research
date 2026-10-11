#!/usr/bin/env python3
"""CPU numerical/gradient/axis and patched-forward tests. No efficacy claim."""
import argparse
import ast
import importlib.util
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import sha, validate_knowledge, write_json
from install import install, patch_model
from prepare import prepare
from run import verify_protocol
from summarize import summarize

BUNDLE = Path(__file__).resolve().parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    package = types.ModuleType('bf_test_model')
    package.__path__ = [str(BUNDLE / 'model')]
    sys.modules['bf_test_model'] = package
    rules = load_file('bf_test_model.bf_logic_loss', BUNDLE / 'model/logic_loss.py')
    adapter = load_file('bf_test_model.bf_logic', BUNDLE / 'model/bf_logic.py')
    return rules, adapter


class Classifier(nn.Module):
    def forward(self, x, w, scale=True):
        result = F.normalize(x, dim=-1) @ F.normalize(w, dim=-1).t()
        return result / 0.05 if scale else result


class FakeFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 768)

    def forward(self, image):
        cls = self.layer(image)
        patches = cls[:, None, :].expand(-1, 4, -1)
        return patches * 0.7, patches * 0.8, patches * 0.9, patches, cls


class FakeBaseline(nn.Module):
    def __init__(self, ds, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_objs, self.num_attrs, self.word_dim = len(ds.objs), len(ds.attrs), 300
        self.obj_embedder = nn.Embedding(self.num_objs, 300)
        self.attr_embedder = nn.Embedding(self.num_attrs, 300)
        self.tot_objs, self.tot_attrs = torch.arange(self.num_objs), torch.arange(self.num_attrs)
        self.train_objs = torch.tensor([i % self.num_objs for i in range(32)])
        self.train_attrs = torch.tensor([i % self.num_attrs for i in range(32)])
        self.feat_extractor = FakeFeatures()
        self.objc_embedder = nn.Linear(768, 300)
        self.attrc_embedder = nn.Linear(900, 300)
        self.compose = nn.Linear(600, 300)
        self.projection = nn.Linear(600, 300)
        self.classifier = Classifier()

    def compose_visual(self, obj, attr):
        return self.projection(torch.cat((obj, attr), dim=1))


class FakeCache:
    def _setup_kcc(self, ds, cfg):
        pass

    def _kcc_obj_embeddings(self, ids):
        return self.obj_embedder(ids)

    def _kcc_attr_embeddings(self, ids):
        return self.attr_embedder(ids)

    def _kcc_calibrate(self, score, obj, attr):
        return score


def model_from_code(code, adapter, attention):
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name == 'DDM')
    namespace = dict(BaselineDDM=FakeBaseline, KCCCacheMixin=FakeCache, CPFLogicAdapter=adapter,
                     torch=torch, F=F, np=np, choose_texts=attention.choose_texts,
                     choose_visual_tokens=attention.choose_visual_tokens)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual CPF class AST>', 'exec'), namespace)
    return namespace['DDM']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, help='Audited local/server KCC0601_v1 source')
    p.add_argument('--original-logic', type=Path, help='Optional author logicloss.py for numerical comparison')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    rules, adapter_module = modules()
    tests = []
    for file in BUNDLE.rglob('*.py'):
        if '__pycache__' not in file.parts:
            ast.parse(file.read_text(encoding='utf-8'),feature_version=(3,8))
    tests.append('all release scripts parse as Python 3.8 syntax')
    torch.manual_seed(23)
    ids = [0, 0, 1, 1]
    tensors = [torch.randn(*shape, requires_grad=True) for shape in ((7,4),(7,2),(7,4,3),(7,5),(7,5,3))]
    loss = rules.LogicLoss(ids, 2)(*tensors)
    loss['loss_logic_raw'].backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.norm() > 0 for x in tensors)
    tests.append('all five logit families receive finite nonzero gradients')
    zero = torch.zeros(8, 4, requires_grad=True)
    rules.pmean_violation(zero).sum().backward()
    assert torch.isfinite(zero.grad).all() and torch.equal(zero.grad, torch.zeros_like(zero))
    tests.append('zero violation has exact zero value and finite zero gradient')
    if a.original_logic:
        original = load_file('bf_original_logicloss', a.original_logic)
        dataset = types.SimpleNamespace(data_name='ut-zappos', hc_dict2idx={0:[0,1],1:[2,3]})
        expected = original.LogicLoss(dataset)(tensors)
        assert torch.allclose(expected, 0.01 * loss['loss_logic_raw'], atol=1e-8, rtol=1e-6)
        tests.append('numerically matches author logicloss.py p=3 beta=0.01')
    order = torch.tensor([2,0,3,1])
    shuffled = rules.LogicLoss([ids[i] for i in order], 2)(tensors[0][:,order], tensors[1], tensors[2][:,order], tensors[3], tensors[4])
    assert torch.allclose(shuffled['loss_logic_raw'], loss['loss_logic_raw'], atol=1e-7)
    tests.append('aligned object reorder leaves the logic loss invariant')
    # Description slots independently compete across primitive classes, not across slots.
    obj = tensors[0].softmax(dim=1)
    oe = tensors[2].softmax(dim=1)
    direct = rules.pmean_violation(obj[:,:,None] * (1 - oe)).mean()
    assert torch.allclose(direct, loss['loss_oe'])
    tests.append('OE softmax and batch aggregation axes match author code')
    try:
        rules.LogicLoss([0,0], 2)
        raise AssertionError('Empty parent accepted')
    except ValueError:
        tests.append('uncovered/empty parent rejected')
    knowledge = json.loads((BUNDLE / 'assets/ut-zap50k/knowledge.json').read_text(encoding='utf-8'))
    labels = {'attrs': sorted(knowledge['ae']), 'objs': sorted(knowledge['oe'])}
    parents = validate_knowledge(knowledge, labels)
    tests.append('UT knowledge covers exactly 12 objects / 16 attributes / 3 descriptions')
    with tempfile.TemporaryDirectory(prefix='bf_cpu_test_') as td:
        root = Path(td)
        bank = root / 'bank'
        bank.mkdir()
        tensors_bank = {'parent': F.normalize(torch.randn(len(parents),300),dim=-1),
                        'oe': F.normalize(torch.randn(12,3,300),dim=-1),
                        'ae': F.normalize(torch.randn(16,3,300),dim=-1)}
        torch.save(tensors_bank, bank / 'bank.pt')
        object_parent = [next(i for i,p in enumerate(parents) if o in knowledge['oi'][p]) for o in labels['objs']]
        write_json(bank / 'manifest.json', {'labels':labels,'parents':parents,'object_parent':object_parent,
                                          'official_author_data':False,'bank_sha256':sha(bank/'bank.pt')})
        ds = types.SimpleNamespace(**labels)
        cfg = types.SimpleNamespace(MODEL=types.SimpleNamespace(kcc_logic=True,kcc_beta=0.01,
                                    bf_logic_bank=str(bank),bf_logic_q=3.0,bf_allow_reconstructed=True))
        module = adapter_module.CPFLogicAdapter(ds,cfg)
        vobj, vattr = torch.randn(7,300,requires_grad=True), torch.randn(7,300,requires_grad=True)
        classification = Classifier()
        logits_obj = classification(vobj,torch.randn(12,300))
        logits_attr = classification(vattr,torch.randn(16,300))
        result = module(vobj,vattr,logits_obj,logits_attr,classification)
        for term in ('loss_oi','loss_oe','loss_ae'):
            grads = torch.autograd.grad(result[term],(vobj,vattr),retain_graph=True,allow_unused=True)
            assert sum(float(g.square().sum()) for g in grads if g is not None) > 0
        assert torch.equal(result['loss_logic_weighted'],result['loss_logic_raw'] * 0.01)
        assert list(module.parameters()) == [] and module.state_dict() == {}
        tests.append('adapter connects each rule to visual features; beta once; no added parameters/state keys')
        module.beta = 0
        assert module(vobj,vattr,logits_obj,logits_attr,classification)['loss_logic_weighted'].item() == 0
        tests.append('beta=0 adapter exact-zero bypass')
        if a.source:
            copied = root / 'installed'
            install(a.source,copied)
            source_code = (a.source/'models/kcc_cpf.py').read_text(encoding='utf-8')
            attention = load_file('bf_attention',a.source/'models/ddm_utils.py')
            original_model = model_from_code(source_code,adapter_module.CPFLogicAdapter,attention)
            patched_model = model_from_code(patch_model(source_code),adapter_module.CPFLogicAdapter,attention)
            torch.manual_seed(31)
            baseline = original_model(ds,cfg).eval()
            torch.manual_seed(31)
            patched = patched_model(ds,cfg).eval()
            assert all(torch.equal(value,patched.state_dict()[name]) for name,value in baseline.state_dict().items())
            batch = {'img':torch.randn(7,8),'obj':torch.arange(7)%12,
                     'attr':torch.arange(7)%16,'pair':torch.arange(7)%32}
            native = baseline.train_forward(batch)
            patched.bf_logic.beta = 0
            disabled = patched.train_forward(batch)
            assert torch.equal(native['loss_total'],disabled['loss_total'])
            patched.bf_logic.beta = 0.01
            active = patched.train_forward(batch)
            assert torch.equal(native['loss_total'],active['loss_cpf'])
            assert torch.equal(active['loss_total'],active['loss_cpf']+active['loss_logic_weighted'])
            active['loss_total'].backward()
            for model in (baseline,patched):
                model.val_attrs,model.val_objs=model.train_attrs,model.train_objs
                model.val_pairs=list(range(32))
            assert torch.equal(baseline.val_forward(batch)['pred'],patched.val_forward(batch)['pred'])
            tests.append('actual CPF forward AST + tensor-stub backbone: native loss preserved, beta0 exact identity, active backward, inference exact identity')
            assert sha(a.source/'train.py') == json.loads((copied/'bf_install.json').read_text(encoding='utf-8'))['original_guarded_hashes']['train.py']
            tests.append('installer validates original hashes, copies independently, and keeps original train.py unchanged')
            # Full preparation/summary test with synthetic files, explicitly NOT real training evidence.
            fixture_root = root / 'fixture_project'
            fixture_src = fixture_root / 'src/KCC0601_v1'
            shutil.copytree(a.source,fixture_src,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc','*.zip','kcc_mvp_backup*'))
            cache = fixture_root / 'knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k'
            cache.mkdir(parents=True)
            write_json(cache/'labels.json',labels)
            write_json(cache/'manifest.json',{'fixture_only':True})
            write_json(cache/'cache_manifest.json',{'fixture_only':True})
            import yaml
            basic = {'config_name':'ut-zappos-czsl','DATASET':{'name':'ut-zappos','splitname':'compositional-split-natural'},
                     'EVAL':{},'MODEL':{'wordembs':'glove','cosine_cls_temp':0.05},
                     'TRAIN':{'max_epoch':10,'aug_epoch':37}}
            old_configs = fixture_root/'experiments/kcc0601_glove_mvp_v1/configs'
            old_configs.mkdir(parents=True)
            for group,lam in (('A',0.0),('E',0.1)):
                c = json.loads(json.dumps(basic))
                c['MODEL'].update(kcc_text=group=='E',kcc_calib=group=='E',kcc_lambda=lam)
                path=old_configs/f'mvp_ut-zappos_{group}_legacy_s3351_l{lam:g}.yml'
                path.write_text(yaml.safe_dump(c),encoding='utf-8')
            from build_bank import tokenize
            words={w for s in list(knowledge['oi'])+[t for group in ('oe','ae') for texts in knowledge[group].values() for t in texts] for w in tokenize(s)}
            glove=fixture_root/'data/glove/glove.6B.300d.txt'
            glove.parent.mkdir(parents=True)
            glove.write_text(''.join(w+' '+ ' '.join(str(float(x)) for x in torch.randn(300))+'\n' for w in sorted(words)),encoding='utf-8')
            data=fixture_root/'data/ut-zap50k'
            splits=data/'compositional-split-natural'
            splits.mkdir(parents=True)
            for split in ('train','val','test'):
                (splits/(split+'_pairs.txt')).write_text('Canvas Boots.Ankle\n',encoding='utf-8')
            (data/'metadata_compositional-split-natural.t7').write_bytes(b'fixture-only')
            fixture_work=root/'fixture_work'
            prepare(fixture_root,fixture_work,3351,allow_reconstructed=True)
            prepare(fixture_root,fixture_work,3351,allow_reconstructed=True)
            protocol=json.loads((fixture_work/'experiments/protocol/seed_3351.json').read_text(encoding='utf-8'))
            smoke=fixture_work/'experiments/smoke'
            smoke.mkdir()
            for group in ('B','F'):
                write_json(smoke/f'{group}_s3351.json',{'passed':True,'fixture_only':True,
                    'config_sha256':protocol['configs'][group]['sha256'],'install_sha256':protocol['install_sha256'],
                    'logic_manifest_sha256':protocol['logic_bank_manifest_sha256']})
            verify_protocol(fixture_work,3351)
            assert yaml.safe_load(Path(protocol['configs']['F']['path']).read_text(encoding='utf-8'))['MODEL']['kcc_lambda']==0.1
            for group,auc in (('A_ref',0.40),('B',0.41),('E_ref',0.38),('F',0.39)):
                result_dir=Path(protocol['configs'][group]['run'])
                result_dir.mkdir(parents=True)
                (result_dir/'status.txt').write_text('train_and_val_complete\n',encoding='utf-8')
                (result_dir/'train.log').write_text('BF_LOGIC epoch=10 iter=1 cpf=1 oi=.1 oe=.1 ae=.1 weighted=.003\n',encoding='utf-8')
                for setting in ('cw','ow'):
                    write_json(result_dir/f'eval_val_{setting}/metrics.json',{'phase':'val','seed':3351,'variant':group,
                        'metrics_raw':{'AUC':auc,'best_hm':0.5,'best_seen':0.6,'best_unseen':0.6},
                        'best_epoch':10,'checkpoint_sha256':'fixture-only'})
            summarize(fixture_work,[3351],root/'fixture_summary.md')
            summary=(root/'fixture_summary.md').read_text(encoding='utf-8')
            assert '| 3351 | cw | +1.000 | +1.000 | -1.000 |' in summary
            tests.append('synthetic end-to-end fixture: prepare idempotent, protocol verification, original F lambda=0.1, paired summary signs/units correct')
    report = {'passed': True, 'tests': tests, 'n_checks':len(tests), 'device':'CPU',
              'limitations':['synthetic tensor gradients/identity are not real-image CUDA smoke or training results',
                             'full GPU B/F training has not been run by this test']}
    if a.out:
        write_json(a.out,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
