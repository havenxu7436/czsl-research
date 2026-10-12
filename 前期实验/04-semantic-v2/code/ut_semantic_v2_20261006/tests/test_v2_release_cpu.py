"""Local release test using real UT metadata and source, but synthetic GloVe vectors.

This does NOT validate real server embeddings, image files, CUDA, or performance.
Only torchvision's unused transforms import is stubbed for parse_split on CPU.
The original dataset.parse_split and original cache builder run unmodified.
"""
import ast
import contextlib
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
from transition.bank import construct, derangement
from transition.common import read_json, sha
from transition.v2_prepare import native_identity, verify_assets


class V2Release(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="ut-v2-release-")
        cls.work = Path(cls.temp.name)
        cls.root = cls.work / "source_root"
        cls.backup = Path(sys.argv[1]).resolve()
        cls.project = cls.work / "src"
        subprocess.run([sys.executable, str(BUNDLE / "install.py"),
            "--source", str(cls.backup / "src/KCC0601_v1"), "--destination", str(cls.project)], check=True)
        # Load the installed module with its own package path; verify installed code rather than bundle-only code.
        sys.path.insert(0, str(cls.project))
        for name in list(sys.modules):
            if name == "transition" or name.startswith("transition."):
                del sys.modules[name]
        import transition.v2_prepare as prep
        cls.prep = prep
        assets, source, labels, review, groups = prep.verify_assets(cls.project)
        cls.assets, cls.source, cls.labels = assets, source, labels
        data = cls.root / "data/ut-zap50k"
        shutil.copytree(cls.backup / "data/ut-zap50k/compositional-split-natural", data / "compositional-split-natural")
        shutil.copy2(cls.backup / "data/ut-zap50k/metadata_compositional-split-natural.t7", data)
        old_source = cls.root / "knowledge/kcc0601_v1/trusted_v1/ut-zap50k"
        shutil.copytree(cls.backup / "knowledge/kcc0601_v1/trusted_v1/ut-zap50k", old_source)
        run = cls.root / "experiments/kcc0601_glove_mvp_v1/runs/mvp_ut-zappos_C_legacy_s3351_l0"
        run.mkdir(parents=True)
        shutil.copy2(cls.backup / "experiments/kcc0601_glove_mvp_v1/runs/mvp_ut-zappos_C_legacy_s3351_l0/train.log", run)
        builder = prep.builder_module(cls.project)
        cls.builder = builder
        aliases, token_aliases = builder.baseline_aliases(cls.project / "models/word_embedding_utils.py")
        needed = set(t for seq in builder.FILLERS for t in seq)
        for axis in ("attrs", "objs"):
            for label in labels[axis]:
                needed.update(builder.name_tokens(label, aliases, token_aliases))
        for knowledge in (source, old_source):
            for name in ("ae_dict.json", "oe_dict.json"):
                for texts in read_json(knowledge / name).values():
                    for text in texts:
                        needed.update(builder.description_tokens(text))
        rng = np.random.RandomState(9)
        glove_path = cls.root / "data/glove/glove.6B.300d.txt"
        glove_path.parent.mkdir(parents=True)
        with glove_path.open("w", encoding="utf-8", newline="\n") as f:
            for token in sorted(needed):
                vector = rng.normal(size=300).astype(np.float32)
                f.write(token + " " + " ".join(format(float(x), ".8g") for x in vector) + "\n")
        cls.glove_path = glove_path
        # The test never constructs image transforms; original parse_split only reads pair lists.
        tv = types.ModuleType("torchvision")
        tf = types.ModuleType("torchvision.transforms")
        tv.transforms = tf
        cls.saved_tv = {k: sys.modules.get(k) for k in ("torchvision", "torchvision.transforms")}
        sys.modules["torchvision"], sys.modules["torchvision.transforms"] = tv, tf
        cfg = cls.work / "old_builder_cfg.yml"
        cfg.write_bytes(yaml.safe_dump(read_json(assets / "protocol/base_effective_config.json")).encode())
        torch.set_num_threads(1)
        builder.build(SimpleNamespace(project_root=cls.project, cfg=cfg, data_root=data,
            knowledge_dir=old_source, output_dir=cls.root / "knowledge/kcc0601_v1/cache_trusted_v1/ut-zap50k",
            glove_file=glove_path, descriptions_per_class=3, prior_source="template",
            knowledge_status="bulk_trusted_unverified", seed=91001))

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.saved_tv.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        cls.temp.cleanup()

    def test_01_review_and_real_metadata(self):
        assets, source, labels, review, groups = self.prep.verify_assets(self.project)
        gallery = read_json(assets / "review_evidence/manifest.json")
        data = self.root / "data/ut-zap50k"
        for name in ("metadata_compositional-split-natural.t7", "compositional-split-natural/train_pairs.txt"):
            expected = gallery["inputs_sha256"]["/data/xhw/czsl_repro/data/ut-zap50k/" + name]
            self.assertEqual(sha(data / name), expected)
        records = torch.load(data / "metadata_compositional-split-natural.t7", map_location="cpu")
        train = {r["image"]:r for r in records if r["set"] == "train"}
        self.assertEqual(len(train), 22998)
        unique = set()
        for group in groups:
            with Image.open(assets / "review_evidence" / group["board"]) as board:
                board.load()
            for row in group["selected"]:
                self.assertIn(row["image"], train)
                self.assertEqual((row["attr"], row["obj"]), (train[row["image"]]["attr"], train[row["image"]]["obj"]))
                unique.add(row["image"])
        self.assertEqual(len(unique), 528)
        self.assertEqual(review["decisions"], {"keep":30,"rewrite":54})

    def test_02_end_to_end_synthetic_encoder(self):
        # Native image files are not in the downloaded result backup. They are checked on the server.
        original = self.prep.data_integrity
        self.prep.data_integrity = lambda root, assets, decode: {"local_test_skip": "Native train/val image files absent locally"}
        try:
            self.prep.prepare(self.project, self.root, self.work, decode_train=False)
        finally:
            self.prep.data_integrity = original
        frozen, reports = self.prep.verify_prepared(self.project, self.root, self.work)
        self.assertEqual(len(frozen["config_sha256"]),12)
        self.assertEqual(reports["a"]["OOV_fraction"],0)
        self.assertEqual(reports["o"]["OOV_fraction"],0)
        before = sha(self.work / "prepared.json")
        self.prep.prepare(self.project, self.root, self.work)
        self.assertEqual(before, sha(self.work / "prepared.json"))

    def test_03_tampered_knowledge_rejected(self):
        path = self.source / "ae_dict.json"
        before = path.read_bytes()
        try:
            path.write_bytes(before.replace(b"woven canvas fabric", b"invalid altered text"))
            with self.assertRaisesRegex(ValueError,"asset changed"):
                self.prep.verify_assets(self.project)
        finally:
            path.write_bytes(before)

    def test_04_tampered_config_rejected(self):
        path = self.work / "experiments/semantic_v2/configs/semantic_ut-zappos_C_text_s0_e10.yml"
        before = path.read_bytes()
        cfg = yaml.safe_load(before)
        cfg["TRAIN"]["lr"] = 0.02
        try:
            path.write_bytes(yaml.safe_dump(cfg).encode())
            with self.assertRaisesRegex(ValueError, "Native protocol differs"):
                self.prep.config_integrity(self.project,self.root,self.work)
        finally:
            path.write_bytes(before)

    def test_05_fallback_and_shuffle_control(self):
        from transition.bank import construct, derangement
        labels = ["left","right"]
        raw = torch.randn(2,300)
        glove = {"visible":torch.randn(300),"surface":torch.randn(300)}
        table, records = construct(labels,{"left":["missingword","visible","surface"],
            "right":["surface","visible surface","visible"]},raw,glove,self.builder,[1,0])
        self.assertTrue(records[0]["fallback"])
        self.assertTrue(torch.equal(table["shuffle_units"][:,0],table["units"][:,0]))
        self.assertTrue(torch.equal(table["shuffle_units"][:,1:],table["units"][[1,0],1:]))
        self.assertLess(float(table["residual"].sum(1).abs().max()),1e-6)
        self.assertFalse(any(i==p for i,p in enumerate(derangement(16,1729))))

    def test_06_yacs_log_format_and_real_config_merges(self):
        from transition.runtime import read_cfg
        from transition.v2_checks import verify_native_log
        index = read_json(self.work / "experiments/semantic_v2/config_index_semantic_e10.json")
        folder = self.work / "log_format_test"
        folder.mkdir()
        for path in index.values():
            cfg = read_cfg(path)
            (folder / "train.log").write_bytes((str(cfg)+'\nRandom seed: 0\n').encode())
            verify_native_log(folder,cfg)
        bad = cfg.clone()
        bad.TRAIN.lr = 0.02
        with self.assertRaisesRegex(ValueError,"Native logged training config differs"):
            verify_native_log(folder,bad)

    def test_07_post_audit_and_paired_identity_checks(self):
        from transition.runtime import read_cfg
        from transition.v2_checks import post_audit, paired_bias0
        from transition.common import write_json, code_manifest
        from itertools import product
        import hashlib
        index = read_json(self.work / "experiments/semantic_v2/config_index_semantic_e10.json")
        data = self.root / "data/ut-zap50k"
        metadata = torch.load(data / "metadata_compositional-split-natural.t7", map_location="cpu")
        gold = [r for r in metadata if r["set"]=='val']
        aidx = {v:i for i,v in enumerate(self.labels["attrs"])}
        oidx = {v:i for i,v in enumerate(self.labels["objs"])}
        cw = sorted(set(tuple(s.split()) for split in ('train','val','test')
                       for s in (data/'compositional-split-natural'/(split+'_pairs.txt')).read_text().splitlines()))
        for rid,path in index.items():
            cfg=read_cfg(path)
            run=Path(cfg.TRAIN.kcc_run_dir)
            (run/'meta').mkdir(parents=True)
            (run/'checkpoints').mkdir()
            # Deliberately synthetic checkpoint bytes and metrics: only the audit mechanism is tested.
            checkpoint=run/'checkpoints/best_auc.pth'
            checkpoint.write_bytes(b'LOCAL SYNTHETIC AUDIT FIXTURE; NOT TRAINED WEIGHTS')
            digest=sha(checkpoint)
            (run/'effective_config.yml').write_bytes(cfg.dump().encode())
            (run/'train.log').write_bytes((str(cfg)+'\nRandom seed: '+str(cfg.TRAIN.seed)+'\n').encode())
            write_json(run/'status.json',{'status':'train_and_val_complete'})
            write_json(run/'train_complete.json',{'epochs':10,'best_epoch':6,'checkpoint_sha256':digest})
            write_json(run/'meta/code_manifest.json',code_manifest(self.project))
            shutil.copy2(self.work/'cache/ut-zap50k/manifest.json',run/'meta/cache_manifest.json')
            shutil.copy2(self.work/'bank/ut-zap50k/manifest.json',run/'meta/bank_manifest.json')
            for setting in ('cw','ow'):
                folder=run/('eval_val_'+setting)
                folder.mkdir()
                pairs=cw if setting=='cw' else list(product(self.labels['attrs'],self.labels['objs']))
                pidx={tuple(p):i for i,p in enumerate(pairs)}
                target=np.array([pidx[r['attr'],r['obj']] for r in gold])
                pred=target.copy()
                pred[::3]=(pred[::3]+1)%len(pairs)
                if cfg.TRAIN.kcc_variant=='C_text':
                    pred[::6]=target[::6]
                np.savez_compressed(folder/'labels.npz',attr=np.array([aidx[r['attr']] for r in gold]),
                    obj=np.array([oidx[r['obj']] for r in gold]),pair=target,top1_at_bias0=pred)
                write_json(folder/'image_ids.json',[r['image'] for r in gold])
                write_json(folder/'candidates.json',pairs)
                eval_cfg=cfg.clone()
                eval_cfg.EVAL.open_world=setting=='ow'
                write_json(folder/'metrics.json',{'phase':'val','setting':setting,'variant':cfg.TRAIN.kcc_variant,
                    'seed':int(cfg.TRAIN.seed),'checkpoint_sha256':digest,'training_regime':'native_semantic',
                    'bank_manifest_sha256':sha(self.work/'bank/ut-zap50k/manifest.json'),
                    'config_sha256':hashlib.sha256(eval_cfg.dump().encode()).hexdigest(),
                    'n_images':len(gold),'n_candidates':len(pairs),
                    'metrics_raw':{'AUC':0.4,'best_hm':0.5,'best_seen':0.6,'best_unseen':0.5}})
        out=self.work/'post_audit_test.json'
        post_audit(self.project,self.root,self.work,out)
        audit=read_json(out)
        self.assertTrue(audit['integrity_passed'])
        self.assertEqual(len(audit['paired_bias0_errors']['comparisons']),18)
        path=Path(next(iter(index.values())))
        cfg=read_cfg(path)
        target=Path(cfg.TRAIN.kcc_run_dir)/'eval_val_cw/image_ids.json'
        before=target.read_bytes()
        ids=read_json(target)
        ids[0],ids[1]=ids[1],ids[0]
        try:
            write_json(target,ids)
            with self.assertRaisesRegex(ValueError,'Paired validation|VAL image order'):
                paired_bias0(index,self.root)
        finally:
            target.write_bytes(before)

    def test_08_future_ekr_uses_v2_anchors_and_separate_directory(self):
        from transition.configs import make
        from transition.common import write_json
        gate = self.work/'fixture_gate_true.json'
        write_json(gate,{'test_fixture_only':True,'datasets':{'ut-zappos':{'proceed':True}}})
        out = self.work/'experiments/ekr_v2_pilot'
        args = SimpleNamespace(root=self.root,exp=out,legacy_exp=self.root/'experiments/kcc0601_glove_mvp_v1',
            bank_root=self.work/'bank',cache_root=self.work/'cache',semantic_exp=self.work/'experiments/semantic_v2',
            stage='branch',datasets=['ut-zappos'],seeds=[0],epochs=3,variants=['R0','R3','R5','R7'],
            use_legacy_anchor=False,regime='frozen',smoke_only=False,gate=gate,transfer_from=None)
        make(args)
        index=read_json(out/'config_index_branch_e3.json')
        self.assertEqual(len(index),4)
        anchors=set()
        for path in index.values():
            cfg=yaml.safe_load(Path(path).read_text())
            self.assertEqual(cfg['MODEL']['kcc_cache'],str(self.work/'cache/ut-zap50k'))
            self.assertEqual(cfg['MODEL']['description_bank'],str(self.work/'bank/ut-zap50k'))
            self.assertEqual(cfg['MODEL']['branch_regime'],'frozen')
            self.assertEqual(cfg['TRAIN']['max_epoch'],3)
            anchors.add(cfg['MODEL']['branch_anchor'])
            self.assertIn(out,Path(cfg['TRAIN']['kcc_run_dir']).parents)
        self.assertEqual(anchors,{str(self.work/'experiments/semantic_v2/runs/semantic_ut-zappos_C_text_s0_e10/checkpoints/best_auc.pth')})
        write_json(gate,{'test_fixture_only':True,'datasets':{'ut-zappos':{'proceed':False}}})
        with self.assertRaisesRegex(ValueError,'screening gate did not pass'):
            make(args)


if __name__ == "__main__":
    if len(sys.argv)<2:
        raise SystemExit("Pass the full week5 local backup files directory")
    backup = sys.argv[1]
    # Keep backup available for setup while avoiding unittest interpreting it as a test name.
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(V2Release)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
