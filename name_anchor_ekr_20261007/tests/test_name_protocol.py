import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from transition import name_plan as p


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        root=Path(self.temp.name)
        self.work=root/'new'; self.previous=root/'previous'
        self.checkpoint=self.previous/'experiments/semantic_v2/runs/semantic_ut-zappos_A_unit_s0_e10/checkpoints/best_auc.pth'
        self.checkpoint.parent.mkdir(parents=True);self.checkpoint.write_bytes(b'fake-checkpoint-for-guard-test')
        self.bank=self.previous/'bank/ut-zap50k'
        self.bank.mkdir(parents=True);(self.bank/'manifest.json').write_text('{}')
        (self.work/'src').mkdir(parents=True)
        p.write_json(self.work/'src/name_reference.json',{'bank_manifest_sha256':p.sha(self.bank/'manifest.json')})
        self.protocol={'work':str(self.work),'previous':str(self.previous),
            'frozen_inputs_sha256':{str(self.checkpoint):p.sha(self.checkpoint)}}
        p.write_json(p.protocol_path(self.work),self.protocol)
        reference=json.loads((Path(__file__).resolve().parents[1]/'reference.json').read_text(encoding='utf-8'))
        self.template=copy.deepcopy(reference['base_effective_config'])
        self.template['MODEL'].update(semantic_text='unit',kcc_text=True,branch_mode='ekr',
            branch_regime='frozen',branch_anchor=str(self.checkpoint),
            branch_anchor_sha256=p.sha(self.checkpoint),description_bank=str(self.bank))
        self.template['TRAIN'].update(seed=0,kcc_variant='N_clean',name_protocol_file=str(p.protocol_path(self.work)),
            name_protocol_sha256=p.sha(p.protocol_path(self.work)))
        self.cfg_path=self.work/'configs/frozen.yml'
        self.cfg_path.parent.mkdir(parents=True)
        self.cfg_path.write_text(yaml.safe_dump(self.template),encoding='utf-8')
        p.write_json(self.work/'protocol/config_index.json',{
            p.rid('N_clean',0):{'config':str(self.cfg_path),'sha256':p.sha(self.cfg_path)}})
        self.cfg=p.read_cfg(self.cfg_path)
        self.replay=self.work/'diagnostics/full_scores/replay_audit.json'
        p.write_json(self.replay,{'passed':True,'protocol_sha256':p.sha(p.protocol_path(self.work)),
            'inputs_sha256':{}})
        self.protocol_patch=patch.object(p,'validate_protocol',return_value=self.protocol)
        self.bank_patch=patch.object(p,'load_bank',return_value=({},{}))
        self.protocol_patch.start();self.bank_patch.start()

    def tearDown(self):
        self.protocol_patch.stop();self.bank_patch.stop();self.temp.cleanup()

    def test_real_yacs_config_and_replay_unlock(self):
        self.assertEqual(p.validate_config(self.cfg,require_replay=True),self.protocol)

    def test_wrong_same_seed_anchor_rejected(self):
        self.cfg.MODEL.branch_anchor=str(self.checkpoint.parent/'another_seed.pth')
        with self.assertRaisesRegex(ValueError,'same-seed'):
            p.validate_config(self.cfg)

    def test_failed_replay_cannot_unlock_training(self):
        result=p.read_json(self.replay);result['passed']=False;p.write_json(self.replay,result)
        with self.assertRaisesRegex(ValueError,'replay audit'):
            p.validate_config(self.cfg,require_replay=True)

    def test_changed_effective_config_rejected(self):
        self.cfg.MODEL.branch_eta_max=0.2
        with self.assertRaisesRegex(ValueError,'Effective config'):
            p.validate_config(self.cfg)

    def test_changed_full_score_evidence_rejected(self):
        result=p.read_json(self.replay);result['inputs_sha256']={'tampered':'x'};p.write_json(self.replay,result)
        with self.assertRaisesRegex(ValueError,'diagnostic input'):
            p.validate_config(self.cfg,require_replay=True)


