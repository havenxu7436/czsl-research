"""Seal pilot inputs from the independently audited Stage A export."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-a', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if (root / 'reference.json').exists():
        raise FileExistsError('Preserve existing fixed pilot reference')
    def J(path):
        return json.loads(path.read_text(encoding='utf-8'))
    audit = J(args.audit)
    if not audit['passed'] or audit['independent_native_metric_checks'] != 12:
        raise ValueError('Independent Stage A audit not passed')
    reference = J(args.stage_a / 'release/reference.json')
    reference['schema'] = 'conditional_attribute_pilot_v1'
    manifest = J(args.stage_a / 'manifest.json')['files']
    for name, spec in manifest.items():
        if sha(args.stage_a / name) != spec['sha256']:
            raise ValueError('Stage A extracted bytes changed')
    if sha(root / 'head.py') != manifest['release/head.py']['sha256']:
        raise ValueError('Do not change the audited head for this pilot')
    prepared = J(args.stage_a / 'stageA/prepared.json')
    reference['stageA_base'] = '/var/tmp/czsl_conditional_stageA_20261009/base'
    reference['stageA_files'] = {reference['stageA_base'] + '/' + name: digest
                                for name, digest in prepared['files_sha256'].items()}
    reference['stageA_files'][reference['stageA_base'] + '/prepared.json'] = sha(args.stage_a / 'stageA/prepared.json')
    reference['control_protocol'] = J(args.stage_a / 'stageA/control_protocol.json')
    reference['stageA_zip_sha256'] = '72709f5f7bb39c5494caa596ceff1314006d1ab0ebfc91ce7c0e7d71e196a0c3'
    reference['independent_stageA_audit_sha256'] = sha(args.audit)
    reference['audited_head_sha256'] = sha(root / 'head.py')
    reference['pilot_settings'] = dict(seed=0, epochs=10, batch_size=256, rank=32,
                                      temperature=0.05, cap=0.10, lr=0.001, weight_decay=0.0001,
                                      train_mechanics_epochs=1, train_holdout_fraction=0.20,
                                      fold_id='sha256(image_id), sorted within TRAIN pair; lowest 20%, min 1',
                                      data_order_seed_base=21009000,
                                      static_prior=0, logic=0, backbone_optimizer_steps=0,
                                      selection='first strict maximum VAL CW AUC over epochs 0..10')
    reference['budget_gate'] = dict(pk_minus_baseline_cw_auc_pp=0.20, pk_minus_name_cw_auc_pp=0.10,
                                  pk_minus_mean_permutations_cw_auc_pp=0.10,
                                  cw_hm_floor_pp=-0.5, ow_auc_floor_pp=-0.5, ow_best_unseen_floor_pp=-2.0,
                                  gate_is_budget_rule_not_significance=True)
    (root / 'reference.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (root / 'stageA_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('PILOT_REFERENCE_BUILT')


if __name__ == '__main__':
    main()
