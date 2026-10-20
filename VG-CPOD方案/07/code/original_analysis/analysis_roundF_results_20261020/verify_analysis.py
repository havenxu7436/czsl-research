"""Cross-check the additional F1 logs, server summaries and mechanism calculations."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'server_results_roundF_20261020'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    data=read(OUT/'analysis.json');mechanism=read(OUT/'mechanism.json')
    summary=read(ARCHIVE/'roundF_20261020/round_f_summary.json')
    manifest=read(ROOT/'server_results_20261018/vgcpod_v1_runs/cache_ut_clip_l14/features_manifest.json')
    assert len(manifest['pairs']['train'])==83
    assert len(set(map(tuple,manifest['pairs']['train']))|set(map(tuple,manifest['pairs']['test'])))==101
    errors=dict(support_coefficient=0.,pair_weighted=0.,support_weighted=0.,batch_residual=0.)
    for seed in (3351,3352,3353):
        run=ARCHIVE/'roundF_20261020/F1/ut-zap50k/V1'/('seed%d'%seed)
        for entry in read(run/'history.json'):
            train=entry['train'];coefficient=1.5*.05/(1+.05)
            errors['support_coefficient']=max(errors['support_coefficient'],abs(train['support_weight']-coefficient))
            errors['pair_weighted']=max(errors['pair_weighted'],abs(train['pair_weighted']-1.5*train['pair']))
            errors['support_weighted']=max(errors['support_weighted'],abs(train['support_weighted']-coefficient*train['support']))
            errors['batch_residual']=max(errors['batch_residual'],train['task_reconstruction_error'])
        for setting in ('val_cw','val_ow','test_cw','test_ow'):
            native=next(r for r in data['metrics'] if r['case']=='F1' and r['seed']==seed and r['setting']==setting)
            server=next(r for r in summary['runs'] if r['seed']==seed)
            for key in ('AUC_native','AUC_exact','HM_at_val_bias','gt_object_attr_acc_unseen','support_mass_unseen'):
                assert abs(native[key]-server[setting+'_'+key])<1e-12
        for case in ('E1','F1'):
            a=next(r for r in data['diagnostics'] if r['case']==case and r['seed']==seed and r['phase']=='test')
            b=next(r for r in mechanism['diagnostics'] if r['case']==case and r['seed']==seed and r['phase']=='test' and r['group']=='unseen')
            assert abs(a['unseen_joint_ceiling']-b['unseen_rank_ceiling'])<1e-9
    assert errors['support_coefficient']<1e-14
    assert errors['pair_weighted']<1e-12 and errors['support_weighted']<1e-12
    assert errors['batch_residual']<1e-6
    release=read(ARCHIVE/'vgcpod_v1_roundF/release_manifest.json')
    for name,expected in release['files_sha256'].items():
        assert hashlib.sha256((ARCHIVE/'vgcpod_v1_roundF'/name).read_bytes()).hexdigest()==expected,name
    result=dict(max_train_log_errors=errors,server_summary_matches_recomputed_rows=True,
        ranking_ceiling_matches_exact_evaluator=True,release_files_verified=len(release['files_sha256']),
        planned_beta075_support_weight=1.5*.075/1.075,
        planned_cases=dict(G1=dict(rho=.15,beta=.075,gate_input='local',gate_eta=.25),
                           G0=dict(rho=.15,beta=.075,gate_input='none')))
    (OUT/'supplementary_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
