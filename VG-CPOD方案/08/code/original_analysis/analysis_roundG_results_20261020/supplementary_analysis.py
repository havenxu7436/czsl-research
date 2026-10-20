"""Verify G summaries and show that fixed gate means can be absorbed by static CP."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'server_results_roundG_20261020'
CACHE=ROOT/'server_results_20261018/vgcpod_v1_runs/cache_ut_clip_l14'
sys.path.insert(0,str(ARCHIVE/'vgcpod_v1_roundG'))
from ranking_diagnostics import diagnose,METRICS

torch.set_num_threads(1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    spec=importlib.util.spec_from_file_location('g_prediction_helpers',ROOT/'analysis_20261018/analyze_results.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    data=read(OUT/'analysis.json');server=read(ARCHIVE/'roundG_20261020/round_g_summary.json')
    ranking=read(ARCHIVE/'roundG_20261020/ranking_diagnostics.json')
    manifest=read(CACHE/'features_manifest.json')
    new_rows=[r for r in data['metrics'] if r['case'] in ('G0','G1')]
    for row in new_rows:
        other=next(r for r in server['runs'] if r['case']==row['case'] and r['seed']==row['seed'])
        for key in ('AUC_native','AUC_exact','HM_at_val_bias','gt_object_attr_acc_unseen','support_mass_unseen'):
            assert abs(row[key]-other[row['setting']+'_'+key])<1e-12
    rank_errors={k:0. for k in METRICS};checked=0
    groups={}
    for row in ranking['runs']:
        key=(row['case'],row['seed'],row['selection'],row['condition'],row['phase'])
        groups.setdefault(key,[]).append(row)
    for (case,seed,selection,condition,phase),rows in groups.items():
        if case.startswith('A'):
            base=ROOT/'server_results_roundA_confirm_20261019/roundA_20261019'
        else:base=ARCHIVE/('round%s_20261020'%case[0])
        if condition=='train_mean':
            folder='mean_gate' if selection=='native' else 'mean_gate_exact'
            path=base/folder/case/('seed%d'%seed)/'train_mean'
        else:
            path=next((base/case).glob('ut-zap50k/V*/seed%d'%seed))
            if selection=='exact':path=path/'exact_selection'
        with np.load(path/'predictions'/(phase+'_ow.npz'),allow_pickle=False) as archive:
            raw={k:archive[k].copy() for k in archive.files}
        actual,_=diagnose(raw,manifest,phase)
        for row in rows:
            other=next(r for r in actual if r['group']==row['group'])
            for name in METRICS:rank_errors[name]=max(rank_errors[name],abs(row[name]-other[name]))
            checked+=1
    assert max(rank_errors.values())<1e-12
    release=read(ARCHIVE/'vgcpod_v1_roundG/release_manifest.json')
    for name,fingerprint in release['files_sha256'].items():
        assert hashlib.sha256((ARCHIVE/'vgcpod_v1_roundG'/name).read_bytes()).hexdigest()==fingerprint,name
    with np.load(CACHE/'text_features.npz') as archive:
        eo=torch.from_numpy(archive['eo'].copy()).double()
        ea=torch.from_numpy(archive['ea'].copy()).double()
    singular=torch.linalg.svdvals(eo);rank=int(torch.linalg.matrix_rank(eo))
    assert rank==eo.shape[0]
    pinv=torch.linalg.pinv(eo)
    folding=[]
    generator=torch.Generator().manual_seed(621)
    visual=torch.randn(19,eo.shape[1],dtype=torch.float64,generator=generator)
    visual=visual/visual.norm(dim=-1,keepdim=True)
    for selection,folder in [('native','mean_gate'),('exact','mean_gate_exact')]:
        for seed in (3351,3352,3353):
            base=ARCHIVE/'roundG_20261020'
            run=base/'G1/ut-zap50k/V1'/('seed%d'%seed)
            checkpoint=helper.checkpoint(run/'checkpoints'/('best_'+selection+'.pt'))
            model=checkpoint['model'];cfg=checkpoint['config']
            with np.load(base/folder/'G1'/('seed%d'%seed)/'train_mean_gamma.npz') as archive:
                gamma=torch.from_numpy(archive['mean_gamma'].copy()).double()
            v=eo@model['Av'].double().T
            target=v*gamma
            folded_Av_T=pinv@target
            folded_v=eo@folded_Av_T
            u=ea@model['Au'].double().T
            readout=visual@model['Wc'].double()
            direct=cfg['rho']*cfg['kappa']*torch.einsum('br,jr,ir->bji',readout,target,u)
            equivalent=cfg['rho']*cfg['kappa']*torch.einsum('br,jr,ir->bji',readout,folded_v,u)
            folding.append(dict(seed=seed,selection=selection,epoch=checkpoint['epoch'],
                factor_max_error=float((target-folded_v).abs().max()),
                random_visual_delta_max_error=float((direct-equivalent).abs().max())))
    assert max(r['random_visual_delta_max_error'] for r in folding)<1e-12
    train_log_errors=dict(support_coefficient=0.,weighted_pair=0.,weighted_support=0.,batch_residual=0.)
    for case in ('G0','G1'):
        for seed in (3351,3352,3353):
            run=ARCHIVE/'roundG_20261020'/case/'ut-zap50k/V1'/('seed%d'%seed)
            for entry in read(run/'history.json'):
                t=entry['train'];weight=9/86
                if case=='G0':assert t['orth']==0 and t['gamma_image_variance']==0
                train_log_errors['support_coefficient']=max(train_log_errors['support_coefficient'],abs(t['support_weight']-weight))
                train_log_errors['weighted_pair']=max(train_log_errors['weighted_pair'],abs(t['pair_weighted']-1.5*t['pair']))
                train_log_errors['weighted_support']=max(train_log_errors['weighted_support'],abs(t['support_weighted']-weight*t['support']))
                train_log_errors['batch_residual']=max(train_log_errors['batch_residual'],t['task_reconstruction_error'])
    result=dict(server_main_summary_matches=True,ranking_rows_verified=checked,ranking_max_errors=rank_errors,
        release_files_verified=len(release['files_sha256']),train_log_errors=train_log_errors,
        object_text_shape=list(eo.shape),object_text_rank=rank,
        object_text_condition_number=float(singular[0]/singular[-1]),static_gate_folding=folding,
        folding_rule='Full row-rank fixed object vocabulary: Av_eff.T = pinv(Eo) @ ((Eo @ Av.T) * mean_gamma)')
    (OUT/'supplementary_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
