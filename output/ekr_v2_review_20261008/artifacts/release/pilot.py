"""Bounded UT seed-0 pilot. No modification of old source, banks, or gates."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F
from core import AnchorModel, IncrementRouter

BUNDLE = Path(__file__).resolve().parent
VARIANTS = ('V_name_align','V_clean_align','V_shuffle_align','V_clean_fixed')
METRICS = ('AUC','best_hm','best_seen','best_unseen','hm_seen','hm_unseen')
SETTINGS = dict(seed=0,epochs=3,rank=32,value_rank=32,eta_max=.1,temperature=.2,
                score_temperature=.05,learning_rate=.001,weight_decay=.0001,batch_size=64)


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def J(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def W(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(str(temporary),str(path))


def save_checkpoint(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    torch.save(value,str(temporary));os.replace(str(temporary),str(path))


def release_id():return sha(BUNDLE/'SHA256SUMS')


def check_inputs(work):
    subprocess.run([sys.executable,str(BUNDLE/'verify_release.py')],check=True)
    reference=J(BUNDLE/'reference.json')
    require(str(work)==reference['work'],'Use the pinned reviewed name work')
    for path,expected in reference['inputs_sha256'].items():
        require(Path(path).is_file() and sha(path)==expected,'Reviewed input changed or missing: '+path)
    for relative,expected in reference['code_sha256'].items():
        require(sha(work/'src'/relative)==expected,'Old source changed: '+relative)
    sys.path.insert(0,str(work/'src'));os.chdir(str(work/'src'))
    from transition.name_plan import validate_protocol
    validate_protocol(work,full=True)
    gate=J(work/'reports/pilot/decision.json')
    require(gate['proceed'] is False and gate['seeds']==[0],'The pinned old pilot must remain unchanged')
    old_diagnostics=J(reference['mechanism_integrity_path'])
    require(old_diagnostics['passed'] and old_diagnostics['raw_argmax_exact_replay'],
            'Reviewed mechanism replay did not pass')
    require(torch.__version__=='1.13.1+cu117' and torch.cuda.is_available(),
            'Use CPF PyTorch 1.13.1+cu117 and CUDA')
    return reference


def check_output(out,work):
    require(str(out).startswith('/var/tmp/czsl_xhw_ekr_v2_align_'),
            'Use a separate /var/tmp/czsl_xhw_ekr_v2_align_ output version')
    require(out!=work and work not in out.parents and out not in work.parents,'Output overlaps old work')
    if out.exists():
        protocol=J(out/'protocol.json')
        require(protocol['release_sha256']==release_id(),'Bundle changed since output preparation')
        require(protocol['output']==str(out) and protocol['work']==str(work),'Output identity changed')
        require(protocol['settings']==SETTINGS and protocol['variants']==list(VARIANTS),'Pilot budget changed')
        if (out/'prepare_checks.json').is_file():
            for relative,expected in J(out/'prepare_checks.json')['baseline_files_sha256'].items():
                require(sha(out/relative)==expected,'Prepared baseline changed: '+relative)
    require(shutil.disk_usage(out.parent).free>=5*1024**3,'Need at least 5 GiB free on output disk')


def make_model(reference,variant):
    from transition.name_runtime import read_cfg,trainset,load_branch
    from transition.common import seed_all
    original=read_cfg(reference['name_config'])
    require(original.TRAIN.seed==0 and original.MODEL.branch_mode=='name_only','Wrong name anchor config')
    ds=trainset(original)
    name_model,saved=load_branch(original,ds,reference['name_checkpoint'])
    require(saved['epoch']==3,'Use the reviewed selected seed-0 name anchor')
    seed_all(SETTINGS['seed'])
    router=IncrementRouter(name_model.bank,variant,rank=SETTINGS['rank'],value_rank=SETTINGS['value_rank'],
        eta_max=SETTINGS['eta_max'],temperature=SETTINGS['temperature']).to('cuda:0')
    model=AnchorModel(name_model,router).to('cuda:0')
    return model,ds,original


def make_cfg(original,run,variant):
    cfg=original.clone()
    cfg.TRAIN.kcc_run_dir=str(run);cfg.TRAIN.kcc_variant=variant
    cfg.TRAIN.seed=0;cfg.TRAIN.batch_size=64;cfg.TRAIN.max_epoch=3
    return cfg


def next_attempt(run,prefix):
    folder=run/prefix
    for attempt in range(10000):
        target=folder/('attempt_%04d'%attempt)
        if not target.exists():return target
    raise ValueError('Too many preserved attempts')


def frozen_state(model):
    from transition.common import state_digest
    return dict(base=state_digest(model.base),name=state_digest(model.name_anchor))


def prepare(work,out,reference):
    require(not out.exists(),'Output exists; train/resume it or choose a new output version')
    subprocess.run([sys.executable,str(BUNDLE/'tests.py')],check=True)
    out.mkdir()
    protocol=dict(version=BUNDLE.name,work=str(work),output=str(out),release_sha256=release_id(),
        reference_sha256=sha(BUNDLE/'reference.json'),settings=SETTINGS,variants=list(VARIANTS),
        old_pilot_proceed=False,phase='UT seed-0 exploratory pilot',test_allowed=False,
        common_anchor='frozen A_unit + frozen reviewed N_name',
        primary='V_clean_align',epoch_selection='first strict CW maximum among epochs 0,1,2,3',
        pilot_threshold_pp=.2,alignment_ablation_threshold_pp=.1,
        guards=dict(cw_hm_pp=-.5,ow_auc_pp=-.5,ow_best_unseen_pp=-2),
        note='Input shuffle control is not a random semantic graph; no static K; no robustness loss')
    W(out/'protocol.json',protocol);W(out/'status.json',dict(status='preparing'))
    shutil.copy2(BUNDLE/'reference.json',out/'reference.json')
    for p in BUNDLE.iterdir():
        if p.is_file():
            destination=out/'release'/p.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(p,destination)
    model,ds,original=make_model(reference,'V_clean_align')
    from transition.common import state_digest
    from transition.evaluate import evaluate
    initial=frozen_state(model)
    expected=reference['base_state_digest']
    require(initial['base']==expected,'A_unit frozen state differs from reviewed pilot')
    cfg=make_cfg(original,out/'baseline','B_name')
    checks=[]
    for setting in ('cw','ow'):
        rec=evaluate(model,cfg,'val',setting,out/'baseline'/('val_'+setting),save_scores=True,force_off=True)
        target=reference['name_metrics'][setting]['metrics_raw']
        difference=max(abs(100*(rec['metrics_raw'][key]-target[key])) for key in METRICS)
        require(difference<=.05,'Frozen name anchor replay drift')
        oldlabels=Path(reference['name_config']).parent/('eval_val_'+setting)/'labels.npz'
        with np.load(str(oldlabels),allow_pickle=False) as old,np.load(str(out/'baseline'/('val_'+setting)/'labels.npz'),allow_pickle=False) as new:
            require(all(np.array_equal(old[k],new[k]) for k in ('attr','obj','pair','top1_at_bias0')),
                    'Name baseline label/prediction replay mismatch')
        checks.append(dict(setting=setting,max_metric_difference_pp=difference,raw_predictions_equal=True))
    init=[]
    from transition.common import seed_all
    for variant in VARIANTS:
        seed_all(0);router=IncrementRouter(model.bank,variant)
        init.append(dict(variant=variant,state_digest=state_digest(router),
            registered_params=sum(p.numel() for p in router.parameters()),
            trainable_params=sum(p.numel() for p in router.parameters() if p.requires_grad)))
    require(len({r['state_digest'] for r in init})==1,'Control initial parameters differ')
    require(initial==frozen_state(model),'Preparation changed frozen weights')
    check_inputs(work)
    W(out/'prepare_checks.json',dict(passed=True,initialization=init,replay=checks,frozen_state=initial,
        training_performed=False,common_baseline_checkpoint_sha256=sha(reference['name_checkpoint']),
        baseline_files_sha256={str(p.relative_to(out)):sha(p) for p in sorted((out/'baseline').rglob('*')) if p.is_file()}))
    W(out/'status.json',dict(status='prepared'))
    print('V2 PREPARED; OLD GATE REMAINS FALSE',flush=True)


def rng_state(generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),loader=generator.get_state())


def restore_rng(value,generator):
    random.setstate(value['python']);np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch']);torch.cuda.set_rng_state_all(value['cuda'])
    generator.set_state(value['loader'])


def branch_snapshot(model,epoch,protocol,variant):
    return dict(router_state={k:v.detach().cpu().clone() for k,v in model.router.state_dict().items()},
        epoch=epoch,variant=variant,protocol_sha256=protocol,release_sha256=release_id())


def check_completed(run,out,variant):
    complete=J(run/'complete.json')
    require(complete['variant']==variant and complete['protocol_sha256']==sha(out/'protocol.json'),
            'Completed run identity mismatch')
    require(sha(run/'checkpoints/best_increment.pth')==complete['best_checkpoint_sha256'],
            'Completed increment checkpoint changed')
    require(complete['completed_epochs']==3 and complete['frozen_state_unchanged'],
            'Incomplete or changed frozen anchor')
    for setting in ('cw','ow'):
        folder=run/complete['final_eval_folders'][setting]
        for name,digest in complete['final_eval_sha256'][setting].items():
            require(sha(folder/name)==digest,'Completed evaluation changed: '+str(folder/name))
    return complete


def train_one(work,out,reference,variant,resume):
    run=out/'runs'/variant
    if (run/'complete.json').is_file():
        check_completed(run,out,variant)
        print('VERIFIED COMPLETE; SKIP',variant,flush=True);return
    existing=run.exists()
    require(not existing or resume,'Partial run exists; preserve it and use the resume command')
    model,ds,original=make_model(reference,variant)
    from transition.common import candidates,move,seed_all,seed_worker,state_digest
    from transition.evaluate import evaluate
    expected=J(out/'prepare_checks.json')
    require(state_digest(model.router)==expected['initialization'][0]['state_digest'],
            'Formal initialization differs from prepared controls')
    frozen=frozen_state(model)
    require(frozen==expected['frozen_state'],'Frozen baseline changed')
    cfg=make_cfg(original,run,variant)
    optimizer=torch.optim.AdamW([p for p in model.router.parameters() if p.requires_grad],
        lr=SETTINGS['learning_rate'],weight_decay=SETTINGS['weight_decay'])
    seed_all(0)
    generator=torch.Generator().manual_seed(0)
    loader=torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True,num_workers=cfg.TRAIN.num_workers,
        pin_memory=True,worker_init_fn=seed_worker,generator=generator,persistent_workers=False)
    protocol_hash=sha(out/'protocol.json')
    last_path=run/'checkpoints/last_epoch.pth'
    if existing:
        require(last_path.is_file(),'No committed epoch state; preserve this run and use a new output version')
        last=torch.load(str(last_path),map_location='cpu')
        require(last['protocol_sha256']==protocol_hash and last['variant']==variant and
                last['release_sha256']==release_id(),'Resume provenance mismatch')
        best_path=run/last['best_checkpoint_relative']
        require(sha(best_path)==last['best_checkpoint_sha256'],'Selected checkpoint changed before resume')
        model.router.load_state_dict(last['router_state'],strict=True)
        optimizer.load_state_dict(last['optimizer'])
        restore_rng(last['rng'],generator)
        first_epoch=last['epoch']+1;best=last['best_auc'];best_epoch=last['best_epoch'];records=last['records']
        require(0<=last['epoch']<=3 and len(records)==last['epoch'],'Invalid committed epoch budget')
        print('RESUME AFTER COMMITTED EPOCH',last['epoch'],variant,flush=True)
    else:
        run.mkdir(parents=True)
        (run/'source_anchor_config.yml').write_text(original.dump(),encoding='utf-8')
        W(run/'effective_settings.json',dict(settings=SETTINGS,variant=variant,
            num_workers=int(cfg.TRAIN.num_workers),value_alignment_trainable=not variant.endswith('_fixed'),
            registered_params=sum(p.numel() for p in model.router.parameters()),
            trainable_params=sum(p.numel() for p in model.router.parameters() if p.requires_grad),
            source_config_sha256=sha(reference['name_config']),common_anchor_frozen=True))
        first_epoch=1;best=J(out/'baseline/val_cw/metrics.json')['metrics_raw']['AUC'];best_epoch=0;records=[]
        best_path=run/'checkpoints/selected_epoch_00.pth'
        save_checkpoint(best_path,branch_snapshot(model,0,protocol_hash,variant))
        initial=branch_snapshot(model,0,protocol_hash,variant)
        initial.update(optimizer=optimizer.state_dict(),rng=rng_state(generator),best_auc=best,
            best_epoch=0,best_checkpoint_sha256=sha(best_path),records=records,
            best_checkpoint_relative=str(best_path.relative_to(run)))
        save_checkpoint(last_path,initial)
    log_path=run/('train_attempt_%s.log'%time.time_ns())
    started=time.time()
    with log_path.open('w',encoding='utf-8') as log:
        for epoch in range(first_epoch,4):
            model.train(True);ai,oi,pairs=candidates(model,ds,train=True)
            n=0;loss_total=0;gradient_names=[]
            for index,batch in enumerate(loader):
                batch=move(batch,'cuda:0')
                anchor,feature=model.components(batch,ai,oi)
                increment=model.router(feature,ai,oi)
                loss=F.cross_entropy((anchor+increment)/.05,batch['pair'])
                require(torch.isfinite(loss),'Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                gradient_names=[name for name,p in model.router.named_parameters()
                    if p.grad is not None and float(p.grad.abs().sum())>0]
                require(all(torch.isfinite(p.grad).all() for p in model.router.parameters() if p.grad is not None),
                        'Nonfinite increment gradient')
                require(all(p.grad is None for module in (model.base,model.name_anchor) for p in module.parameters()),
                        'Frozen baseline received gradients')
                optimizer.step();size=len(batch['pair']);n+=size;loss_total+=float(loss.detach())*size
                if index%25==0:
                    line='epoch={} batch={}/{} loss={:.6f} eta={:.6f} gradients={}\n'.format(
                        epoch,index+1,len(loader),float(loss.detach()),float(model.router.eta.detach()),len(gradient_names))
                    log.write(line);log.flush();print(line,end='',flush=True)
            require(n==len(ds),'Training image count mismatch')
            folder=next_attempt(run,'epochs/epoch_%02d_cw'%epoch)
            record=evaluate(model,cfg,'val','cw',folder,save_scores=True)
            auc=record['metrics_raw']['AUC']
            if auc>best:
                best=auc;best_epoch=epoch
                best_path=run/'checkpoints'/('selected_epoch_%02d_%s.pth'%(epoch,time.time_ns()))
                save_checkpoint(best_path,branch_snapshot(model,epoch,protocol_hash,variant))
            require(frozen==frozen_state(model),'Frozen baseline state changed')
            records.append(dict(epoch=epoch,loss_mean=loss_total/n,cw_auc_pct=100*auc,
                cw_hm_pct=100*record['metrics_raw']['best_hm'],eta=float(model.router.eta.detach()),
                gradient_names_at_last_batch=gradient_names,evaluation=str(folder.relative_to(run)),
                elapsed_attempt_seconds=time.time()-started))
            committed=branch_snapshot(model,epoch,protocol_hash,variant)
            committed.update(optimizer=optimizer.state_dict(),rng=rng_state(generator),best_auc=best,
                best_epoch=best_epoch,best_checkpoint_sha256=sha(best_path),records=records,
                best_checkpoint_relative=str(best_path.relative_to(run)))
            save_checkpoint(last_path,committed);W(run/'committed_epochs.json',records)
            print('COMMITTED EPOCH',epoch,variant,'SELECTED EPOCH',best_epoch,flush=True)
    final_best=run/'checkpoints/best_increment.pth'
    if final_best.exists():
        require(sha(final_best)==sha(best_path),'Final selected checkpoint differs from committed state')
    else:
        shutil.copy2(best_path,final_best)
    best_path=final_best
    selected=torch.load(str(best_path),map_location='cpu')
    model.router.load_state_dict(selected['router_state'],strict=True)
    model.evaluation_checkpoint_sha256=sha(best_path)
    final_folders={};final_hashes={}
    for setting in ('cw','ow'):
        folder=next_attempt(run,'final_'+setting)
        record=evaluate(model,cfg,'val',setting,folder,save_scores=True)
        record.update(v2_variant=variant,selected_epoch=best_epoch,
            common_name_checkpoint_sha256=sha(reference['name_checkpoint']),protocol_sha256=protocol_hash)
        W(folder/'metrics.json',record)
        final_folders[setting]=str(folder.relative_to(run))
        final_hashes[setting]={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
    require(frozen_state(model)==frozen,'Frozen baseline state changed after final evaluation')
    check_inputs(work)
    W(run/'complete.json',dict(variant=variant,protocol_sha256=protocol_hash,completed_epochs=3,
        selected_epoch=best_epoch,best_checkpoint_sha256=sha(best_path),frozen_state_unchanged=True,
        frozen_state=frozen,final_eval_folders=final_folders,final_eval_sha256=final_hashes))
    print('V2 RUN COMPLETE',variant,'SELECTED EPOCH',best_epoch,flush=True)


def train(work,out,reference,resume):
    require(J(out/'prepare_checks.json')['passed'],'Prepare must pass')
    W(out/'status.json',dict(status='training'))
    for variant in VARIANTS:
        train_one(work,out,reference,variant,resume)
        torch.cuda.empty_cache()
    W(out/'status.json',dict(status='pilot_training_complete'))


def decision(records,baseline,threshold=.2):
    clean=records['V_clean_align']
    comparisons={name:100*(clean['cw']['metrics_raw']['AUC']-value['cw']['metrics_raw']['AUC'])
        for name,value in [('B_name',baseline),('V_name_align',records['V_name_align']),
                           ('V_shuffle_align',records['V_shuffle_align'])]}
    conditions={name+'_cw_gain_at_least_0.2pp':delta>=threshold for name,delta in comparisons.items()}
    for name,value in [('B_name',baseline),('V_name_align',records['V_name_align'])]:
        for setting,key,tolerance in [('cw','best_hm',.5),('ow','AUC',.5),('ow','best_unseen',2)]:
            delta=100*(clean[setting]['metrics_raw'][key]-value[setting]['metrics_raw'][key])
            conditions[name+'_'+setting+'_'+key+'_guard']=delta>=-tolerance
    alignment_delta=100*(clean['cw']['metrics_raw']['AUC']-records['V_clean_fixed']['cw']['metrics_raw']['AUC'])
    return dict(stage='v2_seed0_pilot',proceed=all(conditions.values()),conditions=conditions,
        paired_cw_auc_deltas_pp=comparisons,alignment_over_fixed_delta_pp=alignment_delta,
        alignment_gain_supported_at_0_1pp=alignment_delta>=.1,old_pilot_proceed=False,
        seeds=[0],statistical_proof=False,
        next=('freeze a separate multi-seed confirmation protocol' if all(conditions.values()) else
              'stop this v2 pilot budget; preserve every result; do not run old stage C'))


def report(work,out,reference):
    from transition.common import state_digest
    baseline={s:J(out/'baseline'/('val_'+s)/'metrics.json') for s in ('cw','ow')}
    records={};rows=[]
    for variant in VARIANTS:
        run=out/'runs'/variant;complete=check_completed(run,out,variant)
        records[variant]={s:J(run/complete['final_eval_folders'][s]/'metrics.json') for s in ('cw','ow')}
        settings=J(run/'effective_settings.json')
        require(settings['settings']==SETTINGS and settings['source_config_sha256']==sha(reference['name_config']),
                'Run settings not matched to the pinned protocol')
        for setting,record in records[variant].items():
            base=baseline[setting]
            with np.load(str(run/complete['final_eval_folders'][setting]/'labels.npz'),allow_pickle=False) as lab, np.load(str(out/'baseline'/('val_'+setting)/'labels.npz'),allow_pickle=False) as anchor:
                require(all(np.array_equal(lab[k],anchor[k]) for k in ('attr','obj','pair')),'Final labels not paired')
            require(J(run/complete['final_eval_folders'][setting]/'candidates.json')==
                    J(out/'baseline'/('val_'+setting)/'candidates.json'),'Final candidates not paired')
            require(J(run/complete['final_eval_folders'][setting]/'image_ids.json')==
                    J(out/'baseline'/('val_'+setting)/'image_ids.json'),'Final images not paired')
            m=record['metrics_raw'];bm=base['metrics_raw']
            rows.append(dict(variant=variant,seed=0,setting=setting,selected_epoch=complete['selected_epoch'],
                trainable_params=settings['trainable_params'],auc_pct=100*m['AUC'],
                delta_auc_vs_name_anchor_pp=100*(m['AUC']-bm['AUC']),hm_pct=100*m['best_hm'],
                best_unseen_pct=100*m['best_unseen'],eta=record['diagnostics']['eta'],
                null_mean=record['diagnostics']['null_mean'],value_drift_mean=record['diagnostics']['value_drift_mean']))
    require(len({J(out/'runs'/v/'effective_settings.json')['trainable_params'] for v in VARIANTS[:3]})==1,
            'Main attribution controls have unequal trainable capacity')
    check_inputs(work)
    result=decision(records,baseline)
    result['protocol_sha256']=sha(out/'protocol.json')
    W(out/'reports/decision.json',result)
    with (out/'reports/results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    W(out/'reports/integrity.json',dict(passed=True,matched_main_controls=3,completed_runs=4,
        seed=0,epochs_each=3,frozen_A_unit_and_name_anchor=True,labels_candidates_images_paired=True,
        old_pilot_proceed=False,alignment_fixed_is_an_ablation_not_equal_trainable_capacity_control=True))
    W(out/'status.json',dict(status='pilot_report_complete',proceed=result['proceed']))
    print('V2 PILOT DECISION',json.dumps(result),flush=True)
    pack(out)


def pack(out):
    require(J(out/'status.json')['status']=='pilot_report_complete','Report must complete before packing')
    require(J(out/'reports/decision.json')['protocol_sha256']==sha(out/'protocol.json'),
            'Report protocol changed before packing')
    for variant in VARIANTS:check_completed(out/'runs'/variant,out,variant)
    prepared=J(out/'prepare_checks.json')
    for relative,expected in prepared.get('baseline_files_sha256',{}).items():
        require(sha(out/relative)==expected,'Prepared baseline changed before packing')
    target=out.parent/(out.name+'_review.zip')
    require(not target.exists(),'Review ZIP exists; preserve it')
    files=[p for p in sorted(out.rglob('*')) if p.is_file() and p.suffix not in ('.pth','.pt','.tmp')
           and p.name!='review_manifest.json']
    manifest={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in files}
    W(out/'review_manifest.json',dict(files_sha256=manifest,weights_included=False))
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[out/'review_manifest.json']:
            z.write(p,str(p.relative_to(out)).replace('\\','/'))
    print('DOWNLOAD V2 REVIEW ZIP',target,'SHA256',sha(target),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','train','resume','report','pack'))
    p.add_argument('--work',default='/var/tmp/czsl_xhw_name_ekr_20261007')
    p.add_argument('--out',default='/var/tmp/czsl_xhw_ekr_v2_align_20261007')
    args=p.parse_args();work=Path(args.work).resolve();out=Path(args.out).resolve()
    if args.command=='pack':pack(out);return
    reference=check_inputs(work);check_output(out,work)
    if args.command=='prepare':prepare(work,out,reference)
    elif args.command in ('train','resume'):train(work,out,reference,args.command=='resume')
    else:report(work,out,reference)


if __name__=='__main__':main()
