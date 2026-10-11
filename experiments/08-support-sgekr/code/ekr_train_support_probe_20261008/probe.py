"""CPU diagnostic of cached TRAIN features; no optimizer, new images, VAL or TEST."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
import numpy as np

BUNDLE = Path(__file__).resolve().parent
VERSION = 'ekr_train_support_probe_20261008'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def J(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def W(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def C(p, rows):
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)

def evidence_vectors(units, mask):
    # Match raw fixed values: average unit vectors; do not re-normalize their mean.
    require(mask.any(1).all(), 'Empty attribute evidence')
    return (units*mask[...,None]).sum(1)/mask.sum(1,keepdims=True)/2

def stats(scores, gold, allowed):
    require(allowed[np.arange(len(gold)),gold].all(), 'Gold missing from diagnostic rivals')
    s = np.where(allowed, scores, -np.inf)
    competitor = s.copy();competitor[np.arange(len(gold)),gold] = -np.inf
    require(np.isfinite(competitor.max(1)).all(), 'Need at least one competing attribute')
    margin = s[np.arange(len(gold)),gold]-competitor.max(1)
    return margin, s.argmax(1)==gold

def analyze(features, labels, image_ids, bank, attrs, objs):
    x = unit(np.asarray(features, dtype=np.float64))
    a, o, pair = [labels[k].astype(int) for k in ('attr','obj','pair')]
    n = len(a);na = len(attrs)
    require(x.shape==(n,300) and np.isfinite(x).all(), 'Wrong cached feature array')
    require(len(image_ids)==n and len(set(image_ids))==n, 'Image IDs are not unique/paired')
    require(sorted(set(a))==list(range(na)), 'Primitive coverage changed')
    name = unit(np.asarray(bank['unit'], dtype=np.float64))/2
    clean = np.asarray(bank['units'], dtype=np.float64);cm = np.asarray(bank['mask'],bool)
    shuffled = np.asarray(bank['shuffle_units'],dtype=np.float64);sm = np.asarray(bank['shuffle_mask'],bool)
    require(clean.shape==shuffled.shape==(na,4,300), 'Wrong bank slot shape')
    require(np.allclose(clean[:,0],name*2,atol=1e-5) and np.allclose(shuffled[:,0],name*2,atol=1e-5),
            'Own names changed in the shuffle control')
    vectors={'name':name,'correct_mean':evidence_vectors(clean,cm),
             'shuffle_mean':evidence_vectors(shuffled,sm)}
    scores={k:x@v.T for k,v in vectors.items()}
    all_allowed=np.ones((n,na),bool)
    availability=np.zeros((len(objs),na),bool);availability[o,a]=True
    same_obj=availability[o]
    # Exactly three of six images from each TRAIN pair in each diagnostic fold.
    fold=np.zeros(n,int)
    for pi in sorted(set(pair)):
        ids=np.where(pair==pi)[0]
        require(len(ids)==6, 'Expected reviewed six-image-per-pair TRAIN cache')
        ordered=sorted(ids,key=lambda i:image_ids[i])
        fold[ordered[1::2]]=1
    summaries=[];by_attr=[];slot_rows=[];contrasts={};crossfit_rows=[]
    for view,allowed in [('all_attributes',all_allowed),('same_object_train_attributes',same_obj)]:
        metrics={k:stats(v,a,allowed) for k,v in scores.items()}
        for k,(m,correct) in metrics.items():
            summaries.append(dict(view=view,case=k,n_images=n,attribute_accuracy_pct=100*float(correct.mean()),
                macro_attribute_accuracy_pct=100*float(np.mean([correct[a==ai].mean() for ai in range(na)])),
                mean_margin=float(m.mean()),macro_attribute_margin=float(np.mean([m[a==ai].mean() for ai in range(na)]))))
        for left,right in [('correct_mean','name'),('correct_mean','shuffle_mean')]:
            per_attribute=[]
            for ai in range(na):
                selected=a==ai
                delta=float((metrics[left][0]-metrics[right][0])[selected].mean())
                per_attribute.append(delta)
                by_attr.append(dict(view=view,comparison=left+'-'+right,attribute=attrs[ai],
                    n_images=int(selected.sum()),mean_margin_delta=delta,
                    delta_attribute_accuracy_pp=100*float((metrics[left][1][selected].astype(float)-
                                                         metrics[right][1][selected].astype(float)).mean())))
            contrasts[view+'/'+left+'-'+right]=dict(macro_margin_delta=float(np.mean(per_attribute)),
                positive_attributes=int((np.array(per_attribute)>0).sum()),n_attributes=na,
                fold_macro_margin_deltas=[float(np.mean([(metrics[left][0]-metrics[right][0])[(a==ai)&(fold==fi)].mean()
                    for ai in range(na)])) for fi in (0,1)])
        # Visual prototypes are a diagnostic reference, not a deployed classifier.
        for fi in (0,1):
            fit=fold!=fi;held=fold==fi;proto=[]
            for ai in range(na):
                object_means=[x[fit&(a==ai)&(o==oi)].mean(0) for oi in sorted(set(o[fit&(a==ai)]))]
                require(bool(object_means), 'No prototype fit examples for primitive')
                proto.append(np.mean(object_means,axis=0))
            prototype=unit(np.asarray(proto))/2
            m,c=stats(x[held]@prototype.T,a[held],allowed[held])
            crossfit_rows.append(dict(view=view,held_fold=fi,n_images=int(held.sum()),
                case='visual_prototype_reference',attribute_accuracy_pct=100*float(c.mean()),
                macro_attribute_margin=float(np.mean([m[a[held]==ai].mean() for ai in range(na)]))))
    for ai in range(na):
        own=x[a==ai];other=x[a!=ai]
        for r in range(4):
            if not cm[ai,r]:
                continue
            own_mean=float((own@clean[ai,r]/2).mean());other_mean=float((other@clean[ai,r]/2).mean())
            slot_rows.append(dict(attribute=attrs[ai],slot=r,slot_kind='name' if r==0 else 'AE',
                n_positive_images=len(own),own_attribute_evidence_mean=own_mean,
                other_attribute_evidence_mean=other_mean,one_vs_rest_separation=own_mean-other_mean))
    primary=contrasts['same_object_train_attributes/correct_mean-shuffle_mean']
    name_check=contrasts['same_object_train_attributes/correct_mean-name']
    require(na==16, 'This frozen screen expects the reviewed UT attribute inventory')
    conditions=dict(correct_over_shuffle_macro_margin_at_least_0_01=primary['macro_margin_delta']>=.01,
        correct_over_shuffle_positive_on_both_folds=all(d>0 for d in primary['fold_macro_margin_deltas']),
        correct_over_shuffle_positive_on_at_least_11_attributes=primary['positive_attributes']>=11,
        correct_not_below_name_macro_margin=name_check['macro_margin_delta']>=0,
        correct_not_below_name_on_both_folds=all(d>=0 for d in name_check['fold_macro_margin_deltas']))
    decision=dict(stage='TRAIN_cached_support_diagnostic',support_screen_passed=all(conditions.values()),
        conditions=conditions,contrasts=contrasts,n_images=n,n_attributes=na,
        old_v2_gate_proceed=False,official_EKR_confirmation_allowed=False,new_optimizer_steps=0,
        val_or_test_features_used=False,new_model_performance_claim=False,
        note='Same already-trained CPF; internal TRAIN folds are not independent model-validation data. '
             'Uniform fixed evidence only; a negative screen does not rule out learned attention or another representation.',
        next=('A separate TRAIN-grounded fixed-value pilot may be designed; do not launch it automatically'
              if all(conditions.values()) else
              'Stop blind tuning of this GloVe score-residual setup; inspect representation/data grounding before a new hypothesis'))
    return dict(summary=summaries,by_attribute=by_attr,slots=slot_rows,crossfit=crossfit_rows,
        decision=decision,folds=fold.tolist(),unit_features=x)

def verify_bundle():
    for line in (BUNDLE/'SHA256SUMS').read_text(encoding='ascii').splitlines():
        expected,name=line.split('  ',1)
        require(sha(BUNDLE/name)==expected, 'Probe release changed: '+name)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='/var/tmp/czsl_xhw_ekr_train_support_20261008')
    args=p.parse_args();verify_bundle();ref=J(BUNDLE/'reference.json')
    import torch
    require(torch.__version__=='1.13.1+cu117', 'Use the absolute CPF Python interpreter')
    for remote,expected in ref['files_sha256'].items():
        require(Path(remote).is_file() and sha(remote)==expected, 'Pinned input changed/missing: '+remote)
    old=J(Path(ref['v2_output'])/'reports/decision.json')
    require(old['proceed'] is False, 'Preserve the reviewed failed v2 gate')
    cache=Path(ref['train_cache']);bank_root=Path(ref['bank'])
    manifest=J(bank_root/'manifest.json')
    for name,expected in manifest['files_sha256'].items():
        require(sha(bank_root/name)==expected, 'Bank manifest mismatch: '+name)
    raw_bank=torch.load(str(bank_root/'bank.pt'),map_location='cpu')
    bank={k:raw_bank['a'][k].detach().cpu().numpy() for k in ('unit','units','mask','shuffle_units','shuffle_mask')}
    metadata=J(bank_root/'labels.json');pairs=J(cache/'candidates.json')
    with np.load(cache/'labels.npz',allow_pickle=False) as f:
        labels={k:f[k].copy() for k in ('attr','obj','pair')}
    for pi,ai,oi in zip(labels['pair'],labels['attr'],labels['obj']):
        require(pairs[int(pi)]==[metadata['attrs'][int(ai)],metadata['objs'][int(oi)]], 'Label order differs from bank')
    ids=J(cache/'image_ids.json');selection=J(Path(ref['diagnostic_output'])/'train_selection.json')
    require(selection['n_images']==len(ids)==498 and selection['n_pairs']==83, 'TRAIN cache selection changed')
    result=analyze(np.load(cache/'attr_features.npy',allow_pickle=False),labels,ids,bank,metadata['attrs'],metadata['objs'])
    out=Path(args.out).resolve()
    require(str(out).startswith('/var/tmp/czsl_xhw_ekr_train_support_'), 'Use a new dedicated /var/tmp output')
    require(not out.exists(), 'Output exists; preserve it and select a new output version')
    require(out.parent.exists(), 'Output parent must exist')
    require(out not in Path(ref['v2_output']).parents and out!=Path(ref['v2_output']), 'Output overlaps old work')
    out.mkdir()
    W(out/'protocol.json',dict(version=VERSION,release_sha256=sha(BUNDLE/'SHA256SUMS'),
        reference_sha256=sha(BUNDLE/'reference.json'),files_sha256=ref['files_sha256'],
        feature_source='reviewed frozen A_unit TRAIN cache, six images per TRAIN pair',
        primary_view='same_object_train_attributes',fixed_rules='two image folds, object-balanced visual reference',
        optimizer_steps=0,model_weights_loaded=False,val_or_test_used=False,old_gates_modified=False))
    W(out/'decision.json',result['decision'])
    W(out/'fold_assignment.json',dict(image_ids=ids,folds=result['folds']))
    for name,key in [('summary.csv','summary'),('per_attribute.csv','by_attribute'),('slot_support.csv','slots'),('prototype_reference.csv','crossfit')]:
        C(out/name,result[key])
    W(out/'integrity.json',dict(passed=True,source_checks=len(ref['files_sha256']),
        bank_manifest_passed=True,cached_train_images=498,train_pairs=83,new_images_read=0,
        gpu_used=False,optimizer_steps=0,old_v2_gate_proceed=False))
    for remote,expected in ref['files_sha256'].items():
        require(sha(remote)==expected, 'Source changed during CPU probe')
    files=sorted(q for q in out.iterdir() if q.is_file())
    W(out/'review_manifest.json',dict(files_sha256={q.name:sha(q) for q in files},weights_included=False))
    archive=out.parent/(out.name+'_review.zip')
    require(not archive.exists(), 'Review ZIP exists; preserve it')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for q in sorted(out.iterdir()):
            if q.is_file():
                z.write(q,q.name)
    print('TRAIN SUPPORT PROBE COMPLETE',json.dumps(result['decision']),flush=True)
    print('DOWNLOAD REVIEW ZIP',archive,'SHA256',sha(archive),flush=True)

if __name__=='__main__':
    main()
