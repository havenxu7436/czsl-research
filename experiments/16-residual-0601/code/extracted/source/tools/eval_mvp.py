import argparse, hashlib, json, re, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from config import cfg
from dataset import CompositionDataset
from models.kcc_cpf import DDM
import evaluator_ge

def sha(path):
    p=Path(path)
    if not p.is_file(): return None
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def scalar(x):
    if torch.is_tensor(x): x=x.detach().cpu().numpy()
    z=np.asarray(x)
    return float(z.item()) if z.size==1 else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--load',required=True)
    p.add_argument('--phase',choices=['val','test'],required=True)
    p.add_argument('--setting',choices=['cw','ow'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--save-scores',action='store_true')
    p.add_argument('opts',nargs=argparse.REMAINDER)
    a=p.parse_args()
    cfg.merge_from_file(a.cfg); cfg.merge_from_list(a.opts)
    assert bool(cfg.EVAL.open_world)==(a.setting=='ow'), 'setting/config mismatch'
    cfg.EVAL.topk=1
    run=Path(getattr(cfg.TRAIN,'kcc_run_dir',str(Path(a.out).parent)))
    frozen_cache=run/'meta/cache_manifest.json'
    live_cache=Path(cfg.MODEL.kcc_cache)/'cache_manifest.json'
    if frozen_cache.is_file() and sha(frozen_cache)!=sha(live_cache):
        raise ValueError('live knowledge cache differs from the training manifest')
    dest=Path(a.out); dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metrics.json').exists() or (dest/'scores.dat').exists():
        raise FileExistsError('refusing overwrite: '+str(dest))
    device=torch.device('cuda:0')
    trainset=CompositionDataset(phase='train',split=cfg.DATASET.splitname,
               open_world=cfg.EVAL.open_world,cfg=cfg)
    ds=CompositionDataset(phase=a.phase,split=cfg.DATASET.splitname,
               open_world=cfg.EVAL.open_world,cfg=cfg)
    dl=torch.utils.data.DataLoader(ds,batch_size=cfg.TRAIN.test_batch_size,
               shuffle=False,num_workers=cfg.TRAIN.num_workers)
    model=DDM(trainset,cfg).to(device)
    model.load_state_dict(torch.load(a.load,map_location=device),strict=True)
    model.eval()
    model.val_attrs=torch.tensor([ds.attr2idx[x] for x,y in ds.pairs],device=device)
    model.val_objs=torch.tensor([ds.obj2idx[y] for x,y in ds.pairs],device=device)
    model.val_pairs=ds.pairs
    ev=evaluator_ge.Evaluator(ds,model)
    n=len(ds); nc=len(ds.pairs)
    print('phase=',a.phase,'setting=',a.setting,'images=',n,'candidates=',nc,
          'score_disk_GiB=',4*n*nc/(1024**3),flush=True)
    truth_a=np.empty(n,dtype=np.int64); truth_o=np.empty(n,dtype=np.int64)
    truth_p=np.empty(n,dtype=np.int64)
    # 标明bias=0的top5，不能冒称best-HM bias预测。
    top5=np.empty((n,min(5,nc)),dtype=np.int64)
    torch.cuda.reset_peak_memory_stats()
    start=time.time()
    with tempfile.TemporaryDirectory(prefix='score_tmp_',dir=str(dest)) as td:
        scorepath=dest/'scores.dat' if a.save_scores else Path(td)/'scores.dat'
        mm=np.memmap(str(scorepath),mode='w+',dtype=np.float32,shape=(n,nc))
        offset=0
        with torch.no_grad():
            for batch in dl:
                for k,v in batch.items():
                    if torch.is_tensor(v): batch[k]=v.to(device,non_blocking=True)
                out=model(batch)
                mat=torch.stack([out['scores'][pair] for pair in ds.pairs],dim=1)
                mat=mat.detach().float().cpu()
                if not torch.isfinite(mat).all(): raise ValueError('nonfinite scores')
                b=mat.size(0); sl=slice(offset,offset+b)
                mm[sl]=mat.numpy()
                truth_a[sl]=batch['attr'].cpu().numpy()
                truth_o[sl]=batch['obj'].cpu().numpy()
                truth_p[sl]=batch['pair'].cpu().numpy()
                top5[sl]=torch.topk(mat,min(5,nc),dim=1).indices.numpy()
                offset+=b
                del out,mat
        assert offset==n
        mm.flush()
        matrix=torch.from_numpy(mm)
        pred={pair:matrix[:,i] for i,pair in enumerate(ds.pairs)}
        ga=torch.from_numpy(truth_a); go=torch.from_numpy(truth_o); gp=torch.from_numpy(truth_p)
        results=ev.score_model(pred,go,bias=1e3,topk=1)
        stats=ev.evaluate_predictions(results,ga,go,gp,pred,topk=1)
        vals={k:scalar(v) for k,v in stats.items() if scalar(v) is not None}
        for required in ['AUC','best_hm','best_seen','best_unseen']:
            if required not in vals: raise ValueError('missing metric '+required)
            if not np.isfinite(vals[required]): raise ValueError('nonfinite metric')
        run=Path(getattr(cfg.TRAIN,'kcc_run_dir',str(dest.parent)))
        best_epoch=None
        if (run/'train.log').exists():
            matches=re.findall(r'Loading validation-best checkpoint from epoch (\d+)',
                              (run/'train.log').read_text(errors='replace'))
            if matches: best_epoch=int(matches[-1])
        meta={
            'run_id':run.name,'stage':getattr(cfg.TRAIN,'kcc_stage','audit'),
            'dataset':cfg.DATASET.name,'variant':getattr(cfg.TRAIN,'kcc_variant','A'),
            'seed':int(cfg.TRAIN.seed),'phase':a.phase,'setting':a.setting,'topk':1,
            'score_mode':getattr(cfg.MODEL,'kcc_score_mode','legacy'),
            'control':getattr(cfg.MODEL,'kcc_control','none'),
            'lambda_k':float(getattr(cfg.MODEL,'kcc_lambda',0)),
            'beta':float(getattr(cfg.MODEL,'kcc_beta',0)),
            'best_epoch':best_epoch,'checkpoint_sha256':sha(a.load),
            'config_sha256':hashlib.sha256(cfg.dump().encode()).hexdigest(),
            'source_manifest_sha256':sha(run/'meta/source.sha256'),
            'knowledge_manifest_sha256':sha(run/'meta/cache_manifest.json'),
            'dataset_manifest_sha256':sha(run/'meta/dataset_hashes.txt'),
            'n_images':n,'n_candidates':nc,'eval_seconds':time.time()-start,
            'peak_eval_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
            'score_dtype':'float32','score_shape':[n,nc],
            'scores_saved':a.save_scores,'metrics_raw':vals}
        (dest/'config_eval_effective.yml').write_text(cfg.dump())
        (dest/'candidates.json').write_text(json.dumps(ds.pairs,ensure_ascii=False))
        np.savez_compressed(str(dest/'labels.npz'),attr=truth_a,obj=truth_o,pair=truth_p,
                            top5_indices_at_bias0=top5)
        # image_id需来自dataset真实顺序，不按文件夹重新排序。
        image_ids=[item[0] for item in ds.data]
        (dest/'image_ids.json').write_text(json.dumps(image_ids,ensure_ascii=False))
        (dest/'metrics.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        print('Top 1',json.dumps(vals,ensure_ascii=False),flush=True)
        del results,pred,matrix,mm

if __name__=='__main__':
    main()
