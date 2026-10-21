"""One shared TRAIN-only object-query mean for the H1 residual gate."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from .common import read_json,sha,write_json


def prepare_gate_statistics(cache,folder,chunk_size=256):
    cache,folder=Path(cache),Path(folder)
    manifest=read_json(cache/'features_manifest.json')
    features=np.load(cache/'train/fo.npy',mmap_mode='r')
    count,objects,dimension=features.shape
    if count!=manifest['counts']['train'] or objects!=len(manifest['labels']['objs']):
        raise ValueError('TRAIN object-query feature shape differs from manifest')
    identity=dict(schema='vgcpod_train_fo_mean_v1',feature_manifest_sha256=sha(cache/'features_manifest.json'),
        train_fo_sha256=manifest['files_sha256']['train/fo.npy'],train_images=count,
        object_names=manifest['labels']['objs'],feature_dimension=dimension)
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/'train_object_query_mean.npz'
    if target.exists():
        with np.load(target,allow_pickle=False) as saved:
            if json.loads(str(saved['identity_json'].item()))!=identity:
                raise ValueError('Existing TRAIN mean belongs to a different feature cache')
            mean64=saved['mean_fo_float64'].copy()
        reused=True
    else:
        if sha(cache/'train/fo.npy')!=identity['train_fo_sha256']:
            raise ValueError('TRAIN object-query feature checksum differs from manifest')
        total=np.zeros((objects,dimension),dtype=np.float64)
        for start in range(0,count,chunk_size):
            total+=np.asarray(features[start:start+chunk_size],dtype=np.float64).sum(axis=0,dtype=np.float64)
        mean64=total/count
        mean32=mean64.astype(np.float32)
        writing=target.with_name(target.name+'.writing.npz')
        np.savez_compressed(writing,mean_fo=mean32,mean_fo_float64=mean64,
            identity_json=np.array(json.dumps(identity,sort_keys=True)))
        os.replace(str(writing),str(target))
        reused=False
    write_json(folder/'train_object_query_mean.json',dict(identity,npz_sha256=sha(target),
        mean_shape=list(mean64.shape),accumulation_dtype='float64',model_buffer_dtype='float32',
        mean_vector_norms=np.linalg.norm(mean64,axis=-1).tolist(),
        origin='Every TRAIN image contributes to every object query; labels are not used'))
    print('TRAIN_GATE_MEAN_REUSE' if reused else 'TRAIN_GATE_MEAN_COMPLETE',target,'images',count,flush=True)
    return target


def load_gate_mean(cache,cfg,manifest):
    if 'gate_statistics_path' not in cfg or 'gate_statistics_sha256' not in cfg:
        raise ValueError('TRAIN-centered gates need the prepared runtime config from the experiment launcher')
    path=Path(cfg['gate_statistics_path'])
    if sha(path)!=cfg['gate_statistics_sha256']:
        raise ValueError('TRAIN gate mean differs from the run configuration')
    with np.load(path,allow_pickle=False) as saved:
        identity=json.loads(str(saved['identity_json'].item()))
        mean=saved['mean_fo'].copy()
    if (identity['feature_manifest_sha256']!=sha(Path(cache)/'features_manifest.json') or
        identity['object_names']!=manifest['labels']['objs'] or
        identity['train_images']!=manifest['counts']['train']):
        raise ValueError('TRAIN gate mean and feature cache differ')
    return torch.from_numpy(mean).float()
