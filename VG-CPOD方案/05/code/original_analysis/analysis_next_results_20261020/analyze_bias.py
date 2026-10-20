"""Compare VAL-native vs VAL-exact calibration with fixed saved TEST scores."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
NEW = ROOT/'server_results_next_20261020'
torch.set_num_threads(1)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    manifest = read(ROOT/'server_results_20261018/vgcpod_v1_runs/cache_ut_clip_l14/features_manifest.json')
    labels = manifest['labels']
    train = set(map(tuple,manifest['pairs']['train']))
    rows = []
    for case in ('A5','C2','D1','D1_mean'):
        for seed in (3351,3352,3353):
            if case=='A5':
                run = ROOT/'server_results_roundA_confirm_20261019/roundA_20261019/A5/ut-zap50k/V1'/('seed%d'%seed)
            elif case=='D1_mean':
                run = NEW/'roundD_20261020/mean_gate/D1'/('seed%d'%seed)/'train_mean'
            else:
                run = NEW/('roundC_20261020' if case=='C2' else 'roundD_20261020')/case/'ut-zap50k/V1'/('seed%d'%seed)
            metrics = read(run/'metrics.json')['metrics']
            for world in ('cw','ow'):
                with np.load(run/'predictions'/('test_'+world+'.npz')) as d:
                    pairs = list(map(tuple,d['candidate_pairs'].tolist()))
                    index = {p:i for i,p in enumerate(pairs)}
                    seen = np.array([p in train for p in pairs])
                    truth = np.array([index[(labels['attrs'][a],labels['objs'][o])]
                        for a,o in zip(d['attribute_truth'],d['object_truth'])])
                    sm = seen[truth]
                    for method in ('native','exact'):
                        scores = torch.from_numpy(d['log_probabilities'].copy())
                        bias = metrics['val_'+world]['bias_'+method]
                        scores[:,~torch.from_numpy(seen)] += bias
                        correct = scores.topk(1,dim=1).indices[:,0].numpy()==truth
                        s,u = correct[sm].mean(),correct[~sm].mean()
                        rows.append(dict(case=case,seed=seed,world=world,bias_method=method,
                            bias=bias,seen=float(100*s),unseen=float(100*u),hm=float(200*s*u/(s+u))))
    with (OUT/'tables/val_bias_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Bias comparison rows:',len(rows))


if __name__=='__main__':
    main()
