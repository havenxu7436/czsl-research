import csv
from pathlib import Path
import numpy as np
from .common import read_json,write_json


def report(args):
    root=Path(args.work); rows=[]
    for file in sorted(root.rglob('metrics.json')):
        value=read_json(file)
        if 'variant' not in value: continue
        for phase in ('val','test'):
            for world,m in value[phase].items():
                rows.append({'variant':value['variant'],'seed':value['seed'],'epoch':value['epoch'],
                    'phase':phase,'world':world,'AUC_native':m['AUC_native'],'AUC_exact':m['AUC_exact'],
                    'HM_oracle_native':m['HM_oracle_native'],'HM_at_val_bias':m['at_val_bias']['HM'],
                    'unseen_at_val_bias':m['at_val_bias']['unseen'],'NLL':m['calibrated']['NLL'],
                    'ECE':m['calibrated']['ECE'],'graph_variance':m['graph_variance']})
    if not rows: raise ValueError('No completed result rows under '+str(root))
    with (root/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    comparisons=[]
    for phase in ('val','test'):
        for world in ('cw','ow'):
            group=[r for r in rows if r['phase']==phase and r['world']==world]
            full={r['seed']:r for r in group if r['variant']=='robust'}
            for control in sorted(set(r['variant'] for r in group)-{'robust'}):
                other={r['seed']:r for r in group if r['variant']==control}
                seeds=sorted(set(full)&set(other)); diffs=[full[s]['AUC_native']-other[s]['AUC_native'] for s in seeds]
                if diffs: comparisons.append({'phase':phase,'world':world,'control':control,'seeds':seeds,
                    'differences_pp':diffs,'mean_pp':float(np.mean(diffs)),
                    'std_pp':float(np.std(diffs,ddof=1)) if len(diffs)>1 else 0.})
    write_json(root/'paired_comparisons.json',comparisons)
    noise_rows=[]
    for file in sorted(root.rglob('noise_*.json')):
        groups={}
        for value in read_json(file):
            for world in ('cw','ow'):
                key=(value['seed'],value['type'],value['kind'],world)
                groups.setdefault(key,[]).append((value['rate'],value['test'][world]['AUC_native']))
        for (seed,ty,kind,world),curve in groups.items():
            curve=sorted(set(curve)); xx,yy=zip(*curve)
            if len(xx)<2:continue
            trapz=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
            noise_rows.append({'variant':file.parent.name,'noise_seed':seed,'type':ty,'kind':kind,'world':world,
                'max_noise_rate':max(xx),'RAUC':float(trapz(yy,xx)/(max(xx)-min(xx)))})
    write_json(root/'robustness_summary.json',noise_rows)
    print('REPORT',root/'summary.csv',flush=True)
