"""Actual completed results only; same-budget paired comparisons."""
import argparse
import math
import re
from pathlib import Path
import numpy as np
from common import read, write, sha


def stats(values):
    return dict(n=len(values),mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)) if len(values)>1 else None)


def summarize(work):
    work = Path(work)
    p = read(work/'protocol.json')
    states, rows, curves = [], [], {}
    for j in p['jobs']:
        run = Path(j['run'])
        state = read(run/'status.json') if (run/'status.json').exists() else {'state':'not_started'}
        states.append(dict(variant=j['variant'],base_variant=j['base_variant'],seed=j['seed'],tier=j['tier'],**state))
        if state['state'] != 'complete': continue
        if state.get('epochs') != p['epochs']: raise ValueError('Mismatched budget')
        text = (run/'train.log').read_text(encoding='utf-8',errors='replace')
        curves[j['variant']] = [{'epoch':int(e),'VAL_CW_AUC_pct':100*float(v)} for e,v in
                   re.findall(r'Val/Test Epoch (\d+): AUC=([\d.eE+-]+)',text)]
        identities = set()
        for phase in ('val','test'):
            for setting in ('cw','ow'):
                m = read(run/('eval_'+phase+'_'+setting)/'metrics.json')
                identities.add(m['checkpoint_sha256'])
                values = {k+'_pct':100*float(m['metrics_raw'][k]) for k in ('AUC','best_hm','best_seen','best_unseen')}
                if not all(math.isfinite(v) for v in values.values()): raise ValueError('Nonfinite metric')
                rows.append(dict(variant=j['variant'],base_variant=j['base_variant'],seed=j['seed'],phase=phase,setting=setting,
                                 best_epoch=m['best_epoch'],checkpoint_sha256=m['checkpoint_sha256'],**values))
        if len(identities) != 1: raise ValueError('Four evaluations did not share checkpoint')
    aggregates = []
    for base in sorted({r['base_variant'] for r in rows}):
        for phase in ('val','test'):
            for setting in ('cw','ow'):
                group = [r for r in rows if r['base_variant']==base and r['phase']==phase and r['setting']==setting]
                if group: aggregates.append(dict(variant=base,phase=phase,setting=setting,seeds=[r['seed'] for r in group],
                    AUC_pct=stats([r['AUC_pct'] for r in group]),HM_pct=stats([r['best_hm_pct'] for r in group])))
    lookup = {(r['base_variant'],r['seed'],r['phase'],r['setting']):r for r in rows}
    comparisons = [('Rfull','A_native'),('Rtext','R0'),('Rprior','R0'),('Rmvp','Rtext'),
                   ('Rexternal','R0'),('Rmvp_logic','Rmvp'),('Rfeature','Rmvp'),('Rfull','Rfeature'),
                   ('Rfull','Rexternal'),('Rfull','Rmvp_logic'),('R0','R0_native_attention')]
    comparisons += [('Rfull',name) for name in sorted({j['base_variant'] for j in p['jobs']}) if name.startswith('Rfull_')]
    differences = []
    for left,right in comparisons:
        for phase in ('val','test'):
            for setting in ('cw','ow'):
                pairs = [(s,lookup.get((left,s,phase,setting)),lookup.get((right,s,phase,setting))) for s in p['seeds']]
                pairs = [(s,x,y) for s,x,y in pairs if x and y]
                if pairs: differences.append(dict(comparison=left+'-'+right,phase=phase,setting=setting,seeds=[s for s,x,y in pairs],
                    AUC_pp=stats([x['AUC_pct']-y['AUC_pct'] for s,x,y in pairs]),HM_pp=stats([x['best_hm_pct']-y['best_hm_pct'] for s,x,y in pairs])))
    core = all(s['state']=='complete' for s in states if s['tier']==0)
    done = all(s['state']=='complete' for s in states)
    result = dict(core_complete=core,all_planned_complete=done,states=states,rows=rows,aggregates=aggregates,
                  paired_differences=differences,val_curves=curves,not_covered=p['not_covered'],protocol_sha256=sha(work/'protocol.json'))
    write(work/'summary/summary.json',result)
    def display(s):
        return '{:.4f} ± {:.4f}'.format(s['mean'],s['sample_std']) if s['sample_std'] is not None else '{:.4f} (n=1)'.format(s['mean'])
    lines = ['# 0601 统一修订版：实际结果','', '这是公式修订后的新模型，不与旧 r2 / 30epoch 结果混合计算均值。', '',
             '全部公式统一修正；{}epoch；seeds={}；suite={}；完整={}。'.format(p['epochs'],p['seeds'],p['suite'],done), '',
             '|组|划分|设置|n|AUC (%)|HM (%)|','|---|---|---|---:|---|---|']
    lines += ['|{}|{}|{}|{}|{}|{}|'.format(r['variant'],r['phase'],r['setting'],len(r['seeds']),display(r['AUC_pct']),display(r['HM_pct'])) for r in aggregates]
    lines += ['', '## 配对差分（百分点）','', '|比较|划分|设置|n|ΔAUC|ΔHM|','|---|---|---|---:|---|---|']
    lines += ['|{}|{}|{}|{}|{}|{}|'.format(r['comparison'],r['phase'],r['setting'],len(r['seeds']),display(r['AUC_pp']),display(r['HM_pp'])) for r in differences]
    lines += ['', '## 边界', '', p['formula'], '', p['logic'], '', p['knowledge'], '', p['test_policy'], '',
              'AUC/HM 的 bias 曲线是评测统计，不等于可以在 TEST 上挑部署 bias。', '', '未完成/未覆盖：']
    lines += ['- '+v for v in p['not_covered']]
    (work/'summary/0601_统一修订版训练汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return core,done


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    summarize(p.parse_args().work)
