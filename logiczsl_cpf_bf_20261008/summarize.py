#!/usr/bin/env python3
"""Collect paired VAL results and actual logic logs into Markdown/CSV."""
import argparse
import csv
import io
import json
import re
from pathlib import Path
from common import write_new


def summarize(work, seeds, out):
    work, out = Path(work), Path(out)
    rows = []
    messages = []
    for seed in seeds:
        protocol = json.loads((work / 'experiments/protocol' / f'seed_{seed}.json').read_text(encoding='utf-8'))
        for variant, spec in protocol['configs'].items():
            run = Path(spec['run'])
            if not (run / 'status.txt').is_file() or (run / 'status.txt').read_text().strip() != 'train_and_val_complete':
                messages.append(f'seed={seed} {variant}: incomplete; no efficacy conclusion')
                continue
            training_log = (run / 'train.log').read_text(encoding='utf-8', errors='replace')
            active = re.findall(r'BF_LOGIC epoch=(\d+) iter=(\d+) cpf=([^ ]+) oi=([^ ]+) oe=([^ ]+) ae=([^ ]+) weighted=([^\s]+)', training_log)
            if variant in ('B', 'F'):
                messages.append(f'seed={seed} {variant}: {len(active)} logged logic observations (not every batch)')
                if active:
                    last = active[-1]
                    messages.append(f'  last: epoch={last[0]}, cpf={last[2]}, oi={last[3]}, oe={last[4]}, ae={last[5]}, weighted={last[6]}')
            for setting in ('cw', 'ow'):
                result = json.loads((run / ('eval_val_' + setting) / 'metrics.json').read_text(encoding='utf-8'))
                if result['phase'] != 'val' or result['seed'] != seed or result['variant'] != variant:
                    raise ValueError('Result provenance mismatch')
                metrics = result['metrics_raw']
                rows.append({'seed': seed, 'variant': variant, 'setting': setting,
                             **{name: 100 * metrics[name] for name in ('AUC','best_hm','best_seen','best_unseen')},
                             'best_epoch': result['best_epoch'], 'checkpoint_sha256': result['checkpoint_sha256']})
    lines = ['# CPF + LOGICZSL B/F actual training and VAL summary', '',
             'These are reconstructed-knowledge, native CPF results, not original Logic-Troika paper scores.',
             'First-round protocol: 10 epochs, beta=0.01, q=3; F keeps legacy lambda=0.1. No TEST metrics.', '',
             '| seed | variant | setting | AUC% | HM% | Seen% | Unseen% | best epoch |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['variant']} | {row['setting']} | {row['AUC']:.3f} | {row['best_hm']:.3f} | {row['best_seen']:.3f} | {row['best_unseen']:.3f} | {row['best_epoch']} |")
    lines += ['', '## Paired differences (percentage points)', '',
              '| seed | setting | B - A_ref AUC | F - E_ref AUC | F - A_ref AUC |',
              '|---|---|---:|---:|---:|']
    differences = []
    for seed in seeds:
        for setting in ('cw', 'ow'):
            mapping = {r['variant']: r for r in rows if r['seed'] == seed and r['setting'] == setting}
            if set(mapping) != {'A_ref', 'B', 'E_ref', 'F'}:
                continue
            delta = [mapping['B']['AUC'] - mapping['A_ref']['AUC'],
                     mapping['F']['AUC'] - mapping['E_ref']['AUC'],
                     mapping['F']['AUC'] - mapping['A_ref']['AUC']]
            differences.append((setting, delta))
            lines.append(f'| {seed} | {setting} | {delta[0]:+.3f} | {delta[1]:+.3f} | {delta[2]:+.3f} |')
    if len(seeds) > 1:
        lines += ['', 'Mean paired differences are descriptive only; n<3 is not multi-seed confirmation.']
        for setting in ('cw','ow'):
            values = [d for s,d in differences if s == setting]
            if len(values) == len(seeds):
                mean = [sum(v[i] for v in values)/len(values) for i in range(3)]
                lines.append(f"{setting}: n={len(values)}, B-A_ref={mean[0]:+.3f}, F-E_ref={mean[1]:+.3f}, F-A_ref={mean[2]:+.3f} pp.")
    lines += ['', '## Training evidence / incomplete jobs', ''] + [f'- {m}' for m in messages]
    lines += ['', '## Interpretation limits', '',
              '- Positive F-E_ref with F-A_ref <= 0 means logic may help E but does not establish an overall improvement over CPF.',
              '- Single-seed changes do not prove stable improvement. Keep beta/knowledge unchanged before paired seed 0 and 2026 runs.',
              '- Do not compare these native CPF/GloVe VAL values directly against original CLIP-based TEST values.',
              '- The old static calibration/text findings are not evidence that this logic-loss combination succeeds or fails.', '']
    write_new(out, '\n'.join(lines))
    fields = ['seed','variant','setting','AUC','best_hm','best_seen','best_unseen','best_epoch','checkpoint_sha256']
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    write_new(out.with_suffix('.csv'), buffer.getvalue())
    print('SUMMARY_READY', out, 'completed_setting_rows=', len(rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=Path('/var/tmp/czsl_xhw_bf_20261008'))
    p.add_argument('--seeds', type=int, nargs='+', default=[3351])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    summarize(a.work, a.seeds, a.out)
