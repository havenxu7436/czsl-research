"""Read-only status and empirical epoch timing; no GPU/control mutations."""
import argparse
import re
import statistics
from pathlib import Path
from common import read


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--seeds', default='all')
    a = p.parse_args()
    protocol = read(a.work / 'protocol.json')
    seeds = set(protocol['seeds']) if a.seeds == 'all' else {int(s) for s in a.seeds.split(',')}
    complete = 0
    print('JOB                         GPU STATE         EPOCH     MEDIAN_MIN/EPOCH')
    selected = [j for j in protocol['jobs'] if j['seed'] in seeds]
    for j in selected:
        run = Path(j['run'])
        s = read(run / 'status.json') if (run / 'status.json').exists() else {'state': 'not_started'}
        log = (run / 'train.log').read_text(encoding='utf-8', errors='replace') if (run / 'train.log').exists() else ''
        epochs = [(int(e), float(t)) for e, t in re.findall(r'FULL_EPOCH_FINISHED epoch=(\d+) seconds=([\d.]+)', log)]
        finished = max([e for e, t in epochs] or [0])
        median = '{:.2f}'.format(statistics.median([t for e, t in epochs[-5:]]) / 60) if epochs else '-'
        complete += s['state'] == 'complete'
        print('{:<27} {:<3} {:<13} {:>2}/{:<3}    {}'.format(j['variant'], str(s.get('gpu', '-')), s['state'], finished, protocol['epochs'], median))
    print('SELECTED_COMPLETE {}/{}; planned_all_seeds={} jobs={}'.format(complete, len(selected), protocol['seeds'], len(protocol['jobs'])))
    print('Epoch timing includes TRAIN+VAL+durable checkpoint; smoke/startup/final CW+OW evaluation are extra.')


if __name__ == '__main__':
    main()
