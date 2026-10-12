"""Run isolated Stage A checks in separate native-source processes, then export."""
import argparse
import subprocess
import sys
from pathlib import Path

from common import ROOT, new_output, read_json, verify_release, write_json, zip_destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    args = parser.parse_args()
    verify_release()
    ref = read_json(ROOT / 'reference.json')
    args.export = zip_destination(args.export, ref, [args.work])
    work = new_output(args.work, ref, free_mib=2048)
    codes = {}
    for name, script, target in (('bf_probe', 'probe_bf.py', 'bf_probe'), ('prepare', 'prepare.py', 'base')):
        cmd = [sys.executable, str(ROOT / script), '--out', str(work / target)]
        print('RUN_CHECK', name, flush=True)
        with (work / (name + '.log')).open('x', encoding='utf-8') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        codes[name] = result.returncode
        # Keep forensic evidence even for a pre-output guard failure, never claim pass.
        if not (work / target).exists():
            (work / target).mkdir()
            write_json(work / target / 'failure.json', dict(error='Child failed before creating output', return_code=result.returncode))
        print('CHECK_EXIT', name, result.returncode, flush=True)
    write_json(work / 'driver.json', dict(return_codes=codes, formal_training_started=False,
                                        original_experiments_overwritten=False))
    cmd = [sys.executable, str(ROOT / 'export_results.py'), '--base', str(work / 'base'),
           '--probe', str(work / 'bf_probe'), '--logs', str(work), '--out', str(args.export)]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    if any(codes.values()):
        print('STAGE_A_FAILED_DIAGNOSTICS_EXPORTED: download ZIP; do not start formal training.', flush=True)
        raise SystemExit(1)
    print('STAGE_A_COMPLETE: download ZIP for audit before formal pilot.', flush=True)


if __name__ == '__main__':
    main()
