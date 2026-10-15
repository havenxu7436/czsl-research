#!/usr/bin/env bash
# Fresh protocol only. Never terminates old jobs or edits old results.
set -euo pipefail
release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cpf_python=/data/xhw/miniconda3/envs/cpf/bin/python
revision_root=/data/xhw/czsl_repro
revision_work=/var/tmp/czsl_0601_revision_20261014_fix
revision_suite=all
revision_seeds=3351,3352,3353
revision_gpus=1,2,3
revision_workers=3
while [ "$#" -gt 0 ]; do
  case "$1" in
    --work) revision_work="$2"; shift 2;;
    --suite) revision_suite="$2"; shift 2;;
    --seeds) revision_seeds="$2"; shift 2;;
    --gpus) revision_gpus="$2"; shift 2;;
    --max-workers) revision_workers="$2"; shift 2;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2;;
  esac
done
case "$revision_work" in
  /var/tmp/czsl_0601_revision_*|/data0/*/czsl_0601_revision_*) ;;
  *) printf 'Use a new explicitly named /var/tmp/czsl_0601_revision_* or /data0/<user>/czsl_0601_revision_* work directory.\n' >&2; exit 2;;
esac
case "$revision_suite" in core|small|all) ;; *) printf 'Use core,small,all. External suite requires independently supplied feasibility JSON; see README.\n' >&2; exit 2;; esac
command -v tmux >/dev/null
test -x "$cpf_python"
if [ -e "$revision_work" ]; then
  printf 'STOP: preserve existing work: %s. See explicit resume commands in README.\n' "$revision_work" >&2
  exit 1
fi
revision_session="rev0601_${revision_suite}_1014_fix"
revision_driver="${revision_work}_driver.log"
revision_export="${revision_work}.zip"
if [ -e "$revision_driver" ] || [ -e "$revision_export" ] || tmux has-session -t "$revision_session" 2>/dev/null; then
  printf 'STOP: preserve existing driver/export/session.\n' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
"$cpf_python" - "$release_dir" "$revision_work" "$revision_suite" "$revision_seeds" "$revision_gpus" "$revision_workers" <<'PY'
import sys, shutil, subprocess
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from prepare_revision import CORE,SMALL
from common import verify_release
verify_release(Path(sys.argv[1]))
seeds=[int(s) for s in sys.argv[4].split(',')]
if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0:
    raise SystemExit('STOP: seeds must be distinct nonnegative integers')
workers=int(sys.argv[6])
if workers<1: raise SystemExit('STOP: max-workers must be positive')
indices={int(s.strip()) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits'],text=True).splitlines() if s.strip()}
requested=indices if sys.argv[5]=='auto' else {int(s) for s in sys.argv[5].split(',')}
if not requested or not requested.issubset(indices):
    raise SystemExit('STOP: requested GPU index does not exist')
sys.path.insert(0,str(Path(sys.argv[1])/'source'))
import train  # Import the actual training entry and its runtime dependencies.
import torch
if not torch.cuda.is_available(): raise SystemExit('STOP: CPF environment cannot access CUDA')
print('RUNTIME_PREFLIGHT_OK torch={} gpus={} workers={}'.format(torch.__version__,sorted(requested),workers),flush=True)
work=Path(sys.argv[2]); parent=work.parent
while not parent.exists(): parent=parent.parent
n=(len(CORE) if sys.argv[3]=='core' else 1+len(SMALL) if sys.argv[3]=='small' else len(CORE)+len(SMALL))*len(sys.argv[4].split(','))
needed=10+2*n
free=shutil.disk_usage(parent).free/(1024**3)
print('DISK_PREFLIGHT jobs={} free_GiB={:.1f} required_GiB={} mount_path={}'.format(n,free,needed,parent),flush=True)
if free<needed: raise SystemExit('STOP: insufficient disk for the selected suite; choose another writable mount or a smaller separately named experiment suite. All formula fixes remain identical.')
PY
"$cpf_python" "$release_dir/test_revision.py"
"$cpf_python" "$release_dir/prepare_revision.py" --root "$revision_root" --work "$revision_work" --suite "$revision_suite" --epochs 10 --seeds "$revision_seeds"
printf -v revision_command '%q -u %q --work %q --gpus %q --max-workers %q --export %q > %q 2>&1' \
  "$cpf_python" "$release_dir/run_suite.py" "$revision_work" "$revision_gpus" "$revision_workers" "$revision_export" "$revision_driver"
tmux new-session -d -s "$revision_session" "$revision_command"
printf 'STARTED session=%s\nWORK=%s\nLOG=%s\nEXPORT=%s\n' "$revision_session" "$revision_work" "$revision_driver" "$revision_export"
