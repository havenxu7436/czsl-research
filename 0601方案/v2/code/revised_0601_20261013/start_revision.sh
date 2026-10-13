#!/usr/bin/env bash
# Fresh protocol only. Never terminates old jobs or edits old results.
set -euo pipefail
release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cpf_python=/data/xhw/miniconda3/envs/cpf/bin/python
revision_root=/data/xhw/czsl_repro
revision_work=/var/tmp/czsl_0601_revision_20261013
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
revision_session="rev0601_${revision_suite}_1013"
revision_driver="${revision_work}_driver.log"
revision_export="${revision_work}.zip"
if [ -e "$revision_driver" ] || [ -e "$revision_export" ] || tmux has-session -t "$revision_session" 2>/dev/null; then
  printf 'STOP: preserve existing driver/export/session.\n' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
"$cpf_python" - "$release_dir" "$revision_work" "$revision_suite" "$revision_seeds" <<'PY'
import sys, shutil
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from prepare_revision import CORE,SMALL
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
