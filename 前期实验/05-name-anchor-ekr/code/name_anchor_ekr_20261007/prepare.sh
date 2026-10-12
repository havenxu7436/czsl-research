#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$CPF_PY" "$BUNDLE/verify_release.py"
"$CPF_PY" -c 'import sys,torch,torchvision; from torch.nn.modules.pooling import MaxPool2d; assert sys.version_info[:2]==(3,8); assert torch.__version__.startswith("1.13.1"); assert torchvision.__version__.startswith("0.14.1"); assert torch.cuda.is_available(); print("NAME ENV PASSED",sys.executable,torch.cuda.get_device_name(0))'
df -h -- "$NAME_PREVIOUS" /var/tmp
"$CPF_PY" - "$NAME_WORK" <<'PY'
import shutil,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve()
assert p.parent.is_dir(), 'Parent directory is absent'
assert shutil.disk_usage(str(p.parent)).free >= 5*1024**3, 'Need at least 5 GiB for this staged exploration'
PY
"$CPF_PY" "$BUNDLE/install.py" --source "$NAME_PREVIOUS/src" --destination "$NAME_WORK/src"
cd -- "$NAME_WORK/src"
"$CPF_PY" -u -m transition.name_plan prepare --work "$NAME_WORK" --previous "$NAME_PREVIOUS"
printf 'PREPARED. Next: bash %s/replay_scores.sh\n' "$BUNDLE"
