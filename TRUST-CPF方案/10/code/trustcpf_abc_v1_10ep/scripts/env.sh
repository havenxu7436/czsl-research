#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f server_env.sh ]]; then
  printf '%s\n' 'Run python run.py setup --source ... --cache ... --work ... first.'
  exit 2
fi
source server_env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD"

