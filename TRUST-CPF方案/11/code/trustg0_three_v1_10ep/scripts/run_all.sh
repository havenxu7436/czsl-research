#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
if [ "$#" -eq 0 ]; then set -- 0 1 2; fi
exec python -u scripts/run_queue.py "$@"
