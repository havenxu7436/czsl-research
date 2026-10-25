#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
exec "$CPF_PY" scripts/run_queue.py "$@"

