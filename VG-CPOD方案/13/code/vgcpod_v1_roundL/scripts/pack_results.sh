#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
"$CPF_PY" run.py report --work "$CPF_WORK"
# All archive member names are ASCII, including Markdown file names.
"$CPF_PY" scripts/pack_results.py
