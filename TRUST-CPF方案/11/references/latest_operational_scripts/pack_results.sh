#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u scripts/pack_results.py
