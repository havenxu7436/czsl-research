#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py setup --source /data/xhw/experiments/trustcpf_pure_v1_10ep
mkdir -p logs
echo TRUST_G0_READY_GPUS_0_1_2
