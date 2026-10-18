#!/usr/bin/env bash
# Override these variables before sourcing/running scripts. Paths follow the archived server.
export CPF_PY="${CPF_PY:-python}"
export VG_DATA="${VG_DATA:-/data/xhw/czsl_repro/data/ut-zap50k}"
export VG_CLIP="${VG_CLIP:-/data/xhw/.cache/clip/ViT-L-14.pt}"
export VG_WORK="${VG_WORK:-/data/xhw/vgcpod_v1_runs}"
export VG_CACHE="${VG_CACHE:-$VG_WORK/cache_ut_clip_l14}"
export VG_OUT="${VG_OUT:-$VG_WORK/main}"
export VG_GPU="${VG_GPU:-0}"
export CUDA_VISIBLE_DEVICES="$VG_GPU"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
