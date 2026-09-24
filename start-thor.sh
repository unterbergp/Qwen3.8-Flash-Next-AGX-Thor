#!/usr/bin/env bash
# Jetson AGX Thor profile with native context. Explicit environment wins.
# The model, credentials, image and port still come from .env.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DOCKER_GPU_RUNTIME="${DOCKER_GPU_RUNTIME:-nvidia}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export YARN="${YARN:-0}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
export MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-0}"
# Thor/Marlin warmup needs more overhead than the Spark estimate (5.6 GiB).
export OVERHEAD_GIB="${OVERHEAD_GIB:-14}"
# Native context needs about 7.2 GiB BF16 KV; allow budget rounding headroom.
export KV_TARGET_GIB="${KV_TARGET_GIB:-8}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-NONE}"
export COMPILATION_MODE="${COMPILATION_MODE:-0}"
# The image auto-selects CUTLASS MoE, which failed warmup on Thor.
export MOE_BACKEND="${MOE_BACKEND:-marlin}"
export GDN_DECODE_KERNEL="${GDN_DECODE_KERNEL:-triton}"
exec bash "$SCRIPT_DIR/start.sh" "$@"
