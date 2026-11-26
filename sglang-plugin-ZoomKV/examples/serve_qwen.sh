#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PORT="${PORT:-30000}"

export SGLANG_PLUGINS="${SGLANG_PLUGINS:-sglang_zoomkv}"
export SGLANG_ZOOMKV_FALLBACK_BACKEND="${SGLANG_ZOOMKV_FALLBACK_BACKEND:-flashinfer}"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port "${PORT}" \
  --prefill-attention-backend "${SGLANG_ZOOMKV_FALLBACK_BACKEND}" \
  --decode-attention-backend zoomkv \
  --disable-cuda-graph \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --max-running-requests 1 \
  --tp-size 1

