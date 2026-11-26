#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/zoomkv_sglang/models/Qwen3-8B}"
PORT="${PORT:-30000}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="127.0.0.1,localhost,::1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# This machine's PyTorch is cu129 while its newest installed toolkit is 12.4.
# CUDA 12.2 cannot compile the current PyTorch C++20 headers.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CC="${CC:-/usr/bin/g++-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++-11}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin /usr/bin/g++-11}"

export SGLANG_PLUGINS="${SGLANG_PLUGINS:-sglang_zoomkv}"
export SGLANG_ZOOMKV_OFFLOAD="${SGLANG_ZOOMKV_OFFLOAD:-false}"
export SGLANG_ZOOMKV_MAX_NEW_TOKENS="${SGLANG_ZOOMKV_MAX_NEW_TOKENS:-256}"
export SGLANG_ZOOMKV_FALLBACK_BACKEND="${SGLANG_ZOOMKV_FALLBACK_BACKEND:-flashinfer}"

sglang serve \
  --model-path "${MODEL_PATH}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --context-length 1024 \
  --max-total-tokens 65536 \
  --mem-fraction-static 0.70 \
  --prefill-attention-backend flashinfer \
  --decode-attention-backend zoomkv \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --max-prefill-tokens 32768 \
  --max-running-requests 64 \
  --tp-size 2 \
  --page-size 1 \
  --disable-custom-all-reduce
