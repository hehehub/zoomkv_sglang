#!/usr/bin/env bash
# One-shot smoke test: Qwen3-4B, TP=2, batch=2, ZoomKV offload.
# Default input is long enough to leave the dense-prefill threshold and short
# enough to finish on 40GB A100s. Override with INPUT_LEN=131072 for 128k.
set -euo pipefail

ROOT=/root/zoomkv_sglang
MODEL=${MODEL:-$ROOT/models/Qwen3-4B}
PORT=${PORT:-33041}
INPUT_LEN=${INPUT_LEN:-4096}
OUTPUT_LEN=${OUTPUT_LEN:-32}
BATCH=2
TP=2
OUT_DIR=${OUT_DIR:-/root/zoomkv_e2e}
LOG="$OUT_DIR/smoke_4b_tp2_bs2_server.log"
JSONL="$OUT_DIR/smoke_4b_tp2_bs2.jsonl"
CONTEXT=$((INPUT_LEN + OUTPUT_LEN + 256))
MAX_TOKENS=$((INPUT_LEN * BATCH + OUTPUT_LEN * BATCH + 4096))
SERVER_PID=""

set +u
source /root/env/zoomkv_sglang/bin/activate
set -u
export LD_LIBRARY_PATH="/root/env/zoomkv_sglang/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/root/env/zoomkv_sglang
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
cd "$ROOT"
mkdir -p "$OUT_DIR"

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  pkill -f "sglang.launch_server" 2>/dev/null || true
  sleep 2
}
trap stop_server EXIT

echo "model=$MODEL tp=$TP batch=$BATCH input=$INPUT_LEN output=$OUTPUT_LEN gpus=$CUDA_VISIBLE_DEVICES"
stop_server
: >"$LOG"

YARN=()
ALLOW_LONG=0
if (( CONTEXT > 40960 )); then
  ALLOW_LONG=1
  YARN=(--json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4,"original_max_position_embeddings":40960}}')
fi

env \
  SGLANG_PLUGINS=sglang_zoomkv \
  SGLANG_ZOOMKV_OFFLOAD=true \
  SGLANG_ZOOMKV_PREFILL_ALL_FIRST=1 \
  SGLANG_ZOOMKV_PREFILL_COHORT="$BATCH" \
  SGLANG_ZOOMKV_PREFILL_DENSE_LAYERS="${SGLANG_ZOOMKV_PREFILL_DENSE_LAYERS:-4}" \
  SGLANG_ZOOMKV_PREFILL_BLOCK="${SGLANG_ZOOMKV_PREFILL_BLOCK:-1024}" \
  SGLANG_ZOOMKV_FINAL_TOPK=100 \
  SGLANG_ZOOMKV_FUSED_QUEST=0 \
  SGLANG_ZOOMKV_QUEST_GRAPH=1 \
  SGLANG_ZOOMKV_BREAK_DECODE_GRAPH=0 \
  SGLANG_ZOOMKV_MAX_NEW_TOKENS="$OUTPUT_LEN" \
  SGLANG_ZOOMKV_FALLBACK_BACKEND=flashinfer \
  POLAR_QUEST_LARGE_RATIO=0.3 \
  POLAR_QUEST_SMALL_RATIO=0.3 \
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="$ALLOW_LONG" \
  python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 \
    --context-length "$CONTEXT" \
    --max-total-tokens "$MAX_TOKENS" \
    --mem-fraction-static 0.75 \
    --prefill-attention-backend zoomkv \
    --decode-attention-backend zoomkv \
    --cuda-graph-backend-decode full \
    --cuda-graph-max-bs-decode "$BATCH" \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-prefill-tokens "$INPUT_LEN" \
    --max-running-requests "$BATCH" \
    --tp-size "$TP" \
    --page-size 1 \
    --disable-custom-all-reduce \
    "${YARN[@]}" \
    >>"$LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 900); do
  if grep -q "The server is fired up and ready to roll" "$LOG"; then
    ready=1
    break
  fi
  if grep -qE "CUDA out of memory|Scheduler hit an exception" "$LOG"; then
    echo "FAIL server error, see $LOG"
    tail -n 40 "$LOG"
    exit 1
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL server exited, see $LOG"
    tail -n 40 "$LOG"
    exit 1
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "FAIL server not ready, see $LOG"
  tail -n 40 "$LOG"
  exit 1
fi
echo "server ready"

set +e
python -m sglang.benchmark.serving \
  --backend sglang --host 127.0.0.1 --port "$PORT" \
  --dataset-name random-ids --tokenize-prompt \
  --random-range-ratio 1.0 --seed 42 \
  --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$BATCH" --max-concurrency "$BATCH" \
  --warmup-requests 0 \
  --tag smoke_4b_tp2_bs2 \
  --output-file "$JSONL"
client=$?
set -e

echo "-------- server markers --------"
grep -E "sparse prefill|holding decode|parked prefill|decode CUDA graph|Scheduler hit an exception|CUDA out of memory" "$LOG" | tail -n 40 || true
if [[ "$client" != "0" ]]; then
  echo "FAIL client exit $client, see $LOG and $JSONL"
  exit 1
fi
if ! grep -q "sparse prefill" "$LOG"; then
  echo "FAIL client finished but sparse prefill did not run, see $LOG"
  exit 1
fi
echo "PASS 4B tp=$TP batch=$BATCH input=$INPUT_LEN"
echo "log=$LOG"
echo "jsonl=$JSONL"
