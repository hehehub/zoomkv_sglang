#!/usr/bin/env bash
# Four prefill timings on Qwen3-4B, TP=2, batch=2:
#   baseline  dense FlashInfer, copy KV after the forward
#   overlap   dense FlashInfer, copy KV while the next layer runs
#   sparse    ZoomKV sparse prefill, copy KV after the forward
#   both      ZoomKV sparse prefill and overlapped copy
# Default input is 128k tokens. The number to compare is total_ms: enter
# prefill until that request's retrieval KV copy has finished. Two requests
# are prefilled one after another.
set -u

ROOT=/root/zoomkv_sglang
MODEL=${MODEL:-$ROOT/models/Qwen3-4B}
PORT=${PORT:-33041}
INPUT_LEN=${INPUT_LEN:-131072}
OUTPUT_LEN=${OUTPUT_LEN:-32}
BATCH=2
TP=2
OUT_DIR=${OUT_DIR:-/root/zoomkv_e2e}
SUMMARY="$OUT_DIR/prefill_ablation_4b_summary.txt"
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
  if [[ -n "${SERVER_PID}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  pkill -f "sglang.launch_server" 2>/dev/null || true
  sleep 3
}
trap stop_server EXIT

run_one() {
  local mode="$1"
  local overlap="$2"
  local dense="$3"
  local prefill_backend="$4"
  local log="$OUT_DIR/prefill_ablation_${mode}.log"
  local jsonl="$OUT_DIR/prefill_ablation_${mode}.jsonl"
  local context=$((INPUT_LEN + OUTPUT_LEN + 256))
  local max_tokens=$((INPUT_LEN * BATCH + OUTPUT_LEN * BATCH + 4096))
  local yarn=()
  local allow_long=0
  # This Qwen3-4B checkpoint is trained to 40960. SGLang ignores the YaRN
  # factor when original_max_position_embeddings is set, so a longer request
  # has to be allowed explicitly. Factor 4 covers 163840 positions.
  if (( context > 40960 )); then
    allow_long=1
    yarn=(--json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4,"original_max_position_embeddings":40960}}')
  fi

  echo "======== ${mode}: overlap=${overlap} dense_layers=${dense} prefill=${prefill_backend} ========"
  stop_server
  : >"$log"

  env \
    SGLANG_PLUGINS=sglang_zoomkv \
    SGLANG_ZOOMKV_OFFLOAD=true \
    SGLANG_ZOOMKV_PREFILL_ALL_FIRST=1 \
    SGLANG_ZOOMKV_PREFILL_COHORT="$BATCH" \
    SGLANG_ZOOMKV_PREFILL_MODE="$mode" \
    SGLANG_ZOOMKV_PREFILL_OVERLAP="$overlap" \
    SGLANG_ZOOMKV_PREFILL_DENSE_LAYERS="$dense" \
    SGLANG_ZOOMKV_PREFILL_BLOCK="${SGLANG_ZOOMKV_PREFILL_BLOCK:-1024}" \
    SGLANG_ZOOMKV_FINAL_TOPK=100 \
    SGLANG_ZOOMKV_FUSED_QUEST=0 \
    SGLANG_ZOOMKV_QUEST_GRAPH=1 \
    SGLANG_ZOOMKV_BREAK_DECODE_GRAPH=0 \
    SGLANG_ZOOMKV_MAX_NEW_TOKENS="$OUTPUT_LEN" \
    SGLANG_ZOOMKV_FALLBACK_BACKEND=flashinfer \
    POLAR_QUEST_LARGE_RATIO=0.3 \
    POLAR_QUEST_SMALL_RATIO=0.3 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="$allow_long" \
    python -m sglang.launch_server \
      --model-path "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dtype bfloat16 \
      --context-length "$context" \
      --max-total-tokens "$max_tokens" \
      --mem-fraction-static 0.75 \
      --prefill-attention-backend "$prefill_backend" \
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
      "${yarn[@]}" \
      >>"$log" 2>&1 &
  SERVER_PID=$!

  local ready=0
  local _
  for _ in $(seq 1 900); do
    if grep -q "The server is fired up and ready to roll" "$log"; then
      ready=1
      break
    fi
    if grep -qE "CUDA out of memory|Scheduler hit an exception" "$log"; then
      echo "FAIL ${mode}: server error, see $log"
      tail -n 30 "$log"
      return 1
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "FAIL ${mode}: server exited, see $log"
      tail -n 30 "$log"
      return 1
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "FAIL ${mode}: server not ready, see $log"
    tail -n 30 "$log"
    return 1
  fi

  python -m sglang.benchmark.serving \
    --backend sglang --host 127.0.0.1 --port "$PORT" \
    --dataset-name random-ids --tokenize-prompt \
    --random-range-ratio 1.0 --seed 42 \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$BATCH" --max-concurrency "$BATCH" \
    --warmup-requests 0 \
    --tag "prefill_${mode}" \
    --output-file "$jsonl"
  local client=$?
  if [[ "$client" != "0" ]]; then
    echo "FAIL ${mode}: client exit ${client}, see $log"
    return 1
  fi
  if [[ "$mode" == "sparse" || "$mode" == "both" ]]; then
    if ! grep -q "sparse prefill" "$log"; then
      echo "FAIL ${mode}: sparse prefill did not run, see $log"
      return 1
    fi
  elif grep -q "sparse prefill" "$log"; then
    echo "FAIL ${mode}: sparse prefill ran but this mode should stay dense, see $log"
    return 1
  fi
  echo "OK ${mode}, log=$log"
  return 0
}

echo "model=$MODEL tp=$TP batch=$BATCH input=$INPUT_LEN output=$OUTPUT_LEN gpus=$CUDA_VISIBLE_DEVICES"
: >"$SUMMARY"
: >"$OUT_DIR/prefill_ablation_status.txt"

# mode overlap dense_layers prefill_backend
run_one baseline 0 999 flashinfer
echo "baseline $?" >>"$OUT_DIR/prefill_ablation_status.txt"
run_one overlap 1 999 zoomkv
echo "overlap $?" >>"$OUT_DIR/prefill_ablation_status.txt"
run_one sparse 0 4 zoomkv
echo "sparse $?" >>"$OUT_DIR/prefill_ablation_status.txt"
run_one both 1 4 zoomkv
echo "both $?" >>"$OUT_DIR/prefill_ablation_status.txt"

python3 - <<PY
import re
from pathlib import Path

out_dir = Path("$OUT_DIR")
input_len = $INPUT_LEN
modes = ["baseline", "overlap", "sparse", "both"]
names = {
    "baseline": "原始 prefill",
    "overlap": "只遮盖",
    "sparse": "只稀疏化",
    "both": "遮盖+稀疏化",
}
status = {}
status_path = out_dir / "prefill_ablation_status.txt"
if status_path.exists():
    for line in status_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            status[parts[0]] = parts[1]
pat = re.compile(
    r"prefill_time mode=(\S+) rank=(-?\d+) seq=(-?\d+) bs=(\d+) "
    r"overlap=(\d) sparse=(\d+) forward_ms=([0-9.]+) "
    r"offload_ms=([0-9.]+) total_ms=([0-9.]+)"
)
rows = []
print()
print(f"prefill 计时  input={input_len}  batch=2  tp=2")
print("total_ms：从进入这条 prefill，到这条请求的 KV 拷贝结束。")
print("两条请求串行，cohort 是两条之和，取两张卡里更慢的那张。")
print("遮盖打开时，拷贝叠在 forward 里面，offload_ms 只剩前向返回后的收尾。")
print("模式 | 状态 | cohort_ms | 单条均值_ms | forward_ms | offload_ms")
for mode in modes:
    log = out_dir / f"prefill_ablation_{mode}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    per_rank = {}
    for match in pat.finditer(text):
        if match.group(1) != mode:
            continue
        seq = int(match.group(3))
        if seq != input_len:
            continue
        rank = int(match.group(2))
        bucket = per_rank.setdefault(rank, [])
        bucket.append(
            (
                float(match.group(7)),
                float(match.group(8)),
                float(match.group(9)),
            )
        )
    ok = status.get(mode) == "0" and any(len(v) >= 2 for v in per_rank.values())
    state = "OK" if ok else "FAIL"
    if not per_rank:
        line = f"{names[mode]} | {state} | - | - | - | -"
        print(line)
        rows.append(line)
        continue
    slower = max(per_rank, key=lambda rank: sum(item[2] for item in per_rank[rank]))
    samples = per_rank[slower]
    total = sum(item[2] for item in samples)
    forward = sum(item[0] for item in samples)
    offload = sum(item[1] for item in samples)
    mean = total / len(samples)
    line = (
        f"{names[mode]} | {state} | {total:.1f} | {mean:.1f} | "
        f"{forward:.1f} | {offload:.1f}"
    )
    print(line)
    rows.append(line)
    detail = ", ".join(f"{item[2]:.1f}" for item in samples)
    print(f"  rank {slower} 两条 total_ms: {detail}")

summary = out_dir / "prefill_ablation_4b_summary.txt"
summary.write_text(
    "prefill ablation Qwen3-4B TP=2 batch=2 input=%s\n" % input_len
    + "\n".join(rows)
    + "\n"
)
print()
print(f"summary={summary}")
PY

echo "logs:"
echo "  $OUT_DIR/prefill_ablation_baseline.log"
echo "  $OUT_DIR/prefill_ablation_overlap.log"
echo "  $OUT_DIR/prefill_ablation_sparse.log"
echo "  $OUT_DIR/prefill_ablation_both.log"
