#!/usr/bin/env python3
"""Benchmark an SGLang ZoomKV TP=2 server at batch size 32/64.

This is adapted from flagos-ai/vllm-plugin-FL's serving benchmark.  The
important differences are:

* it invokes ``sglang.benchmark.serving`` instead of ``vllm bench serve``;
* requests use exact token IDs so every decode batch remains lockstep;
* test cases cover batch size 32/64 and request count 64/128;
* lengths fit Qwen3-8B bf16 plus two GPU KV copies on two 24 GiB RTX 4090s.

Expected server capacity:

* ``--tp-size 2``
* ``--context-length 1024``
* ``--max-total-tokens 65536``
* ``--max-running-requests 64``
* ``--max-prefill-tokens 32768``
* ``SGLANG_ZOOMKV_MAX_NEW_TOKENS=256``
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

RUNS = 4
SKIP_FIRST = 1
RECOMMENDED_CONTEXT_LENGTH = 1_024
RECOMMENDED_MAX_TOTAL_TOKENS = 65_536
RECOMMENDED_ZOOMKV_MAX_NEW_TOKENS = 256

# (input length, output length, concurrency, number of prompts)
DEFAULT_TEST_CASES = [
    (512, 256, 32, 64),
    (512, 256, 32, 128),
    (512, 256, 64, 64),
    (512, 256, 64, 128),
]

PATTERNS = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "benchmark_duration": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "total_input_tokens": r"Total input tokens:\s+([0-9.]+)",
    "total_output_tokens": r"Total generated tokens:\s+([0-9.]+)",
    "request_throughput": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "output_throughput": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "peak_output_throughput": r"Peak output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_token_throughput": r"Total token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
    "median_itl_ms": r"Median ITL \(ms\):\s+([0-9.]+)",
    "p99_itl_ms": r"P99 ITL \(ms\):\s+([0-9.]+)",
}

COLUMNS = [
    "Prefill",
    "Decode",
    "Conc",
    "Num Prompts",
    "Successful Requests",
    "Run Status",
    "Benchmark Duration (s)",
    "Total Input Tokens",
    "Total Output Tokens",
    "Req/s",
    "Output tok/s",
    "Peak Output tok/s",
    "Total tok/s",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
]

METRIC_COLUMNS = {
    "Benchmark Duration (s)": "benchmark_duration",
    "Total Input Tokens": "total_input_tokens",
    "Total Output Tokens": "total_output_tokens",
    "Req/s": "request_throughput",
    "Output tok/s": "output_throughput",
    "Peak Output tok/s": "peak_output_throughput",
    "Total tok/s": "total_token_throughput",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "Median TTFT (ms)": "median_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms",
    "Median TPOT (ms)": "median_tpot_ms",
    "P99 TPOT (ms)": "p99_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "Median ITL (ms)": "median_itl_ms",
    "P99 ITL (ms)": "p99_itl_ms",
}


def existing_path(value: str) -> str:
    path = os.path.abspath(os.path.expanduser(value))
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"model path does not exist: {path}")
    return path


def parse_test_cases(value: str) -> list[tuple[int, int, int, int]]:
    try:
        raw_cases = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc

    if not isinstance(raw_cases, list) or not raw_cases:
        raise argparse.ArgumentTypeError("test cases must be a non-empty list")

    cases = []
    for index, case in enumerate(raw_cases):
        if (
            not isinstance(case, list)
            or len(case) != 4
            or any(type(item) is not int or item <= 0 for item in case)
        ):
            raise argparse.ArgumentTypeError(
                f"case {index} must contain four positive integers"
            )
        if case[2] not in (32, 64):
            raise argparse.ArgumentTypeError(
                f"case {index}: concurrency must be 32 or 64"
            )
        if case[3] not in (64, 128):
            raise argparse.ArgumentTypeError(
                f"case {index}: number of prompts must be 64 or 128"
            )
        cases.append(tuple(case))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang ZoomKV with TP=2 and batch size 32/64."
    )
    parser.add_argument("--model", type=existing_path, required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--test-cases",
        type=parse_test_cases,
        help='JSON list such as "[[512,256,32,64],[512,256,64,128]]"',
    )
    parser.add_argument(
        "--context-length", type=int, default=RECOMMENDED_CONTEXT_LENGTH
    )
    parser.add_argument(
        "--max-total-tokens", type=int, default=RECOMMENDED_MAX_TOTAL_TOKENS
    )
    parser.add_argument(
        "--zoomkv-max-new-tokens",
        type=int,
        default=RECOMMENDED_ZOOMKV_MAX_NEW_TOKENS,
    )
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--skip-first", type=int, default=SKIP_FIRST)
    parser.add_argument(
        "--output-dir", default="benchmark_results/zoomkv_gpu_only"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, cases: list[tuple[int, int, int, int]]) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive")
    if args.max_total_tokens <= 0:
        raise ValueError("--max-total-tokens must be positive")
    if args.zoomkv_max_new_tokens <= 0:
        raise ValueError("--zoomkv-max-new-tokens must be positive")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if not 0 <= args.skip_first < args.runs:
        raise ValueError("--skip-first must satisfy 0 <= skip-first < runs")

    for input_len, output_len, concurrency, _ in cases:
        if concurrency not in (32, 64):
            raise ValueError("ZoomKV throughput cases require concurrency 32 or 64")
        if input_len + output_len > args.context_length:
            raise ValueError(
                f"case {input_len}+{output_len} exceeds server context length "
                f"{args.context_length}"
            )
        if output_len > args.zoomkv_max_new_tokens:
            raise ValueError(
                f"case output length {output_len} exceeds the required "
                "SGLANG_ZOOMKV_MAX_NEW_TOKENS setting "
                f"{args.zoomkv_max_new_tokens}"
            )
        active_tokens = concurrency * (input_len + output_len)
        if active_tokens > args.max_total_tokens:
            raise ValueError(
                f"case needs up to {active_tokens} active KV tokens, exceeding "
                f"the configured pool size {args.max_total_tokens}"
            )


def common_command(args: argparse.Namespace) -> list[str]:
    model_name = args.served_model_name or args.model
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--model",
        model_name,
        "--tokenizer",
        args.model,
        "--dataset-name",
        "random-ids",
        "--tokenize-prompt",
        "--random-range-ratio",
        # SGLang defines this as the minimum/maximum length ratio.  1.0 is
        # required for exact equal-length lockstep batches; 0 means ragged
        # random lengths in [1, requested_length].
        "1.0",
        "--warmup-requests",
        "1",
        "--seed",
        "42",
    ]
    if args.served_model_name:
        command.extend(["--served-model-name", args.served_model_name])
    return command


def benchmark_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(name, None)
    env["NO_PROXY"] = "127.0.0.1,localhost,::1"
    env["no_proxy"] = env["NO_PROXY"]
    return env


def extract_metrics(output: str) -> dict[str, float | None]:
    metrics = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, output, re.IGNORECASE)
        metrics[key] = float(match.group(1)) if match else None
    return metrics


def result_row(
    case: tuple[int, int, int, int],
    metrics: dict[str, float | None],
    *,
    status: str,
) -> dict[str, object]:
    input_len, output_len, concurrency, num_prompts = case
    row: dict[str, object] = {
        "Prefill": input_len,
        "Decode": output_len,
        "Conc": concurrency,
        "Num Prompts": num_prompts,
        "Successful Requests": metrics.get("successful_requests"),
        "Run Status": status,
    }
    row.update(
        {column: metrics.get(metric_key) for column, metric_key in METRIC_COLUMNS.items()}
    )
    return row


def append_csv(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_once(
    case: tuple[int, int, int, int],
    run_id: int,
    base_command: list[str],
    output_dir: Path,
) -> dict[str, float | None]:
    input_len, output_len, _, num_prompts = case
    concurrency = case[2]
    label = f"{input_len}_{output_len}_c{concurrency}"
    jsonl_path = output_dir / f"{label}_run{run_id}.jsonl"
    command = base_command + [
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--output-file",
        str(jsonl_path),
    ]

    print(f"\n{'=' * 80}\nRunning {label}, run {run_id}\n{'=' * 80}")
    print(shlex.join(command), flush=True)
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=benchmark_env(),
    )
    print(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(
            f"benchmark exited with code {process.returncode}: {label}, run {run_id}"
        )
    return extract_metrics(process.stdout)


def average_metrics(
    runs: list[dict[str, float | None]],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in PATTERNS:
        values = [run[key] for run in runs if run.get(key) is not None]
        result[key] = round(mean(values), 2) if values else None
    return result


def main() -> None:
    args = parse_args()
    cases = args.test_cases or DEFAULT_TEST_CASES
    validate_args(args, cases)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / "raw_runs.csv"
    summary_csv = output_dir / "summary.csv"
    base_command = common_command(args)

    print("SGLang ZoomKV GPU-only benchmark")
    print(f"Model: {args.model}")
    print(f"Server: http://127.0.0.1:{args.port}")
    print(f"Expected context length: {args.context_length}")
    print(f"Expected KV pool tokens: {args.max_total_tokens}")
    print(f"Expected ZoomKV max new tokens: {args.zoomkv_max_new_tokens}")
    print(f"Cases: {cases}")
    print(f"Runs: {args.runs}; skip first: {args.skip_first}")
    print(f"Results: {output_dir}")

    for case in cases:
        measured_runs = []
        for run_id in range(1, args.runs + 1):
            metrics = run_once(case, run_id, base_command, output_dir)
            expected = case[3]
            success = metrics.get("successful_requests") == expected
            append_csv(
                raw_csv,
                result_row(case, metrics, status="SUCCESS" if success else "FAILED"),
            )
            measured_runs.append(metrics)

        valid_runs = measured_runs[args.skip_first :]
        if any(run.get("successful_requests") != case[3] for run in valid_runs):
            print(f"Skipping summary for failed case: {case}")
            continue

        averaged = average_metrics(valid_runs)
        append_csv(summary_csv, result_row(case, averaged, status="SUCCESS"))

    print(f"\nRaw CSV: {raw_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
