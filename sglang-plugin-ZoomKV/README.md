# SGLang ZoomKV Plugin

An out-of-tree SGLang plugin that injects the ZoomKV chunk-centric sparse KV
retrieval path without modifying the SGLang source tree.

This first release is an MVP intended to validate the integration:

- Qwen/Llama-style MHA/GQA models with `head_dim=128`
- NVIDIA CUDA, tensor parallel size 1 or 2
- lockstep decode batches of up to 64 equal-length requests
- standard SGLang prefill and ZoomKV decode
- hierarchical Quest filtering, dense/sparse KIVI 4-bit reranking, and either
  CPU KV offload or a complete GPU-resident KV sidecar

It does **not** currently support ragged/variable-length continuous batches,
CUDA Graph, speculative decoding, sliding-window attention, MLA, cross
attention, TP greater than 2, or PP. The batch implementation targets
fixed-length throughput tests; unequal sequence lengths fail explicitly.

## Install

Install a CUDA-enabled SGLang environment first, then install this package:

```bash
cd sglang-plugin-ZoomKV
pip install -e .
```

The CUDA/C++ operators are compiled lazily on their first use. A working CUDA
toolchain and `ninja` are required. The source checkout under `ZoomKV-main` is
not needed after this plugin has been installed.

## Run

Use a normal SGLang prefill backend and select ZoomKV for decode:

```bash
SGLANG_PLUGINS=sglang_zoomkv \
SGLANG_ZOOMKV_FALLBACK_BACKEND=flashinfer \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --prefill-attention-backend flashinfer \
  --decode-attention-backend zoomkv \
  --disable-cuda-graph \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --max-running-requests 64 \
  --tp-size 2
```

`torch_native` is the default fallback and is useful for correctness checks:

```bash
SGLANG_ZOOMKV_FALLBACK_BACKEND=torch_native \
python -m sglang.launch_server ... \
  --prefill-attention-backend torch_native \
  --decode-attention-backend zoomkv
```

The backend rejects unsupported configurations instead of silently returning
incorrect output.

## Configuration

The plugin accepts `SGLANG_ZOOMKV_*` variables. For compatibility with the
standalone ZoomKV release, corresponding `POLAR_*` names are also accepted.

Common settings:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `SGLANG_ZOOMKV_HEAD_DIM` | `128` | Fixed kernel head dimension; other values are rejected |
| `SGLANG_ZOOMKV_BITS` | `4` | Fixed KIVI code width; other values are rejected |
| `SGLANG_ZOOMKV_OFFLOAD` | `true` | `true`: pinned-CPU retrieval KV; `false`: complete GPU-only sidecar |
| `SGLANG_ZOOMKV_SINK_SIZE` | `64` | Always-resident sink tokens |
| `SGLANG_ZOOMKV_LOCAL_SIZE` | `512` | Always-resident recent window |
| `SGLANG_ZOOMKV_FINAL_TOPK` | `1000` | Final retrieved-token budget |
| `SGLANG_ZOOMKV_CANDIDATE_RATIO` | `0.2` | Stage-1 candidate ratio |
| `SGLANG_ZOOMKV_DYNAMIC_UPDATE_INTERVAL` | `512` | Index update interval |
| `SGLANG_ZOOMKV_FULL_ATTENTION_THRESHOLD` | `2000` | Full-attention cutoff |
| `SGLANG_ZOOMKV_MAX_NEW_TOKENS` | `4096` | Sidecar decode capacity |
| `SGLANG_ZOOMKV_FALLBACK_BACKEND` | `torch_native` | Prefill/full fallback |

The lower-level Quest/CDS tuning variables retain their standalone names,
including `POLAR_QUEST_CHUNK`, `POLAR_QUEST_LARGE_CHUNK`,
`POLAR_QUEST_LARGE_RATIO`, `POLAR_QUEST_SMALL_RATIO`,
`POLAR_CDS_DENSE_RATIO`, `POLAR_CDS_DENSE_TOPK`, and
`POLAR_CDS_SPARSE_TOPK`.

For settings accepted under both prefixes, `SGLANG_ZOOMKV_*` takes precedence
over its legacy `POLAR_*` spelling. Invalid numeric, boolean, or unsupported
fixed-kernel values fail during plugin loading, before backend registration.

## How it integrates

Installing the package adds a `sglang.srt.plugins` entry point. During SGLang
startup the plugin:

1. adds `zoomkv` to the attention backend choices;
2. registers a `ZoomKVAttentionBackend`;
3. leaves SGLang's platform, scheduler, model code, and KV allocator unchanged.

Prefill K/V remains in SGLang's regular token pool. On the first decode step,
the backend builds a per-layer ZoomKV sidecar from that pool. Each subsequent
decode step updates the SGLang pool and sidecar, then runs the ZoomKV retrieval
pipeline.

With `SGLANG_ZOOMKV_OFFLOAD=false`, the sidecar retains every token in
chronological order on GPU. Short contexts use full attention; long contexts
use the same Quest/KIVI retrieval metadata, gather selected retrieval tokens
from GPU, and call the same FlashAttention function. Sink, retrieval, and local
zones are disjoint.

With offload enabled, selected indices are copied GPU→pinned CPU, KV rows are
gathered on CPU (the C++ extension is preferred, with `torch.gather` fallback),
then copied explicitly from pinned staging buffers to preallocated GPU outputs.
The production path does not run the legacy `trans_h2d.cu` UVA kernel.

## Tests

CPU-only registration/configuration tests:

```bash
python -m pip install -e ".[test]"
pytest -q tests/unit
```

GPU kernel and integration tests are marked `gpu`:

```bash
pytest -q -m gpu
```

GPU kernel modules are imported inside the marked test functions. On hosts
without CUDA they are skipped before import, so test collection does not
compile native operators. The first native smoke run on a CUDA host may invoke
PyTorch JIT compilation and therefore requires the CUDA toolkit and `ninja`.

## Source provenance

The cache algorithm and CUDA/C++ kernels are adapted from the adjacent
`ZoomKV-main` implementation. The packaging and plugin loading structure
follows SGLang's official out-of-tree plugin API and the
`sglang-plugin-FL-main` reference project.

