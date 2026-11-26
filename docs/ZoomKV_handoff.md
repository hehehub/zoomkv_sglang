# ZoomKV 交接说明

模型是 Qwen3-4B，TP=2，两张 GPU。代码在 GPU 机器的 `/root/zoomkv_sglang/sglang-plugin-ZoomKV`。ZoomKV 只替换 decode 的注意力；prefill 仍然走 FlashInfer。

下面这组 128k 对比是目前的正式结果。输入 131072，输出 768，配置的 batch 是 2，实际 decode 是 batch 1：第二条 128k 要等第一条 decode 结束才开始 prefill。YaRN factor 4.5，context 147456。

| 模式 | 报告 | 墙钟 | 每个 decode token |
|---|---|---:|---:|
| full attention | `nvtx_128k_full.nsys-rep` | 49.12 s | 12.79 ms |
| GPU-only | `nvtx_128k_sparse_gpuonly2.nsys-rep` | 56.09 s | 12.64 ms |
| offload（当前） | `nvtx_128k_sparse_offload3.nsys-rep` | 85.85 s | 17.90 ms |

墙钟是两张卡重叠之后、最早的 CUDA kernel 到最晚的 CUDA kernel，不要把两张卡的时间加起来。token 时间是 CUDA graph 一次重放的耗时（p50）。用 Nsight Systems **2025.3.1** 打开这些 `.nsys-rep`。

GPU-only 的 decode 已经略快于 full attention。端到端更慢，是因为每个请求要先建一次 Quest 索引。offload 在此之外还要把检索区的 KV 放到 CPU，decode 时再经 PCIe 读回来。

## 三种模式

- **full attention**：不加载 ZoomKV。prefill 和 decode 都是 FlashInfer，decode 整段前向在一张 CUDA graph 里。
- **GPU-only**：`SGLANG_ZOOMKV_OFFLOAD=false`。KV 留在 SGLang 的页表里。Quest 选出检索位置后，sparse kernel 直接从页表读。sink、local、新 token 也在这张页表里。
- **offload**：`SGLANG_ZOOMKV_OFFLOAD=true`。检索区 KV 在 CPU 锁页内存。decode 的 CUDA graph 里，GPU 用 UVA 按 top-k 下标把 K/V 从 CPU 读上来，再和页表里的 sink、local、新 token 一起做 sparse attention。

插件会强制 `--cuda-graph-backend-decode full`，并且关掉 radix cache、关掉 chunked prefill。不要再加 `--disable-cuda-graph`。

默认参数在 `sglang_zoomkv/config.py`：sink 64，local 512，`final_topk` 1000，`dynamic_update_interval` 512。这次运行另外设置了 `SGLANG_ZOOMKV_MAX_NEW_TOKENS=768`。

## 要改的文件在哪

根目录是 GPU 机器上的：

`/root/zoomkv_sglang/sglang-plugin-ZoomKV/sglang_zoomkv/`

改完必须停掉服务再启动，进程不会热加载。

还要加速的就这几个文件，按文件夹：

```text
sglang_zoomkv/
├── cache/
│   └── polar_cache_kivi_quest_density_seperate_kernel.py
├── kernels/
│   ├── gather_trans/
│   │   ├── trans_h2d.cu
│   │   └── trans_h2d.py
│   └── sparse_attn/
│       ├── sparse_decode_attn.cu
│       └── sparse_decode_attn_ops.py
└── attention/
    └── backend.py
```

| 文件 | 文件夹 | 要改什么 |
|---|---|---|
| `polar_cache_kivi_quest_density_seperate_kernel.py` | `sglang_zoomkv/cache/` | offload 的主路径。Prefill 把检索 KV 拷到 CPU（`prefill_d2h`），decode 图里做 Quest、UVA gather、sparse attention（`attention_inside_graph`）。优先改这个文件。 |
| `trans_h2d.cu` | `sglang_zoomkv/kernels/gather_trans/` | 从 CPU 锁页内存按 top-k 读 K/V 的 GPU kernel（`h2d_gather_kv_kernel`）。 |
| `trans_h2d.py` | `sglang_zoomkv/kernels/gather_trans/` | 上面这个 kernel 的 Python 入口。改 `.cu` 的参数时一起改。 |
| `sparse_decode_attn.cu` | `sglang_zoomkv/kernels/sparse_attn/` | decode 的 sparse attention。下一步应让它直接读 host 上的 top-k，不要先 gather 到 GPU buffer 再读一遍。 |
| `sparse_decode_attn_ops.py` | `sglang_zoomkv/kernels/sparse_attn/` | sparse kernel 的 Python 入口。改 `.cu` 的参数时一起改。 |
| `backend.py` | `sglang_zoomkv/attention/` | GPU-only 建 Quest 索引的入口（`_ensure_layer_prefill`）。要让端到端超过 full attention 时改这里，不是 offload 的 PCIe 路径。 |

旁边这几个文件现在已经接好，改上面的代码时不要把行为退回去：

| 文件 | 文件夹 | 作用 |
|---|---|---|
| `__init__.py` | `sglang_zoomkv/` | 插件入口。强制 decode 用完整 CUDA graph。 |
| `cuda_graph.py` | `sglang_zoomkv/attention/` | 等真实 batch 到了再抓那一张 decode 图。 |
| `config.py` | `sglang_zoomkv/` | sink、local、top-k、offload 开关。 |
| `quest_pipeline.py` | `sglang_zoomkv/kernels/quest/` | 图里面的 Quest。下标写进固定 buffer，重放时地址不能变。 |
| `staged_h2d.py` | `sglang_zoomkv/kernels/gather_trans/` | UVA 不可用时的旧拷贝。热路径已经不走它。 |

## 现在要改的核心代码


已经做完、不要退回去的部分：

- decode 是一张完整 CUDA graph，不再每个 layer 拆开。入口是 `__init__.py` 里把 `cuda_graph_backend_decode` 设成 `full`，图的捕获在 `attention/cuda_graph.py` 的 `install_decode_graph_break` / `_install_deferred_full_graph`。服务起来时先不抓图，等真实 batch 的 Quest 尺寸定下来再抓。日志里要看到 `capturing one decode CUDA graph for batch 1` 和 `decode CUDA graph capture finished`。
- offload 的热路径不再做「下标 D2H → CPU 同步 → CPU gather → H2D」。`kernels/gather_trans/trans_h2d.cu` 的 `h2d_gather_kv` 在 GPU 上直接读锁页内存。`cudaHostGetDevicePointer` 不能放进 CUDA graph，所以 `cache/polar_cache_kivi_quest_density_seperate_kernel.py` 的 `_try_enable_uva_gather` 会在初始化时把每一层的指针解析好。

offload 还慢，按收益从大到小改这些地方。

### 1. Prefill 把整段检索 KV 拷到 CPU

这是 offload 相对 GPU-only 最大的一段端到端差距。128k 上每个请求大约 8.6 秒，两次请求的 device-to-host 一共 38.4 GB。

位置：`cache/polar_cache_kivi_quest_density_seperate_kernel.py` 里 prefill 分区存储那一段，NVTX 名字是 `prefill_d2h` 和 `sidecar_prefill_copy`。检索区被拷进 `unified_keys_cpu` / `unified_values_cpu`。紧挨着的 `prefill_build_index` 用同一批 key 建 Quest 的 min/max。最后一层会 `_sync_offload_copies()`，等这些拷贝结束才能 decode。

第一条真实请求前面还有大约 8.4 秒的 `sidecar_reset`。服务预热会先占住 batch 1 的槽，128k 请求进来要按 `seq_len 131073` 重建索引。对应日志是 `rebuilding Quest index for seq_len 131073`。

### 2. 图内的 UVA gather

当前 offload 的 decode 图是 17.90 ms，GPU-only 是 12.64 ms，每个 token 大约多 5.3 ms。两个 768 token 的请求加起来大约多 8 秒。

位置：同一个 cache 文件的 `attention_inside_graph`。offload 分支先 `run_hierarchical_quest`，再 `_uva_gather_kv(...)` 把 top-k 写入 `_topk_keys_buffer` / `_topk_values_buffer`，然后 `_launch_sparse_attn` 从页表读 sink 和 local，从这块 buffer 读检索 token。kernel 在 `kernels/gather_trans/trans_h2d.cu` 的 `h2d_gather_kv_kernel`。

单独测过这个 kernel：形状和线上一致（batch 1、4 个 KV head、13 万 token、top-1000、bf16），一层 0.208 ms，36 层 7.5 ms，和 staged 拷贝结果一致，误差是 0。线上图里的增量比这个小一点，因为 GPU-only 本来也要从显存读那 1000 个 token。

36 层、每层大约 2 MB，按 PCIe 大约 25 GB/s 估算，下限在 3 ms 左右。只把现在这个 gather kernel 再写快一点，token 时间仍然会高于 GPU-only 的 12.6 ms。值得做的是让 `kernels/sparse_attn/sparse_decode_attn.cu` 的 sparse kernel 直接读 host 上的 top-k，不要先写进 GPU buffer 再读一遍。检索下标的坐标是 CPU 检索区里的下标（从 0 开始，不含 sink），和 GPU-only 那条「下标加 `sink_size` 再去页表里读」不是同一套坐标，不要混用。

### 3. 若目标是端到端超过 full attention

先看 GPU-only，不看 offload。decode 已经是 12.64 ms 对 12.79 ms。每个请求大约 3.5 到 4 秒花在建 Quest 索引：`attention/backend.py` 的 `_ensure_layer_prefill` 从页表把整段前缀 K/V 取出来，再进 `prefill_build_index`。把输出加长到 context 允许的上限（大约还能再生成 1.6 万 token）也补不回这段一次性开销。batch 提到 3 在 128k 上放不下：单条序列每张卡大约 9.1 GB KV，0.75 × 40 GB 大约是 30 GB 的静态上限。

## 环境

在 GPU 机器上：

```bash
source /root/env/zoomkv_sglang/bin/activate
cd /root/zoomkv_sglang
export CUDA_HOME=/root/env/zoomkv_sglang
export CUDA_VISIBLE_DEVICES=0,1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
```

nsys 是 `/usr/local/bin/nsys`，版本 2025.3.1.0。跟踪参数保持下面这样：

- `--cuda-graph-trace=graph`。不要用 `node`，ZoomKV 会崩。
- 用 `nsys launch --session-new=...` 启动服务，服务就绪之后再 `nsys start`，客户端跑完再 `nsys stop`。
- 不要用 `nsys profile --start-later`，除非同时带 `--session`。

三个终端：第一个起服务，第二个在服务就绪后开始采集，第三个打请求。`nsys start` 必须在客户端之前。确认第二个终端打印的是 `Collection`。

下面每条命令都用了新的 session 和报告名，不会覆盖已经有的 `nvtx_128k_full.nsys-rep`、`nvtx_128k_sparse_gpuonly2.nsys-rep`、`nvtx_128k_sparse_offload3.nsys-rep`。再跑下一轮时继续换名字。

客户端三条模式相同：

```bash
source /root/env/zoomkv_sglang/bin/activate
python -m sglang.benchmark.serving \
  --dataset-name random-ids --tokenize-prompt \
  --random-range-ratio 1.0 --seed 42 \
  --random-input-len 131072 --random-output-len 768 \
  --num-prompts 2 --max-concurrency 2 --warmup-requests 0
```

## full attention

第一个终端：

```bash
source /root/env/zoomkv_sglang/bin/activate
cd /root/zoomkv_sglang
export CUDA_HOME=/root/env/zoomkv_sglang
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
unset SGLANG_PLUGINS
unset SGLANG_ZOOMKV_OFFLOAD
unset SGLANG_ZOOMKV_MAX_NEW_TOKENS
unset SGLANG_ZOOMKV_FALLBACK_BACKEND
export CUDA_VISIBLE_DEVICES=0,1
nsys launch --session-new=full128k2 \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-graph-trace=graph \
  --trace-fork-before-exec=true \
  python -m sglang.launch_server \
  --model-path /root/zoomkv_sglang/models/Qwen3-4B \
  --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.5,"original_max_position_embeddings":32768}}' \
  --port 30000 --dtype bfloat16 \
  --context-length 147456 --max-total-tokens 263680 \
  --mem-fraction-static 0.75 \
  --attention-backend flashinfer \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 2 \
  --disable-radix-cache --chunked-prefill-size -1 \
  --max-prefill-tokens 131072 --max-running-requests 2 \
  --tp-size 2 --page-size 1 --disable-custom-all-reduce
```

服务就绪后，第二个终端：

```bash
nsys start --session=full128k2 --sample=none --cpuctxsw=none --force-overwrite=true -o /root/zoomkv_sglang/nsys_reports/nvtx_128k_full2
```

第三个终端跑上面的客户端。结束后：

```bash
nsys stop --session=full128k2
```

这条不加载 ZoomKV，报告里不应出现 ZoomKV 的 NVTX。每张 GPU 大约 1536 次 graph launch（2 个请求 × 768 个 token）。

## GPU-only

第一个终端：

```bash
source /root/env/zoomkv_sglang/bin/activate
cd /root/zoomkv_sglang
export CUDA_HOME=/root/env/zoomkv_sglang
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_PLUGINS=sglang_zoomkv
export SGLANG_ZOOMKV_OFFLOAD=false
export SGLANG_ZOOMKV_MAX_NEW_TOKENS=768
export SGLANG_ZOOMKV_FALLBACK_BACKEND=flashinfer
export CUDA_VISIBLE_DEVICES=0,1
nsys launch --session-new=zoom128k3 \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-graph-trace=graph \
  --trace-fork-before-exec=true \
  python -m sglang.launch_server \
  --model-path /root/zoomkv_sglang/models/Qwen3-4B \
  --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.5,"original_max_position_embeddings":32768}}' \
  --port 30000 --dtype bfloat16 \
  --context-length 147456 --max-total-tokens 263680 \
  --mem-fraction-static 0.75 \
  --prefill-attention-backend flashinfer \
  --decode-attention-backend zoomkv \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 2 \
  --disable-radix-cache --chunked-prefill-size -1 \
  --max-prefill-tokens 131072 --max-running-requests 2 \
  --tp-size 2 --page-size 1 --disable-custom-all-reduce
```

服务就绪后，第二个终端：

```bash
nsys start --session=zoom128k3 --sample=none --cpuctxsw=none --force-overwrite=true -o /root/zoomkv_sglang/nsys_reports/nvtx_128k_sparse_gpuonly3
```

第三个终端跑客户端。结束后：

```bash
nsys stop --session=zoom128k3
```

日志里以这几行为准：

- `rebuilding Quest index for seq_len 131073`
- `graph local window starts at 130560` 左右（131072 减去 local 窗口 512）
- `capturing one decode CUDA graph for batch 1`，然后 `decode CUDA graph capture finished`

预热那条短请求可能先打出一次很小的 window，忽略它。每张 GPU 大约 1534 次 graph launch：每个请求的第一个 token 用来建索引并抓图，后面 767 个 token 各重放一次。

## offload

第一个终端：

```bash
source /root/env/zoomkv_sglang/bin/activate
cd /root/zoomkv_sglang
export CUDA_HOME=/root/env/zoomkv_sglang
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_PLUGINS=sglang_zoomkv
export SGLANG_ZOOMKV_OFFLOAD=true
export SGLANG_ZOOMKV_MAX_NEW_TOKENS=768
export SGLANG_ZOOMKV_FALLBACK_BACKEND=flashinfer
export CUDA_VISIBLE_DEVICES=0,1
nsys launch --session-new=offload128k4 \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-graph-trace=graph \
  --trace-fork-before-exec=true \
  python -m sglang.launch_server \
  --model-path /root/zoomkv_sglang/models/Qwen3-4B \
  --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.5,"original_max_position_embeddings":32768}}' \
  --port 30000 --dtype bfloat16 \
  --context-length 147456 --max-total-tokens 263680 \
  --mem-fraction-static 0.75 \
  --prefill-attention-backend flashinfer \
  --decode-attention-backend zoomkv \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 2 \
  --disable-radix-cache --chunked-prefill-size -1 \
  --max-prefill-tokens 131072 --max-running-requests 2 \
  --tp-size 2 --page-size 1 --disable-custom-all-reduce
```

服务就绪后，第二个终端：

```bash
nsys start --session=offload128k4 --sample=none --cpuctxsw=none --force-overwrite=true -o /root/zoomkv_sglang/nsys_reports/nvtx_128k_sparse_offload4
```

第三个终端跑客户端。结束后：

```bash
nsys stop --session=offload128k4
```

日志里应出现：

- `offload gather reads pinned CPU KV from the GPU`
- `graph local window starts at 130560` 左右
- `capturing one decode CUDA graph for batch 1`
- `decode CUDA graph capture finished`

如果出现 `decode CUDA graph capture failed`，这次没有收成整图，把 traceback 留下来。成功时每张 GPU 也是大约 1534 次 graph launch。`uva_h2d_gather` 只会在每个请求的第一个 eager token 上出现；后面的 gather 在 graph 里面，`--cuda-graph-trace=graph` 不会单独展开。

报告在 `/root/zoomkv_sglang/nsys_reports/`。
