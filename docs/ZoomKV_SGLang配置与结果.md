# SGLang ZoomKV 配置与结果

## 1. 测试目标

对比同一形状下的 full attention 和 TopK 100 的 ZoomKV。ZoomKV 这八次是 `SGLANG_ZOOMKV_FINAL_TOPK=100` 的客户端 benchmark。full attention 的 prefill 和 decode 都是 FlashInfer，结果在 `/root/zoomkv_sglang/sglang_1029_2_131072_768.jsonl` 第 3 行。

本次测试：

- 模型：Qwen3-4B，BF16
- 输入长度：131072 tokens
- 输出长度：768 tokens
- 配置 batch：2，实际 decode batch：1
- 请求数量：非卸载和卸载都是 2
- 并行 query 数：2
- CUDA Graph 开启，decode 是一张完整图
- 时间是客户端墙钟：整段时间、首 token、每 token。没有套 nsys

八次只换三件事：query×chunk summary 和 topk 是拆开还是合并、源码里的 NVTX 标记在不在、检索 KV 留在 GPU 还是卸载到 CPU。Prefill 都是 FlashInfer dense。没有 Chunked Prefill。

另有一次卸载吞吐测试：16 条请求、并发 16，TopK 100，算子拆开，NVTX 已注释，没有 nsys，decode 仍是一张完整 CUDA Graph。结果在 `/root/zoomkv_e2e/offload_bs16_e2e.jsonl`。

## 2. 测试环境

### 2.1 模型与硬件

| 项目 | 配置 |
|---|---|
| 模型 | Qwen3-4B，BF16，`/root/zoomkv_sglang/models/Qwen3-4B` |
| 位置编码 | YaRN，factor 4.5，`original_max_position_embeddings=32768` |
| GPU / 并行 | 2× A100-SXM4-40GB，TP=2 |
| 上下文 | 147456 |
| 静态显存 | `mem-fraction-static=0.75`，约 30GB/卡 |

### 2.2 调度与显存参数

| 项目 | 配置 |
|---|---|
| page size | 1 |
| radix cache | 关 |
| chunked prefill | 关，`chunked-prefill-size=-1` |
| max-total-tokens | 263680 |
| max-prefill-tokens | 131072 |
| max-running-requests | 2 |
| CUDA Graph | decode 使用 full graph，`--cuda-graph-max-bs-decode=2` |
| 通信 | `--disable-custom-all-reduce` |

单条 128k 序列的 KV 大约 9.1GB/卡。这次的页池按 `max-total-tokens=263680` 配了两条，大约 18GB。`--max-running-requests` 和 `--cuda-graph-max-bs-decode` 都是 2，所以这八次的上限是 2。两条 128k 不会同时处在 decode，实际 decode batch 是 1。

非卸载把整段 KV 留在 GPU 页表里，第三条 128k 再加上每张卡大约 4GB 权重，超过 30GB。卸载在检索区拷到 CPU 之后会把这段 GPU 页还回去，一条请求不再长期占那 9GB。这八次没有把卸载的上限调到 2 以上。两条 128k 如果同时做 prefill，释放页之前池子仍会被占满。

### 2.3 功能开关

| 模式 | Prefill | Decode |
|---|---|---|
| GPU-only | FlashInfer dense | ZoomKV 稀疏，检索 KV 留在 GPU 页表。`SGLANG_ZOOMKV_OFFLOAD=false` |
| Offload | FlashInfer dense | ZoomKV 稀疏，检索 KV 在 CPU 锁页内存，UVA 读回。`SGLANG_ZOOMKV_OFFLOAD=true` |

两种模式除了卸载开关、算子是否合并、NVTX 是否留在源码里，其余服务参数和 benchmark 参数一致。

启动时已确认：

- `SGLANG_ZOOMKV_FINAL_TOPK=100`。日志是 `[ZoomKV] final_topk=100`。
- 拆开时日志是 `[Quest] score and topk stay separate kernels`。合并时日志是 `[Quest] score and topk are one fused kernel`。
- GPU-only 走页表上的 sparse decode，不另存一份检索 KV。
- Offload 的 gather 在 GPU 上直接读 CPU 锁页内存。36 层拷完后，日志里有 `released retrieval GPU pages`，检索区的 GPU 页还回分配器。sink 和 local 留下。
- Decode CUDA Graph 已完成捕获。日志为 `capturing one decode CUDA graph for batch 1` 和 `decode CUDA graph capture finished`。

## 3. ZoomKV 配置

这八次用环境变量把 final TopK 设成 100。代码里的默认值仍是 1000，不改默认值。另外设置了 `SGLANG_ZOOMKV_MAX_NEW_TOKENS=768`。

| 项目 | 值 |
|---|---|
| final TopK | 100，`SGLANG_ZOOMKV_FINAL_TOPK=100` |
| sink | 64，始终全量注意力 |
| local | 512，始终全量注意力 |
| 小 chunk | 8 |
| 大 chunk | 256 |
| 粗筛 | 大 chunk Quest 打分后取 parent TopK |
| 选 TopK | 小 chunk 打分、child TopK、map back |
| 精排 | KIVI 4-bit。chunk 内 dense TopK=16、sparse TopK=8，再做全局 TopK |
| 短序列 | 长度小于 2000 走全量注意力 |
| 索引更新间隔 | 512 |
| 这次 768 个 decode token | 没有中途重建索引。新 token 从 GPU 页表直接读 |

query×chunk summary 的内积和 topk 默认是两个 kernel。`SGLANG_ZOOMKV_FUSED_QUEST=1` 才合成一个 kernel。八次里四种是拆开，四种是合并。

## 4. Benchmark 配置

| 项目 | 配置 |
|---|---|
| 数据 | `random-ids`，`--tokenize-prompt`，`seed=42`，`random-range-ratio=1.0` |
| 输入 | 131072 |
| 输出 | 768 |
| 请求数 | 2 |
| 并发 | 2 |
| warmup | 0。服务自己的短 warmup 会占过 batch-1 槽，正式 128k 请求会重建索引 |
| 结果文件 | `/root/zoomkv_e2e/`，每个配置一个 jsonl，一行 |
| 时间 | 客户端墙钟。整段时间是 `duration`。首 token 是 `mean_ttft_ms`。每 token 是 `mean_tpot_ms`。生成速度是 `1000 / mean_tpot_ms` |

这八次没有用 nsys。nsys 记的是 GPU 时间线，采集本身会拖慢运行。端到端要的是请求从发出到收完的时间，所以用 benchmark 客户端直接计时。

| 配置 | 文件 |
|---|---|
| full attention | `/root/zoomkv_sglang/sglang_1029_2_131072_768.jsonl` 第 3 行 |
| 拆开，无标记，GPU-only | `split_novtx_gpuonly.jsonl` |
| 拆开，有标记，GPU-only | `split_nvtx_gpuonly.jsonl` |
| 合并，无标记，GPU-only | `fused_novtx_gpuonly.jsonl` |
| 合并，有标记，GPU-only | `fused_nvtx_gpuonly.jsonl` |
| 拆开，无标记，卸载 | `split_novtx_offload.jsonl` |
| 拆开，有标记，卸载 | `split_nvtx_offload.jsonl` |
| 合并，无标记，卸载 | `fused_novtx_offload.jsonl` |
| 合并，有标记，卸载 | `fused_nvtx_offload.jsonl` |
| 卸载，16 条，无标记，无 nsys | `/root/zoomkv_e2e/offload_bs16_e2e.jsonl` |

「拆开」是 `SGLANG_ZOOMKV_FUSED_QUEST=0`。「合并」是 `SGLANG_ZOOMKV_FUSED_QUEST=1`。「有标记」是源码里的 NVTX 还在；「无标记」是这些 `nvtx.range_push/pop` 被注释掉。解码仍是一张完整 CUDA Graph，标记不会在每个 token 重放。

## 5. 实验结果

full attention 和 GPU-only 都是客户端墙钟，输入 131072，输出 768，`completed=2`，`max_concurrency=2`，decode 是一张完整 CUDA Graph。full attention 没有 TopK。GPU-only 是下面四行的范围，TopK 100。

| 配置 | 整段时间 | 首 token | 每 token | 生成速度 | 输出吞吐 |
|---|---:|---:|---:|---:|---:|
| full attention | 49.19 s | 26.88 s | 13.01 ms | 76.9 tok/s | 31.22 tok/s |
| GPU-only | 52.11–52.29 s | 29.20–29.30 s | 12.80–12.90 ms | 77.5–78.1 tok/s | 29.38–29.48 tok/s |

GPU-only 每个 token 比 full attention 快 0.1–0.2 ms。整段慢大约 3 秒，差在首 token：29.2 秒对 26.9 秒，这是建 Quest 索引的一次性开销。

八个 ZoomKV 文件的整段时间含 prefill、建索引、拷贝和 decode。输出吞吐是 1536 个输出 token 除以整段时间，prefill 也算在分母里。

| 配置 | 整段时间 | 首 token | 每 token | 生成速度 | 输出吞吐 |
|---|---:|---:|---:|---:|---:|
| 拆开，无标记，GPU-only | 52.11 s | 29.20 s | 12.80 ms | 78.1 tok/s | 29.48 tok/s |
| 拆开，有标记，GPU-only | 52.29 s | 29.30 s | 12.90 ms | 77.5 tok/s | 29.38 tok/s |
| 合并，无标记，GPU-only | 52.20 s | 29.23 s | 12.89 ms | 77.6 tok/s | 29.43 tok/s |
| 合并，有标记，GPU-only | 52.24 s | 29.28 s | 12.88 ms | 77.7 tok/s | 29.40 tok/s |
| 拆开，无标记，卸载 | 61.94 s | 37.63 s | 13.59 ms | 73.6 tok/s | 24.80 tok/s |
| 拆开，有标记，卸载 | 62.36 s | 38.01 s | 13.63 ms | 73.4 tok/s | 24.63 tok/s |
| 合并，无标记，卸载 | 62.32 s | 38.00 s | 13.62 ms | 73.4 tok/s | 24.65 tok/s |
| 合并，有标记，卸载 | 61.40 s | 37.21 s | 13.47 ms | 74.2 tok/s | 25.02 tok/s |

GPU-only 整段在 52.11–52.29 s，每 token 12.80–12.90 ms。卸载整段在 61.40–62.36 s，每 token 13.47–13.63 ms。拆开和合并的差距小于同一种配置自己重复跑的波动。有标记和无标记也几乎一样。

卸载、16 条请求、并发 16 的这一次单独列出。`--max-running-requests` 和 `--cuda-graph-max-bs-decode` 都是 16。算子拆开，NVTX 已注释，没有 nsys，decode 是一张完整 CUDA Graph。`completed=16`。整段时间含 16 条排队 prefill、建索引、拷贝和 decode。输出吞吐是 `output_throughput`，等于 12288 个输出 token 除以整段时间。生成速度仍是 `1000 / mean_tpot_ms`。客户端统计的平均并发是 8.56。

| 配置 | 整段时间 | 首 token | 每 token | 生成速度 | 输出吞吐 |
|---|---:|---:|---:|---:|---:|
| 卸载，16 条，无标记，无 nsys | 1159.66 s | 567.25 s | 69.18 ms | 14.5 tok/s | 10.60 tok/s |

这一次每条平均耗时 620.31 秒。每 token 从 2 条卸载时的 13.47–13.63 ms 变成 69.18 ms。输出吞吐是 10.60 tok/s，低于 2 条卸载的 24.63–25.02 tok/s，因为 16 条 128k 的 prefill 排在同一段墙钟里。

## 6. 结论

在 131072 input / 768 output / 实际 decode batch 1 下，full attention、GPU-only、卸载各 2 个请求。GPU-only 和卸载是 TopK 100。

- full attention 每 token 13.01 ms，生成速度 76.9 tok/s。整段 49.19 s，首 token 26.88 s。
- GPU-only 每 token 12.80–12.90 ms，生成速度 77.5–78.1 tok/s。整段大约 52.2 s，首 token 大约 29.2 s。每个 token 比 full attention 快 0.1–0.2 ms，整段慢大约 3 秒，多出来的时间是建 Quest 索引。
- 卸载每 token 13.47–13.63 ms，生成速度 73.4–74.2 tok/s。整段大约 61.4–62.4 s，首 token 大约 37.2–38.0 s。卸载比 GPU-only 慢的主要部分在首 token，也就是 prefill 时把整段检索 KV 拷到 CPU。
- query×chunk summary 和 topk 拆开或合并，端到端差一个来回波动。按老师的要求，默认保持拆开。
- 源码里留着 NVTX、同时 decode 仍是一张完整 CUDA Graph 时，端到端几乎不变。标记只在抓图时走一次，不会在每个 token 重放。
- 这组没有用 nsys。打开 nsys 之后，采集开销会算进这次运行，尤其把图拆开看分段时，一个 token 的 GPU 工作会被标记首尾拉长。
- 卸载 16 条、并发 16：整段 1159.66 s，首 token 567.25 s，每 token 69.18 ms，生成速度 14.5 tok/s，输出吞吐 10.60 tok/s。输出吞吐就是这次的吞吐量，字段是 `output_throughput`。

## 7. 系统设计

SGLang 继续负责调度、分页 KV、采样和张量并行。ZoomKV 只替换 decode 的注意力。Prefill 两种模式都是 FlashInfer dense，稀疏检索只发生在逐 token 解码。

一条序列在逻辑上分成三段：

| 区域 | 范围 | 注意力 |
|---|---|---|
| sink | 前 64 个 token | 每个 decode token 都看 |
| retrieval | sink 之后、local 之前，按 256 token 对齐后的那一段 | 只看 Quest 选出的 final TopK。这八次是 100 |
| local | 最近 512 个 token，加上对齐余数 | 每个 decode token 都看 |

检索不扫描 13 万 token。先按大 chunk（256）做 query 和 chunk summary 的内积，再在选中的大块里对小 chunk（8）打分，最后用 KIVI 4-bit 精排，得到 final TopK。query×chunk summary 和 topk 是两个 kernel。`SGLANG_ZOOMKV_FUSED_QUEST=1` 才会合成一个 kernel。当前默认是拆开的。

KV 放在哪里：

| 数据 | GPU-only | Offload |
|---|---|---|
| 整段 bf16 KV | GPU 页表，只此一份 | prefill 时先写入 GPU 页表 |
| 检索区 bf16 KV | 留在 GPU 页表 | 拷进 CPU 锁页内存 `unified_keys_cpu` / `unified_values_cpu`。36 层都拷完后，GPU 页表里这一段还回去 |
| sink、local、新 token | GPU 页表 | 留在 GPU 页表，另外在 ZoomKV 的小 GPU 缓冲里留 sink 和 local |
| Quest 索引 | GPU 上的 chunk min/max 和 KIVI 4-bit | 同一份索引留在 GPU。打分不读整段 KV |
| 本步选中的 token | sparse kernel 直接读页表 | UVA 从 CPU 读进一块很小的 GPU 缓冲，再做 sparse attention |

GPU-only 的检索下标要加 `sink_size` 再去页表里读。Offload 的检索下标从检索区的 0 开始。两套下标不能混用。

页表按 `max-total-tokens=263680` 一次分配，大约两条 128k 序列。Offload 把检索页还回去之后，这条请求在 GPU 页表里只剩 sink 和 local。空出来的页可以给下一条 prefill 用。两条 128k 仍不能同时做 prefill，因为 prefill 当时还要把整段写进页表。这八次的 `--max-running-requests` 和 `--cuda-graph-max-bs-decode` 都是 2，所以并发仍是 2。

```text
Scheduler
  │  分配 req_to_token / 页表
  ▼
FlashInfer prefill（dense）
  │  整段 bf16 KV 写入 GPU 页表
  │
  ├─ GPU-only
  │    └─ 用检索区的 key 建立 Quest 索引，不另存一份 KV
  │
  └─ Offload
       ├─ 每一层算完就在旁路 stream 上把检索区拷到 CPU
       ├─ 用同一批 key 建立 Quest 索引
       └─ 36 层拷贝结束后，把检索区的 GPU 页还回分配器
  ▼
decode，每个 token 一张完整 CUDA Graph
  │
  ├─ 大 chunk 打分，再 topk
  ├─ 小 chunk 打分，再 topk，map back
  ├─ KIVI 精排，得到 final TopK 100
  │
  ├─ GPU-only：sparse kernel 读页表
  └─ Offload：UVA 读 CPU 上的 100 个 token，sparse kernel 读 sink、local、新 token
```

## 8. CUDA Graph

第 5 节的速度来自 decode full graph，每个 token 一张图。客户端的每 token 时间还含两张 graph 之间的一小段 CPU 间隔。

- 服务启动时不抓图。等真实 batch 的 Quest 尺寸定下来再抓，避免 warmup 的短序列把 batch-1 槽占住之后，128k 请求复用错误索引。
- GPU-only 的图内包含 Quest 检索和 sparse attention。
- `cudaHostGetDevicePointer` 不能放进图里，所以 UVA 指针在初始化时按层解析好。
- 捕获失败不会拿一张错误的图继续跑。日志里没有 `decode CUDA graph capture finished` 时，第 5 节的数字不能当成完整图的结果。
- 这八次没有把 attention 拆到图外。源码里的 NVTX 只在抓图时执行一次。

## 9. 原始数据

第 5 节由下面这些客户端字段四舍五入得到。ZoomKV 每个文件一行。full attention 是 `sglang_1029_2_131072_768.jsonl` 的第 3 行。前九行 `completed=2`，最后一行 `completed=16`。

| 文件 | tag | duration (s) | mean_ttft_ms | mean_tpot_ms | output_throughput |
|---|---|---:|---:|---:|---:|
| `sglang_1029_2_131072_768.jsonl` 第 3 行 | full attention | 49.192509 | 26884.969 | 13.007707 | 31.224267 |
| `split_novtx_gpuonly.jsonl` | `split_novtx_gpuonly_topk100` | 52.105465 | 29203.610 | 12.803608 | 29.478674 |
| `split_nvtx_gpuonly.jsonl` | `split_nvtx_gpuonly_topk100` | 52.287865 | 29298.991 | 12.902713 | 29.375841 |
| `fused_novtx_gpuonly.jsonl` | `fused_novtx_gpuonly_topk100` | 52.199583 | 29225.861 | 12.885780 | 29.425522 |
| `fused_nvtx_gpuonly.jsonl` | `fused_nvtx_gpuonly_topk100` | 52.236753 | 29278.852 | 12.875039 | 29.404584 |
| `split_novtx_offload.jsonl` | `split_novtx_offload_topk100` | 61.941520 | 37631.335 | 13.592378 | 24.797583 |
| `split_nvtx_offload.jsonl` | `split_nvtx_offload_topk100` | 62.364815 | 38012.558 | 13.629263 | 24.629272 |
| `fused_novtx_offload.jsonl` | `fused_novtx_offload_topk100` | 62.318250 | 37999.209 | 13.619676 | 24.647676 |
| `fused_nvtx_offload.jsonl` | `fused_nvtx_offload_topk100` | 61.396886 | 37208.522 | 13.470766 | 25.017555 |
| `/root/zoomkv_e2e/offload_bs16_e2e.jsonl` | `offload_bs16_e2e` | 1159.657244 | 567251.422 | 69.180766 | 10.596234 |

生成速度是 `1000 / mean_tpot_ms`，不是 `output_throughput`。

## 10. 系统实现

插件入口是 `sglang-plugin-ZoomKV/sglang_zoomkv/`。`SGLANG_PLUGINS=sglang_zoomkv` 之后，`load_plugin` 做三件事：注册名为 `zoomkv` 的注意力后端，把 decode CUDA Graph 固定成 `full`，并关掉 radix cache 和 chunked prefill。要在 nsys 里看每个 token 的分段时，另设 `SGLANG_ZOOMKV_BREAK_DECODE_GRAPH=1`，插件会改成 `breakable`。这八次没有设这个变量。Prefill 后端仍由启动参数指定为 FlashInfer。ZoomKV 自己的 prefill 回退也是 FlashInfer（`SGLANG_ZOOMKV_FALLBACK_BACKEND=flashinfer`），head dim 必须是 128。

| 路径 | 职责 |
|---|---|
| `sglang_zoomkv/__init__.py` | 插件加载，改写服务参数 |
| `attention/register.py` | 向 SGLang 注册 `zoomkv`，并接上 FlashInfer 回退 |
| `attention/backend.py` | `forward_extend` 里按层触发卸载；`forward_decode` 调 cache。36 层都拷完后归还检索页 |
| `attention/cuda_graph.py` | 等真实 batch 的 Quest 尺寸确定后再抓一张 decode 图 |
| `cache/polar_cache_kivi_quest_density_seperate_kernel.py` | 分区、Quest、KIVI、卸载拷贝、UVA gather、sparse attention |
| `kernels/quest/` | 大块和小块的打分。`quest_fused.cu` 是可选的合并 kernel，默认不走 |
| `kernels/topk/` | 独立的 topk |
| `kernels/kivi/` | 4-bit 精排用的 qk dot |
| `kernels/gather_trans/trans_h2d.py` | UVA：GPU kernel 直接读 CPU 锁页 KV |
| `kernels/sparse_attn/` | 只对 sink、local 和选中 token 做 decode attention |
| `config.py` | sink、local、final TopK、offload 开关。运行时用 `SGLANG_ZOOMKV_*` 覆盖。代码默认 TopK 是 1000，这八次用环境变量设成 100 |

Prefill 的实现顺序：

1. FlashInfer 把这一层的 K/V 写入 SGLang 页表。
2. GPU-only 只拿检索区的 key 去建 chunk min/max 和 KIVI packed key，不分配第二份 KV。
3. Offload 在这一层返回前，把检索区拷到一块连续缓冲，再在旁路 stream 上按 head 做异步 D2H。下一层的 attention 和这次拷贝重叠。索引用的是同一块已经转置好的 key，不再转置第二次。
4. 最后一层结束时同步拷贝，然后把 `req_to_token[sink, retrieval_end)` 交还给页分配器。请求结束时 chunk cache 跳过这段，避免释放两次。sink 和 local 的页留下。

Decode 的实现顺序，包在一张 CUDA Graph 里：

1. 大 chunk 打分，再对分数做 topk。
2. 在选中的大块里给小 chunk 打分，再 topk，map 回原始 chunk 下标。
3. KIVI 精排：chunk 内 dense TopK 16、sparse TopK 8，再取 `final_topk`。这八次是 100。
4. GPU-only 用加上 `sink_size` 的下标读页表。Offload 用 UVA 从 CPU 取出这 100 个 token 的 K/V，sparse kernel 再拼上页表里的 sink、local 和新 token。

服务启动时不抓图。短 warmup 会占住 batch-1 槽，128k 请求的序列长度和缓存长度对不上时会重建 Quest 索引，然后按这个真实尺寸抓图。日志里要有 `decode CUDA graph capture finished`。没有这一行时，decode 是逐 kernel 发射的，第 5 节的数字不能当成完整图的结果。
