# `quest_small_topk` 空档问题说明

本文对应 Nsight Systems 中 Hierarchical Quest 第二阶段里 `quest_small_topk` 这一段：NVTX 灰条很长，真正的 CUDA 蓝条却只出现在末尾，前后看起来像“空等”。结论先说清楚：

**空档不是 GPU kernel 在算，也不是 `cudaStreamSynchronize` 之类的显式同步。它是 CPU 在 `float_topk_kernel` launch 之前走完整段 host 包装路径所花的墙钟时间；这段时间里当前 stream 上没有新的工作可执行。**

---

## 1. Nsight 上看到什么

一次典型 decode 时间轴（单位以截图为准）：

| 层级 | 名称 | 时长 |
|---|---|---:|
| 外层 | `zoomkv_attention` | ~1.7 ms |
| | `quest_topk` | ~1.3 ms |
| | `stage1_coarse` | ~608 μs |
| | `topk_select` | ~298 μs |
| 问题段 | **`quest_small_topk`** | **~266 μs** |
| 后续 | `hq_map_back` | ~13 μs |

在 `quest_small_topk` 这一段内部：

1. **左边**刚结束的是上一个 kernel：`quest_sub_chunk_score_kernel`。它的 GPU 执行大约 9–15 μs，launch 也很快返回。
2. **中间**很长一段白色空白：CUDA API 行没有新的 launch，GPU kernel 行也没有新的工作。
3. **右边末端**才出现 `float_topk_kernel` 的 CUDA API 蓝条（约 10 μs 量级），紧接着 NVTX 结束。
4. 再往后才是 `hq_map_back` 对应的 `quest_map_back_kernel`（约 13 μs 的整段 range）。

因此：

- 灰条 `quest_small_topk` 的长度 ≈ **一次 `_topk_3d` / `float_topk_3d` 调用的 CPU 墙钟时间**。
- 蓝条位置 ≈ **真正执行 `float_topk_kernel<<<...>>>` 的时刻**。
- 两者不对齐，是因为 launch 被挡在函数末尾；前面两百多微秒 CPU 在忙，GPU 已经空了。

```
时间 →

quest_small_score          quest_small_topk (~266 μs)              hq_map_back
[score launch + GPU ~10μs] [=========== CPU host path ==========][map launch]
                            ↑ 无 CUDA API                         ↑ float_topk launch
                            GPU 空转                              蓝条贴右端
```

---

## 2. 这段 NVTX 实际包住了什么

打标点在 `_quest_pick_coarse_indices_hierarchical`：

```2376:2405:sglang-plugin-ZoomKV/sglang_zoomkv/cache/polar_cache_kivi_quest_density_seperate_kernel.py
        # --- Stage 2: score sub-chunks within selected large chunks ---
        _small_stage = nvtx_stage(SMALL_CHUNK_SELECT)
        _small_stage.__enter__()
        # nvtx.range_push("quest_small_score")
        sub_scores = self._cached_hq_scores_sub_view
        chunk_min = self._cached_chunk_min_views[layer_idx]
        chunk_max = self._cached_chunk_max_views[layer_idx]
        self._quest_sub_chunk_score(
            raw_q, chunk_min, chunk_max,
            large_idx, sub_scores,
            nk_large, factor,
        )
        # nvtx.range_pop()

        # nvtx.range_push("quest_small_topk")
        sub_topk_pos = self._topk_3d(sub_scores, nk_small)
        # nvtx.range_pop()

        # --- Map back to global small chunk indices ---
        # nvtx.range_push("hq_map_back")
        chunk_idx = self._cached_hq_chunk_idx
        self._quest_map_back(
            large_idx,
            sub_topk_pos,
            chunk_idx,
            factor,
            n_chunks,
        )
        # nvtx.range_pop()
        _small_stage.__exit__(None, None, None)
```

关键事实：

- `quest_small_topk` **只包住一行**：`self._topk_3d(sub_scores, nk_small)`。
- 它**不包含** `quest_sub_chunk_score`，也不包含 `quest_map_back`。
- 因此这段灰条里唯一允许出现的 CUDA kernel，就是 topk 那一次 launch。
- 若蓝条出现在灰条最右端，说明整段调用几乎都花在了 **launch 之前**。

`_topk_3d` 本身不做计算，只是转发：

```2217:2236:sglang-plugin-ZoomKV/sglang_zoomkv/cache/polar_cache_kivi_quest_density_seperate_kernel.py
    def _topk_3d(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k on last dim [bs, kv_heads, L] → indices [bs, kv_heads, k].
        ...
        """
        return float_topk_3d(scores, k)
```

---

## 3. 调用链：从一行 Python 到真正的 `<<<>>>`

完整路径如下。只有最后一步才会在 CUDA API 行上出现蓝条。

```
_quest_pick_coarse_indices_hierarchical
  └─ _topk_3d(sub_scores, nk_small)                  # Python，无 CUDA
       └─ float_topk_3d(scores, k)                   # Python wrapper
            ├─ 4 个 assert
            ├─ load_float_topk_module()              # 每次调用都进
            │    └─ load_jit(...)
            │         ├─ Path.resolve / 拼源文件路径
            │         ├─ _source_signature()         # resolve + is_file + stat + sha256
            │         └─ _load_cached(...)           # lru_cache 命中后返回已加载扩展
            └─ kernel.float_topk_3d(scores, k)       # 进入 C++
                 └─ float_topk_3d_cuda
                      ├─ reshape 成 2D
                      └─ float_topk_cuda
                           ├─ contiguous / 可能 cast
                           ├─ torch::empty({B, k})   # 每次新分配输出
                           ├─ CUDAGuard / 取 stream
                           ├─ cudaFuncSetAttribute   # 每次设置 64KB smem
                           └─ float_topk_kernel<<<>>>  ← 蓝条出现在这里
```

---

## 4. 为什么会空这么久：问题拆解

### 4.1 根因类型

这是 **CPU launch latency / host-path overhead**，不是：

- kernel 内部算得慢；
- 显式 `cudaDeviceSynchronize` / `cudaStreamSynchronize`；
- PCIe / UVA 拷贝卡住（那段在更后面的 `uva_read`）；
- NVTX 标错阶段（标的就是这一次 `_topk_3d`）。

上一个 kernel `quest_sub_chunk_score_kernel` 是异步提交的。GPU 很快跑完后，CPU 还在 Python → JIT 入口 → C++ 分配 → 属性设置这条路上。stream 队列空了，于是出现空白。

### 4.2 对比：为什么旁边的 Quest kernel 没有同样大的空档

同一文件里，Quest 扩展在初始化时加载一次，并保存在 `self._quest_kernel`：

```635:639:sglang-plugin-ZoomKV/sglang_zoomkv/cache/polar_cache_kivi_quest_density_seperate_kernel.py
            from sglang_zoomkv.kernels.quest.quest_ops import load_kernel_module as load_quest_kernel
            self._quest_kernel = load_quest_kernel()
```

热路径直接用已缓存对象发射，并把结果写进**预分配**的 `sub_scores`：

```2256:2262:sglang-plugin-ZoomKV/sglang_zoomkv/cache/polar_cache_kivi_quest_density_seperate_kernel.py
    def _quest_sub_chunk_score(...):
        if self._quest_kernel is not None:
            self._quest_kernel.quest_sub_chunk_score(
                raw_q, chunk_min, chunk_max, large_idx, sub_scores, nk_large, factor,
            )
```

对应 C++ 不分配输出，只 launch：

```302:322:sglang-plugin-ZoomKV/sglang_zoomkv/kernels/quest/quest.cu
    const int bs = raw_q.size(0);
    const int kv_heads = raw_q.size(1);
    c10::cuda::CUDAGuard guard(raw_q.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(nk_large * factor, bs * kv_heads);
    quest_sub_chunk_score_kernel<<<grid, kHeadDim, 0, stream>>>(
        ...
        sub_scores.data_ptr<float>(),
        ...);
```

而 topk 每次都重新走 `load_float_topk_module()`，并且在 C++ 里 `torch::empty` 新输出。两边不对齐，空档就落在 topk 这一侧。

### 4.3 开销来源 A：每次调用都重新进入 JIT 加载入口

```66:86:sglang-plugin-ZoomKV/sglang_zoomkv/kernels/topk/float_topk_ops.py
def float_topk_3d(
    scores: torch.Tensor,
    k: int
) -> torch.Tensor:
    assert scores.is_cuda, "scores must be on CUDA"
    assert scores.dim() == 3, "scores must be 3D [bs, kv_heads, n_chunks]"
    assert k > 0, f"k must be positive, got {k}"
    assert k <= scores.size(2), f"k={k} cannot exceed n_chunks={scores.size(2)}"

    kernel = load_float_topk_module()
    return kernel.float_topk_3d(scores, k)
```

`load_float_topk_module` **没有**把自己的结果缓存为模块级全局句柄；它每次都调用 `load_jit`：

```28:40:sglang-plugin-ZoomKV/sglang_zoomkv/kernels/topk/float_topk_ops.py
def load_float_topk_module(...):
    return load_jit(
        name, path, source_dir=Path(__file__).parent,
        cflags=cflags, cuda_flags=cuda_flags,
    )
```

`load_jit` 在命中 `_load_cached` 之前，每次都做路径解析和源文件签名：

```107:137:sglang-plugin-ZoomKV/sglang_zoomkv/kernels/jit.py
    base = Path(source_dir or os.curdir).resolve()
    resolved = tuple(
        str((base / source).resolve()) if not Path(source).is_absolute() else str(Path(source).resolve())
        for source in sources
    )
    signature = _source_signature(resolved)
    ...
    return _load_cached(
        name,
        resolved,
        signature,
        ...
    )
```

```24:34:sglang-plugin-ZoomKV/sglang_zoomkv/kernels/jit.py
def _source_signature(sources: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for source in sources:
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(...)
        stat = path.stat()
        digest.update(str(path).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(str(stat.st_size).encode())
    return digest.hexdigest()[:16]
```

这里的代价是：

- 多次 `Path.resolve()`；
- `is_file()` / `stat()` 文件系统调用；
- 用 mtime、size 拼 SHA256 得到 cache key；
- 然后才查 `lru_cache`。

扩展**不会重新编译**（缓存命中后 `_load_cached` 直接返回），但这些 host 工作仍然发生在 **每一次 decode、每一个需要 topk 的调用点**。它们不会出现在 CUDA API 行上，所以时间轴显示为空白。

同一层 hierarchical Quest 里，topk 至少会在 large topk、small topk、以及后续 CDS dense topk 各走一遍；问题在 small topk 上被 `quest_small_topk` 单独标出来，所以特别显眼。

### 4.4 开销来源 B：输出 indices 每次现分配

Quest 路径对分数和 map-back 输出都做了复用：

```1888:1899:sglang-plugin-ZoomKV/sglang_zoomkv/cache/polar_cache_kivi_quest_density_seperate_kernel.py
        # Output buffer for hq_map_back; reused every decode step to avoid
        # per-call torch.empty_like allocation on the hot path.
        self._cached_hq_chunk_idx = torch.empty(
            bs, kv_heads, nk_small, dtype=torch.int64, device=self.device,
        )
        ...
        self._cached_hq_scores_sub_view = (
            self._quest_scores_buffer_sub[:bs * kv_heads * nk_large * self._hq_factor]
            .view(bs, kv_heads, nk_large * self._hq_factor)
        )
```

但 `float_topk_cuda` 没有 out-tensor 参数，每次自己分配：

```293:324:sglang-plugin-ZoomKV/sglang_zoomkv/kernels/topk/float_topk.cu
    if (input.scalar_type() != torch::kFloat32) {
        input_float = input.to(torch::kFloat32).contiguous();
    } else {
        input_float = input.contiguous();
    }

    auto indices = torch::empty({B, k},
        torch::TensorOptions().dtype(torch::kInt64).device(input.device()));

    c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr size_t smem = 64 * 1024;
    auto err_attr = cudaFuncSetAttribute(
        float_topk_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem));
    ...
    float_topk_kernel<<<B, BLOCK_SIZE, smem, stream>>>(...);
```

含义：

1. `contiguous()`：即使输入已经连续，也要走一遍 PyTorch 检查/可能的 view 路径。
2. `torch::empty({B, k})`：每层、每次 topk 都向 caching allocator 申请一块 `[bs*kv_heads, k]` 的 int64 缓冲。
3. 缓存分配器命中时通常**不会**调用 `cudaMalloc`，因此 CUDA API 行仍可能是空的；但 CPU 仍要拿分配器锁、在池中找块、构造 Tensor。
4. 这一步发生在 `<<<>>>` **之前**，直接推迟蓝条出现时刻。

这和旁边 `hq_map_back` 形成对比：map-back 写进 `_cached_hq_chunk_idx`，整段只有约 13 μs。

### 4.5 开销来源 C：每次 launch 前设置 dynamic shared memory

`float_topk_kernel` 需要 64 KB dynamic shared memory，超过默认 48 KB 上限，所以 launch 前必须：

```cudaFuncSetAttribute(
    float_topk_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    64 * 1024);
```

这段属性对同一个函数其实只需要设一次，但当前实现写在每次 `float_topk_cuda` 的热路径上。它是真正的 CUDA driver 调用，会和随后的 launch 一起出现在蓝条附近，本身不解释灰条前半段空白；但它进一步增加了“从进入 C++ 到 kernel 真正入队”的延迟。

### 4.6 开销来源 D：Python / pybind 边界

相对 A/B 通常更小，但仍在 launch 前：

- `_topk_3d` → `float_topk_3d` 的 Python 函数调用；
- 4 个 `assert`（含格式化失败路径的字符串准备，成功路径也有属性读取）；
- pybind 把 Tensor 从 Python 传入 C++、再把 indices 传回；
- `float_topk_3d_cuda` 里 `reshape` 成 2D，返回后再 `reshape` 回 3D。

这些同样不产生 CUDA API 蓝条。

---

## 5. 把时间轴和代码对上

以 `quest_small_topk ≈ 266 μs` 为例，事件顺序是：

| 时刻 | 发生了什么 | 时间轴表现 |
|---|---|---|
| T0 | 上一阶段 `quest_sub_chunk_score` launch 并返回 | 左侧短蓝条 |
| T0+~10 μs | GPU 跑完 score kernel | GPU 行变空 |
| T1 | 进入 `quest_small_topk` / `_topk_3d` | 灰条开始 |
| T1 → T2 | assert、`load_jit` 路径解析/stat、取缓存扩展、进入 C++、`contiguous`、`torch::empty`、CUDAGuard | **大段空白** |
| T2 | `cudaFuncSetAttribute` + `float_topk_kernel<<<>>>` | 灰条右端短蓝条 |
| T2 之后很快 | `cudaGetLastError`、reshape、返回 Python、`range_pop` | 灰条结束 |
| T3 | `hq_map_back` launch | 下一段短灰条/蓝条 |

所以：

- **“空等”发生在 T1→T2**；
- **真正执行的蓝条在 T2**；
- GPU 从 score kernel 结束后到 T2 之前一直空闲，因为 CPU 还没把 topk 提交上去。

---

## 6. 问题本质（总结）

1. **现象**：`quest_small_topk` 约 266 μs，而其中 `float_topk_kernel` 只有约 10 μs，且贴在 range 右端。
2. **位置**：`polar_cache_kivi_quest_density_seperate_kernel.py` 的 `_topk_3d` → `float_topk_ops.py` 的 `float_topk_3d` → `float_topk.cu` 的 `float_topk_cuda`。
3. **性质**：异步流水线上的 **host launch 延迟**。短 kernel（约 10 μs）无法掩盖几百微秒级的 Python/JIT/分配包装。
4. **与相邻代码的不对称**：
   - Quest score / map-back：模块常驻 + 输出缓冲复用 → launch 很快；
   - float topk：每次重新走 `load_jit` 入口 + 每次 `torch::empty` 输出 + 每次 `cudaFuncSetAttribute` → launch 被推迟到函数末尾。
5. **影响面**：不只 small topk。同一条 `_topk_3d` / `float_topk_3d` 还用于 large topk、CDS dense topk 等；每层 decode 会重复多次，累计可到毫秒级。

---

## 7. 相关文件索引

| 文件 | 作用 |
|---|---|
| `sglang_zoomkv/cache/polar_cache_kivi_quest_density_seperate_kernel.py` | hierarchical Quest 热路径；`quest_small_topk` 打标；`_topk_3d` |
| `sglang_zoomkv/kernels/topk/float_topk_ops.py` | Python wrapper；每次 `load_float_topk_module()` |
| `sglang_zoomkv/kernels/jit.py` | `load_jit` / `_source_signature`；缓存前仍 resolve+stat |
| `sglang_zoomkv/kernels/topk/float_topk.cu` | `float_topk_kernel` 与 host launch；`torch::empty`、`cudaFuncSetAttribute` |
| `sglang_zoomkv/kernels/quest/quest.cu` | 对照：`quest_sub_chunk_score` / `quest_map_back` 写预分配缓冲 |
| `sglang_zoomkv/kernels/quest/quest_pipeline.py` | graph 路径上同样直接调用 `float_topk_3d`，存在同类问题 |

---

## 8. 改进方向（问题对应关系）

这些不是必须立刻改的结论，但与上述开销一一对应：

1. **缓存扩展句柄**：像 `_quest_kernel` 一样，在 init 或首次调用后保存 `float_topk` 模块，热路径不再进 `load_jit`。
2. **预分配 / out 参数**：为 large/small/CDS 的 indices 准备复用缓冲，让 C++ 写入已有 tensor，去掉热路径 `torch::empty`。
3. **一次性设置 smem 属性**：进程内对 `float_topk_kernel` 只调用一次 `cudaFuncSetAttribute`。
4. **进一步**：把 score → topk → map_back 融成更少次 launch / CUDA Graph，从根上减少短 kernel 之间的 host 空档。

在未改之前，Nsight 上 `quest_small_topk` 的“长灰条 + 右端短蓝条”应解读为：**topk 包装太重，GPU 在等 CPU 提交**，而不是 topk kernel 本身算得慢。
