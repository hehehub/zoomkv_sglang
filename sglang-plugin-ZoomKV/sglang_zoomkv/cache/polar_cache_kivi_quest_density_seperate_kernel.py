"""Polar ANN KV cache — Quest stage-1 + KIVI per-(quest-)chunk per-channel

ZoomKV production backend

Adapted from ``ZoomKV-main/cache_hub/polar_cache_kivi_quest_density_seperate_kernel.py``.


4-bit stage-2 rerank (offload-aware, _seperate_kernel layout).
Drop-in counterpart of
``sglang_zoomkv.cache.polar_cache_pure_rabitq_quest_density_seperate_kernel`` with
RaBitQ stage-2 replaced by KIVI 4-bit:

* Prefill / dynamic update: raw post-RoPE K → per-(quest-)chunk min/max
  (already maintained by ``_compute_and_store_chunk_minmax``) → KIVI 4-bit
  packed-along-head_dim int32 stored in ``key_kivi_packed_gpu`` in
  **chunk-major** layout (A-1):
  ``[bs, kv, n_chunks_max, n_pack, group_size]``.  Within each chunk, the
  ``group_size`` lanes for a fixed dim-block sit at consecutive int32
  positions, so the stage-2 kernel's per-warp packed-K read collapses to a
  single 64-byte coalesced load (vs. 16 strided loads in the older
  token-major layout).
* Decode stage-1: hierarchical or flat Quest chunk picking, only reads
  ``chunk_min`` / ``chunk_max`` / raw query.
* Decode stage-2: CDS dense/sparse classify (unchanged) →
  ``partial_chunk_kivi_qk_dense_sparse_cuda`` (replaces
  ``partial_rerank_chunk_dense_sparse_cuda``) → global top-K.

What got dropped vs. the RaBitQ pipeline:

* SRHT rotation (``fast_hadamard_transform``), diagonal sign matrix
  (``MultiDimBlockQuantizer``), and 3-bit magnitude codebook
  (``_mag_thresholds`` / ``_mag_centers``, ``polar_B`` / ``polar_m``).
* Query encoding (``fused_query_encode`` → ``q_blocks`` / ``q_norm``);
  KIVI uses the GQA-averaged raw query directly.
* ``key_4bit_packed_gpu`` and ``key_block_weight_gpu`` buffers; replaced
  by a single ``key_kivi_packed_gpu`` (int32, packed along head_dim).
* ``batch_encode`` / ``_encode_query_blocks`` /
  ``_rabitq_rerank_on_coarse_indices`` / RaBitQ-only debug branches.

What stayed identical:

* Offload plumbing (CPU pinned KV, copy stream, async D2H, H2D gather).
* Sink + local + retrieval geometry, ``current_seq_len`` bookkeeping,
  ``_update_polar_index`` flush trigger, live carry / skipped tokens.
* Quest stage-1 + CDS classify + fused global top-K.

Activate by setting
``POLAR_CACHE_MODULE=sglang_zoomkv.cache.polar_cache_kivi_quest_density_seperate_kernel``
before launching the runner.  KIVI stage-2 is fixed to CUDA, 4-bit, head_dim=128.
"""
import math
import os
from typing import Any, Optional

import torch
import torch.cuda.nvtx as nvtx

from sglang_zoomkv.kernels.rerank.rerank import (
    partial_chunk_density_scores_cuda,
    dense_mask_from_topk_cuda,
)

from sglang_zoomkv.kernels.kivi.quant_pack import _pack_along_last_dim

try:
    from sglang_zoomkv.kernels.kivi.kivi_qk_dot_ops import (
        partial_chunk_kivi_qk_dense_sparse_cuda,
    )
    _HAS_KIVI_CUDA = True
    _KIVI_CUDA_IMPORT_ERR = ""
except Exception as _e:  # noqa: BLE001 — report a hard init failure below
    partial_chunk_kivi_qk_dense_sparse_cuda = None
    _HAS_KIVI_CUDA = False
    _KIVI_CUDA_IMPORT_ERR = repr(_e)


_KIVI_BITS_DEFAULT = int(os.environ.get("POLAR_KIVI_BITS", "4"))
if _KIVI_BITS_DEFAULT != 4:
    raise ValueError(
        f"KIVI speed path requires POLAR_KIVI_BITS=4; got {_KIVI_BITS_DEFAULT}",
    )


# 导入基类
from sglang_zoomkv.cache.cache import KV_Cache
from sglang_zoomkv.attention.flash_attn_compat import flash_attn_with_kvcache_compat
# Steady Zone 动态更新的间隔已移至实例变量 self.dynamic_update_interval

class ZoomKVCache(KV_Cache):

    def __init__(
        self,
        valid_start,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        max_new_length: int,
        sink_size: int,
        local_size: int,
        core: int,
        nprobe: int,
        cache_unit_size: int,
        cache_cluster_num: int,
        num_gpus: int,
        model_size: int,
        codebook_path: str = '',  # ignored; kept for kwarg-compat with qwen.py
        # NOTE: SuCo 碰撞路径已移除，collision_ratio 仅作 backward-compat 占位
        # （qwen.py 始终往 polar_cache_kwargs 里塞这个键），运行期不再使用。
        collision_ratio: float = 0.3,
        candidate_ratio: float = 0.05,
        final_topk: int = 100,  # 精排阶段：从候选集中选 top-K 个 key
        dynamic_update_interval: int = 512,  # 每累积多少个 tokens 触发更新
        full_attention_threshold: int = 2000,  # prefill + decode < threshold 时使用全量 attention
        enable_offload: bool = True,
        debug_offload_coords: bool = False,  # ignored compatibility knob
        enable_rerank: bool = False,  # ignored compatibility knob
        enable_alpha_correction: bool = False,  # ignored compatibility knob
        enable_rerank_error_ablation: bool = False,  # ignored compatibility knob
        adaptive_topk_enabled: bool = False,  # ignored compatibility knob
        adaptive_topk_threshold: float = 0.9,  # ignored compatibility knob
        adaptive_topk_min_k: int = 100,  # ignored compatibility knob
        adaptive_topk_max_k: int = 400,  # ignored compatibility knob
        **_ignored_compat_kwargs,  # 吸收老调用方仍传入的兼容 kwarg
    ) -> None:
        if dtype != torch.bfloat16:
            raise ValueError(f"ZoomKV requires dtype=torch.bfloat16, got {dtype}")
        if head_dim not in (128, 256):
            raise ValueError(f"ZoomKV requires head_dim=128 or 256, got {head_dim}")
        if num_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_key_value_heads, got "
                f"{num_heads} and {num_key_value_heads}"
            )
        if max_length <= 0 or max_new_length <= 0 or max_new_length >= max_length:
            raise ValueError(
                "max_length/max_new_length must describe a non-empty prefill "
                f"(got {max_length}/{max_new_length})"
            )

        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, num_gpus, model_size)
         
        self.valid_start = valid_start  # 用于 prefill 时确定有效数据起始位置
        self.sink_size = sink_size  # Sink tokens 数量 = 4
        self.local_size = local_size   # Local window 大小 (通过配置文件设置,推荐512)
        # ---- aligned-offload (A 方案) ----
        # 只把 stage-1 对齐粒度整数倍的 retrieval token offload 到 CPU；不足一个对齐块
        # 的余数 R0 留在 GPU，当作 local 的延伸全量 attend（零 H2D，且 CPU retrieval 永远
        # 整除 → skipped=0）。余数恒定 = R0，所以环形缓冲机制不变，只是把 local_size 换成
        # 这里的 _eff_local_size = local_size + R0（prefill 时按真实 prefill 长度确定）。
        self._align_offload = os.environ.get("POLAR_ALIGN_OFFLOAD", "1") != "0"
        self._eff_local_size = self.local_size  # 实际生效的 GPU near-zone 窗口（prefill 时更新）
        self.steady_size = self.sink_size + self._eff_local_size  # Steady Zone 总大小
        # 动态更新间隔：每累积多少个新 tokens 触发一次 local zone 更新
        self.dynamic_update_interval = dynamic_update_interval if dynamic_update_interval is not None else self.local_size
        self.group_size = self.num_heads // self.kv_head # 每个kv head对应的query head的group数量=32/8=4
        self.batch_groups = self.batch_size * self.kv_head # batch_size × kv_head = 1 × 8 = 8
        self.dtype = dtype
        # 保存 max_new_length,用于计算 retrieval zone capacity
        self.max_new_length = max_new_length
        # qwen.py 传入: max_length = real_input_length + max_new_length
        self.input_length = self.max_length - max_new_length  # 实际 prefill 长度 (通过反推得到)
        self.FULL_ATTENTION_THRESHOLD = full_attention_threshold  # prefill + decode < threshold 时全量计算
        self.current_seq_len = [0] * layer_num  # 每层独立跟踪，prefill 后会初始化为 input_length
        _ = codebook_path  # explicitly ignored (KIVI has no codebook)
        _ = collision_ratio  # explicitly ignored; kept for kwarg compat with qwen.py
        _ = (
            debug_offload_coords,
            enable_rerank,
            enable_alpha_correction,
            enable_rerank_error_ablation,
            adaptive_topk_enabled,
            adaptive_topk_threshold,
            adaptive_topk_min_k,
            adaptive_topk_max_k,
        )
        self.candidate_ratio = candidate_ratio
        self.final_topk = final_topk  # 精排阶段的 top-K
        self.enable_offload = bool(enable_offload)
        # ----- Quest-inspired Stage-1 (this file's production retrieval path) -----
        self.quest_chunk_size = max(1, int(os.environ.get("POLAR_QUEST_CHUNK", "8")))
        # 当 retrieval 长度不是 chunk_size 的整数倍时，跨界 chunk 和 trailing partial 的
        # token 没有 chunk min/max 元数据，因此**不进 stage-1 评分**；但它们仍有合法的
        # RaBitQ 4-bit 编码，所以会以"额外候选"身份 cat 进 coarse_indices，**继续走 stage-2
        # rerank**。每次 prefill / decode update 最多产生 chunk_size 个跨界 + chunk_size-1
        # 个 trailing partial token。1024 decode @ dui=512 最多 2 次 flush ⇒ 上界 ~22；
        # slack 128 够 ~16k decode token。
        # live skipped 上界 = chunk_size * hq_factor - 1（hierarchical 时）或 chunk_size - 1（非分层）。
        # 默认 512 足够覆盖 chunk=8 × factor=64 的极端情况；遇到更激进配置时通过 env 调大。
        self.MAX_SKIPPED_INDICES = int(os.environ.get("POLAR_MAX_SKIPPED_INDICES", "512"))
        # ----- Quest coarse selection -----
        self._use_hierarchical_quest = (
            os.environ.get("POLAR_QUEST_HIERARCHICAL", "1")
            not in ("0", "false", "False", "")
        )
        self.quest_coarse_candidate_ratio = float(
            os.environ.get("POLAR_QUEST_CAND_RATIO", str(candidate_ratio))
        )
        _force_cand_raw = os.environ.get("POLAR_FORCE_CAND_TOKENS")
        self._force_cand_tokens = int(_force_cand_raw) if _force_cand_raw else None

        # ----- Hierarchical Quest (two-level chunk selection) -----
        self._hq_large_chunk_size = int(os.environ.get("POLAR_QUEST_LARGE_CHUNK", "256"))
        self._hq_large_ratio = float(os.environ.get("POLAR_QUEST_LARGE_RATIO", "0.3"))
        self._hq_small_ratio = float(os.environ.get("POLAR_QUEST_SMALL_RATIO", "0.3"))
        assert self._hq_large_chunk_size % self.quest_chunk_size == 0, (
            f"POLAR_QUEST_LARGE_CHUNK ({self._hq_large_chunk_size}) must be divisible "
            f"by POLAR_QUEST_CHUNK ({self.quest_chunk_size})"
        )
        self._hq_factor = self._hq_large_chunk_size // self.quest_chunk_size
        self._offload_align_chunk = (
            self._hq_large_chunk_size if self._use_hierarchical_quest else self.quest_chunk_size
        )
        # aligned-offload 依赖 decode 每次 offload 量是 large_chunk 的整数倍，否则
        # cpu_valid_end 会重新错位。dui 默认 = local_size；只要它整除 large_chunk 即可。
        if self.enable_offload and self._align_offload:
            assert self.dynamic_update_interval % self._offload_align_chunk == 0, (
                f"POLAR_ALIGN_OFFLOAD=1 requires dynamic_update_interval "
                f"({self.dynamic_update_interval}) be a multiple of align_chunk "
                f"({self._offload_align_chunk}); 否则 decode offload 会破坏对齐。"
                f" 可调 local_size / dui 或关闭 POLAR_ALIGN_OFFLOAD。"
            )
        # ----- CDS (Coarse → Dense/Sparse) pipeline (always on for this module) -----
        self.cds_dense_ratio = float(os.environ.get("POLAR_CDS_DENSE_RATIO", "0.1"))
        self.cds_dense_topk = int(os.environ.get("POLAR_CDS_DENSE_TOPK", "16"))
        self.cds_sparse_topk = int(os.environ.get("POLAR_CDS_SPARSE_TOPK", "8"))

        # KIVI 4-bit packed-along-head_dim stage-2 codec (no codebook needed).
        self._kivi_bits = _KIVI_BITS_DEFAULT
        self._kivi_feat_per_int = 32 // self._kivi_bits
        if head_dim % self._kivi_feat_per_int != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by "
                f"feat_per_int={self._kivi_feat_per_int} (bits={self._kivi_bits})"
            )
        self._kivi_n_pack = head_dim // self._kivi_feat_per_int
        if not _HAS_KIVI_CUDA:
            raise RuntimeError(
                "KIVI speed path requires the CUDA stage-2 kernel; "
                f"import failed with {_KIVI_CUDA_IMPORT_ERR}"
            )
        if head_dim not in (128, 256):
            raise ValueError(
                f"KIVI CUDA speed path requires head_dim=128 or 256; got {head_dim}"
            )
        # 初始化实际使用的 retrieval zone 长度
        self._actual_retrieval_length = None  # 将在 prefill 时设置
        # Quest stage-1 缓存：仅在 prefill 和 _update_polar_index 时刷新
        self._cached_quest_n_chunks = 0
        self._cached_quest_nk = 0
        self._cached_quest_kv_len = -1
        self._cached_scores_view = None
        self._cached_chunk_min_views = None
        self._cached_chunk_max_views = None
        self._cached_chunk_valid_views = None
        # Optional experiment-only recall logging.  Measures retrieval-zone
        # exact q·k recall for the stage-1 chunk pool and final CDS output.
        self._allayer_recall_check = (
            os.environ.get("POLAR_PIPELINE_RECALL", "0") not in ("0", "false", "False", "")
            or os.environ.get("POLAR_ALLAYER_RECALL", "0") not in ("0", "false", "False", "")
        )
        self._allayer_recall_log_every = int(os.environ.get(
            "POLAR_PIPELINE_RECALL_LOG_EVERY",
            os.environ.get("POLAR_ALLAYER_RECALL_LOG_EVERY", "100"),
        ))
        _recall_k_raw = os.environ.get(
            "POLAR_PIPELINE_RECALL_K",
            os.environ.get("POLAR_ALLAYER_RECALL_K", str(final_topk)),
        )
        self._allayer_recall_k = int(str(_recall_k_raw).split(",")[-1])
        self._allayer_recall_step = 0
        self._allayer_recall_accum = {}
        # Decode counter used to decide when the local/update buffer flushes.
        self._decode_step_counter = 0
        # 上次 buffer reset 时的 counter 值（长 prefill offload 用）
        self._buffer_base_counter = 0
        # 延迟 base 更新标志（确保 update 后的第一个 token 写到 offset 0）
        self._pending_buffer_reset = False
        

        self.device = torch.device(list[Any](layer_mapping.values())[0] if layer_mapping else 'cuda:0')

        # 统一 KV Cache 架构（按生成顺序存储所有 tokens）: 全量容量 = prefill + decode 的总长度
        self.unified_kv_capacity = self.input_length + self.max_new_length

        # ── decode_prepare bubble killer ──
        # Pre-allocated [0, unified_kv_capacity) int64 arange used for both
        # ``sink_indices`` (= self._arange_buffer[:sink_size]) and
        # ``local_indices`` (= self._arange_buffer[local_start:total_len]).
        # Both slices are zero-copy views; previously each compute() call
        # spent ~10us launching two torch.arange kernels, summing to ~360us
        # per decode token across 36 layers and showing up as a pre-stage1
        # bubble in nsys.
        self._arange_buffer = torch.arange(
            0, self.unified_kv_capacity, dtype=torch.long, device=self.device,
        )
        # Per-step memoisation of get_region_indices() output.  All 36 layers
        # in a single decode step see the same ``current_seq_len[layer_idx]``
        # (update_kv_cache increments AFTER compute()), so layer 0 builds the
        # dict and layers 1..35 reuse it via dict-identity check.  Saves
        # ~2us python work + 1 conditional branch per layer.
        self._cached_region_total_len: int = -1
        self._cached_region_info = None
        # Per-layer cache of ``key_kivi_packed_gpu[l][:, :, :n_valid_chunks]``.
        # The slice itself is stride-only (zero kernel launch) but goes
        # through PyTorch's ``__getitem__`` Python dispatch (~3-5us).
        # ``n_valid_chunks`` only advances every CHUNK_SIZE (=16) tokens,
        # so cache hit rate is ~15/16; saves ~3-5us × 36 layers = ~150us
        # per decode token in steady state.
        self._cached_kivi_packed_n_chunks: list = [-1] * layer_num
        self._cached_kivi_packed_views: list = [None] * layer_num

        
        # Offload mode stores sink/local/update on GPU and retrieval on CPU.
        # GPU-only mode stores the complete chronological sidecar on GPU.
        # aligned-offload 会把不足一个 stage-1 对齐块的余数留在 GPU near-zone，
        # 所以 GPU 容量要多预留 (align_chunk - 1)。
        self._eff_local_extra = (self._offload_align_chunk - 1) if self._align_offload else 0
        self.gpu_kv_capacity = (
            self.sink_size + self.local_size + self._eff_local_extra
            + self.dynamic_update_interval
            if self.enable_offload
            else self.unified_kv_capacity
        )

        self._copy_stream = (
            torch.cuda.Stream(device=self.device) if self.enable_offload else None
        )
        # - main_event: 标记源数据准备好（default stream → copy stream 的依赖）
        # - copy_event: 标记该层 D2H 拷贝完成（CPU 数据可安全读取）
        self._main_events = (
            [torch.cuda.Event() for _ in range(layer_num)]
            if self.enable_offload else []
        )
        self._copy_events = (
            [torch.cuda.Event() for _ in range(layer_num)]
            if self.enable_offload else []
        )
        # 让 copy_event 初始处于“已完成”状态，避免第一次 synchronize() 卡住
        if self.enable_offload:
            with torch.cuda.device(self.device):
                for e in self._copy_events:
                    e.record()

        # 保存仍在 D2H 拷贝中的源 tensor 引用，避免 allocator 复用导致潜在数据竞争
        # 每层一个 (keys, values) tuple；拷贝完成后会清理
        self._inflight_offload_buffers = [None for _ in range(layer_num)]
        self._layer_offload_inflight = [False for _ in range(layer_num)]
        # CPU 只存 Retrieval Zone，不含 Sink + Local
        self.cpu_kv_capacity = max(
            0, self.unified_kv_capacity - self.sink_size - self.local_size
        )

        # GPU-only reads the SGLang paged pool in place. Offload still keeps a
        # compact sink/local buffer on GPU and the retrieval zone on CPU.
        self._use_pool_kv = not self.enable_offload
        self._sglang_kv_pool = None
        self._slot_table = None
        self._pool_row_ids = None
        self.unified_keys_gpu = []
        self.unified_values_gpu = []
        if self.enable_offload:
            for ldx in range(layer_num):
                self.unified_keys_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.gpu_kv_capacity, head_dim), dtype=dtype, device=self.device).contiguous())
                self.unified_values_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.gpu_kv_capacity, head_dim), dtype=dtype, device=self.device).contiguous())
        else:
            self.unified_keys_gpu = [None] * layer_num
            self.unified_values_gpu = [None] * layer_num
            print("[Init] gpu-only reads the SGLang KV pool; sidecar KV is not allocated", flush=True)
        self.unified_keys_cpu = []
        self.unified_values_cpu = []
        self._k_scale = []
        self._v_scale = []
        # One scale per head per 128 retrieval tokens. Values are packed int4
        # (two per byte) so six parked 128k prefills stay inside the memory
        # that already held five int8 copies. empty() does not touch the pages.
        self._cpu_kv_block = 128
        if self.enable_offload and self.cpu_kv_capacity > 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            host_empty = getattr(torch.cuda, "host_empty_cache", None)
            if host_empty is not None:
                host_empty()
            n_blocks = (self.cpu_kv_capacity + self._cpu_kv_block - 1) // self._cpu_kv_block
            packed_dim = head_dim // 2
            for ldx in range(layer_num):
                self.unified_keys_cpu.append(torch.empty(
                    (batch_size, self.kv_head, self.cpu_kv_capacity, packed_dim),
                    dtype=torch.uint8, pin_memory=True,
                ))
                self.unified_values_cpu.append(torch.empty(
                    (batch_size, self.kv_head, self.cpu_kv_capacity, packed_dim),
                    dtype=torch.uint8, pin_memory=True,
                ))
                self._k_scale.append(torch.ones(
                    (batch_size, self.kv_head, n_blocks),
                    dtype=torch.float32, device=self.device,
                ))
                self._v_scale.append(torch.ones(
                    (batch_size, self.kv_head, n_blocks),
                    dtype=torch.float32, device=self.device,
                ))
            kv_bytes = (
                2 * layer_num * batch_size * self.kv_head
                * self.cpu_kv_capacity * packed_dim
            )
            print(
                f"[ZoomKV] retrieval KV is packed int4, {kv_bytes / (1024 ** 3):.2f} GiB/rank "
                f"(bf16 would be {4 * kv_bytes / (1024 ** 3):.2f} GiB)",
                flush=True,
            )
            self._log_host_rss("after retrieval KV alloc")

        # 记录 CPU 上已存储的有效范围
        self.cpu_valid_end = [0] * layer_num
        # 预分配 TopK gather 结果的 GPU buffer（避免每次 gather 后分配新内存）
        # 使用 final_topk 参数作为 buffer 大小
        max_topk_size = self.final_topk + self.MAX_SKIPPED_INDICES
        self._topk_keys_buffer = torch.empty(
            (batch_size, self.kv_head, max_topk_size, head_dim),dtype=dtype, device=self.device)
        self._topk_values_buffer = torch.empty(
            (batch_size, self.kv_head, max_topk_size, head_dim),dtype=dtype, device=self.device)

        # 预分配 Offload staging buffers（避免 clone + 避免跨层复用导致数据竞争）
        # 每层一个 staging buffer：否则在 _update_polar_index 的 layer 循环里复用同一个 buffer，
        # 会和异步 copy_stream 产生 race（上一层还在 D2H，下一层就覆盖 src）。
        self._offload_keys_buffer = [
            torch.empty(
                (batch_size, self.kv_head, self.dynamic_update_interval, head_dim),
                dtype=dtype, device=self.device
            )
            for _ in range(layer_num)
        ]
        self._offload_values_buffer = [
            torch.empty(
                (batch_size, self.kv_head, self.dynamic_update_interval, head_dim),
                dtype=dtype, device=self.device
            )
            for _ in range(layer_num)
        ]
        # 仅 offload 模式需要 pinned CPU staging；GPU-only 不分配 CPU 缓存。
        if self.enable_offload:
            self._topk_keys_cpu_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk_size, head_dim),
                dtype=dtype, device='cpu', pin_memory=True
            )
            self._topk_values_cpu_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk_size, head_dim),
                dtype=dtype, device='cpu', pin_memory=True
            )
            self._topk_indices_cpu_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk_size),
                dtype=torch.int64, device='cpu', pin_memory=True
            )
        else:
            self._topk_keys_cpu_buffer = None
            self._topk_values_cpu_buffer = None
            self._topk_indices_cpu_buffer = None

        # 注意：在统一架构下，retrieval_zone_capacity 指的是 unified_kv_capacity
        self.retrieval_zone_capacity = self.unified_kv_capacity
        # Pre-compute chunk capacity here so the chunk-major KIVI packed-K
        # allocation below can size axis-2 by chunks (rather than tokens).
        # ``_chunk_n_max`` is also (re)assigned in the Quest-coarse block
        # below; we set it eagerly here because the KIVI packed buffer
        # always needs it under the A-1 chunk-major layout.
        _kivi_chunk = self.quest_chunk_size
        self._chunk_n_max = (
            self.retrieval_zone_capacity + _kivi_chunk - 1
        ) // _kivi_chunk
        # Offload keeps the Quest index in pinned CPU memory and exposes it
        # through a UVA alias. Score buffers stay on GPU. GPU-only is unchanged.
        self._index_cpu_keepalive = []
        self._index_host_of = {}
        # CPU keeps every request's index so a high batch still fits. The
        # request that is about to decode gets one GPU copy, then each token
        # reads that copy instead of pinned host memory.
        self._offload_index_on_gpu = False

        def alloc_index(shape, dtype, zero=False):
            # A zero-length dim (warmup shorter than one large chunk) cannot be
            # pinned: is_pinned() is false, and uva_alias rejects it.
            if (not self.enable_offload) or any(int(dim) <= 0 for dim in shape):
                made = (
                    torch.zeros(shape, dtype=dtype, device=self.device)
                    if zero
                    else torch.empty(shape, dtype=dtype, device=self.device)
                )
                return made.contiguous()
            # Masks must start false. The packed index and min/max are written
            # before any read of a valid chunk, so they stay uncommitted until
            # the prefill fill. zeros() here was dirtying the whole index at
            # construction and stacking on top of the KV allocation.
            cpu = (
                torch.zeros(shape, dtype=dtype, pin_memory=True)
                if zero
                else torch.empty(shape, dtype=dtype, pin_memory=True)
            )
            self._index_cpu_keepalive.append(cpu)
            from sglang_zoomkv.kernels.gather_trans.trans_h2d import uva_alias
            alias = uva_alias(cpu, self.device)
            self._index_host_of[id(alias)] = cpu
            return alias

        # ============ KIVI 4-bit packed-along-head_dim 存储 (chunk-major) ============
        # A-1 layout: [batch, kv_head, n_chunks_max, n_pack, group_size] int32.
        # Within one chunk, lane k (0..g-1) sits at the innermost axis, so a
        # warp's 16 lanes for a fixed dim-block read 16 consecutive int32
        # elements -> one 64-byte coalesced load (vs. 16 strided loads in
        # the previous token-major layout).  The retrieval-internal chunk
        # id from Quest stage-1 is the axis-2 index directly; the stage-2
        # kernel no longer needs a per-token offset / sink_size shift.
        self.key_kivi_packed_gpu = []
        for ldx in range(layer_num):
            self.key_kivi_packed_gpu.append(
                alloc_index(
                    (batch_size, self.kv_head, self._chunk_n_max,
                     self._kivi_n_pack, _kivi_chunk),
                    torch.int32,
                )
            )

        # ============ Chunk-level raw K min/max metadata (Quest-style upper bound) ============
        # shape: [batch, kv_head, n_chunks_max, head_dim]
        # 数学：score(chunk) = Σ_d max(q_d * max_k_d, q_d * min_k_d)，对应 Quest paper Sec.3.1
        # 存储 ≈ KV cache 的 1/chunk_size（chunk=8 时约 12.5%）
        chunk = self.quest_chunk_size
        self._chunk_n_max = (self.retrieval_zone_capacity + chunk - 1) // chunk
        self._chunk_key_min_gpu = []
        self._chunk_key_max_gpu = []
        self._chunk_key_centroid_gpu = []
        self._chunk_valid_mask_gpu = []
        for ldx in range(layer_num):
            self._chunk_key_min_gpu.append(
                alloc_index((batch_size, self.kv_head, self._chunk_n_max, head_dim), dtype)
            )
            self._chunk_key_max_gpu.append(
                alloc_index((batch_size, self.kv_head, self._chunk_n_max, head_dim), dtype)
            )
            self._chunk_key_centroid_gpu.append(
                alloc_index((batch_size, self.kv_head, self._chunk_n_max, head_dim), dtype)
            )
            self._chunk_valid_mask_gpu.append(
                alloc_index((batch_size, self.kv_head, self._chunk_n_max), torch.bool, zero=True)
            )
        # 记录每层已写入的 chunk 数（== floor(retrieval_token_count / chunk_size)）
        self._chunk_valid_end = [0] * layer_num
        # carry: 把上次未满一个 small chunk 的尾巴 token 暂存下来；下次 update
        # 先与新 token 拼接再切块，避免每次都产生重复 head/tail skipped。
        self._chunk_carry_count = [0] * layer_num
        carry_cap = max(0, chunk - 1)
        self._chunk_carry_keys = []
        self._chunk_merge_buffer = []
        for ldx in range(layer_num):
            self._chunk_carry_keys.append(
                torch.zeros(
                    (batch_size, self.kv_head, carry_cap, head_dim),
                    dtype=dtype,
                    device=self.device,
                ).contiguous()
            )
            # 临时拼接 [carry + new_head] 用：固定 chunk 大小，避免 decode 热路径频繁 cat 分配。
            self._chunk_merge_buffer.append(
                torch.empty(
                    (batch_size, self.kv_head, chunk, head_dim),
                    dtype=dtype,
                    device=self.device,
                ).contiguous()
            )
        # 跨界 chunk + trailing partial 的 retrieval-zone 索引，每层独立维护。
        # shape: [batch, kv_head, MAX_SKIPPED_INDICES] int64；attention 阶段直接 cat 进 topk。
        # 注意：所有 batch / kv_head 共享同一组 skipped indices（chunk 边界与 batch 无关），
        # 但为了和 topk_indices [bs, kv, K] 形状对齐，我们把同一份索引广播到 [bs, kv]。
        self._chunk_skipped_indices_gpu = []
        for ldx in range(layer_num):
            self._chunk_skipped_indices_gpu.append(
                torch.zeros(
                    (batch_size, self.kv_head, self.MAX_SKIPPED_INDICES),
                    dtype=torch.int64,
                    device=self.device,
                ).contiguous()
            )
        self._chunk_skipped_count = [0] * layer_num
        # Pre-create chunk expansion offsets [0, 1, ..., chunk-1] used to fan-out
        # chunk indices to token indices in the hot path.
        self._chunk_offs = torch.arange(chunk, device=self.device, dtype=torch.int64).view(1, 1, 1, chunk)
        # --- Hierarchical Quest: large chunk buffers ---
        self._chunk_n_max_large = self._chunk_n_max // self._hq_factor
        self._chunk_key_min_gpu_large = []
        self._chunk_key_max_gpu_large = []
        self._chunk_valid_mask_gpu_large = []
        for ldx in range(layer_num):
            self._chunk_key_min_gpu_large.append(
                alloc_index(
                    (batch_size, self.kv_head, self._chunk_n_max_large, head_dim), dtype
                )
            )
            self._chunk_key_max_gpu_large.append(
                alloc_index(
                    (batch_size, self.kv_head, self._chunk_n_max_large, head_dim), dtype
                )
            )
            self._chunk_valid_mask_gpu_large.append(
                alloc_index(
                    (batch_size, self.kv_head, self._chunk_n_max_large), torch.bool, zero=True
                )
            )
        if self.enable_offload:
            print(
                "[ZoomKV] offload Quest index lives in pinned CPU memory",
                flush=True,
            )
            self._log_host_rss("after Quest index alloc")
        self._chunk_valid_end_large = [0] * layer_num
        self._quest_scores_buffer_small = torch.empty(
            batch_size * self.kv_head * self._chunk_n_max,
            dtype=torch.float32, device=self.device,
        )
        self._quest_scores_buffer_large = torch.empty(
            batch_size * self.kv_head * self._chunk_n_max_large,
            dtype=torch.float32, device=self.device,
        )
        self._quest_scores_buffer_sub = torch.empty(
            batch_size * self.kv_head * self._chunk_n_max_large * self._hq_factor,
            dtype=torch.float32, device=self.device,
        )

        self.context = 0
        self._preallocated_buffers = {}


        # 设置 PyTorch CPU 线程数（只在第一次初始化时设置）
        if not hasattr(ZoomKVCache, '_threads_initialized'):
            torch.set_num_threads(1)          # 8 关闭多线程，单线程运行
            try:
                torch.set_num_interop_threads(1)  # inter-op 也用 1
            except RuntimeError:
                pass  # 已经设置过或并行工作已启动，忽略
            ZoomKVCache._threads_initialized = True
            # print(f"  [Thread Config] intra-op={torch.get_num_threads()}, inter-op={torch.get_num_interop_threads()}")
        
        self._gather_ext = None
        if self.enable_offload:
            try:
                from sglang_zoomkv.kernels.gather import load_gather_ext
                self._gather_ext = load_gather_ext()
            except Exception as e:
                print(f"[Init] gather ext unavailable, using torch.gather: {e}", flush=True)
        
        # 预加载 Radix TopK CUDA extension (O(n) vs O(n log k))
        try:
            from sglang_zoomkv.kernels.topk import load_radix_topk_ext
            self._radix_topk_ext = load_radix_topk_ext()
        except Exception as e:
            print(f"[Init]   radix_topk FAILED: {e}", flush=True)
            self._radix_topk_ext = None

        # Quest top-k and the final gather-concat each used to load their
        # extension on every call. Compile them once, before the layer loop.
        nvtx.range_push("load_topk_modules")
        try:
            try:
                from sglang_zoomkv.kernels.topk.float_topk_ops import load_float_topk_module
                self._float_topk_ext = load_float_topk_module()
            except Exception as e:
                raise RuntimeError(f"float_topk CUDA kernel compile failed: {e}")
            try:
                from sglang_zoomkv.kernels.topk.float_topk_gather_concat_ops import (
                    load_fused_topk_module,
                )
                self._fused_topk_ext = load_fused_topk_module()
            except Exception as e:
                raise RuntimeError(
                    f"float_topk_gather_concat CUDA kernel compile failed: {e}"
                )
        finally:
            nvtx.range_pop()
            pass

        # Staged copies stay as the fallback. The hot path prefers one GPU
        # kernel that reads the pinned retrieval buffers over PCIe.
        self._uva_gather_kv = None
        self._h2d_gather_kv = None
        self._offload_extra_counts = None
        self._offload_extra_counts_cpu = None
        if self.enable_offload:
            from sglang_zoomkv.kernels.gather_trans.staged_h2d import (
                staged_h2d_gather_kv,
            )
            self._h2d_gather_kv = staged_h2d_gather_kv
            self._try_enable_uva_gather()
        
        # 预加载 CDS density + dense-mask CUDA extension (rerank_topk shares
        # the same .so as the RaBitQ rerank kernel — we only use the
        # density/dense-mask entry points here).
        try:
            from sglang_zoomkv.kernels.rerank.rerank import load_kernel_module
            self._rerank_topk_kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
        except Exception as e:
            raise RuntimeError(f"rerank_topk CUDA kernel compile failed: {e}")

        # 预加载 KIVI stage-2 CUDA kernel.
        try:
            from sglang_zoomkv.kernels.kivi.kivi_qk_dot_ops import _kivi_kernel as _load_kivi_kernel
            self._kivi_kernel = _load_kivi_kernel()
        except Exception as e:
            raise RuntimeError(f"kivi_qk_dot CUDA kernel compile failed: {e}")

        # 预加载 Quest chunk score CUDA kernel
        try:
            from sglang_zoomkv.kernels.quest.quest_ops import load_kernel_module as load_quest_kernel
            self._quest_kernel = load_quest_kernel()
        except Exception as e:
            print(f"[Init]   quest_chunk_score CUDA failed ({e}), falling back to PyTorch", flush=True)
            self._quest_kernel = None

        # Compile the replayed stage markers before the decode graph is captured.
        from sglang_zoomkv.kernels.nvtx_ranges import preload_stage_nvtx
        preload_stage_nvtx()

        # Query-chunk summary dot and top-k stay separate launches.
        # Set SGLANG_ZOOMKV_FUSED_QUEST=1 to merge them into one kernel.
        self._fused_quest_ext = None
        fused_flag = os.environ.get("SGLANG_ZOOMKV_FUSED_QUEST", "0")
        self._fused_quest_enabled = fused_flag not in ("0", "false", "False", "")
        if self._fused_quest_enabled and self._use_hierarchical_quest:
            try:
                from sglang_zoomkv.kernels.quest.quest_fused_ops import load_fused_quest
                self._fused_quest_ext = load_fused_quest()
                print("[Quest] score and topk are one fused kernel", flush=True)
            except Exception as e:
                print(f"[Init]   fused quest CUDA failed ({e}), using the split kernels", flush=True)
                self._fused_quest_ext = None
        else:
            print("[Quest] score and topk stay separate kernels", flush=True)

        # Per-layer CUDA graph of the separate Quest kernels. One replay per
        # layer. Score and top-k stay two launches inside that graph.
        graph_flag = os.environ.get("SGLANG_ZOOMKV_QUEST_GRAPH", "1")
        self._quest_graph_enabled = graph_flag not in ("0", "false", "False", "")
        if self._quest_graph_enabled:
            print("[Quest] per-layer quest graph is on", flush=True)
        else:
            print(
                "[Quest] per-layer quest graph is off; score and topk launch separately",
                flush=True,
            )
        self._quest_graphs = {}
        self._quest_graph_pool = None
        self._quest_graph_broken = False

        # GPU-only: Quest and index attention share one CUDA graph per layer.
        # Offload still gathers through the staged H2D path.
        sparse_flag = os.environ.get("SGLANG_ZOOMKV_SPARSE_ATTN", "1")
        self._sparse_attn_enabled = sparse_flag not in ("0", "false", "False", "")
        self._sparse_ext = None
        self._sparse_splits = 64
        self._sparse_partial_m = None
        self._sparse_partial_l = None
        self._sparse_partial_acc = None
        self._sparse_out = None
        self._sparse_bounds = None
        self._sparse_bounds_cpu = None
        self._sparse_bounds_cached = None
        self._sparse_extra_count = None
        self._sparse_extra_count_cpu = None
        self._sparse_extra_cached = None
        self._sparse_empty_indices = None
        self._sparse_graphs = {}
        self._offload_sparse_graphs = {}
        self._sparse_graph_broken = False
        self._offload_sparse_broken = False
        self._sparse_announced = False
        self._graph_local_start_host = None
        self._graph_window_announced = False
        self._graph_avg_q = None
        self._graph_keepalive = []
        self._workspace_frozen = False
        if self._sparse_attn_enabled:
            try:
                from sglang_zoomkv.kernels.sparse_attn.sparse_decode_attn_ops import (
                    load_sparse_decode_attn,
                )
                self._sparse_ext = load_sparse_decode_attn()
                if self.enable_offload:
                    print("[Init]   offload sparse attention enabled", flush=True)
                else:
                    print("[Init]   gpu-only sparse attention enabled", flush=True)
            except Exception as e:
                print(f"[Init]   sparse attention CUDA failed ({e}), using gather", flush=True)
                self._sparse_attn_enabled = False

        # Keep allocator state warm for speed tests; avoid gc/empty_cache here.

    # 处理单层 prefill 阶段
    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, batch_idx):
        """
        Prefill 阶段入口：
        - bsz==1: 按 batch_idx 把单样本写入对应 cache slot（外部按 batch_idx 循环调用）。
        - bsz>1 (lockstep): 一次性处理整个 batch（要求各样本有效长度/valid_start 一致），
          只在 batch_idx==0 时执行，避免外部仍按 batch_idx 循环导致重复工作。
        两条路径只是 batch 维写入切片不同，统一走 ``_prefill_store``。
        """
        bsz = key_states.shape[0]
        if bsz == 1:
            return self._prefill_store(
                key_states, value_states, layer_idx, dst_offset=batch_idx, src_bsz=1
            )
        # bsz>1：只在 batch_idx==0 时执行一次
        if batch_idx != 0:
            # 保持返回 shape 兼容：去掉 padding（按 batch_idx=0 的 valid_start 规则）
            vs = self.valid_start[0] if isinstance(self.valid_start, (list, tuple)) else int(self.valid_start)
            return key_states[:, vs:, :, :], value_states[:, vs:, :, :]
        return self._prefill_store(
            key_states, value_states, layer_idx, dst_offset=0, src_bsz=bsz
        )

    def _resolve_valid_start(self, dst_offset: int, src_bsz: int) -> int:
        """Resolve the (batch-consistent) ``valid_start`` for this prefill call.

        For ``src_bsz>1`` (lockstep) we require an identical valid_start across
        the involved batch slots (per-batch region划分否则需要更大改动)。
        """
        vs = self.valid_start
        if not hasattr(vs, '__len__'):
            return int(vs)
        if src_bsz > 1:
            assert len(vs) >= dst_offset + src_bsz
            vs0 = int(vs[dst_offset])
            for b in range(dst_offset + 1, dst_offset + src_bsz):
                assert int(vs[b]) == vs0, (
                    f"lockstep multi-batch currently requires same valid_start across batch. "
                    f"valid_start[{dst_offset}]={vs0}, valid_start[{b}]={vs[b]}"
                )
            return vs0
        return int(vs[dst_offset])

    def _prefill_store(self, key_states, value_states, layer_idx, *, dst_offset: int, src_bsz: int):
        """Prefill ``src_bsz`` 个样本到 CPU-offload 或完整 GPU sidecar。

        统一了原 ``_prefill_single_sample`` (dst_offset=batch_idx, src_bsz=1) 与
        ``_prefill_lockstep_batch`` (dst_offset=0, src_bsz=batch_size) 两条路径，
        行为完全一致——区别仅在于 batch 维的写入切片。

        区域划分（zone-internal offset 从 0 起）：
            Sink      = [valid_start, valid_start+sink_size)        → GPU
            Retrieval = [sink_end, max(sink_end, seq_len-local_size)) → 异步 offload 到 CPU
            Local     = [retrieval_end, seq_len)                    → GPU
        KIVI 4-bit packing 折叠进 ``_compute_and_store_chunk_minmax`` —— 每个 chunk
        的 (mn, mx, packed_codes) 三元组在同一处一次写齐，保证 stage-2 读到一致状态。
        """
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert seq_len <= self.input_length
        b_sl = slice(dst_offset, dst_offset + src_bsz)
        is_last_layer = (layer_idx == self.layer_num - 1)
        is_last_chunk = (dst_offset + src_bsz == self.batch_size)

        valid_start = self._resolve_valid_start(dst_offset, src_bsz)
        sink_start = valid_start
        sink_end = valid_start + self.sink_size
        retrieval_start = sink_end
        # 默认 retrieval/local 边界
        raw_retrieval_end = max(retrieval_start, seq_len - self.local_size)
        align_chunk = self._offload_align_chunk
        raw_retrieval_len = raw_retrieval_end - retrieval_start
        if self.enable_offload and self._align_offload and raw_retrieval_len >= align_chunk:
            # 把 offload 到 CPU 的 retrieval 长度向下取整到 align_chunk；不足的余数 R0
            # 留在 GPU，并入 local（_eff_local_size = local_size + R0）。
            aligned_retrieval_len = (raw_retrieval_len // align_chunk) * align_chunk
            r0 = raw_retrieval_len - aligned_retrieval_len
            retrieval_end = retrieval_start + aligned_retrieval_len
            self._eff_local_size = self.local_size + r0
        else:
            retrieval_end = raw_retrieval_end
            self._eff_local_size = self.local_size
        self.steady_size = self.sink_size + self._eff_local_size
        local_start = retrieval_end
        local_end = seq_len
        retrieval_length = retrieval_end - retrieval_start
        valid_length = retrieval_length

        # 首次 prefill：按真实 batch 维重新分配精确大小的 topk buffer（避免每次 slice）
        if self._actual_retrieval_length is None:
            self._actual_retrieval_length = valid_length
            self._refresh_quest_chunk_cache(valid_length)
            bs = self.batch_size
            max_topk = self.final_topk + self.MAX_SKIPPED_INDICES
            self._topk_keys_buffer = torch.empty((bs, self.kv_head, max_topk, self.head_dim), dtype=self.dtype, device=self.device)
            self._topk_values_buffer = torch.empty((bs, self.kv_head, max_topk, self.head_dim), dtype=self.dtype, device=self.device)
            if self.enable_offload:
                self._topk_keys_cpu_buffer = torch.empty((bs, self.kv_head, max_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
                self._topk_values_cpu_buffer = torch.empty((bs, self.kv_head, max_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
                self._topk_indices_cpu_buffer = torch.empty((bs, self.kv_head, max_topk), dtype=torch.int64, device='cpu', pin_memory=True)

        if not self.enable_offload:
            # The tokens already live in the SGLang pool. Build Quest metadata
            # from the prefill keys and do not keep a second copy.
            actual_len = seq_len - valid_start
            if actual_len > self.unified_kv_capacity:
                raise RuntimeError(
                    f"GPU-only sequence length exceeded capacity: {actual_len} > "
                    f"{self.unified_kv_capacity}"
                )
            self.current_seq_len[layer_idx] = actual_len
            if layer_idx == 0:
                self._graph_local_start_host = int(local_start)
                if not self._graph_window_announced:
                    self._graph_window_announced = True
                    print(
                        f"[ZoomKV] graph local window starts at {int(local_start)}",
                        flush=True,
                    )
            if valid_length > 0:
                retrieval_keys = (
                    key_states[:src_bsz, retrieval_start:retrieval_end, :, :]
                    .transpose(1, 2)
                    .contiguous()
                )
                nvtx.range_push("prefill_build_index")
                self._compute_and_store_chunk_minmax(
                    layer_idx,
                    retrieval_keys,
                    abs_token_start=0,
                    batch_slice=b_sl,
                )
                nvtx.range_pop()
            if is_last_layer and is_last_chunk:
                self.context += actual_len
            return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

        # 短 prefill：没有 retrieval zone，全部留在 GPU（不触发 offload）
        if valid_length == 0:
            if layer_idx == 0 and not hasattr(self, '_short_prefill_printed'):
                self._short_prefill_printed = True
            actual_len = min(seq_len - valid_start, self.gpu_kv_capacity)
            nvtx.range_push("prefill_gpu_copy")
            self.unified_keys_gpu[layer_idx][b_sl, :, :actual_len, :] = \
                key_states[:src_bsz, valid_start:valid_start+actual_len, :, :].transpose(1, 2).to(self.dtype)
            self.unified_values_gpu[layer_idx][b_sl, :, :actual_len, :] = \
                value_states[:src_bsz, valid_start:valid_start+actual_len, :, :].transpose(1, 2).to(self.dtype)
            nvtx.range_pop()
            self.current_seq_len[layer_idx] = actual_len
            if is_last_layer and is_last_chunk:
                # context 跟踪单个样本的位置（用于 position_ids），不是 batch * seq_len
                self.context += actual_len
            return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

        # Step1: 分区存储 —— Sink + Local → GPU，Retrieval → CPU（异步 D2H）
        local_len = local_end - local_start
        nvtx.range_push("prefill_gpu_copy")
        sink_k = key_states[:src_bsz, sink_start:sink_end, :, :].transpose(1, 2).to(self.dtype)
        sink_v = value_states[:src_bsz, sink_start:sink_end, :, :].transpose(1, 2).to(self.dtype)
        self.unified_keys_gpu[layer_idx][b_sl, :, :self.sink_size, :] = sink_k
        self.unified_values_gpu[layer_idx][b_sl, :, :self.sink_size, :] = sink_v

        local_k = key_states[:src_bsz, local_start:local_end, :, :].transpose(1, 2).to(self.dtype)
        local_v = value_states[:src_bsz, local_start:local_end, :, :].transpose(1, 2).to(self.dtype)
        self.unified_keys_gpu[layer_idx][b_sl, :, self.sink_size:self.sink_size+local_len, :] = local_k
        self.unified_values_gpu[layer_idx][b_sl, :, self.sink_size:self.sink_size+local_len, :] = local_v
        nvtx.range_pop()

        # Keep several copies in flight. Drop the ones that already finished
        # without blocking. Only the safety cap waits, and prefill's next
        # layer is long enough that the cap is not hit.
        self._retire_offload_layers(keep=8)
        temp_retrieval_keys = None
        if retrieval_length > 0:
            temp_retrieval_keys = key_states[:src_bsz, retrieval_start:retrieval_end, :, :].transpose(1, 2).to(self.dtype).contiguous()
            temp_retrieval_values = value_states[:src_bsz, retrieval_start:retrieval_end, :, :].transpose(1, 2).to(self.dtype).contiguous()
            q_keys, scale_keys = self._quantize_kv_blocks(temp_retrieval_keys)
            q_values, scale_values = self._quantize_kv_blocks(temp_retrieval_values)
            del temp_retrieval_values
            self._inflight_offload_buffers[layer_idx] = (q_keys, q_values)
            self._layer_offload_inflight[layer_idx] = True
            self._async_d2h_retrieval(
                layer_idx,
                q_keys,
                q_values,
                self.unified_keys_cpu[layer_idx][b_sl],
                self.unified_values_cpu[layer_idx][b_sl],
            )
            self._write_block_scales(layer_idx, b_sl, 0, scale_keys, scale_values)
        self.cpu_valid_end[layer_idx] = retrieval_length
        # Positions [start, end) were copied to pinned CPU memory. The SGLang
        # pool slots for that span can be freed once every layer has copied.
        self._retrieval_gpu_span = (int(retrieval_start), int(retrieval_end))
        nvtx.range_pop()

        self.current_seq_len[layer_idx] = seq_len - valid_start
        if layer_idx == 0:
            self._graph_local_start_host = int(local_start)
            if not self._graph_window_announced:
                self._graph_window_announced = True
                print(
                    f"[ZoomKV] graph local window starts at {int(local_start)}",
                    flush=True,
                )

        # Step2: KIVI per-(quest-)chunk per-channel 4-bit encode（折叠进 chunk min/max）
        # Reuse the tensor already transposed for the copy. A second transpose
        # of the same retrieval keys was pure extra traffic.
        nvtx.range_push("prefill_build_index")
        if temp_retrieval_keys is not None:
            self._compute_and_store_chunk_minmax(
                layer_idx, temp_retrieval_keys, abs_token_start=0, batch_slice=b_sl,
            )
            del temp_retrieval_keys
        nvtx.range_pop()

        if is_last_layer and is_last_chunk:
            # context 跟踪单个样本的位置（用于 position_ids），不是 batch * seq_len
            self.context += self.current_seq_len[layer_idx]
            # The copy stream keeps running. The first eager token reads the
            # pool, and stage_for_replay waits before a graph reads CPU memory.
            # Index scoring cannot. Move this request's Quest index onto the
            # GPU before decode capture, once for the whole generation.
            self._stage_offload_index_to_gpu()

        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]


    def merge_cohort(self, caches: list) -> "ZoomKVCache":
        """Stack one-request prefills into this cache, in decode-batch order.

        Each prefill owns its own sidecar. Decode of the whole cohort needs
        one batch dimension, so the retrieval KV and Quest index are
        concatenated here and the per-request caches are released.
        """
        if not caches or caches[0] is not self:
            raise RuntimeError("ZoomKV cohort merge must start from the first cache")
        if len(caches) == 1:
            return self
        if not self.enable_offload:
            raise RuntimeError("ZoomKV cohort merge is for the offload cache")
        ref = caches[0]
        for cache in caches[1:]:
            if int(cache.unified_kv_capacity) != int(ref.unified_kv_capacity):
                raise RuntimeError(
                    "ZoomKV cannot decode these requests together: "
                    "their cache capacities differ"
                )
            if int(cache.kv_head) != int(ref.kv_head) or int(cache.layer_num) != int(ref.layer_num):
                raise RuntimeError("ZoomKV cohort caches do not match this GPU shard")
        same_state = (
            "current_seq_len",
            "cpu_valid_end",
            "_chunk_valid_end",
            "_chunk_valid_end_large",
            "_chunk_skipped_count",
            "_chunk_carry_count",
            "_actual_retrieval_length",
            "_retrieval_gpu_span",
            "context",
            "_graph_local_start_host",
            "_eff_local_size",
        )
        for name in same_state:
            baseline = getattr(ref, name)
            for cache in caches[1:]:
                other = getattr(cache, name)
                if isinstance(baseline, list):
                    matched = list(other) == list(baseline)
                else:
                    matched = other == baseline
                if not matched:
                    raise RuntimeError(
                        "ZoomKV cannot decode these requests together: "
                        f"{name} differs across prefills"
                    )
        for cache in caches:
            cache._sync_offload_copies()

        batch_size = len(caches)
        self.batch_size = batch_size
        self.batch_groups = batch_size * self.kv_head
        self._log_host_rss(f"merge start bs={batch_size}")
        self._cat_cuda_layers(caches, "unified_keys_gpu")
        self._cat_cuda_layers(caches, "unified_values_gpu")
        # Allocate the merged pinned KV once, copy one parked request at a time,
        # and drop that request's buffers before touching the next. Both ranks
        # then barrier so neither starts the index merge while the other still
        # holds a doubled KV working set (that spike was 141→177 GiB).
        self._merge_pinned_attr(caches, "unified_keys_cpu")
        self._merge_pinned_attr(caches, "unified_values_cpu")
        self._reclaim_host_memory()
        self._dist_host_barrier("after merged retrieval KV")
        self._log_host_rss("after merged retrieval KV")
        self._cat_cuda_layers(caches, "_k_scale")
        self._cat_cuda_layers(caches, "_v_scale")
        self._cat_cuda_layers(caches, "_chunk_carry_keys")
        self._cat_cuda_layers(caches, "_chunk_skipped_indices_gpu")
        for cache in caches:
            cache._index_cpu_keepalive = []
        hosts = []
        for name in (
            "key_kivi_packed_gpu",
            "_chunk_key_min_gpu",
            "_chunk_key_max_gpu",
            "_chunk_key_centroid_gpu",
            "_chunk_valid_mask_gpu",
            "_chunk_key_min_gpu_large",
            "_chunk_key_max_gpu_large",
            "_chunk_valid_mask_gpu_large",
        ):
            hosts.extend(self._merge_index_attr(caches, name))
            self._reclaim_host_memory()
            self._log_host_rss(f"after merged index {name}")
        self._index_cpu_keepalive = hosts
        self._offload_index_on_gpu = False
        self._index_stage_decided = False
        self._allow_index_stage = False
        self._realloc_batch_workspace(batch_size)
        self._quest_graphs = {}
        self._sparse_graphs = {}
        self._quest_graph_broken = False
        self._sparse_graph_broken = False
        self._workspace_frozen = False
        self._graph_avg_q = None
        self._pool_row_ids = None
        self._sparse_out = None
        if self._actual_retrieval_length:
            self._refresh_quest_chunk_cache(int(self._actual_retrieval_length))
        self._try_enable_uva_gather()
        for cache in caches[1:]:
            cache._index_cpu_keepalive = []
            cache._index_host_of = {}
            cache.unified_keys_cpu = []
            cache.unified_values_cpu = []
            cache._k_scale = []
            cache._v_scale = []
        self._reclaim_host_memory()
        self._dist_host_barrier("merge done")
        self._log_host_rss(f"merge done bs={batch_size}")
        return self

    def _cat_cuda_layers(self, caches: list, name: str) -> None:
        merged = []
        for layer in range(self.layer_num):
            parts = [getattr(cache, name)[layer] for cache in caches]
            merged.append(torch.cat(parts, dim=0).contiguous())
            for cache in caches:
                getattr(cache, name)[layer] = None
            del parts
        setattr(self, name, merged)

    def _merge_pinned_attr(self, caches: list, name: str) -> None:
        """Merge pinned layer lists with one destination allocation."""
        n_layers = self.layer_num
        merged = []
        for layer in range(n_layers):
            ref = getattr(caches[0], name)[layer]
            shape = list(ref.shape)
            shape[0] = sum(int(getattr(cache, name)[layer].shape[0]) for cache in caches)
            merged.append(torch.empty(shape, dtype=ref.dtype, pin_memory=True))
        for row, cache in enumerate(caches):
            layers = getattr(cache, name)
            for layer in range(n_layers):
                src = layers[layer]
                rows = int(src.shape[0])
                merged[layer][row:row + rows].copy_(src)
                layers[layer] = None
            setattr(cache, name, [])
            self._reclaim_host_memory()
        setattr(self, name, merged)

    def _merge_index_attr(self, caches: list, name: str) -> list:
        """Merge one Quest index field. Copy one parked request, then free it."""
        from sglang_zoomkv.kernels.gather_trans.trans_h2d import uva_alias

        n_layers = self.layer_num
        per_cache_hosts = []
        for cache in caches:
            layer_hosts = []
            for layer in range(n_layers):
                alias = getattr(cache, name)[layer]
                host = cache._index_host_of.pop(id(alias), None)
                if host is None:
                    raise RuntimeError(
                        f"ZoomKV index {name} layer {layer} has no pinned host copy"
                    )
                try:
                    cache._index_cpu_keepalive.remove(host)
                except ValueError:
                    pass
                layer_hosts.append(host)
                getattr(cache, name)[layer] = None
            per_cache_hosts.append(layer_hosts)

        merged_hosts = []
        for layer in range(n_layers):
            ref = per_cache_hosts[0][layer]
            shape = list(ref.shape)
            shape[0] = sum(
                int(per_cache_hosts[i][layer].shape[0]) for i in range(len(caches))
            )
            merged_hosts.append(torch.empty(shape, dtype=ref.dtype, pin_memory=True))

        for row, layer_hosts in enumerate(per_cache_hosts):
            for layer in range(n_layers):
                src = layer_hosts[layer]
                rows = int(src.shape[0])
                merged_hosts[layer][row:row + rows].copy_(src)
                layer_hosts[layer] = None
            self._reclaim_host_memory()

        aliases = []
        for host in merged_hosts:
            alias = uva_alias(host, self.device)
            self._index_host_of[id(alias)] = host
            aliases.append(alias)
        setattr(self, name, aliases)
        return merged_hosts

    @staticmethod
    def _reclaim_host_memory() -> None:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        host_empty = getattr(torch.cuda, "host_empty_cache", None)
        if host_empty is None:
            host_empty = getattr(getattr(torch.cuda, "memory", None), "_host_emptyCache", None)
        if callable(host_empty):
            host_empty()

    def _dist_host_barrier(self, tag: str) -> None:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                print(f"[ZoomKV] host barrier ok ({tag})", flush=True)
        except Exception as exc:
            print(f"[ZoomKV] host barrier skipped ({tag}): {exc}", flush=True)

    @staticmethod
    def _as_pinned(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_pinned():
            return tensor
        pinned = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        pinned.copy_(tensor)
        return pinned

    def _realloc_batch_workspace(self, batch_size: int) -> None:
        max_topk = int(self._topk_keys_buffer.size(2))
        head_dim = self.head_dim
        self._topk_keys_buffer = torch.empty(
            (batch_size, self.kv_head, max_topk, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._topk_values_buffer = torch.empty(
            (batch_size, self.kv_head, max_topk, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._chunk_merge_buffer = [
            torch.empty(
                (batch_size, self.kv_head, self.quest_chunk_size, head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self._offload_keys_buffer = [
            torch.empty(
                (batch_size, self.kv_head, self.dynamic_update_interval, head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self._offload_values_buffer = [
            torch.empty(
                (batch_size, self.kv_head, self.dynamic_update_interval, head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        if self._topk_keys_cpu_buffer is not None:
            self._topk_keys_cpu_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk, head_dim),
                dtype=self.dtype,
                pin_memory=True,
            )
            self._topk_values_cpu_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk, head_dim),
                dtype=self.dtype,
                pin_memory=True,
            )
            self._topk_indices_cpu_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk),
                dtype=torch.int64,
                pin_memory=True,
            )
        self._quest_scores_buffer_small = torch.empty(
            batch_size * self.kv_head * self._chunk_n_max,
            dtype=torch.float32,
            device=self.device,
        )
        self._quest_scores_buffer_large = torch.empty(
            batch_size * self.kv_head * self._chunk_n_max_large,
            dtype=torch.float32,
            device=self.device,
        )
        self._quest_scores_buffer_sub = torch.empty(
            batch_size * self.kv_head * self._chunk_n_max_large * self._hq_factor,
            dtype=torch.float32,
            device=self.device,
        )

    def stage_index_for_decode(self) -> None:
        """Move the Quest index to GPU once, after the decode batch is known."""
        if not self.enable_offload or self._offload_index_on_gpu:
            return
        if getattr(self, "_index_stage_decided", False):
            return
        self._index_stage_decided = True
        self._allow_index_stage = True
        self._stage_offload_index_to_gpu()

    def _defer_index_stage(self) -> bool:
        return os.environ.get("SGLANG_ZOOMKV_PREFILL_ALL_FIRST", "1") not in (
            "0",
            "false",
            "False",
            "",
        )

    def _stage_offload_index_to_gpu(self) -> None:
        """Copy this request's Quest index from pinned CPU onto the GPU once.

        Prefill writes the index through the UVA alias so other requests can
        keep theirs in host memory. Score and KIVI then read the GPU copy for
        every decode token. The copy is outside the CUDA graph.
        """
        if not self.enable_offload or self._offload_index_on_gpu:
            return
        if self._defer_index_stage() and not getattr(self, "_allow_index_stage", False):
            return
        nbytes = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self._index_cpu_keepalive
        )
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        # After KV-pool compact, bs=6 still needs ~2 GiB for the decode graph
        # and activations. 8 GiB was so large the 17 GiB index never staged.
        reserve = 2 * 1024**3
        if nbytes > 0 and free_bytes < nbytes + reserve:
            print(
                "[ZoomKV] Quest index stays in pinned CPU memory for this decode batch "
                f"({nbytes / (1024 ** 3):.1f} GB index, "
                f"{free_bytes / (1024 ** 3):.1f} GB free, "
                f"bs={int(self.batch_size)})",
                flush=True,
            )
            return
        names = (
            "key_kivi_packed_gpu",
            "_chunk_key_min_gpu",
            "_chunk_key_max_gpu",
            "_chunk_key_centroid_gpu",
            "_chunk_valid_mask_gpu",
            "_chunk_key_min_gpu_large",
            "_chunk_key_max_gpu_large",
            "_chunk_valid_mask_gpu_large",
        )
        # Order the copies after the prefill kernels that filled the alias.
        torch.cuda.current_stream(self.device).synchronize()
        for name in names:
            staged = []
            for tensor in getattr(self, name):
                gpu = torch.empty(
                    tensor.shape, dtype=tensor.dtype, device=self.device
                )
                gpu.copy_(tensor)
                staged.append(gpu)
            setattr(self, name, staged)
        self._offload_index_on_gpu = True
        if self._actual_retrieval_length:
            self._refresh_quest_chunk_cache(int(self._actual_retrieval_length))
        torch.cuda.current_stream(self.device).synchronize()
        print(
            "[ZoomKV] offload Quest index staged to GPU for decode",
            flush=True,
        )

    def _try_enable_uva_gather(self) -> None:
        """Gather retrieval KV with a GPU kernel. Fall back if UVA is unavailable."""
        if not self.unified_keys_cpu or self.cpu_kv_capacity <= 0:
            return
        try:
            from sglang_zoomkv.kernels.gather_trans.trans_h2d import h2d_gather_kv_i4

            probe = torch.zeros(
                self.batch_size, self.kv_head, 1, dtype=torch.int64, device=self.device
            )
            # Resolve every layer's pinned pointer now. cudaHostGetDevicePointer
            # is not capturable, and the decode graph launches one gather per layer.
            for layer in range(self.layer_num):
                h2d_gather_kv_i4(
                    self.unified_keys_cpu[layer],
                    self.unified_values_cpu[layer],
                    self._k_scale[layer],
                    self._v_scale[layer],
                    probe,
                    self._topk_keys_buffer,
                    self._topk_values_buffer,
                )
            torch.cuda.current_stream(self.device).synchronize()
        except Exception as exc:
            print(
                f"[ZoomKV] GPU gather from pinned CPU KV unavailable ({exc}); "
                "using staged copies",
                flush=True,
            )
            self._uva_gather_kv = None
            return
        self._uva_gather_kv = h2d_gather_kv_i4
        print(
            "[ZoomKV] offload gather reads packed int4 CPU KV from the GPU",
            flush=True,
        )

    def _log_host_rss(self, tag: str) -> None:
        try:
            rss_kb = 0
            with open("/proc/self/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
            with open("/sys/fs/cgroup/memory.current", "r", encoding="utf-8") as handle:
                current = int(handle.read().strip())
            print(
                f"[ZoomKV] {tag}: rss={rss_kb / (1024 ** 2):.2f} GiB, "
                f"cgroup={current / (1024 ** 3):.2f} GiB",
                flush=True,
            )
        except Exception:
            return

    def _quantize_kv_blocks(self, src: torch.Tensor):
        """Per-head, per-128-token packed int4. src is [batch, head, tokens, 128]."""
        block = self._cpu_kv_block
        batches, heads, length, dim = src.shape
        if int(length) <= 0:
            return (
                torch.empty((batches, heads, 0, dim // 2), dtype=torch.uint8, device=src.device),
                torch.empty((batches, heads, 0), dtype=torch.float32, device=src.device),
            )
        n_blocks = (int(length) + block - 1) // block
        pad = n_blocks * block - int(length)
        work = src if pad == 0 else torch.nn.functional.pad(src, (0, 0, 0, pad))
        view = work.reshape(batches, heads, n_blocks, block, dim).float()
        scale = view.abs().amax(dim=(3, 4)).clamp_min(1e-8) / 7.0
        codes = torch.round(view / scale[:, :, :, None, None]).clamp(-8, 7).to(torch.int16)
        codes = codes.reshape(batches, heads, n_blocks * block, dim)[:, :, :length]
        even = codes[..., 0::2] & 0xF
        odd = codes[..., 1::2] & 0xF
        packed = (even | (odd << 4)).to(torch.uint8).contiguous()
        return packed, scale.contiguous()

    def _write_block_scales(self, layer_idx, batch_slice, start, scale_k, scale_v) -> None:
        block = self._cpu_kv_block
        if int(start) % block != 0:
            raise RuntimeError(
                f"ZoomKV int4 scales need a {block}-token aligned write, got start={start}"
            )
        block0 = int(start) // block
        n_blocks = int(scale_k.shape[-1])
        self._k_scale[layer_idx][batch_slice, :, block0:block0 + n_blocks].copy_(scale_k)
        self._v_scale[layer_idx][batch_slice, :, block0:block0 + n_blocks].copy_(scale_v)

    def _gather_offload_kv(self, layer_idx, indices, dst_k, dst_v) -> None:
        from sglang_zoomkv.kernels.gather_trans.trans_h2d import h2d_gather_kv_i4

        h2d_gather_kv_i4(
            self.unified_keys_cpu[layer_idx],
            self.unified_values_cpu[layer_idx],
            self._k_scale[layer_idx],
            self._v_scale[layer_idx],
            indices,
            dst_k,
            dst_v,
        )

    def _dequant_cpu_span(self, layer_idx, length: int, kind: str) -> torch.Tensor:
        cpu = self.unified_keys_cpu[layer_idx] if kind == "k" else self.unified_values_cpu[layer_idx]
        scale = self._k_scale[layer_idx] if kind == "k" else self._v_scale[layer_idx]
        block = self._cpu_kv_block
        packed = cpu[:, :, :length, :].to(self.device, non_blocking=True)
        low = (packed & 0xF).to(torch.int16)
        high = ((packed >> 4) & 0xF).to(torch.int16)
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        taken = torch.stack((low, high), dim=-1).reshape(
            packed.shape[0], packed.shape[1], length, self.head_dim
        ).float()
        token_blocks = torch.arange(length, device=self.device) // block
        token_scale = scale[:, :, token_blocks]
        return (taken * token_scale.unsqueeze(-1)).to(self.dtype)

    def _fallback_gather_i8(self, layer_idx, indices, dst_k, dst_v) -> None:
        count = int(indices.shape[-1])
        length = max(int(self.cpu_valid_end[layer_idx]), 1)
        safe = indices.clamp(0, length - 1)
        expand = safe.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        for kind, dst in (("k", dst_k), ("v", dst_v)):
            full = self._dequant_cpu_span(layer_idx, length, kind)
            picked = torch.gather(full, 2, expand)
            dst[:, :, :count, :].copy_(picked)

    def _sync_offload_copies(self):
        """等待所有异步 offload 拷贝完成"""
        if not self.enable_offload:
            return
        # 等待 copy stream 上的所有操作完成
        self._copy_stream.synchronize()
        # 清理临时引用
        for ldx in range(self.layer_num):
            self._inflight_offload_buffers[ldx] = None
            self._layer_offload_inflight[ldx] = False



    def _async_d2h_retrieval(self, layer_idx, src_k, src_v, dst_k, dst_v) -> None:
        """Queue one contiguous D2H per head. The CPU call returns immediately.

        A non-contiguous ``copy_`` into ``[:, :, :length]`` falls back to a
        blocking memcpy, which is the long cudaMemcpyAsync on the timeline.
        Each head prefix of a contiguous ``[B, H, S, D]`` buffer is itself
        contiguous, so these are real async copies on the copy stream.
        """
        self._main_events[layer_idx].record()
        batches, heads, length, _ = src_k.shape
        with torch.cuda.stream(self._copy_stream):
            self._main_events[layer_idx].wait()
            for batch in range(int(batches)):
                for head in range(int(heads)):
                    dst_k[batch, head, :length].copy_(src_k[batch, head], non_blocking=True)
                    dst_v[batch, head, :length].copy_(src_v[batch, head], non_blocking=True)
            self._copy_events[layer_idx].record()

    def _retire_offload_layers(self, keep: int) -> None:
        """Release finished copies. Block only when too many are still running."""
        inflight = [i for i, busy in enumerate(self._layer_offload_inflight) if busy]
        still = []
        for layer_idx in inflight:
            if self._copy_events[layer_idx].query():
                self._inflight_offload_buffers[layer_idx] = None
                self._layer_offload_inflight[layer_idx] = False
            else:
                still.append(layer_idx)
        while len(still) > int(keep):
            self._sync_layer_offload(still.pop(0))

    def _sync_layer_offload(self, layer_idx: int):
        """只等待指定层的异步 offload 完成（PQCache 风格：按层精确等待）。"""
        if not self.enable_offload:
            return
        if not self._layer_offload_inflight[layer_idx]:
            return
        self._copy_events[layer_idx].synchronize()
        self._inflight_offload_buffers[layer_idx] = None
        self._layer_offload_inflight[layer_idx] = False


    def get_region_indices(self, layer_idx, device=None):
        """
        动态计算各区域索引（统一 KV cache 架构）
        Args:
            layer_idx: 当前层索引
            device: 目标设备（保留以兼容旧 caller，本实现忽略：所有索引来自
                    pre-allocated ``self._arange_buffer``，已经在 ``self.device``）
        Returns:
            dict: {
                'sink_indices': Tensor - Sink 区域索引   (zero-copy view)
                'local_indices': Tensor - Local + UpdateBuffer 区域索引 (zero-copy view)
                'retrieval_start': int - Retrieval zone 起始 (= sink_size)
                'retrieval_length': int - Retrieval zone 长度
                'total_len': int - 当前总序列长度
                'local_len': int - Local + UpdateBuffer 长度
            }

        Bubble killer (vs. previous implementation):
          * Both ``sink_indices`` and ``local_indices`` are now slices of the
            pre-allocated ``self._arange_buffer``: zero kernel launches and
            zero allocator pressure (was 2× ``torch.arange`` per call =
            ~10us host launch + same on device).
          * Per-step memoisation: all 36 layers in a decode step see the
            same ``current_seq_len`` (update_kv_cache increments AFTER
            compute()).  Layer 0 builds the dict, layers 1..35 reuse it
            via the ``_cached_region_total_len`` fast path.
        """
        del device  # only the pre-allocated buffer's device matters

        total_len = self.current_seq_len[layer_idx]

        # Per-step fast path: same total_len → same region info across the
        # 36 layers within one decode step.  Saves ~2us python work per
        # layer × 35 = ~70us per token.
        if total_len == self._cached_region_total_len and self._cached_region_info is not None:
            return self._cached_region_info

        # 计算 update buffer 大小（区分长短 prefill）
        update_buffer_size = self._compute_update_buffer_size(total_len)

        # 计算 local_start（retrieval_end = local_start）
        # _eff_local_size = local_size + R0：aligned-offload 把不足一个 large_chunk 的余数
        # 留在 GPU near-zone，所以这里的窗口用 eff 而非 local_size，保证
        # local_start == sink + cpu_valid_end（GPU near-zone 紧接 CPU retrieval 尾部）。
        local_end = max(total_len - update_buffer_size, self.sink_size)
        local_start = max(self.sink_size, local_end - self._eff_local_size)
        retrieval_length = max(0, local_start - self.sink_size) # 是"理论值"（滑动窗口认为应该有多少），不是"实际值"（真正 offload 了多少）。

        # Zero-launch index views — slicing a contiguous arange returns a
        # view; downstream callers (``unsqueeze().expand()``) accept views.
        sink_indices = self._arange_buffer[:self.sink_size]
        local_len = total_len - local_start if local_start < total_len else 0
        if local_len > 0:
            local_indices = self._arange_buffer[local_start:total_len]
        else:
            # zero-length view (no allocation)
            local_indices = self._arange_buffer[:0]

        region_info = {
            'sink_indices': sink_indices,
            'local_indices': local_indices,
            'retrieval_start': self.sink_size,
            'retrieval_length': retrieval_length,
            'total_len': total_len,
            'local_len': local_len,
            'local_start': int(local_start),
            'attn_local_end': int(total_len if local_len > 0 else local_start),
        }
        self._cached_region_total_len = total_len
        self._cached_region_info = region_info
        return region_info


    def _compute_update_buffer_size(self, total_len: int) -> int:
        """Update-buffer 内 token 数（区分长/短 prefill），供区域划分逻辑共享。

        长 prefill (input_length >= steady_size)：buffer = (decode 累计) % dui。
        短 prefill：buffer = (total_len - steady_size) % dui。
        steady 之前：无 buffer。
        """
        steady_threshold = self.steady_size  # sink_size + local_size
        if self.input_length >= steady_threshold:
            return max(0, total_len - self.input_length) % self.dynamic_update_interval
        if total_len >= steady_threshold:
            return (total_len - steady_threshold) % self.dynamic_update_interval
        return 0

    def _compute_local_start(self, layer_idx=0):
        """
        轻量版：只计算 local_start，不构建 tensor（用于 _update_polar_index）
        """
        total_len = self.current_seq_len[layer_idx]
        update_buffer_size = self._compute_update_buffer_size(total_len)
        local_end = max(total_len - update_buffer_size, self.sink_size)
        return max(self.sink_size, local_end - self._eff_local_size)


    def bind_sglang_kv(self, kv_pool, slot_table, row_ids):
        """Point GPU-only attention at the SGLang pool and its request slot map."""
        self._sglang_kv_pool = kv_pool
        self._slot_table = slot_table
        n = int(row_ids.shape[0])
        self._pool_row_n = n
        if (
            self._pool_row_ids is None
            or self._pool_row_ids.device != slot_table.device
            or int(self._pool_row_ids.shape[0]) < n
        ):
            self._pool_row_ids = torch.empty(
                int(self.batch_size), dtype=torch.int32, device=slot_table.device
            )
        src = row_ids if row_ids.dtype == torch.int32 else row_ids.to(dtype=torch.int32)
        self._pool_row_ids[:n].copy_(src[:n], non_blocking=True)

    def graph_spec_ready(self) -> bool:
        """True once prefill has frozen Quest sizes and the local-window start."""
        quest_ready = (
            self._sparse_ext is not None
            and self._graph_local_start_host is not None
            and int(getattr(self, "_cached_hq_n_large", 0) or 0) > 0
        )
        if not quest_ready:
            return False
        if self.enable_offload:
            return self._uva_gather_kv is not None and bool(self.unified_keys_cpu)
        return self._use_pool_kv

    def stage_for_replay(self, kv_pool, slot_table, row_ids, seq_len: int) -> None:
        """Refresh the device buffers a captured decode graph reads.

        Called outside the graph, immediately before capture and before every
        replay. Kernel arguments keep these pointers; only the values change.
        """
        hq = int(self.num_heads)
        self._ensure_sparse_workspace(int(self.batch_size), hq)
        if (
            self._graph_avg_q is None
            or self._graph_avg_q.size(0) < int(self.batch_size)
            or self._graph_avg_q.size(1) != self.kv_head
        ):
            if self._workspace_frozen:
                raise RuntimeError(
                    "ZoomKV query workspace cannot be resized after CUDA graph capture"
                )
            self._graph_avg_q = torch.empty(
                int(self.batch_size),
                self.kv_head,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
        self.bind_sglang_kv(kv_pool, slot_table, row_ids)
        cpu = self._sparse_bounds_cpu
        cpu[0] = int(self._graph_local_start_host)
        cpu[1] = int(seq_len)
        self._sparse_bounds_cached = (int(cpu[0]), int(cpu[1]))
        self._sparse_bounds.copy_(cpu, non_blocking=True)
        # Replay never returns to Python, so keep the host length in step with
        # the seq len just staged. Capture warms up at length 1 and must not
        # rewind a real prefix.
        if int(seq_len) > int(self.current_seq_len[0]):
            self.current_seq_len[0] = int(seq_len)
        if self.enable_offload:
            if any(self._layer_offload_inflight):
                self._sync_offload_copies()
            self._stage_offload_extra_counts()
            self._empty_sparse_indices(int(self.batch_size))

    def _offload_gathered_count(self, layer_idx: int) -> int:
        """How many retrieval tokens this layer's graph gather writes."""
        chunk = int(self.quest_chunk_size)
        nk_small = int(getattr(self, "_cached_hq_nk_small", 0) or 0)
        actual_topk = min(int(self.final_topk), nk_small * chunk)
        return actual_topk + int(self._chunk_skipped_count[layer_idx])

    def _stage_offload_extra_counts(self) -> None:
        """Device per-layer gather lengths. The captured kernel only reads them."""
        n_layers = int(self.layer_num)
        if (
            self._offload_extra_counts is None
            or int(self._offload_extra_counts.numel()) != n_layers
        ):
            if self._workspace_frozen or torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "ZoomKV offload gather counts cannot be allocated after CUDA graph capture"
                )
            self._offload_extra_counts = torch.empty(
                n_layers, dtype=torch.int32, device=self.device
            )
            self._offload_extra_counts_cpu = torch.empty(
                n_layers, dtype=torch.int32, pin_memory=True
            )
        cpu = self._offload_extra_counts_cpu
        for layer in range(n_layers):
            cpu[layer] = self._offload_gathered_count(layer)
        self._offload_extra_counts.copy_(cpu, non_blocking=True)

    def attention_inside_graph(self, queries, layer_idx: int):
        """Quest plus sparse attention with no nested graph and no host sync.

        Offload keeps retrieval on CPU: the gather kernel reads pinned memory
        inside this graph. Sink, the local window, and the new token are read
        from the paged pool, which already holds them.
        """
        from sglang_zoomkv.kernels.quest.quest_pipeline import run_hierarchical_quest

        if self._graph_avg_q is None or self._sglang_kv_pool is None:
            raise RuntimeError("ZoomKV graph attention was not staged")
        bs = int(queries.shape[0])
        full_q = queries[:bs, 0]
        avg_q = self._graph_avg_q[:bs]
        self._fill_avg_query(full_q, avg_q)
        n_valid = int(self._cached_quest_n_chunks)
        packed = self.key_kivi_packed_gpu[layer_idx][:, :, :n_valid]
        spec = self._quest_graph_inputs(avg_q, packed, layer_idx, bs)
        if spec is None:
            raise RuntimeError(
                "ZoomKV Quest spec is not ready to be captured into the decode graph"
            )
        keep = None
        if torch.cuda.is_current_stream_capturing():
            keep = []
            self._graph_keepalive.append(keep)
        kwargs = dict(spec)
        kwargs["raw_q"] = avg_q
        kwargs["quest_kernel"] = self._quest_kernel
        kwargs["rerank_kernel"] = self._rerank_topk_kernel
        kwargs["kivi_kernel"] = self._kivi_kernel
        kwargs["keepalive"] = keep
        indices = run_hierarchical_quest(**kwargs)
        k = self._sglang_kv_pool.get_key_buffer(layer_idx)
        v = self._sglang_kv_pool.get_value_buffer(layer_idx)
        if self.enable_offload:
            # Quest returns a prefix of a wider buffer. The gather indexes it as
            # packed [batch, head, topk], so the copy has to be recorded too.
            if not indices.is_contiguous():
                indices = indices.contiguous()
                if keep is not None:
                    keep.append(indices)
            from sglang_zoomkv.kernels.nvtx_ranges import UVA_READ, nvtx_stage
            with nvtx_stage(UVA_READ):
                self._gather_offload_kv(
                    layer_idx,
                    indices,
                    self._topk_keys_buffer,
                    self._topk_values_buffer,
                )
            self._launch_sparse_attn(
                full_q,
                k,
                v,
                self._empty_sparse_indices(bs),
                extra_k=self._topk_keys_buffer,
                extra_v=self._topk_values_buffer,
                extra_count=self._offload_extra_counts[layer_idx:layer_idx + 1],
                paged=True,
            )
        else:
            self._launch_sparse_attn(full_q, k, v, indices)
        return self._sparse_out[:bs].unsqueeze(1)

    def _gather_pool_positions(self, layer_idx, positions):
        """Gather sequence positions from the paged pool into ``[B, H, n, D]``."""
        rows = self._pool_row_ids[: self._pool_row_n].long()
        slots = self._slot_table[rows.unsqueeze(1), positions.long().unsqueeze(0)]
        keys = self._sglang_kv_pool.get_key_buffer(layer_idx)[slots.long()]
        values = self._sglang_kv_pool.get_value_buffer(layer_idx)[slots.long()]
        return keys.transpose(1, 2).contiguous(), values.transpose(1, 2).contiguous()

    def _gather_pool_sequence(self, layer_idx, length):
        positions = torch.arange(int(length), device=self.device)
        return self._gather_pool_positions(layer_idx, positions)

    def _get_full_kv_for_attention(self, layer_idx, sink_indices, local_indices, 
                                    retrieval_start, retrieval_length, device, dtype):
        """
        获取完整的 KV 用于 attention 计算（Sink + Retrieval + Local）。

        Args:
            layer_idx: 层索引
            sink_indices: Sink 区域的全局索引
            local_indices: Local+Buffer 区域的全局索引
            retrieval_start / dtype: 保留以兼容旧 caller（offload 路径不再使用）
            retrieval_length: Retrieval 区域长度（理论值）
            device: 目标设备
        Returns:
            (concat_keys, concat_values): 拼接后的 KV，形状 [bs, kv_head, total_len, head_dim]
        """
        if not self.enable_offload:
            total_len = self.current_seq_len[layer_idx]
            return self._gather_pool_sequence(layer_idx, total_len)

        local_len = local_indices.shape[0]

        if retrieval_length > 0:
            self._sync_layer_offload(layer_idx)

            # Sink+Local 在 GPU 上（先提取出来，保证变量作用域）
            sink_local_end = self.sink_size + local_len
            sink_local_keys = self.unified_keys_gpu[layer_idx][:, :, :sink_local_end, :]
            sink_local_values = self.unified_values_gpu[layer_idx][:, :, :sink_local_end, :]

            cpu_retrieval_len = self.cpu_valid_end[layer_idx]
            if cpu_retrieval_len > 0:
                retrieval_keys = self._dequant_cpu_span(layer_idx, cpu_retrieval_len, "k")
                retrieval_values = self._dequant_cpu_span(layer_idx, cpu_retrieval_len, "v")
                concat_keys = torch.cat([sink_local_keys, retrieval_keys], dim=2)
                concat_values = torch.cat([sink_local_values, retrieval_values], dim=2)
            else:
                # CPU 没有 offload 数据，所有 token 仍在 GPU，直接用 Sink+Local
                concat_keys = sink_local_keys
                concat_values = sink_local_values
        else:
            # 没有 retrieval zone，只用 Sink + Local（仍在 GPU）
            all_indices = torch.cat([sink_indices, local_indices]) if local_len > 0 else sink_indices
            concat_keys = torch.index_select(self.unified_keys_gpu[layer_idx], 2, all_indices)
            concat_values = torch.index_select(self.unified_values_gpu[layer_idx], 2, all_indices)

        return concat_keys, concat_values


    def _update_polar_index(self):
        """
        动态更新:对Steady Zone中旧的Local部分进行Polar编码并更新倒排索引(GPU版本)
        逻辑:
        1. 提取旧的Local tokens [4:68] → 编码 → 追加到统一KV缓存
        2. 把新生成的64个tokens [68:132] 移到Local位置 [4:68]
        3. Steady Zone重置为 [Sink(4) + Local(64)] = 68
        """
        # 计算需要编码的范围：[old_retrieval_end : new_retrieval_end]
        # 关键：这里用的是 _update_polar_index 被调用时的 current_seq_len[0]
        # 这个值是在写入当前 token 之前的值！
        new_retrieval_end = self._compute_local_start(0)
        old_retrieval_end = self._actual_retrieval_length + self.sink_size
        encode_start = old_retrieval_end
        encode_end = new_retrieval_end
        num_new_tokens = encode_end - encode_start
        
        # 边界检查（触发时序已修复，num_new_tokens 应该总是 > 0）
        assert num_new_tokens > 0, f"num_new_tokens={num_new_tokens} <= 0, 触发逻辑有 bug"
        assert encode_start >= self.sink_size, f"encode_start ({encode_start}) < sink_size ({self.sink_size})"
        assert encode_end <= self.current_seq_len[0], f"encode_end ({encode_end}) > current_seq_len ({self.current_seq_len[0]})"

        # 更新 _actual_retrieval_length（编码到的相对长度）
        # Retrieval Zone = [sink_size : encode_end]
        self._actual_retrieval_length = encode_end - self.sink_size
        self._refresh_quest_chunk_cache(self._actual_retrieval_length)

        if not self.enable_offload:
            nvtx.range_push("index_encode")
            positions = torch.arange(encode_start, encode_end, device=self.device)
            for ldx in range(self.layer_num):
                new_retrieval_keys, _values = self._gather_pool_positions(ldx, positions)
                self._compute_and_store_chunk_minmax(
                    ldx,
                    new_retrieval_keys,
                    abs_token_start=encode_start - self.sink_size,
                )
            nvtx.range_pop()
            self.steady_size = self.sink_size + self._eff_local_size
            return
        
        for ldx in range(self.layer_num):
            # Offload-only: encode tokens that are about to leave GPU Local.
            # GPU layout: [sink | local | update_buffer].
            offload_start = self.sink_size
            offload_end = self.sink_size + num_new_tokens
            new_tokens_to_encode = self.unified_keys_gpu[ldx][:, :, offload_start:offload_end, :]

            # KIVI per-(quest-)chunk per-channel encode is folded into
            # _compute_and_store_chunk_minmax — it writes both the
            # chunk_min/max bookkeeping AND the 4-bit packed codes into
            # key_kivi_packed_gpu for every chunk it completes (including
            # the carry-merged chunk).  zone-internal offset = encode_start - sink_size.
            nvtx.range_push("index_encode")
            self._compute_and_store_chunk_minmax(
                ldx,
                new_tokens_to_encode,
                abs_token_start=encode_start - self.sink_size,
            )
            nvtx.range_pop()
            nvtx.range_push("index_offload")

            
            # ===========================================
            # Offload-only update:
            # 1. 把 Local 最老的部分 offload 到 CPU（进入 retrieval zone）
            # 2. 滑动 GPU 上的 Local window（Local 新部分 + Update Buffer → 新 Local）
            # 1. 把 Local 最老的 num_new_tokens 个 tokens offload 到 CPU
            # 这些 tokens 从 GPU local 区域移出，进入 CPU retrieval zone
            # GPU 布局: [0:sink] [sink:sink+local] [sink+local:sink+local+update]
            # Local 最老部分: [sink : sink+num_new_tokens]
            cpu_write_start = self.cpu_valid_end[ldx]
            cpu_write_end = cpu_write_start + num_new_tokens

            # 使用预分配 buffer 替代 clone()，避免内存分配开销
            src_keys = self._offload_keys_buffer[ldx][:, :, :num_new_tokens, :]
            src_values = self._offload_values_buffer[ldx][:, :, :num_new_tokens, :]
            src_keys.copy_(self.unified_keys_gpu[ldx][:, :, self.sink_size:self.sink_size + num_new_tokens, :])
            src_values.copy_(self.unified_values_gpu[ldx][:, :, self.sink_size:self.sink_size + num_new_tokens, :])
            q_keys, scale_keys = self._quantize_kv_blocks(src_keys)
            q_values, scale_values = self._quantize_kv_blocks(src_values)
            self._write_block_scales(ldx, slice(None), cpu_write_start, scale_keys, scale_values)
            self._inflight_offload_buffers[ldx] = (q_keys, q_values)
            self._layer_offload_inflight[ldx] = True

            self._main_events[ldx].record()
            with torch.cuda.stream(self._copy_stream):
                self._main_events[ldx].wait()
                self.unified_keys_cpu[ldx][:, :, cpu_write_start:cpu_write_end, :].copy_(q_keys, non_blocking=True)
                self.unified_values_cpu[ldx][:, :, cpu_write_start:cpu_write_end, :].copy_(q_values, non_blocking=True)
                self._copy_events[ldx].record()

            self.cpu_valid_end[ldx] = cpu_write_end

            # 2. 滑动 GPU Local window（可以与 CPU 拷贝并行）
            # near-zone 窗口大小 = _eff_local_size（= local_size + R0，含 aligned-offload
            # 留在 GPU 的余数）。把 [sink+num_new : sink+eff+num_new] 移到 [sink : sink+eff]
            # 即：原 near-zone 新部分 + 原 Update Buffer → 新 near-zone。
            old_local_end = self.sink_size + self._eff_local_size
            old_buffer_end = old_local_end + num_new_tokens
            new_local_start_src = self.sink_size + num_new_tokens

            # 只有当 num_new_tokens < eff 时源和目标才会重叠，需要 clone
            if num_new_tokens < self._eff_local_size:
                new_local_keys = self.unified_keys_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :].clone()
                new_local_values = self.unified_values_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :].clone()
            else:
                new_local_keys = self.unified_keys_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :]
                new_local_values = self.unified_values_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :]
            self.unified_keys_gpu[ldx][:, :, self.sink_size:old_local_end, :] = new_local_keys
            self.unified_values_gpu[ldx][:, :, self.sink_size:old_local_end, :] = new_local_values
            if num_new_tokens < self._eff_local_size:
                del new_local_keys, new_local_values
            nvtx.range_pop()
        
        # 等待所有层的 offload 完成，这样 retrieval 阶段不需要再检查
        nvtx.range_push("index_sync")
        self._sync_offload_copies()
        nvtx.range_pop()
        
        self.steady_size = self.sink_size + self._eff_local_size



    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """
        统一架构:Decode阶段直接append到unified_keys_gpu
        每生成1个token,直接追加到当前序列末尾
        当积累 dynamic_update_interval 个新tokens时,触发动态更新(编码新tokens到retrieval zone)
        """
        # ============ Step 1: 先写入当前 token ============
        nvtx.range_push("kv_write")
        key_to_add = key_states[:, 0, :, :]
        value_to_add = value_states[:, 0, :, :]

        current_len = self.current_seq_len[layer_idx]
        if not self.enable_offload:
            # The new token was written into the SGLang pool before this call.
            gpu_write_pos = None
        else:
            # Offload GPU layout: [Sink][Local][UpdateBuffer].
            steady_threshold = self.steady_size
            if current_len < steady_threshold:
                gpu_write_pos = current_len
            else:
                if self.input_length >= steady_threshold:
                    tokens_in_buffer = self._decode_step_counter - self._buffer_base_counter
                else:
                    tokens_in_buffer = current_len - steady_threshold
                buffer_offset = tokens_in_buffer % self.dynamic_update_interval
                gpu_write_pos = steady_threshold + buffer_offset

        if gpu_write_pos is not None:
            self.unified_keys_gpu[layer_idx][:, :, gpu_write_pos, :] = key_to_add
            self.unified_values_gpu[layer_idx][:, :, gpu_write_pos, :] = value_to_add
        
        # 更新全局序列长度
        self.current_seq_len[layer_idx] += 1
        nvtx.range_pop()
        
        # ============ Step 2: 在第一层检查是否触发 update（写入后检查！）============
        if layer_idx == 0:
            # print(f"[AFTER WRITE] current_seq_len[0]={self.current_seq_len[0]} (compute 会用这个值)")
            # 递增计数器（在写入 token 之后）
            self._decode_step_counter += 1
            self.context += 1
            
            # 处理延迟的 buffer base 更新（base 应设为触发时的 counter 值，即当前 counter - 1）
            if self._pending_buffer_reset:
                self._buffer_base_counter = self._decode_step_counter - 1
                self._pending_buffer_reset = False
                # print(f"[BASE RESET] old_base={old_base} → new_base={self._buffer_base_counter} (counter={self._decode_step_counter})")
            
            # 检查是否需要触发 update（Buffer 刚好写满 dynamic_update_interval 个）
            current_total_len = self.current_seq_len[0]  # 现在包含刚写入的 token！
            steady_threshold = self.steady_size  # 预计算：sink_size + local_size
            
            if current_total_len > steady_threshold:
                if self.input_length >= steady_threshold:
                    # 长 prefill：用 decode counter 计算 buffer 内 token 数
                    buffer_tokens = self._decode_step_counter - self._buffer_base_counter
                else:
                    # 短 prefill：用序列长度溢出计算
                    buffer_tokens = current_total_len - steady_threshold
                
                # Buffer 刚好满（256 个）时触发
                if buffer_tokens > 0 and buffer_tokens % self.dynamic_update_interval == 0:
                    # print(f"[Decode] 触发 _update_polar_index, buffer_tokens={buffer_tokens}, step={self._decode_step_counter}")
                    nvtx.range_push("update_polar_index")
                    self._update_polar_index()
                    self._pending_buffer_reset = True
                    nvtx.range_pop()

        return None, None


    # ============================================================================
    # 单层Decode 阶段的 Attention 计算
    # 输入:query 向量,  输出:attention 结果
    def compute(self, queries, layer_idx):
        """
        输入: queries [batch_size, seq_len, num_heads, head_dim]  seq_len=1
        输出: attn_out [batch_size, seq_len, num_heads, head_dim] seq_len=1
        Decode计算:
        - 当 total_length < 1000 时:全量 attention(不检索)
        - 当 total_length >= 1000 时:使用Polar ANN检索top-k tokens
        """
        nvtx.range_push("decode_prepare")

        # 统一架构：动态计算各区域索引（预计算 sink/local indices，避免重复切片）
        device = self.device
        region_info = self.get_region_indices(layer_idx, device=device)
        total_length = region_info['total_len']
        sink_indices = region_info['sink_indices']
        local_indices = region_info['local_indices']
        retrieval_start = region_info['retrieval_start']
        retrieval_length = region_info['retrieval_length']
        local_len = region_info['local_len']
        
        # Metadata covers only the retrieval prefix completed at prefill/flush.
        valid_length = (
            self.cpu_valid_end[layer_idx]
            if self.enable_offload
            else int(self._actual_retrieval_length or 0)
        )
        bs = int(queries.shape[0])
        
        # 判断是否需要检索
        # total_length 至少要 > sink + local + final_topk*2，否则 retrieval zone 太小
        # GQA mean stays off this path: the sparse graph averages query heads itself.
        min_total_for_retrieval = max(self.FULL_ATTENTION_THRESHOLD, self.sink_size + self.local_size + self.final_topk * 2)
        use_full_attention = (total_length < min_total_for_retrieval) or (valid_length <= 0)
        nvtx.range_pop()


        if use_full_attention:
            # ============ 全量 Attention 模式 ============
            nvtx.range_push("full_kv_load")
            concat_keys, concat_values = self._get_full_kv_for_attention(layer_idx, sink_indices, local_indices, retrieval_start, retrieval_length, device, queries.dtype)
            nvtx.range_pop()
        else:

            # ============ 检索模式 ============
            # KIVI stage-2 reads `key_kivi_packed_gpu` (chunk-major after
            # A-1) directly by retrieval-internal chunk id.  We slice on
            # the chunk axis to expose only the chunks that have been
            # written so far — Quest stage-1 already filters by
            # ``_chunk_valid_mask_gpu``, but slicing here lets the kernel
            # bounds-check ``real_chunk_id < n_chunks_packed`` cheaply.
            chunk_size = self.quest_chunk_size
            n_valid_chunks = max(0, valid_length // chunk_size)
            # n_valid_chunks grows only when new retrieval chunks become valid
            # after prefill/update flushes. Most decode steps reuse this view.
            if self._cached_kivi_packed_n_chunks[layer_idx] == n_valid_chunks:
                key_kivi_packed = self._cached_kivi_packed_views[layer_idx]
            else:
                key_kivi_packed = self.key_kivi_packed_gpu[layer_idx][:, :, :n_valid_chunks, :, :]
                self._cached_kivi_packed_n_chunks[layer_idx] = n_valid_chunks
                self._cached_kivi_packed_views[layer_idx] = key_kivi_packed

            if (
                self.enable_offload
                and self._layer_offload_inflight[layer_idx]
                and self._copy_events[layer_idx].query()
            ):
                self._inflight_offload_buffers[layer_idx] = None
                self._layer_offload_inflight[layer_idx] = False

            if (
                self.enable_offload
                and self._layer_offload_inflight[layer_idx]
                and self._sparse_attn_enabled
            ):
                # This layer's CPU copy is still running. The tokens are already
                # in the SGLang pool, so this one eager step reads them there.
                nvtx.range_push("gpu_sparse_attn")
                try:
                    sparse_out = self._gpu_sparse_attention(
                        queries,
                        None,
                        key_kivi_packed,
                        layer_idx,
                        bs,
                        region_info,
                        force_pool=True,
                    )
                finally:
                    nvtx.range_pop()
                    pass
                if sparse_out is not None:
                    return sparse_out
                self._sync_layer_offload(layer_idx)

            if not self.enable_offload and self._sparse_attn_enabled:
                nvtx.range_push("gpu_sparse_attn")
                try:
                    sparse_out = self._gpu_sparse_attention(
                        queries, None, key_kivi_packed, layer_idx, bs, region_info,
                    )
                finally:
                    nvtx.range_pop()
                    pass
                if sparse_out is not None:
                    return sparse_out

            query_group = queries[:, 0, :, :]
            query_avg_gpu = query_group.view(
                bs, self.kv_head, self.group_size, self.head_dim
            ).mean(dim=2)
            query_batch = query_avg_gpu.unsqueeze(2)

            nvtx.range_push("quest_topk")
            topk_indices_batch = self.collision_based_topk_batch(
                query=query_batch,
                key_kivi_packed=key_kivi_packed,
                layer_idx=layer_idx,
            )
            nvtx.range_pop()
            
            # if layer_idx == 0:
            #     print(f"[Debug] topk_indices_batch.shape={topk_indices_batch.shape}, final_topk={self.final_topk}", flush=True)

            if self.enable_offload:
                indices = topk_indices_batch
                if indices.dtype != torch.int64:
                    indices = indices.to(dtype=torch.int64)
                if not indices.is_contiguous():
                    indices = indices.contiguous()
                if self._uva_gather_kv is not None:
                    from sglang_zoomkv.kernels.nvtx_ranges import UVA_READ, nvtx_stage
                    nvtx.range_push("uva_h2d_gather")
                    try:
                        with nvtx_stage(UVA_READ):
                            self._gather_offload_kv(
                                layer_idx,
                                indices,
                                self._topk_keys_buffer,
                                self._topk_values_buffer,
                            )
                    finally:
                        nvtx.range_pop()
                        pass
                else:
                    self._fallback_gather_i8(
                        layer_idx,
                        indices,
                        self._topk_keys_buffer,
                        self._topk_values_buffer,
                    )
                if self._sparse_attn_enabled and self._sparse_ext is not None:
                    nvtx.range_push("offload_sparse_attn")
                    try:
                        sparse_out = self._offload_sparse_attention(
                            queries, layer_idx, bs, region_info, int(topk_indices_batch.shape[-1]),
                        )
                    finally:
                        nvtx.range_pop()
                        pass
                    if sparse_out is not None:
                        return sparse_out
                sink_local_end = self.sink_size + local_len
                sink_local_keys = self.unified_keys_gpu[layer_idx][:, :, :sink_local_end, :]
                sink_local_values = self.unified_values_gpu[layer_idx][:, :, :sink_local_end, :]
            else:
                if self._use_pool_kv:
                    raise RuntimeError(
                        "GPU-only attention could not read the SGLang KV pool"
                    )
                # Retrieval indices are zone-relative; full GPU sidecar positions
                # are chronological, so add sink_size exactly once.
                from sglang_zoomkv.cache.gpu_only import (
                    retrieval_to_cache_indices,
                    steady_indices as build_steady_indices,
                )
                nvtx.range_push("gpu_gather")
                cache_indices = retrieval_to_cache_indices(
                    topk_indices_batch, self.sink_size, valid_length
                )
                gather_indices = cache_indices.unsqueeze(-1)
                gather_indices = gather_indices.expand(-1, -1, -1, self.head_dim)
                n_topk = topk_indices_batch.shape[-1]
                torch.gather(
                    self.unified_keys_gpu[layer_idx], 2, gather_indices,
                    out=self._topk_keys_buffer[:, :, :n_topk, :],
                )
                torch.gather(
                    self.unified_values_gpu[layer_idx], 2, gather_indices,
                    out=self._topk_values_buffer[:, :, :n_topk, :],
                )
                steady_indices = build_steady_indices(sink_indices, local_indices)
                sink_local_keys = torch.index_select(
                    self.unified_keys_gpu[layer_idx], 2, steady_indices
                )
                sink_local_values = torch.index_select(
                    self.unified_values_gpu[layer_idx], 2, steady_indices
                )
                nvtx.range_pop()

            nvtx.range_push("compute_attention")
            n_topk = topk_indices_batch.shape[-1]
            concat_keys = torch.cat([sink_local_keys, self._topk_keys_buffer[:, :, :n_topk, :]], dim=2)
            concat_values = torch.cat([sink_local_values, self._topk_values_buffer[:, :, :n_topk, :]], dim=2)
            k_cache = concat_keys.transpose(1, 2)
            v_cache = concat_values.transpose(1, 2)
            attn_out = flash_attn_with_kvcache_compat(
                q=queries,
                k_cache=k_cache,
                v_cache=v_cache,
                causal=False
            )
            del sink_local_keys, sink_local_values, concat_keys, concat_values, k_cache, v_cache
            nvtx.range_pop()
            return attn_out

        # 7. 全量 Attention：cat 已完成，仅 Flash
        k_cache = concat_keys.transpose(1, 2)
        v_cache = concat_values.transpose(1, 2)
        nvtx.range_push("compute_attention")
        attn_out = flash_attn_with_kvcache_compat(
            q=queries,  # [bs, 1, num_heads, head_dim]
            k_cache=k_cache,  # [bs, seq_len, kv_heads, head_dim]
            v_cache=v_cache,  # [bs, seq_len, kv_heads, head_dim]
            causal=False
        )
        del concat_keys, concat_values, k_cache, v_cache
        nvtx.range_pop()
        return attn_out










    # ============================================================================
    # Hierarchical Quest cache refresh
    # ============================================================================
    def _quest_aligned_candidate_tokens(self, kv_len: int) -> int:
        """Flat stage-1 token budget aligned to ``quest_chunk_size``."""
        chunk = self.quest_chunk_size
        if kv_len <= 0:
            return 0
        if kv_len < chunk:
            return kv_len
        cap = (kv_len // chunk) * chunk
        target = (
            self._force_cand_tokens
            if self._force_cand_tokens is not None
            else math.ceil(kv_len * self.quest_coarse_candidate_ratio)
        )
        target_aligned = max(chunk, ((target + chunk - 1) // chunk) * chunk)
        return min(cap, target_aligned)

    def _refresh_quest_chunk_cache(self, kv_len: int):
        """Recompute and cache all stage-1 invariants.

        Called at prefill end and _update_polar_index; the decode hot path
        reads only cached values — no shape queries, no slicing, no arithmetic.
        """
        chunk = self.quest_chunk_size
        self._refresh_live_skipped_indices(kv_len)
        if kv_len < chunk:
            self._cached_quest_n_chunks = 0
            self._cached_quest_nk = 0
            self._cached_quest_kv_len = kv_len
            self._cached_cds_n_dense = 0
            self._cached_cds_density_scores = None
            self._cached_hq_n_large = 0
            self._cached_hq_nk_large = 0
            self._cached_hq_nk_small = 0
            return
        n_chunks = kv_len // chunk
        self._cached_quest_n_chunks = n_chunks
        self._cached_quest_kv_len = kv_len
        bs = self.batch_size
        kv_heads = self.kv_head
        self._cached_chunk_min_views = [
            t[:bs] for t in self._chunk_key_min_gpu
        ]
        self._cached_chunk_max_views = [
            t[:bs] for t in self._chunk_key_max_gpu
        ]
        self._cached_chunk_valid_views = [
            t[:bs] for t in self._chunk_valid_mask_gpu
        ]
        if not self._use_hierarchical_quest:
            target_tokens = self._quest_aligned_candidate_tokens(kv_len)
            nk = max(1, min(n_chunks, target_tokens // chunk))
            self._cached_quest_nk = nk
            self._cached_scores_view = (
                self._quest_scores_buffer_small[:bs * kv_heads * n_chunks]
                .view(bs, kv_heads, n_chunks)
            )
            nk_small_for_cds = nk
            self._cached_cds_n_dense = max(1, int(nk_small_for_cds * self.cds_dense_ratio))
            self._cached_dense_mask = torch.zeros(
                bs, kv_heads, nk_small_for_cds, dtype=torch.bool, device=self.device,
            )
            self._cached_cds_out_scores = torch.empty(
                bs, kv_heads, nk_small_for_cds * self.quest_chunk_size,
                dtype=torch.float32, device=self.device,
            )
            self._cached_cds_out_indices = torch.empty(
                bs, kv_heads, nk_small_for_cds * self.quest_chunk_size,
                dtype=torch.int64, device=self.device,
            )
            self._cached_cds_density_scores = torch.empty(
                bs, kv_heads, nk_small_for_cds, dtype=torch.float32, device=self.device,
            )
            self._cached_cds_final_indices = torch.empty(
                bs, kv_heads, self.final_topk + self.MAX_SKIPPED_INDICES,
                dtype=torch.int64, device=self.device,
            )
            return
        # --- Hierarchical Quest cached views ---
        n_large = n_chunks // self._hq_factor
        nk_large = max(1, int(math.ceil(n_large * self._hq_large_ratio)))
        nk_large = min(nk_large, n_large)
        nk_sub_total = nk_large * self._hq_factor
        nk_small = max(1, int(math.ceil(nk_sub_total * self._hq_small_ratio)))
        nk_small = min(nk_small, nk_sub_total)
        self._cached_hq_n_large = n_large
        self._cached_hq_nk_large = nk_large
        self._cached_hq_nk_small = nk_small
        nk_small_for_cds = nk_small
        # Output buffer for hq_map_back; reused every decode step to avoid
        # per-call torch.empty_like allocation on the hot path.
        self._cached_hq_chunk_idx = torch.empty(
            bs, kv_heads, nk_small, dtype=torch.int64, device=self.device,
        )
        self._cached_hq_scores_large_view = (
            self._quest_scores_buffer_large[:bs * kv_heads * n_large]
            .view(bs, kv_heads, n_large)
        )
        self._cached_hq_scores_sub_view = (
            self._quest_scores_buffer_sub[:bs * kv_heads * nk_large * self._hq_factor]
            .view(bs, kv_heads, nk_large * self._hq_factor)
        )
        self._cached_hq_large_min_views = [
            t[:bs] for t in self._chunk_key_min_gpu_large
        ]
        self._cached_hq_large_max_views = [
            t[:bs] for t in self._chunk_key_max_gpu_large
        ]
        self._cached_hq_large_valid_views = [
            t[:bs] for t in self._chunk_valid_mask_gpu_large
        ]

        # --- CDS cache (shared by hierarchical + single-level) ---
        self._cached_cds_n_dense = max(1, int(nk_small_for_cds * self.cds_dense_ratio))
        self._cached_dense_mask = torch.zeros(
            bs, kv_heads, nk_small_for_cds, dtype=torch.bool, device=self.device,
        )
        self._cached_cds_out_scores = torch.empty(
            bs, kv_heads, nk_small_for_cds * self.quest_chunk_size,
            dtype=torch.float32, device=self.device,
        )
        self._cached_cds_out_indices = torch.empty(
            bs, kv_heads, nk_small_for_cds * self.quest_chunk_size,
            dtype=torch.int64, device=self.device,
        )
        self._cached_cds_density_scores = torch.empty(
            bs, kv_heads, nk_small_for_cds, dtype=torch.float32, device=self.device,
        )
        # Output buffer for the fused topk_gather_concat kernel.  Sized for
        # [bs, kv, final_topk + MAX_SKIPPED_INDICES] so the same buffer fits
        # all sk_count values (caller slices to actual region after launch).
        self._cached_cds_final_indices = torch.empty(
            bs, kv_heads, self.final_topk + self.MAX_SKIPPED_INDICES,
            dtype=torch.int64, device=self.device,
        )

    def _refresh_live_skipped_indices(self, kv_len: int):
        """Recompute the live skipped-token set for current retrieval length.

        skipped 定义为：当前 stage-1 chunk 选择路径无法触达、但必须并入 stage-2
        rerank 的 retrieval-zone token 索引（= retrieval 尾部未对齐到 chunk 边界
        的那点 "live tail"）。每次 _refresh_quest_chunk_cache 都按当前 kv_len
        重算并覆写 buffer，不做历史累计。
        """
        kv_len = max(0, int(kv_len))
        chunk = int(self.quest_chunk_size)
        n_chunks = kv_len // chunk if chunk > 0 else 0
        if self._use_hierarchical_quest:
            covered_small = (n_chunks // self._hq_factor) * self._hq_factor
        else:
            covered_small = n_chunks
        covered_tokens = covered_small * chunk
        skipped_count = max(0, kv_len - covered_tokens)
        # 诊断：看 "多余 / skipped" 这段到底多大（= retrieval_len mod align_chunk）。
        # 只在 refresh（每次 update / prefill 末，非 decode 热路径）打印一次。
        large_chunk = self._offload_align_chunk
        n_trailing_chunks = n_chunks - covered_small        # 完整 small chunk 数
        carry_tail = kv_len - n_chunks * chunk              # 不足 1 个 small chunk 的尾巴
        if os.environ.get("POLAR_DEBUG_SKIPPED", "0") != "0":
            print(
                f"[skipped] retrieval_len(kv_len)={kv_len} align_chunk={large_chunk} "
                f"=> skipped={skipped_count} (= kv_len % align_chunk = {kv_len % large_chunk}); "
                f"其中 trailing 完整 small chunk={n_trailing_chunks} 个 ({n_trailing_chunks * chunk} tok) "
                f"+ carry 残块={carry_tail} tok",
                flush=True,
            )
        if skipped_count > self.MAX_SKIPPED_INDICES:
            raise RuntimeError(
                "[chunk_minmax] live skipped indices overflow: "
                f"kv_len={kv_len}, covered_tokens={covered_tokens}, "
                f"skipped={skipped_count} > MAX={self.MAX_SKIPPED_INDICES}. "
                "增大 POLAR_MAX_SKIPPED_INDICES 或减小 align_chunk。"
            )
        idx_row = None
        if skipped_count > 0:
            idx_row = torch.arange(
                covered_tokens, kv_len, device=self.device, dtype=torch.int64
            ).unsqueeze(0).unsqueeze(0)
        for ldx in range(self.layer_num):
            self._chunk_skipped_count[ldx] = skipped_count
            if idx_row is not None:
                self._chunk_skipped_indices_gpu[ldx][:, :, :skipped_count].copy_(idx_row)

    # ------------------------------------------------------------------
    # Chunk-level raw K min/max metadata maintenance (Quest-style)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _kivi_pack_chunks(
        self,
        layer_idx: int,
        b_sl: slice,
        K_chunked: torch.Tensor,           # [bs_local, kv, n_chunks, chunk, D] bf16
        chunk_start: int,
        chunk_end: int,
    ) -> None:
        """KIVI-encode ``K_chunked`` and write into ``key_kivi_packed_gpu``.

        Storage layout (A-1 chunk-major):

            ``key_kivi_packed_gpu[layer]`` :
                ``[bs, kv, n_chunks_max, n_pack, group_size]`` int32

        ``chunk_start`` / ``chunk_end`` are retrieval-internal chunk
        indices (same as ``_chunk_key_min_gpu`` axis-2).  We write the
        packed codes for the freshly-completed chunks at
        ``[:, :, chunk_start:chunk_end, :, :]`` directly — no token-level
        offset / sink_size shift any more.
        """
        if chunk_end <= chunk_start:
            return
        chunk = self.quest_chunk_size
        n_chunks = chunk_end - chunk_start
        bs_local, kv, n_chunks_ck, g_check, D = K_chunked.shape
        assert n_chunks_ck == n_chunks and g_check == chunk, (
            f"kivi pack shape mismatch: K_chunked={K_chunked.shape}, "
            f"n_chunks={n_chunks}, chunk={chunk}"
        )
        # Per-(quest-)chunk per-channel min/max we just wrote.
        mn = self._chunk_key_min_gpu[layer_idx][
            b_sl, :, chunk_start:chunk_end, :
        ].unsqueeze(-2)                                # [bs, kv, n_chunks, 1, D]
        mx = self._chunk_key_max_gpu[layer_idx][
            b_sl, :, chunk_start:chunk_end, :
        ].unsqueeze(-2)
        levels = (1 << self._kivi_bits) - 1
        # bf16 ladder (matches kivi_simulate_dequant and the CUDA kernel).
        scale = (mx - mn) / float(levels)
        scale_safe = scale.clamp(min=1e-8)
        codes = ((K_chunked - mn) / scale_safe).clamp_(
            0.0, float(levels),
        ).round_().to(torch.int32)
        # Pack along head_dim axis: [bs, kv, n_chunks, chunk, n_pack] int32.
        packed = _pack_along_last_dim(codes, bits=self._kivi_bits)
        # Re-layout to chunk-major: [bs, kv, n_chunks, n_pack, chunk] so
        # that within one chunk the 16 lanes for a fixed dim-block live at
        # consecutive int32 positions.  ``.contiguous()`` materialises the
        # permuted tensor; ``.copy_()`` then does a straight memcpy into
        # the destination slice (which is also stride-1 on its inner axis).
        packed_cm = packed.transpose(-1, -2).contiguous()
        self.key_kivi_packed_gpu[layer_idx][
            b_sl, :, chunk_start:chunk_end, :, :
        ].copy_(packed_cm)




    @torch.inference_mode()
    def _compute_and_store_chunk_minmax(
        self,
        layer_idx: int,
        key_block: torch.Tensor,
        abs_token_start: int,
        batch_slice: Optional[slice] = None,
    ):
        """把一段连续 raw-K 写入 chunk 元数据（small chunk）。

        live-carry 语义：
            - 每层维护 <chunk_size 的 carry tail（上次不足一块的尾巴 token）。
            - 新 update 先用 carry + 本次开头 token 拼满 1 个 chunk，再继续处理。
            - 这样可避免 decode 长跑时周期性 head/tail hole 累积；当前 retrieval 末尾
              不足一块的 token 由 `_refresh_live_skipped_indices(kv_len)` 在下一次
              `_refresh_quest_chunk_cache` 时按 kv_len 重算后写入 skipped buffer。
        Args:
            layer_idx:        目标层
            key_block:        [bs_local, kv, n_tokens, head_dim] 一段连续的 RoPE-后 raw K
                              （要求是写入 retrieval zone 的那批 K，不含 sink/local）
            abs_token_start:  这批 token 在 retrieval zone 内的绝对起始偏移
                              （= 写入 unified_keys_*[..., retrieval_start:retrieval_end] 时的
                              retrieval_start - sink_size）
            batch_slice:      None  → 写入 buffer[:bs_local, ...]（lockstep 全 batch）
                              slice → 写入 buffer[batch_slice, ...]（单样本写入指定 slot）
        """
        chunk = self.quest_chunk_size
        bs_local, kv, n_tokens, dh = key_block.shape
        if n_tokens <= 0:
            return
        b_sl = batch_slice if batch_slice is not None else slice(0, bs_local)

        # ---- 先尝试用 carry 拼出一个完整 chunk ----
        carry_cnt = int(self._chunk_carry_count[layer_idx])
        if carry_cnt > 0:
            need = chunk - carry_cnt
            if n_tokens < need:
                if carry_cnt + n_tokens > self.quest_chunk_size - 1:
                    raise RuntimeError(
                        f"[chunk_minmax] invalid carry append: carry={carry_cnt} + n={n_tokens} > chunk-1={self.quest_chunk_size - 1}"
                    )
                self._chunk_carry_keys[layer_idx][b_sl, :, carry_cnt:carry_cnt + n_tokens, :].copy_(key_block[:, :, :n_tokens, :])
                self._chunk_carry_count[layer_idx] = carry_cnt + n_tokens
                return

            merge = self._chunk_merge_buffer[layer_idx][b_sl, :, :, :]
            merge[:, :, :carry_cnt, :].copy_(self._chunk_carry_keys[layer_idx][b_sl, :, :carry_cnt, :])
            merge[:, :, carry_cnt:, :].copy_(key_block[:, :, :need, :])

            chunk_start = (abs_token_start - carry_cnt) // chunk
            chunk_end = chunk_start + 1
            if chunk_end > self._chunk_n_max:
                raise RuntimeError(
                    f"[chunk_minmax] layer={layer_idx} chunk_end={chunk_end} exceeds "
                    f"preallocated _chunk_n_max={self._chunk_n_max}; check retrieval_zone_capacity."
                )
            self._chunk_key_min_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(
                merge.amin(dim=2, keepdim=True)
            )
            self._chunk_key_max_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(
                merge.amax(dim=2, keepdim=True)
            )
            self._chunk_key_centroid_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(
                merge.mean(dim=2, keepdim=True)
            )
            self._chunk_valid_mask_gpu[layer_idx][b_sl, :, chunk_start:chunk_end].fill_(True)
            if chunk_end > self._chunk_valid_end[layer_idx]:
                self._chunk_valid_end[layer_idx] = chunk_end
            self._update_large_chunk_minmax(layer_idx, chunk_start, chunk_end, b_sl)

            # KIVI pack the freshly-completed (carry + new_head) chunk.
            self._kivi_pack_chunks(
                layer_idx, b_sl,
                merge.view(bs_local, kv, 1, chunk, dh),
                chunk_start=chunk_start, chunk_end=chunk_end,
            )

            key_block = key_block[:, :, need:, :]
            abs_token_start = abs_token_start + need
            n_tokens -= need
            self._chunk_carry_count[layer_idx] = 0
            carry_cnt = 0
            if n_tokens <= 0:
                return

        # 若无 carry 但起点不对齐，说明上游时序/状态不一致（carry 丢失）。
        if abs_token_start % chunk != 0:
            raise RuntimeError(
                "[chunk_minmax] misaligned abs_token_start without carry: "
                f"layer={layer_idx}, abs={abs_token_start}, chunk={chunk}, carry={carry_cnt}"
            )

        # ---- 现在 abs_token_start 一定 chunk-aligned ----
        usable = (n_tokens // chunk) * chunk
        if usable > 0:
            n_chunks = usable // chunk
            chunk_start = abs_token_start // chunk
            chunk_end = chunk_start + n_chunks
            if chunk_end > self._chunk_n_max:
                raise RuntimeError(
                    f"[chunk_minmax] layer={layer_idx} chunk_end={chunk_end} exceeds "
                    f"preallocated _chunk_n_max={self._chunk_n_max}; check retrieval_zone_capacity."
                )
            K_chunked = key_block[:, :, :usable, :].view(bs_local, kv, n_chunks, chunk, dh)
            # amin/amax over the chunk dim → [bs_local, kv, n_chunks, head_dim]
            self._chunk_key_min_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(K_chunked.amin(dim=3))
            self._chunk_key_max_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(K_chunked.amax(dim=3))
            self._chunk_key_centroid_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(K_chunked.mean(dim=3))
            self._chunk_valid_mask_gpu[layer_idx][b_sl, :, chunk_start:chunk_end].fill_(True)
            if chunk_end > self._chunk_valid_end[layer_idx]:
                self._chunk_valid_end[layer_idx] = chunk_end

            self._update_large_chunk_minmax(layer_idx, chunk_start, chunk_end, b_sl)

            # KIVI pack the aligned full-chunk region.
            self._kivi_pack_chunks(
                layer_idx, b_sl, K_chunked,
                chunk_start=chunk_start, chunk_end=chunk_end,
            )

        # ---- 尾部不足一个 small chunk：写入 carry（live，不进历史累计 skipped）----
        tail = n_tokens - usable
        if tail > 0:
            if tail > self.quest_chunk_size - 1:
                raise RuntimeError(
                    f"[chunk_minmax] invalid tail={tail} for chunk={self.quest_chunk_size}"
                )
            self._chunk_carry_keys[layer_idx][b_sl, :, :tail, :].copy_(
                key_block[:, :, usable:usable + tail, :]
            )
            self._chunk_carry_count[layer_idx] = tail
        else:
            self._chunk_carry_count[layer_idx] = 0

    @torch.inference_mode()
    def _update_large_chunk_minmax(
        self,
        layer_idx: int,
        chunk_start: int,
        chunk_end: int,
        b_sl: slice,
    ):
        """Derive large-chunk min/max from base-level (small chunk) results.

        For each large chunk group that has any newly written small chunks,
        re-derive its bbox from the full range of small-chunk min/max values.
        """
        factor = self._hq_factor
        g_start = chunk_start // factor
        g_end = (chunk_end + factor - 1) // factor
        g_end = min(g_end, self._chunk_n_max_large)
        if g_start >= g_end:
            return

        base_min = self._chunk_key_min_gpu[layer_idx][b_sl]
        base_max = self._chunk_key_max_gpu[layer_idx][b_sl]
        valid_end = self._chunk_valid_end[layer_idx]

        for g in range(g_start, g_end):
            b_lo = g * factor
            b_hi = min(b_lo + factor, valid_end)
            if b_hi <= b_lo:
                continue
            self._chunk_key_min_gpu_large[layer_idx][b_sl, :, g, :] = base_min[:, :, b_lo:b_hi, :].amin(dim=2)
            self._chunk_key_max_gpu_large[layer_idx][b_sl, :, g, :] = base_max[:, :, b_lo:b_hi, :].amax(dim=2)
            self._chunk_valid_mask_gpu_large[layer_idx][b_sl, :, g].fill_(True)

        if g_end > self._chunk_valid_end_large[layer_idx]:
            self._chunk_valid_end_large[layer_idx] = g_end



    def _topk_3d(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k on last dim [bs, kv_heads, L] → indices [bs, kv_heads, k].

        Production speed path always uses the fused float top-k CUDA backend.

        Caller contract:
        - ``k`` MUST be a Python int (all current call sites source from
          cached ints set at init).
        - ``scores`` MUST already be fp32; callers pass pre-allocated
          fp32 buffers (``_quest_scores_buffer_large``, ``_quest_scores_buffer_sub``,
          ``_cached_cds_density_scores``, ``_cached_cds_out_scores``).

        Defensive ``int(k)`` / ``k > 0`` / ``scores.float()`` were removed:
        the int cast is no-op (~30 ns) but ``scores.float()`` carried ~1-2 μs
        of PyTorch dispatch overhead even on already-fp32 inputs (≈200-400 μs
        per decode token over 5 × 36 calls).  ``float_topk.cu`` itself
        auto-casts non-fp32 inputs and asserts ``k > 0``, so real misuse is
        still caught loudly.
        """
        return self._float_topk_ext.float_topk_3d(scores, k)




    def _quest_chunk_score(self, raw_q, chunk_min, chunk_max, scores_out, n_chunks, chunk_valid):
        if self._quest_kernel is not None:
            self._quest_kernel.quest_chunk_score(
                raw_q, chunk_min, chunk_max, scores_out, n_chunks, chunk_valid,
            )
            return
        # score(chunk) = sum_d max(q_d * max_d, q_d * min_d)
        q = raw_q.float().unsqueeze(2)
        cmin = chunk_min[:, :, :n_chunks, :].float()
        cmax = chunk_max[:, :, :n_chunks, :].float()
        score = torch.maximum(q * cmax, q * cmin).sum(dim=-1)
        if chunk_valid is not None:
            score = score.masked_fill(~chunk_valid[:, :, :n_chunks], -1e30)
        scores_out.copy_(score)

    def _quest_sub_chunk_score(
        self, raw_q, chunk_min, chunk_max, large_idx, sub_scores, nk_large, factor,
    ):
        if self._quest_kernel is not None:
            self._quest_kernel.quest_sub_chunk_score(
                raw_q, chunk_min, chunk_max, large_idx, sub_scores, nk_large, factor,
            )
            return
        offsets = torch.arange(factor, device=raw_q.device, dtype=large_idx.dtype)
        small_idx = large_idx.unsqueeze(-1) * factor + offsets
        flat_idx = small_idx.reshape(raw_q.shape[0], raw_q.shape[1], nk_large * factor)
        gather_idx = flat_idx.unsqueeze(-1).expand(-1, -1, -1, raw_q.shape[-1])
        cmin = torch.gather(chunk_min.float(), 2, gather_idx)
        cmax = torch.gather(chunk_max.float(), 2, gather_idx)
        q = raw_q.float().unsqueeze(2)
        sub_scores.copy_(torch.maximum(q * cmax, q * cmin).sum(dim=-1))

    def _quest_map_back(self, large_idx, sub_topk_pos, chunk_idx, factor, n_chunks):
        if self._quest_kernel is not None:
            self._quest_kernel.quest_map_back(
                large_idx, sub_topk_pos, chunk_idx, factor, n_chunks,
            )
            return
        # float_topk leaves -1 in slots it did not fill. Gathering that
        # index is a CUDA device-side assert, which kills the rank and
        # surfaces on the other rank as an NCCL TCPStore broken pipe.
        nk_large = large_idx.size(2)
        span = max(nk_large * factor, 1)
        safe_pos = sub_topk_pos.clamp(min=0, max=span - 1)
        large_slot = torch.div(safe_pos, factor, rounding_mode="floor")
        large_slot.clamp_(min=0, max=max(nk_large - 1, 0))
        sub = torch.remainder(safe_pos, factor)
        parent = torch.gather(large_idx, 2, large_slot)
        mapped = parent * factor + sub
        if n_chunks > 0:
            mapped.clamp_(min=0, max=n_chunks - 1)
        else:
            mapped.zero_()
        chunk_idx.copy_(mapped)

    def _quest_pick_coarse_indices_flat(
        self,
        *,
        raw_q: torch.Tensor,
        layer_idx: int,
    ):
        """Single-level Quest: score all small chunks once, then pick top chunks."""
        n_chunks = self._cached_quest_n_chunks
        nk = self._cached_quest_nk
        if n_chunks <= 0 or nk <= 0:
            chunk_idx = torch.zeros(
                raw_q.shape[0], raw_q.shape[1], 1,
                dtype=torch.int64, device=raw_q.device,
            )
            return chunk_idx, 0

        scores_out = self._cached_scores_view
        chunk_min = self._cached_chunk_min_views[layer_idx]
        chunk_max = self._cached_chunk_max_views[layer_idx]
        chunk_valid = self._cached_chunk_valid_views[layer_idx]
        nvtx.range_push("quest_score")
        self._quest_chunk_score(
            raw_q, chunk_min, chunk_max, scores_out, n_chunks, chunk_valid,
        )
        nvtx.range_pop()
        nvtx.range_push("quest_chunk_topk")
        chunk_idx = self._topk_3d(scores_out, nk).clamp(min=0, max=n_chunks - 1)
        nvtx.range_pop()
        return chunk_idx, nk




    def _quest_pick_coarse_indices_hierarchical(
        self,
        *,
        raw_q: torch.Tensor,
        layer_idx: int,
    ):
        """Two-level hierarchical Quest: large chunk → sub-chunk scoring.

        Stage 1: score all large chunks, pick top nk_large.
        Stage 2: for each selected large chunk, score its sub-chunks via
                 offset-based CUDA kernel, then pick top nk_small globally.
        """
        n_chunks = self._cached_quest_n_chunks
        n_large = self._cached_hq_n_large
        nk_large = self._cached_hq_nk_large
        nk_small = self._cached_hq_nk_small
        factor = self._hq_factor

        if n_large <= 0:
            chunk_idx = torch.zeros(raw_q.shape[0], raw_q.shape[1], 1,
                                    dtype=torch.int64, device=raw_q.device)
            return chunk_idx, 0

        # --- Stage 1: score large chunks ---
        from sglang_zoomkv.kernels.nvtx_ranges import (
            LARGE_CHUNK_SELECT,
            SMALL_CHUNK_SELECT,
            nvtx_stage,
        )
        _large_stage = nvtx_stage(LARGE_CHUNK_SELECT)
        _large_stage.__enter__()
        nvtx.range_push("quest_large_score")
        large_scores = self._cached_hq_scores_large_view
        large_min = self._cached_hq_large_min_views[layer_idx]
        large_max = self._cached_hq_large_max_views[layer_idx]
        large_valid = self._cached_hq_large_valid_views[layer_idx]
        self._quest_chunk_score(
            raw_q, large_min, large_max, large_scores, n_large, large_valid,
        )
        nvtx.range_pop()

        nvtx.range_push("quest_large_topk")
        large_idx = self._topk_3d(large_scores, nk_large)
        large_idx = large_idx.clamp(min=0, max=n_large - 1)
        nvtx.range_pop()
        _large_stage.__exit__(None, None, None)

        # --- Stage 2: score sub-chunks within selected large chunks ---
        _small_stage = nvtx_stage(SMALL_CHUNK_SELECT)
        _small_stage.__enter__()
        nvtx.range_push("quest_small_score")
        sub_scores = self._cached_hq_scores_sub_view
        chunk_min = self._cached_chunk_min_views[layer_idx]
        chunk_max = self._cached_chunk_max_views[layer_idx]
        self._quest_sub_chunk_score(
            raw_q, chunk_min, chunk_max,
            large_idx, sub_scores,
            nk_large, factor,
        )
        nvtx.range_pop()

        nvtx.range_push("quest_small_topk")
        sub_topk_pos = self._topk_3d(sub_scores, nk_small)
        nvtx.range_pop()

        # --- Map back to global small chunk indices ---
        nvtx.range_push("hq_map_back")
        chunk_idx = self._cached_hq_chunk_idx
        self._quest_map_back(
            large_idx,
            sub_topk_pos,
            chunk_idx,
            factor,
            n_chunks,
        )
        nvtx.range_pop()
        _small_stage.__exit__(None, None, None)

        return chunk_idx, nk_small




    # ------------------------------------------------------------------
    # CDS: Coarse → Dense/Sparse rerank + final topk
    # ------------------------------------------------------------------
    def _cds_dense_sparse_select(
        self,
        chunk_idx: torch.Tensor,
        bs: int,
        key_kivi_packed: torch.Tensor,
        raw_q: torch.Tensor,
        layer_idx: int,
        skipped_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """KIVI fused dense/sparse rerank + global top-k.

        1. Classify dense chunks via float_topk on centroid @ q
        2. Per-chunk KIVI on-the-fly dequant + raw_q dot + bitonic top-K
        3. Global top-K across nk*chunk_size token candidates.

        NVTX names match the RaBitQ pipeline (``cds_classify`` /
        ``cds_fused_kernel`` / ``cds_global_topk`` / ``cds_skipped``) so
        existing nsys reports line up.
        """
        chunk = self.quest_chunk_size
        # KIVI's dense/sparse "topk" is per-(quest-)chunk, must fit within
        # ``chunk``.  RaBitQ defaults dense_topk=16/sparse_topk=8 are for
        # chunk=16; clamp for smaller chunks.
        sparse_topk = max(1, min(int(self.cds_sparse_topk), chunk))
        dense_topk = max(1, min(int(self.cds_dense_topk), chunk))

        from sglang_zoomkv.kernels.nvtx_ranges import FINE_RERANK, nvtx_stage
        _rerank_stage = nvtx_stage(FINE_RERANK)
        _rerank_stage.__enter__()
        try:
            return self._cds_dense_sparse_select_body(
                chunk_idx, bs, key_kivi_packed, raw_q, layer_idx, skipped_indices,
                chunk, sparse_topk, dense_topk,
            )
        finally:
            _rerank_stage.__exit__(None, None, None)
    def _cds_dense_sparse_select_body(
        self, chunk_idx, bs, key_kivi_packed, raw_q, layer_idx, skipped_indices,
        chunk, sparse_topk, dense_topk,
    ) -> torch.Tensor:
        # --- Step 1: classify dense/sparse by centroid @ q ---
        nvtx.range_push("cds_classify")
        centroid_all = self._chunk_key_centroid_gpu[layer_idx][:bs]
        density_scores = self._cached_cds_density_scores
        partial_chunk_density_scores_cuda(
            chunk_idx, centroid_all, raw_q, density_scores,
            kernel=self._rerank_topk_kernel,
        )
        n_dense = self._cached_cds_n_dense
        dense_positions = self._topk_3d(density_scores, n_dense)
        dense_mask = self._cached_dense_mask
        dense_mask_from_topk_cuda(
            dense_positions, dense_mask, kernel=self._rerank_topk_kernel,
        )
        nvtx.range_pop()

        # --- Step 2: fused KIVI dequant + dot + per-chunk top-K ---
        nvtx.range_push("cds_fused_kernel")
        out_scores = self._cached_cds_out_scores
        out_indices = self._cached_cds_out_indices
        chunk_min_layer = self._chunk_key_min_gpu[layer_idx][:bs]
        chunk_max_layer = self._chunk_key_max_gpu[layer_idx][:bs]
        partial_chunk_kivi_qk_dense_sparse_cuda(
            chunk_ids=chunk_idx,
            dense_mask=dense_mask,
            packed_K=key_kivi_packed,
            chunk_min=chunk_min_layer,
            chunk_max=chunk_max_layer,
            raw_q=raw_q,
            dense_topk=dense_topk,
            sparse_topk=sparse_topk,
            group_size=chunk,
            out_scores=out_scores,
            out_indices=out_indices,
            kernel=self._kivi_kernel,
        )
        nvtx.range_pop()

        # --- Step 3: fused global top-k + gather + skipped concat ---
        nvtx.range_push("cds_global_topk_fused")
        actual_topk = min(self.final_topk, out_scores.shape[-1])
        if skipped_indices is None:
            skipped_for_kernel = self._cached_cds_final_indices[:bs, :, :0]
            sk_count = 0
        else:
            skipped_for_kernel = skipped_indices
            sk_count = skipped_indices.shape[-1]
        out_buf = self._cached_cds_final_indices[:bs]
        self._fused_topk_ext.float_topk_gather_concat_3d(
            out_scores, out_indices, skipped_for_kernel, out_buf, actual_topk
        )
        
        final_indices = out_buf[:, :, :actual_topk + sk_count].contiguous()
        nvtx.range_pop()

        return final_indices





    def collision_based_topk_batch(
        self,
        query,
        key_kivi_packed=None,
        layer_idx=None,
    ):
        """Quest stage-1 + CDS dense/sparse + KIVI 4-bit final topk.

        Name kept for backward compatibility with `compute()` / timing keys.

        Args:
            query:             [bs, kv_heads, 1, head_dim] raw query
            key_kivi_packed:   [bs, kv_heads, n_chunks_pkd, n_pack, group_size]
                               int32 — KIVI packed K (A-1 chunk-major) sliced
                               on the chunk axis to ``n_chunks_pkd``
            layer_idx:         layer index (used by chunk-metadata lookup)
        Returns:
            topk_indices:      [bs, kv_heads, final_topk] retrieval-zone indices
        """
        assert key_kivi_packed is not None, \
            "KIVI path requires key_kivi_packed"

        # KIVI has no query encode work; fold the trivial shape ops into
        # stage1 chunk picking.
        bs, kv_heads, qlen, head_dim = query.shape
        assert qlen == 1, "Decode 阶段每次只处理 1 个 token"
        query_squeezed = query.squeeze(2)  # [bs, kv_heads, head_dim]

        topk_indices = None
        chunk_idx = None
        # One fused kernel or one nested Quest graph is a single decode bar.
        # The split launches below are coarse_filter, topk_select, fine_rerank.
        if topk_indices is None:
            topk_indices, chunk_idx = self._split_quest_topk(
                query_squeezed, key_kivi_packed, layer_idx, bs,
            )

        if self._allayer_recall_check:
            self._allayer_recall_ablation(
                topk_indices, query_squeezed, layer_idx, tag="Final",
            )
            chunk = self.quest_chunk_size
            offsets = torch.arange(chunk, device=chunk_idx.device, dtype=chunk_idx.dtype)
            coarse_tokens = (
                (chunk_idx * chunk).unsqueeze(-1) + offsets
            ).reshape(bs, kv_heads, -1).contiguous()
            self._allayer_recall_ablation(
                coarse_tokens, query_squeezed, layer_idx, tag="CoarsePool",
            )

        return topk_indices

    def _fused_hierarchical_quest(self, raw_q, key_kivi_packed, layer_idx, bs):
        """Run hierarchical Quest + CDS as one CUDA launch. None means fallback."""
        nk_large = self._cached_hq_nk_large
        nk_small = self._cached_hq_nk_small
        n_large = self._cached_hq_n_large
        chunk = self.quest_chunk_size
        if chunk not in (8, 16) or nk_large <= 0 or nk_small <= 0:
            return None
        scratch = self._cached_cds_out_indices
        if scratch.size(-1) < nk_large + nk_small or scratch.size(-1) < nk_small * chunk:
            return None
        n_dense = self._cached_cds_n_dense
        if n_dense <= 0 or n_dense > nk_small:
            return None
        dense_topk = max(1, min(int(self.cds_dense_topk), chunk))
        sparse_topk = max(1, min(int(self.cds_sparse_topk), chunk))
        actual_topk = min(self.final_topk, nk_small * chunk)
        sk_count = self._chunk_skipped_count[layer_idx]
        if actual_topk + sk_count > self._cached_cds_final_indices.size(-1):
            return None
        skipped = self._chunk_skipped_indices_gpu[layer_idx][:bs, :, :max(sk_count, 1)]
        small_min = self._cached_chunk_min_views[layer_idx]
        scratch_b = scratch[:bs]
        scores_b = self._cached_cds_out_scores[:bs]
        final_b = self._cached_cds_final_indices[:bs]
        large_scores = self._cached_hq_scores_large_view
        sub_scores = self._cached_hq_scores_sub_view
        density = self._cached_cds_density_scores[:bs]
        if any(not tensor.is_contiguous() for tensor in (
            scratch_b, scores_b, final_b, large_scores, sub_scores, density, raw_q,
        )):
            return None
        self._fused_quest_ext.fused_hierarchical_quest(
            raw_q,
            self._cached_hq_large_min_views[layer_idx],
            self._cached_hq_large_max_views[layer_idx],
            self._cached_hq_large_valid_views[layer_idx],
            small_min,
            self._cached_chunk_max_views[layer_idx],
            large_scores,
            sub_scores,
            scratch_b,
            self._cached_hq_chunk_idx[:bs],
            self._chunk_key_centroid_gpu[layer_idx][:bs],
            density,
            self._cached_dense_mask[:bs],
            key_kivi_packed,
            scores_b,
            final_b,
            skipped,
            int(n_large),
            int(nk_large),
            int(self._hq_factor),
            int(small_min.size(2)),
            int(nk_small),
            int(self._cached_quest_n_chunks),
            int(n_dense),
            int(dense_topk),
            int(sparse_topk),
            int(self._chunk_key_centroid_gpu[layer_idx].size(2)),
            int(actual_topk),
            int(sk_count),
        )
        return final_b[:, :, :actual_topk + sk_count]

    def _quest_graph_inputs(self, raw_q, key_kivi_packed, layer_idx, bs):
        nk_large = int(self._cached_hq_nk_large)
        nk_small = int(self._cached_hq_nk_small)
        n_large = int(self._cached_hq_n_large)
        n_dense = int(self._cached_cds_n_dense)
        chunk = int(self.quest_chunk_size)
        if chunk not in (8, 16) or nk_large <= 0 or nk_small <= 0 or n_dense <= 0:
            return None
        if n_dense > nk_small:
            return None
        dense_topk = max(1, min(int(self.cds_dense_topk), chunk))
        sparse_topk = max(1, min(int(self.cds_sparse_topk), chunk))
        actual_topk = min(int(self.final_topk), nk_small * chunk)
        sk_count = int(self._chunk_skipped_count[layer_idx])
        final_buf = self._cached_cds_final_indices[:bs]
        if actual_topk + sk_count > final_buf.size(-1):
            return None
        if sk_count > 0:
            skipped = self._chunk_skipped_indices_gpu[layer_idx][:bs, :, :sk_count]
        else:
            skipped = final_buf[:, :, :0]
        return {
            "n_large": n_large,
            "nk_large": nk_large,
            "nk_small": nk_small,
            "n_dense": n_dense,
            "factor": int(self._hq_factor),
            "n_chunks": int(self._cached_quest_n_chunks),
            "dense_topk": dense_topk,
            "sparse_topk": sparse_topk,
            "actual_topk": actual_topk,
            "sk_count": sk_count,
            "large_min": self._cached_hq_large_min_views[layer_idx],
            "large_max": self._cached_hq_large_max_views[layer_idx],
            "large_valid": self._cached_hq_large_valid_views[layer_idx],
            "small_min": self._cached_chunk_min_views[layer_idx],
            "small_max": self._cached_chunk_max_views[layer_idx],
            "large_scores": self._cached_hq_scores_large_view,
            "sub_scores": self._cached_hq_scores_sub_view,
            "chunk_idx": self._cached_hq_chunk_idx[:bs],
            "centroid": self._chunk_key_centroid_gpu[layer_idx][:bs],
            "density": self._cached_cds_density_scores[:bs],
            "dense_mask": self._cached_dense_mask[:bs],
            "packed_k": key_kivi_packed,
            "out_scores": self._cached_cds_out_scores[:bs],
            "out_indices": self._cached_cds_out_indices[:bs],
            "final_indices": final_buf,
            "skipped": skipped,
        }

    def _quest_graph_hierarchical(self, raw_q, key_kivi_packed, layer_idx, bs):
        """Replay the parallel Quest kernels as one CUDA graph. None falls back."""
        spec = self._quest_graph_inputs(raw_q, key_kivi_packed, layer_idx, bs)
        if spec is None:
            return None
        key = (
            bs,
            spec["n_large"], spec["nk_large"], spec["nk_small"], spec["n_dense"],
            spec["factor"], spec["n_chunks"], spec["dense_topk"], spec["sparse_topk"],
            spec["actual_topk"], spec["sk_count"],
            int(spec["packed_k"].data_ptr()), int(spec["packed_k"].size(2)),
            int(spec["large_min"].data_ptr()), int(spec["small_min"].data_ptr()),
            int(spec["centroid"].data_ptr()), int(spec["large_scores"].data_ptr()),
        )
        slot = self._quest_graphs.get(layer_idx)
        if slot is None or slot["key"] != key:
            try:
                slot = self._capture_quest_graph(raw_q, spec, key)
            except Exception as exc:
                if not self._quest_graph_broken:
                    print(f"[Quest] CUDA graph capture failed ({exc}); using fused/split kernels", flush=True)
                self._quest_graph_broken = True
                self._quest_graphs.clear()
                return None
            self._quest_graphs[layer_idx] = slot
            if len(self._quest_graphs) == 1:
                print("[Quest] CUDA graph replay enabled", flush=True)
            return slot["final"]
        slot["q"].copy_(raw_q)
        slot["graph"].replay()
        return slot["final"]

    def _capture_quest_graph(self, raw_q, spec, key):
        from sglang_zoomkv.kernels.quest.quest_pipeline import run_hierarchical_quest

        static_q = torch.empty_like(raw_q)
        static_q.copy_(raw_q)
        keepalive: list = [static_q, spec["skipped"], spec["packed_k"], spec["final_indices"]]
        kwargs = dict(spec)
        kwargs["raw_q"] = static_q
        kwargs["quest_kernel"] = self._quest_kernel
        kwargs["rerank_kernel"] = self._rerank_topk_kernel
        kwargs["kivi_kernel"] = self._kivi_kernel
        kwargs["keepalive"] = keepalive

        if self._quest_graph_pool is None:
            self._quest_graph_pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            run_hierarchical_quest(**kwargs)
            with torch.cuda.graph(graph, pool=self._quest_graph_pool):
                final = run_hierarchical_quest(**kwargs)
        torch.cuda.current_stream().wait_stream(side)
        keepalive.append(final)
        return {"key": key, "graph": graph, "q": static_q, "final": final, "keep": keepalive}

    def _ensure_sparse_workspace(self, bs, hq):
        splits = int(self._sparse_splits)
        if (
            self._sparse_out is not None
            and self._sparse_partial_m is not None
            and self._sparse_out.size(0) >= bs
            and self._sparse_out.size(1) == hq
            and self._sparse_partial_m.size(2) == splits
        ):
            return
        if self._workspace_frozen or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "ZoomKV sparse workspace cannot be resized inside or after CUDA graph capture"
            )
        device = self.device
        cap = max(int(bs), int(self.batch_size))
        self._sparse_partial_m = torch.empty(cap, hq, splits, device=device)
        self._sparse_partial_l = torch.empty(cap, hq, splits, device=device)
        self._sparse_partial_acc = torch.empty(cap, hq, splits, self.head_dim, device=device)
        self._sparse_out = torch.empty(cap, hq, self.head_dim, dtype=self.dtype, device=device)
        self._sparse_bounds = torch.empty(2, dtype=torch.int32, device=device)
        self._sparse_bounds_cpu = torch.empty(2, dtype=torch.int32, pin_memory=True)
        self._sparse_bounds_cached = None
        self._sparse_extra_count = torch.empty(1, dtype=torch.int32, device=device)
        self._sparse_extra_count_cpu = torch.empty(1, dtype=torch.int32, pin_memory=True)
        self._sparse_extra_cached = None
        self._sparse_graphs.clear()
        self._offload_sparse_graphs.clear()

    def _stage_sparse_bounds(self, local_start, local_end):
        # Same window for every layer of this token. Rewriting the pinned
        # staging buffer each layer serializes behind the in-flight H2D.
        key = (int(local_start), int(local_end))
        if key == self._sparse_bounds_cached:
            return
        self._sparse_bounds_cached = key
        cpu = self._sparse_bounds_cpu
        cpu[0] = key[0]
        cpu[1] = key[1]
        self._sparse_bounds.copy_(cpu, non_blocking=True)

    def _stage_extra_count(self, n_extra):
        n_extra = int(n_extra)
        if n_extra == self._sparse_extra_cached:
            return
        self._sparse_extra_cached = n_extra
        cpu = self._sparse_extra_count_cpu
        cpu[0] = n_extra
        self._sparse_extra_count.copy_(cpu, non_blocking=True)

    def _fill_avg_query(self, full_q, avg_q):
        grouped = full_q.view(
            int(full_q.size(0)), self.kv_head, self.group_size, self.head_dim
        )
        torch.mean(grouped, dim=2, out=avg_q)

    def _launch_sparse_attn(
        self, q, k, v, indices, extra_k=None, extra_v=None, extra_count=None, paged=None
    ):
        from sglang_zoomkv.kernels.sparse_attn.sparse_decode_attn_ops import (
            sparse_decode_attention,
        )

        bs = int(q.size(0))
        self._ensure_sparse_workspace(bs, int(q.size(1)))
        use_pool = self._use_pool_kv if paged is None else bool(paged)
        if use_pool:
            k_arg, v_arg = k, v
            slot_table = self._slot_table
            row_ids = self._pool_row_ids[:bs]
        else:
            k_arg, v_arg = k[:bs], v[:bs]
            slot_table = None
            row_ids = None
        count = self._sparse_extra_count if extra_count is None else extra_count
        return sparse_decode_attention(
            q,
            k_arg,
            v_arg,
            indices,
            self._sparse_bounds,
            self._sparse_partial_m[:bs],
            self._sparse_partial_l[:bs],
            self._sparse_partial_acc[:bs],
            self._sparse_out[:bs],
            sink_tokens=int(self.sink_size),
            index_bias=int(self.sink_size),
            splits=int(self._sparse_splits),
            kernel=self._sparse_ext,
            extra_k=None if extra_k is None else extra_k[:bs],
            extra_v=None if extra_v is None else extra_v[:bs],
            extra_count=None if extra_k is None else count,
            slot_table=slot_table,
            row_ids=row_ids,
        )

    def _empty_sparse_indices(self, bs):
        cached = self._sparse_empty_indices
        if cached is None or cached.size(0) < bs or cached.size(1) != self.kv_head:
            self._sparse_empty_indices = torch.empty(
                max(int(bs), int(self.batch_size)),
                self.kv_head,
                0,
                dtype=torch.int64,
                device=self.device,
            )
        return self._sparse_empty_indices[:bs]

    def _offload_sparse_attention(self, queries, layer_idx, bs, region_info, n_topk):
        """Attend sink/local from the packed GPU cache plus the gathered retrieval tokens."""
        if self.head_dim not in (128, 256) or queries.dtype != torch.bfloat16:
            return None
        full_q = queries[:bs, 0]
        if not full_q.is_contiguous():
            full_q = full_q.contiguous()
        k = self.unified_keys_gpu[layer_idx]
        v = self.unified_values_gpu[layer_idx]
        if not k.is_contiguous() or not v.is_contiguous():
            return None
        local_len = int(region_info["local_len"])
        local_start = int(self.sink_size)
        local_end = local_start + local_len
        extra_k = self._topk_keys_buffer
        extra_v = self._topk_values_buffer
        try:
            if not self._offload_sparse_broken and not torch.cuda.is_current_stream_capturing():
                out = self._offload_sparse_graph(
                    full_q, k, v, extra_k, extra_v, layer_idx, bs, local_start, local_end, n_topk,
                )
                if out is not None:
                    return out.unsqueeze(1)
            self._ensure_sparse_workspace(bs, int(full_q.size(1)))
            self._stage_sparse_bounds(local_start, local_end)
            self._stage_extra_count(n_topk)
            self._launch_sparse_attn(
                full_q, k, v, self._empty_sparse_indices(bs), extra_k, extra_v,
            )
        except Exception as exc:
            if not self._offload_sparse_broken:
                print(f"[Quest] offload sparse attention failed ({exc}); using cat", flush=True)
            self._offload_sparse_broken = True
            return None
        return self._sparse_out[:bs].unsqueeze(1)

    def _offload_sparse_graph(
        self, full_q, k, v, extra_k, extra_v, layer_idx, bs, local_start, local_end, n_topk,
    ):
        self._ensure_sparse_workspace(bs, int(full_q.size(1)))
        self._stage_sparse_bounds(local_start, local_end)
        self._stage_extra_count(n_topk)
        slot = self._offload_sparse_graphs.get(layer_idx)
        hq = int(full_q.size(1))
        if (
            slot is not None
            and slot["bs"] == bs
            and slot["hq"] == hq
            and slot["k_ptr"] == int(k.data_ptr())
            and slot["v_ptr"] == int(v.data_ptr())
            and slot["extra_ptr"] == int(extra_k.data_ptr())
        ):
            slot["full_q"].copy_(full_q)
            slot["graph"].replay()
            return slot["out"]
        try:
            slot = self._capture_offload_sparse_graph(
                full_q, k, v, extra_k, extra_v, bs,
            )
        except Exception as exc:
            if not self._offload_sparse_broken:
                print(
                    f"[Quest] offload sparse graph failed ({exc}); using a direct launch",
                    flush=True,
                )
            self._offload_sparse_broken = True
            self._offload_sparse_graphs.clear()
            return None
        self._offload_sparse_graphs[layer_idx] = slot
        if len(self._offload_sparse_graphs) == 1:
            print("[Quest] offload sparse attention CUDA graph enabled", flush=True)
        return slot["out"]

    def _capture_offload_sparse_graph(self, full_q, k, v, extra_k, extra_v, bs):
        static_full = full_q.detach().clone()
        indices = self._empty_sparse_indices(bs)
        keepalive = [static_full, k, v, extra_k, extra_v, indices, self._sparse_extra_count, self._sparse_bounds]

        def body():
            self._launch_sparse_attn(static_full, k, v, indices, extra_k, extra_v)

        if self._quest_graph_pool is None:
            self._quest_graph_pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            body()
            with torch.cuda.graph(graph, pool=self._quest_graph_pool):
                body()
        torch.cuda.current_stream().wait_stream(side)
        keepalive.append(self._sparse_out[:bs])
        return {
            "graph": graph,
            "full_q": static_full,
            "out": self._sparse_out[:bs],
            "keep": keepalive,
            "bs": bs,
            "hq": int(full_q.size(1)),
            "k_ptr": int(k.data_ptr()),
            "v_ptr": int(v.data_ptr()),
            "extra_ptr": int(extra_k.data_ptr()),
        }

    def _gpu_sparse_attention(
        self, queries, query_batch, key_kivi_packed, layer_idx, bs, region_info, force_pool=False,
    ):
        """Quest plus index attention in one graph replay. None keeps the gather path."""
        if self._sparse_ext is None or self.head_dim not in (128, 256):
            return None
        if queries.dtype != torch.bfloat16:
            return None
        full_q = queries[:bs, 0]
        use_pool = bool(force_pool or self._use_pool_kv)
        if use_pool:
            if self._sglang_kv_pool is None or self._slot_table is None:
                return None
            k = self._sglang_kv_pool.get_key_buffer(layer_idx)
            v = self._sglang_kv_pool.get_value_buffer(layer_idx)
            if k.dim() != 3 or v.shape != k.shape:
                raise RuntimeError(
                    "ZoomKV single-cache path expects NHD KV [tokens, heads, dim], "
                    f"got K {tuple(k.shape)} V {tuple(v.shape)}"
                )
        else:
            k = self.unified_keys_gpu[layer_idx]
            v = self.unified_values_gpu[layer_idx]
        if not k.is_contiguous() or not v.is_contiguous():
            return None
        local_start = region_info["local_start"]
        local_end = region_info["attn_local_end"]
        if (
            not force_pool
            and self._quest_graph_enabled
            and not self._quest_graph_broken
            and not self._sparse_graph_broken
            and self._quest_kernel is not None
            and not torch.cuda.is_current_stream_capturing()
        ):
            out = self._sparse_quest_graph(
                full_q, key_kivi_packed, k, v, layer_idx, bs, local_start, local_end,
            )
            if out is not None:
                return out.unsqueeze(1)
        if query_batch is None:
            avg_q = full_q.reshape(
                bs, self.kv_head, self.group_size, self.head_dim
            ).mean(dim=2)
        else:
            avg_q = query_batch[:bs].squeeze(2)
        _chunk_idx = None
        indices = None
        if self._fused_quest_ext is not None:
            indices = self._fused_hierarchical_quest(avg_q, key_kivi_packed, layer_idx, bs)
        if indices is None:
            indices, _chunk_idx = self._split_quest_topk(
                avg_q, key_kivi_packed, layer_idx, bs,
            )
        try:
            self._ensure_sparse_workspace(bs, int(full_q.size(1)))
            self._stage_sparse_bounds(local_start, local_end)
            self._launch_sparse_attn(
                full_q,
                k,
                v,
                indices,
                paged=True if force_pool and not self._use_pool_kv else None,
            )
        except Exception as exc:
            print(f"[Quest] sparse attention failed ({exc}); using gather", flush=True)
            self._sparse_attn_enabled = False
            return None
        return self._sparse_out[:bs].unsqueeze(1)

    def _sparse_slot_matches(self, slot, packed, k, v, full_q, bs, layer_idx):
        views = self._cached_hq_large_min_views
        if not views:
            return False
        large = views[layer_idx]
        return (
            slot["bs"] == bs
            and slot["hq"] == int(full_q.size(1))
            and slot["packed_n"] == int(packed.size(2))
            and slot["packed_ptr"] == int(packed.data_ptr())
            and slot["large_ptr"] == int(large.data_ptr())
            and slot["k_ptr"] == int(k.data_ptr())
            and slot["v_ptr"] == int(v.data_ptr())
        )

    def _sparse_quest_graph(
        self, full_q, packed, k, v, layer_idx, bs, local_start, local_end,
    ):
        slot = self._sparse_graphs.get(layer_idx)
        if slot is not None and self._sparse_slot_matches(
            slot, packed, k, v, full_q, bs, layer_idx,
        ):
            slot["full_q"].copy_(full_q)
            self._stage_sparse_bounds(local_start, local_end)
            slot["graph"].replay()
            return slot["out"]
        spec = self._quest_graph_inputs(full_q, packed, layer_idx, bs)
        if spec is None:
            return None
        self._ensure_sparse_workspace(bs, int(full_q.size(1)))
        self._stage_sparse_bounds(local_start, local_end)
        key = (
            bs,
            spec["n_large"], spec["nk_large"], spec["nk_small"], spec["n_dense"],
            spec["factor"], spec["n_chunks"], spec["dense_topk"], spec["sparse_topk"],
            spec["actual_topk"], spec["sk_count"],
            int(spec["packed_k"].data_ptr()), int(spec["packed_k"].size(2)),
            int(spec["large_min"].data_ptr()), int(spec["small_min"].data_ptr()),
            int(spec["centroid"].data_ptr()), int(spec["large_scores"].data_ptr()),
            int(k.data_ptr()), int(v.data_ptr()), int(full_q.size(1)),
            int(self.sink_size),
        )
        slot = self._sparse_graphs.get(layer_idx)
        if slot is None or slot["key"] != key:
            try:
                slot = self._capture_sparse_quest_graph(full_q, spec, k, v, key)
            except Exception as exc:
                if not self._sparse_graph_broken:
                    print(
                        f"[Quest] sparse attention graph failed ({exc}); using separate launches",
                        flush=True,
                    )
                self._sparse_graph_broken = True
                self._sparse_graphs.clear()
                return None
            self._sparse_graphs[layer_idx] = slot
            if not self._sparse_announced:
                print("[Quest] sparse attention CUDA graph enabled", flush=True)
                self._sparse_announced = True
            return slot["out"]
        slot["full_q"].copy_(full_q)
        self._stage_sparse_bounds(local_start, local_end)
        slot["graph"].replay()
        return slot["out"]

    def _capture_sparse_quest_graph(self, full_q, spec, k, v, key):
        from sglang_zoomkv.kernels.quest.quest_pipeline import run_hierarchical_quest

        static_full = full_q.contiguous()
        if static_full.data_ptr() == full_q.data_ptr():
            static_full = full_q.detach().clone()
        static_q = torch.empty(
            int(full_q.size(0)),
            self.kv_head,
            self.head_dim,
            dtype=full_q.dtype,
            device=full_q.device,
        )
        self._fill_avg_query(static_full, static_q)
        keepalive = [static_q, static_full, spec["skipped"], spec["packed_k"], spec["final_indices"], k, v]
        kwargs = dict(spec)
        kwargs["raw_q"] = static_q
        kwargs["quest_kernel"] = self._quest_kernel
        kwargs["rerank_kernel"] = self._rerank_topk_kernel
        kwargs["kivi_kernel"] = self._kivi_kernel
        kwargs["keepalive"] = keepalive
        bs = int(full_q.size(0))

        def body():
            self._fill_avg_query(static_full, static_q)
            indices = run_hierarchical_quest(**kwargs)
            self._launch_sparse_attn(static_full, k, v, indices)
            return indices

        if self._quest_graph_pool is None:
            self._quest_graph_pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            body()
            with torch.cuda.graph(graph, pool=self._quest_graph_pool):
                final = body()
        torch.cuda.current_stream().wait_stream(side)
        keepalive.append(final)
        keepalive.append(self._sparse_out[:bs])
        return {
            "key": key,
            "graph": graph,
            "q": static_q,
            "full_q": static_full,
            "out": self._sparse_out[:bs],
            "keep": keepalive,
            "bs": bs,
            "hq": int(full_q.size(1)),
            "packed_n": int(spec["packed_k"].size(2)),
            "packed_ptr": int(spec["packed_k"].data_ptr()),
            "large_ptr": int(spec["large_min"].data_ptr()),
            "k_ptr": int(k.data_ptr()),
            "v_ptr": int(v.data_ptr()),
        }

    def _split_quest_topk(self, query_squeezed, key_kivi_packed, layer_idx, bs):
        nvtx.range_push("stage1_coarse")
        sk_count = self._chunk_skipped_count[layer_idx]
        skipped = None
        if sk_count > 0:
            # _chunk_skipped_indices_gpu is filled by torch.arange over the
            # in-range token span (see the ``_compute_and_store_chunk_minmax``
            # tail-skip path), so values are already in [0, kv_len).  No
            # defensive clamp needed.
            skipped = self._chunk_skipped_indices_gpu[layer_idx][:bs, :, :sk_count]

        if self._use_hierarchical_quest:
            chunk_idx, _nk = self._quest_pick_coarse_indices_hierarchical(
                raw_q=query_squeezed,
                layer_idx=layer_idx,
            )
        else:
            chunk_idx, _nk = self._quest_pick_coarse_indices_flat(
                raw_q=query_squeezed,
                layer_idx=layer_idx,
            )
        nvtx.range_pop()  # stage1_chunk_pick


        nvtx.range_push("stage2_rerank")
        topk_indices = self._cds_dense_sparse_select(
            chunk_idx, bs,
            key_kivi_packed,
            query_squeezed, layer_idx,
            skipped_indices=skipped,
        )
        nvtx.range_pop()  # stage2_cds
        return topk_indices, chunk_idx




    def _allayer_recall_ablation(
        self,
        indices: torch.Tensor,
        raw_q: torch.Tensor,
        layer_idx: int,
        tag: str = "Final",
    ):
        """Accumulate exact q·k recall across all layers/heads for one decode step."""
        if layer_idx == 0 and tag == "Final":
            self._allayer_recall_step += 1
            self._allayer_recall_accum = {}
        if self._allayer_recall_step % self._allayer_recall_log_every != 0:
            return

        K = int(self._allayer_recall_k)
        with torch.no_grad():
            if not (self.enable_offload and self.unified_keys_cpu is not None and self.cpu_valid_end is not None):
                return
            cpu_len = int(self.cpu_valid_end[layer_idx])
            if cpu_len < K:
                return
            keys_layer = self._dequant_cpu_span(layer_idx, cpu_len, "k").float()
            q_fp32 = raw_q.to(torch.float32)

            if tag not in self._allayer_recall_accum:
                self._allayer_recall_accum[tag] = [0, 0, 0]  # hit, total, kv_len

            kv_heads = indices.shape[1]
            for h in range(kv_heads):
                exact_scores = torch.matmul(keys_layer[0, h, :cpu_len, :], q_fp32[0, h])
                _, true_topk = torch.topk(exact_scores, k=K, dim=-1)
                true_set = set(true_topk.cpu().tolist())
                cand_set = set(indices[0, h].cpu().tolist())
                self._allayer_recall_accum[tag][0] += len(cand_set & true_set)
                self._allayer_recall_accum[tag][1] += K
                self._allayer_recall_accum[tag][2] = cpu_len

            if layer_idx == self.layer_num - 1 and tag == "CoarsePool":
                self._print_allayer_recall_summary()
                self._allayer_recall_accum = {}


    def _print_allayer_recall_summary(self):
        parts = []
        kv_len = 0
        for tag in ("Final", "CoarsePool"):
            if tag not in self._allayer_recall_accum:
                continue
            hit, total, kv = self._allayer_recall_accum[tag]
            recall = hit / total * 100 if total > 0 else 0
            parts.append(f"{tag} R@{self._allayer_recall_k}={recall:.1f}%")
            kv_len = max(kv_len, kv)
        if parts:
            print(
                f"[Pipeline Recall] step={self._allayer_recall_step} kv={kv_len} "
                f"(avg all layers/heads) | " + " | ".join(parts),
                flush=True,
            )


    def sync(self, layer_idx, batch_idx):
        """
        GPU版本:数据已经在prefill阶段直接存储到GPU,无需额外同步
        """
        pass  # GPU版本中所有数据已在prefill时存储到GPU,无需额外操作











# Historical class name retained for integrations that import ``polar_cache``.
polar_cache = ZoomKVCache

__all__ = ["ZoomKVCache", "polar_cache"]
