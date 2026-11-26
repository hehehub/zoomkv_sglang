"""SGLang attention backend bridge for the ZoomKV cache.

The first prefill layers stay on the dense backend. Later layers use ZoomKV
sparse prefill. Each layer's retrieval KV is queued on the copy stream as
soon as that layer's K/V exist, so the transfer overlaps the next layer.
Decode supports TP=1/2 and a lockstep batch of up to 64 equal-length requests.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.cuda.nvtx as nvtx

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _decode_cuda_graph_backend(server_args: Any) -> str | None:
    """Return the resolved decode CUDA-graph backend, if the process has one."""
    config = getattr(server_args, "cuda_graph_config", None)
    decode = getattr(config, "decode", None)
    backend = getattr(decode, "backend", None)
    if backend:
        return str(backend)
    selected = getattr(server_args, "cuda_graph_backend_decode", None)
    if selected:
        return str(selected)
    if getattr(server_args, "disable_cuda_graph", False) or getattr(
        server_args, "disable_decode_cuda_graph", False
    ):
        return "disabled"
    return None


# req_pool_idx -> retrieval span already returned to the GPU page allocator.
_released_retrieval_spans: dict[int, tuple[int, int]] = {}


def note_released_retrieval(req_pool_idx: int, start: int, end: int) -> None:
    _released_retrieval_spans[int(req_pool_idx)] = (int(start), int(end))


def should_merge_parked_slots(slots: tuple[int, ...], parked_slots) -> bool:
    """Join parked prefills only when decode is actually a multi-request batch."""
    if len(slots) < 2:
        return False
    parked = {int(slot) for slot in parked_slots}
    return all(int(slot) in parked for slot in slots)


def fill_to_decode_perm(fill_order, decode_slots):
    """Map cache rows (prefill fill order) onto the decode batch order.

    gather[cache_row] is the decode-batch index that belongs in that row.
    scatter[decode_index] is the cache row for that decode request.
    Identity when the two orders already match.
    """
    fill = tuple(int(slot) for slot in fill_order)
    decode = tuple(int(slot) for slot in decode_slots)
    if fill == decode:
        return None, None
    if set(fill) != set(decode) or len(fill) != len(decode):
        raise RuntimeError(
            f"ZoomKV fill order {fill} cannot cover decode slots {decode}"
        )
    decode_pos = {slot: idx for idx, slot in enumerate(decode)}
    fill_pos = {slot: idx for idx, slot in enumerate(fill)}
    gather = [decode_pos[slot] for slot in fill]
    scatter = [fill_pos[slot] for slot in decode]
    return gather, scatter


def install_offload_page_release() -> None:
    """Teach chunk-cache completion to skip pages ZoomKV already freed."""
    from sglang.srt.mem_cache.chunk_cache import ChunkCache

    current = ChunkCache.cache_finished_req
    if getattr(current, "_zoomkv_wrapped", False):
        return

    def cache_finished_req(self, req, is_insert=True, *, kv_len_to_handle):
        span = _released_retrieval_spans.pop(getattr(req, "req_pool_idx", None), None)
        if span is None:
            return current(self, req, is_insert, kv_len_to_handle=kv_len_to_handle)
        start, end = span
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len_to_handle
        ]
        protected = int(getattr(req, "cache_protected_len", 0))
        start = max(int(start), protected)
        end = min(int(end), int(kv_len_to_handle))
        parts = []
        if start > protected:
            parts.append(kv_indices[protected:start])
        if end < kv_len_to_handle:
            parts.append(kv_indices[end:kv_len_to_handle])
        if not parts:
            return
        free_index = parts[0] if len(parts) == 1 else torch.cat(parts)
        self.token_to_kv_pool_allocator.free(free_index)

    cache_finished_req._zoomkv_wrapped = True
    ChunkCache.cache_finished_req = cache_finished_req


class ZoomKVAttentionBackend(AttentionBackend):
    """Decode-only ZoomKV backend with an ordinary SGLang prefill fallback."""

    supports_ragged_verify_graph = False
    supports_full_cuda_graph_chunked_prefix = False

    def __init__(self, model_runner: ModelRunner, fallback_backend: AttentionBackend):
        super().__init__()
        self.runner = model_runner
        self.fallback = fallback_backend
        self.config = self._load_config()
        print(
            f"[ZoomKV] final_topk={self.config.final_topk} offload={self.config.offload}",
            flush=True,
        )
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.device = model_runner.device
        self._cache = None
        self._request_slots: tuple[int, ...] | None = None
        self._batch_size = 0
        self._initialized_layers: set[int] = set()
        self._retrieval_pages_released = False
        self._cohort_row = 0
        self._cohort_slot_order: list[int] = []
        self._cohort_gather = None
        self._cohort_scatter = None
        self._kv_pool_compacted = False
        self._joint_decode_prepared = False
        self._zoomkv_full_graph = False
        self._zoomkv_graph_capture = False
        self._zoomkv_graph_failed = False
        self._zoomkv_graphs_ready = False
        self._zoomkv_captured_bs = None

        self.prefill_attention_backend_str = getattr(
            fallback_backend, "prefill_attention_backend_str", None
        )
        self.decode_attention_backend_str = "zoomkv"
        self._validate_runner()
        self._zoomkv_full_graph = (
            _decode_cuda_graph_backend(self.runner.server_args) == "full"
        )

    @staticmethod
    def _load_config():
        from sglang_zoomkv.config import ZoomKVConfig

        return ZoomKVConfig.from_env()

    def _validate_runner(self) -> None:
        args = self.runner.server_args
        model_config = self.runner.model_config
        errors = []
        if getattr(args, "tp_size", 1) not in (1, 2):
            errors.append("tp_size must be 1 or 2")
        if getattr(args, "page_size", 1) != 1:
            errors.append("page_size must be 1")
        if getattr(args, "chunked_prefill_size", -1) != -1:
            errors.append("--chunked-prefill-size -1 is required")
        if not getattr(args, "disable_radix_cache", False):
            errors.append("--disable-radix-cache is required")
        if getattr(model_config, "head_dim", None) not in (128, 256):
            errors.append("model head_dim must be 128 or 256")
        model_dtype = getattr(model_config, "dtype", None)
        if str(model_dtype).lower() not in {
            "torch.bfloat16",
            "bfloat16",
            "bf16",
        }:
            errors.append("model dtype must be bfloat16")
        if getattr(model_config, "attention_arch", None) is not None:
            arch_name = str(model_config.attention_arch).lower()
            if "mla" in arch_name:
                errors.append("MLA models are not supported")
        if errors:
            raise ValueError("ZoomKV MVP configuration error: " + "; ".join(errors))

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        return self.fallback.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # The full graph reads Quest bounds and pool row ids from buffers this
        # method refreshes. It runs before capture and again before every
        # replay, so the recorded kernels see the current token.
        if self._zoomkv_full_graph:
            self._stage_graph_inputs(forward_batch)
            return None
        # Capture plans the FlashInfer fallback at the seq-len fill value so
        # the graph buffers exist. Replaying that plan against the real batch
        # writes a 4096-token page table through a wrapper planned for length
        # 1. The fault is asynchronous and shows up in the next KV store.
        # ZoomKV's own decode path does not read this wrapper.
        if not in_capture:
            return None
        return self.fallback.init_forward_metadata_out_graph(
            forward_batch, in_capture=True
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        if self._zoomkv_full_graph:
            return None
        return self.fallback.init_forward_metadata_in_graph(forward_batch)

    def _stage_graph_inputs(self, forward_batch: ForwardBatch) -> None:
        cache = self._cache
        if cache is None or not cache.graph_spec_ready():
            return
        cpu_lens = getattr(forward_batch, "seq_lens_cpu", None)
        row_ids = getattr(forward_batch, "req_pool_indices", None)
        if cpu_lens is None or row_ids is None:
            return
        cache.stage_for_replay(
            self.token_to_kv_pool,
            self.req_to_token_pool.req_to_token,
            self._align_to_cache_rows(row_ids[: int(cache.batch_size)]),
            int(cpu_lens[0]),
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.fallback.get_cuda_graph_seq_len_fill_value()

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # Eager ragged batches still use the FlashInfer fallback. The full
        # graph itself does not replay that plan.
        self.fallback.init_cuda_graph_state(max_bs, max_num_tokens)
        return None

    def prepare_prefill_offload(self, forward_batch: ForwardBatch) -> None:
        """Allocate the sidecar before layer 0.

        Allocating it inside the first layer stacks the index on top of that
        layer's activations and has killed the rank. The per-layer copy still
        runs from ``forward_extend`` and overlaps the next attention.
        """
        if not bool(self.config.enable_offload):
            return
        if not bool(getattr(self.config, "prefill_overlap", True)):
            return
        if (
            torch.cuda.is_current_stream_capturing()
            or getattr(self, "_zoomkv_graph_capture", False)
        ):
            return
        batch_size = int(getattr(forward_batch, "batch_size", 0) or 0)
        if batch_size <= 0:
            return
        layers = self._radix_layers()
        if not layers:
            return
        layer = layers[0]
        slots, seq_lens = self._batch_state(forward_batch, int(layer.layer_id))
        if len(set(seq_lens)) != 1:
            return
        seq_len = int(seq_lens[0])
        if seq_len <= int(self.config.sink_size) + int(self.config.local_size):
            return
        cached_len = (
            int(self._cache.current_seq_len[0]) if self._cache is not None else None
        )
        stale = (
            self._cache is None
            or self._request_slots != slots
            or cached_len != seq_len
        )
        if not stale:
            return
        dtype = self.token_to_kv_pool.get_key_buffer(int(layer.layer_id)).dtype
        self._reset_for_batch(slots, seq_len, layer, dtype, prefer_cohort=True)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs: Any,
    ):
        if self._sparse_prefill_this_layer(q, k, v, layer, forward_batch):
            out = self._forward_sparse_prefill(
                q, k, v, layer, forward_batch, save_kv_cache
            )
        else:
            out = self.fallback.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        # K/V for this layer are ready. Queue the CPU copy on the copy stream
        # and return without waiting, so the next layer's attention overlaps it.
        if (
            bool(self.config.enable_offload)
            and bool(getattr(self.config, "prefill_overlap", True))
            and not torch.cuda.is_current_stream_capturing()
            and not getattr(self, "_zoomkv_graph_capture", False)
        ):
            try:
                self._offload_layer_now(k, v, layer, forward_batch)
            except Exception as exc:
                print(
                    f"[ZoomKV] layer {int(layer.layer_id)} offload deferred ({exc})",
                    flush=True,
                )
        return out

    def _sparse_prefill_this_layer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> bool:
        dense_layers = int(getattr(self.config, "prefill_dense_layers", 4))
        if int(layer.layer_id) < dense_layers:
            return False
        if torch.cuda.is_current_stream_capturing() or getattr(self, "_zoomkv_graph_capture", False):
            return False
        if k is None or v is None or q is None:
            return False
        try:
            _slots, seq_lens = self._batch_state(forward_batch, int(layer.layer_id))
        except Exception:
            return False
        if len(set(seq_lens)) != 1:
            return False
        seq_len = int(seq_lens[0])
        keep_dense = max(
            int(self.config.full_attention_threshold),
            int(self.config.sink_size) + int(self.config.local_size),
        )
        if seq_len <= keep_dense:
            return False
        if int(layer.v_head_dim) != int(layer.qk_head_dim):
            return False
        batch_size = len(seq_lens)
        try:
            rows = q.reshape(-1, int(layer.tp_q_head_num), int(layer.qk_head_dim)).shape[0]
        except RuntimeError:
            return False
        if rows != batch_size * seq_len:
            self._warn_offload_once(
                "sparse prefill skipped: "
                f"q rows {rows} != batch {batch_size} * seq {seq_len}"
            )
            return False
        if not getattr(self, "_sparse_prefill_announced", False):
            self._sparse_prefill_announced = True
            print(
                "[ZoomKV] prefill layers "
                f"[0, {dense_layers}) stay dense; later layers use ZoomKV "
                f"sparse prefill (block={int(self.config.prefill_block)}, "
                f"sink={int(self.config.sink_size)}, "
                f"local={int(self.config.local_size)}, "
                f"topk={int(self.config.final_topk)})",
                flush=True,
            )
        return True

    def _forward_sparse_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ) -> torch.Tensor:
        from sglang_zoomkv.attention.prefill_sparse import sparse_causal_prefill

        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(forward_batch.out_cache_loc),
                k,
                v,
            )
        _slots, seq_lens = self._batch_state(forward_batch, int(layer.layer_id))
        batch_size = len(seq_lens)
        seq_len = int(seq_lens[0])
        q_heads = int(layer.tp_q_head_num)
        kv_heads = int(layer.tp_k_head_num)
        dim = int(layer.qk_head_dim)
        scale = float(getattr(layer, "scaling", None) or (dim ** -0.5))
        query = q.reshape(batch_size, seq_len, q_heads, dim)
        keys = k.reshape(batch_size, seq_len, kv_heads, dim)
        values = v.reshape(batch_size, seq_len, kv_heads, int(layer.v_head_dim))
        nvtx.range_push("sparse_prefill")
        try:
            output = sparse_causal_prefill(
                query,
                keys,
                values,
                sink=int(self.config.sink_size),
                local=int(self.config.local_size),
                topk=int(self.config.final_topk),
                block=int(self.config.prefill_block),
                scale=scale,
            )
        finally:
            nvtx.range_pop()
        return output.reshape(q.shape[0], q_heads * int(layer.v_head_dim))

    def _warn_offload_once(self, reason: str) -> None:
        seen = getattr(self, "_offload_warnings", None)
        if seen is None:
            seen = set()
            self._offload_warnings = seen
        if reason in seen:
            return
        seen.add(reason)
        print(f"[ZoomKV] prefill offload skipped ({reason})", flush=True)

    def _prefill_kv_tensors(
        self,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        batch_size: int,
        seq_len: int,
    ):
        """Prefer the prefill K/V. Fall back to the pool the fallback just wrote."""
        if (
            k is not None
            and v is not None
            and k.dim() == 3
            and int(k.shape[0]) == batch_size * seq_len
        ):
            keys = k.reshape(
                batch_size, seq_len, layer.tp_k_head_num, layer.qk_head_dim
            )
            values = v.reshape(
                batch_size, seq_len, layer.tp_v_head_num, layer.v_head_dim
            )
            return keys, values
        self._warn_offload_once(
            "prefill K/V is not the full sequence; copying this layer from the KV pool"
        )
        request_rows = forward_batch.req_pool_indices[:batch_size].long()
        token_indices = self.req_to_token_pool.req_to_token[
            request_rows, :seq_len
        ].long()
        keys = self.token_to_kv_pool.get_key_buffer(int(layer.layer_id))[token_indices]
        values = self.token_to_kv_pool.get_value_buffer(int(layer.layer_id))[
            token_indices
        ]
        keys = keys.reshape(
            batch_size, seq_len, layer.tp_k_head_num, layer.qk_head_dim
        )
        values = values.reshape(
            batch_size, seq_len, layer.tp_v_head_num, layer.v_head_dim
        )
        return keys, values

    def _offload_layer_now(self, k, v, layer: RadixAttention, forward_batch: ForwardBatch) -> None:
        """Offload one layer's retrieval KV as soon as prefill produced it."""
        if (
            torch.cuda.is_current_stream_capturing()
            or getattr(self, "_zoomkv_graph_capture", False)
        ):
            return
        batch_size = int(forward_batch.batch_size)
        if batch_size <= 0:
            return
        layer_id = int(layer.layer_id)
        slots, seq_lens = self._batch_state(forward_batch, layer_id)
        if len(set(seq_lens)) != 1:
            self._warn_offload_once("ragged prefill sequence lengths")
            return
        seq_len = int(seq_lens[0])
        if seq_len <= int(self.config.sink_size) + int(self.config.local_size):
            return
        cached_len = (
            int(self._cache.current_seq_len[0]) if self._cache is not None else None
        )
        stale = self._cache is None or self._request_slots != slots or (
            layer_id == 0 and cached_len != seq_len
        )
        if stale:
            if layer_id != 0:
                return
            nvtx.range_push("sidecar_reset")
            try:
                dtype = getattr(k, "dtype", None)
                if dtype is None:
                    dtype = self.token_to_kv_pool.get_key_buffer(layer_id).dtype
                self._reset_for_batch(
                    slots, seq_len, layer, dtype, prefer_cohort=True
                )
            finally:
                nvtx.range_pop()
                pass
        if layer_id in self._initialized_layers:
            return
        keys, values = self._prefill_kv_tensors(
            k, v, layer, forward_batch, batch_size, seq_len
        )
        nvtx.range_push("layer_offload")
        try:
            # Quantize, build the Quest index, and copy to CPU on the copy
            # stream. The main stream goes on to the next layer immediately.
            copy_stream = getattr(self._cache, "_copy_stream", None)
            row = int(getattr(self, "_cohort_row", 0))
            if copy_stream is None:
                self._cache.prefill_update_kv_cache(None, keys, values, layer_id, row)
            else:
                ready = torch.cuda.Event()
                ready.record()
                copy_stream.wait_event(ready)
                with torch.cuda.stream(copy_stream):
                    self._cache.prefill_update_kv_cache(
                        None, keys, values, layer_id, row
                    )
            self._initialized_layers.add(layer_id)
            self._release_retrieval_pages_if_ready(forward_batch)
        finally:
            nvtx.range_pop()

            pass
    def offload_finished_prefill(self, forward_batch: ForwardBatch) -> None:
        """Copy retrieval KV out and free its pages after the prefill forward returns.

        Doing this inside a layer allocates the sidecar while that layer's
        activations are still live, and rank 0 dies mid-forward. The pool
        already holds the KV once the forward returns, so the copy happens
        then, before the scheduler looks for the next prefill.
        """
        if not bool(self.config.enable_offload):
            return
        if (
            torch.cuda.is_current_stream_capturing()
            or getattr(self, "_zoomkv_graph_capture", False)
        ):
            return
        batch_size = int(getattr(forward_batch, "batch_size", 0) or 0)
        if batch_size <= 0:
            return
        layers = self._radix_layers()
        if not layers:
            print("[ZoomKV] prefill offload found no attention layers", flush=True)
            return
        _, seq_lens = self._batch_state(forward_batch, int(layers[0].layer_id))
        seq = seq_lens[0] if seq_lens else 0
        print(
            f"[ZoomKV] prefill forward done, offloading {batch_size} request(s), seq={seq}",
            flush=True,
        )
        for layer in layers:
            self._offload_layer_now(None, None, layer, forward_batch)

    def _radix_layers(self):
        cached = getattr(self, "_radix_layer_list", None)
        if cached:
            return cached
        found = []
        model = getattr(self.runner, "model", None)
        if model is not None:
            for module in model.modules():
                if module.__class__.__name__ == "RadixAttention":
                    found.append(module)
        found.sort(key=lambda layer: int(layer.layer_id))
        self._radix_layer_list = found
        return found

    def _release_retrieval_pages_if_ready(self, forward_batch: ForwardBatch) -> None:
        """Return retrieval pages to the GPU pool after the CPU copy has finished.

        Sink and local positions stay allocated. The pinned CPU tensors keep the
        retrieval KV. Chunk-cache completion skips this span so it is not freed
        a second time.
        """
        if self._retrieval_pages_released or self._cache is None:
            return
        install_offload_page_release()
        layer_count = int(self.runner.model_config.num_attention_layers)
        if len(self._initialized_layers) != layer_count:
            return
        span = getattr(self._cache, "_retrieval_gpu_span", None)
        if span is None:
            return
        start, end = int(span[0]), int(span[1])
        if end <= start:
            return
        allocator = getattr(self.runner, "token_to_kv_pool_allocator", None)
        page_size = int(getattr(allocator, "page_size", 1) or 1)
        if allocator is None or page_size != 1:
            print(
                "[ZoomKV] retrieval GPU pages kept "
                f"(allocator={allocator is not None}, page_size={page_size})",
                flush=True,
            )
            self._retrieval_pages_released = True
            return
        self._cache._sync_offload_copies()
        batch_size = int(forward_batch.batch_size)
        rows = forward_batch.req_pool_indices[:batch_size].detach().to(torch.int64).cpu()
        parts = [
            self.req_to_token_pool.req_to_token[int(row), start:end]
            for row in rows.tolist()
        ]
        allocator.free(torch.cat(parts))
        for row in rows.tolist():
            note_released_retrieval(int(row), start, end)
        self._retrieval_pages_released = True
        print(
            f"[ZoomKV] released retrieval GPU pages [{start}, {end}) "
            f"for {batch_size} request(s); sink and local stay on GPU",
            flush=True,
        )
        # Park immediately. Waiting until the next _reset_for_batch deadlocks
        # the last request in a prefill-all-first wave: decode wants parked==N,
        # but the N-th cache is only parked when decode starts.
        self._park_finished_prefill()
        parked = getattr(self, "_parked_caches", None) or {}
        print(
            f"[ZoomKV] parked prefill slot={self._request_slots} "
            f"({len(parked)} parked)",
            flush=True,
        )

    def _batch_state(
        self, forward_batch: ForwardBatch, layer_id: int = 0
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        batch_size = int(forward_batch.batch_size)
        if batch_size <= 0 or batch_size > 64:
            raise RuntimeError(
                f"ZoomKV supports decode batch_size in [1, 64], got {batch_size}"
            )
        # The static CUDA-graph ForwardBatch is reused for every token, and a
        # new cohort overwrites req_pool_indices in that same buffer. Refresh
        # slots and host seq lens on layer 0 of each token; later layers reuse
        # them. seq_lens_cpu is already on the host — reading it 36 times, or
        # reading the GPU seq_lens at all, shows up as a per-layer sync.
        slot_key = (id(forward_batch), batch_size)
        if layer_id == 0 or getattr(self, "_slot_key", None) != slot_key:
            cpu_lens = getattr(forward_batch, "seq_lens_cpu", None)
            if cpu_lens is not None:
                seq_lens = tuple(int(value) for value in cpu_lens[:batch_size].tolist())
            else:
                seq_lens = tuple(
                    int(value) for value in forward_batch.seq_lens[:batch_size].tolist()
                )
            slots = tuple(
                int(value)
                for value in forward_batch.req_pool_indices[:batch_size].tolist()
            )
            self._slot_key = slot_key
            self._state_slots = slots
            self._state_seq_lens = seq_lens
        else:
            slots = self._state_slots
            seq_lens = self._state_seq_lens
        return slots, seq_lens

    def _prefill_cohort_size(self) -> int:
        """Target sidecar batch width for prefill-all-first waves."""
        import os

        from sglang_zoomkv.schedule import prefill_all_first_enabled

        if not prefill_all_first_enabled():
            return 1
        env = os.environ.get("SGLANG_ZOOMKV_PREFILL_COHORT", "").strip()
        if env:
            return max(1, int(env))
        return max(1, int(getattr(self.runner.server_args, "max_running_requests", 1) or 1))

    def _park_finished_prefill(self) -> None:
        """Keep a finished one-request prefill until the cohort decodes together."""
        if not getattr(self, "_retrieval_pages_released", False):
            return
        slots = getattr(self, "_request_slots", None)
        cache = getattr(self, "_cache", None)
        if not slots or cache is None or len(slots) != 1:
            return
        runner = getattr(self, "runner", None)
        model_config = getattr(runner, "model_config", None)
        layer_count = getattr(model_config, "num_attention_layers", None)
        if layer_count is None:
            return
        if len(getattr(self, "_initialized_layers", ())) != int(layer_count):
            return
        parked = getattr(self, "_parked_caches", None)
        if parked is None:
            parked = {}
            self._parked_caches = parked
        slot = int(slots[0])
        if slot in parked:
            return
        parked[slot] = cache
        order = getattr(self, "_cohort_slot_order", None)
        if order is None:
            order = []
            self._cohort_slot_order = order
        order.append(slot)
        # Shrink the pool before the scheduler allocates decode slots.
        # Doing it later leaves out_cache_loc pointing past the new size_limit.
        cohort = self._prefill_cohort_size()
        if len(parked) >= cohort > 1:
            seq_len = int(cache.current_seq_len[0])
            self._reclaim_idle_kv_pool(tuple(int(s) for s in order), seq_len)

    def _align_to_cache_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        gather = getattr(self, "_cohort_gather", None)
        if not gather:
            return tensor
        idx = torch.as_tensor(gather, dtype=torch.int64, device=tensor.device)
        return tensor.index_select(0, idx)

    def _align_from_cache_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        scatter = getattr(self, "_cohort_scatter", None)
        if not scatter:
            return tensor
        idx = torch.as_tensor(scatter, dtype=torch.int64, device=tensor.device)
        return tensor.index_select(0, idx)

    def _attach_filled_cohort(self, slots: tuple[int, ...], cache) -> None:
        order = tuple(int(slot) for slot in (getattr(self, "_cohort_slot_order", ()) or ()))
        gather, scatter = fill_to_decode_perm(order, slots)
        self._cache = cache
        self._request_slots = slots
        self._batch_size = len(slots)
        self._cohort_row = 0
        self._cohort_gather = gather
        self._cohort_scatter = scatter
        layer_count = int(self.runner.model_config.num_attention_layers)
        self._initialized_layers = set(range(layer_count))
        self._retrieval_pages_released = True
        self._parked_caches = {}
        self._cohort_slot_order = list(order)
        extra = ""
        if gather is not None:
            extra = f" (aligned fill={order} -> decode={slots})"
        print(
            f"[ZoomKV] prefill cohort complete, decoding {len(slots)} "
            f"requests together (row-filled, no merge){extra}",
            flush=True,
        )

    def prepare_joint_decode(self, forward_batch: ForwardBatch) -> None:
        """Attach the filled sidecar, reclaim idle KV pages, stage Quest on GPU.

        Must run before CUDA-graph capture. Otherwise the graph bakes the CPU
        UVA index in and TPOT stays ~400 ms.
        """
        if getattr(self, "_joint_decode_prepared", False):
            return
        slots, seq_lens = self._batch_state(forward_batch, 0)
        if len(slots) < 2 or not seq_lens:
            return
        parked = getattr(self, "_parked_caches", None) or {}
        if should_merge_parked_slots(slots, parked):
            cache = parked[int(slots[0])]
            shared = cache is not None and all(
                parked[int(slot)] is cache for slot in slots
            )
            if shared and int(cache.batch_size) >= len(slots):
                self._attach_filled_cohort(slots, cache)
        if self._cache is None or self._request_slots != slots:
            return
        loc = getattr(forward_batch, "out_cache_loc", None)
        extra = None
        if loc is not None:
            extra = loc[: int(forward_batch.batch_size)]
        self._reclaim_idle_kv_pool(slots, int(seq_lens[0]), extra_locs=extra)
        cache = self._cache
        if cache is not None and bool(getattr(cache, "enable_offload", False)):
            cache._index_stage_decided = False
            cache._allow_index_stage = True
            cache.stage_index_for_decode()
        self._joint_decode_prepared = True

    def _reclaim_idle_kv_pool(
        self,
        slots: tuple[int, ...],
        seq_len: int,
        extra_locs: torch.Tensor | None = None,
    ) -> None:
        """Copy live sink/local/decode pages into a small pool and free the rest.

        Prefill needs ~144k tokens. After retrieval is offloaded each request
        only keeps sink+local plus new decode tokens, but the 9.9 GiB tensor
        stays reserved. bs=6 Quest index is 17.2 GiB and only 13.6 GiB is free
        until this compact runs.
        """
        if getattr(self, "_kv_pool_compacted", False):
            return
        pool = self.token_to_kv_pool
        allocator = getattr(self.runner, "token_to_kv_pool_allocator", None)
        table = self.req_to_token_pool.req_to_token
        if pool is None or allocator is None or not getattr(pool, "k_buffer", None):
            return
        if int(getattr(allocator, "page_size", 1) or 1) != 1:
            print("[ZoomKV] KV compact skipped (page_size != 1)", flush=True)
            return
        span = getattr(self._cache, "_retrieval_gpu_span", None)
        start = int(span[0]) if span else 0
        end = int(span[1]) if span else 0
        seq_len = min(int(seq_len), int(table.shape[1]))
        parts = []
        for slot in slots:
            row = table[int(slot), :seq_len]
            if end > start:
                parts.append(row[:start])
                parts.append(row[end:seq_len])
            else:
                parts.append(row)
        if extra_locs is not None and extra_locs.numel() > 0:
            parts.append(extra_locs.reshape(-1).to(dtype=torch.int64))
        live = torch.cat([part.reshape(-1) for part in parts if part.numel() > 0])
        live = live[live > 0].to(dtype=torch.int64).unique()
        n_live = int(live.numel())
        if n_live <= 0:
            return
        max_new = int(getattr(self.config, "max_new_tokens", 768))
        new_size = n_live + len(slots) * (max_new + 8) + 8192
        old_size = int(pool.size)
        if new_size >= old_size:
            print(
                f"[ZoomKV] KV compact skipped (need {new_size} >= pool {old_size})",
                flush=True,
            )
            return
        device = live.device
        free_before, _ = torch.cuda.mem_get_info(self.device)
        try:
            new_idx = torch.arange(1, n_live + 1, dtype=torch.int64, device=device)
            mapper = torch.zeros(
                int(pool.k_buffer[0].shape[0]) + 1,
                dtype=torch.int64,
                device=device,
            )
            mapper[live] = new_idx
            for slot in slots:
                row = table[int(slot), :seq_len]
                if end > start:
                    row[:start] = mapper[row[:start].long()]
                    row[end:seq_len] = mapper[row[end:seq_len].long()]
                else:
                    table[int(slot), :seq_len] = mapper[row.long()]
            if extra_locs is not None and extra_locs.numel() > 0:
                extra_locs.copy_(
                    mapper[extra_locs.long().clamp(0, mapper.numel() - 1)]
                )
            rows = new_size + int(pool.page_size)
            for layer in range(int(pool.layer_num)):
                old_k = pool.k_buffer[layer]
                old_v = pool.v_buffer[layer]
                new_k = torch.zeros(
                    (rows, *old_k.shape[1:]), dtype=old_k.dtype, device=old_k.device
                )
                new_v = torch.zeros(
                    (rows, *old_v.shape[1:]), dtype=old_v.dtype, device=old_v.device
                )
                new_k[new_idx] = old_k[live]
                new_v[new_idx] = old_v[live]
                pool.k_buffer[layer] = new_k
                pool.v_buffer[layer] = new_v
                del old_k, old_v
            pool.size = new_size
            if hasattr(pool, "_init_data_ptrs_and_strides"):
                pool._init_data_ptrs_and_strides()
            allocator.size = new_size
            if hasattr(self.runner, "max_total_num_tokens"):
                self.runner.max_total_num_tokens = new_size
            allocator.free_pages = torch.arange(
                n_live + 1, new_size + 1, dtype=torch.int64, device=device
            )
            allocator.release_pages = torch.empty(
                (0,), dtype=torch.int64, device=device
            )
            import gc

            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self._kv_pool_compacted = True
            free_after, _ = torch.cuda.mem_get_info(self.device)
            print(
                f"[ZoomKV] compacted KV pool {old_size} -> {new_size} tokens "
                f"(live={n_live}); gpu_free "
                f"{free_before / (1024 ** 3):.1f} -> {free_after / (1024 ** 3):.1f} GiB",
                flush=True,
            )
        except Exception as exc:
            print(f"[ZoomKV] KV compact failed ({exc}); index may stay on CPU", flush=True)

    def _reset_for_batch(
        self,
        slots: tuple[int, ...],
        prefix_len: int,
        layer: RadixAttention,
        dtype: torch.dtype,
        *,
        prefer_cohort: bool = False,
    ) -> None:
        from sglang_zoomkv.cache import ZoomKVCache

        self._park_finished_prefill()
        parked = getattr(self, "_parked_caches", None) or {}
        cohort = self._prefill_cohort_size()
        max_new_tokens = int(getattr(self.config, "max_new_tokens", 4096))
        self._cohort_gather = None
        self._cohort_scatter = None

        # Decode of the full wave: every request already wrote a row of one
        # sidecar. Do not concatenate pinned copies — Torch 2.13 never returns
        # that host memory, and the cgroup SIGKILLs at ~177 GiB.
        if should_merge_parked_slots(slots, parked):
            cache = parked[int(slots[0])]
            shared = cache is not None and all(
                parked[int(slot)] is cache for slot in slots
            )
            if shared and int(cache.batch_size) >= len(slots):
                self._attach_filled_cohort(slots, cache)
                if self._cache is not None and bool(
                    getattr(self._cache, "enable_offload", False)
                ):
                    self._cache._index_stage_decided = False
                    self._cache._allow_index_stage = True
                    self._cache.stage_index_for_decode()
                self._joint_decode_prepared = True
                return
            raise RuntimeError(
                "ZoomKV needs a single row-filled sidecar to decode this "
                f"batch of {len(slots)}; refusing merge_cohort"
            )

        # Next one-request prefill of an in-progress wave: keep the same sidecar.
        if (
            prefer_cohort
            and len(slots) == 1
            and cohort > 1
            and self._cache is not None
            and int(getattr(self._cache, "batch_size", 0)) == cohort
            and int(getattr(self._cache, "max_length", 0)) >= prefix_len + max_new_tokens
            and len(parked) < cohort
            and int(slots[0]) not in parked
        ):
            self._cohort_row = len(parked)
            self._request_slots = slots
            self._batch_size = 1
            self._initialized_layers.clear()
            self._retrieval_pages_released = False
            print(
                f"[ZoomKV] cohort row fill row={self._cohort_row}/{cohort} "
                f"slot={slots[0]}",
                flush=True,
            )
            return

        if len(slots) == 1 and parked:
            parked.pop(int(slots[0]), None)

        # Drop the previous sidecar before allocating the next. Parked prefills
        # stay reachable from _parked_caches.
        self._cache = None
        self._initialized_layers.clear()
        self._joint_decode_prepared = False
        if not (
            prefer_cohort
            and len(slots) == 1
            and cohort > 1
            and parked
            and len(parked) < cohort
        ):
            self._cohort_slot_order = []
        layer_count = int(self.runner.model_config.num_attention_layers)
        alloc_bs = len(slots)
        # Prefill of the first request in the wave: one sidecar wide enough for
        # the whole joint decode. Server warmup decode is prefer_cohort=False,
        # so it stays batch_size=1 and cannot reshape 6 rows into 1.
        if prefer_cohort and len(slots) == 1 and cohort > 1 and not parked:
            alloc_bs = cohort
        self._cache = ZoomKVCache(
            valid_start=0,
            layer_num=layer_count,
            batch_size=alloc_bs,
            max_length=prefix_len + max_new_tokens,
            num_key_value_heads=layer.tp_k_head_num,
            num_heads=layer.tp_q_head_num,
            head_dim=layer.qk_head_dim,
            dtype=dtype,
            layer_mapping={i: str(self.device) for i in range(layer_count)},
            max_new_length=max_new_tokens,
            sink_size=int(self.config.sink_size),
            local_size=int(self.config.local_size),
            core=1,
            nprobe=1,
            cache_unit_size=1,
            cache_cluster_num=1,
            num_gpus=1,
            model_size=0,
            candidate_ratio=float(self.config.candidate_ratio),
            final_topk=int(self.config.final_topk),
            dynamic_update_interval=int(self.config.dynamic_update_interval),
            full_attention_threshold=int(self.config.full_attention_threshold),
            enable_offload=bool(self.config.enable_offload),
        )
        self._request_slots = slots
        self._batch_size = len(slots)
        self._cohort_row = 0
        self._retrieval_pages_released = False
        logger.info(
            "Initialized ZoomKV sidecar for %s request slots (prefix=%s, capacity=%s)",
            alloc_bs,
            prefix_len,
            prefix_len + max_new_tokens,
        )
        if alloc_bs != len(slots):
            print(
                f"[ZoomKV] pre-allocated cohort sidecar bs={alloc_bs} "
                f"(first slot={slots[0]})",
                flush=True,
            )

    def _ensure_layer_prefill(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        prefix_len: int,
    ) -> None:
        assert self._cache is not None
        layer_id = int(layer.layer_id)
        if layer_id in self._initialized_layers:
            return

        # First decode step of this layer: copy the SGLang pool into the sidecar.
        nvtx.range_push("sidecar_prefill_copy")
        try:
            request_rows = forward_batch.req_pool_indices[: self._batch_size].long()
            token_indices = self.req_to_token_pool.req_to_token[
                request_rows, :prefix_len
            ].long()
            keys = self.token_to_kv_pool.get_key_buffer(layer_id)[token_indices]
            values = self.token_to_kv_pool.get_value_buffer(layer_id)[token_indices]
            keys = keys.reshape(
                self._batch_size,
                prefix_len,
                layer.tp_k_head_num,
                layer.qk_head_dim,
            )
            values = values.reshape(
                self._batch_size,
                prefix_len,
                layer.tp_v_head_num,
                layer.v_head_dim,
            )
            self._cache.prefill_update_kv_cache(
                None, keys.contiguous(), values.contiguous(), layer_id, 0
            )
            self._initialized_layers.add(layer_id)
            if bool(self.config.enable_offload):
                self._release_retrieval_pages_if_ready(forward_batch)
        finally:
            nvtx.range_pop()

            pass
    def _forward_decode_captured(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ):
        """Attention body recorded into the parent decode graph.

        No host sync, no cache reset, and no nested graph. Sequence bounds and
        request rows were staged before this replay.
        """
        cache = self._cache
        if cache is None:
            raise RuntimeError("ZoomKV CUDA graph capture ran before the cache existed")
        if layer.qk_head_dim not in (128, 256) or layer.v_head_dim != layer.qk_head_dim:
            raise RuntimeError("ZoomKV requires qk/v head dimensions of 128 or 256")
        if save_kv_cache:
            token_count = int(k.shape[0])
            loc = forward_batch.out_cache_loc[:token_count]
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(loc),
                k[: loc.shape[0]].contiguous(),
                v[: loc.shape[0]].contiguous(),
            )
        batch_size = int(q.shape[0])
        query = q.reshape(
            batch_size, 1, layer.tp_q_head_num, layer.qk_head_dim
        ).contiguous()
        query = self._align_to_cache_rows(query)
        output = cache.attention_inside_graph(query, int(layer.layer_id))
        output = self._align_from_cache_rows(output)
        if int(output.shape[0]) != int(q.shape[0]):
            raise RuntimeError(
                "ZoomKV graph attention returned "
                f"{int(output.shape[0])} rows for a {int(q.shape[0])}-request decode"
            )
        return output.reshape(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs: Any,
    ):
        if kwargs:
            raise RuntimeError(
                "ZoomKV MVP does not support extra attention inputs: "
                + ", ".join(sorted(kwargs))
            )
        if layer.is_cross_attention or layer.sliding_window_size not in (-1, None):
            raise RuntimeError(
                "ZoomKV MVP supports decoder self-attention without sliding window"
            )
        if k is None or v is None:
            raise RuntimeError("ZoomKV MVP does not support cross-layer KV sharing")
        if self._zoomkv_graph_capture or torch.cuda.is_current_stream_capturing():
            return self._forward_decode_captured(
                q, k, v, layer, forward_batch, save_kv_cache
            )

        slots, seq_lens = self._batch_state(
            forward_batch, int(getattr(layer, "layer_id", 0))
        )
        if len(set(seq_lens)) != 1:
            # SGLang startup warmup and general continuous batching may be
            # ragged. Keep those requests correct through the native backend;
            # fixed-length throughput cohorts use the ZoomKV batched path.
            return self.fallback.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )
        seq_len = seq_lens[0]
        batch_size = len(slots)
        prefix_len = seq_len - 1
        if prefix_len <= 0:
            return self.fallback.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )
        if layer.qk_head_dim not in (128, 256) or layer.v_head_dim != layer.qk_head_dim:
            raise RuntimeError("ZoomKV requires qk/v head dimensions of 128 or 256")
        if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
            raise RuntimeError(
                "ZoomKV requires bfloat16 Q/K/V tensors; got "
                f"{q.dtype}, {k.dtype}, {v.dtype}"
            )

        if save_kv_cache:
            token_count = int(k.shape[0])
            loc = forward_batch.out_cache_loc[:token_count]
            nvtx.range_push("sglang_kv_store")
            try:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(loc),
                    k[: loc.shape[0]].contiguous(),
                    v[: loc.shape[0]].contiguous(),
                )
            finally:
                nvtx.range_pop()

                pass
        # A new request often reuses the slot freed by server warmup. That
        # warmup is a few tokens, so its cached length is far below the real
        # prefix. The next token of the SAME request is exactly one past the
        # cached length; anything else is a different sequence and must rebuild
        # the Quest index from the SGLang pool.
        cached_len = (
            int(self._cache.current_seq_len[0]) if self._cache is not None else None
        )
        # q has len(slots) rows; a leftover cohort sidecar still has batch_size=N.
        wide_cache = (
            self._cache is not None
            and len(slots) == 1
            and int(getattr(self._cache, "batch_size", 0)) > 1
        )
        cache_is_stale = (
            self._cache is None
            or self._request_slots != slots
            or wide_cache
            or (int(layer.layer_id) == 0 and cached_len + 1 != int(seq_len))
        )
        if cache_is_stale:
            if cached_len is not None and int(layer.layer_id) == 0:
                print(
                    f"[ZoomKV] rebuilding Quest index for seq_len {int(seq_len)} "
                    f"(cached length was {cached_len})",
                    flush=True,
                )
            self._zoomkv_graphs_ready = False
            self._zoomkv_captured_bs = None
            nvtx.range_push("sidecar_reset")
            try:
                self._reset_for_batch(slots, prefix_len, layer, k.dtype)
            finally:
                nvtx.range_pop()

                pass
        self._ensure_layer_prefill(layer, forward_batch, prefix_len)
        assert self._cache is not None
        # Offload's first token reads the pool while the CPU copy is still
        # running. Later replays bind again in stage_for_replay.
        self._cache.bind_sglang_kv(
            self.token_to_kv_pool,
            self.req_to_token_pool.req_to_token,
            self._align_to_cache_rows(forward_batch.req_pool_indices[:batch_size]),
        )
        if int(layer.layer_id) == 0:
            self._cache.stage_index_for_decode()
        nvtx.range_push("decode_kv_update")
        try:
            self._cache.decode_update_kv_cache(
                self._align_to_cache_rows(
                    k.reshape(
                        batch_size, 1, layer.tp_k_head_num, layer.qk_head_dim
                    ).contiguous()
                ),
                self._align_to_cache_rows(
                    v.reshape(
                        batch_size, 1, layer.tp_v_head_num, layer.v_head_dim
                    ).contiguous()
                ),
                int(layer.layer_id),
            )
        finally:
            nvtx.range_pop()
            pass
        query = q.reshape(
            batch_size, 1, layer.tp_q_head_num, layer.qk_head_dim
        ).contiguous()
        query = self._align_to_cache_rows(query)
        nvtx.range_push("zoomkv_attention")
        try:
            output = self._cache.compute(query, int(layer.layer_id))
        finally:
            nvtx.range_pop()
            pass
        output = self._align_from_cache_rows(output)
        if int(output.shape[0]) != int(q.shape[0]):
            raise RuntimeError(
                "ZoomKV attention returned "
                f"{int(output.shape[0])} rows for a {int(q.shape[0])}-request decode"
            )
        return output.reshape(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def support_triton(self):
        return False

