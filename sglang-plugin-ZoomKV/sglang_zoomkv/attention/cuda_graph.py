"""Record ZoomKV decode into one full CUDA graph.

The graph is captured on the first uniform batch, after prefill has frozen
Quest sizes. Offload gathers retrieval KV inside that graph; it does not
break the graph at attention.
"""

from __future__ import annotations

_INSTALLED = False


def _strong_ref_break(inner):
    """Graph break that keeps q, k, and v alive across replays.

    ``eager_on_graph`` weak-refs tensor inputs. That is safe only for storage
    pinned by the CUDA-graph pool. Decode q/k/v are views of the QKV projection,
    and a weak ref lets the caching allocator reuse that storage after capture.
    The KV store then reads a freed pointer and CUDA reports an illegal access
    at the store kernel.
    """
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        _copy_output,
        _current_capture_var,
    )

    def wrapper(*args, **kwargs):
        capture = _current_capture_var.get()
        if capture is None:
            return inner(*args, **kwargs)

        capture._end_current_segment()
        if capture._barrier_fn is not None:
            capture._barrier_fn()

        output = inner(*args, **kwargs)
        captured_inner = inner
        captured_args = args
        captured_kwargs = kwargs
        captured_output = output

        def replay_fn():
            new_out = captured_inner(*captured_args, **captured_kwargs)
            return _copy_output(captured_output, new_out)

        capture.cuda_graph._break_fns.append(replay_fn)
        capture._begin_new_segment()
        return output

    return wrapper


def _zoomkv_decode_backend(attn_backend):
    """Return the ZoomKV backend, including when SGLang wraps it in a hybrid.

    ``--prefill-attention-backend flashinfer`` puts FlashInfer and ZoomKV
    behind one hybrid object. The decode runner stores that hybrid, so the
    full-graph flag and the cache live one level down, on ``decode_backend``.
    """
    backend = attn_backend
    seen: set[int] = set()
    while backend is not None and id(backend) not in seen:
        seen.add(id(backend))
        if hasattr(backend, "_zoomkv_full_graph"):
            return backend
        backend = getattr(backend, "decode_backend", None)
    return None


def _uniform_cpu_lens(forward_batch, batch_size: int) -> bool:
    cpu_lens = getattr(forward_batch, "seq_lens_cpu", None)
    if cpu_lens is None or batch_size <= 0:
        return False
    lengths = [int(value) for value in cpu_lens[:batch_size].tolist()]
    return len(set(lengths)) == 1 and lengths[0] > 1


def _install_deferred_full_graph() -> None:
    """Capture the decode graph after prefill, not at process startup.

    Startup capture sees a sequence length of 1, so Quest's host-side sizes
    would be frozen at zero and every replay would skip retrieval. Waiting
    until the real batch exists bakes the retrieval length that this cohort
    actually uses.
    """
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
    from sglang.srt.runtime_context import get_parallel

    original_capture = DecodeCudaGraphRunner.capture
    original_can_run = DecodeCudaGraphRunner.can_run_graph

    def capture(self):
        backend = _zoomkv_decode_backend(self.attn_backend)
        if backend is None or not backend._zoomkv_full_graph:
            return original_capture(self)
        if getattr(backend, "_zoomkv_graph_capture", False):
            return original_capture(self)
        self.warmup()
        if get_parallel().tp_rank == 0:
            print(
                "[ZoomKV] decode CUDA graph deferred until the first full uniform batch",
                flush=True,
            )
        return None

    def can_run_graph(self, forward_batch):
        backend = _zoomkv_decode_backend(self.attn_backend)
        if backend is None or not backend._zoomkv_full_graph:
            return original_can_run(self, forward_batch)
        if getattr(backend, "_zoomkv_graph_failed", False):
            return False
        if getattr(backend, "_zoomkv_graph_capture", False):
            return False
        cache = getattr(backend, "_cache", None)
        batch_size = int(forward_batch.batch_size)
        if batch_size > int(self.max_bs):
            return False
        if (
            cache is None
            or not cache.graph_spec_ready()
            or int(cache.batch_size) != batch_size
            or not _uniform_cpu_lens(forward_batch, batch_size)
        ):
            return False
        # Replay only continues the indexed request. A new request reuses the
        # same pool slot, and its seq_len is not one past the cached length;
        # the eager path must rebuild Quest before any graph is captured.
        live_len = int(forward_batch.seq_lens_cpu[0])
        if int(cache.current_seq_len[0]) + 1 != live_len:
            return False
        captured_bs = getattr(backend, "_zoomkv_captured_bs", None)
        if getattr(backend, "_zoomkv_graphs_ready", False) and captured_bs != batch_size:
            backend._zoomkv_graphs_ready = False
        if not getattr(backend, "_zoomkv_graphs_ready", False):
            # Compact the idle KV pool and stage Quest onto the GPU before
            # capture, otherwise the graph bakes the CPU UVA path in.
            backend.prepare_joint_decode(forward_batch)
            cache = getattr(backend, "_cache", None)
            if cache is None or not cache.graph_spec_ready():
                return False
            backend._zoomkv_graph_capture = True
            try:
                if get_parallel().tp_rank == 0:
                    print(
                        f"[ZoomKV] capturing one decode CUDA graph for batch {batch_size}",
                        flush=True,
                    )
                graph_backend = getattr(self, "backend", None)
                if graph_backend is not None:
                    getattr(graph_backend, "_graphs", {}).clear()
                    getattr(graph_backend, "_outputs", {}).clear()
                self.capture_bs = [batch_size]
                with model_capture_mode():
                    original_capture(self)
                cache._workspace_frozen = True
                backend._zoomkv_captured_bs = batch_size
                backend._zoomkv_graphs_ready = True
                if get_parallel().tp_rank == 0:
                    print("[ZoomKV] decode CUDA graph capture finished", flush=True)
            except Exception as exc:
                backend._zoomkv_graph_failed = True
                cache._workspace_frozen = False
                print(
                    f"[ZoomKV] decode CUDA graph capture failed ({exc}); staying eager",
                    flush=True,
                )
                import traceback

                traceback.print_exc()
                return False
            finally:
                backend._zoomkv_graph_capture = False
        return original_can_run(self, forward_batch)

    DecodeCudaGraphRunner.capture = capture
    DecodeCudaGraphRunner.can_run_graph = can_run_graph


def _install_attention_break() -> None:
    """Run ZoomKV attention outside the captured segments.

    Breakable replay calls this every layer, so the coarse filter, top-k,
    fine rerank and UVA read show up as their own NVTX ranges and kernels.
    Full-graph capture never sets the breakable context, and the wrapper
    then calls the original method.
    """
    from sglang_zoomkv.attention.backend import ZoomKVAttentionBackend

    current = ZoomKVAttentionBackend.forward_decode
    if getattr(current, "_zoomkv_attention_break", False):
        return
    wrapped = _strong_ref_break(current)
    wrapped._zoomkv_attention_break = True
    ZoomKVAttentionBackend.forward_decode = wrapped


def _install_prefill_offload() -> None:
    """Offload retrieval KV after FlashInfer prefill returns.

    The hybrid prefill path never enters ZoomKV, so the retrieval pages used
    to stay allocated. Copying them out inside a layer also kills the process:
    the sidecar is allocated while that layer is still live. Wait until
    ``ModelRunner.forward`` has returned, then copy from the pool and free
    the pages before the next schedule.
    """
    import os
    import time
    import traceback

    from sglang.srt.model_executor.model_runner import ModelRunner

    current = ModelRunner.forward
    if getattr(current, "_zoomkv_prefill_offload", False):
        return

    def _extend_seq(forward_batch) -> tuple[int, int]:
        batch_size = int(getattr(forward_batch, "batch_size", 0) or 0)
        cpu_lens = getattr(forward_batch, "seq_lens_cpu", None)
        if cpu_lens is None or batch_size <= 0:
            return batch_size, -1
        return batch_size, int(cpu_lens[0])

    def _prefill_label(zoomkv) -> tuple[str, int, int]:
        cfg = getattr(zoomkv, "config", None)
        overlap = int(bool(getattr(cfg, "prefill_overlap", True)))
        dense = int(getattr(cfg, "prefill_dense_layers", 4))
        runner = getattr(zoomkv, "runner", None)
        model_config = getattr(runner, "model_config", None)
        layer_count = int(getattr(model_config, "num_attention_layers", 0) or 0)
        sparse = int(layer_count > 0 and dense < layer_count)
        label = os.environ.get("SGLANG_ZOOMKV_PREFILL_MODE", "").strip()
        if not label:
            if overlap and sparse:
                label = "both"
            elif overlap:
                label = "overlap"
            elif sparse:
                label = "sparse"
            else:
                label = "baseline"
        return label, overlap, sparse

    def forward(self, forward_batch, *args, **kwargs):
        mode = getattr(forward_batch, "forward_mode", None)
        is_extend = mode is not None and mode.is_extend()
        started = time.perf_counter() if is_extend else 0.0
        zoomkv = None
        if is_extend:
            zoomkv = _zoomkv_decode_backend(getattr(self, "attn_backend", None))
            if zoomkv is not None and bool(
                getattr(getattr(zoomkv, "config", None), "prefill_overlap", True)
            ):
                try:
                    zoomkv.prepare_prefill_offload(forward_batch)
                except Exception:
                    traceback.print_exc()
                    print(
                        "[ZoomKV] prefill sidecar alloc deferred until after forward",
                        flush=True,
                    )
        forward_started = time.perf_counter() if is_extend else 0.0
        out = current(self, forward_batch, *args, **kwargs)
        forward_done = time.perf_counter() if is_extend else 0.0
        if not is_extend:
            return out
        if zoomkv is None:
            zoomkv = _zoomkv_decode_backend(getattr(self, "attn_backend", None))
        if zoomkv is not None:
            try:
                zoomkv.offload_finished_prefill(forward_batch)
            except Exception:
                traceback.print_exc()
                print("[ZoomKV] prefill offload failed; pages stay on GPU", flush=True)
        finished = time.perf_counter()
        batch_size, seq_len = _extend_seq(forward_batch)
        label, overlap, sparse = (
            _prefill_label(zoomkv) if zoomkv is not None else ("baseline", 0, 0)
        )
        rank = getattr(self, "tp_rank", None)
        if rank is None:
            rank = getattr(getattr(self, "ps", None), "tp_rank", -1)
        print(
            "[ZoomKV] prefill_time "
            f"mode={label} rank={int(rank)} seq={seq_len} bs={batch_size} "
            f"overlap={overlap} sparse={sparse} "
            f"forward_ms={(forward_done - forward_started) * 1000.0:.1f} "
            f"offload_ms={(finished - forward_done) * 1000.0:.1f} "
            f"total_ms={(finished - started) * 1000.0:.1f}",
            flush=True,
        )
        return out

    forward._zoomkv_prefill_offload = True
    ModelRunner.forward = forward


def install_decode_graph_break() -> None:
    """Capture decode after prefill. Offload leaves attention outside the graph."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_deferred_full_graph()
    _install_attention_break()
    _install_prefill_offload()
    _INSTALLED = True
