"""NVTX stage names for the offload decode path.

These are ordinary push/pop ranges on the thread that runs attention, so a
replay shows coarse_filter, topk_select, fine_rerank and uva_read instead of
one solid decode bar.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch.cuda.nvtx as nvtx

COARSE_FILTER = 0
TOPK_SELECT = 1
FINE_RERANK = 2
UVA_READ = 3

# Older call sites.
LARGE_CHUNK_SELECT = COARSE_FILTER
SMALL_CHUNK_SELECT = TOPK_SELECT

_NAMES = ("coarse_filter", "topk_select", "fine_rerank", "uva_read")
_ext = None
_ext_failed = False


def preload_stage_nvtx():
    """Load the host-node NVTX extension before decode graph capture.

    Ranges pushed with torch.cuda.nvtx during capture are not replayed.
    cudaLaunchHostFunc is, so nsys shows these names on every decode step.
    """
    global _ext, _ext_failed
    if _ext is not None or _ext_failed:
        return _ext
    try:
        from sglang_zoomkv.kernels.jit import load_jit

        _ext = load_jit(
            "stage_nvtx",
            "stage_nvtx.cpp",
            source_dir=Path(__file__).parent,
            cflags=["-O3", "-std=c++17"],
            with_cuda=True,
        )
        _ext.warmup()
    except Exception as exc:
        _ext_failed = True
        _ext = None
        print(
            f"[ZoomKV] NVTX host ranges unavailable ({exc}); using Python NVTX",
            flush=True,
        )
    return _ext


@contextmanager
def nvtx_stage(stage: int):
    stage = int(stage)
    ext = _ext if _ext is not None else preload_stage_nvtx()
    if ext is not None:
        ext.range_begin(stage)
    else:
        nvtx.range_push(_NAMES[stage])
    try:
        yield
    finally:
        if ext is not None:
            ext.range_end(stage)
        else:
            nvtx.range_pop()
