"""Python wrapper for the fused float-topk + gather + skipped-concat kernel.

Companion to ``float_topk_ops.float_topk_3d``: replaces the 3-launch chain at
the tail of ``_cds_dense_sparse_select`` with a single CUDA launch that writes
into a caller-provided buffer (CUDA-Graph friendly).

Caller contract:
    out[:, :, :k]            ← lut[:, :, argtop_k(scores)]
    out[:, :, k:k+sk_count]  ← skipped[:, :, :sk_count]

``out`` is pre-allocated (size (bs, kv, k + max_sk)).  Caller slices to the
useful region ``out[:, :, :k + sk_count]`` after the kernel returns.

Shapes / dtypes:
    scores  : [bs, kv, L]    fp32
    lut     : [bs, kv, L]    int64
    skipped : [bs, kv, sk]   int64   (sk may be 0; pass empty slice)
    out     : [bs, kv, ≥k+sk] int64  (contiguous)

Returns the input ``out`` tensor (for chaining); no allocation.
"""

import os
import warnings
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import torch
from sglang_zoomkv.kernels.jit import load_jit


@lru_cache()
def _prepare_for_load() -> str:
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


def load_fused_topk_module(
    path: str = "float_topk_gather_concat.cu",
    name: str = "float_topk_gather_concat_ext",
    *,
    build: str = "build",
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
) -> Any:
    del build
    if (
        path == "float_topk_gather_concat.cu"
        and name == "float_topk_gather_concat_ext"
        and cflags is None
        and cuda_flags is None
    ):
        return _default_fused_topk_module()
    return load_jit(
        name, path, source_dir=Path(__file__).parent,
        cflags=cflags, cuda_flags=cuda_flags,
    )


@lru_cache(maxsize=1)
def _default_fused_topk_module() -> Any:
    return load_jit(
        "float_topk_gather_concat_ext",
        "float_topk_gather_concat.cu",
        source_dir=Path(__file__).parent,
    )


def float_topk_gather_concat_3d(
    scores: torch.Tensor,
    lut: torch.Tensor,
    skipped: torch.Tensor,
    out: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Fused top-k + indirect gather + skipped-tail concat.

    Args:
        scores  : [bs, kv, L] fp32     – ranking scores
        lut     : [bs, kv, L] int64    – token-id lookup table; value at
                                         ``lut[b, h, i]`` is the global token
                                         id corresponding to score
                                         ``scores[b, h, i]``
        skipped : [bs, kv, sk] int64   – pre-computed skipped token ids to
                                         append; ``sk`` may be 0
        out     : [bs, kv, ≥k+sk] int64 – contiguous output buffer (caller
                                          pre-allocated; caller slices to
                                          ``out[:, :, :k + sk]`` after the
                                          call)
        k       : Python int           – number of top-k elements

    Returns:
        ``out`` (same buffer; returned for convenience).
    """
    assert scores.is_cuda and lut.is_cuda and skipped.is_cuda and out.is_cuda, \
        "all tensors must be on CUDA"
    assert scores.dim() == 3 and lut.dim() == 3 and skipped.dim() == 3 and out.dim() == 3, \
        "all tensors must be 3D"
    assert scores.dtype == torch.float32, f"scores must be fp32; got {scores.dtype}"
    assert lut.dtype == torch.int64, f"lut must be int64; got {lut.dtype}"
    assert skipped.dtype == torch.int64, f"skipped must be int64; got {skipped.dtype}"
    assert out.dtype == torch.int64, f"out must be int64; got {out.dtype}"
    assert scores.shape == lut.shape, f"scores/lut shape mismatch: {scores.shape} vs {lut.shape}"
    assert skipped.shape[:2] == scores.shape[:2], \
        f"skipped (bs, kv) mismatch: {skipped.shape[:2]} vs {scores.shape[:2]}"
    assert out.shape[:2] == scores.shape[:2], \
        f"out (bs, kv) mismatch: {out.shape[:2]} vs {scores.shape[:2]}"

    sk = skipped.shape[2]
    assert k > 0 and k <= scores.shape[2], f"k={k} out of range [1, {scores.shape[2]}]"
    assert out.shape[2] >= k + sk, \
        f"out last dim={out.shape[2]} must be ≥ k+sk={k + sk}"
    assert out.is_contiguous(), "out must be contiguous"

    kernel = _default_fused_topk_module()
    kernel.float_topk_gather_concat_3d(scores, lut, skipped, out, int(k))
    return out
