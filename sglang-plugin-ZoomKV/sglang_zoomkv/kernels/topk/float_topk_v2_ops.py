"""
Float Top-K v2: radix select with global-memory candidate buffers.

Fixes shared-memory overflow in float_topk when scores are highly clustered
(e.g. Quest bbox upper-bound scores).

Usage:
    from sglang_zoomkv.kernels.topk.float_topk_v2_ops import float_topk_v2, float_topk_v2_3d
    indices = float_topk_v2(scores_2d, k)
    indices = float_topk_v2_3d(scores_3d, k)
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


def load_float_topk_v2_module(
    path: str = "float_topk_v2.cu",
    name: str = "float_topk_v2_ext",
    *,
    build: str = "build",
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
) -> Any:
    del build
    if (
        path == "float_topk_v2.cu"
        and name == "float_topk_v2_ext"
        and cflags is None
        and cuda_flags is None
    ):
        return _default_float_topk_v2_module()
    return load_jit(
        name, path, source_dir=Path(__file__).parent,
        cflags=cflags, cuda_flags=cuda_flags,
    )


@lru_cache(maxsize=1)
def _default_float_topk_v2_module() -> Any:
    return load_jit(
        "float_topk_v2_ext",
        "float_topk_v2.cu",
        source_dir=Path(__file__).parent,
    )


def float_topk_v2(
    scores: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Float Top-K v2 using radix select with global-memory buffers (2D input).

    Args:
        scores: [B, L] float tensor on CUDA
        k: number of top elements to select

    Returns:
        indices: [B, k] int64 tensor
    """
    assert scores.is_cuda, "scores must be on CUDA"
    assert scores.dim() == 2, "scores must be 2D [B, L]"
    assert k > 0, f"k must be positive, got {k}"
    assert k <= scores.size(1), f"k={k} cannot exceed L={scores.size(1)}"

    return _default_float_topk_v2_module().float_topk_v2(scores, k)


def float_topk_v2_3d(
    scores: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Float Top-K v2 using radix select with global-memory buffers (3D input).

    Args:
        scores: [bs, kv_heads, n_chunks] float tensor on CUDA
        k: number of top elements to select

    Returns:
        indices: [bs, kv_heads, k] int64 tensor
    """
    assert scores.is_cuda, "scores must be on CUDA"
    assert scores.dim() == 3, "scores must be 3D [bs, kv_heads, n_chunks]"
    assert k > 0, f"k must be positive, got {k}"
    assert k <= scores.size(2), f"k={k} cannot exceed n_chunks={scores.size(2)}"

    return _default_float_topk_v2_module().float_topk_v2_3d(scores, k)
