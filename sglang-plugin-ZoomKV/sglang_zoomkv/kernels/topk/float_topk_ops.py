"""
Float Top-K CUDA ops with multi-round radix select

Usage:
    from sglang_zoomkv.kernels.topk.float_topk_ops import float_topk, float_topk_3d
    indices = float_topk_3d(scores, k)
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


def load_float_topk_module(
    path: str = "float_topk.cu",
    name: str = "float_topk_ext",
    *,
    build: str = "build",
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
) -> Any:
    del build
    if (
        path == "float_topk.cu"
        and name == "float_topk_ext"
        and cflags is None
        and cuda_flags is None
    ):
        return _default_float_topk_module()
    return load_jit(
        name, path, source_dir=Path(__file__).parent,
        cflags=cflags, cuda_flags=cuda_flags,
    )


@lru_cache(maxsize=1)
def _default_float_topk_module() -> Any:
    """Load float_topk.cu once per process, not once per top-k call."""
    return load_jit(
        "float_topk_ext",
        "float_topk.cu",
        source_dir=Path(__file__).parent,
    )


def float_topk(
    scores: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Float Top-K using multi-round radix select (2D input)

    Args:
        scores: [B, L] float tensor
        k: number of top elements

    Returns:
        indices: [B, k] int64 tensor
    """
    assert scores.is_cuda, "scores must be on CUDA"
    assert scores.dim() == 2, "scores must be 2D [B, L]"
    assert k > 0, f"k must be positive, got {k}"
    assert k <= scores.size(1), f"k={k} cannot exceed L={scores.size(1)}"

    return _default_float_topk_module().float_topk(scores, k)


def float_topk_3d(
    scores: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Float Top-K using multi-round radix select (3D input)

    Args:
        scores: [bs, kv_heads, n_chunks] float tensor
        k: number of top elements

    Returns:
        indices: [bs, kv_heads, k] int64 tensor
    """
    assert scores.is_cuda, "scores must be on CUDA"
    assert scores.dim() == 3, "scores must be 3D [bs, kv_heads, n_chunks]"
    assert k > 0, f"k must be positive, got {k}"
    assert k <= scores.size(2), f"k={k} cannot exceed n_chunks={scores.size(2)}"

    return _default_float_topk_module().float_topk_3d(scores, k)


if __name__ == "__main__":
    # Quick test
    print("Quick test...")
    x = torch.randn(2, 1000, dtype=torch.float32, device='cuda')

    print(f"Input shape: {x.shape}")

    indices = float_topk(x, k=100)
    print(f"float_topk output shape: {indices.shape}")

    # Compare with torch.topk
    _, ref_indices = torch.topk(x, k=100, dim=-1)

    # Check overlap
    overlap = 0
    for i in range(x.size(0)):
        ref_set = set(ref_indices[i].cpu().tolist())
        test_set = set(indices[i].cpu().tolist())
        overlap += len(ref_set & test_set)

    total = x.size(0) * 100
    print(f"Overlap with torch.topk: {overlap}/{total} ({overlap/total*100:.1f}%)")

    print("Done!")
