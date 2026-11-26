"""
Fused per-chunk top-k with index gather (CUDA).

Usage:
    from sglang_zoomkv.kernels.topk.chunk_topk_ops import chunk_topk_gather
    top_scores, top_indices = chunk_topk_gather(scores_4d, indices_4d, k=2)
"""

import os
import warnings
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Tuple

import torch
from sglang_zoomkv.kernels.jit import load_jit


@lru_cache()
def _prepare_for_load() -> str:
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


def load_chunk_topk_module(
    path: str = "chunk_topk.cu",
    name: str = "chunk_topk_ext",
    *,
    build: str = "build",
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
) -> Any:
    del build
    if (
        path == "chunk_topk.cu"
        and name == "chunk_topk_ext"
        and cflags is None
        and cuda_flags is None
    ):
        return _default_chunk_topk_module()
    return load_jit(
        name, path, source_dir=Path(__file__).parent,
        cflags=cflags, cuda_flags=cuda_flags,
    )


@lru_cache(maxsize=1)
def _default_chunk_topk_module() -> Any:
    return load_jit(
        "chunk_topk_ext",
        "chunk_topk.cu",
        source_dir=Path(__file__).parent,
    )


def chunk_topk_gather(
    scores: torch.Tensor,
    indices: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused per-chunk top-k selection with index gather.

    Args:
        scores:  [bs, kv_heads, nk, chunk_size]  float tensor (chunk_size ≤ 32)
        indices: [bs, kv_heads, nk, chunk_size]  int64 tensor
        k:       number of top elements per chunk

    Returns:
        (top_scores, top_indices): each [bs, kv_heads, nk, k]
    """
    kernel = _default_chunk_topk_module()
    return kernel.chunk_topk_gather(
        scores.contiguous(),
        indices.contiguous(),
        k,
    )
