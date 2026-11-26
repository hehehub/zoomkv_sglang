"""Production staged CPU gather without UVA host-pointer dereferencing."""

import torch

from sglang_zoomkv.kernels.gather import load_gather_ext
from .staged_h2d import staged_h2d_gather_kv


def h2d_gather_kv(
    src_k_caches,
    src_v_caches,
    selected_indices,
    dst_k_caches,
    dst_v_caches,
):
    """Backward-compatible five-argument API implemented with staged copies."""
    batch, heads, topk = selected_indices.shape
    dim = src_k_caches.shape[-1]
    staging_k = torch.empty(
        (batch, heads, topk, dim),
        dtype=src_k_caches.dtype,
        device="cpu",
        pin_memory=True,
    )
    staging_v = torch.empty(
        (batch, heads, topk, dim),
        dtype=src_v_caches.dtype,
        device="cpu",
        pin_memory=True,
    )
    staging_indices = torch.empty(
        (batch, heads, topk),
        dtype=torch.int64,
        device="cpu",
        pin_memory=True,
    )
    try:
        gather_ext = load_gather_ext()
    except Exception:
        gather_ext = None
    return staged_h2d_gather_kv(
        src_k_caches,
        src_v_caches,
        selected_indices,
        dst_k_caches,
        dst_v_caches,
        staging_k,
        staging_v,
        staging_indices,
        gather_ext=gather_ext,
    )

__all__ = ["h2d_gather_kv", "staged_h2d_gather_kv"]

