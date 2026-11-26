"""Non-UVA CPU KV gather followed by an explicit host-to-device copy."""

from __future__ import annotations

from typing import Any

import torch


def staged_h2d_gather_kv(
    src_k_caches: torch.Tensor,
    src_v_caches: torch.Tensor,
    selected_indices: torch.Tensor,
    dst_k_caches: torch.Tensor,
    dst_v_caches: torch.Tensor,
    staging_k: torch.Tensor,
    staging_v: torch.Tensor,
    staging_indices: torch.Tensor,
    *,
    gather_ext: Any | None = None,
) -> None:
    """Gather pinned CPU KV without a CUDA kernel dereferencing host memory.

    The transfer boundaries are intentional: indices are copied from CUDA to
    pinned host memory and completed before CPU code reads them; gathered KV is
    then copied explicitly from pinned staging buffers to preallocated CUDA
    outputs.
    """
    if src_k_caches.shape != src_v_caches.shape or src_k_caches.ndim != 4:
        raise ValueError("source K/V must have equal [B,H,N,D] shapes")
    if selected_indices.ndim != 3 or selected_indices.dtype != torch.int64:
        raise ValueError("selected_indices must be int64 [B,H,K]")
    if not selected_indices.is_cuda:
        raise ValueError("selected_indices must be on CUDA")
    if not (src_k_caches.is_cpu and src_v_caches.is_cpu):
        raise ValueError("source K/V must be on CPU")
    if not (src_k_caches.is_pinned() and src_v_caches.is_pinned()):
        raise ValueError("source K/V must be pinned")
    if src_k_caches.dtype != torch.bfloat16:
        raise ValueError(f"ZoomKV staged gather requires bfloat16, got {src_k_caches.dtype}")
    if src_k_caches.shape[-1] not in (128, 256):
        raise ValueError(
            "ZoomKV staged gather requires head_dim=128 or 256, "
            f"got {src_k_caches.shape[-1]}"
        )

    batch, heads, _, dim = src_k_caches.shape
    topk = selected_indices.shape[-1]
    expected_indices = (batch, heads, topk)
    expected_kv = (batch, heads, topk, dim)
    if selected_indices.shape != expected_indices:
        raise ValueError("selected_indices batch/head shape does not match source K/V")
    for name, tensor, cpu in (
        ("staging_k", staging_k, True),
        ("staging_v", staging_v, True),
        ("dst_k", dst_k_caches, False),
        ("dst_v", dst_v_caches, False),
    ):
        if tensor.shape[0] != batch or tensor.shape[1] != heads:
            raise ValueError(f"{name} batch/head shape mismatch")
        if tensor.shape[-2] < topk or tensor.shape[-1] != dim:
            raise ValueError(f"{name} capacity/shape mismatch")
        if tensor.is_cpu != cpu:
            raise ValueError(f"{name} must be on {'CPU' if cpu else 'CUDA'}")
    if staging_indices.shape[:2] != (batch, heads) or staging_indices.shape[-1] < topk:
        raise ValueError("staging_indices capacity/shape mismatch")
    if not staging_indices.is_cpu:
        raise ValueError("staging_indices must be on CPU")
    if not (staging_indices.is_pinned() and staging_k.is_pinned() and staging_v.is_pinned()):
        raise ValueError("CPU staging buffers must be pinned")
    if staging_indices.dtype != torch.int64:
        raise ValueError("staging_indices must be int64")
    if not all(
        tensor.dtype == src_k_caches.dtype
        for tensor in (src_v_caches, staging_k, staging_v, dst_k_caches, dst_v_caches)
    ):
        raise ValueError("source, staging, and destination KV dtypes must match")

    host_indices = staging_indices[:, :, :topk]
    host_k = staging_k[:, :, :topk, :]
    host_v = staging_v[:, :, :topk, :]

    # D2H completion is required before CPU gather code may read the indices.
    host_indices.copy_(selected_indices, non_blocking=True)
    torch.cuda.current_stream(selected_indices.device).synchronize()

    if gather_ext is not None:
        gather_ext.gather_kv_out_cpu(src_k_caches, host_indices, host_k)
        gather_ext.gather_kv_out_cpu(src_v_caches, host_indices, host_v)
    else:
        expanded = host_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
        torch.gather(src_k_caches, 2, expanded, out=host_k)
        torch.gather(src_v_caches, 2, expanded, out=host_v)

    dst_k_caches[:, :, :topk, :].copy_(host_k, non_blocking=True)
    dst_v_caches[:, :, :topk, :].copy_(host_v, non_blocking=True)


__all__ = ["staged_h2d_gather_kv"]
