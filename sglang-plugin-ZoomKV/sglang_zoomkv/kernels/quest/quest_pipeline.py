"""Hierarchical Quest + CDS as a fixed sequence of parallel kernels.

The sequence is what a per-layer CUDA graph replays. Every tensor the kernels
read or write is supplied by the caller so a replay uses the same addresses.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_zoomkv.kernels.kivi.kivi_qk_dot_ops import (
    partial_chunk_kivi_qk_dense_sparse_cuda,
)
from sglang_zoomkv.kernels.rerank.rerank import (
    dense_mask_from_topk_cuda,
    partial_chunk_density_scores_cuda,
)
from sglang_zoomkv.kernels.nvtx_ranges import (
    FINE_RERANK,
    LARGE_CHUNK_SELECT,
    SMALL_CHUNK_SELECT,
    nvtx_stage,
)
from sglang_zoomkv.kernels.topk.float_topk_gather_concat_ops import (
    float_topk_gather_concat_3d,
)
from sglang_zoomkv.kernels.topk.float_topk_ops import float_topk_3d


def run_hierarchical_quest(
    *,
    quest_kernel: Any,
    rerank_kernel: Any,
    kivi_kernel: Any,
    raw_q: torch.Tensor,
    large_min: torch.Tensor,
    large_max: torch.Tensor,
    large_valid: torch.Tensor,
    small_min: torch.Tensor,
    small_max: torch.Tensor,
    large_scores: torch.Tensor,
    sub_scores: torch.Tensor,
    chunk_idx: torch.Tensor,
    centroid: torch.Tensor,
    density: torch.Tensor,
    dense_mask: torch.Tensor,
    packed_k: torch.Tensor,
    out_scores: torch.Tensor,
    out_indices: torch.Tensor,
    final_indices: torch.Tensor,
    skipped: torch.Tensor,
    n_large: int,
    nk_large: int,
    factor: int,
    nk_small: int,
    n_chunks: int,
    n_dense: int,
    dense_topk: int,
    sparse_topk: int,
    actual_topk: int,
    sk_count: int,
    keepalive: list | None = None,
) -> torch.Tensor:
    """Run one head-batch of hierarchical Quest. Append captured outputs to ``keepalive``."""

    def keep(tensor: torch.Tensor) -> torch.Tensor:
        if keepalive is not None:
            keepalive.append(tensor)
        return tensor

    # Host nodes, not Python NVTX: a captured decode graph replays these.
    with nvtx_stage(LARGE_CHUNK_SELECT):
        quest_kernel.quest_chunk_score(
            raw_q, large_min, large_max, large_scores, int(n_large), large_valid,
        )
        large_idx = keep(float_topk_3d(large_scores, int(nk_large)))
        large_idx.clamp_(min=0, max=int(n_large) - 1)
    with nvtx_stage(SMALL_CHUNK_SELECT):
        quest_kernel.quest_sub_chunk_score(
            raw_q, small_min, small_max, large_idx, sub_scores, int(nk_large), int(factor),
        )
        sub_pos = keep(float_topk_3d(sub_scores, int(nk_small)))
        quest_kernel.quest_map_back(
            large_idx, sub_pos, chunk_idx, int(factor), int(n_chunks),
        )
    with nvtx_stage(FINE_RERANK):
        partial_chunk_density_scores_cuda(
            chunk_idx, centroid, raw_q, density, kernel=rerank_kernel,
        )
        dense_pos = keep(float_topk_3d(density, int(n_dense)))
        dense_mask_from_topk_cuda(dense_pos, dense_mask, kernel=rerank_kernel)
        partial_chunk_kivi_qk_dense_sparse_cuda(
            chunk_ids=chunk_idx,
            dense_mask=dense_mask,
            packed_K=packed_k,
            chunk_min=small_min,
            chunk_max=small_max,
            raw_q=raw_q,
            dense_topk=int(dense_topk),
            sparse_topk=int(sparse_topk),
            group_size=int(packed_k.size(-1)),
            out_scores=out_scores,
            out_indices=out_indices,
            kernel=kivi_kernel,
        )
        float_topk_gather_concat_3d(
            out_scores, out_indices, skipped, final_indices, int(actual_topk),
        )
    return keep(final_indices[:, :, : int(actual_topk) + int(sk_count)])
