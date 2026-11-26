"""GPU-only decode attention that reads KV by index."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from sglang_zoomkv.kernels.jit import load_jit


@lru_cache(maxsize=1)
def load_sparse_decode_attn() -> Any:
    return load_jit(
        "sparse_decode_attn_ext",
        "sparse_decode_attn.cu",
        source_dir=Path(__file__).parent,
        cuda_flags=("-O3", "-std=c++20"),
    )


def sparse_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    bounds: torch.Tensor,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    out: torch.Tensor,
    *,
    sink_tokens: int,
    index_bias: int,
    splits: int,
    kernel: Any | None = None,
    extra_k: torch.Tensor | None = None,
    extra_v: torch.Tensor | None = None,
    extra_count: torch.Tensor | None = None,
    slot_table: torch.Tensor | None = None,
    row_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attend over sink ``[0, sink)``, local ``bounds``, indexed tokens, and an optional extra buffer.

    ``slot_table`` is the SGLang request-to-token map. With it, ``k`` and ``v`` are the
    paged pool ``[tokens, kv_heads, dim]`` and ``row_ids`` selects one row per batch element.
    """
    ext = kernel or load_sparse_decode_attn()
    args = (
        q, k, v, indices, bounds, partial_m, partial_l, partial_acc, out,
        int(sink_tokens), int(index_bias), int(splits),
    )
    if slot_table is None and extra_k is None:
        ext.sparse_decode_attn(*args)
    elif slot_table is None:
        ext.sparse_decode_attn(*args, extra_k, extra_v, extra_count)
    else:
        ext.sparse_decode_attn(
            *args, extra_k, extra_v, extra_count, slot_table, row_ids,
        )
    return out
