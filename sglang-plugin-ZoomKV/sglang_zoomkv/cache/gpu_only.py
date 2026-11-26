"""Small, device-agnostic state helpers for the full GPU sidecar."""

from __future__ import annotations

from typing import Any


def append_position(current_length: int, capacity: int) -> int:
    """Return the chronological append position or fail before overflow."""
    if current_length < 0 or current_length >= capacity:
        raise RuntimeError(
            f"GPU-only sidecar capacity exceeded: {current_length + 1} > {capacity}"
        )
    return current_length


def retrieval_to_cache_indices(
    retrieval_indices: Any,
    sink_size: int,
    retrieval_length: int,
) -> torch.Tensor:
    """Map retrieval-zone offsets to full-cache positions exactly once."""
    if str(retrieval_indices.dtype) not in {"torch.int64", "int64"}:
        raise ValueError("retrieval indices must be int64")
    if retrieval_indices.numel():
        if bool((retrieval_indices < 0).any()) or bool(
            (retrieval_indices >= retrieval_length).any()
        ):
            raise IndexError("retrieval index is outside the encoded retrieval zone")
    return retrieval_indices + sink_size


def steady_indices(
    sink_indices: Any,
    local_indices: Any,
) -> Any:
    """Join disjoint sink/local regions without adding retrieval tokens."""
    import torch

    return torch.cat((sink_indices, local_indices))


__all__ = ["append_position", "retrieval_to_cache_indices", "steady_indices"]
