"""ZoomKV sparse causal prefill.

Each query block keeps the sink, a local window, and the Quest top chunks of
the prefix. The block itself stays causal. Prefix keys are fully visible to
every query in the block, which matches a bottom-right causal mask when they
are concatenated in front of the block.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_causal_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sink: int,
    local: int,
    topk: int,
    block: int,
    scale: float,
    chunk: int = 16,
) -> torch.Tensor:
    """Return attention output shaped like ``q``: ``[B, S, Hq, D]``.

    ``k`` and ``v`` are ``[B, S, Hkv, D]``. Query heads must be a multiple of
    the KV heads.
    """
    if q.dim() != 4 or k.dim() != 4 or v.shape != k.shape:
        raise ValueError("q/k/v must be [B,S,H,D] and k/v shapes must match")
    batch, seq_len, q_heads, dim = q.shape
    kv_heads = int(k.shape[2])
    if q_heads % kv_heads != 0:
        raise ValueError(f"query heads {q_heads} are not divisible by kv heads {kv_heads}")
    if int(k.shape[1]) != seq_len or int(v.shape[1]) != seq_len:
        raise ValueError("k/v sequence length must match q")
    group = q_heads // kv_heads
    sink = max(0, min(int(sink), seq_len))
    local = max(1, int(local))
    block = max(1, int(block))
    topk = max(1, int(topk))
    chunk = max(1, int(chunk))

    q_h = q.transpose(1, 2).contiguous()
    k_h = k.transpose(1, 2).contiguous()
    v_h = v.transpose(1, 2).contiguous()
    # Quest bounds are anchored at the sink. Every later block reuses this
    # one reduction instead of scanning a longer prefix from scratch.
    bounds = _chunk_bounds(k_h[:, :, sink:, :], chunk) if sink < seq_len else None
    out = torch.empty(batch, q_heads, seq_len, dim, dtype=q.dtype, device=q.device)

    for start in range(0, seq_len, block):
        end = min(seq_len, start + block)
        rep = _block_query(q_h[:, :, start:end, :], group)
        prefix_k, prefix_v = _prefix_kv(
            k_h, v_h, start, sink, local, topk, chunk, rep, bounds
        )
        ctx_k = torch.cat((prefix_k, k_h[:, :, start:end, :]), dim=2)
        ctx_v = torch.cat((prefix_v, v_h[:, :, start:end, :]), dim=2)
        out[:, :, start:end, :] = _attend(
            q_h[:, :, start:end, :], ctx_k, ctx_v, scale
        )
    return out.transpose(1, 2).contiguous()


def _attend(
    q_block: torch.Tensor,
    ctx_k: torch.Tensor,
    ctx_v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Bottom-right causal attention of one query block against its context.

    The prefix is visible to every query. The block itself is causal. On CUDA
    those are two Flash calls merged by log-sum-exp. Padding the queries out
    to a square and running one causal kernel also attends the dummy rows, and
    that extra work is most of the gap versus dense FlashInfer at short lengths.
    """
    prefix = int(ctx_k.shape[2]) - int(q_block.shape[2])
    if prefix < 0:
        raise ValueError(
            f"context length {int(ctx_k.shape[2])} is shorter than the query block"
        )
    if q_block.is_cuda and q_block.dtype in (torch.float16, torch.bfloat16):
        return _attend_flash(q_block, ctx_k, ctx_v, prefix, scale)
    return _attend_padded(q_block, ctx_k, ctx_v, prefix, scale)


def _attend_flash(q_block, ctx_k, ctx_v, prefix: int, scale: float) -> torch.Tensor:
    block_k = ctx_k[:, :, prefix:, :]
    block_v = ctx_v[:, :, prefix:, :]
    out_block, lse_block = _flash(q_block, block_k, block_v, True, scale)
    if prefix == 0:
        return out_block
    out_prefix, lse_prefix = _flash(
        q_block, ctx_k[:, :, :prefix, :], ctx_v[:, :, :prefix, :], False, scale
    )
    winner = torch.maximum(lse_prefix, lse_block)
    weight_prefix = torch.exp(lse_prefix - winner).unsqueeze(-1)
    weight_block = torch.exp(lse_block - winner).unsqueeze(-1)
    return (out_prefix * weight_prefix + out_block * weight_block) / (
        weight_prefix + weight_block
    )


def _flash(q, k, v, causal: bool, scale: float):
    if k.shape[1] != q.shape[1]:
        group = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        q, k, v, 0.0, causal, scale=scale
    )
    return out, lse


def _attend_padded(q_block, ctx_k, ctx_v, prefix: int, scale: float) -> torch.Tensor:
    query = q_block
    if prefix:
        query = torch.cat(
            (
                q_block.new_zeros(
                    q_block.shape[0], q_block.shape[1], prefix, q_block.shape[3]
                ),
                q_block,
            ),
            dim=2,
        )
    attended = F.scaled_dot_product_attention(
        query,
        ctx_k,
        ctx_v,
        is_causal=True,
        scale=scale,
        enable_gqa=True,
    )
    if prefix:
        attended = attended[:, :, prefix:, :]
    return attended


def _block_query(q_block: torch.Tensor, group: int) -> torch.Tensor:
    """Mean query of one block, reduced to one vector per KV head."""
    if group != 1:
        batch, q_heads, length, dim = q_block.shape
        q_block = q_block.reshape(batch, q_heads // group, group, length, dim).mean(dim=2)
    return q_block.mean(dim=2)


def _chunk_bounds(keys: torch.Tensor, chunk: int):
    """Per-chunk min and max of ``keys`` ``[B, H, S, D]``, from the front."""
    length = int(keys.shape[2])
    n_chunks = length // chunk
    if n_chunks <= 0:
        return None
    usable = n_chunks * chunk
    view = keys[:, :, :usable, :].reshape(
        keys.shape[0], keys.shape[1], n_chunks, chunk, keys.shape[-1]
    )
    return view.amin(dim=3), view.amax(dim=3)


def _prefix_kv(
    k_h: torch.Tensor,
    v_h: torch.Tensor,
    start: int,
    sink: int,
    local: int,
    topk: int,
    chunk: int,
    rep: torch.Tensor,
    bounds,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keys and values strictly before ``start`` that this block may see."""
    if start <= 0:
        empty_k = k_h[:, :, :0, :]
        return empty_k, v_h[:, :, :0, :]
    sink_end = min(sink, start)
    local_start = max(sink_end, start - local)
    parts_k = [k_h[:, :, :sink_end, :]]
    parts_v = [v_h[:, :, :sink_end, :]]
    if local_start > sink_end:
        picked_k, picked_v = _quest_pick(
            k_h[:, :, sink_end:local_start, :],
            v_h[:, :, sink_end:local_start, :],
            rep,
            topk,
            chunk,
            bounds if sink_end == sink else None,
        )
        parts_k.append(picked_k)
        parts_v.append(picked_v)
    if start > local_start:
        parts_k.append(k_h[:, :, local_start:start, :])
        parts_v.append(v_h[:, :, local_start:start, :])
    return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)


def _quest_pick(
    keys: torch.Tensor,
    values: torch.Tensor,
    rep: torch.Tensor,
    topk: int,
    chunk: int,
    bounds=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the same token positions from K and V with one Quest score."""
    length = int(keys.shape[2])
    if length <= topk:
        return keys, values
    n_chunks = length // chunk
    if n_chunks <= 0:
        return keys[:, :, :topk, :], values[:, :, :topk, :]
    if bounds is None:
        usable = n_chunks * chunk
        view = keys[:, :, :usable, :].reshape(
            keys.shape[0], keys.shape[1], n_chunks, chunk, keys.shape[-1]
        )
        cmin = view.amin(dim=3)
        cmax = view.amax(dim=3)
    else:
        cmin, cmax = bounds
        cmin = cmin[:, :, :n_chunks, :]
        cmax = cmax[:, :, :n_chunks, :]
    score = torch.maximum(rep.unsqueeze(2) * cmax, rep.unsqueeze(2) * cmin).sum(dim=-1)
    n_keep = min(n_chunks, max(1, (topk + chunk - 1) // chunk))
    chosen = score.topk(n_keep, dim=-1).indices.sort(dim=-1).values
    token_base = chosen.unsqueeze(-1) * chunk
    token_off = torch.arange(chunk, device=keys.device)
    token_idx = (token_base + token_off).reshape(keys.shape[0], keys.shape[1], n_keep * chunk)
    if int(token_idx.shape[-1]) > topk:
        token_idx = token_idx[:, :, :topk]
    index = token_idx.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
    return torch.gather(keys, 2, index), torch.gather(values, 2, index)
