"""Fused hierarchical Quest must select the same tokens as the split kernels."""

import torch

from sglang_zoomkv.kernels.kivi.kivi_qk_dot_ops import partial_chunk_kivi_qk_dense_sparse_cuda
from sglang_zoomkv.kernels.quest.quest_fused_ops import load_fused_quest
from sglang_zoomkv.kernels.quest.quest_ops import load_kernel_module
from sglang_zoomkv.kernels.rerank.rerank import (
    dense_mask_from_topk_cuda,
    partial_chunk_density_scores_cuda,
)
from sglang_zoomkv.kernels.topk.float_topk_gather_concat_ops import float_topk_gather_concat_3d
from sglang_zoomkv.kernels.topk.float_topk_ops import float_topk_3d


def _reference(q, large_min, large_max, large_valid, small_min, small_max,
               centroid, packed, skipped, n_large, nk_large, factor, nk_small,
               n_chunks, n_dense, dense_topk, sparse_topk, actual_topk):
    quest = load_kernel_module()
    bs, kv, _ = q.shape
    large_scores = torch.empty(bs, kv, n_large, device=q.device, dtype=torch.float32)
    quest.quest_chunk_score(q, large_min, large_max, large_scores, n_large, large_valid)
    large_idx = float_topk_3d(large_scores, nk_large).clamp(min=0, max=n_large - 1)
    n_sub = nk_large * factor
    sub_scores = torch.empty(bs, kv, n_sub, device=q.device, dtype=torch.float32)
    quest.quest_sub_chunk_score(q, small_min, small_max, large_idx, sub_scores, nk_large, factor)
    sub_pos = float_topk_3d(sub_scores, nk_small)
    chunk_idx = torch.empty(bs, kv, nk_small, device=q.device, dtype=torch.int64)
    quest.quest_map_back(large_idx, sub_pos, chunk_idx, factor, n_chunks)

    density = torch.empty(bs, kv, nk_small, device=q.device, dtype=torch.float32)
    partial_chunk_density_scores_cuda(chunk_idx, centroid, q, density)
    dense_pos = float_topk_3d(density, n_dense)
    dense_mask = torch.zeros(bs, kv, nk_small, device=q.device, dtype=torch.bool)
    dense_mask_from_topk_cuda(dense_pos, dense_mask)

    cand = nk_small * packed.size(-1)
    out_scores = torch.empty(bs, kv, cand, device=q.device, dtype=torch.float32)
    out_indices = torch.empty(bs, kv, cand, device=q.device, dtype=torch.int64)
    partial_chunk_kivi_qk_dense_sparse_cuda(
        chunk_idx, dense_mask, packed, small_min, small_max, q,
        dense_topk, sparse_topk, packed.size(-1), out_scores, out_indices,
    )
    final = torch.empty(bs, kv, actual_topk + skipped.size(-1), device=q.device, dtype=torch.int64)
    float_topk_gather_concat_3d(out_scores, out_indices, skipped, final, actual_topk)
    return final


def main():
    # Small shape, then a 20k-token retrieval shape with the production ratios.
    run_case(n_large=32, factor=4, chunk=8, nk_large=10, nk_small=12,
             n_dense=2, dense_topk=8, sparse_topk=4, actual_topk=16, sk=3)
    run_case(n_large=80, factor=32, chunk=8, nk_large=24, nk_small=231,
             n_dense=23, dense_topk=8, sparse_topk=8, actual_topk=256, sk=5)
    print("fused quest matches split kernels")


def run_case(*, n_large, factor, chunk, nk_large, nk_small, n_dense,
             dense_topk, sparse_topk, actual_topk, sk):
    torch.manual_seed(0)
    device = "cuda"
    bs, kv, head = 2, 4, 128
    n_chunks = n_large * factor

    q = torch.randn(bs, kv, head, device=device, dtype=torch.bfloat16)
    large_min = torch.randn(bs, kv, n_large, head, device=device, dtype=torch.bfloat16)
    large_max = large_min + torch.rand(bs, kv, n_large, head, device=device, dtype=torch.bfloat16).abs()
    large_valid = torch.ones(bs, kv, n_large, device=device, dtype=torch.bool)
    large_valid[:, :, 3] = False
    small_min = torch.randn(bs, kv, n_chunks, head, device=device, dtype=torch.bfloat16)
    small_max = small_min + torch.rand(bs, kv, n_chunks, head, device=device, dtype=torch.bfloat16).abs()
    centroid = torch.randn(bs, kv, n_chunks, head, device=device, dtype=torch.bfloat16)
    packed = torch.randint(0, 2**31 - 1, (bs, kv, n_chunks, head // 8, chunk), device=device, dtype=torch.int32)
    skipped = torch.arange(sk, device=device, dtype=torch.int64).view(1, 1, sk).expand(bs, kv, sk).contiguous()

    large_scores = torch.empty(bs, kv, n_large, device=device, dtype=torch.float32)
    sub_scores = torch.empty(bs, kv, nk_large * factor, device=device, dtype=torch.float32)
    scratch = torch.empty(bs, kv, nk_small * chunk, device=device, dtype=torch.int64)
    chunk_idx = torch.empty(bs, kv, nk_small, device=device, dtype=torch.int64)
    density = torch.empty(bs, kv, nk_small, device=device, dtype=torch.float32)
    dense_mask = torch.zeros(bs, kv, nk_small, device=device, dtype=torch.bool)
    kivi_scores = torch.empty(bs, kv, nk_small * chunk, device=device, dtype=torch.float32)
    final = torch.empty(bs, kv, actual_topk + sk, device=device, dtype=torch.int64)

    ext = load_fused_quest()
    ext.fused_hierarchical_quest(
        q, large_min, large_max, large_valid, small_min, small_max,
        large_scores, sub_scores, scratch, chunk_idx, centroid, density, dense_mask,
        packed, kivi_scores, final, skipped,
        n_large, nk_large, factor, n_chunks, nk_small, n_chunks, n_dense,
        dense_topk, sparse_topk, n_chunks, actual_topk, sk,
    )
    torch.cuda.synchronize()
    ref = _reference(
        q, large_min, large_max, large_valid, small_min, small_max,
        centroid, packed, skipped, n_large, nk_large, factor, nk_small,
        n_chunks, n_dense, dense_topk, sparse_topk, actual_topk,
    )
    torch.cuda.synchronize()

    got = final[:, :, :actual_topk].sort(dim=-1).values
    exp = ref[:, :, :actual_topk].sort(dim=-1).values
    tail_ok = torch.equal(final[:, :, actual_topk:], ref[:, :, actual_topk:])
    if torch.equal(got, exp) and tail_ok:
        print(f"match n_large={n_large} chunk={chunk} topk={actual_topk}")
        return
    mismatch = (got != exp).sum().item()
    print("topk mismatches", mismatch, "of", got.numel())
    print("fused", got[0, 0].tolist())
    print("split", exp[0, 0].tolist())
    print("tail fused", final[0, 0, actual_topk:].tolist(), "split", ref[0, 0, actual_topk:].tolist())
    raise SystemExit(1)


if __name__ == "__main__":
    main()
