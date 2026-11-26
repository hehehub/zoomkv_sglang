"""CUDA graph replay of hierarchical Quest must match the eager kernel sequence."""

import torch

from sglang_zoomkv.kernels.kivi.kivi_qk_dot_ops import _kivi_kernel
from sglang_zoomkv.kernels.quest.quest_ops import load_kernel_module
from sglang_zoomkv.kernels.quest.quest_pipeline import run_hierarchical_quest
from sglang_zoomkv.kernels.rerank.rerank import load_kernel_module as load_rerank


def _alloc(device, bs, kv, n_large, factor, chunk, nk_large, nk_small, sk):
    n_chunks = n_large * factor
    head = 128
    q = torch.randn(bs, kv, head, device=device, dtype=torch.bfloat16)
    large_min = torch.randn(bs, kv, n_large, head, device=device, dtype=torch.bfloat16)
    large_max = large_min + torch.rand_like(large_min).abs()
    large_valid = torch.ones(bs, kv, n_large, device=device, dtype=torch.bool)
    large_valid[:, :, 1] = False
    small_min = torch.randn(bs, kv, n_chunks, head, device=device, dtype=torch.bfloat16)
    small_max = small_min + torch.rand_like(small_min).abs()
    centroid = torch.randn(bs, kv, n_chunks, head, device=device, dtype=torch.bfloat16)
    packed = torch.randint(0, 2**31 - 1, (bs, kv, n_chunks, head // 8, chunk), device=device, dtype=torch.int32)
    skipped = torch.arange(sk, device=device, dtype=torch.int64).view(1, 1, sk).expand(bs, kv, sk).contiguous()
    buffers = {
        "large_scores": torch.empty(bs, kv, n_large, device=device, dtype=torch.float32),
        "sub_scores": torch.empty(bs, kv, nk_large * factor, device=device, dtype=torch.float32),
        "chunk_idx": torch.empty(bs, kv, nk_small, device=device, dtype=torch.int64),
        "density": torch.empty(bs, kv, nk_small, device=device, dtype=torch.float32),
        "dense_mask": torch.empty(bs, kv, nk_small, device=device, dtype=torch.bool),
        "out_scores": torch.empty(bs, kv, nk_small * chunk, device=device, dtype=torch.float32),
        "out_indices": torch.empty(bs, kv, nk_small * chunk, device=device, dtype=torch.int64),
        "final_indices": torch.empty(bs, kv, 256 + sk, device=device, dtype=torch.int64),
    }
    static = {
        "raw_q": q,
        "large_min": large_min,
        "large_max": large_max,
        "large_valid": large_valid,
        "small_min": small_min,
        "small_max": small_max,
        "centroid": centroid,
        "packed_k": packed,
        "skipped": skipped,
        "n_large": n_large,
        "nk_large": nk_large,
        "factor": factor,
        "nk_small": nk_small,
        "n_chunks": n_chunks,
        "n_dense": 4,
        "dense_topk": min(8, chunk),
        "sparse_topk": min(4, chunk),
        "actual_topk": 16,
        "sk_count": sk,
    }
    return static, buffers


def _run(static, buffers, quest, rerank, kivi, q, keepalive=None):
    args = dict(static)
    args.update(buffers)
    args["raw_q"] = q
    return run_hierarchical_quest(
        quest_kernel=quest,
        rerank_kernel=rerank,
        kivi_kernel=kivi,
        keepalive=keepalive,
        **args,
    )


def main():
    torch.manual_seed(0)
    device = "cuda"
    bs, kv = 2, 4
    n_large, factor, chunk, nk_large, nk_small, sk = 32, 4, 8, 10, 12, 3
    static, buffers = _alloc(device, bs, kv, n_large, factor, chunk, nk_large, nk_small, sk)
    quest = load_kernel_module()
    rerank = load_rerank("rerank_topk.cu", "rerank_topk")
    kivi = _kivi_kernel()

    eager_buf = {name: tensor.clone() for name, tensor in buffers.items()}
    ref = _run(static, eager_buf, quest, rerank, kivi, static["raw_q"]).clone()

    static_q = static["raw_q"].clone()
    keepalive = []
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.stream(side):
        _run(static, buffers, quest, rerank, kivi, static_q, keepalive)
        with torch.cuda.graph(graph, pool=pool):
            final = _run(static, buffers, quest, rerank, kivi, static_q, keepalive)
    torch.cuda.current_stream().wait_stream(side)
    got = final.clone()
    assert torch.equal(got.sort(-1).values, ref.sort(-1).values), "capture result diverged"
    assert torch.equal(got[:, :, 16:], ref[:, :, 16:])

    for step in range(2):
        new_q = torch.randn_like(static_q)
        ref2 = _run(static, eager_buf, quest, rerank, kivi, new_q).clone()
        static_q.copy_(new_q)
        graph.replay()
        torch.cuda.synchronize()
        got2 = final.clone()
        if not torch.equal(got2.sort(-1).values, ref2.sort(-1).values):
            print("replay mismatch", step)
            print(got2[0, 0].tolist())
            print(ref2[0, 0].tolist())
            raise SystemExit(1)
        if not torch.equal(got2[:, :, 16:], ref2[:, :, 16:]):
            raise SystemExit("skipped tail mismatch")
    print("quest cuda graph replay matches eager kernels")


if __name__ == "__main__":
    main()
