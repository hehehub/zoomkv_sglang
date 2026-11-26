"""Index attention must match a dense softmax over the same tokens, including graph replay."""

import torch

from sglang_zoomkv.kernels.sparse_attn.sparse_decode_attn_ops import (
    load_sparse_decode_attn,
    sparse_decode_attention,
)


def _reference(q, k, v, indices, sink, local_start, local_end, index_bias):
    scale = 128 ** -0.5
    b, hq, dim = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    outs = []
    for bi in range(b):
        heads = []
        for h in range(hq):
            kv = h // group
            cols = list(range(sink))
            cols.extend(range(local_start, local_end))
            extra = (indices[bi, kv] + index_bias).tolist()
            cols.extend(int(x) for x in extra)
            kk = k[bi, kv, cols].float()
            vv = v[bi, kv, cols].float()
            score = torch.matmul(kk, q[bi, h].float()) * scale
            prob = torch.softmax(score, dim=0)
            heads.append(torch.matmul(prob, vv))
        outs.append(torch.stack(heads, 0))
    return torch.stack(outs, 0).to(torch.bfloat16)


def _run(ext, q, k, v, indices, bounds, sink, bias, splits=8):
    b, hq, _ = q.shape
    partial_m = torch.empty(b, hq, splits, device=q.device)
    partial_l = torch.empty(b, hq, splits, device=q.device)
    partial_acc = torch.empty(b, hq, splits, 128, device=q.device)
    out = torch.empty_like(q)
    sparse_decode_attention(
        q, k, v, indices, bounds, partial_m, partial_l, partial_acc, out,
        sink_tokens=sink, index_bias=bias, splits=splits, kernel=ext,
    )
    return out


def main():
    torch.manual_seed(0)
    device = "cuda"
    ext = load_sparse_decode_attn()
    b, hq, hkv, s, k = 2, 8, 2, 2048, 64
    sink, local_start, local_end, bias = 64, 1400, 1900, 64
    q = torch.randn(b, hq, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(b, hkv, s, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn(b, hkv, s, 128, device=device, dtype=torch.bfloat16)
    indices = torch.randint(0, 1200, (b, hkv, k), device=device, dtype=torch.int64)
    bounds = torch.tensor([local_start, local_end], device=device, dtype=torch.int32)
    got = _run(ext, q, key, value, indices, bounds, sink, bias)
    ref = _reference(q, key, value, indices, sink, local_start, local_end, bias)
    err = (got.float() - ref.float()).abs().max().item()
    print(f"max abs err {err:.5f}")
    if err > 0.05:
        raise SystemExit(1)

    static_q = q.clone()
    static_bounds = bounds.clone()
    splits = 8
    partial_m = torch.empty(b, hq, splits, device=device)
    partial_l = torch.empty(b, hq, splits, device=device)
    partial_acc = torch.empty(b, hq, splits, 128, device=device)
    out = torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _args = dict(
            q=static_q, k=key, v=value, indices=indices, bounds=static_bounds,
            partial_m=partial_m, partial_l=partial_l, partial_acc=partial_acc, out=out,
            sink_tokens=sink, index_bias=bias, splits=splits, kernel=ext,
        )
        sparse_decode_attention(**_args)
        with torch.cuda.graph(graph):
            sparse_decode_attention(**_args)
    torch.cuda.current_stream().wait_stream(side)

    new_q = torch.randn_like(q)
    new_end = 2048
    ref2 = _reference(new_q, key, value, indices, sink, local_start, new_end, bias)
    static_q.copy_(new_q)
    static_bounds.copy_(torch.tensor([local_start, new_end], device=device, dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    err2 = (out.float() - ref2.float()).abs().max().item()
    print(f"replay max abs err {err2:.5f}")
    if err2 > 0.05:
        raise SystemExit(1)

    wide = torch.randint(0, 1200, (b, hkv, k + 400), device=device, dtype=torch.int64)
    wide.copy_(torch.nn.functional.pad(indices, (0, 400)))
    sliced = wide[:, :, :k]
    if sliced.is_contiguous():
        raise SystemExit("expected a strided index slice")
    got3 = _run(ext, q, key, value, sliced, bounds, sink, bias)
    err3 = (got3.float() - ref.float()).abs().max().item()
    print(f"strided index max abs err {err3:.5f}")
    if err3 > 0.05:
        raise SystemExit(1)
    for splits in (32, 64, 128):
        got_s = _run(ext, q, key, value, sliced, bounds, sink, bias, splits=splits)
        err_s = (got_s.float() - ref.float()).abs().max().item()
        print(f"splits{splits} max abs err {err_s:.5f}")
        if err_s > 0.05:
            raise SystemExit(1)

    # Offload shape: sink and local live in the packed GPU cache, retrieval tokens
    # were gathered into a side buffer and are not indexed in the main cache.
    sink_e, local_s, local_e = 4, 20, 28
    extra_n = 6
    idx_e = torch.zeros(b, hkv, 0, device=device, dtype=torch.int64)
    extra_k = torch.randn(b, hkv, extra_n, 128, device=device, dtype=torch.bfloat16)
    extra_v = torch.randn(b, hkv, extra_n, 128, device=device, dtype=torch.bfloat16)
    extra_count = torch.tensor([extra_n], device=device, dtype=torch.int32)
    bounds_e = torch.tensor([local_s, local_e], device=device, dtype=torch.int32)
    partial_m = torch.empty(b, hq, 8, device=device)
    partial_l = torch.empty(b, hq, 8, device=device)
    partial_acc = torch.empty(b, hq, 8, 128, device=device)
    out_e = torch.empty_like(q)
    sparse_decode_attention(
        q, key, value, idx_e, bounds_e, partial_m, partial_l, partial_acc, out_e,
        sink_tokens=sink_e, index_bias=0, splits=8, kernel=ext,
        extra_k=extra_k, extra_v=extra_v, extra_count=extra_count,
    )
    ref_parts = []
    for bi in range(b):
        heads = []
        for h in range(hq):
            kv = h // (hq // hkv)
            cols = list(range(sink_e)) + list(range(local_s, local_e))
            kk = torch.cat([key[bi, kv, cols].float(), extra_k[bi, kv].float()], 0)
            vv = torch.cat([value[bi, kv, cols].float(), extra_v[bi, kv].float()], 0)
            score = torch.matmul(kk, q[bi, h].float()) * (128 ** -0.5)
            prob = torch.softmax(score, dim=0)
            heads.append(torch.matmul(prob, vv))
        ref_parts.append(torch.stack(heads, 0))
    ref_e = torch.stack(ref_parts, 0).to(torch.bfloat16)
    err_e = (out_e.float() - ref_e.float()).abs().max().item()
    print(f"extra buffer max abs err {err_e:.5f}")
    if err_e > 0.05:
        raise SystemExit(1)

    # Paged pool: sequence position maps through a slot table into a shared KV buffer.
    nslots = b * s
    pool_k = torch.zeros(nslots, hkv, 128, device=device, dtype=torch.bfloat16)
    pool_v = torch.zeros(nslots, hkv, 128, device=device, dtype=torch.bfloat16)
    slot_table = torch.zeros(8, s, device=device, dtype=torch.int32)
    row_ids = torch.tensor([2, 6], device=device, dtype=torch.int32)
    perm_all = torch.randperm(nslots, device=device)
    for bi, row in enumerate(row_ids.tolist()):
        perm = perm_all[bi * s:(bi + 1) * s]
        pool_k[perm] = key[bi].permute(1, 0, 2)
        pool_v[perm] = value[bi].permute(1, 0, 2)
        slot_table[row, :s] = perm.to(torch.int32)
    partial_m = torch.empty(b, hq, 8, device=device)
    partial_l = torch.empty(b, hq, 8, device=device)
    partial_acc = torch.empty(b, hq, 8, 128, device=device)
    out_p = torch.empty_like(q)
    sparse_decode_attention(
        q, pool_k, pool_v, indices, bounds, partial_m, partial_l, partial_acc, out_p,
        sink_tokens=sink, index_bias=bias, splits=8, kernel=ext,
        slot_table=slot_table, row_ids=row_ids,
    )
    err_p = (out_p.float() - ref.float()).abs().max().item()
    print(f"paged pool max abs err {err_p:.5f}")
    if err_p > 0.05:
        raise SystemExit(1)

    def bench(splits):
        bb, hq, hkv, ss, kk = 4, 16, 4, 8192, 1024
        qn = torch.randn(bb, hq, 128, device=device, dtype=torch.bfloat16)
        kn = torch.randn(bb, hkv, ss, 128, device=device, dtype=torch.bfloat16)
        vn = torch.randn(bb, hkv, ss, 128, device=device, dtype=torch.bfloat16)
        idx = torch.randint(0, 6000, (bb, hkv, kk), device=device, dtype=torch.int64)
        wide_idx = torch.empty(bb, hkv, kk + 300, device=device, dtype=torch.int64)
        wide_idx[:, :, :kk] = idx
        idx = wide_idx[:, :, :kk]
        bd = torch.tensor([7000, 7600], device=device, dtype=torch.int32)
        partial_m = torch.empty(bb, hq, splits, device=device)
        partial_l = torch.empty(bb, hq, splits, device=device)
        partial_acc = torch.empty(bb, hq, splits, 128, device=device)
        out = torch.empty_like(qn)
        def once():
            sparse_decode_attention(
                qn, kn, vn, idx, bd, partial_m, partial_l, partial_acc, out,
                sink_tokens=64, index_bias=64, splits=splits, kernel=ext,
            )
        for _ in range(10):
            once()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        iters = 50
        for _ in range(iters):
            once()
        end.record()
        end.synchronize()
        print(f"bench splits {splits} us {(start.elapsed_time(end) / iters) * 1000:.1f}")

    for splits in (32, 64, 128):
        bench(splits)
    print("sparse decode attention matches reference")


if __name__ == "__main__":
    main()
