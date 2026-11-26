// Decode attention over sink, a contiguous local window, and retrieved positions.
// One replay reads KV in place. No gather, cat, or intermediate KV buffer.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kDim = 128;
constexpr float kScale = 0.08838834764831845f;  // 1/sqrt(128)

struct Float4 {
    float x, y, z, w;
};

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ Float4 load_bf16x4(const __nv_bfloat16* p) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    const float2 f0 = __bfloat1622float2(a);
    const float2 f1 = __bfloat1622float2(b);
    return {f0.x, f0.y, f1.x, f1.y};
}

__device__ __forceinline__ int64_t sparse_token_pos(
    int t,
    int sink_tokens,
    int n_local,
    int local_start,
    const int64_t* indices,
    long long idx_base,
    long long idx_stride_k,
    int index_bias) {
    if (t < sink_tokens) {
        return t;
    }
    if (t < sink_tokens + n_local) {
        return static_cast<int64_t>(local_start) + (t - sink_tokens);
    }
    const int ki = t - sink_tokens - n_local;
    return indices[idx_base + static_cast<long long>(ki) * idx_stride_k] + index_bias;
}

__device__ __forceinline__ void load_kv_token(
    int t,
    int n_tokens,
    int sink_tokens,
    int n_local,
    int local_start,
    int K,
    const int64_t* indices,
    long long idx_base,
    long long idx_stride_k,
    int index_bias,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    long long kv_base,
    int lane,
    int S,
    int b,
    int kv,
    int extra_count,
    const __nv_bfloat16* extra_k,
    const __nv_bfloat16* extra_v,
    long long extra_stride_b,
    long long extra_stride_h,
    long long extra_stride_s,
    const int32_t* slots,
    long long slot_stride,
    const int32_t* row_ids,
    int pool_heads,
    bool& valid,
    Float4& kk,
    Float4& vv) {
    if (t >= n_tokens) {
        valid = false;
        return;
    }
    const int index_end = sink_tokens + n_local + K;
    if (t >= index_end) {
        const int ei = t - index_end;
        if (extra_k == nullptr || ei >= extra_count) {
            valid = false;
            return;
        }
        const long long off = static_cast<long long>(b) * extra_stride_b
            + static_cast<long long>(kv) * extra_stride_h
            + static_cast<long long>(ei) * extra_stride_s
            + lane * 4;
        valid = true;
        kk = load_bf16x4(extra_k + off);
        vv = load_bf16x4(extra_v + off);
        return;
    }
    const int64_t pos = sparse_token_pos(
        t, sink_tokens, n_local, local_start, indices, idx_base, idx_stride_k, index_bias);
    valid = pos >= 0 && pos < static_cast<int64_t>(S);
    if (!valid) {
        return;
    }
    long long off;
    if (slots != nullptr) {
        const int row = row_ids[b];
        const int slot = slots[static_cast<long long>(row) * slot_stride + pos];
        off = (static_cast<long long>(slot) * pool_heads + kv) * kDim + lane * 4;
    } else {
        off = (kv_base + pos) * kDim + lane * 4;
    }
    kk = load_bf16x4(k + off);
    vv = load_bf16x4(v + off);
}

// One warp owns one KV head and every query head in its group, so K and V are
// loaded once per token instead of once per query head.
template <int Group>
__global__ void __launch_bounds__(128, 2)
sparse_partial_gqa(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int64_t* __restrict__ indices,
    long long idx_stride_b,
    long long idx_stride_h,
    long long idx_stride_k,
    const int32_t* __restrict__ bounds,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    const int32_t* __restrict__ extra_count_ptr,
    const __nv_bfloat16* __restrict__ extra_k,
    const __nv_bfloat16* __restrict__ extra_v,
    long long extra_stride_b,
    long long extra_stride_h,
    long long extra_stride_s,
    int B,
    int Hq,
    int Hkv,
    int S,
    int K,
    int sink_tokens,
    int index_bias,
    int splits,
    const int32_t* __restrict__ slots,
    long long slot_stride,
    const int32_t* __restrict__ row_ids) {
    const int warp = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int total = B * Hkv * splits;
    if (warp >= total) {
        return;
    }
    const int split = warp % splits;
    const int kv = (warp / splits) % Hkv;
    const int b = warp / (splits * Hkv);
    const int32_t local_start = bounds[0];
    const int32_t local_end = bounds[1];
    const int n_local = local_end > local_start ? (local_end - local_start) : 0;
    const int extra_count = extra_count_ptr != nullptr ? extra_count_ptr[0] : 0;
    const int n_tokens = sink_tokens + n_local + K + extra_count;

    Float4 qv[Group];
#pragma unroll
    for (int g = 0; g < Group; ++g) {
        const int hq = kv * Group + g;
        qv[g] = load_bf16x4(q + (static_cast<long long>(b) * Hq + hq) * kDim + lane * 4);
    }
    const long long kv_base = (static_cast<long long>(b) * Hkv + kv) * S;
    const long long idx_base = static_cast<long long>(b) * idx_stride_b
        + static_cast<long long>(kv) * idx_stride_h;

    float acc[Group][4];
    float m_i[Group];
    float l_i[Group];
#pragma unroll
    for (int g = 0; g < Group; ++g) {
        acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
        m_i[g] = -1e30f;
        l_i[g] = 0.f;
    }

    bool valid = false;
    Float4 kk, vv;
    load_kv_token(
        split, n_tokens, sink_tokens, n_local, local_start, K, indices, idx_base, idx_stride_k,
        index_bias, k, v, kv_base, lane, S, b, kv, extra_count, extra_k, extra_v,
        extra_stride_b, extra_stride_h, extra_stride_s, slots, slot_stride, row_ids, Hkv,
        valid, kk, vv);
    for (int t = split; t < n_tokens; t += splits) {
        bool next_valid = false;
        Float4 next_kk, next_vv;
        load_kv_token(
            t + splits, n_tokens, sink_tokens, n_local, local_start, K, indices, idx_base,
            idx_stride_k, index_bias, k, v, kv_base, lane, S, b, kv, extra_count, extra_k,
            extra_v, extra_stride_b, extra_stride_h, extra_stride_s, slots, slot_stride,
            row_ids, Hkv, next_valid, next_kk, next_vv);
        if (valid) {
            float score[Group];
#pragma unroll
            for (int g = 0; g < Group; ++g) {
                const float partial = qv[g].x * kk.x + qv[g].y * kk.y + qv[g].z * kk.z + qv[g].w * kk.w;
                score[g] = warp_sum(partial) * kScale;
            }
#pragma unroll
            for (int g = 0; g < Group; ++g) {
                const float m_next = fmaxf(m_i[g], score[g]);
                const float alpha = __expf(m_i[g] - m_next);
                const float p = __expf(score[g] - m_next);
                l_i[g] = l_i[g] * alpha + p;
                acc[g][0] = acc[g][0] * alpha + p * vv.x;
                acc[g][1] = acc[g][1] * alpha + p * vv.y;
                acc[g][2] = acc[g][2] * alpha + p * vv.z;
                acc[g][3] = acc[g][3] * alpha + p * vv.w;
                m_i[g] = m_next;
            }
        }
        valid = next_valid;
        kk = next_kk;
        vv = next_vv;
    }

#pragma unroll
    for (int g = 0; g < Group; ++g) {
        const int hq = kv * Group + g;
        const long long slot = (static_cast<long long>(b) * Hq + hq) * splits + split;
        if (lane == 0) {
            partial_m[slot] = m_i[g];
            partial_l[slot] = l_i[g];
        }
        float* acc_out = partial_acc + slot * kDim + lane * 4;
        acc_out[0] = acc[g][0];
        acc_out[1] = acc[g][1];
        acc_out[2] = acc[g][2];
        acc_out[3] = acc[g][3];
    }
}

__global__ void sparse_finalize_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    __nv_bfloat16* __restrict__ out,
    int Hq,
    int splits) {
    const int b = blockIdx.x;
    const int hq = blockIdx.y;
    const int tid = threadIdx.x;
    const long long head = static_cast<long long>(b) * Hq + hq;
    float m_i = -1e30f;
    for (int s = 0; s < splits; ++s) {
        m_i = fmaxf(m_i, partial_m[head * splits + s]);
    }
    float l_i = 0.f;
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) {
        const long long slot = head * splits + s;
        const float alpha = __expf(partial_m[slot] - m_i);
        l_i += partial_l[slot] * alpha;
        acc += partial_acc[slot * kDim + tid] * alpha;
    }
    const float value = l_i > 0.f ? acc / l_i : 0.f;
    out[head * kDim + tid] = __float2bfloat16(value);
}

constexpr int kDim256 = 256;
constexpr float kScale256 = 0.0625f;  // 1/sqrt(256)

struct Float8 {
    float v[8];
};

__device__ __forceinline__ Float8 load_bf16x8(const __nv_bfloat16* p) {
    const Float4 a = load_bf16x4(p);
    const Float4 b = load_bf16x4(p + 4);
    Float8 out;
    out.v[0] = a.x;
    out.v[1] = a.y;
    out.v[2] = a.z;
    out.v[3] = a.w;
    out.v[4] = b.x;
    out.v[5] = b.y;
    out.v[6] = b.z;
    out.v[7] = b.w;
    return out;
}

__device__ __forceinline__ void load_kv_token_256(
    int t,
    int n_tokens,
    int sink_tokens,
    int n_local,
    int local_start,
    int K,
    const int64_t* indices,
    long long idx_base,
    long long idx_stride_k,
    int index_bias,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    long long kv_base,
    int lane,
    int S,
    int b,
    int kv,
    int extra_count,
    const __nv_bfloat16* extra_k,
    const __nv_bfloat16* extra_v,
    long long extra_stride_b,
    long long extra_stride_h,
    long long extra_stride_s,
    const int32_t* slots,
    long long slot_stride,
    const int32_t* row_ids,
    int pool_heads,
    bool& valid,
    Float8& kk,
    Float8& vv) {
    if (t >= n_tokens) {
        valid = false;
        return;
    }
    const int index_end = sink_tokens + n_local + K;
    if (t >= index_end) {
        const int ei = t - index_end;
        if (extra_k == nullptr || ei >= extra_count) {
            valid = false;
            return;
        }
        const long long off = static_cast<long long>(b) * extra_stride_b
            + static_cast<long long>(kv) * extra_stride_h
            + static_cast<long long>(ei) * extra_stride_s
            + lane * 8;
        valid = true;
        kk = load_bf16x8(extra_k + off);
        vv = load_bf16x8(extra_v + off);
        return;
    }
    const int64_t pos = sparse_token_pos(
        t, sink_tokens, n_local, local_start, indices, idx_base, idx_stride_k, index_bias);
    valid = pos >= 0 && pos < static_cast<int64_t>(S);
    if (!valid) {
        return;
    }
    long long off;
    if (slots != nullptr) {
        const int row = row_ids[b];
        const int slot = slots[static_cast<long long>(row) * slot_stride + pos];
        off = (static_cast<long long>(slot) * pool_heads + kv) * kDim256 + lane * 8;
    } else {
        off = (kv_base + pos) * kDim256 + lane * 8;
    }
    kk = load_bf16x8(k + off);
    vv = load_bf16x8(v + off);
}

template <int Group>
__global__ void __launch_bounds__(128, 2)
sparse_partial_gqa_256(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int64_t* __restrict__ indices,
    long long idx_stride_b,
    long long idx_stride_h,
    long long idx_stride_k,
    const int32_t* __restrict__ bounds,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    const int32_t* __restrict__ extra_count_ptr,
    const __nv_bfloat16* __restrict__ extra_k,
    const __nv_bfloat16* __restrict__ extra_v,
    long long extra_stride_b,
    long long extra_stride_h,
    long long extra_stride_s,
    int B,
    int Hq,
    int Hkv,
    int S,
    int K,
    int sink_tokens,
    int index_bias,
    int splits,
    const int32_t* __restrict__ slots,
    long long slot_stride,
    const int32_t* __restrict__ row_ids) {
    const int warp = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int total = B * Hkv * splits;
    if (warp >= total) {
        return;
    }
    const int split = warp % splits;
    const int kv = (warp / splits) % Hkv;
    const int b = warp / (splits * Hkv);
    const int32_t local_start = bounds[0];
    const int32_t local_end = bounds[1];
    const int n_local = local_end > local_start ? (local_end - local_start) : 0;
    const int extra_count = extra_count_ptr != nullptr ? extra_count_ptr[0] : 0;
    const int n_tokens = sink_tokens + n_local + K + extra_count;

    Float8 qv[Group];
#pragma unroll
    for (int g = 0; g < Group; ++g) {
        const int hq = kv * Group + g;
        qv[g] = load_bf16x8(q + (static_cast<long long>(b) * Hq + hq) * kDim256 + lane * 8);
    }
    const long long kv_base = (static_cast<long long>(b) * Hkv + kv) * S;
    const long long idx_base = static_cast<long long>(b) * idx_stride_b
        + static_cast<long long>(kv) * idx_stride_h;

    float acc[Group][8];
    float m_i[Group];
    float l_i[Group];
#pragma unroll
    for (int g = 0; g < Group; ++g) {
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            acc[g][d] = 0.f;
        }
        m_i[g] = -1e30f;
        l_i[g] = 0.f;
    }

    bool valid = false;
    Float8 kk, vv;
    load_kv_token_256(
        split, n_tokens, sink_tokens, n_local, local_start, K, indices, idx_base, idx_stride_k,
        index_bias, k, v, kv_base, lane, S, b, kv, extra_count, extra_k, extra_v,
        extra_stride_b, extra_stride_h, extra_stride_s, slots, slot_stride, row_ids, Hkv,
        valid, kk, vv);
    for (int t = split; t < n_tokens; t += splits) {
        bool next_valid = false;
        Float8 next_kk, next_vv;
        load_kv_token_256(
            t + splits, n_tokens, sink_tokens, n_local, local_start, K, indices, idx_base,
            idx_stride_k, index_bias, k, v, kv_base, lane, S, b, kv, extra_count, extra_k,
            extra_v, extra_stride_b, extra_stride_h, extra_stride_s, slots, slot_stride,
            row_ids, Hkv, next_valid, next_kk, next_vv);
        if (valid) {
            float score[Group];
#pragma unroll
            for (int g = 0; g < Group; ++g) {
                float partial = 0.f;
#pragma unroll
                for (int d = 0; d < 8; ++d) {
                    partial += qv[g].v[d] * kk.v[d];
                }
                score[g] = warp_sum(partial) * kScale256;
            }
#pragma unroll
            for (int g = 0; g < Group; ++g) {
                const float m_next = fmaxf(m_i[g], score[g]);
                const float alpha = __expf(m_i[g] - m_next);
                const float p = __expf(score[g] - m_next);
                l_i[g] = l_i[g] * alpha + p;
#pragma unroll
                for (int d = 0; d < 8; ++d) {
                    acc[g][d] = acc[g][d] * alpha + p * vv.v[d];
                }
                m_i[g] = m_next;
            }
        }
        valid = next_valid;
        kk = next_kk;
        vv = next_vv;
    }

#pragma unroll
    for (int g = 0; g < Group; ++g) {
        const int hq = kv * Group + g;
        const long long slot = (static_cast<long long>(b) * Hq + hq) * splits + split;
        if (lane == 0) {
            partial_m[slot] = m_i[g];
            partial_l[slot] = l_i[g];
        }
        float* acc_out = partial_acc + slot * kDim256 + lane * 8;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            acc_out[d] = acc[g][d];
        }
    }
}

__global__ void sparse_finalize_kernel_256(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    __nv_bfloat16* __restrict__ out,
    int Hq,
    int splits) {
    const int b = blockIdx.x;
    const int hq = blockIdx.y;
    const int tid = threadIdx.x;
    const long long head = static_cast<long long>(b) * Hq + hq;
    float m_i = -1e30f;
    for (int s = 0; s < splits; ++s) {
        m_i = fmaxf(m_i, partial_m[head * splits + s]);
    }
    float l_i = 0.f;
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) {
        const long long slot = head * splits + s;
        const float alpha = __expf(partial_m[slot] - m_i);
        l_i += partial_l[slot] * alpha;
        acc += partial_acc[slot * kDim256 + tid] * alpha;
    }
    const float value = l_i > 0.f ? acc / l_i : 0.f;
    out[head * kDim256 + tid] = __float2bfloat16(value);
}

void sparse_decode_attn(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor indices,
    torch::Tensor bounds,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    torch::Tensor out,
    int64_t sink_tokens,
    int64_t index_bias,
    int64_t splits,
    c10::optional<torch::Tensor> extra_k,
    c10::optional<torch::Tensor> extra_v,
    c10::optional<torch::Tensor> extra_count,
    c10::optional<torch::Tensor> slot_table,
    c10::optional<torch::Tensor> row_ids) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "sparse attn tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q/k/v must be contiguous");
    TORCH_CHECK(bounds.is_contiguous(), "bounds must be contiguous");
    TORCH_CHECK(indices.dim() == 3, "indices must be [B,Hkv,K]");
    TORCH_CHECK(partial_m.is_contiguous() && partial_l.is_contiguous() && partial_acc.is_contiguous(), "partials must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16, "k/v must be bf16");
    TORCH_CHECK(indices.scalar_type() == torch::kLong, "indices must be int64");
    TORCH_CHECK(bounds.scalar_type() == torch::kInt && bounds.numel() == 2, "bounds must be int32[2]");
    TORCH_CHECK(q.dim() == 3 && (q.size(2) == 128 || q.size(2) == 256), "q must be [B,Hq,128 or 256]");
    const int dim = static_cast<int>(q.size(2));
    const bool paged = slot_table.has_value() && slot_table->defined() && slot_table->numel() > 0;
    const int B = static_cast<int>(q.size(0));
    const int Hq = static_cast<int>(q.size(1));
    int Hkv = 0;
    int S = 0;
    const int32_t* slots_ptr = nullptr;
    const int32_t* row_ptr = nullptr;
    long long slot_stride = 0;
    if (paged) {
        TORCH_CHECK(k.dim() == 3 && v.sizes() == k.sizes(), "paged k/v must be [tokens,Hkv,D]");
        TORCH_CHECK(k.size(2) == dim, "head_dim must be 128 or 256");
        TORCH_CHECK(slot_table->is_cuda() && slot_table->scalar_type() == torch::kInt && slot_table->dim() == 2, "slot table must be CUDA int32 [rows,context]");
        TORCH_CHECK(row_ids.has_value() && row_ids->is_cuda() && row_ids->scalar_type() == torch::kInt && row_ids->numel() >= B, "row_ids must be CUDA int32[B]");
        Hkv = static_cast<int>(k.size(1));
        S = static_cast<int>(slot_table->size(1));
        slots_ptr = slot_table->data_ptr<int32_t>();
        slot_stride = slot_table->stride(0);
        row_ptr = row_ids->data_ptr<int32_t>();
    } else {
        TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "q [B,Hq,D], k/v [B,Hkv,S,D]");
        TORCH_CHECK(k.size(3) == dim && v.sizes() == k.sizes(), "head_dim must be 128 or 256");
        TORCH_CHECK(k.size(0) == B, "batch/kv shape mismatch");
        Hkv = static_cast<int>(k.size(1));
        S = static_cast<int>(k.size(2));
    }
    const int K = static_cast<int>(indices.size(2));
    TORCH_CHECK(indices.size(0) == B && indices.size(1) == Hkv, "batch/kv shape mismatch");
    TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "query heads must be divisible by kv heads");
    const int group = Hq / Hkv;
    TORCH_CHECK(group == 1 || group == 2 || group == 4 || group == 6 || group == 8, "query group must be 1, 2, 4, 6, or 8");
    TORCH_CHECK(splits > 0 && splits <= 128, "splits must be in 1..128");
    TORCH_CHECK(partial_m.size(0) == B && partial_m.size(1) == Hq && partial_m.size(2) == splits, "partial_m shape");
    TORCH_CHECK(partial_acc.size(3) == dim, "partial_acc head_dim");
    TORCH_CHECK(out.sizes() == q.sizes(), "out shape must match q");

    c10::cuda::CUDAGuard guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int warps = B * Hkv * static_cast<int>(splits);
    const int threads = 128;
    const int blocks = (warps + (threads / 32) - 1) / (threads / 32);
    const auto q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    const auto k_ptr = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
    const auto v_ptr = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
    const auto idx_ptr = indices.data_ptr<int64_t>();
    const auto bounds_ptr = bounds.data_ptr<int32_t>();
    const auto m_ptr = partial_m.data_ptr<float>();
    const auto l_ptr = partial_l.data_ptr<float>();
    const auto acc_ptr = partial_acc.data_ptr<float>();
    const int sink = static_cast<int>(sink_tokens);
    const int bias = static_cast<int>(index_bias);
    const int nsplit = static_cast<int>(splits);
    const long long stride_b = indices.stride(0);
    const long long stride_h = indices.stride(1);
    const long long stride_k = indices.stride(2);
    const bool has_extra = extra_k.has_value() && extra_k->numel() > 0;
    const int32_t* extra_count_ptr = nullptr;
    const __nv_bfloat16* extra_k_ptr = nullptr;
    const __nv_bfloat16* extra_v_ptr = nullptr;
    long long extra_stride_b = 0, extra_stride_h = 0, extra_stride_s = 0;
    if (has_extra) {
        TORCH_CHECK(extra_v.has_value() && extra_v->sizes() == extra_k->sizes(), "extra V must match extra K");
        TORCH_CHECK(extra_k->is_cuda() && extra_k->scalar_type() == torch::kBFloat16, "extra K must be CUDA bf16");
        TORCH_CHECK(extra_count.has_value() && extra_count->is_cuda() && extra_count->numel() == 1, "extra_count must be a CUDA scalar");
        TORCH_CHECK(extra_k->size(0) == B && extra_k->size(1) == Hkv && extra_k->size(3) == dim, "extra K shape");
        extra_count_ptr = extra_count->data_ptr<int32_t>();
        extra_k_ptr = reinterpret_cast<const __nv_bfloat16*>(extra_k->data_ptr());
        extra_v_ptr = reinterpret_cast<const __nv_bfloat16*>(extra_v->data_ptr());
        extra_stride_b = extra_k->stride(0);
        extra_stride_h = extra_k->stride(1);
        extra_stride_s = extra_k->stride(2);
    }
    auto launch = [&](auto kernel) {
        kernel<<<blocks, threads, 0, stream>>>(
            q_ptr, k_ptr, v_ptr, idx_ptr, stride_b, stride_h, stride_k, bounds_ptr,
            m_ptr, l_ptr, acc_ptr, extra_count_ptr, extra_k_ptr, extra_v_ptr,
            extra_stride_b, extra_stride_h, extra_stride_s,
            B, Hq, Hkv, S, K, sink, bias, nsplit, slots_ptr, slot_stride, row_ptr);
    };
    if (dim == 128) {
        if (group == 1) {
            launch(sparse_partial_gqa<1>);
        } else if (group == 2) {
            launch(sparse_partial_gqa<2>);
        } else if (group == 4) {
            launch(sparse_partial_gqa<4>);
        } else if (group == 6) {
            launch(sparse_partial_gqa<6>);
        } else {
            launch(sparse_partial_gqa<8>);
        }
        sparse_finalize_kernel<<<dim3(B, Hq), kDim, 0, stream>>>(
            partial_m.data_ptr<float>(),
            partial_l.data_ptr<float>(),
            partial_acc.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            Hq,
            static_cast<int>(splits));
    } else {
        if (group == 1) {
            launch(sparse_partial_gqa_256<1>);
        } else if (group == 2) {
            launch(sparse_partial_gqa_256<2>);
        } else if (group == 4) {
            launch(sparse_partial_gqa_256<4>);
        } else if (group == 6) {
            launch(sparse_partial_gqa_256<6>);
        } else {
            launch(sparse_partial_gqa_256<8>);
        }
        sparse_finalize_kernel_256<<<dim3(B, Hq), 256, 0, stream>>>(
            partial_m.data_ptr<float>(),
            partial_l.data_ptr<float>(),
            partial_acc.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            Hq,
            static_cast<int>(splits));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "sparse_decode_attn",
        &sparse_decode_attn,
        "Index-based decode attention",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("indices"),
        py::arg("bounds"),
        py::arg("partial_m"),
        py::arg("partial_l"),
        py::arg("partial_acc"),
        py::arg("out"),
        py::arg("sink_tokens"),
        py::arg("index_bias"),
        py::arg("splits"),
        py::arg("extra_k") = py::none(),
        py::arg("extra_v") = py::none(),
        py::arg("extra_count") = py::none(),
        py::arg("slot_table") = py::none(),
        py::arg("row_ids") = py::none());
}
