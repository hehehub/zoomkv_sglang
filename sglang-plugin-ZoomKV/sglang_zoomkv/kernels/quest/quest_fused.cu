// One-launch hierarchical Quest + CDS.
//
// Replaces the per-layer chain
//   quest_chunk_score, float_topk, quest_sub_chunk_score, float_topk,
//   quest_map_back, partial_chunk_density_scores, float_topk,
//   mask_from_topk, partial_chunk_kivi_qk_dense_sparse,
//   float_topk_gather_concat
// with a single block per (batch, kv head). Scoring reductions follow the
// 128-thread tree in quest.cu / the density kernel. Top-k is the same
// 4-round radix select as float_topk.cu. KIVI dequant matches
// kivi_qk_dot.cu (bf16 mul/add, fp32 accumulation, .x then .y).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kBlock = 1024;
constexpr int kHeadDim = 128;
constexpr int kGroups = kBlock / kHeadDim;  // 8
constexpr int kRadix = 256;
constexpr float kInvalidScore = -1e30f;

__device__ __forceinline__ uint32_t float_to_ordered_uint32(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint8_t extract_byte(uint32_t x, int byte_offset) {
    return static_cast<uint8_t>((x >> byte_offset) & 0xFF);
}

__device__ void naive_topk(int64_t* indices, int length, int k) {
    const int tid = threadIdx.x;
    for (int i = tid; i < k; i += kBlock) {
        indices[i] = (i < length) ? i : -1;
    }
}

// Radix select copied from float_topk.cu. `in` is one contiguous row.
// Writes k positions (sentinel -1) to `out`. All threads in the block must
// call this together.
__device__ void radix_topk_row(
    const float* __restrict__ in,
    int64_t* __restrict__ out,
    int length,
    int k,
    int* s_input_idx
) {
    const int tid = threadIdx.x;
    for (int i = tid; i < k; i += kBlock) {
        out[i] = -1;
    }
    __syncthreads();

    if (length <= k) {
        naive_topk(out, length, k);
        return;
    }

    __shared__ int s_histogram_buf[2][kRadix + 1];
    __shared__ int s_counter;
    __shared__ int s_threshold_bin;
    __shared__ int s_num_input[2];
    __shared__ int s_last_remain;

    int* s_histogram = s_histogram_buf[0];
    const int smem_idx_size = (64 * 1024 / static_cast<int>(sizeof(int))) / 2;
    int topk_remain = k;

    if (tid < kRadix + 1) {
        s_histogram_buf[0][tid] = 0;
        s_histogram_buf[1][tid] = 0;
    }
    if (tid == 0) {
        s_counter = 0;
        s_num_input[0] = 0;
    }
    __syncthreads();

    for (int idx = tid; idx < length; idx += kBlock) {
        uint32_t ordered = float_to_ordered_uint32(in[idx]);
        uint8_t bin = extract_byte(ordered, 24);
        atomicAdd(&s_histogram_buf[0][bin], 1);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (tid < kRadix) {
            int j = 1 << i;
            int k_idx = i & 1;
            int value = s_histogram_buf[k_idx][tid];
            if (tid < kRadix - j) {
                value += s_histogram_buf[k_idx][tid + j];
            }
            s_histogram_buf[k_idx ^ 1][tid] = value;
        }
        __syncthreads();
    }
    s_histogram = s_histogram_buf[0];

    if (tid < kRadix && s_histogram[tid] > topk_remain && s_histogram[tid + 1] <= topk_remain) {
        s_threshold_bin = tid;
    }
    __syncthreads();

    const int threshold_bin_r0 = s_threshold_bin;
    topk_remain -= s_histogram[threshold_bin_r0 + 1];

    if (topk_remain == 0) {
        for (int idx = tid; idx < length; idx += kBlock) {
            uint32_t ordered = float_to_ordered_uint32(in[idx]);
            uint8_t bin = extract_byte(ordered, 24);
            if (bin > threshold_bin_r0) {
                int pos = atomicAdd(&s_counter, 1);
                if (pos < k) out[pos] = idx;
            }
        }
        return;
    }

    if (tid < kRadix + 1) {
        s_histogram_buf[0][tid] = 0;
        s_histogram_buf[1][tid] = 0;
    }
    __syncthreads();

    for (int idx = tid; idx < length; idx += kBlock) {
        float val = in[idx];
        uint32_t ordered = float_to_ordered_uint32(val);
        uint8_t bin = extract_byte(ordered, 24);
        if (bin > threshold_bin_r0) {
            int pos = atomicAdd(&s_counter, 1);
            if (pos < k) out[pos] = idx;
        } else if (bin == threshold_bin_r0) {
            int pos = atomicAdd(&s_num_input[0], 1);
            if (pos < smem_idx_size) {
                s_input_idx[pos] = idx;
                uint8_t sub_bin = extract_byte(ordered, 16);
                atomicAdd(&s_histogram_buf[0][sub_bin], 1);
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int round = 1; round < 4; ++round) {
        const int r_idx = (round - 1) % 2;
        const int num_input = min(s_num_input[r_idx], smem_idx_size);
        const int byte_offset = 24 - round * 8;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (tid < kRadix) {
                int j = 1 << i;
                int k_idx = i & 1;
                int value = s_histogram_buf[k_idx][tid];
                if (tid < kRadix - j) {
                    value += s_histogram_buf[k_idx][tid + j];
                }
                s_histogram_buf[k_idx ^ 1][tid] = value;
            }
            __syncthreads();
        }
        s_histogram = s_histogram_buf[0];

        if (tid < kRadix && s_histogram[tid] > topk_remain && s_histogram[tid + 1] <= topk_remain) {
            s_threshold_bin = tid;
            s_last_remain = topk_remain - s_histogram[tid + 1];
        }
        if (tid == 0) {
            s_num_input[r_idx ^ 1] = 0;
        }
        __syncthreads();

        const int threshold_bin = s_threshold_bin;
        topk_remain -= s_histogram[threshold_bin + 1];

        if (topk_remain == 0) {
            for (int i = tid; i < num_input; i += kBlock) {
                int idx = s_input_idx[r_idx * smem_idx_size + i];
                uint32_t ordered = float_to_ordered_uint32(in[idx]);
                uint8_t bin = extract_byte(ordered, byte_offset);
                if (bin > threshold_bin) {
                    int pos = atomicAdd(&s_counter, 1);
                    if (pos < k) out[pos] = idx;
                }
            }
            return;
        }

        if (tid < kRadix + 1) {
            s_histogram_buf[0][tid] = 0;
            s_histogram_buf[1][tid] = 0;
        }
        __syncthreads();

        for (int i = tid; i < num_input; i += kBlock) {
            int idx = s_input_idx[r_idx * smem_idx_size + i];
            float val = in[idx];
            uint32_t ordered = float_to_ordered_uint32(val);
            uint8_t bin = extract_byte(ordered, byte_offset);
            if (bin > threshold_bin) {
                int pos = atomicAdd(&s_counter, 1);
                if (pos < k) out[pos] = idx;
            } else if (bin == threshold_bin) {
                if (round == 3) {
                    int pos = atomicAdd(&s_last_remain, -1);
                    if (pos > 0) {
                        out[k - pos] = idx;
                    }
                } else {
                    int pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
                    if (pos < smem_idx_size) {
                        s_input_idx[(r_idx ^ 1) * smem_idx_size + pos] = idx;
                        uint8_t sub_bin = extract_byte(ordered, byte_offset - 8);
                        atomicAdd(&s_histogram_buf[0][sub_bin], 1);
                    }
                }
            }
        }
        __syncthreads();
    }
}

struct FusedParams {
    const __nv_bfloat16* q;
    const __nv_bfloat16* large_min;
    const __nv_bfloat16* large_max;
    const uint8_t* large_valid;
    const __nv_bfloat16* small_min;
    const __nv_bfloat16* small_max;
    float* large_scores;
    float* sub_scores;
    int64_t* scratch;
    int64_t* chunk_idx;
    const __nv_bfloat16* centroid;
    float* density;
    uint8_t* dense_mask;
    const int32_t* packed_k;
    float* kivi_scores;
    int64_t* final_indices;
    const int64_t* skipped;

    int kv_heads;
    int n_large;
    int nk_large;
    int factor;
    int n_small_cap;
    int nk_small;
    int n_chunks;
    int n_dense;
    int dense_topk;
    int sparse_topk;
    int group_size;
    int n_pack;
    int n_packed;
    int n_centroid;
    int actual_topk;
    int sk_count;
    int scratch_stride;

    int64_t q_b, q_h, q_d;
    int64_t lmin_b, lmin_h, lmin_c, lmin_d;
    int64_t lmax_b, lmax_h, lmax_c, lmax_d;
    int64_t lvalid_b, lvalid_h, lvalid_c;
    int64_t smin_b, smin_h, smin_c, smin_d;
    int64_t smax_b, smax_h, smax_c, smax_d;
    int64_t lscore_b, lscore_h, lscore_c;
    int64_t sscore_b, sscore_h, sscore_c;
    int64_t scratch_b, scratch_h;
    int64_t chunk_b, chunk_h, chunk_k;
    int64_t cent_b, cent_h, cent_c, cent_d;
    int64_t dens_b, dens_h, dens_k;
    int64_t mask_b, mask_h, mask_k;
    int64_t pk_b, pk_h, pk_c, pk_pack;
    int64_t ks_b, ks_h, ks_k;
    int64_t fin_b, fin_h;
    int64_t sk_b, sk_h, sk_k;
};

template <int HeadDim>
__device__ void kivi_chunks(
    const FusedParams& p,
    const __nv_bfloat16* q,
    const int64_t* chunk_ids,
    const uint8_t* dense_mask,
    const __nv_bfloat16* cmin,
    const __nv_bfloat16* cmax,
    const int32_t* packed,
    float* scores,
    int64_t* indices,
    int nk
) {
    const int tid = threadIdx.x;
    const int chunk = p.group_size;
    const int chunks_per_warp = 32 / chunk;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int sub_id = lane_id / chunk;
    const int lane_in_chunk = lane_id - sub_id * chunk;
    const int warps = kBlock / 32;
    const int step = warps * chunks_per_warp;

    __shared__ __nv_bfloat16 q_sm[HeadDim];
    if (tid < HeadDim) {
        q_sm[tid] = q[tid * p.q_d];
    }
    __syncthreads();

    for (int base = 0; base < nk; base += step) {
        const int chunk_slot = base + warp_id * chunks_per_warp + sub_id;
        const bool valid_slot = chunk_slot < nk;
        int64_t real_chunk_id = 0;
        bool is_dense = false;
        if (valid_slot) {
            real_chunk_id = chunk_ids[chunk_slot * p.chunk_k];
            is_dense = dense_mask[chunk_slot * p.mask_k] != 0;
        }
        const bool in_range = valid_slot && real_chunk_id >= 0 && real_chunk_id < p.n_packed;
        int token_idx = static_cast<int>(real_chunk_id) * chunk + lane_in_chunk;
        float score_f = kInvalidScore;
        if (in_range) {
            const __nv_bfloat16* mn =
                cmin + real_chunk_id * p.smin_c;
            const __nv_bfloat16* mx =
                cmax + real_chunk_id * p.smax_c;
            const int32_t* pack =
                packed + real_chunk_id * p.pk_c + lane_in_chunk;
            const __nv_bfloat16 levels = __float2bfloat16(15.f);
            float acc = 0.f;
            for (int i = 0; i < p.n_pack; ++i) {
                // Scalar loads: a uint4 cast here faults when shared q or a
                // per-chunk min row is only 2-byte aligned.
                __nv_bfloat16 qv[8], mnv[8], mxv[8], scv[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    qv[j] = q_sm[i * 8 + j];
                    mnv[j] = mn[(i * 8 + j) * p.smin_d];
                    mxv[j] = mx[(i * 8 + j) * p.smax_d];
                    scv[j] = __hdiv(__hsub(mxv[j], mnv[j]), levels);
                }
                const uint32_t bits = static_cast<uint32_t>(__ldg(pack + i * p.pk_pack));
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const uint32_t code_lo = (bits >> (pair * 8 + 0)) & 0xFu;
                    const uint32_t code_hi = (bits >> (pair * 8 + 4)) & 0xFu;
                    const __nv_bfloat162 code_pair = __halves2bfloat162(
                        __int2bfloat16_rn(static_cast<int>(code_lo)),
                        __int2bfloat16_rn(static_cast<int>(code_hi)));
                    const __nv_bfloat162 sc_pair = __halves2bfloat162(scv[pair * 2], scv[pair * 2 + 1]);
                    const __nv_bfloat162 mn_pair = __halves2bfloat162(mnv[pair * 2], mnv[pair * 2 + 1]);
                    const __nv_bfloat162 q_pair = __halves2bfloat162(qv[pair * 2], qv[pair * 2 + 1]);
                    const __nv_bfloat162 mul_pair = __hmul2(code_pair, sc_pair);
                    const __nv_bfloat162 k_pair = __hadd2(mul_pair, mn_pair);
                    const __nv_bfloat162 prod_pair = __hmul2(k_pair, q_pair);
                    acc += __low2float(prod_pair);
                    acc += __high2float(prod_pair);
                }
            }
            score_f = acc;
        }

#pragma unroll
        for (int size = 2; size <= chunk; size <<= 1) {
            for (int j = size >> 1; j >= 1; j >>= 1) {
                float other_score = __shfl_xor_sync(0xffffffff, score_f, j);
                int other_idx = __shfl_xor_sync(0xffffffff, token_idx, j);
                const bool desc = ((lane_in_chunk & size) == 0);
                const bool is_lower = ((lane_in_chunk & j) == 0);
                const bool want_larger = (desc == is_lower);
                const bool do_swap = want_larger ? (other_score > score_f)
                                                 : (other_score < score_f);
                if (do_swap) {
                    score_f = other_score;
                    token_idx = other_idx;
                }
            }
        }

        if (valid_slot) {
            const int out_base = chunk_slot * chunk + lane_in_chunk;
            const int keep = is_dense ? p.dense_topk : p.sparse_topk;
            if (lane_in_chunk < keep) {
                scores[out_base] = score_f;
                indices[out_base] = static_cast<int64_t>(token_idx);
            } else {
                scores[out_base] = kInvalidScore;
                indices[out_base] = 0;
            }
        }
    }
    __syncthreads();
}

template <int HeadDim>
__global__ void fused_hierarchical_quest_kernel(FusedParams p) {
    constexpr int kHeadDim = HeadDim;
    constexpr int kGroups = kBlock / HeadDim;
    extern __shared__ int s_input_idx[];
    __shared__ float red[kGroups][kHeadDim];
    const int bh = blockIdx.x;
    const int b = bh / p.kv_heads;
    const int h = bh - b * p.kv_heads;
    const int tid = threadIdx.x;

    const __nv_bfloat16* q = p.q + b * p.q_b + h * p.q_h;
    float* large_scores = p.large_scores + b * p.lscore_b + h * p.lscore_h;
    float* sub_scores = p.sub_scores + b * p.sscore_b + h * p.sscore_h;
    int64_t* scratch = p.scratch + b * p.scratch_b + h * p.scratch_h;
    int64_t* chunk_idx = p.chunk_idx + b * p.chunk_b + h * p.chunk_h;
    float* density = p.density + b * p.dens_b + h * p.dens_h;
    uint8_t* dense_mask = p.dense_mask + b * p.mask_b + h * p.mask_h;
    float* kivi_scores = p.kivi_scores + b * p.ks_b + h * p.ks_h;
    int64_t* final_out = p.final_indices + b * p.fin_b + h * p.fin_h;

    // ----- large-chunk Quest scores -----
    {
        const int group = tid / kHeadDim;
        const int lane = tid - group * kHeadDim;
        const __nv_bfloat16* lmin = p.large_min + b * p.lmin_b + h * p.lmin_h;
        const __nv_bfloat16* lmax = p.large_max + b * p.lmax_b + h * p.lmax_h;
        const uint8_t* valid = p.large_valid + b * p.lvalid_b + h * p.lvalid_h;
        for (int base = 0; base < p.n_large; base += kGroups) {
            const int chunk = base + group;
            const bool in = chunk < p.n_large;
            const bool keep = in && valid[chunk * p.lvalid_c];
            float term = 0.f;
            if (keep) {
                const float qv = __bfloat162float(q[lane * p.q_d]);
                const __nv_bfloat16* cmin = lmin + static_cast<int64_t>(chunk) * p.lmin_c;
                const __nv_bfloat16* cmax = lmax + static_cast<int64_t>(chunk) * p.lmax_c;
                const float lo = __bfloat162float(cmin[lane * p.lmin_d]);
                const float hi = __bfloat162float(cmax[lane * p.lmax_d]);
                term = fmaxf(qv * hi, qv * lo);
            }
            red[group][lane] = term;
            __syncthreads();
#pragma unroll
            for (int offset = kHeadDim / 2; offset > 0; offset >>= 1) {
                if (lane < offset) red[group][lane] += red[group][lane + offset];
                __syncthreads();
            }
            if (lane == 0 && in) {
                large_scores[chunk * p.lscore_c] = keep ? red[group][0] : kInvalidScore;
            }
            __syncthreads();
        }
    }

    // large_scores may have stride. Radix wants a contiguous row. Production
    // buffers are contiguous (stride == 1). Enforce that on the host.
    radix_topk_row(large_scores, scratch, p.n_large, p.nk_large, s_input_idx);
    __syncthreads();
    for (int i = tid; i < p.nk_large; i += kBlock) {
        int64_t v = scratch[i];
        if (v < 0) v = 0;
        if (v > p.n_large - 1) v = p.n_large - 1;
        scratch[i] = v;
    }
    __syncthreads();

    // ----- sub-chunk scores for the selected large chunks -----
    {
        const int group = tid / kHeadDim;
        const int lane = tid - group * kHeadDim;
        const int n_sub = p.nk_large * p.factor;
        const __nv_bfloat16* smin = p.small_min + b * p.smin_b + h * p.smin_h;
        const __nv_bfloat16* smax = p.small_max + b * p.smax_b + h * p.smax_h;
        for (int base = 0; base < n_sub; base += kGroups) {
            const int slot = base + group;
            const bool in = slot < n_sub;
            float term = 0.f;
            bool keep = false;
            if (in) {
                const int large_slot = slot / p.factor;
                const int sub = slot - large_slot * p.factor;
                const int64_t parent = scratch[large_slot];
                const int64_t chunk = parent * p.factor + sub;
                keep = parent >= 0 && chunk >= 0 && chunk < p.n_small_cap;
                if (keep) {
                    const float qv = __bfloat162float(q[lane * p.q_d]);
                    const __nv_bfloat16* cmin = smin + chunk * p.smin_c;
                    const __nv_bfloat16* cmax = smax + chunk * p.smax_c;
                    const float lo = __bfloat162float(cmin[lane * p.smin_d]);
                    const float hi = __bfloat162float(cmax[lane * p.smax_d]);
                    term = fmaxf(qv * hi, qv * lo);
                }
            }
            red[group][lane] = term;
            __syncthreads();
#pragma unroll
            for (int offset = kHeadDim / 2; offset > 0; offset >>= 1) {
                if (lane < offset) red[group][lane] += red[group][lane + offset];
                __syncthreads();
            }
            if (lane == 0 && in) {
                sub_scores[slot * p.sscore_c] = keep ? red[group][0] : kInvalidScore;
            }
            __syncthreads();
        }
    }

    int64_t* sub_pos = scratch + p.nk_large;
    radix_topk_row(sub_scores, sub_pos, p.nk_large * p.factor, p.nk_small, s_input_idx);
    __syncthreads();

    // ----- map back to small-chunk ids -----
    const int span = p.nk_large * p.factor;
    for (int k = tid; k < p.nk_small; k += kBlock) {
        int64_t pos = sub_pos[k];
        int64_t safe = pos;
        if (safe < 0) safe = 0;
        else if (safe > span - 1) safe = span - 1;
        const int64_t large_slot = safe / p.factor;
        const int64_t sub = safe - large_slot * p.factor;
        int64_t parent = 0;
        if (large_slot >= 0 && large_slot < p.nk_large) {
            parent = scratch[large_slot];
        }
        int64_t mapped = parent * static_cast<int64_t>(p.factor) + sub;
        if (p.n_chunks > 0) {
            if (mapped < 0) mapped = 0;
            else if (mapped >= p.n_chunks) mapped = p.n_chunks - 1;
        } else {
            mapped = 0;
        }
        chunk_idx[k * p.chunk_k] = mapped;
    }
    __syncthreads();

    // ----- centroid density on the selected small chunks -----
    {
        const int group = tid / kHeadDim;
        const int lane = tid - group * kHeadDim;
        const __nv_bfloat16* cent = p.centroid + b * p.cent_b + h * p.cent_h;
        for (int base = 0; base < p.nk_small; base += kGroups) {
            const int slot = base + group;
            const bool in = slot < p.nk_small;
            float term = 0.f;
            bool keep = false;
            if (in) {
                const int64_t chunk_id = chunk_idx[slot * p.chunk_k];
                keep = chunk_id >= 0 && chunk_id < p.n_centroid;
                if (keep) {
                    const float qv = __bfloat162float(q[lane * p.q_d]);
                    const float cv = __bfloat162float(
                        cent[chunk_id * p.cent_c + lane * p.cent_d]);
                    term = qv * cv;
                }
            }
            red[group][lane] = term;
            __syncthreads();
#pragma unroll
            for (int offset = kHeadDim / 2; offset > 0; offset >>= 1) {
                if (lane < offset) red[group][lane] += red[group][lane + offset];
                __syncthreads();
            }
            if (lane == 0 && in) {
                density[slot * p.dens_k] = keep ? red[group][0] : kInvalidScore;
            }
            __syncthreads();
        }
    }

    // Density positions reuse scratch[0:n_dense]. large_idx is no longer read.
    radix_topk_row(density, scratch, p.nk_small, p.n_dense, s_input_idx);
    __syncthreads();
    for (int i = tid; i < p.nk_small; i += kBlock) {
        dense_mask[i * p.mask_k] = 0;
    }
    __syncthreads();
    for (int i = tid; i < p.n_dense; i += kBlock) {
        const int64_t pos = scratch[i];
        if (pos >= 0 && pos < p.nk_small) {
            dense_mask[pos * p.mask_k] = 1;
        }
    }
    __syncthreads();

    // ----- KIVI dequant + per-chunk top-k -----
    kivi_chunks<kHeadDim>(
        p, q, chunk_idx, dense_mask,
        p.small_min + b * p.smin_b + h * p.smin_h,
        p.small_max + b * p.smax_b + h * p.smax_h,
        p.packed_k + b * p.pk_b + h * p.pk_h,
        kivi_scores, scratch, p.nk_small);

    const int cand = p.nk_small * p.group_size;
    radix_topk_row(kivi_scores, final_out, cand, p.actual_topk, s_input_idx);
    __syncthreads();
    for (int i = tid; i < p.actual_topk; i += kBlock) {
        const int64_t pos = final_out[i];
        int64_t token = 0;
        if (pos >= 0 && pos < cand) {
            token = scratch[pos];
        }
        final_out[i] = token;
    }
    if (p.sk_count > 0 && p.skipped != nullptr) {
        const int64_t* sk = p.skipped + b * p.sk_b + h * p.sk_h;
        for (int i = tid; i < p.sk_count; i += kBlock) {
            final_out[p.actual_topk + i] = sk[i * p.sk_k];
        }
    }
}

}  // namespace

void fused_hierarchical_quest(
    at::Tensor raw_q,
    at::Tensor large_min,
    at::Tensor large_max,
    at::Tensor large_valid,
    at::Tensor small_min,
    at::Tensor small_max,
    at::Tensor large_scores,
    at::Tensor sub_scores,
    at::Tensor scratch,
    at::Tensor chunk_idx,
    at::Tensor centroid,
    at::Tensor density,
    at::Tensor dense_mask,
    at::Tensor packed_k,
    at::Tensor kivi_scores,
    at::Tensor final_indices,
    at::Tensor skipped,
    int64_t n_large,
    int64_t nk_large,
    int64_t factor,
    int64_t n_small_cap,
    int64_t nk_small,
    int64_t n_chunks,
    int64_t n_dense,
    int64_t dense_topk,
    int64_t sparse_topk,
    int64_t n_centroid,
    int64_t actual_topk,
    int64_t sk_count
) {
    TORCH_CHECK(raw_q.is_cuda() && raw_q.scalar_type() == at::kBFloat16);
    TORCH_CHECK(raw_q.dim() == 3 && (raw_q.size(2) == 128 || raw_q.size(2) == 256));
    TORCH_CHECK(large_scores.is_contiguous() && sub_scores.is_contiguous());
    TORCH_CHECK(scratch.is_contiguous() && kivi_scores.is_contiguous());
    TORCH_CHECK(density.is_contiguous() && final_indices.is_contiguous());
    TORCH_CHECK(large_min.stride(-1) == 1 && large_max.stride(-1) == 1);
    TORCH_CHECK(small_min.stride(-1) == 1 && small_max.stride(-1) == 1);
    TORCH_CHECK(packed_k.dim() == 5 && packed_k.stride(-1) == 1);
    TORCH_CHECK(n_large > 0 && nk_large > 0 && factor > 0 && nk_small > 0);
    TORCH_CHECK(n_dense > 0 && n_dense <= nk_small);
    TORCH_CHECK(actual_topk > 0);
    const int group = static_cast<int>(packed_k.size(4));
    TORCH_CHECK(group == 8 || group == 16);
    TORCH_CHECK(dense_topk >= 1 && dense_topk <= group);
    TORCH_CHECK(sparse_topk >= 1 && sparse_topk <= group);
    TORCH_CHECK(scratch.size(2) >= nk_large + nk_small);
    TORCH_CHECK(scratch.size(2) >= nk_small * group);
    TORCH_CHECK(kivi_scores.size(2) >= nk_small * group);
    TORCH_CHECK(final_indices.size(2) >= actual_topk + sk_count);
    TORCH_CHECK(large_scores.size(2) >= n_large);
    TORCH_CHECK(sub_scores.size(2) >= nk_large * factor);

    TORCH_CHECK(packed_k.size(3) * 8 == raw_q.size(2), "packed head dim must match q");

    const int bs = raw_q.size(0);
    const int kv = raw_q.size(1);
    FusedParams p{};
    p.q = reinterpret_cast<const __nv_bfloat16*>(raw_q.data_ptr());
    p.large_min = reinterpret_cast<const __nv_bfloat16*>(large_min.data_ptr());
    p.large_max = reinterpret_cast<const __nv_bfloat16*>(large_max.data_ptr());
    p.large_valid = static_cast<const uint8_t*>(large_valid.data_ptr());
    p.small_min = reinterpret_cast<const __nv_bfloat16*>(small_min.data_ptr());
    p.small_max = reinterpret_cast<const __nv_bfloat16*>(small_max.data_ptr());
    p.large_scores = large_scores.data_ptr<float>();
    p.sub_scores = sub_scores.data_ptr<float>();
    p.scratch = scratch.data_ptr<int64_t>();
    p.chunk_idx = chunk_idx.data_ptr<int64_t>();
    p.centroid = reinterpret_cast<const __nv_bfloat16*>(centroid.data_ptr());
    p.density = density.data_ptr<float>();
    p.dense_mask = static_cast<uint8_t*>(dense_mask.data_ptr());
    p.packed_k = packed_k.data_ptr<int32_t>();
    p.kivi_scores = kivi_scores.data_ptr<float>();
    p.final_indices = final_indices.data_ptr<int64_t>();
    p.skipped = sk_count > 0 ? skipped.data_ptr<int64_t>() : nullptr;
    p.kv_heads = kv;
    p.n_large = static_cast<int>(n_large);
    p.nk_large = static_cast<int>(nk_large);
    p.factor = static_cast<int>(factor);
    p.n_small_cap = static_cast<int>(n_small_cap);
    p.nk_small = static_cast<int>(nk_small);
    p.n_chunks = static_cast<int>(n_chunks);
    p.n_dense = static_cast<int>(n_dense);
    p.dense_topk = static_cast<int>(dense_topk);
    p.sparse_topk = static_cast<int>(sparse_topk);
    p.group_size = group;
    p.n_pack = static_cast<int>(packed_k.size(3));
    p.n_packed = static_cast<int>(packed_k.size(2));
    p.n_centroid = static_cast<int>(n_centroid);
    p.actual_topk = static_cast<int>(actual_topk);
    p.sk_count = static_cast<int>(sk_count);
    p.scratch_stride = static_cast<int>(scratch.size(2));

    p.q_b = raw_q.stride(0); p.q_h = raw_q.stride(1); p.q_d = raw_q.stride(2);
    p.lmin_b = large_min.stride(0); p.lmin_h = large_min.stride(1);
    p.lmin_c = large_min.stride(2); p.lmin_d = large_min.stride(3);
    p.lmax_b = large_max.stride(0); p.lmax_h = large_max.stride(1);
    p.lmax_c = large_max.stride(2); p.lmax_d = large_max.stride(3);
    p.lvalid_b = large_valid.stride(0); p.lvalid_h = large_valid.stride(1);
    p.lvalid_c = large_valid.stride(2);
    p.smin_b = small_min.stride(0); p.smin_h = small_min.stride(1);
    p.smin_c = small_min.stride(2); p.smin_d = small_min.stride(3);
    p.smax_b = small_max.stride(0); p.smax_h = small_max.stride(1);
    p.smax_c = small_max.stride(2); p.smax_d = small_max.stride(3);
    p.lscore_b = large_scores.stride(0); p.lscore_h = large_scores.stride(1);
    p.lscore_c = large_scores.stride(2);
    p.sscore_b = sub_scores.stride(0); p.sscore_h = sub_scores.stride(1);
    p.sscore_c = sub_scores.stride(2);
    p.scratch_b = scratch.stride(0); p.scratch_h = scratch.stride(1);
    p.chunk_b = chunk_idx.stride(0); p.chunk_h = chunk_idx.stride(1);
    p.chunk_k = chunk_idx.stride(2);
    p.cent_b = centroid.stride(0); p.cent_h = centroid.stride(1);
    p.cent_c = centroid.stride(2); p.cent_d = centroid.stride(3);
    p.dens_b = density.stride(0); p.dens_h = density.stride(1); p.dens_k = density.stride(2);
    p.mask_b = dense_mask.stride(0); p.mask_h = dense_mask.stride(1);
    p.mask_k = dense_mask.stride(2);
    p.pk_b = packed_k.stride(0); p.pk_h = packed_k.stride(1);
    p.pk_c = packed_k.stride(2); p.pk_pack = packed_k.stride(3);
    p.ks_b = kivi_scores.stride(0); p.ks_h = kivi_scores.stride(1);
    p.ks_k = kivi_scores.stride(2);
    p.fin_b = final_indices.stride(0); p.fin_h = final_indices.stride(1);
    p.sk_b = sk_count > 0 ? skipped.stride(0) : 0;
    p.sk_h = sk_count > 0 ? skipped.stride(1) : 0;
    p.sk_k = sk_count > 0 ? skipped.stride(2) : 1;

    // Radix reads score rows as contiguous. Stride on the last dim must be 1,
    // which contiguous() above already required for the score buffers.
    TORCH_CHECK(p.lscore_c == 1 && p.sscore_c == 1 && p.dens_k == 1 && p.ks_k == 1);

    c10::cuda::CUDAGuard guard(raw_q.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = 64 * 1024;
    auto launch = [&](auto kernel) {
        auto err_attr = cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem);
        TORCH_CHECK(err_attr == cudaSuccess, "fused quest shared-memory opt-in failed: ",
                    cudaGetErrorString(err_attr));
        kernel<<<bs * kv, kBlock, smem, stream>>>(p);
    };
    if (raw_q.size(2) == 256) {
        launch(fused_hierarchical_quest_kernel<256>);
    } else {
        launch(fused_hierarchical_quest_kernel<128>);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_hierarchical_quest launch failed: ",
                cudaGetErrorString(cudaGetLastError()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_hierarchical_quest", &fused_hierarchical_quest);
}
