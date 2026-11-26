// Quest stage-1 scoring.
// score(chunk) = sum_d max(q_d * max_d, q_d * min_d)
// Matches the PyTorch fallback in polar_cache_kivi_quest_density_seperate_kernel.py.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace {

constexpr int kHeadDim = 128;
constexpr float kInvalidScore = -1e30f;

template <int kHeadDim>
__device__ __forceinline__ float block_sum(float value, float* shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
#pragma unroll
    for (int offset = kHeadDim / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    return shared[0];
}

template <int kHeadDim>
__device__ __forceinline__ float quest_dot(
    const __nv_bfloat16* q,
    const __nv_bfloat16* cmin,
    const __nv_bfloat16* cmax,
    int64_t q_stride,
    int64_t min_stride,
    int64_t max_stride,
    float* shared
) {
    const int tid = threadIdx.x;
    const float qv = __bfloat162float(q[tid * q_stride]);
    const float lo = __bfloat162float(cmin[tid * min_stride]);
    const float hi = __bfloat162float(cmax[tid * max_stride]);
    const float term = fmaxf(qv * hi, qv * lo);
    return block_sum<kHeadDim>(term, shared);
}

template <int kHeadDim>
__global__ void quest_chunk_score_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ cmin,
    const __nv_bfloat16* __restrict__ cmax,
    const uint8_t* __restrict__ valid,
    float* __restrict__ scores,
    int n_chunks,
    int kv_heads,
    int64_t q_b,
    int64_t q_h,
    int64_t q_d,
    int64_t min_b,
    int64_t min_h,
    int64_t min_c,
    int64_t min_d,
    int64_t max_b,
    int64_t max_h,
    int64_t max_c,
    int64_t max_d,
    int64_t valid_b,
    int64_t valid_h,
    int64_t valid_c,
    int64_t score_b,
    int64_t score_h,
    int64_t score_c,
    int has_valid
) {
    const int chunk = blockIdx.x;
    const int bh = blockIdx.y;
    if (chunk >= n_chunks) {
        return;
    }
    const int b = bh / kv_heads;
    const int h = bh - b * kv_heads;

    __shared__ float shared[kHeadDim];
    float score = kInvalidScore;
    const bool keep = !has_valid || valid[b * valid_b + h * valid_h + chunk * valid_c];
    if (keep) {
        score = quest_dot<kHeadDim>(
            q + b * q_b + h * q_h,
            cmin + b * min_b + h * min_h + chunk * min_c,
            cmax + b * max_b + h * max_h + chunk * max_c,
            q_d, min_d, max_d, shared);
    } else if (threadIdx.x == 0) {
        // Keep the unused threads from skipping __syncthreads inside quest_dot.
    }
    if (!keep) {
        // quest_dot was not called, so do not read shared[].
    }
    if (threadIdx.x == 0) {
        scores[b * score_b + h * score_h + chunk * score_c] = score;
    }
}

template <int kHeadDim>
__global__ void quest_sub_chunk_score_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ cmin,
    const __nv_bfloat16* __restrict__ cmax,
    const int64_t* __restrict__ large_idx,
    float* __restrict__ scores,
    int nk_large,
    int factor,
    int n_cap,
    int kv_heads,
    int64_t q_b,
    int64_t q_h,
    int64_t q_d,
    int64_t min_b,
    int64_t min_h,
    int64_t min_c,
    int64_t min_d,
    int64_t max_b,
    int64_t max_h,
    int64_t max_c,
    int64_t max_d,
    int64_t idx_b,
    int64_t idx_h,
    int64_t idx_k,
    int64_t score_b,
    int64_t score_h,
    int64_t score_k
) {
    const int slot = blockIdx.x;
    const int n_slots = nk_large * factor;
    if (slot >= n_slots) {
        return;
    }
    const int bh = blockIdx.y;
    const int b = bh / kv_heads;
    const int h = bh - b * kv_heads;
    const int large_slot = slot / factor;
    const int sub = slot - large_slot * factor;

    __shared__ float shared[kHeadDim];
    const int64_t parent = large_idx[b * idx_b + h * idx_h + large_slot * idx_k];
    const int64_t chunk = parent * factor + sub;
    float score = kInvalidScore;
    if (parent >= 0 && chunk >= 0 && chunk < n_cap) {
        score = quest_dot<kHeadDim>(
            q + b * q_b + h * q_h,
            cmin + b * min_b + h * min_h + chunk * min_c,
            cmax + b * max_b + h * max_h + chunk * max_c,
            q_d, min_d, max_d, shared);
    }
    if (threadIdx.x == 0) {
        scores[b * score_b + h * score_h + slot * score_k] = score;
    }
}

}  // namespace

namespace {

__global__ void quest_map_back_kernel_linear(
    const int64_t* __restrict__ large_idx,
    const int64_t* __restrict__ sub_pos,
    int64_t* __restrict__ chunk_idx,
    int factor,
    int n_chunks,
    int nk_large,
    int nk_small,
    int kv_heads,
    int bs,
    int64_t large_b,
    int64_t large_h,
    int64_t large_k,
    int64_t pos_b,
    int64_t pos_h,
    int64_t pos_k,
    int64_t out_b,
    int64_t out_h,
    int64_t out_k
) {
    const int total = bs * kv_heads * nk_small;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
         index += blockDim.x * gridDim.x) {
        const int k = index % nk_small;
        const int tmp = index / nk_small;
        const int h = tmp % kv_heads;
        const int b = tmp / kv_heads;
        const int64_t pos = sub_pos[b * pos_b + h * pos_h + k * pos_k];
        // Match the PyTorch fallback: clamp the position into the scored
        // sub-chunk span first (float_topk leaves -1 in unfilled slots),
        // then parent * factor + sub, then clamp into the real chunk range.
        int64_t span = static_cast<int64_t>(nk_large) * static_cast<int64_t>(factor);
        if (span < 1) {
            span = 1;
        }
        int64_t safe = pos;
        if (safe < 0) {
            safe = 0;
        } else if (safe > span - 1) {
            safe = span - 1;
        }
        const int64_t large_slot = safe / factor;
        const int64_t sub = safe - large_slot * factor;
        int64_t parent = 0;
        if (large_slot >= 0 && large_slot < nk_large) {
            parent = large_idx[b * large_b + h * large_h + large_slot * large_k];
        }
        int64_t mapped = parent * static_cast<int64_t>(factor) + sub;
        if (n_chunks > 0) {
            if (mapped < 0) {
                mapped = 0;
            } else if (mapped >= n_chunks) {
                mapped = n_chunks - 1;
            }
        } else {
            mapped = 0;
        }
        chunk_idx[b * out_b + h * out_h + k * out_k] = mapped;
    }
}

void check_bf16_4d(const at::Tensor& tensor, const char* name, int64_t head_dim) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::BFloat16, name, " must be bfloat16");
    TORCH_CHECK(tensor.dim() == 4, name, " must be [bs, kv, chunks, D]");
    TORCH_CHECK(tensor.size(3) == head_dim, name, " head_dim must be 128 or 256");
}

}  // namespace

void quest_chunk_score(
    at::Tensor raw_q,
    at::Tensor chunk_min,
    at::Tensor chunk_max,
    at::Tensor scores_out,
    int64_t n_chunks,
    at::Tensor chunk_valid
) {
    check_bf16_4d(chunk_min, "chunk_min", raw_q.size(2));
    check_bf16_4d(chunk_max, "chunk_max", raw_q.size(2));
    TORCH_CHECK(raw_q.is_cuda() && raw_q.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(raw_q.dim() == 3 && (raw_q.size(2) == 128 || raw_q.size(2) == 256));
    TORCH_CHECK(scores_out.is_cuda() && scores_out.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(scores_out.dim() == 3);
    TORCH_CHECK(n_chunks > 0);
    TORCH_CHECK(n_chunks <= chunk_min.size(2));
    TORCH_CHECK(scores_out.size(2) >= n_chunks);

    const int bs = raw_q.size(0);
    const int kv_heads = raw_q.size(1);
    TORCH_CHECK(chunk_min.size(0) >= bs && chunk_min.size(1) == kv_heads);
    TORCH_CHECK(scores_out.size(0) == bs && scores_out.size(1) == kv_heads);

    const bool has_valid = chunk_valid.defined() && chunk_valid.numel() > 0;
    if (has_valid) {
        TORCH_CHECK(chunk_valid.is_cuda() && chunk_valid.scalar_type() == at::ScalarType::Bool);
        TORCH_CHECK(chunk_valid.dim() == 3);
    }

    c10::cuda::CUDAGuard guard(raw_q.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(n_chunks, bs * kv_heads);
    auto launch = [&](auto kernel, int threads) {
        kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(raw_q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(chunk_min.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(chunk_max.data_ptr()),
            has_valid ? static_cast<const uint8_t*>(chunk_valid.data_ptr()) : nullptr,
            scores_out.data_ptr<float>(),
            static_cast<int>(n_chunks),
            kv_heads,
            raw_q.stride(0), raw_q.stride(1), raw_q.stride(2),
            chunk_min.stride(0), chunk_min.stride(1), chunk_min.stride(2), chunk_min.stride(3),
            chunk_max.stride(0), chunk_max.stride(1), chunk_max.stride(2), chunk_max.stride(3),
            has_valid ? chunk_valid.stride(0) : 0,
            has_valid ? chunk_valid.stride(1) : 0,
            has_valid ? chunk_valid.stride(2) : 0,
            scores_out.stride(0), scores_out.stride(1), scores_out.stride(2),
            has_valid ? 1 : 0);
    };
    if (raw_q.size(2) == 256) {
        launch(quest_chunk_score_kernel<256>, 256);
    } else {
        launch(quest_chunk_score_kernel<128>, 128);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "quest_chunk_score launch failed");
}

void quest_sub_chunk_score(
    at::Tensor raw_q,
    at::Tensor chunk_min,
    at::Tensor chunk_max,
    at::Tensor large_idx,
    at::Tensor sub_scores,
    int64_t nk_large,
    int64_t factor
) {
    check_bf16_4d(chunk_min, "chunk_min", raw_q.size(2));
    check_bf16_4d(chunk_max, "chunk_max", raw_q.size(2));
    TORCH_CHECK(raw_q.is_cuda() && raw_q.dim() == 3 && (raw_q.size(2) == 128 || raw_q.size(2) == 256));
    TORCH_CHECK(large_idx.is_cuda() && large_idx.scalar_type() == at::ScalarType::Long && large_idx.dim() == 3);
    TORCH_CHECK(sub_scores.is_cuda() && sub_scores.scalar_type() == at::ScalarType::Float && sub_scores.dim() == 3);
    TORCH_CHECK(nk_large > 0 && factor > 0);
    TORCH_CHECK(large_idx.size(2) >= nk_large);
    TORCH_CHECK(sub_scores.size(2) >= nk_large * factor);

    const int bs = raw_q.size(0);
    const int kv_heads = raw_q.size(1);
    c10::cuda::CUDAGuard guard(raw_q.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(nk_large * factor, bs * kv_heads);
    auto launch = [&](auto kernel, int threads) {
        kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(raw_q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(chunk_min.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(chunk_max.data_ptr()),
            large_idx.data_ptr<int64_t>(),
            sub_scores.data_ptr<float>(),
            static_cast<int>(nk_large),
            static_cast<int>(factor),
            static_cast<int>(chunk_min.size(2)),
            kv_heads,
            raw_q.stride(0), raw_q.stride(1), raw_q.stride(2),
            chunk_min.stride(0), chunk_min.stride(1), chunk_min.stride(2), chunk_min.stride(3),
            chunk_max.stride(0), chunk_max.stride(1), chunk_max.stride(2), chunk_max.stride(3),
            large_idx.stride(0), large_idx.stride(1), large_idx.stride(2),
            sub_scores.stride(0), sub_scores.stride(1), sub_scores.stride(2));
    };
    if (raw_q.size(2) == 256) {
        launch(quest_sub_chunk_score_kernel<256>, 256);
    } else {
        launch(quest_sub_chunk_score_kernel<128>, 128);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "quest_sub_chunk_score launch failed");
}

void quest_map_back(
    at::Tensor large_idx,
    at::Tensor sub_topk_pos,
    at::Tensor chunk_idx,
    int64_t factor,
    int64_t n_chunks
) {
    TORCH_CHECK(large_idx.is_cuda() && sub_topk_pos.is_cuda() && chunk_idx.is_cuda());
    TORCH_CHECK(large_idx.scalar_type() == at::ScalarType::Long);
    TORCH_CHECK(sub_topk_pos.scalar_type() == at::ScalarType::Long);
    TORCH_CHECK(chunk_idx.scalar_type() == at::ScalarType::Long);
    TORCH_CHECK(factor > 0);
    const int bs = large_idx.size(0);
    const int kv_heads = large_idx.size(1);
    const int nk_large = large_idx.size(2);
    const int nk_small = sub_topk_pos.size(2);
    TORCH_CHECK(chunk_idx.size(2) >= nk_small);

    c10::cuda::CUDAGuard guard(large_idx.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const int total = bs * kv_heads * nk_small;
    const int blocks = std::max(1, (total + threads - 1) / threads);
    quest_map_back_kernel_linear<<<blocks, threads, 0, stream>>>(
        large_idx.data_ptr<int64_t>(),
        sub_topk_pos.data_ptr<int64_t>(),
        chunk_idx.data_ptr<int64_t>(),
        static_cast<int>(factor),
        static_cast<int>(n_chunks),
        nk_large,
        nk_small,
        kv_heads,
        bs,
        large_idx.stride(0), large_idx.stride(1), large_idx.stride(2),
        sub_topk_pos.stride(0), sub_topk_pos.stride(1), sub_topk_pos.stride(2),
        chunk_idx.stride(0), chunk_idx.stride(1), chunk_idx.stride(2));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "quest_map_back launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quest_chunk_score", &quest_chunk_score);
    m.def("quest_sub_chunk_score", &quest_sub_chunk_score);
    m.def("quest_map_back", &quest_map_back);
}
