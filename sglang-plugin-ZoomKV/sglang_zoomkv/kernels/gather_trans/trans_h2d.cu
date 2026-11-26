#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mutex>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <unordered_map>
#include <cuda_runtime.h>
#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------------------------------------------------------------------
// CPU pinned host pointer  ->  UVA device pointer cache.
//
// Why we need this:
//   This translation unit launches kernels that dereference *host-pinned*
//   memory directly via UVA (so the GPU PCIe-reads from CPU RAM).  When the
//   process is profiled by nsys 2023.2.3, CUPTI's cudaLaunchKernel intercept
//   validates each kernel-argument pointer; if it sees the *host-side* virtual
//   address of a mapped pinned buffer it can SIGSEGV inside cudaLaunchKernel
//   on first launch.
//
//   The portable fix is to pass a bona fide *device-side* alias of the same
//   pinned buffer obtained via cudaHostGetDevicePointer().  On UVA systems
//   the returned pointer maps to the exact same physical host pages, so the
//   kernel's actual PCIe traffic is unchanged — there is no copy and no
//   measurable runtime cost.  We cache the translation per host pointer so
//   the per-call overhead is a single hashmap lookup (sub-microsecond).
//
//   Lifetime: callers (KIVI / RaBitQ pure-rabitq cache modules) allocate
//   their CPU pinned KV buffers once at __init__ and reuse them for the
//   whole session, so caching by host pointer is safe.
// ---------------------------------------------------------------------------
namespace {
inline void* host_pinned_to_uva_dptr(void* host_ptr) {
    static std::mutex mu;
    static std::unordered_map<void*, void*> cache;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(host_ptr);
        if (it != cache.end()) {
            return it->second;
        }
    }
    void* dev_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostGetDevicePointer failed (is the buffer host-pinned and mapped?): ",
                cudaGetErrorString(err));
    {
        std::lock_guard<std::mutex> lk(mu);
        cache.emplace(host_ptr, dev_ptr);
    }
    return dev_ptr;
}

// Resolve a tensor's data pointer to something nsys/CUPTI is happy to see
// at kernel-launch time.  CUDA tensors pass through unchanged; CPU pinned
// tensors get a UVA device-pointer alias.
inline void* resolve_kernel_arg_ptr(const at::Tensor& t) {
    void* p = t.data_ptr();
    if (t.is_cuda()) {
        return p;
    }
    // Tensor must be a CPU tensor here; require it to be host-pinned so that
    // cudaHostGetDevicePointer is well-defined.
    TORCH_CHECK(t.is_pinned(),
                "h2d_gather_kv: CPU source tensor must be host-pinned (allocated with pin_memory=True)");
    return host_pinned_to_uva_dptr(p);
}
}  // namespace


// One warp copies several scattered tokens and keeps their PCIe loads in
// flight before writing them back. A one-token block issues a single 16-byte
// load per thread and spends the launch on latency.
template<int DIM, int TOKENS>
__global__
void h2d_gather_kv_kernel(
    void* __restrict__ src_k_caches_ptr,
    void* __restrict__ src_v_caches_ptr,
    void* __restrict__ dst_k_caches_ptr,
    void* __restrict__ dst_v_caches_ptr,
    const int64_t* __restrict__ selected_indices_ptr,
    int src_len,
    int dst_len,
    int topk
) {
    constexpr int uint4_per_dim = DIM * 2 / 16; // 256 / 16 = 16
    const int heads = static_cast<int>(gridDim.y);
    const int token0 = static_cast<int>(blockIdx.z) * TOKENS;
    const int tidx = static_cast<int>(threadIdx.x);
    const long long head_base = (static_cast<long long>(blockIdx.x) * heads + blockIdx.y) * topk;

    int64_t index[TOKENS];
#pragma unroll
    for (int i = 0; i < TOKENS; ++i) {
        const int token = token0 + i;
        index[i] = -1;
        if (token < topk) {
            index[i] = __ldg(selected_indices_ptr + head_base + token);
        }
    }

    const uint4 zeros = make_uint4(0, 0, 0, 0);
    constexpr int nvec = uint4_per_dim / 16;
    if (tidx < 16) {
        uint4 regs[TOKENS][nvec];
#pragma unroll
        for (int i = 0; i < TOKENS; ++i) {
            const int token = token0 + i;
            const bool ok = token < topk && index[i] >= 0 && index[i] < static_cast<int64_t>(src_len);
#pragma unroll
            for (int v = 0; v < nvec; ++v) {
                if (!ok) {
                    regs[i][v] = zeros;
                } else {
                    const uint4* src = reinterpret_cast<const uint4*>(src_k_caches_ptr);
                    src += ((static_cast<long long>(blockIdx.x) * heads + blockIdx.y) * src_len + index[i]) * uint4_per_dim;
                    regs[i][v] = src[tidx + v * 16];
                }
            }
        }
#pragma unroll
        for (int i = 0; i < TOKENS; ++i) {
            const int token = token0 + i;
            if (token >= topk) {
                continue;
            }
            uint4* dst = reinterpret_cast<uint4*>(dst_k_caches_ptr);
            dst += ((static_cast<long long>(blockIdx.x) * heads + blockIdx.y) * dst_len + token) * uint4_per_dim;
#pragma unroll
            for (int v = 0; v < nvec; ++v) {
                dst[tidx + v * 16] = regs[i][v];
            }
        }
    } else if (tidx < 32) {
        const int v_tidx = tidx - 16;
        uint4 regs[TOKENS][nvec];
#pragma unroll
        for (int i = 0; i < TOKENS; ++i) {
            const int token = token0 + i;
            const bool ok = token < topk && index[i] >= 0 && index[i] < static_cast<int64_t>(src_len);
#pragma unroll
            for (int v = 0; v < nvec; ++v) {
                if (!ok) {
                    regs[i][v] = zeros;
                } else {
                    const uint4* src = reinterpret_cast<const uint4*>(src_v_caches_ptr);
                    src += ((static_cast<long long>(blockIdx.x) * heads + blockIdx.y) * src_len + index[i]) * uint4_per_dim;
                    regs[i][v] = src[v_tidx + v * 16];
                }
            }
        }
#pragma unroll
        for (int i = 0; i < TOKENS; ++i) {
            const int token = token0 + i;
            if (token >= topk) {
                continue;
            }
            uint4* dst = reinterpret_cast<uint4*>(dst_v_caches_ptr);
            dst += ((static_cast<long long>(blockIdx.x) * heads + blockIdx.y) * dst_len + token) * uint4_per_dim;
#pragma unroll
            for (int v = 0; v < nvec; ++v) {
                dst[v_tidx + v * 16] = regs[i][v];
            }
        }
    }
}

void h2d_gather_kv(
    at::Tensor& src_k_caches, // bs, heads, kv_len, dim
    at::Tensor& src_v_caches, // bs, heads, kv_len, dim
    at::Tensor& selected_indices, // bs, heads, topk
    at::Tensor& dst_k_caches, // bs, heads, max_topk, dim
    at::Tensor& dst_v_caches // bs, heads, max_topk, dim
){
    unsigned int bs = src_k_caches.size(0);
    unsigned int heads = src_k_caches.size(1);
    unsigned int src_len = src_k_caches.size(2);
    unsigned int dim = src_k_caches.size(3);
    unsigned int topk = selected_indices.size(2);
    unsigned int dst_len = dst_k_caches.size(2);

    // Resolve the UVA alias on the device that will launch the kernel.
    // cudaHostGetDevicePointer follows the current context.
    const c10::Device dev = dst_k_caches.device();
    c10::cuda::CUDAGuard device_guard(dev);

    void * src_k_caches_ptr = resolve_kernel_arg_ptr(src_k_caches);
    void * src_v_caches_ptr = resolve_kernel_arg_ptr(src_v_caches);
    void * dst_k_caches_ptr = dst_k_caches.data_ptr();
    void * dst_v_caches_ptr = dst_v_caches.data_ptr();
    void * selected_indices_ptr = selected_indices.data_ptr();

    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int kTokensPerBlock = 8;
    if (dim != 128 && dim != 256) {
        TORCH_CHECK(false, "The dim must be 128 or 256.");
    }
    if (topk == 0) {
        return;
    }
    TORCH_CHECK(selected_indices.is_contiguous(), "selected_indices must be contiguous");
    TORCH_CHECK(selected_indices.is_cuda(), "selected_indices must be CUDA");
    const unsigned int blocks_z = (topk + kTokensPerBlock - 1) / kTokensPerBlock;
    dim3 grid = {bs, heads, blocks_z};
    if (dim == 256) {
        h2d_gather_kv_kernel<256, kTokensPerBlock><<<grid, 32, 0, stream>>>(
            src_k_caches_ptr,
            src_v_caches_ptr,
            dst_k_caches_ptr,
            dst_v_caches_ptr,
            reinterpret_cast<const int64_t*>(selected_indices_ptr),
            static_cast<int>(src_len),
            static_cast<int>(dst_len),
            static_cast<int>(topk)
        );
    } else {
        h2d_gather_kv_kernel<128, kTokensPerBlock><<<grid, 32, 0, stream>>>(
            src_k_caches_ptr,
            src_v_caches_ptr,
            dst_k_caches_ptr,
            dst_v_caches_ptr,
            reinterpret_cast<const int64_t*>(selected_indices_ptr),
            static_cast<int>(src_len),
            static_cast<int>(dst_len),
            static_cast<int>(topk)
        );
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "update_cache_cnt kernel failed:", ::cudaGetErrorString(result));

}

// Retrieval KV is stored as int8 with one fp32 scale per 128 tokens per head.
// The kernel dequantizes into the bf16 top-k buffers. Wider than a byte load
// is unnecessary: 128 int8 values are 128 bytes, 8 per thread.
template<int DIM, int TOKENS, int BLOCK>
__global__
void h2d_gather_kv_i8_kernel(
    const int8_t* __restrict__ src_k,
    const int8_t* __restrict__ src_v,
    const float* __restrict__ scale_k,
    const float* __restrict__ scale_v,
    __nv_bfloat16* __restrict__ dst_k,
    __nv_bfloat16* __restrict__ dst_v,
    const int64_t* __restrict__ indices,
    int src_len,
    int dst_len,
    int topk,
    int n_blocks
) {
    constexpr int elems = DIM / 16;
    const int heads = static_cast<int>(gridDim.y);
    const int token0 = static_cast<int>(blockIdx.z) * TOKENS;
    const int tidx = static_cast<int>(threadIdx.x);
    const long long head = static_cast<long long>(blockIdx.x) * heads + blockIdx.y;

    int64_t index[TOKENS];
#pragma unroll
    for (int i = 0; i < TOKENS; ++i) {
        const int token = token0 + i;
        index[i] = -1;
        if (token < topk) {
            index[i] = __ldg(indices + head * topk + token);
        }
    }

    const int lane = tidx < 16 ? tidx : tidx - 16;
    const bool is_k = tidx < 16;
    if (tidx >= 32) {
        return;
    }
    const int8_t* src = is_k ? src_k : src_v;
    const float* scale = is_k ? scale_k : scale_v;
    __nv_bfloat16* dst = is_k ? dst_k : dst_v;

#pragma unroll
    for (int i = 0; i < TOKENS; ++i) {
        const int token = token0 + i;
        uint4 packed = make_uint4(0, 0, 0, 0);
        const bool ok = token < topk && index[i] >= 0 && index[i] < static_cast<int64_t>(src_len);
        if (ok) {
            const int block = static_cast<int>(index[i] / BLOCK);
            float s = 0.f;
            if (block >= 0 && block < n_blocks) {
                s = __ldg(scale + head * n_blocks + block);
            }
            const int8_t* sp = src + (head * src_len + index[i]) * DIM + lane * elems;
            __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
            for (int j = 0; j < elems; ++j) {
                out[j] = __float2bfloat16(static_cast<float>(sp[j]) * s);
            }
        }
        if (token < topk) {
            uint4* dp = reinterpret_cast<uint4*>(dst + (head * dst_len + token) * DIM + lane * elems);
            *dp = packed;
        }
    }
}

void h2d_gather_kv_i8(
    at::Tensor& src_k_caches,
    at::Tensor& src_v_caches,
    at::Tensor& scale_k,
    at::Tensor& scale_v,
    at::Tensor& selected_indices,
    at::Tensor& dst_k_caches,
    at::Tensor& dst_v_caches
) {
    unsigned int bs = src_k_caches.size(0);
    unsigned int heads = src_k_caches.size(1);
    unsigned int src_len = src_k_caches.size(2);
    unsigned int dim = src_k_caches.size(3);
    unsigned int topk = selected_indices.size(2);
    unsigned int dst_len = dst_k_caches.size(2);
    unsigned int n_blocks = scale_k.size(2);

    TORCH_CHECK(src_k_caches.scalar_type() == at::kChar, "src_k must be int8");
    TORCH_CHECK(src_v_caches.scalar_type() == at::kChar, "src_v must be int8");
    TORCH_CHECK(dst_k_caches.scalar_type() == at::kBFloat16, "dst_k must be bf16");
    TORCH_CHECK(dst_v_caches.scalar_type() == at::kBFloat16, "dst_v must be bf16");
    TORCH_CHECK(scale_k.scalar_type() == at::kFloat, "scale_k must be float32");
    TORCH_CHECK(scale_v.scalar_type() == at::kFloat, "scale_v must be float32");
    TORCH_CHECK(src_k_caches.is_pinned() && src_v_caches.is_pinned(), "int8 KV must be pinned");
    TORCH_CHECK(scale_k.is_cuda() && scale_v.is_cuda(), "scales must be CUDA");
    TORCH_CHECK(scale_k.is_contiguous() && scale_v.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(src_k_caches.is_contiguous() && src_v_caches.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst_k_caches.is_contiguous() && dst_v_caches.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(selected_indices.is_contiguous() && selected_indices.is_cuda(), "indices must be contiguous CUDA");
    TORCH_CHECK(dim == 128, "The dim must be 128.");
    if (topk == 0) {
        return;
    }

    const c10::Device dev = dst_k_caches.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kTokensPerBlock = 8;
    const unsigned int blocks_z = (topk + kTokensPerBlock - 1) / kTokensPerBlock;
    dim3 grid = {bs, heads, blocks_z};
    h2d_gather_kv_i8_kernel<128, kTokensPerBlock, 128><<<grid, 32, 0, stream>>>(
        reinterpret_cast<const int8_t*>(resolve_kernel_arg_ptr(src_k_caches)),
        reinterpret_cast<const int8_t*>(resolve_kernel_arg_ptr(src_v_caches)),
        scale_k.data_ptr<float>(),
        scale_v.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dst_k_caches.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dst_v_caches.data_ptr()),
        selected_indices.data_ptr<int64_t>(),
        static_cast<int>(src_len),
        static_cast<int>(dst_len),
        static_cast<int>(topk),
        static_cast<int>(n_blocks)
    );
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "h2d_gather_kv_i8 failed:", ::cudaGetErrorString(result));
}

// Packed int4 retrieval KV. Two signed nibbles per byte, even dim in the low
// nibble. One fp32 scale per 128 tokens per head. Output is bf16.
template<int TOKENS, int BLOCK, int OUT_DIM>
__global__
void h2d_gather_kv_i4_kernel(
    const uint8_t* __restrict__ src_k,
    const uint8_t* __restrict__ src_v,
    const float* __restrict__ scale_k,
    const float* __restrict__ scale_v,
    __nv_bfloat16* __restrict__ dst_k,
    __nv_bfloat16* __restrict__ dst_v,
    const int64_t* __restrict__ indices,
    int src_len,
    int dst_len,
    int topk,
    int n_blocks
) {
    constexpr int out_dim = OUT_DIM;
    constexpr int packed_dim = OUT_DIM / 2;
    constexpr int words = OUT_DIM / 128;
    const int heads = static_cast<int>(gridDim.y);
    const int token0 = static_cast<int>(blockIdx.z) * TOKENS;
    const int tidx = static_cast<int>(threadIdx.x);
    const long long head = static_cast<long long>(blockIdx.x) * heads + blockIdx.y;

    int64_t index[TOKENS];
#pragma unroll
    for (int i = 0; i < TOKENS; ++i) {
        const int token = token0 + i;
        index[i] = -1;
        if (token < topk) {
            index[i] = __ldg(indices + head * topk + token);
        }
    }

    if (tidx >= 32) {
        return;
    }
    const int lane = tidx < 16 ? tidx : tidx - 16;
    const bool is_k = tidx < 16;
    const uint8_t* src = is_k ? src_k : src_v;
    const float* scale = is_k ? scale_k : scale_v;
    __nv_bfloat16* dst = is_k ? dst_k : dst_v;

#pragma unroll
    for (int i = 0; i < TOKENS; ++i) {
        const int token = token0 + i;
        uint4 packed_out[words];
        const bool ok = token < topk && index[i] >= 0 && index[i] < static_cast<int64_t>(src_len);
#pragma unroll
        for (int w = 0; w < words; ++w) {
            packed_out[w] = make_uint4(0, 0, 0, 0);
        }
        if (ok) {
            const int block = static_cast<int>(index[i] / BLOCK);
            float s = 0.f;
            if (block >= 0 && block < n_blocks) {
                s = __ldg(scale + head * n_blocks + block);
            }
#pragma unroll
            for (int w = 0; w < words; ++w) {
                const int piece = lane * words + w;
                const uint8_t* sp = src + (head * src_len + index[i]) * packed_dim + piece * 4;
                const unsigned word = *reinterpret_cast<const unsigned*>(sp);
                __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(&packed_out[w]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int nib = static_cast<int>((word >> (j * 4)) & 0xFu);
                    if (nib >= 8) {
                        nib -= 16;
                    }
                    out[j] = __float2bfloat16(static_cast<float>(nib) * s);
                }
            }
        }
        if (token < topk) {
#pragma unroll
            for (int w = 0; w < words; ++w) {
                const int piece = lane * words + w;
                uint4* dp = reinterpret_cast<uint4*>(
                    dst + (head * dst_len + token) * out_dim + piece * 8);
                *dp = packed_out[w];
            }
        }
    }
}

void h2d_gather_kv_i4(
    at::Tensor& src_k_caches,
    at::Tensor& src_v_caches,
    at::Tensor& scale_k,
    at::Tensor& scale_v,
    at::Tensor& selected_indices,
    at::Tensor& dst_k_caches,
    at::Tensor& dst_v_caches
) {
    unsigned int bs = src_k_caches.size(0);
    unsigned int heads = src_k_caches.size(1);
    unsigned int src_len = src_k_caches.size(2);
    unsigned int packed_dim = src_k_caches.size(3);
    unsigned int topk = selected_indices.size(2);
    unsigned int dst_len = dst_k_caches.size(2);
    unsigned int n_blocks = scale_k.size(2);

    TORCH_CHECK(src_k_caches.scalar_type() == at::kByte, "src_k must be uint8 packed int4");
    TORCH_CHECK(src_v_caches.scalar_type() == at::kByte, "src_v must be uint8 packed int4");
    TORCH_CHECK(packed_dim == 64 || packed_dim == 128, "packed int4 dim must be 64 or 128");
    TORCH_CHECK(dst_k_caches.size(3) == packed_dim * 2, "dst head dim must be twice the packed dim");
    TORCH_CHECK(dst_k_caches.scalar_type() == at::kBFloat16, "dst_k must be bf16");
    TORCH_CHECK(dst_v_caches.scalar_type() == at::kBFloat16, "dst_v must be bf16");
    TORCH_CHECK(scale_k.scalar_type() == at::kFloat && scale_v.scalar_type() == at::kFloat, "scales must be float32");
    TORCH_CHECK(src_k_caches.is_pinned() && src_v_caches.is_pinned(), "int4 KV must be pinned");
    TORCH_CHECK(scale_k.is_cuda() && scale_v.is_cuda(), "scales must be CUDA");
    TORCH_CHECK(scale_k.is_contiguous() && scale_v.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(src_k_caches.is_contiguous() && src_v_caches.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst_k_caches.is_contiguous() && dst_v_caches.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(selected_indices.is_contiguous() && selected_indices.is_cuda(), "indices must be contiguous CUDA");
    if (topk == 0) {
        return;
    }

    const c10::Device dev = dst_k_caches.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kTokensPerBlock = 8;
    const unsigned int blocks_z = (topk + kTokensPerBlock - 1) / kTokensPerBlock;
    dim3 grid = {bs, heads, blocks_z};
    if (packed_dim == 128) {
        h2d_gather_kv_i4_kernel<kTokensPerBlock, 128, 256><<<grid, 32, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(resolve_kernel_arg_ptr(src_k_caches)),
            reinterpret_cast<const uint8_t*>(resolve_kernel_arg_ptr(src_v_caches)),
            scale_k.data_ptr<float>(),
            scale_v.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(dst_k_caches.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dst_v_caches.data_ptr()),
            selected_indices.data_ptr<int64_t>(),
            static_cast<int>(src_len),
            static_cast<int>(dst_len),
            static_cast<int>(topk),
            static_cast<int>(n_blocks)
        );
    } else {
        h2d_gather_kv_i4_kernel<kTokensPerBlock, 128, 128><<<grid, 32, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(resolve_kernel_arg_ptr(src_k_caches)),
            reinterpret_cast<const uint8_t*>(resolve_kernel_arg_ptr(src_v_caches)),
            scale_k.data_ptr<float>(),
            scale_v.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(dst_k_caches.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dst_v_caches.data_ptr()),
            selected_indices.data_ptr<int64_t>(),
            static_cast<int>(src_len),
            static_cast<int>(dst_len),
            static_cast<int>(topk),
            static_cast<int>(n_blocks)
        );
    }
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "h2d_gather_kv_i4 failed:", ::cudaGetErrorString(result));
}


// Replay the previous token's gathered rows from GPU memory. PCIe is used
// only for indices that were not in that set. The cache is replaced after
// the gather so the next token sees this selection. CPU rows already stored
// in the cache must not be overwritten; decode only appends retrieval KV.
template<int DIM, int TOKENS>
__global__
void h2d_gather_kv_cached_kernel(
    const uint4* __restrict__ src_k,
    const uint4* __restrict__ src_v,
    uint4* __restrict__ dst_k,
    uint4* __restrict__ dst_v,
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ cache_idx,
    const uint4* __restrict__ cache_k,
    const uint4* __restrict__ cache_v,
    int src_len,
    int dst_len,
    int topk,
    int cache_len
) {
    constexpr int uint4_per_dim = DIM * 2 / 16;
    const int heads = static_cast<int>(gridDim.y);
    const int token0 = static_cast<int>(blockIdx.z) * TOKENS;
    const int tidx = static_cast<int>(threadIdx.x);
    const long long head = static_cast<long long>(blockIdx.x) * heads + blockIdx.y;

    __shared__ int64_t index_s[TOKENS];
    __shared__ int slot_s[TOKENS];
    if (tidx < TOKENS) {
        const int token = token0 + tidx;
        int64_t index = -1;
        int slot = -1;
        if (token < topk) {
            index = indices[head * topk + token];
            if (index >= 0 && index < static_cast<int64_t>(src_len)) {
                const int nsearch = topk < cache_len ? topk : cache_len;
                const int64_t* row = cache_idx + head * cache_len;
                for (int j = 0; j < nsearch; ++j) {
                    if (row[j] == index) {
                        slot = j;
                        break;
                    }
                }
            }
        }
        index_s[tidx] = index;
        slot_s[tidx] = slot;
    }
    __syncthreads();

    const uint4 zeros = make_uint4(0, 0, 0, 0);
    if (tidx < 16) {
        uint4 regs[TOKENS];
#pragma unroll
        for (int i = 0; i < TOKENS; ++i) {
            const int token = token0 + i;
            const int64_t index = index_s[i];
            const int slot = slot_s[i];
            const bool ok = token < topk && index >= 0 && index < static_cast<int64_t>(src_len);
            if (!ok) {
                regs[i] = zeros;
            } else if (slot >= 0) {
                const uint4* src = cache_k + (head * cache_len + slot) * uint4_per_dim;
                regs[i] = src[tidx];
            } else {
                const uint4* src = src_k + (head * src_len + index) * uint4_per_dim;
                regs[i] = src[tidx];
            }
        }
#pragma unroll
        for (int i = 0; i < TOKENS; ++i) {
            const int token = token0 + i;
            if (token >= topk) {
                continue;
            }
            uint4* dst = dst_k + (head * dst_len + token) * uint4_per_dim;
            dst[tidx] = regs[i];
        }
    } else if (tidx < 32) {
        const int v_tidx = tidx - 16;
        uint4 regs[TOKENS];
#pragma unroll
        for (int i = 0; i < TOKENS; ++i) {
            const int token = token0 + i;
            const int64_t index = index_s[i];
            const int slot = slot_s[i];
            const bool ok = token < topk && index >= 0 && index < static_cast<int64_t>(src_len);
            if (!ok) {
                regs[i] = zeros;
            } else if (slot >= 0) {
                const uint4* src = cache_v + (head * cache_len + slot) * uint4_per_dim;
                regs[i] = src[v_tidx];
            } else {
                const uint4* src = src_v + (head * src_len + index) * uint4_per_dim;
                regs[i] = src[v_tidx];
            }
        }
#pragma unroll
        for (int i = 0; i < TOKENS; ++i) {
            const int token = token0 + i;
            if (token >= topk) {
                continue;
            }
            uint4* dst = dst_v + (head * dst_len + token) * uint4_per_dim;
            dst[v_tidx] = regs[i];
        }
    }
}

__global__
void h2d_gather_cache_store_kernel(
    const uint4* __restrict__ dst_k,
    const uint4* __restrict__ dst_v,
    const int64_t* __restrict__ indices,
    uint4* __restrict__ cache_k,
    uint4* __restrict__ cache_v,
    int64_t* __restrict__ cache_idx,
    int dst_len,
    int cache_len,
    int topk
) {
    constexpr int uint4_per_dim = 16; // DIM 128 bf16
    const int token = static_cast<int>(blockIdx.z);
    if (token >= topk) {
        return;
    }
    const int tidx = static_cast<int>(threadIdx.x);
    const int heads = static_cast<int>(gridDim.y);
    const long long head = static_cast<long long>(blockIdx.x) * heads + blockIdx.y;
    if (tidx < 16) {
        const uint4* src = dst_k + (head * dst_len + token) * uint4_per_dim;
        uint4* dst = cache_k + (head * cache_len + token) * uint4_per_dim;
        dst[tidx] = src[tidx];
    } else if (tidx < 32) {
        const int v_tidx = tidx - 16;
        const uint4* src = dst_v + (head * dst_len + token) * uint4_per_dim;
        uint4* dst = cache_v + (head * cache_len + token) * uint4_per_dim;
        dst[v_tidx] = src[v_tidx];
    }
    if (tidx == 0) {
        cache_idx[head * cache_len + token] = indices[head * topk + token];
    }
}

void h2d_gather_kv_cached(
    at::Tensor& src_k_caches,
    at::Tensor& src_v_caches,
    at::Tensor& selected_indices,
    at::Tensor& dst_k_caches,
    at::Tensor& dst_v_caches,
    at::Tensor& cache_k,
    at::Tensor& cache_v,
    at::Tensor& cache_idx
) {
    const unsigned int bs = src_k_caches.size(0);
    const unsigned int heads = src_k_caches.size(1);
    const unsigned int src_len = src_k_caches.size(2);
    const unsigned int dim = src_k_caches.size(3);
    const unsigned int topk = selected_indices.size(2);
    const unsigned int dst_len = dst_k_caches.size(2);
    const unsigned int cache_len = cache_k.size(2);

    TORCH_CHECK(dim == 128, "The dim must be 128.");
    if (topk == 0) {
        return;
    }
    TORCH_CHECK(selected_indices.is_contiguous(), "selected_indices must be contiguous");
    TORCH_CHECK(selected_indices.is_cuda(), "selected_indices must be CUDA");
    TORCH_CHECK(selected_indices.scalar_type() == at::kLong, "selected_indices must be int64");
    TORCH_CHECK(selected_indices.size(0) == bs && selected_indices.size(1) == heads,
                "selected_indices batch/head must match the KV source");
    TORCH_CHECK(dst_k_caches.is_cuda() && dst_v_caches.is_cuda(), "destination KV must be CUDA");
    TORCH_CHECK(dst_k_caches.is_contiguous() && dst_v_caches.is_contiguous(),
                "destination KV must be contiguous");
    TORCH_CHECK(dst_len >= topk, "destination KV is shorter than topk");
    TORCH_CHECK(cache_k.is_cuda() && cache_v.is_cuda() && cache_idx.is_cuda(),
                "gather cache must be CUDA");
    TORCH_CHECK(cache_k.is_contiguous() && cache_v.is_contiguous() && cache_idx.is_contiguous(),
                "gather cache must be contiguous");
    TORCH_CHECK(cache_k.sizes() == cache_v.sizes(), "cache K/V shapes must match");
    TORCH_CHECK(cache_k.size(0) == bs && cache_k.size(1) == heads && cache_k.size(3) == dim,
                "gather cache batch/head/dim must match the KV source");
    TORCH_CHECK(cache_len >= topk, "gather cache is shorter than topk");
    TORCH_CHECK(cache_idx.scalar_type() == at::kLong, "cache indices must be int64");
    TORCH_CHECK(cache_idx.size(0) == bs && cache_idx.size(1) == heads && cache_idx.size(2) == cache_len,
                "cache index shape must be [batch, heads, cache_len]");
    TORCH_CHECK(cache_k.scalar_type() == dst_k_caches.scalar_type(), "cache dtype must match destination");

    const c10::Device dev = dst_k_caches.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int kTokensPerBlock = 8;
    const unsigned int blocks_z = (topk + kTokensPerBlock - 1) / kTokensPerBlock;
    dim3 gather_grid = {bs, heads, blocks_z};
    h2d_gather_kv_cached_kernel<128, kTokensPerBlock><<<gather_grid, 32, 0, stream>>>(
        reinterpret_cast<const uint4*>(resolve_kernel_arg_ptr(src_k_caches)),
        reinterpret_cast<const uint4*>(resolve_kernel_arg_ptr(src_v_caches)),
        reinterpret_cast<uint4*>(dst_k_caches.data_ptr()),
        reinterpret_cast<uint4*>(dst_v_caches.data_ptr()),
        selected_indices.data_ptr<int64_t>(),
        cache_idx.data_ptr<int64_t>(),
        reinterpret_cast<const uint4*>(cache_k.data_ptr()),
        reinterpret_cast<const uint4*>(cache_v.data_ptr()),
        static_cast<int>(src_len),
        static_cast<int>(dst_len),
        static_cast<int>(topk),
        static_cast<int>(cache_len)
    );
    const auto gather_err = cudaGetLastError();
    TORCH_CHECK(gather_err == cudaSuccess,
                "h2d_gather_kv_cached kernel failed: ",
                cudaGetErrorString(gather_err));

    dim3 store_grid = {bs, heads, topk};
    h2d_gather_cache_store_kernel<<<store_grid, 32, 0, stream>>>(
        reinterpret_cast<const uint4*>(dst_k_caches.data_ptr()),
        reinterpret_cast<const uint4*>(dst_v_caches.data_ptr()),
        selected_indices.data_ptr<int64_t>(),
        reinterpret_cast<uint4*>(cache_k.data_ptr()),
        reinterpret_cast<uint4*>(cache_v.data_ptr()),
        cache_idx.data_ptr<int64_t>(),
        static_cast<int>(dst_len),
        static_cast<int>(cache_len),
        static_cast<int>(topk)
    );
    const auto store_err = cudaGetLastError();
    TORCH_CHECK(store_err == cudaSuccess,
                "h2d_gather_cache_store kernel failed: ",
                cudaGetErrorString(store_err));
}


// ---------------------------------------------------------------------------
// Packed variant: directly writes the gathered topk rows into the *final*
// flash_attn_varlen packed buffer, using per-(batch, head) variable K.
//
// Layout / semantics:
//   src_k/src_v       : CPU pinned [bs, kv_heads, src_len, DIM]   (bf16)
//   dst_k_packed      : GPU        [total_k, DIM]                  (bf16)
//   dst_v_packed      : GPU        [total_k, DIM]                  (bf16)
//
//   indices_packed    : GPU        [total_topk]   int64
//                       Concatenated absolute KV indices, one segment per
//                       (b, h) pair, in row-major (b, h) order. The length
//                       of the (b, h) segment equals adaptive_k[b, h].
//
//   cu_seqlens_topk   : GPU        [V + 1]        int32   (V = bs * kv_heads)
//                       Prefix sums of adaptive_k. cu_seqlens_topk[vh] is the
//                       starting offset of (b, h)'s segment inside
//                       indices_packed.
//
//   cu_seqlens_k      : GPU        [V + 1]        int32
//                       Prefix sums of (sl_end + adaptive_k). Determines the
//                       starting row of (b, h)'s segment inside the *final*
//                       packed attention buffer (the same buffer used by
//                       flash_attn_varlen_func). The first sl_end rows of
//                       each segment are reserved for sink+local KV that the
//                       caller fills separately on GPU; this kernel only
//                       writes rows [cu_seqlens_k[vh] + sl_end ..
//                                    cu_seqlens_k[vh + 1]).
//
//   sl_end            : int   (= sink_size + local_size)
//   global_k          : int   (= max(adaptive_k) over all heads; grid Y dim)
//
// Grid layout: (V, global_k, 1). Each block handles a single output row of
// the packed buffer for one (b, h) pair. Threads 0..15 copy the K row, 16..31
// copy the V row (uint4-vectorized, DIM=128 only).
// ---------------------------------------------------------------------------
template<int DIM>
__global__
void h2d_gather_kv_kernel_packed(
    void* __restrict__ src_k_ptr,
    void* __restrict__ src_v_ptr,
    void* __restrict__ dst_k_packed_ptr,
    void* __restrict__ dst_v_packed_ptr,
    void* __restrict__ indices_packed_ptr,
    void* __restrict__ cu_seqlens_topk_ptr,
    void* __restrict__ cu_seqlens_k_ptr,
    int src_len,
    int kv_heads,
    int sl_end
) {
    const int vh = blockIdx.x;
    const int slot = blockIdx.y;
    const int tidx = threadIdx.x;

    const int* cu_topk = reinterpret_cast<int*>(cu_seqlens_topk_ptr);
    const int seg_start = cu_topk[vh];
    const int seg_len = cu_topk[vh + 1] - seg_start;
    if (slot >= seg_len) return;

    const int b = vh / kv_heads;
    const int h = vh - b * kv_heads;

    const int packed_topk_pos = seg_start + slot;
    const int64_t index = __ldg(
        reinterpret_cast<int64_t*>(indices_packed_ptr) + packed_topk_pos);

    const int* cu_k = reinterpret_cast<int*>(cu_seqlens_k_ptr);
    const int dst_row = cu_k[vh] + sl_end + slot;

    constexpr int uint4_per_dim = DIM * 2 / 16; // bf16 -> 2B/elem; 16B/uint4

    if (DIM == 128) {
        const int64_t src_off =
            (int64_t)b * kv_heads * src_len * uint4_per_dim
          + (int64_t)h * src_len * uint4_per_dim
          + (int64_t)index * uint4_per_dim;
        const int64_t dst_off = (int64_t)dst_row * uint4_per_dim;

        if (tidx < 16) {
            uint4* src = reinterpret_cast<uint4*>(src_k_ptr) + src_off;
            uint4* dst = reinterpret_cast<uint4*>(dst_k_packed_ptr) + dst_off;
            dst[tidx] = src[tidx];
        } else if (tidx < 32) {
            uint4* src = reinterpret_cast<uint4*>(src_v_ptr) + src_off;
            uint4* dst = reinterpret_cast<uint4*>(dst_v_packed_ptr) + dst_off;
            const int v_tidx = tidx - 16;
            dst[v_tidx] = src[v_tidx];
        }
    }
}




void h2d_gather_kv_packed(
    at::Tensor& src_k_caches,    // CPU pinned [bs, kv_heads, src_len, dim]
    at::Tensor& src_v_caches,    // CPU pinned [bs, kv_heads, src_len, dim]
    at::Tensor& dst_k_packed,    // GPU         [total_k, dim]
    at::Tensor& dst_v_packed,    // GPU         [total_k, dim]
    at::Tensor& indices_packed,  // GPU         [total_topk] int64
    at::Tensor& cu_seqlens_topk, // GPU         [V + 1]      int32
    at::Tensor& cu_seqlens_k,    // GPU         [V + 1]      int32
    int64_t sl_end,
    int64_t global_k             // grid Y dim (= max adaptive_k)
) {
    c10::cuda::CUDAGuard device_guard(dst_k_packed.device());
    void* src_k_ptr = resolve_kernel_arg_ptr(src_k_caches);
    void* src_v_ptr = resolve_kernel_arg_ptr(src_v_caches);
    void* dst_k_ptr = dst_k_packed.data_ptr();
    void* dst_v_ptr = dst_v_packed.data_ptr();
    void* indices_ptr = indices_packed.data_ptr();
    void* cu_topk_ptr = cu_seqlens_topk.data_ptr();
    void* cu_k_ptr = cu_seqlens_k.data_ptr();

    const int bs = src_k_caches.size(0);
    const int kv_heads = src_k_caches.size(1);
    const int src_len = src_k_caches.size(2);
    const int dim = src_k_caches.size(3);
    const int V = bs * kv_heads;

    if (global_k <= 0 || V <= 0) return;

    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    dim3 grid = {(unsigned)V, (unsigned)global_k, 1u};
    if (dim == 128) {
        h2d_gather_kv_kernel_packed<128><<<grid, 32, 0, stream>>>(
            src_k_ptr, src_v_ptr,
            dst_k_ptr, dst_v_ptr,
            indices_ptr, cu_topk_ptr, cu_k_ptr,
            src_len, kv_heads, (int)sl_end
        );
    } else {
        TORCH_CHECK(false, "The dim must be 128.");
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "h2d_gather_kv_packed kernel failed:",
                ::cudaGetErrorString(result));
}


at::Tensor uva_alias(const at::Tensor& cpu, int64_t device_index) {
    TORCH_CHECK(cpu.device().is_cpu(), "uva_alias expects a CPU tensor");
    TORCH_CHECK(cpu.is_pinned(), "uva_alias expects pinned host memory");
    TORCH_CHECK(cpu.is_contiguous(), "uva_alias expects a contiguous tensor");
    TORCH_CHECK(cpu.numel() > 0, "uva_alias expects a non-empty tensor");
    const auto index = static_cast<c10::DeviceIndex>(device_index);
    c10::cuda::CUDAGuard guard(index);
    void* dev = host_pinned_to_uva_dptr(cpu.data_ptr());
    auto opts = at::TensorOptions()
        .dtype(cpu.scalar_type())
        .device(at::Device(at::kCUDA, index));
    return at::from_blob(dev, cpu.sizes(), cpu.strides(), [](void*) {}, opts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("h2d_gather_kv", &h2d_gather_kv);
  m.def("h2d_gather_kv_i8", &h2d_gather_kv_i8);
  m.def("h2d_gather_kv_i4", &h2d_gather_kv_i4);
  m.def("h2d_gather_kv_cached", &h2d_gather_kv_cached);
  m.def("h2d_gather_kv_packed", &h2d_gather_kv_packed);
  m.def("uva_alias", &uva_alias);
}