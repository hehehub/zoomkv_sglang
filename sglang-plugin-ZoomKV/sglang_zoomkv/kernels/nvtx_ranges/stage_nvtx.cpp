// NVTX ranges that survive CUDA graph replay.
//
// torch.cuda.nvtx ranges pushed while a graph is captured are not replayed.
// cudaLaunchHostFunc is captured as a host node, so the callback runs on every
// replay and the range covers the GPU work ordered between the two nodes.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <nvtx3/nvToolsExt.h>

#include <atomic>

namespace {

constexpr int kStages = 4;

// Names are what nsys shows. Keep them stable.
const char* const kNames[kStages] = {
    "large_chunk_select",
    "small_chunk_select",
    "fine_rerank",
    "uva_read",
};

std::atomic<uint64_t> kIds[kStages];

void start_stage(int stage) {
    kIds[stage].store(nvtxRangeStartA(kNames[stage]), std::memory_order_release);
}

void end_stage(int stage) {
    nvtxRangeEnd(static_cast<nvtxRangeId_t>(kIds[stage].load(std::memory_order_acquire)));
}

void start0(void*) { start_stage(0); }
void end0(void*) { end_stage(0); }
void start1(void*) { start_stage(1); }
void end1(void*) { end_stage(1); }
void start2(void*) { start_stage(2); }
void end2(void*) { end_stage(2); }
void start3(void*) { start_stage(3); }
void end3(void*) { end_stage(3); }

using HostFn = void (*)(void*);

HostFn kStart[kStages] = {start0, start1, start2, start3};
HostFn kEnd[kStages] = {end0, end1, end2, end3};

}  // namespace

static void launch_mark(int stage, bool begin) {
    TORCH_CHECK(stage >= 0 && stage < 4, "ZoomKV NVTX stage out of range");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    cudaError_t err = cudaLaunchHostFunc(
        stream, begin ? kStart[stage] : kEnd[stage], nullptr);
    TORCH_CHECK(err == cudaSuccess, "cudaLaunchHostFunc failed: ", cudaGetErrorString(err));
}

static void warmup() {
    nvtxRangeId_t id = nvtxRangeStartA("zoomkv_nvtx_warmup");
    nvtxRangeEnd(id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warmup", &warmup);
    m.def("range_begin", [](int stage) { launch_mark(stage, true); });
    m.def("range_end", [](int stage) { launch_mark(stage, false); });
}
