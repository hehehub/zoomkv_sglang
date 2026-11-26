"""Opt-in CUDA smoke tests.

Kernel wrappers are imported only inside tests, after the CUDA skip decision,
so collection on CPU hosts never starts native compilation.
"""

import pytest


pytestmark = pytest.mark.gpu


def _torch_or_skip_cuda():
    try:
        import torch
    except Exception as exc:  # broken/CPU-only environments should still collect
        pytest.skip(f"PyTorch CUDA runtime is unavailable: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable PyTorch runtime")
    return torch


def test_cuda_kernel_wrapper_imports():
    _torch_or_skip_cuda()
    from sglang_zoomkv.kernels.jit import load_jit
    from sglang_zoomkv.kernels.topk.float_topk_ops import float_topk

    assert callable(load_jit)
    assert callable(float_topk)


def test_float_topk_native_kernel_smoke():
    torch = _torch_or_skip_cuda()
    from sglang_zoomkv.kernels.topk.float_topk_ops import float_topk

    scores = torch.tensor(
        [[0.25, 3.0, -1.0, 2.0]], device="cuda", dtype=torch.float32
    )
    indices = float_topk(scores, 2)

    assert set(indices.cpu().tolist()[0]) == {1, 3}


def test_staged_h2d_gather_fallback_smoke():
    torch = _torch_or_skip_cuda()
    from sglang_zoomkv.kernels.gather_trans.staged_h2d import (
        staged_h2d_gather_kv,
    )

    shape = (1, 1, 6, 128)
    src_k = torch.arange(6 * 128, dtype=torch.bfloat16).reshape(shape).pin_memory()
    src_v = (src_k + 1).pin_memory()
    indices = torch.tensor([[[4, 1]]], dtype=torch.int64, device="cuda")
    dst_k = torch.empty(1, 1, 2, 128, dtype=torch.bfloat16, device="cuda")
    dst_v = torch.empty_like(dst_k)
    stage_k = torch.empty(1, 1, 2, 128, dtype=torch.bfloat16, pin_memory=True)
    stage_v = torch.empty_like(stage_k, pin_memory=True)
    stage_i = torch.empty(1, 1, 2, dtype=torch.int64, pin_memory=True)

    staged_h2d_gather_kv(
        src_k, src_v, indices, dst_k, dst_v, stage_k, stage_v, stage_i,
        gather_ext=None,
    )
    torch.cuda.synchronize()

    expected_k = src_k[:, :, [4, 1], :]
    expected_v = src_v[:, :, [4, 1], :]
    assert torch.equal(dst_k.cpu(), expected_k)
    assert torch.equal(dst_v.cpu(), expected_v)
