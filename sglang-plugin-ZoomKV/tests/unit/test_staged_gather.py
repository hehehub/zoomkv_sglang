import importlib
import sys
from types import SimpleNamespace


class FakeTensor:
    def __init__(self, shape, dtype, *, cpu, pinned=False, name="tensor"):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.is_cpu = cpu
        self.is_cuda = not cpu
        self._pinned = pinned
        self.name = name
        self.device = "cpu" if cpu else "cuda:0"
        self.copies = []

    def is_pinned(self):
        return self._pinned

    def __getitem__(self, _index):
        return self

    def copy_(self, source, non_blocking=False):
        self.copies.append((source.name, non_blocking))
        return self


def test_staged_gather_never_calls_uva_kernel(monkeypatch):
    events = []
    fake_torch = SimpleNamespace(
        int64="int64",
        bfloat16="bfloat16",
        cuda=SimpleNamespace(
            current_stream=lambda _device: SimpleNamespace(
                synchronize=lambda: events.append("indices-synchronized")
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(
        sys.modules, "sglang_zoomkv.kernels.gather_trans.staged_h2d", raising=False
    )
    monkeypatch.delitem(
        sys.modules, "sglang_zoomkv.kernels.gather_trans", raising=False
    )
    staged_h2d = importlib.import_module(
        "sglang_zoomkv.kernels.gather_trans.staged_h2d"
    )

    src_k = FakeTensor((1, 2, 10, 128), "bfloat16", cpu=True, pinned=True, name="src-k")
    src_v = FakeTensor((1, 2, 10, 128), "bfloat16", cpu=True, pinned=True, name="src-v")
    indices = FakeTensor((1, 2, 3), "int64", cpu=False, name="gpu-indices")
    dst_k = FakeTensor((1, 2, 8, 128), "bfloat16", cpu=False, name="dst-k")
    dst_v = FakeTensor((1, 2, 8, 128), "bfloat16", cpu=False, name="dst-v")
    stage_k = FakeTensor((1, 2, 8, 128), "bfloat16", cpu=True, pinned=True, name="stage-k")
    stage_v = FakeTensor((1, 2, 8, 128), "bfloat16", cpu=True, pinned=True, name="stage-v")
    stage_i = FakeTensor((1, 2, 8), "int64", cpu=True, pinned=True, name="stage-i")

    class GatherExt:
        def gather_kv_out_cpu(self, source, host_indices, target):
            events.append(("cpu-gather", source.name, target.name))

    staged_h2d.staged_h2d_gather_kv(
        src_k, src_v, indices, dst_k, dst_v, stage_k, stage_v, stage_i,
        gather_ext=GatherExt(),
    )

    assert stage_i.copies == [("gpu-indices", True)]
    assert events == [
        "indices-synchronized",
        ("cpu-gather", "src-k", "stage-k"),
        ("cpu-gather", "src-v", "stage-v"),
    ]
    assert dst_k.copies == [("stage-k", True)]
    assert dst_v.copies == [("stage-v", True)]
