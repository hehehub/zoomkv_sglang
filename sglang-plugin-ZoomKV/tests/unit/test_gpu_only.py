import pytest
from types import SimpleNamespace

from sglang_zoomkv.cache.gpu_only import (
    append_position,
    retrieval_to_cache_indices,
    steady_indices,
)


class FakeIndices:
    dtype = "int64"

    def __init__(self, values):
        self.values = list(values)

    def numel(self):
        return len(self.values)

    def __lt__(self, value):
        return SimpleNamespace(any=lambda: any(v < value for v in self.values))

    def __ge__(self, value):
        return SimpleNamespace(any=lambda: any(v >= value for v in self.values))

    def __add__(self, value):
        return FakeIndices([v + value for v in self.values])


def test_gpu_only_indices_keep_zones_disjoint_and_ordered(monkeypatch):
    import sys

    sink = FakeIndices([0, 1])
    local = FakeIndices([7, 8, 9])
    retrieval = FakeIndices([0, 2, 4])
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cat=lambda tensors: FakeIndices(
            tensors[0].values + tensors[1].values
        )),
    )

    assert steady_indices(sink, local).values == [0, 1, 7, 8, 9]
    assert retrieval_to_cache_indices(retrieval, 2, 5).values == [2, 4, 6]


def test_gpu_only_retrieval_index_validation_prevents_zone_overlap():
    with pytest.raises(IndexError, match="encoded retrieval zone"):
        retrieval_to_cache_indices(FakeIndices([5]), sink_size=2, retrieval_length=5)


def test_gpu_only_append_is_chronological_and_capacity_checked():
    assert append_position(12, 16) == 12
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        append_position(16, 16)
