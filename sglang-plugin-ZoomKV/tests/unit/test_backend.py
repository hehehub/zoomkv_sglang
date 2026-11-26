import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


class FakeTensor:
    def __init__(self, value=None, shape=(), dtype="float32"):
        self.value = value
        self.shape = tuple(shape)
        self.dtype = dtype

    def item(self):
        return self.value

    def reshape(self, *shape):
        return FakeTensor(self.value, shape, self.dtype)

    def contiguous(self):
        return self

    def long(self):
        return self

    def __getitem__(self, index):
        if isinstance(self.value, (list, tuple)):
            return FakeTensor(self.value[index], (), self.dtype)
        return FakeTensor(self.value, (), self.dtype)


@pytest.fixture
def backend_class(monkeypatch):
    torch_module = ModuleType("torch")
    torch_module.Tensor = FakeTensor
    torch_module.dtype = type
    torch_module.bfloat16 = "bfloat16"
    torch_module.ones = lambda *shape: FakeTensor(shape=shape)
    torch_module.zeros = lambda shape: FakeTensor(shape=shape)
    torch_module.tensor = lambda value: FakeTensor(value=value, shape=(len(value),))
    base_module = ModuleType(
        "sglang.srt.layers.attention.base_attn_backend"
    )

    class AttentionBackend:
        def __init__(self):
            pass

    base_module.AttentionBackend = AttentionBackend
    pool_module = ModuleType("sglang.srt.mem_cache.memory_pool")
    pool_module.KVWriteLoc = lambda value: ("write-location", value)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.base_attn_backend",
        base_module,
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.mem_cache.memory_pool", pool_module
    )
    monkeypatch.delitem(
        sys.modules, "sglang_zoomkv.attention.backend", raising=False
    )

    module = importlib.import_module("sglang_zoomkv.attention.backend")
    yield module.ZoomKVAttentionBackend
    sys.modules.pop("sglang_zoomkv.attention.backend", None)


def _valid_runner(**server_arg_overrides):
    server_args = {
        "tp_size": 1,
        "disable_cuda_graph": True,
        "chunked_prefill_size": -1,
        "disable_radix_cache": True,
    }
    server_args.update(server_arg_overrides)
    return SimpleNamespace(
        server_args=SimpleNamespace(**server_args),
        model_config=SimpleNamespace(
            head_dim=128, attention_arch="MHA", dtype="bfloat16"
        ),
    )


def test_validate_runner_accepts_supported_envelope(backend_class):
    backend = backend_class.__new__(backend_class)
    backend.runner = _valid_runner()

    backend._validate_runner()

    backend.runner = _valid_runner(tp_size=2)
    backend._validate_runner()


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (_valid_runner(tp_size=4), "tp_size must be 1 or 2"),
        (_valid_runner(disable_cuda_graph=False), "disable-cuda-graph"),
        (_valid_runner(chunked_prefill_size=2048), "chunked-prefill-size"),
        (_valid_runner(disable_radix_cache=False), "disable-radix-cache"),
    ],
)
def test_validate_runner_rejects_unsupported_server_state(
    backend_class, runner, message
):
    backend = backend_class.__new__(backend_class)
    backend.runner = runner

    with pytest.raises(ValueError, match=message):
        backend._validate_runner()


def test_validate_runner_rejects_model_shape_and_mla(backend_class):
    backend = backend_class.__new__(backend_class)
    backend.runner = _valid_runner()
    backend.runner.model_config = SimpleNamespace(
        head_dim=64, attention_arch="MLA"
    )

    with pytest.raises(ValueError, match="head_dim.*MLA"):
        backend._validate_runner()


def test_validate_runner_rejects_non_bfloat16_model(backend_class):
    backend = backend_class.__new__(backend_class)
    backend.runner = _valid_runner()
    backend.runner.model_config.dtype = "float16"

    with pytest.raises(ValueError, match="dtype must be bfloat16"):
        backend._validate_runner()


def test_batch_state_returns_slots_and_lengths(backend_class):
    backend = backend_class.__new__(backend_class)
    batch = SimpleNamespace(
        batch_size=2,
        req_pool_indices=FakeTensor([0, 1], (2,)),
        seq_lens=FakeTensor([4, 4], (2,)),
    )

    assert backend._batch_state(batch) == ((0, 1), (4, 4))


def test_ragged_batch_delegates_to_fallback(backend_class):
    backend = backend_class.__new__(backend_class)
    expected = object()
    backend.fallback = SimpleNamespace(
        forward_decode=lambda *args, **kwargs: expected
    )
    layer = SimpleNamespace(
        is_cross_attention=False, sliding_window_size=-1
    )
    batch = SimpleNamespace(
        batch_size=2,
        req_pool_indices=FakeTensor([0, 1], (2,)),
        seq_lens=FakeTensor([4, 5], (2,)),
    )

    assert (
        backend.forward_decode(
            FakeTensor(shape=(2,)),
            FakeTensor(shape=(2,)),
            FakeTensor(shape=(2,)),
            layer,
            batch,
        )
        is expected
    )


def test_reset_constructs_cache_with_current_production_api(
    backend_class, monkeypatch
):
    captured = {}
    cache_instance = SimpleNamespace()
    cache_module = ModuleType("sglang_zoomkv.cache")

    def make_cache(**kwargs):
        captured.update(kwargs)
        return cache_instance

    cache_module.ZoomKVCache = make_cache
    monkeypatch.setitem(sys.modules, "sglang_zoomkv.cache", cache_module)
    backend = backend_class.__new__(backend_class)
    backend.config = SimpleNamespace(
        max_new_tokens=32,
        sink_size=8,
        local_size=16,
        candidate_ratio=0.25,
        final_topk=12,
        dynamic_update_interval=4,
        full_attention_threshold=64,
        enable_offload=True,
    )
    backend.runner = SimpleNamespace(
        model_config=SimpleNamespace(num_attention_layers=3)
    )
    backend.device = "cuda:0"
    backend._initialized_layers = {0, 1}
    layer = SimpleNamespace(
        tp_k_head_num=2,
        tp_q_head_num=4,
        qk_head_dim=128,
    )

    backend._reset_for_batch((7, 8), 100, layer, "bfloat16")

    assert backend._cache is cache_instance
    assert backend._request_slots == (7, 8)
    assert backend._batch_size == 2
    assert backend._initialized_layers == set()
    assert captured["batch_size"] == 2
    assert captured["max_length"] == 132
    assert captured["max_new_length"] == 32
    assert captured["layer_num"] == 3
    assert captured["layer_mapping"] == {
        0: "cuda:0",
        1: "cuda:0",
        2: "cuda:0",
    }
    assert captured["head_dim"] == 128
    assert captured["enable_offload"] is True


def test_reset_passes_gpu_only_mode_to_cache(backend_class, monkeypatch):
    captured = {}
    cache_module = ModuleType("sglang_zoomkv.cache")
    cache_module.ZoomKVCache = lambda **kwargs: captured.update(kwargs) or object()
    monkeypatch.setitem(sys.modules, "sglang_zoomkv.cache", cache_module)
    backend = backend_class.__new__(backend_class)
    backend.config = SimpleNamespace(
        max_new_tokens=32,
        sink_size=8,
        local_size=16,
        candidate_ratio=0.25,
        final_topk=12,
        dynamic_update_interval=4,
        full_attention_threshold=64,
        enable_offload=False,
    )
    backend.runner = SimpleNamespace(
        model_config=SimpleNamespace(num_attention_layers=1)
    )
    backend.device = "cuda:0"
    backend._initialized_layers = set()

    backend._reset_for_batch(
        (0,), 100,
        SimpleNamespace(tp_k_head_num=2, tp_q_head_num=4, qk_head_dim=128),
        "bfloat16",
    )

    assert captured["enable_offload"] is False


def test_forward_decode_updates_both_caches_and_shapes_output(
    backend_class,
):
    calls = []

    class FakeTokenPool:
        def set_kv_buffer(self, *args):
            calls.append(("sglang-cache", args))

    class FakeCache:
        current_seq_len = [2, 2]

        def decode_update_kv_cache(self, key, value, layer_id):
            calls.append(("sidecar", key.shape, value.shape, layer_id))

        def compute(self, query, layer_id):
            calls.append(("compute", query.shape, layer_id))
            return FakeTensor(shape=(1, 1, 2, 128))

    backend = backend_class.__new__(backend_class)
    backend.token_to_kv_pool = FakeTokenPool()
    backend._cache = FakeCache()
    backend._request_slots = (3,)
    backend._batch_size = 1
    backend._ensure_layer_prefill = (
        lambda layer, batch, prefix_len: calls.append(
            ("prefill", layer.layer_id, prefix_len)
        )
    )
    layer = SimpleNamespace(
        layer_id=1,
        is_cross_attention=False,
        sliding_window_size=-1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        tp_q_head_num=2,
        qk_head_dim=128,
        v_head_dim=128,
    )
    batch = SimpleNamespace(
        batch_size=1,
        req_pool_indices=FakeTensor([3], (1,)),
        seq_lens=FakeTensor([3], (1,)),
        out_cache_loc=FakeTensor([7], (1,)),
    )
    q = FakeTensor(shape=(1, 256), dtype="bfloat16")
    k = FakeTensor(shape=(1, 128), dtype="bfloat16")
    v = FakeTensor(shape=(1, 128), dtype="bfloat16")

    output = backend.forward_decode(q, k, v, layer, batch)

    assert output.shape == (1, 256)
    assert [call[0] for call in calls] == [
        "sglang-cache",
        "prefill",
        "sidecar",
        "compute",
    ]


def test_forward_decode_supports_lockstep_batch(backend_class):
    calls = []

    class FakeCache:
        current_seq_len = [8]

        def decode_update_kv_cache(self, key, value, layer_id):
            calls.append(("update", key.shape, value.shape, layer_id))

        def compute(self, query, layer_id):
            calls.append(("compute", query.shape, layer_id))
            return FakeTensor(shape=(2, 1, 4, 128))

    backend = backend_class.__new__(backend_class)
    backend.token_to_kv_pool = SimpleNamespace(
        set_kv_buffer=lambda *args: calls.append(("sglang", args))
    )
    backend._cache = FakeCache()
    backend._request_slots = (3, 4)
    backend._batch_size = 2
    backend._ensure_layer_prefill = lambda *args: None
    layer = SimpleNamespace(
        layer_id=0,
        is_cross_attention=False,
        sliding_window_size=-1,
        tp_k_head_num=2,
        tp_v_head_num=2,
        tp_q_head_num=4,
        qk_head_dim=128,
        v_head_dim=128,
    )
    batch = SimpleNamespace(
        batch_size=2,
        req_pool_indices=FakeTensor([3, 4], (2,)),
        seq_lens=FakeTensor([9, 9], (2,)),
        out_cache_loc=FakeTensor([7, 8], (2,)),
    )

    output = backend.forward_decode(
        FakeTensor(shape=(2, 512), dtype="bfloat16"),
        FakeTensor(shape=(2, 256), dtype="bfloat16"),
        FakeTensor(shape=(2, 256), dtype="bfloat16"),
        layer,
        batch,
    )

    assert output.shape == (2, 512)
    assert calls[1] == ("update", (2, 1, 2, 128), (2, 1, 2, 128), 0)
    assert calls[2] == ("compute", (2, 1, 4, 128), 0)


def test_short_decode_delegates_to_fallback(backend_class):
    expected = object()
    backend = backend_class.__new__(backend_class)
    backend.fallback = SimpleNamespace(
        forward_decode=lambda *args, **kwargs: expected
    )
    layer = SimpleNamespace(
        is_cross_attention=False, sliding_window_size=None
    )
    batch = SimpleNamespace(
        batch_size=1,
        req_pool_indices=FakeTensor([0], (1,)),
        seq_lens=FakeTensor([1], (1,)),
    )

    assert (
        backend.forward_decode(
            FakeTensor(shape=(1,)),
            FakeTensor(shape=(1,)),
            FakeTensor(shape=(1,)),
            layer,
            batch,
        )
        is expected
    )


def test_decode_rejects_unsupported_attention_inputs(backend_class):
    backend = backend_class.__new__(backend_class)
    layer = SimpleNamespace(
        is_cross_attention=False, sliding_window_size=-1
    )

    with pytest.raises(RuntimeError, match="extra attention inputs: custom_mask"):
        backend.forward_decode(
            FakeTensor(shape=(1,)),
            FakeTensor(shape=(1,)),
            FakeTensor(shape=(1,)),
            layer,
            object(),
            custom_mask=object(),
        )


def test_decode_rejects_non_bfloat16_tensors(backend_class):
    backend = backend_class.__new__(backend_class)
    backend._cache = SimpleNamespace(current_seq_len=[1])
    backend._request_slots = (0,)
    backend._batch_size = 1
    backend.token_to_kv_pool = SimpleNamespace(set_kv_buffer=lambda *args: None)
    backend._ensure_layer_prefill = lambda *args: None
    layer = SimpleNamespace(
        layer_id=0,
        is_cross_attention=False,
        sliding_window_size=-1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        tp_q_head_num=1,
        qk_head_dim=128,
        v_head_dim=128,
    )
    batch = SimpleNamespace(
        batch_size=1,
        req_pool_indices=FakeTensor([0], (1,)),
        seq_lens=FakeTensor([2], (1,)),
        out_cache_loc=FakeTensor([1], (1,)),
    )

    with pytest.raises(RuntimeError, match="bfloat16 Q/K/V"):
        backend.forward_decode(
            FakeTensor(shape=(1, 128), dtype="float16"),
            FakeTensor(shape=(1, 128), dtype="float16"),
            FakeTensor(shape=(1, 128), dtype="float16"),
            layer,
            batch,
        )
