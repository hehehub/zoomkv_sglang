import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def register_module(monkeypatch):
    registry = ModuleType("sglang.srt.layers.attention.attention_registry")
    registry.ATTENTION_BACKENDS = {}

    def register(name):
        def decorate(factory):
            registry.ATTENTION_BACKENDS[name] = factory
            return factory

        return decorate

    registry.register_attention_backend = register
    server_args = ModuleType("sglang.srt.server_args")
    server_args.ATTENTION_BACKEND_CHOICES = ()
    server_args.add_attention_backend_choices = lambda names: None
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.attention_registry",
        registry,
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.delitem(
        sys.modules, "sglang_zoomkv.attention.register", raising=False
    )

    module = importlib.import_module("sglang_zoomkv.attention.register")
    yield module, registry
    sys.modules.pop("sglang_zoomkv.attention.register", None)


def test_import_registers_zoomkv_without_loading_backend(register_module):
    module, registry = register_module

    assert registry.ATTENTION_BACKENDS["zoomkv"] is module.create_zoomkv_backend
    assert "sglang_zoomkv.attention.backend" not in sys.modules


def test_factory_builds_fallback_then_zoomkv_backend(
    register_module, monkeypatch
):
    module, _ = register_module
    runner = SimpleNamespace(model_config=SimpleNamespace(head_dim=128))
    fallback = object()
    calls = []

    monkeypatch.setenv("SGLANG_ZOOMKV_FALLBACK_BACKEND", "fake")
    monkeypatch.setattr(
        module,
        "get_attention_backend_factory",
        lambda name: (
            lambda value: calls.append(("fallback", name, value)) or fallback
        ),
    )

    class FakeBackend:
        def __init__(self, value, fallback_backend):
            calls.append(("backend", value, fallback_backend))

    monkeypatch.setattr(module, "load_backend_class", lambda: FakeBackend)

    result = module.create_zoomkv_backend(runner)

    assert isinstance(result, FakeBackend)
    assert calls == [
        ("fallback", "fake", runner),
        ("backend", runner, fallback),
    ]


@pytest.mark.parametrize(
    ("head_dim", "fallback", "message"),
    [
        (64, "torch_native", "model reports head_dim=64"),
        (128, "", "cannot be empty"),
        (128, "zoomkv", "cannot be 'zoomkv'"),
    ],
)
def test_factory_rejects_invalid_runtime(
    register_module, monkeypatch, head_dim, fallback, message
):
    module, _ = register_module
    monkeypatch.setenv("SGLANG_ZOOMKV_FALLBACK_BACKEND", fallback)
    runner = SimpleNamespace(model_config=SimpleNamespace(head_dim=head_dim))

    with pytest.raises(ValueError, match=message):
        module.create_zoomkv_backend(runner)
