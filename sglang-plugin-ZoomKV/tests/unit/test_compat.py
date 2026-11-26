from types import SimpleNamespace

import pytest

from sglang_zoomkv import compat


def test_register_backend_uses_decorator_and_extends_cli_choices(monkeypatch):
    calls = []
    registry = SimpleNamespace(
        register_attention_backend=lambda name: (
            lambda factory: calls.append(("register", name, factory))
        )
    )
    server_args = SimpleNamespace(
        ATTENTION_BACKEND_CHOICES=("flashinfer",),
        add_attention_backend_choices=lambda names: calls.append(("choices", names)),
    )
    modules = {
        "sglang.srt.layers.attention.attention_registry": registry,
        "sglang.srt.server_args": server_args,
    }
    monkeypatch.setattr(compat, "import_module", modules.__getitem__)

    factory = object()
    compat.register_attention_backend("zoomkv", factory)

    assert calls == [
        ("register", "zoomkv", factory),
        ("choices", ["zoomkv"]),
    ]


def test_register_backend_supports_dictionary_registry(monkeypatch):
    backends = {}
    modules = {
        "sglang.srt.layers.attention.attention_registry": SimpleNamespace(
            ATTENTION_BACKENDS=backends
        ),
        "sglang.srt.server_args": SimpleNamespace(
            ATTENTION_BACKEND_CHOICES=("zoomkv",),
            add_attention_backend_choices=lambda names: pytest.fail(
                f"unexpected duplicate choices: {names}"
            ),
        ),
    }
    monkeypatch.setattr(compat, "import_module", modules.__getitem__)

    factory = object()
    compat.register_attention_backend("zoomkv", factory)

    assert backends == {"zoomkv": factory}


def test_missing_registry_has_actionable_error(monkeypatch):
    monkeypatch.setattr(
        compat,
        "import_module",
        lambda name: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="does not expose"):
        compat.register_attention_backend("zoomkv", object())


def test_get_backend_factory_reports_available_backends(monkeypatch):
    registry = SimpleNamespace(ATTENTION_BACKENDS={"torch_native": object()})
    monkeypatch.setattr(compat, "import_module", lambda name: registry)

    with pytest.raises(ValueError, match="available backends: torch_native"):
        compat.get_attention_backend_factory("missing")


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        (SimpleNamespace(head_dim=128), 128),
        (SimpleNamespace(hf_text_config=SimpleNamespace(head_size=64)), 64),
        (
            SimpleNamespace(
                hf_config=SimpleNamespace(hidden_size=4096, num_attention_heads=32)
            ),
            128,
        ),
        (None, None),
    ],
)
def test_get_runner_head_dim_across_sglang_layouts(model_config, expected):
    assert compat.get_runner_head_dim(SimpleNamespace(model_config=model_config)) == expected
