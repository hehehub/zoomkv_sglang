"""Environment-backed configuration for the ZoomKV plugin."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _env_value(
    environ: Mapping[str, str],
    *names: str,
) -> str | None:
    """Read the first present name from an ordered canonical/legacy list."""
    for name in names:
        if name in environ:
            return environ[name]
    return None


def _parse_int(value: str | None, *, name: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _parse_bool(value: str | None, *, name: str, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}"
    )


def _parse_float(value: str | None, *, name: str, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value.strip())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


@dataclass(frozen=True, slots=True)
class ZoomKVConfig:
    """Supported ZoomKV settings.

    The current kernels intentionally expose only their validated operating
    point: 128-dimensional heads (Qwen3-4B/8B) or 256-dimensional heads
    (Qwen3.5-27B full attention) and 4-bit codes. KV storage may use pinned
    host offload (default) or a full GPU-resident sidecar.
    """

    head_dim: int = 128
    bits: int = 4
    offload: bool = True
    sink_size: int = 64
    local_size: int = 512
    final_topk: int = 1000
    candidate_ratio: float = 0.2
    dynamic_update_interval: int = 512
    full_attention_threshold: int = 2000
    max_new_tokens: int = 4096
    # First layers stay dense. Later layers use ZoomKV sparse prefill.
    prefill_dense_layers: int = 4
    prefill_block: int = 1024
    # Copy this layer's KV on the side stream while the next layer runs.
    # Off means the copy starts only after the whole prefill forward returns.
    prefill_overlap: bool = True

    def __post_init__(self) -> None:
        if self.head_dim not in (128, 256):
            raise ValueError(
                f"ZoomKV supports head_dim=128 or head_dim=256 only, got {self.head_dim}"
            )
        if self.bits != 4:
            raise ValueError(f"ZoomKV supports bits=4 only, got {self.bits}")
        if self.sink_size < 0:
            raise ValueError(f"sink_size must be non-negative, got {self.sink_size}")
        if self.local_size <= 0:
            raise ValueError(f"local_size must be positive, got {self.local_size}")
        if self.final_topk <= 0:
            raise ValueError(f"final_topk must be positive, got {self.final_topk}")
        if not 0.0 < self.candidate_ratio <= 1.0:
            raise ValueError(
                "candidate_ratio must be in the interval (0, 1], "
                f"got {self.candidate_ratio}"
            )
        if self.dynamic_update_interval <= 0:
            raise ValueError(
                "dynamic_update_interval must be positive, "
                f"got {self.dynamic_update_interval}"
            )
        if self.full_attention_threshold < 0:
            raise ValueError(
                "full_attention_threshold must be non-negative, "
                f"got {self.full_attention_threshold}"
            )
        if self.max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {self.max_new_tokens}"
            )
        if self.prefill_dense_layers < 0:
            raise ValueError(
                "prefill_dense_layers must be non-negative, "
                f"got {self.prefill_dense_layers}"
            )
        if self.prefill_block <= 0:
            raise ValueError(
                f"prefill_block must be positive, got {self.prefill_block}"
            )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "ZoomKVConfig":
        """Construct a validated config from canonical and legacy variables.

        Canonical ``SGLANG_ZOOMKV_*`` variables always take precedence. Legacy
        ``POLAR_*`` spellings remain accepted for migration.
        """
        env = os.environ if environ is None else environ
        return cls(
            head_dim=_parse_int(
                _env_value(
                    env, "SGLANG_ZOOMKV_HEAD_DIM", "POLAR_HEAD_DIM"
                ),
                name="SGLANG_ZOOMKV_HEAD_DIM",
                default=128,
            ),
            bits=_parse_int(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_BITS",
                    "SGLANG_ZOOMKV_KIVI_BITS",
                    "POLAR_BITS",
                    "POLAR_KIVI_BITS",
                ),
                name="SGLANG_ZOOMKV_BITS",
                default=4,
            ),
            offload=_parse_bool(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_OFFLOAD",
                    "SGLANG_ZOOMKV_ENABLE_OFFLOAD",
                    "POLAR_OFFLOAD",
                    "POLAR_ENABLE_OFFLOAD",
                ),
                name="SGLANG_ZOOMKV_OFFLOAD",
                default=True,
            ),
            sink_size=_parse_int(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_SINK_SIZE",
                    "SGLANG_ZOOMKV_SINK_TOKENS",
                    "POLAR_SINK_SIZE",
                    "POLAR_SINK_TOKENS",
                ),
                name="SGLANG_ZOOMKV_SINK_SIZE",
                default=64,
            ),
            local_size=_parse_int(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_LOCAL_SIZE",
                    "SGLANG_ZOOMKV_LOCAL_WINDOW",
                    "POLAR_LOCAL_SIZE",
                    "POLAR_LOCAL_WINDOW",
                    "POLAR_STATIC_PATTERN_END",
                ),
                name="SGLANG_ZOOMKV_LOCAL_SIZE",
                default=512,
            ),
            final_topk=_parse_int(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_FINAL_TOPK",
                    "SGLANG_ZOOMKV_TOP_K",
                    "POLAR_FINAL_TOPK",
                    "POLAR_TOP_K",
                ),
                name="SGLANG_ZOOMKV_FINAL_TOPK",
                default=1000,
            ),
            candidate_ratio=_parse_float(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_CANDIDATE_RATIO",
                    "POLAR_CANDIDATE_RATIO",
                ),
                name="SGLANG_ZOOMKV_CANDIDATE_RATIO",
                default=0.2,
            ),
            dynamic_update_interval=_parse_int(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_DYNAMIC_UPDATE_INTERVAL",
                    "SGLANG_ZOOMKV_UPDATE_INTERVAL",
                    "POLAR_DYNAMIC_UPDATE_INTERVAL",
                    "POLAR_UPDATE_INTERVAL",
                ),
                name="SGLANG_ZOOMKV_DYNAMIC_UPDATE_INTERVAL",
                default=512,
            ),
            full_attention_threshold=_parse_int(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_FULL_ATTENTION_THRESHOLD",
                    "POLAR_FULL_ATTENTION_THRESHOLD",
                ),
                name="SGLANG_ZOOMKV_FULL_ATTENTION_THRESHOLD",
                default=2000,
            ),
            max_new_tokens=_parse_int(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_MAX_NEW_TOKENS",
                    "POLAR_MAX_NEW_TOKENS",
                ),
                name="SGLANG_ZOOMKV_MAX_NEW_TOKENS",
                default=4096,
            ),
            prefill_dense_layers=_parse_int(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_PREFILL_DENSE_LAYERS",
                    "POLAR_PREFILL_DENSE_LAYERS",
                ),
                name="SGLANG_ZOOMKV_PREFILL_DENSE_LAYERS",
                default=4,
            ),
            prefill_block=_parse_int(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_PREFILL_BLOCK",
                    "POLAR_PREFILL_BLOCK",
                ),
                name="SGLANG_ZOOMKV_PREFILL_BLOCK",
                default=1024,
            ),
            prefill_overlap=_parse_bool(
                _env_value(
                    env,
                    "SGLANG_ZOOMKV_PREFILL_OVERLAP",
                    "POLAR_PREFILL_OVERLAP",
                ),
                name="SGLANG_ZOOMKV_PREFILL_OVERLAP",
                default=True,
            ),
        )

    @property
    def top_k(self) -> int:
        return self.final_topk

    @property
    def local_window(self) -> int:
        return self.local_size

    @property
    def sink_tokens(self) -> int:
        return self.sink_size

    @property
    def enable_offload(self) -> bool:
        return self.offload

    @property
    def update_interval(self) -> int:
        return self.dynamic_update_interval


def get_config(environ: Mapping[str, str] | None = None) -> ZoomKVConfig:
    """Return a freshly parsed, validated configuration."""
    return ZoomKVConfig.from_env(environ)
