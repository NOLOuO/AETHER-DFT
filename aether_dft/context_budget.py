from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from dft_app.llm.provider_presets import build_provider_model_config

from .model_catalog import resolve_effective_model_id, split_model_id

DEFAULT_CONTEXT_WINDOW_TOKENS = 1_000_000
DEFAULT_CONTEXT_RESERVE_TOKENS = 64_000
DEFAULT_AUTO_COMPACT_RATIO = 0.72
DEFAULT_CONTEXT_GUARD_RATIO = 0.88
APPROX_CHARS_PER_TOKEN = 3


@dataclass(frozen=True)
class ContextBudget:
    model_id: str
    context_window_tokens: int
    reserve_tokens: int
    usable_tokens: int
    approx_chars_per_token: int
    usable_chars: int
    auto_compact_ratio: float
    auto_compact_chars: int
    guard_ratio: float
    guard_chars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounded_ratio(raw: str, *, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return min(0.97, max(0.10, value))


def _reserve_tokens() -> int:
    try:
        return max(0, int(os.getenv("AETHER_DFT_CONTEXT_RESERVE_TOKENS", str(DEFAULT_CONTEXT_RESERVE_TOKENS))))
    except ValueError:
        return DEFAULT_CONTEXT_RESERVE_TOKENS


def _auto_compact_ratio() -> float:
    return _bounded_ratio(os.getenv("AETHER_DFT_AUTO_COMPACT_RATIO", str(DEFAULT_AUTO_COMPACT_RATIO)), default=DEFAULT_AUTO_COMPACT_RATIO)


def _guard_ratio() -> float:
    return _bounded_ratio(os.getenv("AETHER_DFT_CONTEXT_GUARD_RATIO", str(DEFAULT_CONTEXT_GUARD_RATIO)), default=DEFAULT_CONTEXT_GUARD_RATIO)


def current_context_window_tokens(model_id: str | None = None) -> int:
    resolved = model_id or resolve_effective_model_id()
    provider_id, model_name = split_model_id(resolved)
    config = build_provider_model_config(provider_id, model_name)
    return int(config.get("context_window") or DEFAULT_CONTEXT_WINDOW_TOKENS)


def usable_context_tokens(model_id: str | None = None) -> int:
    window = current_context_window_tokens(model_id)
    reserve = _reserve_tokens()
    return max(8_000, window - reserve)


def usable_context_chars(model_id: str | None = None) -> int:
    override = os.getenv("AETHER_DFT_CONTEXT_MAX_CHARS", "").strip()
    if override:
        return max(12_000, int(override))
    return usable_context_tokens(model_id) * APPROX_CHARS_PER_TOKEN


def context_budget(model_id: str | None = None) -> ContextBudget:
    resolved = model_id or resolve_effective_model_id()
    window = current_context_window_tokens(resolved)
    reserve = min(_reserve_tokens(), max(0, window - 8_000))
    usable_tokens_value = max(8_000, window - reserve)
    usable_chars_value = usable_context_chars(resolved)
    compact_ratio = _auto_compact_ratio()
    guard_ratio = _guard_ratio()
    return ContextBudget(
        model_id=resolved,
        context_window_tokens=window,
        reserve_tokens=reserve,
        usable_tokens=usable_tokens_value,
        approx_chars_per_token=APPROX_CHARS_PER_TOKEN,
        usable_chars=usable_chars_value,
        auto_compact_ratio=compact_ratio,
        auto_compact_chars=max(8_000, int(usable_chars_value * compact_ratio)),
        guard_ratio=guard_ratio,
        guard_chars=max(8_000, int(usable_chars_value * guard_ratio)),
    )
