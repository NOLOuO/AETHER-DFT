from __future__ import annotations

import os

from dft_app.llm.provider_presets import build_provider_model_config

from .model_catalog import resolve_effective_model_id, split_model_id

DEFAULT_CONTEXT_WINDOW_TOKENS = 1_000_000
DEFAULT_CONTEXT_RESERVE_TOKENS = 64_000
APPROX_CHARS_PER_TOKEN = 3


def current_context_window_tokens(model_id: str | None = None) -> int:
    resolved = model_id or resolve_effective_model_id()
    provider_id, model_name = split_model_id(resolved)
    config = build_provider_model_config(provider_id, model_name)
    return int(config.get("context_window") or DEFAULT_CONTEXT_WINDOW_TOKENS)


def usable_context_tokens(model_id: str | None = None) -> int:
    window = current_context_window_tokens(model_id)
    reserve = int(os.getenv("AETHER_DFT_CONTEXT_RESERVE_TOKENS", str(DEFAULT_CONTEXT_RESERVE_TOKENS)))
    return max(8_000, window - reserve)


def usable_context_chars(model_id: str | None = None) -> int:
    override = os.getenv("AETHER_DFT_CONTEXT_MAX_CHARS", "").strip()
    if override:
        return max(12_000, int(override))
    return usable_context_tokens(model_id) * APPROX_CHARS_PER_TOKEN
