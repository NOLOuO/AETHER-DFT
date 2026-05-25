from __future__ import annotations

from copy import deepcopy
import os
from typing import Any


PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "timeout_seconds": 180,
        "max_tokens": 4096,
        "supports_temperature": False,
        "default_temperature": None,
        "supports_top_p": False,
        "default_top_p": None,
        "models": [
            {
                "id": "deepseek-v4-pro",
                "label": "DeepSeek V4 Pro",
                "api_model": "deepseek-v4-pro",
                "note": "DeepSeek 官方 V4 Pro；OpenAI-compatible 调用；thinking mode enabled；上下文窗口 1M tokens。",
                "extra_body": {"thinking": {"type": "enabled"}},
                "reasoning_effort": "max",
                "context_window": 1_000_000,
            },
        ],
    },
    "bailian": {
        "label": "阿里百炼",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "timeout_seconds": 180,
        "max_tokens": 1600,
        "supports_temperature": True,
        "default_temperature": 0.2,
        "supports_top_p": True,
        "default_top_p": 0.95,
        "models": [
            {
                "id": "qwen3.7-max",
                "label": "Qwen 3.7 Max",
                "api_model": "qwen3.7-max",
                "note": "通过阿里百炼 DashScope OpenAI-compatible 接口调用；上下文窗口 1M tokens。",
                "extra_body": {"enable_thinking": True},
                "context_window": 1_000_000,
            },
        ],
    },
}


def list_provider_ids() -> list[str]:
    return list(PROVIDER_PRESETS.keys())


def get_provider(provider_id: str) -> dict[str, Any]:
    if provider_id not in PROVIDER_PRESETS:
        raise KeyError(f"未知 provider: {provider_id}")
    return deepcopy(PROVIDER_PRESETS[provider_id])


def list_models(provider_id: str) -> list[dict[str, Any]]:
    provider = get_provider(provider_id)
    return provider.get("models", [])


def default_model_id(provider_id: str) -> str:
    models = list_models(provider_id)
    if not models:
        raise KeyError(f"provider={provider_id} 未配置默认模型")
    return str(models[0]["id"])


def build_provider_model_config(provider_id: str, model_id: str | None = None) -> dict[str, Any]:
    provider = get_provider(provider_id)
    models = provider.pop("models", [])
    target_model_id = model_id or default_model_id(provider_id)
    selected = None
    for item in models:
        if item["id"] == target_model_id:
            selected = deepcopy(item)
            break
    if selected is None:
        raise KeyError(f"provider={provider_id} 下不存在 model={target_model_id}")

    config = deepcopy(provider)
    base_url_env = str(config.get("base_url_env", "") or "").strip()
    if base_url_env and os.getenv(base_url_env, "").strip():
        config["base_url"] = os.getenv(base_url_env, "").strip()
    config["provider_id"] = provider_id
    config["model_id"] = selected["id"]
    config["model_label"] = selected["label"]
    config["model"] = selected["api_model"]
    config["model_note"] = selected.get("note", "")
    config["extra_body"] = selected.get("extra_body", {})
    if "context_window" in selected:
        config["context_window"] = selected.get("context_window")
    if "max_output_tokens" in selected:
        config["max_output_tokens"] = selected.get("max_output_tokens")
    if "reasoning_effort" in selected:
        config["reasoning_effort"] = selected.get("reasoning_effort")
    if "temperature" in selected:
        config["default_temperature"] = selected.get("temperature")
    if "top_p" in selected:
        config["default_top_p"] = selected.get("top_p")
    return config
