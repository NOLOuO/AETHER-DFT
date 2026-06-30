"""独立 LLM 客户端：直接调用 OpenAI-compatible API。

默认以本仓 provider 配置为主，同时提供 OpenAI-compatible
`call_openai_compatible_result` 适配接口，便于结果解释服务复用。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-reasoner",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "minimax": {
        "base_url": "https://api.minimaxi.com/v1",
        "default_model": "minimax-m2.7",
        "env_key": "MINIMAX_API_KEY",
    },
    "bailian": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen3.5-plus",
        "env_key": "DASHSCOPE_API_KEY",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "kimi-k2.5",
        "env_key": "KIMI_API_KEY",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-5",
        "env_key": "GLM_API_KEY",
    },
}

DEFAULT_RACE_CANDIDATES: list[tuple[str, str]] = [
    ("deepseek", "deepseek-reasoner"),
    ("minimax", "minimax-m2.7"),
    ("bailian", "qwen3.5-plus"),
]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_KEY_FILE = _PROJECT_ROOT / "api_keys.local.json"
_LEGACY_KEY_FILE = _PROJECT_ROOT / "api_keys.json"
_PROVIDER_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "kimi": ("kimi", "moonshot"),
    "moonshot": ("moonshot", "kimi"),
}


def _key_file_candidates() -> list[Path]:
    paths = [_KEY_FILE, _LEGACY_KEY_FILE]
    extra_paths = os.getenv("AETHER_DFT_API_KEYS_PATHS", "").strip()
    if extra_paths:
        paths.extend(Path(item.strip()) for item in extra_paths.split(";") if item.strip())
    deduped: list[Path] = []
    for path in paths:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _normalize_key_value(value: Any) -> str:
    if isinstance(value, dict):
        for field in ("api_key", "key", "token", "value"):
            candidate = str(value.get(field) or "").strip()
            if candidate:
                return candidate
        return ""
    return str(value or "").strip()


def _load_api_key(provider: str) -> str:
    """按优先级查找 API key：环境变量 → 本项目 key 文件 → 显式环境变量指定文件。"""
    prov_cfg = PROVIDERS.get(provider, {})
    env_key = prov_cfg.get("env_key", f"{provider.upper()}_API_KEY")

    key = os.getenv(env_key, "").strip()
    if key:
        return key

    lookup_names = _PROVIDER_KEY_ALIASES.get(provider, (provider,))
    candidates = _key_file_candidates()
    for path in candidates:
        if not path.exists():
            continue
        try:
            keys = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(keys, dict):
            continue
        for lookup_name in lookup_names:
            key = _normalize_key_value(keys.get(lookup_name))
            if key:
                return key

    tried_files = ", ".join(str(path) for path in candidates)
    raise ValueError(
        f"未找到 {provider} 的 API key（尝试过环境变量 {env_key}，以及文件: {tried_files}）"
    )


def maybe_strip_markdown_fence(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def call_llm(
    messages: list[dict[str, str]],
    *,
    provider: str = "deepseek",
    model: str | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.3,
    timeout: int = 60,
) -> dict[str, Any]:
    """调用单个 provider。"""
    return _call_llm_impl(
        messages,
        provider=provider,
        model=model,
        api_key_override=None,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )


def call_llm_race(
    messages: list[dict[str, str]],
    *,
    candidates: list[tuple[str, str]] | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.3,
    timeout: int = 45,
) -> dict[str, Any]:
    """并行调用多个 provider，返回最先成功的结果。"""
    resolved_candidates = candidates or list(DEFAULT_RACE_CANDIDATES)
    available = list_available_providers(resolved_candidates)
    if not available:
        raise RuntimeError("没有可用的 LLM provider（请检查 API key 配置）")

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(available)) as pool:
        futures = {
            pool.submit(
                _call_llm_impl,
                messages,
                provider=provider,
                model=model,
                api_key_override=None,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            ): (provider, model)
            for provider, model in available
        }
        for future in as_completed(futures):
            provider, model = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                errors.append(f"{provider}/{model}: {exc}")
                continue
            for pending in futures:
                pending.cancel()
            return result

    raise RuntimeError("所有 LLM 调用均失败: " + "; ".join(errors))


def list_available_providers(
    candidates: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """返回当前已配置 key 的 provider/model 列表。"""
    available: list[tuple[str, str]] = []
    for provider, model in (candidates or DEFAULT_RACE_CANDIDATES):
        try:
            _load_api_key(provider)
        except ValueError:
            continue
        available.append((provider, model))
    return available


def call_openai_compatible_result(
    provider_id: str,
    model_id: str | None,
    api_key: str,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    *,
    temperature: float | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """OpenAI-compatible 调用接口。"""
    if provider_id in {"", "auto"}:
        provider_id = "auto"
    if provider_id == "auto":
        result = call_llm_race(
            messages,
            max_tokens=max_tokens or 2000,
            temperature=0.3 if temperature is None else temperature,
            timeout=timeout,
        )
    else:
        result = _call_llm_impl(
            messages,
            provider=provider_id,
            model=model_id,
            api_key_override=(api_key or "").strip() or None,
            max_tokens=max_tokens or 2000,
            temperature=0.3 if temperature is None else temperature,
            timeout=timeout,
        )
    return {
        "provider": result["provider"],
        "model": result["model"],
        "content": maybe_strip_markdown_fence(result["content"]),
        "usage": result.get("usage") or {},
        "raw": result.get("raw") or {},
    }


def call_openai_compatible(
    provider_id: str,
    model_id: str | None,
    api_key: str,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    *,
    temperature: float | None = None,
    timeout: int = 60,
) -> str:
    return str(
        call_openai_compatible_result(
            provider_id,
            model_id,
            api_key,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )["content"]
    )


def _call_llm_impl(
    messages: list[dict[str, str]],
    *,
    provider: str,
    model: str | None,
    api_key_override: str | None,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    prov_cfg = PROVIDERS.get(provider)
    if not prov_cfg:
        raise ValueError(f"未知 provider: {provider}（可选: {', '.join(PROVIDERS.keys())}）")

    api_key = (api_key_override or "").strip() or _load_api_key(provider)
    resolved_model = model or prov_cfg["default_model"]
    url = f"{prov_cfg['base_url']}/chat/completions"
    payload = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"{provider}/{resolved_model} HTTP {exc.code}: {body[:500]}") from exc
    except Exception as exc:
        raise RuntimeError(f"{provider}/{resolved_model} 请求失败: {exc}") from exc

    choices = data.get("choices") or []
    content = ""
    if choices:
        message = choices[0].get("message") or {}
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            text_chunks = []
            for item in raw_content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"text", "output_text"} and item.get("text"):
                    text_chunks.append(str(item["text"]))
            content = "\n".join(text_chunks)

    return {
        "provider": provider,
        "model": resolved_model,
        "content": content,
        "usage": data.get("usage") or {},
        "raw": data,
    }
