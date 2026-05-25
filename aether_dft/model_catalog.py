from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from dft_app.llm.key_store import load_api_keys
from dft_app.llm.provider_presets import build_provider_model_config, get_provider, list_models, list_provider_ids

from .paths import PROJECT_ROOT, RUNTIME_DIR

DEFAULT_MODEL_ID = "deepseek:deepseek-v4-pro"
PREFERENCES_PATH = RUNTIME_DIR / "model-preferences.json"


@dataclass(frozen=True)
class ModelDescriptor:
    model_id: str
    provider_id: str
    model_name: str
    api_model: str
    display_name: str
    base_url: str
    api_key_env: str
    available: bool
    default: bool = False
    source: str = "builtin"
    note: str = ""
    context_window: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_model_id(provider_id: str, model_name: str) -> str:
    return f"{provider_id}:{model_name}"


def split_model_id(model_id: str) -> tuple[str, str]:
    if ":" not in model_id:
        raise ValueError("模型 ID 必须形如 provider:model，例如 deepseek:deepseek-v4-pro")
    provider_id, model_name = model_id.split(":", 1)
    if not provider_id or not model_name:
        raise ValueError("模型 ID 必须形如 provider:model，例如 deepseek:deepseek-v4-pro")
    return provider_id, model_name


def _provider_has_key(provider_id: str, api_key_env: str, app_root: Path = PROJECT_ROOT) -> bool:
    if os.getenv(api_key_env, "").strip():
        return True
    return bool(str(load_api_keys(app_root).get(provider_id, "")).strip())


def load_model_catalog(app_root: Path = PROJECT_ROOT) -> dict[str, ModelDescriptor]:
    catalog: dict[str, ModelDescriptor] = {}
    for provider_id in list_provider_ids():
        provider_config = get_provider(provider_id)
        provider_models = list_models(provider_id)
        if provider_models:
            for item in provider_models:
                model_name = str(item.get("id") or item.get("api_model"))
                if not model_name:
                    continue
                config = build_provider_model_config(provider_id, model_name)
                model_id = canonical_model_id(provider_id, model_name)
                catalog[model_id] = ModelDescriptor(
                    model_id=model_id,
                    provider_id=provider_id,
                    model_name=model_name,
                    api_model=str(config["model"]),
                    display_name=f"{config.get('label', provider_id)} / {config.get('model_label', model_name)}",
                    base_url=str(config["base_url"]),
                    api_key_env=str(config["api_key_env"]),
                    available=_provider_has_key(provider_id, str(config["api_key_env"]), app_root),
                    default=model_id == DEFAULT_MODEL_ID,
                    note=str(config.get("model_note", "")),
                    context_window=int(config["context_window"]) if config.get("context_window") else None,
                )
        else:
            # Provider with no fixed model list (e.g. OpenAI/custom-compatible) still appears as custom-capable.
            base_url_env = str(provider_config.get("base_url_env", "") or "").strip()
            base_url = os.getenv(base_url_env, "").strip() if base_url_env else ""
            if not base_url:
                base_url = str(provider_config.get("base_url", "") or "")
            catalog[canonical_model_id(provider_id, "<custom>")] = ModelDescriptor(
                model_id=canonical_model_id(provider_id, "<custom>"),
                provider_id=provider_id,
                model_name="<custom>",
                api_model="<custom>",
                display_name=f"{provider_config.get('label', provider_id)} / <custom>",
                base_url=base_url,
                api_key_env=str(provider_config["api_key_env"]),
                available=_provider_has_key(provider_id, str(provider_config["api_key_env"]), app_root),
                default=False,
                source="custom-provider",
                note="此 provider 允许传入任意 OpenAI-compatible model name。",
            )
    return catalog


def _load_preferences() -> dict[str, Any]:
    if not PREFERENCES_PATH.exists():
        return {"global_default_model_id": DEFAULT_MODEL_ID}
    try:
        data = json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"global_default_model_id": DEFAULT_MODEL_ID}
    if not isinstance(data, dict):
        return {"global_default_model_id": DEFAULT_MODEL_ID}
    preferred = str(data.get("global_default_model_id") or DEFAULT_MODEL_ID)
    try:
        provider_id, model_name = split_model_id(preferred)
        build_provider_model_config(provider_id, model_name)
    except Exception:
        preferred = DEFAULT_MODEL_ID
    data["global_default_model_id"] = preferred
    return data


def save_preferences(preferences: dict[str, Any]) -> None:
    PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFERENCES_PATH.write_text(json.dumps(preferences, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_effective_model_id() -> str:
    env_provider = os.getenv("AETHER_DFT_LLM_PROVIDER", "").strip() or os.getenv("SEMI_DFT_LLM_PROVIDER", "").strip()
    env_model = os.getenv("AETHER_DFT_LLM_MODEL", "").strip() or os.getenv("SEMI_DFT_LLM_MODEL", "").strip()
    if env_provider and env_model:
        return canonical_model_id(env_provider, env_model)
    preferred = str(_load_preferences().get("global_default_model_id") or DEFAULT_MODEL_ID)
    return preferred or DEFAULT_MODEL_ID


def resolve_effective_provider_model() -> tuple[str, str]:
    return split_model_id(resolve_effective_model_id())


def set_default_model(model_id: str) -> dict[str, Any]:
    provider_id, model_name = split_model_id(model_id)
    build_provider_model_config(provider_id, model_name)
    preferences = _load_preferences()
    preferences["global_default_model_id"] = model_id
    save_preferences(preferences)
    return preferences


def format_model_table(catalog: dict[str, ModelDescriptor] | None = None, current: str | None = None) -> str:
    catalog = catalog or load_model_catalog()
    current = current or resolve_effective_model_id()
    lines = ["Available OpenAI-compatible models:"]
    for model_id in sorted(catalog):
        item = catalog[model_id]
        marker = "*" if model_id == current else " "
        status = "available" if item.available else f"missing-key:{item.api_key_env}"
        default = " default" if item.default else ""
        lines.append(f"{marker} {model_id:<28} [{status}{default}] {item.display_name}")
        if item.context_window:
            lines[-1] += f" ctx={item.context_window:,}"
    if current not in catalog:
        provider_id, model_name = split_model_id(current)
        config = build_provider_model_config(provider_id, model_name)
        status = "available" if _provider_has_key(provider_id, str(config["api_key_env"])) else f"missing-key:{config['api_key_env']}"
        lines.append(f"* {current:<28} [{status} custom] {config.get('label', provider_id)} / {model_name}")
        if config.get("context_window"):
            lines[-1] += f" ctx={int(config['context_window']):,}"
    return "\n".join(lines)
