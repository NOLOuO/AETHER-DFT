from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


KEYS_FILE_NAME = "api_keys.local.json"


def keys_file_path(app_root: Path) -> Path:
    return app_root / KEYS_FILE_NAME


def _normalize_key_value(value: Any) -> str:
    if isinstance(value, dict):
        for field in ("api_key", "key", "token", "value"):
            candidate = str(value.get(field) or "").strip()
            if candidate:
                return candidate
        return ""
    return str(value or "").strip()


def load_api_keys(app_root: Path) -> dict[str, str]:
    paths = [keys_file_path(app_root)]
    extra_paths = os.getenv("AETHER_DFT_API_KEYS_PATHS", "").strip()
    if extra_paths:
        paths.extend(Path(item.strip()) for item in extra_paths.split(";") if item.strip())
    path = next((candidate for candidate in paths if candidate.exists()), None)
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    normalized = {str(key): _normalize_key_value(value) for key, value in data.items()}
    return {key: value for key, value in normalized.items() if value}


def resolve_api_key(
    app_root: Path,
    *,
    aliases: Iterable[str] = (),
    env_names: Iterable[str] = (),
) -> str | None:
    api_keys = load_api_keys(app_root)
    for alias in aliases:
        value = str(api_keys.get(alias, "")).strip()
        if value:
            return value
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


def save_api_keys(app_root: Path, api_keys: dict[str, str]) -> None:
    path = keys_file_path(app_root)
    payload = {str(key): {"api_key": str(value).strip()} for key, value in api_keys.items() if str(value).strip()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
