from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .builtin_config import BUILTIN_ANALYSIS_MODELS, BUILTIN_SUMMARY_MODEL
from .key_store import KEYS_FILE_NAME, load_api_keys
from .llm_client import call_openai_compatible_result
from .provider_presets import build_provider_model_config, get_provider, list_models, list_provider_ids


class DomesticCopilotLLM:
    """项目内置的大模型调用适配层。"""

    def __init__(self, app_root: Path | None = None):
        self.app_root = app_root or self._default_app_root()

    def is_available(self) -> bool:
        if load_api_keys(self.app_root):
            return True
        for provider_id in list_provider_ids():
            provider = get_provider(provider_id)
            if os.getenv(str(provider["api_key_env"]), "").strip():
                return True
        return False

    def resolve_default_model(self) -> tuple[str, str]:
        provider_id = os.getenv("AETHER_DFT_LLM_PROVIDER", "").strip() or os.getenv("SEMI_DFT_LLM_PROVIDER", "").strip()
        model_id = os.getenv("AETHER_DFT_LLM_MODEL", "").strip() or os.getenv("SEMI_DFT_LLM_MODEL", "").strip()
        if provider_id and model_id:
            return provider_id, model_id

        preference_model = self._load_aether_preference_model()
        if preference_model:
            return preference_model

        summary_model = BUILTIN_SUMMARY_MODEL.strip()
        if summary_model:
            return self._find_provider_for_model(summary_model), summary_model

        fallback_model = (BUILTIN_ANALYSIS_MODELS[0] if BUILTIN_ANALYSIS_MODELS else "minimax-m2.7")
        return self._find_provider_for_model(fallback_model), fallback_model

    def call_messages(
        self,
        messages: list[dict[str, str]],
        *,
        provider_id: str | None = None,
        model_id: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        mode = self._resolve_call_mode()
        if mode == "external":
            return self._call_messages_external(
                messages,
                provider_id=provider_id,
                model_id=model_id,
                max_tokens=max_tokens,
            )
        return self.call_messages_inline(
            messages,
            provider_id=provider_id,
            model_id=model_id,
            max_tokens=max_tokens,
        )

    def call_messages_inline(
        self,
        messages: list[dict[str, Any]],
        *,
        provider_id: str | None = None,
        model_id: str | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream_callback: Any | None = None,
    ) -> dict[str, Any]:
        resolved_provider, resolved_model = (
            (provider_id, model_id)
            if provider_id and model_id
            else self.resolve_default_model()
        )
        api_keys = load_api_keys(self.app_root)
        api_key = str(api_keys.get(resolved_provider, "")).strip()
        result = call_openai_compatible_result(
            resolved_provider,
            resolved_model,
            api_key,
            messages,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stream_callback=stream_callback,
        )
        result["provider_id"] = resolved_provider
        result["model_id"] = resolved_model
        return result

    def call_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        provider_id: str | None = None,
        model_id: str | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        stream_callback: Any | None = None,
    ) -> dict[str, Any]:
        return self.call_messages_inline(
            messages,
            provider_id=provider_id,
            model_id=model_id,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stream_callback=stream_callback,
        )

    def _call_messages_external(
        self,
        messages: list[dict[str, str]],
        *,
        provider_id: str | None = None,
        model_id: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        python_executable = os.getenv("SEMI_DFT_LLM_EXTERNAL_PYTHON", sys.executable).strip()
        if not python_executable:
            raise RuntimeError("未配置可用的外部 Python 解释器")

        command = [
            python_executable,
            "-m",
            "dft_app.llm.external_call_helper",
        ]
        payload = {
            "app_root": str(self.app_root),
            "messages": messages,
            "provider_id": provider_id,
            "model_id": model_id,
            "max_tokens": max_tokens,
        }
        child_env = os.environ.copy()
        child_env["SEMI_DFT_LLM_CALL_MODE"] = "inline"
        child_env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            env=child_env,
            encoding="utf-8",
            errors="replace",
            timeout=int(os.getenv("SEMI_DFT_LLM_EXTERNAL_TIMEOUT_SECONDS", "240")),
            check=False,
        )
        if process.returncode != 0:
            stderr = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"外部 LLM helper 调用失败: {stderr or '未知错误'}")

        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("外部 LLM helper 返回了非法 JSON") from exc
        if not isinstance(result, dict):
            raise RuntimeError("外部 LLM helper 返回结构不合法")
        return result

    def _load_aether_preference_model(self) -> tuple[str, str] | None:
        preference_path = self.app_root / "runtime" / "model-preferences.json"
        if not preference_path.exists():
            return None
        try:
            payload = json.loads(preference_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        model_id = str(payload.get("global_default_model_id") or "").strip()
        if ":" not in model_id:
            return None
        provider_id, model_name = model_id.split(":", 1)
        if provider_id and model_name:
            return provider_id, model_name
        return None

    @staticmethod
    def _resolve_call_mode() -> str:
        mode = os.getenv("SEMI_DFT_LLM_CALL_MODE", "external").strip().lower()
        if mode not in {"external", "inline"}:
            return "external"
        return mode

    @staticmethod
    def _default_app_root() -> Path:
        env_path = os.getenv("SEMI_DFT_DOMESTIC_APP_ROOT")
        if env_path:
            return Path(env_path)
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _find_provider_for_model(model_id: str) -> str:
        for provider_id in list_provider_ids():
            for item in list_models(provider_id):
                if item.get("id") == model_id:
                    return provider_id
        raise KeyError(f"在项目内置 provider 列表中未找到模型: {model_id}")

    def describe_runtime(self) -> dict[str, Any]:
        local_key_file = self.app_root / KEYS_FILE_NAME
        api_keys = load_api_keys(self.app_root)
        return {
            "app_root": str(self.app_root),
            "call_mode": self._resolve_call_mode(),
            "default_model": {
                "provider": self.resolve_default_model()[0],
                "model": self.resolve_default_model()[1],
            },
            "configured_providers": sorted(api_keys.keys()),
            "local_key_file": str(local_key_file),
            "local_key_file_exists": local_key_file.exists(),
            "env_key_overrides": [
                get_provider(provider_id)["api_key_env"]
                for provider_id in list_provider_ids()
                if os.getenv(str(get_provider(provider_id)["api_key_env"]), "").strip()
            ],
            "analysis_models": list(BUILTIN_ANALYSIS_MODELS),
            "summary_model": BUILTIN_SUMMARY_MODEL,
        }
