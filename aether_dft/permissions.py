from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
from typing import Any

from .paths import RUNTIME_DIR

DEFAULT_PERMISSION_MODE = "dev"
PERMISSIONS_PATH = RUNTIME_DIR / "permissions.json"
_MODEL_PERMISSION_PREFIXES = ("_permission_", "_approval_")


def strip_model_permission_fields(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Copy tool arguments while discarding model-controlled approval claims."""
    return {
        str(key): value
        for key, value in dict(arguments or {}).items()
        if not str(key).startswith(_MODEL_PERMISSION_PREFIXES)
    }


def approval_scope_digest(tool_name: str, arguments: dict[str, Any] | None) -> str:
    """Return a stable digest binding an approval to one tool call payload."""
    envelope = {
        "tool_name": str(tool_name),
        "arguments": strip_model_permission_fields(arguments),
    }
    canonical = json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class OneTimeApprovalStore:
    """In-memory, single-use approvals that are unforgeable by model arguments."""

    def __init__(self) -> None:
        self._pending: dict[str, str] = {}
        self._lock = threading.Lock()

    def issue(self, tool_name: str, arguments: dict[str, Any] | None) -> str:
        token = secrets.token_urlsafe(32)
        digest = approval_scope_digest(tool_name, arguments)
        with self._lock:
            self._pending[token] = digest
        return token

    def consume(
        self,
        token: str | None,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        if not token:
            return False, "approval_missing"
        with self._lock:
            expected_digest = self._pending.pop(str(token), None)
        if expected_digest is None:
            return False, "approval_invalid_or_replayed"
        actual_digest = approval_scope_digest(tool_name, arguments)
        if not hmac.compare_digest(expected_digest, actual_digest):
            return False, "approval_scope_mismatch"
        return True, "one_time_approval_consumed"


def normalize_permission_mode(value: str | None) -> str:
    raw = " ".join(str(value or "").strip().lower().split())
    aliases = {
        "": DEFAULT_PERMISSION_MODE,
        "dev": "dev",
        "full": "dev",
        "auto": "dev",
        "autonomous": "dev",
        "开发": "dev",
        "完全开发": "dev",
        "ask": "ask",
        "confirm": "ask",
        "approval": "ask",
        "safe": "ask",
        "需要同意": "ask",
        "需要用户同意": "ask",
    }
    if raw not in aliases:
        raise ValueError("permission mode 必须是 dev/完全开发 或 ask/需要用户同意")
    return aliases[raw]


def _load_payload() -> dict[str, Any]:
    if not PERMISSIONS_PATH.exists():
        return {"mode": DEFAULT_PERMISSION_MODE}
    try:
        data = json.loads(PERMISSIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"mode": DEFAULT_PERMISSION_MODE}
    return data if isinstance(data, dict) else {"mode": DEFAULT_PERMISSION_MODE}


def get_permission_mode() -> str:
    try:
        return normalize_permission_mode(str(_load_payload().get("mode") or DEFAULT_PERMISSION_MODE))
    except ValueError:
        return DEFAULT_PERMISSION_MODE


def set_permission_mode(mode: str) -> dict[str, Any]:
    normalized = normalize_permission_mode(mode)
    payload = {"mode": normalized, "label": permission_mode_label(normalized)}
    PERMISSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERMISSIONS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def permission_mode_label(mode: str | None = None) -> str:
    normalized = normalize_permission_mode(mode or get_permission_mode())
    if normalized == "ask":
        return "需要用户同意"
    return "完全开发"


def permission_policy_text(mode: str | None = None) -> str:
    normalized = normalize_permission_mode(mode or get_permission_mode())
    if normalized == "ask":
        return (
            "Permission mode: ask / 需要用户同意。读取、分析、规划类动作可直接做；"
            "写文件、提交作业、修改项目状态、外部副作用动作前必须先向用户确认。"
        )
    return (
        "Permission mode: dev / 完全开发。清晰、低风险、可逆的读取/写入/整理动作直接推进；"
        "只有删除、覆盖、git reset、真实提交集群作业、安装/卸载包等破坏性或高副作用动作才需要确认。"
    )


def should_allow_tool(*, read_only: bool, mode: str | None = None, explicit_permission: bool = False) -> tuple[bool, str]:
    normalized = normalize_permission_mode(mode or get_permission_mode())
    if read_only:
        return True, "read-only tool allowed"
    if explicit_permission:
        return True, "explicit permission granted"
    if normalized == "dev":
        return True, "dev mode allows non-destructive tool execution"
    return False, "ask mode requires user approval before non-read-only tool execution"
