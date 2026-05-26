from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from aether_dft.session_store import AetherSessionStore
from aether_dft.prompt_engine import render_compiled_system_prompt
from aether_dft.permissions import get_permission_mode, permission_mode_label, should_allow_tool

from .session import HarnessSessionStore
from .tool_registry import ToolRegistry

DISCUSSION_MAX_STEPS = 4
EXECUTION_MAX_STEPS = 15


def _runtime_log_path() -> Path:
    from aether_dft.paths import ensure_runtime_dir

    return ensure_runtime_dir("logs") / "harness-events.jsonl"


def log_event(event: str, payload: dict[str, Any]) -> Path:
    path = _runtime_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _sanitize_fragment(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "item"


def _clean_text(value: Any) -> str:
    return str(value or "").encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(messages, ensure_ascii=False, default=str).encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def _persist_tool_output(*, tool_name: str, tool_call_id: str, payload: Any) -> Path:
    from aether_dft.paths import ensure_runtime_dir

    outputs_dir = ensure_runtime_dir("tool_outputs")
    output_path = outputs_dir / f"{_sanitize_fragment(tool_call_id)}_{_sanitize_fragment(tool_name)}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output_path


def _render_tool_visible_result(*, tool_name: str, tool_call_id: str, payload: Any, limit: int = 12000) -> tuple[str, Path | None]:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if len(rendered) <= limit:
        return rendered, None
    output_path = _persist_tool_output(tool_name=tool_name, tool_call_id=tool_call_id, payload=payload)
    preview_limit = max(1200, limit - 800)
    preview = rendered[:preview_limit].rstrip()
    visible = json.dumps(
        {
            "status": getattr(payload, "get", lambda *_: None)("status") if isinstance(payload, dict) else None,
            "persisted_output_path": str(output_path),
            "preview": preview,
            "note": "full tool output persisted locally",
        },
        ensure_ascii=False,
        indent=2,
    )
    if len(visible) > limit:
        visible = visible[: limit - 20].rstrip() + "\n...[truncated]"
    return visible, output_path


def preflight(*, project: str | None = None) -> dict[str, Any]:
    from aether_dft import paths
    from aether_dft.prompt_engine import load_base_system_prompt, render_compiled_system_prompt

    base_prompt = load_base_system_prompt()
    compiled_prompt = render_compiled_system_prompt(project=project)
    checks = {
        "config/system_prompt.md": (paths.PROJECT_ROOT / "aether_dft" / "prompt_assets" / "system_chemistry.md").exists(),
        "dft_app": (paths.PROJECT_ROOT / "dft_app").exists(),
        "dft_shared": (paths.PROJECT_ROOT / "dft_shared").exists(),
        "智能体架构.md": (paths.PROJECT_ROOT / "智能体架构.md").exists(),
    }
    runtime = {
        "session_dir": str(paths.ensure_runtime_dir("sessions")),
        "context_dir": str(paths.ensure_runtime_dir("context")),
        "log_dir": str(paths.ensure_runtime_dir("logs")),
    }
    return {
        "checks": checks,
        "prompt": {
            "base_prompt": base_prompt,
            "base_prompt_length": len(base_prompt),
            "compiled_prompt": compiled_prompt,
            "compiled_prompt_length": len(compiled_prompt),
        },
        "runtime": runtime,
    }


def require_permission(action: str, *, destructive: bool = False) -> dict[str, Any]:
    mode = get_permission_mode()
    if destructive:
        allowed, reason = False, "destructive action always requires explicit user approval"
    else:
        allowed, reason = should_allow_tool(read_only=True, mode=mode)
    payload = {
        "action": action,
        "destructive": destructive,
        "permission_mode": mode,
        "permission_label": permission_mode_label(mode),
        "allowed": allowed,
        "reason": reason,
    }
    log_event("permission_check", payload)
    return payload


def infer_turn_mode(prompt: str) -> str:
    text = str(prompt or "").lower()
    execution_markers = [
        "提交",
        "集群",
        "slurm",
        "sbatch",
        "生成输入",
        "incar",
        "poscar",
        "建模",
        "构建",
        "建一个",
        "生成结构",
        "build",
        "run",
        "跑计算",
        "计算文件",
        "计算包",
        "开始计算",
        "同步",
        "sync",
        "fetch",
        "monitor",
        "vasp",
    ]
    return "execution" if any(marker in text for marker in execution_markers) else "discussion"


class AgentHarness:
    def __init__(
        self,
        *,
        adapter: Any,
        registry: ToolRegistry | None = None,
        sessions: Any | None = None,
        allow_cluster_submit: bool = False,
        permission_mode: str | None = None,
    ):
        self.adapter = adapter
        self.registry = registry or ToolRegistry(allow_cluster_submit=allow_cluster_submit, permission_mode=permission_mode)
        self.sessions = sessions or HarnessSessionStore()
        self.allow_cluster_submit = allow_cluster_submit

    def run_turn(
        self,
        prompt: str,
        *,
        project: str | None = None,
        session_id: str | None = None,
        max_tokens: int | None = None,
        max_steps: int | None = None,
        progress_callback: Any | None = None,
        permission_prompt_callback: Any | None = None,
    ) -> dict[str, Any]:
        session_store = self.sessions.store if hasattr(self.sessions, "store") else self.sessions
        interaction_mode = infer_turn_mode(prompt)
        if max_steps is None:
            max_steps = EXECUTION_MAX_STEPS if interaction_mode == "execution" else DISCUSSION_MAX_STEPS
        session_id = session_store.ensure_session(session_id=session_id, project=project, first_prompt=prompt)
        session_context = ""
        if session_id and hasattr(session_store, "build_session_context"):
            try:
                session_context = session_store.build_session_context(session_id)
            except Exception:
                session_context = ""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": render_compiled_system_prompt(project=project, session_context=session_context)},
            {"role": "user", "content": prompt},
        ]
        messages = _clean_messages(messages)
        tool_executions: list[dict[str, Any]] = []
        finish_reason = "stop"
        response = ""
        tools = self.registry.openai_tool_schemas()
        started_at = datetime.now().astimezone()
        if progress_callback:
            progress_callback({"event": "turn_start", "session_id": session_id, "model_id": getattr(getattr(self.adapter, "runtime", None), "model_id", "")})
        for step_index in range(max_steps):
            if progress_callback:
                progress_callback({"event": "model_request", "step": step_index + 1, "max_steps": max_steps})
            reply = self.adapter.chat(messages, tools=tools, tool_choice="auto", max_tokens=max_tokens)
            finish_reason = str(reply.get("finish_reason") or "stop")
            tool_calls = reply.get("tool_calls") or []
            content = str(reply.get("content") or "")
            if tool_calls:
                assistant_message: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": tool_calls}
                reasoning_content = str(reply.get("reasoning_content") or "").strip()
                if reasoning_content:
                    assistant_message["reasoning_content"] = reasoning_content
                messages.append(assistant_message)
                messages = _clean_messages(messages)
                for call in tool_calls:
                    func = call.get("function") or {}
                    name = str(func.get("name") or "")
                    raw_args = func.get("arguments") or "{}"
                    if progress_callback:
                        progress_callback({"event": "tool_start", "step": step_index + 1, "name": name, "arguments": raw_args})
                    result = self.registry.run_tool(name, raw_args)
                    if (
                        isinstance(result.get("result"), dict)
                        and result["result"].get("status") == "permission_required"
                        and permission_prompt_callback is not None
                    ):
                        permission_payload = dict(result["result"])
                        if progress_callback:
                            progress_callback(
                                {
                                    "event": "tool_permission_required",
                                    "step": step_index + 1,
                                    "name": name,
                                    "permission_mode": permission_payload.get("permission_mode"),
                                    "permission_label": permission_payload.get("permission_label"),
                                    "message": permission_payload.get("message"),
                                }
                            )
                        approved = bool(
                            permission_prompt_callback(
                                {
                                    "tool_name": name,
                                    "arguments": raw_args,
                                    "permission_mode": permission_payload.get("permission_mode"),
                                    "permission_label": permission_payload.get("permission_label"),
                                    "message": permission_payload.get("message"),
                                    "reason": permission_payload.get("reason"),
                                }
                            )
                        )
                        if approved:
                            rerun_arguments = dict(result.get("arguments") or {})
                            rerun_arguments["_permission_granted"] = True
                            result = self.registry.run_tool(name, rerun_arguments)
                            if progress_callback:
                                progress_callback(
                                    {
                                        "event": "tool_permission_granted",
                                        "step": step_index + 1,
                                        "name": name,
                                        "permission_mode": permission_payload.get("permission_mode"),
                                        "permission_label": permission_payload.get("permission_label"),
                                    }
                                )
                        elif progress_callback:
                            progress_callback(
                                {
                                    "event": "tool_permission_denied",
                                    "step": step_index + 1,
                                    "name": name,
                                    "permission_mode": permission_payload.get("permission_mode"),
                                    "permission_label": permission_payload.get("permission_label"),
                                }
                            )
                    persisted_output_path = None
                    visible, persisted_output_path = _render_tool_visible_result(
                        tool_name=name,
                        tool_call_id=str(call.get("id") or ""),
                        payload=result["result"],
                    )
                    result_record = dict(result)
                    if persisted_output_path is not None:
                        result_record["persisted_output_path"] = str(persisted_output_path)
                    tool_executions.append(result_record)
                    if progress_callback:
                        progress_callback(
                            {
                                "event": "tool_finish",
                                "step": step_index + 1,
                                "name": name,
                                "status": result.get("result", {}).get("status") if isinstance(result.get("result"), dict) else None,
                                "persisted_output_path": str(persisted_output_path) if persisted_output_path is not None else "",
                            }
                        )
                    messages.append({"role": "tool", "name": name, "tool_call_id": call.get("id"), "content": visible})
                    messages = _clean_messages(messages)
                continue
            response = _clean_text(content)
            messages.append({"role": "assistant", "content": response})
            messages = _clean_messages(messages)
            break
        else:
            finish_reason = "tool_loop_limit"

        record = {
            "project": project,
            "prompt": prompt,
            "response": response,
            "finish_reason": finish_reason,
            "interaction_mode": interaction_mode,
            "max_steps_used": max_steps,
            "tool_executions": tool_executions,
            "session_id": session_id,
            "model_id": getattr(getattr(self.adapter, "runtime", None), "model_id", ""),
            "started_at": started_at.isoformat(timespec="seconds"),
            "elapsed_seconds": round((datetime.now().astimezone() - started_at).total_seconds(), 3),
        }
        transcript_path = session_store.append_turn(session_id, record)
        record["record_path"] = str(transcript_path)
        return record
