from __future__ import annotations

import hashlib
import html
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from aether_dft.context_budget import usable_context_chars
from aether_dft.session_store import AetherSessionStore
from aether_dft.prompt_engine import render_compiled_system_prompt
from aether_dft.permissions import get_permission_mode, permission_mode_label, should_allow_tool

from .session import HarnessSessionStore
from .tool_registry import ToolRegistry

DISCUSSION_MAX_STEPS = 8
EXECUTION_MAX_STEPS = 15
MAX_TOOL_CALLS_PER_STEP = 8
MAX_MUTATING_TOOL_CALLS_PER_STEP = 3
TOOL_HEARTBEAT_SECONDS = 1.5
TOOL_VISIBLE_RESULT_LIMIT = 8_000
TOKEN_GUARD_USAGE_RATIO = 0.88
TOKEN_GUARD_MIN_STEPS = 2
_TEXT_TOOL_CALL_MARKERS = (
    "<｜｜DSML｜｜tool_calls",
    "<ï½œï½œDSMLï½œï½œtool_calls",
    "<|tool_calls|",
)


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


def _contains_tool_markup(value: str) -> bool:
    text = str(value or "")
    return any(
        marker in text
        for marker in (
            "<｜｜DSML｜｜tool_calls>",
            "<ï½œï½œDSMLï½œï½œtool_calls>",
            "<|tool_calls|>",
            "<tool_call",
            "<invoke name=",
            "<ï½œï½œDSMLï½œï½œinvoke name=",
            "</invoke>",
        )
    )


def _parse_scalar_tool_value(value: str, *, force_string: bool) -> Any:
    text = html.unescape(str(value or "")).strip()
    if force_string:
        return text
    if not text:
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _extract_text_tool_calls(value: str) -> list[dict[str, Any]]:
    """Parse DSML-like text tool calls emitted by some OpenAI-compatible models.

    DeepSeek/Qwen-compatible backends occasionally serialize tool calls into
    assistant content instead of the structured ``tool_calls`` field.  Treating
    that content as a final answer makes the agent look stuck even though the
    model selected the right next tools.  This parser is intentionally narrow:
    it only accepts explicit ``invoke name=...`` blocks with parameter tags and
    converts them back into ordinary OpenAI-style tool calls.
    """

    text = str(value or "")
    if "invoke name=" not in text:
        return []
    if not any(marker in text for marker in _TEXT_TOOL_CALL_MARKERS):
        return []
    calls: list[dict[str, Any]] = []
    invoke_re = re.compile(r"<[^<>\n]*invoke\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</[^<>\n]*invoke>", re.DOTALL)
    param_re = re.compile(
        r"<[^<>\n]*parameter\s+name=[\"']([^\"']+)[\"'](?:\s+string=[\"']([^\"']+)[\"'])?[^>]*>(.*?)</[^<>\n]*parameter>",
        re.DOTALL,
    )
    for index, match in enumerate(invoke_re.finditer(text)):
        name = html.unescape(match.group(1)).strip()
        if not name:
            continue
        body = match.group(2)
        args: dict[str, Any] = {}
        for param in param_re.finditer(body):
            key = html.unescape(param.group(1)).strip()
            if not key:
                continue
            force_string = str(param.group(2) or "").lower() == "true"
            args[key] = _parse_scalar_tool_value(param.group(3), force_string=force_string)
        digest = hashlib.sha1(f"{name}:{index}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}".encode("utf-8")).hexdigest()[:10]
        calls.append(
            {
                "id": f"text_tool_call_{digest}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    return calls


def _tool_markup_fallback() -> str:
    return (
        "本轮已经拿到工具证据，但模型在最终回复阶段仍尝试输出工具调用标记。"
        "为避免把未执行的工具调用误当成结果，我已拦截这些标记；请继续追问，"
        "我会基于已保存的 trace 用自然语言总结证据和下一步。"
    )


def _tool_evidence_fallback(tool_executions: list[dict[str, Any]], *, reason: str = "") -> str:
    """Return a deterministic natural-language summary when final LLM text is unusable.

    This is a safety net, not a fixed workflow: the model already chose and ran
    tools.  We only translate those completed tool results into a readable
    answer when the final response contains unexecuted tool-call markup.
    """

    if not tool_executions:
        return _tool_markup_fallback()
    lines: list[str] = []
    intro = "模型最终回复含未执行工具标记，我已拦截；下面只基于本轮已经执行过的工具证据总结。"
    if reason:
        intro += f" 触发原因：{reason}。"
    lines.append(intro)
    lines.append("")
    lines.append("已执行工具证据：")
    seen_counts: dict[str, int] = {}
    for item in tool_executions[-12:]:
        name = str(item.get("name") or "tool")
        seen_counts[name] = seen_counts.get(name, 0) + 1
        label = name if seen_counts[name] == 1 else f"{name}#{seen_counts[name]}"
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        status = str(result.get("status") or "unknown")
        parts = [f"- {label}: {status}"]
        message = str(result.get("message") or "").strip()
        if message:
            parts.append(message[:180])
        if name == "cluster_my_jobs" and "count" in result:
            parts.append(f"队列 job 数 {result.get('count')}")
            jobs = result.get("jobs") if isinstance(result.get("jobs"), list) else []
            running = [job for job in jobs if str(job.get("scheduler_state") or "").upper() == "RUNNING"]
            if running:
                parts.append(
                    "RUNNING: "
                    + ", ".join(
                        f"{job.get('job_id')}:{job.get('name')}@{job.get('node')}"
                        for job in running[:4]
                    )
                )
        if name == "cluster_probe" and isinstance(result.get("details"), dict):
            probe = result["details"].get("probe") or {}
            if isinstance(probe, dict):
                parts.append(
                    "probe="
                    + ", ".join(
                        f"{key}:{probe.get(key)}"
                        for key in ("hostname", "sbatch", "squeue")
                        if probe.get(key)
                    )
                )
        if "remote_run_root" in result:
            parts.append(f"remote={result.get('remote_run_root')}")
        if "last_toten_ev" in result:
            parts.append(f"TOTEN={result.get('last_toten_ev')} eV")
        if "last_free_energy_ev" in result and result.get("last_free_energy_ev") != result.get("last_toten_ev"):
            parts.append(f"F={result.get('last_free_energy_ev')} eV")
        if "max_force_ev_a" in result:
            parts.append(f"max_force={result.get('max_force_ev_a')} eV/Å")
        if "rms_force_ev_a" in result:
            parts.append(f"rms_force={result.get('rms_force_ev_a')} eV/Å")
        if "accuracy_reached" in result:
            parts.append(f"SCF_accuracy={result.get('accuracy_reached')}")
        if "ionic_steps_seen" in result:
            parts.append(f"ionic_steps={result.get('ionic_steps_seen')}")
        if "convergence_score" in result:
            parts.append(f"convergence_score={result.get('convergence_score')}")
        if "oscillating" in result:
            parts.append(f"oscillating={result.get('oscillating')}")
        if "persisted_output_path" in result:
            parts.append(f"full={result.get('persisted_output_path')}")
        if "category" in result:
            parts.append(f"category={result.get('category')}")
        if isinstance(result.get("tool_names"), list):
            parts.append("tools=" + ", ".join(str(tool) for tool in result["tool_names"][:8]))
        lines.append("；".join(str(part) for part in parts if str(part).strip()) + "。")
    lines.append("")
    lines.append("结论：本轮没有执行新的未确认工具调用；如果上面有 unavailable/error，下一步应补齐缺失路径或改用已验证的 remote_run_root/job_id 后再读。")
    return "\n".join(lines)


def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(messages, ensure_ascii=False, default=str).encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def _persist_tool_output(*, tool_name: str, tool_call_id: str, payload: Any) -> Path:
    from aether_dft.paths import ensure_runtime_dir

    outputs_dir = ensure_runtime_dir("tool_outputs")
    output_path = outputs_dir / f"{_sanitize_fragment(tool_call_id)}_{_sanitize_fragment(tool_name)}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output_path


def _tool_artifact_meta(*, tool_name: str, tool_call_id: str, output_path: Path | None) -> dict[str, Any]:
    if output_path is None:
        return {}
    meta: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "persisted_output_path": str(output_path),
    }
    try:
        raw = output_path.read_bytes()
        meta["persisted_output_sha256"] = hashlib.sha256(raw).hexdigest()
        meta["persisted_output_bytes"] = len(raw)
    except Exception:
        pass
    return meta


def _microcompact_tool_result(payload: Any, *, output_path: Path | None = None, tool_name: str = "", tool_call_id: str = "", preview_limit: int = 900) -> Any:
    if not isinstance(payload, dict):
        text = _clean_text(payload)
        if len(text) <= preview_limit:
            return payload
        compacted: dict[str, Any] = {
            "status": None,
            "microcompacted": True,
            "preview": text[:preview_limit].rstrip(),
            "note": "large scalar tool result compacted; full output persisted when path is present",
        }
        compacted.update(_tool_artifact_meta(tool_name=tool_name, tool_call_id=tool_call_id, output_path=output_path))
        return compacted

    compact: dict[str, Any] = {
        "status": payload.get("status"),
        "microcompacted": True,
    }
    for key in (
        "message",
        "project",
        "run_id",
        "remote_run_root",
        "checkpoint_path",
        "learning_path",
        "progress_path",
        "state_path",
        "persisted_output_path",
        "guidance",
        "verdict",
        "last_toten_ev",
        "last_free_energy_ev",
        "max_force_ev_a",
        "rms_force_ev_a",
        "accuracy_reached",
        "ionic_steps_seen",
        "last_energy_ev",
    ):
        if key in payload:
            compact[key] = payload.get(key)
    compact.update(_tool_artifact_meta(tool_name=tool_name, tool_call_id=tool_call_id, output_path=output_path))

    rendered = json.dumps(payload, ensure_ascii=False, default=str)
    compact["preview"] = rendered[:preview_limit].rstrip()
    compact["note"] = "full tool output is persisted outside prompt context"
    return compact


def _messages_char_count(messages: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(message, ensure_ascii=False, default=str)) for message in messages)


def _token_guard_status(messages: list[dict[str, Any]], *, model_id: str | None = None) -> dict[str, Any]:
    try:
        budget = usable_context_chars(model_id)
    except Exception:
        budget = usable_context_chars(None)
    used = _messages_char_count(messages)
    ratio = used / budget if budget else 0.0
    return {
        "used_chars": used,
        "budget_chars": budget,
        "usage_ratio": ratio,
        "should_finalize": ratio >= TOKEN_GUARD_USAGE_RATIO,
    }


def _render_tool_visible_result(*, tool_name: str, tool_call_id: str, payload: Any, limit: int = TOOL_VISIBLE_RESULT_LIMIT) -> tuple[str, Path | None]:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if len(rendered) <= limit:
        return rendered, None
    output_path = _persist_tool_output(tool_name=tool_name, tool_call_id=tool_call_id, payload=payload)
    preview_limit = max(900, limit // 5)
    preview = rendered[:preview_limit].rstrip()
    visible = json.dumps(
        {
            "status": getattr(payload, "get", lambda *_: None)("status") if isinstance(payload, dict) else None,
            "microcompacted": True,
            "persisted_output_path": str(output_path),
            "preview": preview,
            "note": "full tool output persisted locally; prompt context receives only this compact preview",
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


def _run_registry_tool_with_heartbeat(
    registry: ToolRegistry,
    name: str,
    raw_args: Any,
    *,
    step: int,
    progress_callback: Any | None,
) -> dict[str, Any]:
    if progress_callback is None:
        return registry.run_tool(name, raw_args)

    stop = threading.Event()

    def heartbeat() -> None:
        started = time.perf_counter()
        tick = 0
        while not stop.wait(TOOL_HEARTBEAT_SECONDS):
            tick += 1
            progress_callback(
                {
                    "event": "tool_progress",
                    "step": step,
                    "name": name,
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                    "tick": tick,
                    "message": "工具仍在运行；这不是卡死，正在等待外部 I/O 或计算返回。",
                }
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return registry.run_tool(name, raw_args)
    finally:
        stop.set()
        thread.join(timeout=0.2)


def infer_turn_mode(prompt: str) -> str:
    """Return the tool-schema exposure mode for a turn.

    Natural-language prompts are intentionally not keyword-routed here. The
    model receives the lean discussion surface plus capability discovery, then
    decides which concrete tools to unlock. Execution mode is reserved for
    explicit machine-readable overrides from CLI/tests/advanced callers.
    """

    text = str(prompt or "").lower()
    if any(tag in text for tag in ("[execution-mode]", "[execution]", "<execution-mode>")):
        return "execution"
    if any(tag in text for tag in ("[discussion-mode]", "[discussion]", "<discussion-mode>")):
        return "discussion"
    return "discussion"



class AgentHarness:
    def __init__(
        self,
        *,
        adapter: Any,
        registry: ToolRegistry | None = None,
        sessions: Any | None = None,
        allow_cluster_submit: bool = False,
        permission_mode: str | None = None,
        human_question_handler: Any | None = None,
    ):
        self.adapter = adapter
        self.registry = registry or ToolRegistry(
            allow_cluster_submit=allow_cluster_submit,
            permission_mode=permission_mode,
            human_question_handler=human_question_handler,
        )
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
        stream_callback: Any | None = None,
    ) -> dict[str, Any]:
        session_store = self.sessions.store if hasattr(self.sessions, "store") else self.sessions
        if hasattr(self.registry, "default_project"):
            self.registry.default_project = project
        interaction_mode = infer_turn_mode(prompt)
        if max_steps is None:
            max_steps = EXECUTION_MAX_STEPS if interaction_mode == "execution" else DISCUSSION_MAX_STEPS
        session_id = session_store.ensure_session(session_id=session_id, project=project, first_prompt=prompt)
        if session_id and hasattr(session_store, "compact_if_needed"):
            try:
                compact_result = session_store.compact_if_needed(session_id, reason="run_turn_preflight")
                if progress_callback and compact_result.get("status") == "ok":
                    progress_callback(
                        {
                            "event": "session_auto_compacted",
                            "session_id": session_id,
                            "compacted_turn_count": compact_result.get("compacted_turn_count"),
                            "compact_summary_chars": compact_result.get("compact_summary_chars"),
                            "approx_chars_before": compact_result.get("approx_chars_before"),
                            "auto_compact_threshold_chars": (compact_result.get("context_budget") or {}).get("auto_compact_chars"),
                        }
                    )
            except Exception as exc:
                log_event(
                    "session_auto_compact_failed",
                    {"session_id": session_id, "error": str(exc), "project": project},
                )
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
        discovered_tool_names: set[str] = set()

        def refresh_tool_schemas() -> list[dict[str, Any]]:
            try:
                return self.registry.openai_tool_schemas(interaction_mode=interaction_mode, include_tool_names=sorted(discovered_tool_names))
            except TypeError:
                try:
                    return self.registry.openai_tool_schemas(interaction_mode=interaction_mode)
                except TypeError:
                    return self.registry.openai_tool_schemas()

        tools = refresh_tool_schemas()
        started_at = datetime.now().astimezone()
        force_final_reply_after_audit = False
        force_final_reply_message = ""

        def append_tool_result(call: dict[str, Any], name: str, result: dict[str, Any], step_number: int) -> dict[str, Any]:
            persisted_output_path = None
            visible, persisted_output_path = _render_tool_visible_result(
                tool_name=name,
                tool_call_id=str(call.get("id") or ""),
                payload=result["result"],
            )
            result_record = dict(result)
            if persisted_output_path is not None:
                result_record["persisted_output_path"] = str(persisted_output_path)
                result_record["result"] = _microcompact_tool_result(
                    result["result"],
                    output_path=persisted_output_path,
                    tool_name=name,
                    tool_call_id=str(call.get("id") or ""),
                )
            tool_executions.append(result_record)
            if progress_callback:
                progress_callback(
                    {
                        "event": "tool_finish",
                        "step": step_number,
                        "name": name,
                        "status": result.get("result", {}).get("status") if isinstance(result.get("result"), dict) else None,
                        "persisted_output_path": str(persisted_output_path) if persisted_output_path is not None else "",
                        "microcompacted": persisted_output_path is not None,
                    }
                )
            messages.append({"role": "tool", "name": name, "tool_call_id": call.get("id"), "content": visible})
            return result_record

        if progress_callback:
            progress_callback({"event": "turn_start", "session_id": session_id, "model_id": getattr(getattr(self.adapter, "runtime", None), "model_id", "")})
        try:
            for step_index in range(max_steps):
                if progress_callback:
                    progress_callback({"event": "model_request", "step": step_index + 1, "max_steps": max_steps})
                tools_for_step = [] if force_final_reply_after_audit else tools
                tool_choice_for_step = "none" if force_final_reply_after_audit else "auto"
                if force_final_reply_after_audit:
                    messages.append(
                        {
                            "role": "system",
                            "content": force_final_reply_message or (
                                "behavior_audit 已完成。现在必须给用户一个简短、证据化的自然语言结论；"
                                "不要再调用工具。若仍有后续动作，只把它列为下一步。"
                            ),
                        }
                    )
                    messages = _clean_messages(messages)
                chat_kwargs: dict[str, Any] = {
                    "tools": tools_for_step,
                    "tool_choice": tool_choice_for_step,
                    "max_tokens": max_tokens,
                }
                if stream_callback is not None:
                    chat_kwargs["stream_callback"] = stream_callback
                try:
                    reply = self.adapter.chat(messages, **chat_kwargs)
                except TypeError as exc:
                    if "stream_callback" not in chat_kwargs or "stream_callback" not in str(exc):
                        raise
                    chat_kwargs.pop("stream_callback", None)
                    reply = self.adapter.chat(messages, **chat_kwargs)
                finish_reason = str(reply.get("finish_reason") or "stop")
                tool_calls = reply.get("tool_calls") or []
                content = str(reply.get("content") or "")
                if not tool_calls:
                    text_tool_calls = _extract_text_tool_calls(content)
                    if text_tool_calls:
                        tool_calls = text_tool_calls
                        finish_reason = "text_tool_calls"
                        content = re.split(r"<[^<>\n]*tool_calls[^>]*>", content, maxsplit=1)[0].strip()
                        log_event(
                            "text_tool_calls_parsed",
                            {
                                "session_id": session_id,
                                "step": step_index + 1,
                                "tool_names": [str((call.get("function") or {}).get("name") or "") for call in tool_calls],
                            },
                        )
                if tool_calls:
                    assistant_message: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": tool_calls}
                    reasoning_content = str(reply.get("reasoning_content") or "").strip()
                    if reasoning_content:
                        assistant_message["reasoning_content"] = reasoning_content
                    messages.append(assistant_message)
                    messages = _clean_messages(messages)
                    read_only_checker = getattr(self.registry, "is_read_only_tool", lambda _name: True)
                    parallel_safe_checker = getattr(self.registry, "is_parallel_safe_tool", lambda _name: True)
                    parsed_calls: list[tuple[int, dict[str, Any], str, Any, bool]] = []
                    for call_index, call in enumerate(tool_calls):
                        func = call.get("function") or {}
                        name = str(func.get("name") or "")
                        raw_args = func.get("arguments") or "{}"
                        parsed_calls.append((call_index, call, name, raw_args, bool(read_only_checker(name))))
                    executable_count = min(len(parsed_calls), MAX_TOOL_CALLS_PER_STEP)
                    human_question_indexes = [
                        idx for idx, item in enumerate(parsed_calls[:executable_count]) if item[2] == "auto_human_question"
                    ]
                    if human_question_indexes:
                        step_number = step_index + 1
                        first_question_index = human_question_indexes[0]
                        for parsed_index, (call_index, call, name, raw_args, _read_only) in enumerate(parsed_calls[:executable_count]):
                            if parsed_index == first_question_index:
                                if progress_callback:
                                    progress_callback({"event": "tool_start", "step": step_number, "name": name, "arguments": raw_args})
                                result = _run_registry_tool_with_heartbeat(
                                    self.registry,
                                    name,
                                    raw_args,
                                    step=step_number,
                                    progress_callback=progress_callback,
                                )
                            else:
                                result = {
                                    "name": name,
                                    "arguments": raw_args,
                                    "result": {
                                        "status": "deferred_until_human_answer",
                                        "message": (
                                            "同一轮包含 auto_human_question；harness 已优先提问并延后其他工具，"
                                            "避免在关键人类判断前执行写入/提交/并行查询。请等待答案后在下一轮继续。"
                                        ),
                                    },
                                }
                            append_tool_result(call, name, result, step_number)
                        for call_index, call, name, raw_args, _read_only in parsed_calls[executable_count:]:
                            append_tool_result(
                                call,
                                name,
                                {
                                    "name": name,
                                    "arguments": raw_args,
                                    "result": {
                                        "status": "blocked",
                                        "message": "auto_human_question 已优先执行；超出本轮工具上限的调用未执行。",
                                    },
                                },
                                step_number,
                            )
                        messages = _clean_messages(messages)
                        continue
                    parallel_read_only = (
                        executable_count > 1
                        and all(item[4] for item in parsed_calls[:executable_count])
                        and all(bool(parallel_safe_checker(item[2])) for item in parsed_calls[:executable_count])
                    )
                    if parallel_read_only:
                        step_number = step_index + 1
                        if progress_callback:
                            progress_callback(
                                {
                                    "event": "tool_parallel_start",
                                    "step": step_number,
                                    "count": executable_count,
                                    "names": [item[2] for item in parsed_calls[:executable_count]],
                                }
                            )
                        parallel_results: dict[int, dict[str, Any]] = {}
                        with ThreadPoolExecutor(max_workers=min(executable_count, 6)) as executor:
                            futures = {}
                            for call_index, call, name, raw_args, _read_only in parsed_calls[:executable_count]:
                                if progress_callback:
                                    progress_callback({"event": "tool_start", "step": step_number, "name": name, "arguments": raw_args, "parallel": True})
                                futures[
                                    executor.submit(
                                        _run_registry_tool_with_heartbeat,
                                        self.registry,
                                        name,
                                        raw_args,
                                        step=step_number,
                                        progress_callback=progress_callback,
                                    )
                                ] = (call_index, call, name)
                            for future in as_completed(futures):
                                call_index, _call, name = futures[future]
                                try:
                                    parallel_results[call_index] = future.result()
                                except Exception as exc:
                                    parallel_results[call_index] = {"name": name, "arguments": {}, "result": {"status": "error", "message": str(exc)}}
                        for call_index, call, name, _raw_args, _read_only in parsed_calls[:executable_count]:
                            result_record = append_tool_result(call, name, parallel_results[call_index], step_number)
                            if name == "aether_discover_tools" and isinstance(result_record.get("result"), dict):
                                newly_discovered = {
                                    str(item)
                                    for item in parallel_results[call_index]["result"].get("tool_names", [])
                                    if str(item).strip()
                                }
                                if newly_discovered:
                                    before_count = len(discovered_tool_names)
                                    discovered_tool_names.update(newly_discovered)
                                    if len(discovered_tool_names) != before_count:
                                        tools = refresh_tool_schemas()
                                        if progress_callback:
                                            progress_callback(
                                                {
                                                    "event": "tool_schema_unlocked",
                                                    "step": step_number,
                                                    "tool_names": sorted(newly_discovered),
                                                    "available_tool_count": len(tools),
                                                }
                                            )
                        for call_index, call, name, raw_args, _read_only in parsed_calls[executable_count:]:
                            append_tool_result(
                                call,
                                name,
                                {
                                    "name": name,
                                    "arguments": raw_args,
                                    "result": {
                                        "status": "blocked",
                                        "message": (
                                            f"单轮已执行 {MAX_TOOL_CALLS_PER_STEP} 个工具调用（你刚才一次请求了 {len(tool_calls)} 个）；"
                                            "余下调用本轮不会执行。请先用自然语言总结已拿到的证据，或把剩余调用拆到下一轮。"
                                        ),
                                    },
                                },
                                step_number,
                            )
                        messages = _clean_messages(messages)
                        guard = _token_guard_status(messages, model_id=getattr(getattr(self.adapter, "runtime", None), "model_id", None))
                        if step_index + 1 >= TOKEN_GUARD_MIN_STEPS and guard["should_finalize"]:
                            force_final_reply_after_audit = True
                            force_final_reply_message = (
                                "上下文预算接近上限，harness 已自动停止继续调用工具。"
                                "现在必须用自然语言总结已取得证据、已写入/未写入内容和最小下一步；不要再调用工具。"
                            )
                            if progress_callback:
                                progress_callback({"event": "token_guard_finalize", "step": step_index + 1, **guard})
                        continue
                    mutating_calls_seen = 0
                    for call_index, call in enumerate(tool_calls):
                        func = call.get("function") or {}
                        name = str(func.get("name") or "")
                        raw_args = func.get("arguments") or "{}"
                        if progress_callback:
                            progress_callback({"event": "tool_start", "step": step_index + 1, "name": name, "arguments": raw_args})
                        read_only = bool(read_only_checker(name))
                        if not read_only:
                            mutating_calls_seen += 1
                        if call_index >= MAX_TOOL_CALLS_PER_STEP:
                            result = {
                                "name": name,
                                "arguments": raw_args,
                                "result": {
                                    "status": "blocked",
                                    "message": (
                                        f"单轮已执行 {MAX_TOOL_CALLS_PER_STEP} 个工具调用（你刚才一次请求了 {len(tool_calls)} 个）；"
                                        "余下调用本轮不会执行。这不是工具失败，而是 harness 为防止一次性过量调用而暂停。"
                                        "请先用自然语言总结已拿到的证据，或把剩余调用拆到下一轮。"
                                    ),
                                },
                            }
                        elif mutating_calls_seen > MAX_MUTATING_TOOL_CALLS_PER_STEP:
                            result = {
                                "name": name,
                                "arguments": raw_args,
                                "result": {
                                    "status": "blocked",
                                    "message": (
                                        f"单轮有副作用工具调用已超过上限 {MAX_MUTATING_TOOL_CALLS_PER_STEP}；本调用未执行。"
                                        "这不是工具失败，而是 harness 为避免连续写入/提交等副作用而暂停。"
                                        "请先总结已完成写入和证据，再把剩余副作用动作拆到下一轮并明确为什么需要。"
                                    ),
                                },
                            }
                        else:
                            result = _run_registry_tool_with_heartbeat(
                                self.registry,
                                name,
                                raw_args,
                                step=step_index + 1,
                                progress_callback=progress_callback,
                            )
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
                        result_record = append_tool_result(call, name, result, step_index + 1)
                        messages = _clean_messages(messages)
                        if name == "aether_discover_tools" and isinstance(result.get("result"), dict):
                            newly_discovered = {
                                str(item)
                                for item in result["result"].get("tool_names", [])
                                if str(item).strip()
                            }
                            if newly_discovered:
                                before_count = len(discovered_tool_names)
                                discovered_tool_names.update(newly_discovered)
                                if len(discovered_tool_names) != before_count:
                                    tools = refresh_tool_schemas()
                                    if progress_callback:
                                        progress_callback(
                                            {
                                                "event": "tool_schema_unlocked",
                                                "step": step_index + 1,
                                                "tool_names": sorted(newly_discovered),
                                                "available_tool_count": len(tools),
                                            }
                                        )
                        if name == "behavior_audit":
                            force_final_reply_after_audit = True
                            force_final_reply_message = (
                                "behavior_audit 已完成。现在必须给用户一个简短、证据化的自然语言结论；"
                                "不要再调用工具。若仍有后续动作，只把它列为下一步。"
                            )
                    guard = _token_guard_status(messages, model_id=getattr(getattr(self.adapter, "runtime", None), "model_id", None))
                    if step_index + 1 >= TOKEN_GUARD_MIN_STEPS and guard["should_finalize"]:
                        force_final_reply_after_audit = True
                        force_final_reply_message = (
                            "上下文预算接近上限，harness 已自动停止继续调用工具。"
                            "现在必须用自然语言总结已取得证据、已写入/未写入内容和最小下一步；不要再调用工具。"
                        )
                        if progress_callback:
                            progress_callback({"event": "token_guard_finalize", "step": step_index + 1, **guard})
                    continue
                response = _clean_text(content)
                if _contains_tool_markup(response):
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "上一条内容包含未执行的工具调用标记。现在工具调用已关闭；"
                                "不要输出 DSML、invoke、JSON 工具参数或任何工具标记。"
                                "只用自然语言总结已经完成的证据、未完成的动作和最小下一步。"
                            ),
                        }
                    )
                    messages = _clean_messages(messages)
                    try:
                        retry_reply = self.adapter.chat(
                            messages,
                            tools=[],
                            tool_choice="none",
                            max_tokens=max(int(max_tokens or 0), 1000),
                        )
                        retry_content = _clean_text(str(retry_reply.get("content") or ""))
                        response = (
                            _tool_evidence_fallback(tool_executions, reason="final reply still contained tool markup")
                            if _contains_tool_markup(retry_content)
                            else (retry_content or _tool_evidence_fallback(tool_executions, reason="empty final reply"))
                        )
                        finish_reason = "tool_markup_finalized"
                    except Exception:
                        response = _tool_evidence_fallback(tool_executions, reason="final reply retry failed")
                messages.append({"role": "assistant", "content": response})
                messages = _clean_messages(messages)
                if finish_reason == "length":
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "上一条回复因为 token 限制被截断。现在不要调用工具，不要表格，"
                                "用不超过 220 个中文字给出完整结论：当前证据、关键决策点、最小下一动作。"
                            ),
                        }
                    )
                    messages = _clean_messages(messages)
                    if progress_callback:
                        progress_callback({"event": "final_reply_after_length", "session_id": session_id, "tool_count": len(tool_executions)})
                    try:
                        retry_reply = self.adapter.chat(
                            messages,
                            tools=[],
                            tool_choice="none",
                            max_tokens=max(int(max_tokens or 0), 1200),
                        )
                        retry_content = _clean_text(str(retry_reply.get("content") or ""))
                        if retry_content:
                            response = retry_content
                            finish_reason = "length_finalized"
                            messages.append({"role": "assistant", "content": response})
                            messages = _clean_messages(messages)
                    except Exception:
                        response = (response + "\n\n[提示] 上一条回复被模型截断；请继续追问，我会基于已获得工具证据给出更短结论。").strip()
                break
            else:
                finish_reason = "tool_loop_limit"
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "本轮工具步数已经用尽。现在必须停止调用工具，基于已经获得的工具结果给用户一个简短、"
                            "诚实、证据化的自然语言总结；如果还缺工具结果，把它列为下一步，不要编造。"
                        ),
                    }
                )
                messages = _clean_messages(messages)
                if progress_callback:
                    progress_callback({"event": "final_reply_after_tool_limit", "session_id": session_id, "tool_count": len(tool_executions)})
                try:
                    final_kwargs: dict[str, Any] = {
                        "tools": [],
                        "tool_choice": "none",
                        "max_tokens": max_tokens,
                    }
                    final_reply = self.adapter.chat(messages, **final_kwargs)
                    final_content = _clean_text(str(final_reply.get("content") or ""))
                    if final_content:
                        if _contains_tool_markup(final_content):
                            messages.append(
                                {
                                    "role": "system",
                                    "content": (
                                        "刚才仍然输出了未执行的工具调用标记。最终回复禁止工具标记；"
                                        "不要写 DSML、invoke、JSON 参数。只用自然语言说明："
                                        "已验证的证据、没有执行的写回/提交动作、下一步。"
                                    ),
                                }
                            )
                            messages = _clean_messages(messages)
                            retry_reply = self.adapter.chat(
                                messages,
                                tools=[],
                                tool_choice="none",
                                max_tokens=max(int(max_tokens or 0), 1000),
                            )
                            retry_content = _clean_text(str(retry_reply.get("content") or ""))
                            final_content = (
                                _tool_evidence_fallback(tool_executions, reason="tool-loop final reply still contained tool markup")
                                if _contains_tool_markup(retry_content)
                                else (retry_content or _tool_evidence_fallback(tool_executions, reason="empty tool-loop final reply"))
                            )
                        response = final_content
                        finish_reason = "tool_loop_limit_finalized"
                        messages.append({"role": "assistant", "content": response})
                        messages = _clean_messages(messages)
                except Exception as exc:
                    response = _clean_text(response or f"工具步数已达上限，且最终总结生成失败：{exc}")

        except KeyboardInterrupt:
            finish_reason = "user_interrupted"
            response = _clean_text(response or "用户中断，本轮 partial trace 已保存。")
            if progress_callback:
                progress_callback({"event": "turn_interrupted", "session_id": session_id, "tool_count": len(tool_executions)})

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
