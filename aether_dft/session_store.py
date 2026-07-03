from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .context_budget import context_budget, usable_context_chars, usable_context_tokens
from .paths import ensure_runtime_dir

SESSION_CONTEXT_MAX_TOKENS = usable_context_tokens()
SESSION_CONTEXT_MAX_CHARS = usable_context_chars()
SESSION_COMPACTION_TRIGGER_CHARS = SESSION_CONTEXT_MAX_CHARS
SESSION_COMPACTION_KEEP_RECENT_TURNS = 80
SESSION_COMPACT_SUMMARY_MAX_CHARS = min(240_000, max(6_000, SESSION_CONTEXT_MAX_CHARS // 10))
SESSION_RECENT_CONTEXT_DEFAULT_TURNS = 10_000


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clean_text(value: Any) -> str:
    return str(value or "").encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _clean_jsonable(value: Any) -> Any:
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        return [_clean_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {_clean_text(key): _clean_jsonable(item) for key, item in value.items()}
    return value


def _derive_session_title(text: Any, *, limit: int = 42) -> str:
    """Create a stable human-readable title without calling an external model."""

    cleaned = " ".join(_clean_text(text).split()).strip()
    if not cleaned:
        return "New research chat"
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    project: str | None
    created_at: str
    updated_at: str
    turn_count: int
    title: str
    first_prompt: str
    last_response: str
    pending_turn_status: str = ""
    pending_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AetherSessionStore:
    """Repo-local session persistence for the AETHER scientific harness.

    The format is intentionally simple and inspectable:
    - ``.aether/runtime/sessions/sessions.json`` is the index.
    - ``.aether/runtime/sessions/<session_id>/state.json`` is the latest state.
    - ``.aether/runtime/sessions/<session_id>/transcript.jsonl`` is the append-only transcript.
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or ensure_runtime_dir("sessions")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "sessions.json"

    def _session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def _state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "state.json"

    def _transcript_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "transcript.jsonl"

    def _project_session_ref_dir(self, project: str | None) -> Path | None:
        if not project:
            return None
        try:
            from .research_workspace import resolve_research_project
        except Exception:
            return None
        paths = resolve_research_project(project)
        if paths is None:
            return None
        return paths.root / ".aether" / "sessions"

    def _project_session_ref_path(self, state: dict[str, Any]) -> Path | None:
        session_id = str(state.get("session_id") or "").strip()
        if not session_id:
            return None
        ref_dir = self._project_session_ref_dir(state.get("project"))
        if ref_dir is None:
            return None
        return ref_dir / f"{session_id}.json"

    def _write_project_session_reference(self, state: dict[str, Any]) -> Path | None:
        """Mirror lightweight session metadata into ``research/<project>``.

        The canonical transcript stays under ``.aether/runtime/sessions`` so the
        harness has one durable append-only store.  This reference makes the
        project directory self-describing without copying full conversations
        into human-maintained research notes.
        """

        ref_path = self._project_session_ref_path(state)
        if ref_path is None:
            return None
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        session_id = str(state.get("session_id") or "")
        reference = {
            "session_id": session_id,
            "project": state.get("project"),
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            "turn_count": int(state.get("turn_count") or 0),
            "title": str(state.get("title") or _derive_session_title(state.get("first_prompt"))),
            "first_prompt": str(state.get("first_prompt") or ""),
            "last_response": str(state.get("last_response") or ""),
            "pending_turn": state.get("pending_turn") if isinstance(state.get("pending_turn"), dict) else None,
            "canonical_state": str(self._state_path(session_id)),
            "canonical_transcript": str(self._transcript_path(session_id)),
            "note": "Lightweight project-facing index; canonical transcript remains in .aether/runtime/sessions.",
        }
        ref_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")

        index_path = ref_path.parent / "sessions.json"
        try:
            existing = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
        except Exception:
            existing = []
        if not isinstance(existing, list):
            existing = []
        entries = [entry for entry in existing if isinstance(entry, dict) and entry.get("session_id") != session_id]
        entries.insert(0, reference)
        index_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        return ref_path

    def project_session_reference_path(self, session_id: str) -> Path | None:
        """Return the ``research/<project>`` reference path for a session if any."""

        try:
            state = self.load_state(session_id)
        except Exception:
            return None
        return self._project_session_ref_path(state)

    def _load_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def _save_index(self, entries: list[dict[str, Any]]) -> None:
        self.index_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_state(self, session_id: str, state: dict[str, Any]) -> None:
        self._state_path(session_id).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        entries = [entry for entry in self._load_index() if entry.get("session_id") != session_id]
        entries.insert(0, state)
        self._save_index(entries)
        self._write_project_session_reference(state)

    def start_session(self, *, project: str | None = None, first_prompt: str = "", session_id: str | None = None) -> str:
        session_id = session_id or f"session_{uuid4().hex[:12]}"
        now = _now_iso()
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "session_id": session_id,
            "project": project,
            "created_at": now,
            "updated_at": now,
            "turn_count": 0,
            "title": _derive_session_title(first_prompt),
            "first_prompt": first_prompt,
            "last_response": "",
            "compact_summary": "",
            "compacted_turn_count": 0,
        }
        self._write_state(session_id, state)
        return session_id

    def ensure_session(self, *, session_id: str | None = None, project: str | None = None, first_prompt: str = "") -> str:
        if session_id and self._state_path(session_id).exists():
            return session_id
        if session_id:
            return self.start_session(project=project, first_prompt=first_prompt, session_id=session_id)
        return self.start_session(project=project, first_prompt=first_prompt)

    def record_pending_turn(
        self,
        session_id: str,
        *,
        prompt: str,
        project: str | None = None,
        model_id: str | None = None,
        status: str = "in_progress",
        error: str | None = None,
    ) -> dict[str, Any]:
        """Remember an in-flight user prompt without writing a fake transcript turn."""

        state = self.load_state(session_id)
        now = _now_iso()
        pending = {
            "prompt": str(prompt or ""),
            "project": project if project is not None else state.get("project"),
            "model_id": model_id,
            "status": status,
            "error": str(error or ""),
            "created_at": now,
            "updated_at": now,
        }
        state["pending_turn"] = pending
        state["updated_at"] = now
        if project is not None:
            state["project"] = project
        if not state.get("first_prompt"):
            state["first_prompt"] = str(prompt or "")
        if not state.get("title") or state.get("title") == "New research chat":
            state["title"] = _derive_session_title(state.get("first_prompt") or prompt)
        self._write_state(session_id, state)
        return pending

    def mark_pending_turn_failed(self, session_id: str, *, error: str) -> dict[str, Any] | None:
        state = self.load_state(session_id)
        pending = state.get("pending_turn")
        if not isinstance(pending, dict) or not str(pending.get("prompt") or "").strip():
            return None
        pending["status"] = "failed"
        pending["error"] = str(error or "")
        pending["updated_at"] = _now_iso()
        state["pending_turn"] = pending
        state["updated_at"] = pending["updated_at"]
        self._write_state(session_id, state)
        return pending

    def clear_pending_turn(self, session_id: str) -> None:
        state = self.load_state(session_id)
        if "pending_turn" not in state:
            return
        state.pop("pending_turn", None)
        state["updated_at"] = _now_iso()
        self._write_state(session_id, state)

    def pending_turn(self, session_id: str) -> dict[str, Any] | None:
        pending = self.load_state(session_id).get("pending_turn")
        if isinstance(pending, dict) and str(pending.get("prompt") or "").strip():
            return pending
        return None

    def rename_session(self, session_id: str, title: str) -> dict[str, Any]:
        state = self.load_state(session_id)
        cleaned = " ".join(str(title or "").split()).strip()
        if not cleaned:
            raise ValueError("session title 不能为空。")
        state["title"] = _derive_session_title(cleaned, limit=80)
        state["updated_at"] = _now_iso()
        self._write_state(session_id, state)
        return state

    def latest_session_id(self, *, project: str | None = None) -> str | None:
        for entry in self._load_index():
            if project and entry.get("project") != project:
                continue
            session_id = str(entry.get("session_id") or "")
            if session_id and self._state_path(session_id).exists():
                return session_id
        return None

    def list_sessions(self, *, project: str | None = None, limit: int = 20) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        for entry in self._load_index():
            if project and entry.get("project") != project:
                continue
            session_id = str(entry.get("session_id") or "")
            if not session_id or not self._state_path(session_id).exists():
                continue
            pending = entry.get("pending_turn") if isinstance(entry.get("pending_turn"), dict) else {}
            summaries.append(
                SessionSummary(
                    session_id=session_id,
                    project=entry.get("project"),
                    created_at=str(entry.get("created_at") or ""),
                    updated_at=str(entry.get("updated_at") or ""),
                    turn_count=int(entry.get("turn_count") or 0),
                    title=str(entry.get("title") or _derive_session_title(entry.get("first_prompt"))),
                    first_prompt=str(entry.get("first_prompt") or ""),
                    last_response=str(entry.get("last_response") or ""),
                    pending_turn_status=str(pending.get("status") or ""),
                    pending_prompt=str(pending.get("prompt") or ""),
                )
            )
            if len(summaries) >= limit:
                break
        return summaries

    def load_state(self, session_id: str) -> dict[str, Any]:
        path = self._state_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"session 不存在: {session_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def append_turn(self, session_id: str, record: dict[str, Any]) -> Path:
        record = _clean_jsonable(record)
        state = self.load_state(session_id)
        now = _now_iso()
        state["updated_at"] = now
        state["turn_count"] = int(state.get("turn_count") or 0) + 1
        if not state.get("first_prompt"):
            state["first_prompt"] = str(record.get("prompt") or "")
        if not state.get("title") or state.get("title") == "New research chat":
            state["title"] = _derive_session_title(state.get("first_prompt") or record.get("prompt"))
        state["last_response"] = _clean_text(record.get("response"))
        state["project"] = record.get("project", state.get("project"))
        state.pop("pending_turn", None)
        self._state_path(session_id).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_project_session_reference(state)

        transcript_record = {
            "type": "turn",
            "timestamp": now,
            "session_id": session_id,
            "record": record,
        }
        transcript_path = self._transcript_path(session_id)
        with transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(transcript_record, ensure_ascii=False) + "\n")

        entries = [entry for entry in self._load_index() if entry.get("session_id") != session_id]
        entries.insert(0, state)
        self._save_index(entries)
        self._maybe_compact_session(session_id, reason="append_turn")
        return transcript_path

    def _read_transcript_rows(self, session_id: str) -> list[dict[str, Any]]:
        rows, _diagnostics = self._read_transcript_rows_with_recovery(session_id)
        return rows

    @staticmethod
    def _sanitize_transcript_row(row: Any) -> tuple[dict[str, Any] | None, int]:
        """Return an API-safe transcript row plus skipped nested item count.

        Session files are append-only JSONL and may contain half-written rows,
        stale fields, or malformed tool records after interruption.  Resume
        should repair what it can and drop only the broken fragments instead of
        crashing the whole conversation.
        """

        if not isinstance(row, dict):
            return None, 0
        record = row.get("record")
        if not isinstance(record, dict):
            return None, 0
        prompt = _clean_text(record.get("prompt"))
        response = _clean_text(record.get("response"))
        raw_tools = record.get("tool_executions")
        skipped_tools = 0
        tools: list[dict[str, Any]] = []
        if isinstance(raw_tools, list):
            for item in raw_tools:
                if not isinstance(item, dict) or not str(item.get("name") or "").strip():
                    skipped_tools += 1
                    continue
                cleaned_tool = {
                    "name": _clean_text(item.get("name")).strip(),
                    "arguments": item.get("arguments") if isinstance(item.get("arguments"), dict) else {},
                    "result": item.get("result") if isinstance(item.get("result"), dict) else {},
                }
                for key in ("status", "duration_s", "started_at", "finished_at"):
                    if key in item:
                        cleaned_tool[key] = _clean_jsonable(item.get(key))
                tools.append(_clean_jsonable(cleaned_tool))
        if not prompt.strip() and not response.strip() and not tools:
            return None, skipped_tools
        clean_record = dict(record)
        clean_record["prompt"] = prompt
        clean_record["response"] = response
        if tools:
            clean_record["tool_executions"] = tools
        else:
            clean_record.pop("tool_executions", None)
        clean_row = dict(row)
        clean_row["type"] = str(clean_row.get("type") or "turn")
        clean_row["record"] = _clean_jsonable(clean_record)
        return _clean_jsonable(clean_row), skipped_tools

    def _read_transcript_rows_with_recovery(self, session_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        path = self._transcript_path(session_id)
        diagnostics = {
            "status": "ok",
            "session_id": session_id,
            "path": str(path),
            "raw_rows": 0,
            "kept_rows": 0,
            "invalid_json_rows": 0,
            "malformed_rows": 0,
            "skipped_tool_records": 0,
        }
        if not path.exists():
            return [], diagnostics
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            diagnostics["raw_rows"] += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                diagnostics["invalid_json_rows"] += 1
                continue
            sanitized, skipped_tools = self._sanitize_transcript_row(raw)
            diagnostics["skipped_tool_records"] += skipped_tools
            if sanitized is None:
                diagnostics["malformed_rows"] += 1
                continue
            rows.append(sanitized)
        diagnostics["kept_rows"] = len(rows)
        if (
            diagnostics["invalid_json_rows"]
            or diagnostics["malformed_rows"]
            or diagnostics["skipped_tool_records"]
        ):
            diagnostics["status"] = "recovered"
        return rows, diagnostics

    def read_transcript(self, session_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._read_transcript_rows(session_id)
        return rows[-limit:]

    def search_transcript(self, session_id: str, *, query: str = "", limit: int = 20) -> list[dict[str, Any]]:
        rows = self._read_transcript_rows(session_id)
        needle = " ".join(str(query or "").lower().split())
        matches: list[dict[str, Any]] = []
        for row in reversed(rows):
            record = row.get("record") or {}
            haystack = " ".join(
                [
                    str(record.get("prompt") or ""),
                    str(record.get("response") or ""),
                    json.dumps(record.get("tool_executions") or [], ensure_ascii=False, default=str),
                ]
            ).lower()
            if not needle or needle in haystack:
                matches.append(row)
            if len(matches) >= limit:
                break
        return list(reversed(matches))

    def rank_sessions(
        self,
        *,
        query: str,
        project: str | None = None,
        exclude_session_id: str | None = None,
        limit: int = 50,
        max_results: int = 8,
        selector: Any | None = None,
        semantic: bool = True,
    ) -> dict[str, Any]:
        from .session_search import rank_session_summaries

        sessions = [
            item
            for item in self.list_sessions(project=project, limit=limit)
            if not exclude_session_id or item.session_id != exclude_session_id
        ]
        return rank_session_summaries(
            query,
            sessions,
            transcript_loader=lambda sid: self.read_transcript(sid, limit=8),
            max_results=max_results,
            selector=selector,
            semantic=semantic,
        )

    def analyze_context(self, session_id: str) -> dict[str, Any]:
        """Explain what is consuming resumable session context.

        The output is intentionally approximate and character-based because it
        must work without provider tokenizers.  It still gives the model/user a
        concrete diagnosis: whether long-term context is dominated by human
        prompts, assistant summaries, or large tool results such as OUTCAR/logs.
        """

        state = self.load_state(session_id)
        rows = self._read_transcript_rows(session_id)
        budget = context_budget()
        buckets = {
            "user_prompt_chars": 0,
            "assistant_response_chars": 0,
            "tool_argument_chars": 0,
            "tool_result_chars": 0,
            "other_record_chars": 0,
            "compact_summary_chars": len(str(state.get("compact_summary") or "")),
        }
        tool_results: dict[str, int] = {}
        tool_requests: dict[str, int] = {}
        large_turns: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            record = row.get("record") if isinstance(row, dict) else {}
            if not isinstance(record, dict):
                continue
            prompt_chars = len(str(record.get("prompt") or ""))
            response_chars = len(str(record.get("response") or ""))
            buckets["user_prompt_chars"] += prompt_chars
            buckets["assistant_response_chars"] += response_chars
            turn_tool_chars = 0
            for tool in record.get("tool_executions") or []:
                if not isinstance(tool, dict):
                    continue
                name = str(tool.get("name") or "unknown")
                arg_chars = len(json.dumps(tool.get("arguments") or {}, ensure_ascii=False, default=str))
                result_chars = len(json.dumps(tool.get("result") or {}, ensure_ascii=False, default=str))
                buckets["tool_argument_chars"] += arg_chars
                buckets["tool_result_chars"] += result_chars
                tool_requests[name] = tool_requests.get(name, 0) + arg_chars
                tool_results[name] = tool_results.get(name, 0) + result_chars
                turn_tool_chars += arg_chars + result_chars
            known = {"prompt", "response", "tool_executions"}
            other_chars = len(json.dumps({k: v for k, v in record.items() if k not in known}, ensure_ascii=False, default=str))
            buckets["other_record_chars"] += other_chars
            turn_chars = prompt_chars + response_chars + turn_tool_chars + other_chars
            if turn_chars >= 2000:
                large_turns.append(
                    {
                        "turn": index,
                        "chars": turn_chars,
                        "prompt": self._collapse_text(record.get("prompt"), limit=120),
                        "tools": self._tool_trail(list(record.get("tool_executions") or []), limit=4),
                    }
                )
        total = sum(int(value) for value in buckets.values())

        def _top(mapping: dict[str, int], *, limit: int = 8) -> list[dict[str, Any]]:
            return [
                {"name": name, "chars": chars, "percent": round((chars / total) * 100, 2) if total else 0.0}
                for name, chars in sorted(mapping.items(), key=lambda item: item[1], reverse=True)[:limit]
            ]

        top_buckets = [
            {"name": name, "chars": chars, "percent": round((chars / total) * 100, 2) if total else 0.0}
            for name, chars in sorted(buckets.items(), key=lambda item: item[1], reverse=True)
            if chars
        ]
        recommendations = self._context_recommendations(
            total=total,
            buckets=buckets,
            tool_results=tool_results,
            large_turns=large_turns,
            compact_summary_chars=int(buckets.get("compact_summary_chars") or 0),
        )
        return {
            "status": "ok",
            "session_id": session_id,
            "turn_count": len(rows),
            "context_budget": budget.to_dict(),
            "approx_total_chars": total,
            "auto_compact_threshold_chars": budget.auto_compact_chars,
            "auto_compact_usage_percent": round((total / budget.auto_compact_chars) * 100, 2) if budget.auto_compact_chars else 0.0,
            "buckets": buckets,
            "top_buckets": top_buckets,
            "top_tool_results": _top(tool_results),
            "top_tool_requests": _top(tool_requests),
            "large_turns": large_turns[-8:],
            "recommendations": recommendations,
            "guidance": (
                "Use this as a context diagnosis, not a scientific conclusion. "
                "Large tool_result buckets should usually be microcompacted/persisted as artifacts before long follow-up chats."
            ),
        }

    @staticmethod
    def _context_recommendations(
        *,
        total: int,
        buckets: dict[str, int],
        tool_results: dict[str, int],
        large_turns: list[dict[str, Any]],
        compact_summary_chars: int,
    ) -> list[dict[str, Any]]:
        recommendations: list[dict[str, Any]] = []
        budget = context_budget()
        if total >= budget.auto_compact_chars:
            recommendations.append(
                {
                    "priority": "high",
                    "action": "auto_compact_due",
                    "reason": "session context crossed the automatic compaction threshold",
                    "threshold_chars": budget.auto_compact_chars,
                    "command": "/compact",
                }
            )
        elif total >= int(budget.auto_compact_chars * 0.75):
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "compact_session",
                    "reason": "session context is approaching the automatic compaction threshold",
                    "command": "/compact 12",
                }
            )
        tool_result_chars = int(buckets.get("tool_result_chars") or 0)
        if total and tool_result_chars / max(total, 1) >= 0.35:
            recommendations.append(
                {
                    "priority": "high",
                    "action": "persist_large_tool_outputs",
                    "reason": "tool results dominate context; keep paths/digests in chat and persist OUTCAR/log/manifests as artifacts",
                    "candidate_tools": ["outcar analyze", "research_learning_capture", "candidate_outcome_record"],
                }
            )
        if large_turns:
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "review_large_turns",
                    "reason": "some turns are unusually large; summarize their scientific conclusion before continuing",
                    "large_turn_count": len(large_turns),
                }
            )
        heavy_tools = [name for name, _chars in sorted(tool_results.items(), key=lambda item: item[1], reverse=True)[:3]]
        if any(name in {"cluster_job_partial_outcar", "job_partial_outcar", "vasp_output_scan", "cluster_job_tail_log"} for name in heavy_tools):
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "replace_raw_vasp_output_with_dossier",
                    "reason": "raw VASP/log output should become an evidence dossier plus paths, not repeated chat context",
                    "candidate_tools": ["cluster_remote_fetch", "result_interpret", "research_learning_capture"],
                }
            )
        if compact_summary_chars:
            recommendations.append(
                {
                    "priority": "low",
                    "action": "trust_compact_summary_but_verify_latest_artifacts",
                    "reason": "older turns are summarized; current scientific claims should still be checked against project files and latest evidence",
                }
            )
        if not recommendations:
            recommendations.append(
                {
                    "priority": "low",
                    "action": "continue",
                    "reason": "context is healthy; continue natural-language research work",
                }
            )
        return recommendations

    @staticmethod
    def _collapse_text(value: Any, *, limit: int = 220) -> str:
        text = " ".join(_clean_text(value).split()).strip()
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    def _turn_digest_line(self, turn: dict[str, Any], *, turn_no: int | None = None, text_limit: int = 180) -> str:
        record = dict(turn.get("record") or {})
        prompt = self._collapse_text(record.get("prompt"), limit=text_limit)
        response = self._collapse_text(record.get("response"), limit=text_limit)
        tool_trail = self._tool_trail(list(record.get("tool_executions") or []))
        prefix = f"turn {turn_no}: " if turn_no is not None else ""
        tools = f" tools=[{'; '.join(tool_trail)}]" if tool_trail else ""
        return f"- {prefix}user={prompt or 'n/a'} | assistant={response or 'n/a'}{tools}"

    def _tool_trail(self, tool_executions: list[Any], *, limit: int = 5) -> list[str]:
        """Compact recent tool activity without pulling large results into context."""

        trail: list[str] = []
        for item in tool_executions[-limit:]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            status = str(result.get("status") or result.get("verdict") or "").strip()
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            arg_bits: list[str] = []
            for key in ("project", "material", "adsorbate", "category", "query", "run_id", "job_id", "task_type"):
                value = arguments.get(key) if isinstance(arguments, dict) else None
                if value not in (None, "", [], {}):
                    arg_bits.append(f"{key}={self._collapse_text(value, limit=36)}")
                if len(arg_bits) >= 3:
                    break
            suffix_parts = []
            if status:
                suffix_parts.append(status)
            if arg_bits:
                suffix_parts.append(", ".join(arg_bits))
            suffix = f"({'; '.join(suffix_parts)})" if suffix_parts else ""
            trail.append(f"{name}{suffix}")
        return trail

    def _build_compact_summary(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        lines = [
            "## Compacted Session Summary",
            "",
            "This is an automatic extractive summary of older turns. Prefer project files and latest turns when facts conflict.",
        ]
        max_summary_turns = 300
        start_turn = max(1, len(rows) - max_summary_turns + 1)
        for offset, turn in enumerate(rows[-max_summary_turns:], start=start_turn):
            lines.append(self._turn_digest_line(turn, turn_no=offset, text_limit=170))
        text = "\n".join(lines).strip()
        if len(text) <= SESSION_COMPACT_SUMMARY_MAX_CHARS:
            return text
        return text[: SESSION_COMPACT_SUMMARY_MAX_CHARS - 15].rstrip() + "\n...[compacted]"

    def _session_context_record_chars(self, rows: list[dict[str, Any]]) -> int:
        return sum(len(json.dumps(row.get("record") or row, ensure_ascii=False, default=str)) for row in rows)

    def _maybe_compact_session(self, session_id: str, *, reason: str = "automatic") -> dict[str, Any] | None:
        rows = self._read_transcript_rows(session_id)
        if len(rows) <= SESSION_COMPACTION_KEEP_RECENT_TURNS:
            return None
        approx_chars = self._session_context_record_chars(rows)
        budget = context_budget()
        if approx_chars < budget.auto_compact_chars:
            return None
        return self.compact_session(
            session_id,
            keep_recent=SESSION_COMPACTION_KEEP_RECENT_TURNS,
            trigger="automatic",
            reason=reason,
            approx_chars_before=approx_chars,
        )

    def compact_if_needed(self, session_id: str, *, reason: str = "before_prompt") -> dict[str, Any]:
        """Run Codex/Claude-Code-style automatic compaction if the session is large.

        The transcript remains append-only.  Only the model-visible resume
        context changes: older turns are replaced by an extractive compact
        summary plus recent turn trails.
        """

        result = self._maybe_compact_session(session_id, reason=reason)
        if result is not None:
            return result
        rows = self._read_transcript_rows(session_id)
        budget = context_budget()
        approx_chars = self._session_context_record_chars(rows)
        return {
            "status": "skipped",
            "reason": "below_auto_compact_threshold",
            "session_id": session_id,
            "turn_count": len(rows),
            "approx_chars": approx_chars,
            "auto_compact_threshold_chars": budget.auto_compact_chars,
            "context_budget": budget.to_dict(),
        }

    def compact_session(
        self,
        session_id: str,
        *,
        keep_recent: int = SESSION_COMPACTION_KEEP_RECENT_TURNS,
        trigger: str = "manual",
        reason: str = "",
        approx_chars_before: int | None = None,
    ) -> dict[str, Any]:
        """Write a compact summary for older turns without deleting transcript rows."""

        keep_recent = max(1, int(keep_recent or SESSION_COMPACTION_KEEP_RECENT_TURNS))
        rows = self._read_transcript_rows(session_id)
        if len(rows) <= keep_recent:
            state = self.load_state(session_id)
            return {
                "status": "skipped",
                "reason": "not_enough_turns",
                "session_id": session_id,
                "turn_count": len(rows),
                "keep_recent": keep_recent,
                "compacted_turn_count": int(state.get("compacted_turn_count") or 0),
            }
        older_rows = rows[:-keep_recent]
        summary = self._build_compact_summary(older_rows)
        if not summary:
            return {
                "status": "skipped",
                "reason": "empty_summary",
                "session_id": session_id,
                "turn_count": len(rows),
                "keep_recent": keep_recent,
            }
        state = self.load_state(session_id)
        budget = context_budget()
        approx_chars_before = approx_chars_before if approx_chars_before is not None else self._session_context_record_chars(rows)
        state["compact_summary"] = summary
        state["compacted_turn_count"] = len(older_rows)
        state["last_compacted_at"] = _now_iso()
        state["last_compact_trigger"] = trigger
        state["last_compact_reason"] = reason
        state["compact_keep_recent_turns"] = keep_recent
        state["last_compact_stats"] = {
            "approx_chars_before": approx_chars_before,
            "compact_summary_chars": len(summary),
            "kept_recent_turns": keep_recent,
            "context_budget": budget.to_dict(),
            "auto_compact_threshold_chars": budget.auto_compact_chars,
        }
        self._state_path(session_id).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        entries = [entry for entry in self._load_index() if entry.get("session_id") != session_id]
        entries.insert(0, state)
        self._save_index(entries)
        self._write_project_session_reference(state)
        return {
            "status": "ok",
            "session_id": session_id,
            "turn_count": len(rows),
            "keep_recent": keep_recent,
            "compacted_turn_count": len(older_rows),
            "compact_summary_chars": len(summary),
            "trigger": trigger,
            "reason": reason,
            "approx_chars_before": approx_chars_before,
            "context_budget": budget.to_dict(),
        }

    def build_session_context(self, session_id: str, *, limit: int | None = None, max_chars: int | None = None) -> str:
        """Summarize recent turns so a resumed session actually carries forward context."""

        state = self.load_state(session_id)
        if limit is None and hasattr(self, "compact_if_needed"):
            self.compact_if_needed(session_id, reason="build_session_context")
            state = self.load_state(session_id)
        max_chars = max_chars or usable_context_chars()
        if limit is None and int(state.get("compacted_turn_count") or 0) > 0:
            default_limit = int(state.get("compact_keep_recent_turns") or SESSION_COMPACTION_KEEP_RECENT_TURNS)
        else:
            default_limit = SESSION_RECENT_CONTEXT_DEFAULT_TURNS
        recent_turns = self.read_transcript(session_id, limit=limit or default_limit)
        budget = context_budget()

        start_turn = max(1, int(state.get("turn_count") or 0) - len(recent_turns) + 1)
        lines = [
            "## Session Context",
            "",
            f"- session_id: {session_id}",
            f"- project: {state.get('project') or 'none'}",
            f"- turn_count: {int(state.get('turn_count') or 0)}",
            f"- model_usable_context_tokens: {usable_context_tokens()}",
            f"- model_context_window_tokens: {budget.context_window_tokens}",
            f"- auto_compact_threshold_chars: {budget.auto_compact_chars}",
            f"- session_context_char_budget: {max_chars}",
            "- this context is resume-only and should not be treated as new facts",
        ]
        compact_summary = str(state.get("compact_summary") or "").strip()
        if compact_summary:
            lines.extend(
                [
                    f"- compacted_turn_count: {int(state.get('compacted_turn_count') or 0)}",
                    f"- last_compact_trigger: {state.get('last_compact_trigger') or 'unknown'}",
                    f"- last_compact_reason: {state.get('last_compact_reason') or ''}",
                    "",
                    compact_summary,
                ]
            )
        pending = state.get("pending_turn")
        if isinstance(pending, dict) and str(pending.get("prompt") or "").strip():
            lines.extend(
                [
                    "",
                    "## Pending Turn",
                    f"- status: {pending.get('status') or 'in_progress'}",
                    f"- updated_at: {pending.get('updated_at') or ''}",
                    f"- user_prompt: {self._collapse_text(pending.get('prompt'), limit=360)}",
                    "- note: this prompt has not received a completed assistant answer; continue it only when the user asks to continue/retry",
                ]
            )
        if recent_turns:
            lines.extend(["", "### Recent Turns"])
            for offset, turn in enumerate(recent_turns, start=start_turn):
                record = dict(turn.get("record") or {})
                prompt = self._collapse_text(record.get("prompt"))
                response = self._collapse_text(record.get("response"))
                tool_trail = self._tool_trail(list(record.get("tool_executions") or []))
                lines.append(f"- turn {offset} user: {prompt or 'n/a'}")
                lines.append(f"  assistant: {response or 'n/a'}")
                if tool_trail:
                    lines.append(f"  tool_trail: {'; '.join(tool_trail)}")

        text = "\n".join(lines).strip()
        if len(text) <= max_chars:
            return text
        keep_head = max(2600, max_chars // 2)
        keep_tail = max(1800, max_chars - keep_head - 40)
        return f"{text[:keep_head].rstrip()}\n...\n{text[-keep_tail:].lstrip()}"

    def resume_payload(self, *, session_id: str | None = None, project: str | None = None, limit: int = 8) -> dict[str, Any]:
        resolved = session_id or self.latest_session_id(project=project)
        if not resolved:
            return {"status": "empty", "session_id": None, "state": None, "recent_turns": []}
        if not self._state_path(resolved).exists():
            return {"status": "missing", "session_id": resolved, "state": None, "recent_turns": []}
        state = self.load_state(resolved)
        recent_turns, recovery = self._read_transcript_rows_with_recovery(resolved)
        recent_turns = recent_turns[-limit:]
        return {
            "status": "ok",
            "session_id": resolved,
            "state": state,
            "recent_turns": recent_turns,
            "recovery": recovery,
            "session_context": self.build_session_context(resolved),
        }
