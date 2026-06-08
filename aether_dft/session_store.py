from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .context_budget import usable_context_chars, usable_context_tokens
from .paths import ensure_runtime_dir

SESSION_CONTEXT_MAX_TOKENS = usable_context_tokens()
SESSION_CONTEXT_MAX_CHARS = usable_context_chars()
SESSION_COMPACTION_TRIGGER_CHARS = SESSION_CONTEXT_MAX_CHARS
SESSION_COMPACTION_KEEP_RECENT_TURNS = 80
SESSION_COMPACT_SUMMARY_MAX_CHARS = min(240_000, max(6_000, SESSION_CONTEXT_MAX_CHARS // 10))


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


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    project: str | None
    created_at: str
    updated_at: str
    turn_count: int
    first_prompt: str
    last_response: str

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
            "first_prompt": first_prompt,
            "last_response": "",
            "compact_summary": "",
            "compacted_turn_count": 0,
        }
        self._state_path(session_id).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        entries = [entry for entry in self._load_index() if entry.get("session_id") != session_id]
        entries.insert(0, state)
        self._save_index(entries)
        return session_id

    def ensure_session(self, *, session_id: str | None = None, project: str | None = None, first_prompt: str = "") -> str:
        if session_id and self._state_path(session_id).exists():
            return session_id
        if session_id:
            return self.start_session(project=project, first_prompt=first_prompt, session_id=session_id)
        return self.start_session(project=project, first_prompt=first_prompt)

    def latest_session_id(self, *, project: str | None = None) -> str | None:
        for entry in self._load_index():
            if project and entry.get("project") != project:
                continue
            session_id = str(entry.get("session_id") or "")
            if session_id:
                return session_id
        return None

    def list_sessions(self, *, project: str | None = None, limit: int = 20) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        for entry in self._load_index():
            if project and entry.get("project") != project:
                continue
            summaries.append(
                SessionSummary(
                    session_id=str(entry.get("session_id") or ""),
                    project=entry.get("project"),
                    created_at=str(entry.get("created_at") or ""),
                    updated_at=str(entry.get("updated_at") or ""),
                    turn_count=int(entry.get("turn_count") or 0),
                    first_prompt=str(entry.get("first_prompt") or ""),
                    last_response=str(entry.get("last_response") or ""),
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
        state["last_response"] = _clean_text(record.get("response"))
        state["project"] = record.get("project", state.get("project"))
        self._state_path(session_id).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

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
        self._maybe_compact_session(session_id)
        return transcript_path

    def _read_transcript_rows(self, session_id: str) -> list[dict[str, Any]]:
        path = self._transcript_path(session_id)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(_clean_jsonable(json.loads(line)))
            except json.JSONDecodeError:
                continue
        return rows

    def read_transcript(self, session_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._read_transcript_rows(session_id)
        return rows[-limit:]

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

    def _maybe_compact_session(self, session_id: str) -> None:
        rows = self._read_transcript_rows(session_id)
        if len(rows) <= SESSION_COMPACTION_KEEP_RECENT_TURNS:
            return
        approx_chars = sum(len(json.dumps(row.get("record") or row, ensure_ascii=False, default=str)) for row in rows)
        if approx_chars < usable_context_chars():
            return
        older_rows = rows[:-SESSION_COMPACTION_KEEP_RECENT_TURNS]
        summary = self._build_compact_summary(older_rows)
        if not summary:
            return
        state = self.load_state(session_id)
        state["compact_summary"] = summary
        state["compacted_turn_count"] = len(older_rows)
        state["last_compacted_at"] = _now_iso()
        self._state_path(session_id).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        entries = [entry for entry in self._load_index() if entry.get("session_id") != session_id]
        entries.insert(0, state)
        self._save_index(entries)

    def build_session_context(self, session_id: str, *, limit: int | None = None, max_chars: int | None = None) -> str:
        """Summarize recent turns so a resumed session actually carries forward context."""

        state = self.load_state(session_id)
        max_chars = max_chars or usable_context_chars()
        recent_turns = self.read_transcript(session_id, limit=limit or 10_000)
        if not recent_turns:
            return ""

        start_turn = max(1, int(state.get("turn_count") or 0) - len(recent_turns) + 1)
        lines = [
            "## Session Context",
            "",
            f"- session_id: {session_id}",
            f"- project: {state.get('project') or 'none'}",
            f"- turn_count: {int(state.get('turn_count') or 0)}",
            f"- model_usable_context_tokens: {usable_context_tokens()}",
            f"- session_context_char_budget: {max_chars}",
            "- this context is resume-only and should not be treated as new facts",
        ]
        compact_summary = str(state.get("compact_summary") or "").strip()
        if compact_summary:
            lines.extend(
                [
                    f"- compacted_turn_count: {int(state.get('compacted_turn_count') or 0)}",
                    "",
                    compact_summary,
                ]
            )
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
        state = self.load_state(resolved)
        return {
            "status": "ok",
            "session_id": resolved,
            "state": state,
            "recent_turns": self.read_transcript(resolved, limit=limit),
            "session_context": self.build_session_context(resolved),
        }
