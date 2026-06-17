from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .paths import ensure_runtime_dir
from .project_state import project_paths


TERMINAL_STATES = {"done", "cancelled", "canceled", "completed", "dismissed"}


def _now() -> datetime:
    return datetime.now().astimezone()


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def _followup_path(project: str | None = None) -> Path:
    if project:
        root = project_paths(project).root / ".aether"
    else:
        root = ensure_runtime_dir("followups")
    root.mkdir(parents=True, exist_ok=True)
    return root / "followups.json"


def _load(project: str | None = None) -> list[dict[str, Any]]:
    path = _followup_path(project)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        rows = data.get("followups")
    else:
        rows = data
    return rows if isinstance(rows, list) else []


def _save(rows: list[dict[str, Any]], project: str | None = None) -> None:
    path = _followup_path(project)
    body = {
        "version": 1,
        "updated_at": _now_iso(),
        "followups": rows,
    }
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_project(project: Any) -> str | None:
    text = str(project or "").strip()
    return text or None


def schedule_followup(
    *,
    project: str | None = None,
    prompt: str,
    due_at: str | None = None,
    interval_minutes: int | None = None,
    title: str | None = None,
    related_job_id: str | None = None,
    related_run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a project-local scientific follow-up prompt.

    This is a durable reminder/checkpoint queue, not a daemon.  The model may
    inspect due items at chat start or when the user asks, then decide which
    evidence-gathering tools are appropriate.
    """

    project = _normalize_project(project)
    text = str(prompt or "").strip()
    if not text:
        return {"status": "error", "message": "prompt 不能为空。"}
    due_dt = _parse_dt(due_at)
    if due_dt is None:
        if interval_minutes is None:
            return {"status": "error", "message": "需要 due_at ISO 时间或 interval_minutes。"}
        try:
            minutes = max(1, min(int(interval_minutes), 60 * 24 * 365))
        except (TypeError, ValueError):
            return {"status": "error", "message": "interval_minutes 必须是整数。"}
        due_dt = _now() + timedelta(minutes=minutes)
    try:
        interval_int = int(interval_minutes) if interval_minutes is not None else None
    except (TypeError, ValueError):
        return {"status": "error", "message": "interval_minutes 必须是整数。"}
    if interval_int is not None:
        interval_int = max(1, min(interval_int, 60 * 24 * 365))
    entry = {
        "id": f"followup_{uuid4().hex[:10]}",
        "project": project,
        "title": str(title or "").strip() or text[:80],
        "prompt": text,
        "due_at": due_dt.isoformat(timespec="seconds"),
        "interval_minutes": interval_int,
        "related_job_id": str(related_job_id or "").strip(),
        "related_run_id": str(related_run_id or "").strip(),
        "status": "scheduled",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }
    rows = [row for row in _load(project) if isinstance(row, dict) and row.get("id") != entry["id"]]
    rows.insert(0, entry)
    _save(rows[:200], project)
    return {
        "status": "ok",
        "path": str(_followup_path(project)),
        "followup": entry,
        "guidance": "这是未来要重新检查的科研意图；到期后模型应先收集证据再回答，不要把 prompt 当已完成事实。",
    }


def list_followups(*, project: str | None = None, include_done: bool = False, limit: int = 50) -> dict[str, Any]:
    project = _normalize_project(project)
    try:
        limit_int = max(1, min(int(limit or 50), 200))
    except (TypeError, ValueError):
        return {"status": "error", "message": "limit 必须是整数。", "followups": []}
    rows = [row for row in _load(project) if isinstance(row, dict)]
    if not include_done:
        rows = [row for row in rows if str(row.get("status") or "").lower() not in TERMINAL_STATES]
    rows.sort(key=lambda item: str(item.get("due_at") or ""))
    return {
        "status": "ok",
        "path": str(_followup_path(project)),
        "count": len(rows[:limit_int]),
        "followups": rows[:limit_int],
    }


def due_followups(*, project: str | None = None, now: str | None = None, limit: int = 20) -> dict[str, Any]:
    project = _normalize_project(project)
    now_dt = _parse_dt(now) or _now()
    try:
        limit_int = max(1, min(int(limit or 20), 100))
    except (TypeError, ValueError):
        return {"status": "error", "message": "limit 必须是整数。", "followups": []}
    due: list[dict[str, Any]] = []
    for row in _load(project):
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").lower() in TERMINAL_STATES:
            continue
        due_at = _parse_dt(row.get("due_at"))
        if due_at is not None and due_at <= now_dt:
            item = dict(row)
            item["evidence_goals"] = _evidence_goals(item)
            due.append(item)
    due.sort(key=lambda item: str(item.get("due_at") or ""))
    return {
        "status": "ok",
        "path": str(_followup_path(project)),
        "now": now_dt.isoformat(timespec="seconds"),
        "count": len(due[:limit_int]),
        "followups": due[:limit_int],
        "guidance": "这些是到期的科研 follow-up。模型应把它们当作待验证意图，自主选择 job/status/log/result/research 工具补证据。",
    }


def complete_followup(
    followup_id: str,
    *,
    project: str | None = None,
    status: str = "done",
    note: str | None = None,
    reschedule: bool = True,
) -> dict[str, Any]:
    project = _normalize_project(project)
    fid = str(followup_id or "").strip()
    if not fid:
        return {"status": "error", "message": "followup_id 不能为空。"}
    rows = [row for row in _load(project) if isinstance(row, dict)]
    target: dict[str, Any] | None = None
    now = _now()
    for row in rows:
        if str(row.get("id") or "") == fid:
            target = row
            break
    if target is None:
        return {"status": "missing", "message": f"未找到 follow-up: {fid}", "path": str(_followup_path(project))}
    interval = target.get("interval_minutes")
    if reschedule and interval:
        try:
            minutes = max(1, min(int(interval), 60 * 24 * 365))
        except (TypeError, ValueError):
            minutes = 0
        if minutes:
            target["due_at"] = (now + timedelta(minutes=minutes)).isoformat(timespec="seconds")
            target["last_completed_at"] = now.isoformat(timespec="seconds")
            target["last_note"] = str(note or "")
            target["status"] = "scheduled"
            target["updated_at"] = now.isoformat(timespec="seconds")
            _save(rows, project)
            return {"status": "rescheduled", "path": str(_followup_path(project)), "followup": target}
    target["status"] = str(status or "done").strip() or "done"
    target["completed_at"] = now.isoformat(timespec="seconds")
    target["completion_note"] = str(note or "")
    target["updated_at"] = now.isoformat(timespec="seconds")
    _save(rows, project)
    return {"status": "ok", "path": str(_followup_path(project)), "followup": target}


def _evidence_goals(item: dict[str, Any]) -> list[dict[str, Any]]:
    goals: list[dict[str, Any]] = []
    if item.get("related_job_id"):
        goals.append(
            {
                "goal": "refresh_related_job",
                "evidence_needed": "scheduler state and recent log/progress evidence for the related job",
                "candidate_tools": ["job_watch_snapshot", "cluster_job_status_brief", "cluster_job_tail_log", "cluster_job_progress_estimate"],
            }
        )
    if item.get("related_run_id"):
        goals.append(
            {
                "goal": "inspect_related_run",
                "evidence_needed": "local run report, VASP output scan, and result interpretation if outputs exist",
                "candidate_tools": ["dft_run_report", "vasp_output_scan", "result_interpret"],
            }
        )
    if not goals:
        goals.append(
            {
                "goal": "decide_needed_evidence",
                "evidence_needed": "project/session/research context sufficient to decide what to check next",
                "candidate_tools": ["project_continuity_digest", "research_onboarding_context", "aether_capability_map"],
            }
        )
    return goals
