from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

from .followups import due_followups, list_followups, schedule_followup
from .paths import ensure_runtime_dir
from .project_state import project_paths


DEFAULT_MONITOR_INTERVAL_HOURS = 4
DEFAULT_DAILY_REPORT_TIME = "18:00"


def _now() -> datetime:
    return datetime.now().astimezone()


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")


def _normalize_project(project: Any) -> str | None:
    text = str(project or "").strip()
    return text or None


def _next_daily_due(time_text: Any) -> str:
    text = str(time_text or DEFAULT_DAILY_REPORT_TIME).strip()
    hour = 18
    minute = 0
    try:
        parts = text.split(":", 2)
        hour = max(0, min(int(parts[0]), 23))
        if len(parts) > 1:
            minute = max(0, min(int(parts[1]), 59))
    except Exception:
        hour, minute = 18, 0
    now = _now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return target.isoformat(timespec="seconds")


def auto_state_path(project: str | None = None) -> Path:
    project = _normalize_project(project)
    if project:
        root = project_paths(project).root / ".aether"
    else:
        root = ensure_runtime_dir("auto_mode")
    root.mkdir(parents=True, exist_ok=True)
    return root / "auto_mode.json"


def _default_state(project: str | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "enabled": False,
        "project": _normalize_project(project),
        "research_goal": "",
        "status": "idle",
        "monitor_interval_hours": DEFAULT_MONITOR_INTERVAL_HOURS,
        "daily_report_time": DEFAULT_DAILY_REPORT_TIME,
        "allow_literature_search": True,
        "allow_structure_build": True,
        "allow_cluster_submit": False,
        "allow_research_writeback": True,
        "human_questions": [],
        "open_questions": [],
        "last_checkpoint": {},
        "created_at": "",
        "updated_at": "",
    }


def load_auto_state(project: str | None = None) -> dict[str, Any]:
    path = auto_state_path(project)
    state = _default_state(project)
    if not path.exists():
        return state
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return state
    if isinstance(data, dict):
        state.update(data)
    state["project"] = _normalize_project(state.get("project") or project)
    return state


def save_auto_state(state: dict[str, Any], *, project: str | None = None) -> dict[str, Any]:
    resolved_project = _normalize_project(project or state.get("project"))
    state = {**_default_state(resolved_project), **state}
    state["project"] = resolved_project
    state["updated_at"] = _now_iso()
    if not state.get("created_at"):
        state["created_at"] = state["updated_at"]
    path = auto_state_path(resolved_project)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "path": str(path), "state": state}


def configure_auto_mode(
    *,
    project: str | None = None,
    enabled: bool,
    research_goal: str | None = None,
    monitor_interval_hours: int | None = None,
    daily_report_time: str | None = None,
    allow_cluster_submit: bool | None = None,
    allow_structure_build: bool | None = None,
    allow_literature_search: bool | None = None,
    allow_research_writeback: bool | None = None,
    reset_questions: bool = False,
) -> dict[str, Any]:
    state = load_auto_state(project)
    goal = str(research_goal if research_goal is not None else state.get("research_goal") or "").strip()
    if enabled and not goal:
        return {
            "status": "needs_goal",
            "message": "开启 /auto 需要明确 research_goal。人类只需给目标；不清楚的地方 AI 后续会问。",
            "state": state,
        }
    state["enabled"] = bool(enabled)
    state["status"] = "active" if enabled else "paused"
    if goal:
        state["research_goal"] = goal
    if monitor_interval_hours is not None:
        try:
            state["monitor_interval_hours"] = max(1, min(int(monitor_interval_hours), 24 * 30))
        except (TypeError, ValueError):
            return {"status": "error", "message": "monitor_interval_hours 必须是整数。", "state": state}
    if daily_report_time is not None:
        state["daily_report_time"] = str(daily_report_time or DEFAULT_DAILY_REPORT_TIME).strip() or DEFAULT_DAILY_REPORT_TIME
    if allow_cluster_submit is not None:
        state["allow_cluster_submit"] = bool(allow_cluster_submit)
    if allow_structure_build is not None:
        state["allow_structure_build"] = bool(allow_structure_build)
    if allow_literature_search is not None:
        state["allow_literature_search"] = bool(allow_literature_search)
    if allow_research_writeback is not None:
        state["allow_research_writeback"] = bool(allow_research_writeback)
    if reset_questions:
        state["human_questions"] = []
        state["open_questions"] = []
    saved = save_auto_state(state, project=project)
    if enabled:
        _ensure_auto_followups(saved["state"])
    return {
        **saved,
        "guidance": (
            "Auto mode stores the human research goal and periodic evidence intents. "
            "It does not hard-code a pipeline: the model decides whether literature, structure, cluster, analysis, writeback, or a human question is needed next."
        ),
    }


def _ensure_auto_followups(state: dict[str, Any]) -> None:
    project = _normalize_project(state.get("project"))
    goal = str(state.get("research_goal") or "").strip()
    if not goal:
        return
    existing = list_followups(project=project, include_done=False, limit=200).get("followups") or []
    kinds = {
        str((item.get("metadata") or {}).get("auto_kind") or "")
        for item in existing
        if isinstance(item, dict) and (item.get("metadata") or {}).get("auto_mode")
    }
    interval_hours = int(state.get("monitor_interval_hours") or DEFAULT_MONITOR_INTERVAL_HOURS)
    if "monitor" not in kinds:
        schedule_followup(
            project=project,
            title="Auto monitor",
            prompt=(
                "AUTO MODE periodic check. Research goal: "
                f"{goal}. Inspect project/session/follow-up/job evidence, decide whether to search literature, build/check structures, "
                "monitor/fetch/analyze calculations, write back learning, or ask the human one blocking question."
            ),
            interval_minutes=interval_hours * 60,
            metadata={"auto_mode": True, "auto_kind": "monitor"},
        )
    if "daily_report" not in kinds:
        schedule_followup(
            project=project,
            title="Auto daily report",
            prompt=(
                "AUTO MODE daily report. Summarize progress toward the research goal, evidence collected, calculations running/completed, "
                "blockers/questions for the human, and the next autonomous focus. Goal: "
                f"{goal}"
            ),
            due_at=_next_daily_due(state.get("daily_report_time")),
            interval_minutes=24 * 60,
            metadata={"auto_mode": True, "auto_kind": "daily_report", "daily_report_time": state.get("daily_report_time")},
        )


def checkpoint_auto_mode(
    *,
    project: str | None = None,
    status: str | None = None,
    observation: str | None = None,
    decision: str | None = None,
    evidence_refs: list[str] | None = None,
    next_focus: str | None = None,
    open_questions: list[str] | None = None,
    human_questions: list[str] | None = None,
) -> dict[str, Any]:
    state = load_auto_state(project)
    if status:
        state["status"] = str(status)
    if open_questions is not None:
        state["open_questions"] = [str(item).strip() for item in open_questions if str(item).strip()]
    if human_questions is not None:
        state["human_questions"] = [str(item).strip() for item in human_questions if str(item).strip()]
    checkpoint = {
        "updated_at": _now_iso(),
        "observation": str(observation or "").strip(),
        "decision": str(decision or "").strip(),
        "evidence_refs": [str(item).strip() for item in (evidence_refs or []) if str(item).strip()],
        "next_focus": str(next_focus or "").strip(),
    }
    state["last_checkpoint"] = checkpoint
    return save_auto_state(state, project=project)


def auto_mode_status(*, project: str | None = None, include_due: bool = True) -> dict[str, Any]:
    state = load_auto_state(project)
    payload: dict[str, Any] = {
        "status": "ok",
        "path": str(auto_state_path(project)),
        "state": state,
        "policy": {
            "human_role": "Set/adjust the research goal and answer AI questions.",
            "ai_role": "Autonomously gather evidence, choose tools, execute allowed DFT work, monitor, analyze, write back, and report.",
            "ask_human_when": [
                "research goal or success metric is ambiguous",
                "multiple scientifically different branches are plausible and costly",
                "credentials/permissions/cluster submission policy blocks progress",
                "a destructive/irreversible action is needed",
            ],
            "not_fixed_workflow": True,
        },
    }
    if include_due:
        payload["due_followups"] = due_followups(project=state.get("project") or project, limit=10)
        payload["scheduled_followups"] = list_followups(project=state.get("project") or project, limit=10)
    return payload


def build_auto_mode_digest(*, project: str | None = None) -> str:
    status = auto_mode_status(project=project, include_due=True)
    state = status.get("state") or {}
    if not state.get("enabled"):
        return ""
    lines = [
        "Auto mode is ON.",
        f"- project: {state.get('project') or project or 'none'}",
        f"- research_goal: {state.get('research_goal')}",
        f"- status: {state.get('status')}",
        f"- monitor_interval_hours: {state.get('monitor_interval_hours')}",
        f"- daily_report_time: {state.get('daily_report_time')}",
        f"- allow_literature_search: {state.get('allow_literature_search')}",
        f"- allow_structure_build: {state.get('allow_structure_build')}",
        f"- allow_cluster_submit: {state.get('allow_cluster_submit')}",
        f"- allow_research_writeback: {state.get('allow_research_writeback')}",
        "",
        "Autonomy contract: human sets/adjusts the research goal and answers blocking questions; AI decides the next evidence/action step.",
        "Ask the human only for ambiguity, materially branching costly choices, missing credentials/permissions, or destructive/irreversible actions.",
        "Do not follow a fixed literature→structure→submit pipeline; choose the smallest evidence/action loop that advances the goal.",
    ]
    last = state.get("last_checkpoint") if isinstance(state.get("last_checkpoint"), dict) else {}
    if last:
        lines.extend(
            [
                "",
                "Last auto checkpoint:",
                f"- observation: {last.get('observation') or ''}",
                f"- decision: {last.get('decision') or ''}",
                f"- next_focus: {last.get('next_focus') or ''}",
            ]
        )
    questions = state.get("human_questions") or []
    if questions:
        lines.extend(["", "Questions currently blocking/benefiting from human answer:"])
        lines.extend(f"- {item}" for item in questions[:5])
    due = ((status.get("due_followups") or {}).get("followups") or [])[:5]
    if due:
        lines.extend(["", "Due auto/follow-up intents:"])
        for item in due:
            lines.append(f"- {item.get('title') or item.get('id')} due_at={item.get('due_at')} prompt={str(item.get('prompt') or '')[:240]}")
    return "\n".join(lines).strip()
