from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

from .followups import complete_followup, due_followups, list_followups, schedule_followup
from .paths import ensure_runtime_dir
from .project_state import project_paths


DEFAULT_MONITOR_INTERVAL_HOURS = 4
DEFAULT_DAILY_REPORT_TIME = "18:00"
AUTO_COMPUTATIONAL_STRATEGY = (
    "Computational strategy: do not over-invest in hand-perfecting a single model or structure. "
    "When the search space is uncertain, enumerate a diverse candidate set, run cheap sanity/quality filters, "
    "batch-submit the scientifically plausible candidates when allowed, then prune and refine from calculated evidence. "
    "Human time is scarce; compute is the lever."
)


def _now() -> datetime:
    return datetime.now().astimezone()


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")


def _normalize_project(project: Any) -> str | None:
    text = str(project or "").strip()
    return text or None


def _collapse(value: Any, *, limit: int = 260) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _extract_goal_from_text(text: str) -> str:
    """Extract a project goal from existing project/session evidence.

    This is structural evidence extraction, not natural-language command
    routing.  It looks for explicit goal/focus fields first, then falls back to
    the first substantive project/session statement.
    """

    lines = [line.strip(" \t#-*•") for line in str(text or "").splitlines()]
    explicit_prefixes = (
        "research_goal",
        "goal",
        "objective",
        "current_focus",
        "研究目标",
        "科研目标",
        "目标",
        "当前重点",
        "当前问题",
    )
    for line in lines:
        if not line:
            continue
        lowered = line.lower()
        for prefix in explicit_prefixes:
            if lowered.startswith(prefix.lower()):
                parts = line.split(":", 1) if ":" in line else line.split("：", 1)
                value = parts[1] if len(parts) > 1 else line
                value = value.strip(" -：:")
                if len(value) >= 8:
                    return _collapse(value)
    for line in lines:
        lowered = line.lower()
        if lowered in {"session context", "recent turns", "project state markdown", "project metadata", "new research chat"}:
            continue
        if len(line) >= 12 and not lowered.startswith(
            (
                "status",
                "created",
                "updated",
                "session_id",
                "project",
                "turn_count",
                "model_usable_context",
                "session_context",
                "this context",
            )
        ):
            return _collapse(line)
    return ""


def infer_research_goal(
    *,
    project: str | None = None,
    session_store: Any | None = None,
    session_id: str | None = None,
    max_chars: int = 8000,
) -> dict[str, Any]:
    """Infer an auto-mode goal from project files and existing conversation.

    `/auto` is a switch, so the normal path should not force the human to
    retype a goal already present in project state or chat history.
    """

    evidence: list[dict[str, str]] = []
    project = _normalize_project(project)
    if project:
        try:
            from .project_state import read_project_context_digest

            text = read_project_context_digest(project)
            if text:
                evidence.append({"source": "project_context_digest", "text": text[:max_chars]})
        except Exception:
            pass
        try:
            paths = project_paths(project)
            for name, path in (
                ("project_metadata", paths.metadata),
                ("project_state", paths.state),
                ("project_state_md", paths.state_md),
                ("project_progress", paths.progress),
            ):
                if path.exists():
                    evidence.append({"source": name, "text": path.read_text(encoding="utf-8", errors="replace")[:max_chars]})
        except Exception:
            pass
    if session_store is not None:
        if session_id and hasattr(session_store, "load_state"):
            try:
                state = session_store.load_state(session_id)
                text = "\n".join(
                    str(state.get(key) or "")
                    for key in ("title", "first_prompt", "last_response")
                    if str(state.get(key) or "").strip()
                )
                if text:
                    evidence.append({"source": "current_session_state", "text": text})
            except Exception:
                pass
        if session_id and hasattr(session_store, "build_session_context"):
            try:
                text = session_store.build_session_context(session_id, max_chars=max_chars)
                if text:
                    evidence.append({"source": "current_session_context", "text": text})
            except Exception:
                pass
        if hasattr(session_store, "list_sessions"):
            try:
                for item in session_store.list_sessions(project=project, limit=5):
                    parts = [
                        getattr(item, "title", ""),
                        getattr(item, "first_prompt", ""),
                        getattr(item, "last_response", ""),
                        getattr(item, "pending_prompt", ""),
                    ]
                    text = "\n".join(str(part or "") for part in parts if str(part or "").strip())
                    if text:
                        evidence.append({"source": f"session_summary:{getattr(item, 'session_id', '')}", "text": text})
            except Exception:
                pass
    for item in evidence:
        candidate = _extract_goal_from_text(item["text"])
        if candidate:
            return {
                "status": "ok",
                "goal": candidate,
                "source": item["source"],
                "evidence_sources": [entry["source"] for entry in evidence],
            }
    return {
        "status": "empty",
        "goal": "",
        "source": "",
        "evidence_sources": [entry["source"] for entry in evidence],
    }


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
            "It does not hard-code a pipeline: the model decides whether literature, structure, cluster, analysis, writeback, or a human question is needed next. "
            "Default bias: convert uncertainty into candidate sets and let calculations filter them."
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
    if "initial_advance" not in kinds:
        schedule_followup(
            project=project,
            title="Auto initial advance",
            prompt=(
                "AUTO MODE initial advance. Research goal: "
                f"{goal}. Start from current project/session evidence and take the first useful autonomous scientific action. "
                "Do not wait for a manual tick. If the goal is broad, create or inspect a multi-candidate campaign, gather immediate evidence, "
                "build safe local structures when useful, and ask the human only if a blocking ambiguity or permission issue remains."
            ),
            due_at=_now_iso(),
            metadata={"auto_mode": True, "auto_kind": "initial_advance"},
        )
    if "monitor" not in kinds:
        schedule_followup(
            project=project,
            title="Auto monitor",
            prompt=(
                "AUTO MODE periodic check. Research goal: "
                f"{goal}. Inspect project/session/follow-up/job evidence, decide whether to search literature, build/check structures, "
                "monitor/fetch/analyze calculations, write back learning, or ask the human one blocking question. "
                "Prefer broad candidate enumeration plus cheap filtering over hand-perfecting a single model when uncertainty remains."
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
                "candidate-space coverage, blockers/questions for the human, and the next autonomous focus. Goal: "
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
            "computational_strategy": {
                "principle": "Human supplies the end-to-end research objective; AI turns uncertainty into computable candidate spaces.",
                "default_bias": "Enumerate diverse plausible candidates, cheaply filter, batch calculate, then refine from results.",
                "state_board": "Use auto_campaign_* tools to persist candidates, quality filters, run/job bindings, parsed results, next batches, and pruning decisions.",
                "avoid": "Do not spend human/model effort trying to hand-perfect one structure before using computation unless the candidate space is genuinely tiny.",
                "ask_human_only_for": "Goal ambiguity, costly branch choice, missing permissions/credentials, or irreversible/destructive action.",
            },
        },
    }
    if include_due:
        payload["due_followups"] = due_followups(project=state.get("project") or project, limit=10)
        payload["scheduled_followups"] = list_followups(project=state.get("project") or project, limit=10)
        try:
            from .auto_campaign import list_campaigns

            payload["active_campaigns"] = list_campaigns(project=state.get("project") or project or "", include_closed=False, limit=5)
        except Exception as exc:
            payload["active_campaigns"] = {"status": "error", "message": str(exc), "campaigns": []}
    return payload


def collect_due_auto_intents(
    *,
    project: str | None = None,
    now: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Return a single model prompt for due autonomous work.

    This is the important distinction for autonomous mode:
    the user does not manually advance a fixed step.  The runtime checks the
    durable follow-up queue and, only when something is due, asks the model to
    decide the smallest next scientific action from the available evidence.
    """

    state = load_auto_state(project)
    if not state.get("enabled"):
        return {
            "status": "disabled",
            "should_run": False,
            "state": state,
            "followups": [],
            "followup_ids": [],
            "prompt": "",
        }
    project_name = state.get("project") or project
    due = due_followups(project=project_name, now=now, limit=limit)
    if due.get("status") != "ok":
        return {
            "status": "error",
            "should_run": False,
            "state": state,
            "followups": [],
            "followup_ids": [],
            "prompt": "",
            "message": due.get("message") or "无法读取 due follow-ups。",
        }
    followups = [item for item in (due.get("followups") or []) if isinstance(item, dict)]
    if not followups:
        return {
            "status": "idle",
            "should_run": False,
            "state": state,
            "followups": [],
            "followup_ids": [],
            "prompt": "",
        }
    safe_followups = []
    for item in followups:
        safe_followups.append(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "prompt": item.get("prompt"),
                "due_at": item.get("due_at"),
                "interval_minutes": item.get("interval_minutes"),
                "related_job_id": item.get("related_job_id"),
                "related_run_id": item.get("related_run_id"),
                "evidence_goals": item.get("evidence_goals") or [],
                "metadata": item.get("metadata") or {},
            }
        )
    prompt = (
        "[execution-mode]\n"
        "AUTO MODE DUE WORK: /auto is enabled for this project. The human should not have to type a manual command.\n"
        "Use tools to inspect project/session/research/job evidence, then decide the smallest useful next scientific action toward the research goal.\n"
        "Do not follow a fixed pipeline. Literature search, structure building, cluster submission, monitoring, parsing, writeback, or asking one human question are all optional and evidence-driven.\n"
        f"{AUTO_COMPUTATIONAL_STRATEGY}\n"
        "Campaign state board: first inspect auto_campaign_status/list for this project; start one if the goal needs multi-candidate exploration and none exists. Register generated candidates, bind run_id/job_id after build/submit, update results after monitor/fetch/parse, and use prune_plan/next_batch to manage compute resources.\n"
        "Operational bias: if several adsorption sites, orientations, conformers, spin/charge states, coverages, or pathways are plausible, build/rank a candidate set instead of arguing for one perfect choice. Use preflight/quality checks to avoid obviously bad jobs, then let DFT results prune.\n"
        "If a cluster submission is scientifically necessary, respect permission/auto state and submit only when allowed. If blocked, ask exactly one concise human question.\n"
        "Before finishing, call auto_mode_checkpoint with observation, decision, evidence_refs, next_focus, and any human_questions.\n\n"
        f"Project: {project_name or 'none'}\n"
        f"Research goal: {state.get('research_goal') or ''}\n"
        f"Auto status: {state.get('status') or 'idle'}\n"
        f"Allowed actions: literature={state.get('allow_literature_search')}, structure_build={state.get('allow_structure_build')}, "
        f"cluster_submit={state.get('allow_cluster_submit')}, research_writeback={state.get('allow_research_writeback')}\n"
        "Due intents JSON:\n"
        f"{json.dumps(safe_followups, ensure_ascii=False, indent=2)}"
    )
    return {
        "status": "ok",
        "should_run": True,
        "state": state,
        "followups": safe_followups,
        "followup_ids": [str(item.get("id") or "") for item in safe_followups if item.get("id")],
        "prompt": prompt,
    }


def complete_due_auto_intents(
    *,
    project: str | None = None,
    followup_ids: list[str] | None = None,
    note: str | None = None,
    reschedule: bool = True,
) -> dict[str, Any]:
    """Mark due autonomous intents as processed after a model turn.

    Recurring monitor/daily-report follow-ups are rescheduled; one-off
    follow-ups are completed.  This keeps the background loop from repeatedly
    firing the same intent after the model has already inspected it.
    """

    ids = [str(item or "").strip() for item in (followup_ids or []) if str(item or "").strip()]
    results = []
    for followup_id in ids:
        results.append(
            complete_followup(
                followup_id,
                project=project,
                note=note or "Processed by /auto background loop.",
                reschedule=reschedule,
            )
        )
    return {"status": "ok", "count": len(results), "results": results}


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
        AUTO_COMPUTATIONAL_STRATEGY,
        "Default execution pattern under uncertainty: enumerate candidates → cheap quality/preflight filters → batch calculations when allowed → parse/prune/refine → report only decisions and blockers.",
        "Use auto_campaign_* as the project state board for multi-candidate campaigns: register candidates, bind runs/jobs, track results, choose next batch, and prune.",
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
