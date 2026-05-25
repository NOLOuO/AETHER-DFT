from __future__ import annotations

from typing import Any

from .project_state import append_progress
from .recommendations import recommend_next_tasks
from .research_workspace import append_research_progress, resolve_research_project


def _collapse_text(value: str, *, limit: int = 96) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def summarize_research_turn(record: dict[str, Any], *, project: str | None = None) -> dict[str, Any]:
    """Attach project-aware follow-up data to one agent turn.

    The harness returns the factual turn record; this helper turns it into a
    research-progress artifact by adding:
    - a compact completion summary
    - blockers when the loop hit a limit or ended abnormally
    - recommended next steps when a project context exists
    - a persisted project progress entry
    """

    prompt = str(record.get("prompt") or "").strip()
    response = str(record.get("response") or "").strip()
    tool_executions = record.get("tool_executions") or []

    completed: list[str] = []
    if prompt:
        completed.append(f"完成一次科研对话：{_collapse_text(prompt)}")
    if response:
        completed.append(f"模型回复摘要：{_collapse_text(response)}")
    if tool_executions:
        tool_names = sorted({str(item.get("name") or "").strip() for item in tool_executions if str(item.get("name") or "").strip()})
        if tool_names:
            completed.append(f"本轮工具调用：{', '.join(tool_names)}")
    if not completed:
        completed.append("完成一次科研推进。")

    finish_reason = str(record.get("finish_reason") or "").strip()
    blockers: list[str] = []
    if finish_reason == "tool_loop_limit":
        blockers.append("工具调用轮数达到上限，需要继续拆分当前问题。")
    elif finish_reason and finish_reason not in {"stop", "end_turn", "tool_calls"}:
        blockers.append(f"本轮结束原因：{finish_reason}")

    focus = prompt or response or None
    recommendations = recommend_next_tasks(project, focus=focus)
    next_steps: list[str] = []
    for item in recommendations[:3]:
        title = str(item.get("title") or "").strip()
        command = str(item.get("command") or "").strip()
        if title and command:
            next_steps.append(f"{title}；{command}")
        elif title:
            next_steps.append(title)
    if not next_steps:
        next_steps.append("继续基于当前项目上下文推进下一步科研任务。")

    progress_path = None
    research_progress_path = None
    if project:
        progress_path = append_progress(project, completed=completed, blockers=blockers, next_steps=next_steps)
        if resolve_research_project(project) is not None:
            research_result = append_research_progress(project, completed=completed, blockers=blockers, next_steps=next_steps)
            if research_result.get("status") == "ok":
                research_progress_path = str(research_result.get("progress_path") or "")

    payload = dict(record)
    payload["project"] = project
    payload["recommendations"] = recommendations
    payload["progress"] = {
        "completed": completed,
        "blockers": blockers,
        "next_steps": next_steps,
            "progress_path": str(progress_path) if progress_path else None,
            "research_progress_path": research_progress_path,
        }
    if progress_path:
        payload["project_progress_path"] = str(progress_path)
    return payload
