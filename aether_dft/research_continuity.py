from __future__ import annotations

"""Continuity primitives for a long-running computational-chemistry project.

These helpers do not prescribe a pipeline.  They assemble the current evidence
state so the model can decide whether the next move should be discussion,
structure modeling, cluster monitoring, result interpretation, or write-back.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .knowledge import search_notes
from .project_state import append_progress, project_paths, read_project_context, write_project_state
from .research_workspace import read_research_onboarding_context


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clean_list(values: Any) -> list[str]:
    """Coerce model-authored list-ish payloads into a clean string list.

    Real tool calls sometimes send semicolon/newline separated strings for
    fields declared as arrays. Treat that as recoverable input instead of
    failing or iterating over characters.
    """

    if values is None:
        items: list[Any] = []
    elif isinstance(values, str):
        items = re.split(r"[\n;；]+", values)
    elif isinstance(values, (list, tuple, set)):
        items = list(values)
    else:
        items = [values]

    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        text = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", text).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", str(value or "").strip()).strip("-")
    return text[:60] or "cycle"


def _recent_runs(limit: int = 8) -> list[dict[str, Any]]:
    try:
        from dft_app.storage import RecordStore

        return RecordStore(Path.cwd()).list_runs(limit=limit)
    except Exception:
        return []


def _run_matches_project(run: dict[str, Any], project: str | None) -> bool:
    if not project:
        return True
    needle = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", project.lower())
    haystack = " ".join(str(run.get(key) or "") for key in ("task_id", "run_id", "run_root")).lower()
    return bool(needle and needle in re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", haystack))


def _classify_recent_results(recent_results: list[dict[str, Any]] | None) -> dict[str, Any]:
    results = [item for item in (recent_results or []) if isinstance(item, dict)]
    statuses = [str(item.get("verdict") or item.get("status") or "").lower() for item in results]
    has_no_outputs = any(status in {"no_outputs", "missing", "unavailable"} for status in statuses)
    has_unconverged = any(status in {"finished_not_converged", "failed", "error", "blocked"} for status in statuses)
    has_converged = any(status in {"finished_converged", "converged", "ok"} for status in statuses)
    return {
        "count": len(results),
        "has_no_outputs": has_no_outputs,
        "has_unconverged_or_failed": has_unconverged,
        "has_converged_like_evidence": has_converged,
        "statuses": statuses,
    }


def build_project_continuity_digest(
    project: str | None = None,
    *,
    focus: str | None = None,
    recent_results: list[dict[str, Any]] | None = None,
    max_chars: int = 9000,
) -> dict[str, Any]:
    """Read-only handoff digest for the next model turn.

    The output deliberately lists evidence and missing evidence instead of
    forcing a workflow.  The model can use it to choose tools responsibly.
    """

    project_clean = str(project or "").strip() or None
    project_context = read_project_context(project_clean, max_chars=3500) if project_clean else ""
    research = read_research_onboarding_context(project_clean, max_chars=3500)
    notes_query = " ".join(item for item in [focus, "candidate_outcome adsorption failure template"] if item)
    notes = search_notes(project_clean, notes_query)[:5] if project_clean else []
    runs = [run for run in _recent_runs(limit=20) if _run_matches_project(run, project_clean)][:6]
    result_state = _classify_recent_results(recent_results)

    open_loops: list[dict[str, Any]] = []
    suggested_tools: list[str] = []
    if not project_context.strip() and research.get("status") == "empty":
        open_loops.append({"kind": "context", "message": "缺少项目状态/research 入职上下文；先建立或选择 project。"})
        suggested_tools.extend(["project_state_read", "research_onboarding_context", "discussion_state_snapshot"])
    if runs and any(str(run.get("overall_status") or "").lower() in {"running", "pending"} for run in runs):
        open_loops.append({"kind": "cluster", "message": "存在本地记录的 running/pending run；应先查 job 状态或回拉日志。"})
        suggested_tools.extend(["cluster_job_status_brief", "cluster_job_tail_log", "cluster_remote_fetch"])
    if result_state["has_no_outputs"]:
        open_loops.append({"kind": "outputs", "message": "近期结果缺少 OUTCAR/OSZICAR 证据；不能给最终科学结论。"})
        suggested_tools.extend(["cluster_remote_fetch", "result_interpret"])
    if result_state["has_unconverged_or_failed"]:
        open_loops.append({"kind": "diagnosis", "message": "近期结果含失败/未收敛状态；优先做参数/日志诊断，不急着扩展候选。"})
        suggested_tools.extend(["cluster_job_tail_log", "research_vasp_template_resolve", "research_learning_capture"])
    if result_state["has_converged_like_evidence"]:
        open_loops.append({"kind": "writeback", "message": "已有可用结果迹象；应复核结构/能量定义并写回 candidate outcome 或 Learning。"})
        suggested_tools.extend(["structure_displacement_compare", "candidate_outcome_record", "research_learning_capture"])
    if not open_loops:
        open_loops.append({"kind": "next_decision", "message": "没有明显阻塞；模型应按科研目标选择讨论、建模或执行下一步。"})
        suggested_tools.extend(["research_proposal_plan", "structure_modeling_intent_plan", "cluster_execution_intent_plan"])

    digest = {
        "status": "ok",
        "project": project_clean or "",
        "focus": str(focus or ""),
        "generated_at": _now(),
        "evidence": {
            "project_context_present": bool(project_context.strip()),
            "research_context_status": research.get("status"),
            "research_files_read": research.get("files_read", []),
            "knowledge_notes": [{"title": item.get("title"), "path": item.get("path"), "score": item.get("score")} for item in notes],
            "recent_runs": runs,
            "recent_results": result_state,
        },
        "open_loops": open_loops,
        "suggested_tools": list(dict.fromkeys(suggested_tools)),
        "not_a_fixed_program": "这是连续性摘要，不是流程表；模型必须根据证据选择、跳过或回退工具。",
    }
    rendered = json.dumps(digest, ensure_ascii=False, indent=2, default=str)
    if len(rendered) > max_chars:
        digest["truncated"] = True
        digest["digest_text"] = rendered[:max_chars].rstrip() + f"\n...[truncated to {max_chars} chars]"
    return digest


@dataclass(frozen=True)
class ResearchCycleCheckpoint:
    checkpoint_id: str
    project: str
    goal: str
    current_decision: str
    evidence_refs: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_research_cycle_checkpoint(
    *,
    project: str,
    goal: str,
    current_decision: str,
    evidence_refs: list[str] | None = None,
    open_questions: list[str] | None = None,
    blockers: list[str] | None = None,
    next_steps: list[str] | None = None,
    run_ids: list[str] | None = None,
    candidate_ids: list[str] | None = None,
    update_project_state: bool = True,
) -> dict[str, Any]:
    """Persist a flexible research-cycle checkpoint for later continuation."""

    project_clean = str(project or "").strip()
    if not project_clean:
        return {"status": "error", "message": "project 不能为空。"}
    goal_clean = str(goal or "").strip()
    decision_clean = str(current_decision or "").strip()
    if not goal_clean or not decision_clean:
        return {"status": "error", "message": "goal 和 current_decision 都不能为空。"}

    paths = project_paths(project_clean)
    checkpoint_id = f"cycle_{uuid4().hex[:8]}"
    checkpoint = ResearchCycleCheckpoint(
        checkpoint_id=checkpoint_id,
        project=paths.slug,
        goal=goal_clean,
        current_decision=decision_clean,
        evidence_refs=_clean_list(evidence_refs),
        open_questions=_clean_list(open_questions),
        blockers=_clean_list(blockers),
        next_steps=_clean_list(next_steps),
        run_ids=_clean_list(run_ids),
        candidate_ids=_clean_list(candidate_ids),
    )
    directory = paths.root / "cycle_checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{checkpoint_id}-{_slug(goal_clean)}.json"
    payload = checkpoint.to_dict()
    payload["path"] = str(output)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    progress_path = append_progress(
        paths.slug,
        completed=[f"研究循环 checkpoint `{checkpoint_id}`：{decision_clean}"],
        blockers=checkpoint.blockers,
        next_steps=checkpoint.next_steps,
    )
    state_path = None
    if update_project_state:
        state_path = write_project_state(
            paths.slug,
            {
                "current_focus": goal_clean,
                "latest_decision": decision_clean,
                "latest_checkpoint_id": checkpoint_id,
                "evidence_refs": checkpoint.evidence_refs,
                "blockers": checkpoint.blockers,
                "next_steps": checkpoint.next_steps,
                "run_ids": checkpoint.run_ids,
                "candidate_ids": checkpoint.candidate_ids,
            },
        )
    return {
        "status": "ok",
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(output),
        "progress_path": str(progress_path),
        "state_path": str(state_path) if state_path else None,
        "checkpoint": payload,
        "guidance": "checkpoint 只记录当前科研判断与证据；下一轮仍应先读 continuity digest 再自主选工具。",
    }


def audit_evidence_claims(claims: list[Any] | None = None, evidence_items: list[Any] | None = None) -> dict[str, Any]:
    """Check that scientific claims explicitly reference evidence.

    ``claims`` can be strings or objects with ``claim``/``evidence_refs``.
    ``evidence_items`` can be strings or objects with ``id``/``path``/``source``.
    """

    evidence_ids: set[str] = set()
    for item in evidence_items or []:
        if isinstance(item, dict):
            for key in ("id", "path", "source", "note_id", "run_id"):
                if item.get(key):
                    evidence_ids.add(str(item[key]))
        elif str(item).strip():
            evidence_ids.add(str(item).strip())

    audited: list[dict[str, Any]] = []
    unsupported = 0
    for raw in claims or []:
        if isinstance(raw, dict):
            text = str(raw.get("claim") or raw.get("text") or "").strip()
            refs = _clean_list(raw.get("evidence_refs") or raw.get("refs") or [])
            confidence = str(raw.get("confidence") or "").strip() or "unspecified"
        else:
            text = str(raw or "").strip()
            refs = []
            confidence = "unspecified"
        missing_refs = [ref for ref in refs if evidence_ids and ref not in evidence_ids]
        supported = bool(refs) and not missing_refs
        if not supported:
            unsupported += 1
        audited.append(
            {
                "claim": text,
                "evidence_refs": refs,
                "confidence": confidence,
                "supported": supported,
                "missing_refs": missing_refs,
            }
        )
    return {
        "status": "ok",
        "claim_count": len(audited),
        "evidence_count": len(evidence_ids),
        "unsupported_count": unsupported,
        "verdict": "all_supported" if audited and unsupported == 0 else "needs_evidence",
        "claims": audited,
        "guidance": "没有 evidence_refs 的 claim 只能作为假设/下一步，不应写成科研结论。",
    }
