from __future__ import annotations

"""Volatile prompt digests for live cluster/research context."""

import json
import os
import re
from pathlib import Path

from .research_workspace import resolve_research_project


def _normalized(value: str) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _slug_tokens(value: str) -> set[str]:
    raw = str(value or "").strip().lower()
    return {
        token
        for token in re.split(r"[^a-z0-9\u4e00-\u9fff]+", raw)
        if token
    }


def _record_matches_project(store: object, rec: dict, project: str | None) -> bool:
    if not project:
        return True
    resolved = resolve_research_project(project)
    project_names = {str(project)}
    if resolved:
        project_names.add(resolved.slug)
    needle_tokens: set[str] = set()
    for name in project_names:
        needle_tokens.update(_slug_tokens(name))
    haystacks = [str(rec.get("task_id") or ""), str(rec.get("run_id") or ""), str(rec.get("run_root") or "")]
    run_root = rec.get("run_root")
    if run_root and hasattr(store, "load_run_record"):
        try:
            full = store.load_run_record(Path(run_root))
            haystacks.extend(str(item) for item in getattr(full, "tags", []) or [])
            notes = getattr(full, "notes", {}) or {}
            haystacks.append(json.dumps(notes, ensure_ascii=False, default=str))
        except Exception:
            pass
    haystack_tokens: set[str] = set()
    for value in haystacks:
        haystack_tokens.update(_slug_tokens(value))
        path = Path(str(value))
        for part in path.parts:
            haystack_tokens.update(_slug_tokens(part))
    return bool(needle_tokens and needle_tokens.intersection(haystack_tokens))


def build_cluster_runtime_digest(*, project: str | None = None, limit: int = 5) -> str:
    """Summarize locally known active remote jobs without doing live SSH."""

    try:
        from dft_app.storage import RecordStore
    except Exception:
        return ""
    try:
        store = RecordStore(Path.cwd())
        runs = store.list_runs(limit=80)
    except Exception:
        return ""
    rows: list[str] = []
    for rec in runs:
        if not _record_matches_project(store, rec, project):
            continue
        job_id = str(rec.get("scheduler_job_id") or "").strip()
        status = str(rec.get("overall_status") or "").strip()
        phase = str(rec.get("current_phase") or "").strip()
        if not job_id or status.lower() in {"completed", "failed", "cancelled"}:
            continue
        rows.append(
            f"- job `{job_id}` task `{rec.get('task_id')}` run `{rec.get('run_id')}` "
            f"status={status or 'unknown'} phase={phase or 'unknown'}"
        )
        if len(rows) >= limit:
            break
    if not rows:
        scope = f" for project `{project}`" if project else ""
        return (
            f"No AETHER-recorded active/partial cluster jobs in the local run store{scope}. "
            "This is not live scheduler evidence and does not prove the user's cluster queue is empty. "
            "If the user asks status, use `cluster_my_jobs` or a specific job realtime tool before claiming live job state."
        )
    return (
        "AETHER-recorded active/partial cluster jobs from the local run store "
        "(not live scheduler evidence; use realtime cluster tools before claiming current state):\n"
        + "\n".join(rows)
    )


def build_job_watch_digest(*, project: str | None = None, limit: int = 5) -> str:
    """Summarize AETHER-submitted Slurm jobs remembered by the local watcher.

    This is intentionally local-only and does not SSH.  It gives the model a
    cheap resume hook for ambiguous references to prior AETHER-submitted work
    while keeping live cluster checks explicit through ``job_watch_snapshot``
    with ``live_check=true`` or the realtime cluster tools.
    """

    try:
        from .job_watcher import snapshot
    except Exception:
        return ""
    payload = snapshot(live_check=False, limit=max(limit * 4, limit))
    if payload.get("status") not in {"ok", "partial"}:
        return ""
    rows: list[str] = []
    for job in payload.get("jobs") or []:
        if project and not _job_watch_matches_project(job, project):
            continue
        bits = [
            f"job `{job.get('job_id')}`",
            f"task `{job.get('task_id') or 'unknown'}`",
            f"run `{job.get('run_id') or 'unknown'}`",
            f"state={job.get('last_known_state') or 'unknown'}",
        ]
        if job.get("cluster_alias"):
            bits.append(f"cluster={job.get('cluster_alias')}")
        if job.get("remote_run_root"):
            bits.append(f"remote={job.get('remote_run_root')}")
        followups = job.get("followup_options") or []
        if followups:
            goals = [str(item.get("goal") or "") for item in followups if isinstance(item, dict) and item.get("goal")]
            if goals:
                bits.append("followup_goals=" + ",".join(goals[:4]))
        rows.append("- " + " ".join(bits))
        if len(rows) >= limit:
            break
    if not rows:
        scope = f" for project `{project}`" if project else ""
        return (
            f"No locally watched AETHER-submitted jobs{scope}. "
            "The model should decide whether the user needs queue evidence, session context, or a remembered AETHER submission before choosing tools."
        )
    return (
        "AETHER job watcher remembers these submitted jobs (local index; not live SSH):\n"
        + "\n".join(rows)
        + "\nTreat followup_goals as optional evidence goals, not a fixed workflow; choose tools only when the user's intent requires that evidence."
    )


def _job_watch_matches_project(job: dict, project: str) -> bool:
    resolved = resolve_research_project(project)
    names = {str(project)}
    if resolved:
        names.add(resolved.slug)
    needle_tokens: set[str] = set()
    for name in names:
        needle_tokens.update(_slug_tokens(name))
    haystack_tokens: set[str] = set()
    for key in ("task_id", "run_id", "run_root", "remote_run_root", "job_script", "cluster_alias"):
        value = str(job.get(key) or "")
        haystack_tokens.update(_slug_tokens(value))
        path = Path(value)
        for part in path.parts:
            haystack_tokens.update(_slug_tokens(part))
    return bool(needle_tokens and needle_tokens.intersection(haystack_tokens))


def build_followup_digest(*, project: str | None = None, limit: int = 5) -> str:
    """Summarize due/upcoming project follow-ups without running their actions."""

    try:
        from .followups import due_followups, list_followups
    except Exception:
        return ""
    due = due_followups(project=project, limit=limit)
    upcoming = list_followups(project=project, limit=limit)
    rows: list[str] = []
    if due.get("followups"):
        rows.append("Due research follow-ups (intent only; gather evidence before answering):")
        for item in due.get("followups") or []:
            bits = [
                f"id=`{item.get('id')}`",
                f"title={item.get('title') or 'untitled'}",
                f"due_at={item.get('due_at')}",
            ]
            if item.get("related_job_id"):
                bits.append(f"job={item.get('related_job_id')}")
            if item.get("related_run_id"):
                bits.append(f"run={item.get('related_run_id')}")
            goals = item.get("evidence_goals") or []
            if goals:
                bits.append("evidence_goals=" + ",".join(str(goal.get("goal") or "") for goal in goals[:3] if isinstance(goal, dict)))
            rows.append("- " + " ".join(bits))
    else:
        followups = upcoming.get("followups") or []
        if followups:
            rows.append("Upcoming research follow-ups:")
            for item in followups[:limit]:
                rows.append(f"- id=`{item.get('id')}` title={item.get('title') or 'untitled'} due_at={item.get('due_at')} status={item.get('status') or 'scheduled'}")
    if not rows:
        return ""
    rows.append("Treat follow-ups as reminders/check intents, not facts or fixed workflows; decide evidence tools from the user goal.")
    return "\n".join(rows)


def build_auto_mode_digest(*, project: str | None = None) -> str:
    try:
        from .auto_mode import build_auto_mode_digest as _build
    except Exception:
        return ""
    try:
        return _build(project=project)
    except Exception:
        return ""


def build_research_workspace_digest(*, project: str | None = None, max_chars: int = 1800) -> str:
    paths = resolve_research_project(project)
    if paths is None:
        return ""
    lines: list[str] = [f"Research project: `{paths.slug}`", f"Root: `{paths.root}`"]
    learning = paths.root / "Learning"
    if learning.exists():
        titles = [item.stem for item in sorted(learning.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]]
        if titles:
            lines.append("Recent Learning notes: " + "；".join(titles))
    if paths.progress.exists():
        text = paths.progress.read_text(encoding="utf-8", errors="replace")
        lines.append("Progress tail:\n" + text[-max_chars:].strip())
    return "\n".join(lines)[:max_chars]


def _preload_semantic_priors_enabled() -> bool:
    raw = os.getenv("AETHER_DFT_PRELOAD_SEMANTIC_PRIORS", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_relevant_priors_digest(*, project: str | None = None, query: str | None = None, max_items: int = 3) -> str:
    if not query:
        return ""
    try:
        from .knowledge import search_for_system, search_notes
    except Exception:
        return ""
    max_items = max(1, min(int(max_items or 3), 8))
    semantic = _preload_semantic_priors_enabled()
    try:
        result = search_for_system(
            extra_terms=[query],
            project_priority=project,
            max_results=max_items,
            semantic=semantic,
        )
    except Exception:
        try:
            notes = search_notes(project or "", query)[:max_items]
        except Exception:
            return ""
        rows = []
        for note in notes:
            rows.append(f"- {note.get('title') or note.get('path')}: {str(note.get('excerpt') or '')[:160]}")
        return "\n".join(rows)

    matches = result.get("matches") or []
    if not matches:
        return ""
    method = str(result.get("selection_method") or ("semantic" if semantic else "lexical"))
    rows = [
        (
            f"Relevant project/research priors ({method} preload; lightweight, not exhaustive). "
            "If this is insufficient, call `knowledge_search_for_system` explicitly with the actual material/adsorbate; "
            "prioritize warnings/gotchas/避坑 over ordinary API notes."
        )
    ]
    for item in matches[:max_items]:
        title = item.get("title") or item.get("path") or "untitled"
        source = item.get("source") or "knowledge"
        project_slug = item.get("project")
        reason = item.get("semantic_reason")
        preview = str(item.get("preview") or "")[:180].replace("\n", " ")
        meta = f"{source}"
        if project_slug:
            meta += f"; project={project_slug}"
        if reason:
            meta += f"; reason={str(reason)[:80]}"
        rows.append(f"- {title} ({meta}): {preview}")
    return "\n".join(rows)
