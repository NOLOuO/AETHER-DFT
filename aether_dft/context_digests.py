from __future__ import annotations

"""Volatile prompt digests for live cluster/research context."""

from pathlib import Path
import json
import re

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
        return f"No locally recorded active cluster jobs{scope}. If the user asks status, use `cluster_my_jobs` or a specific job realtime tool."
    return "Locally recorded active/partial cluster jobs:\n" + "\n".join(rows)


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


def build_relevant_priors_digest(*, project: str | None = None, query: str | None = None, max_items: int = 3) -> str:
    if not query:
        return ""
    try:
        from .knowledge import search_notes
    except Exception:
        return ""
    try:
        notes = search_notes(project or "", query)[:max_items]
    except Exception:
        return ""
    rows = []
    for note in notes:
        rows.append(f"- {note.get('title') or note.get('path')}: {str(note.get('excerpt') or '')[:160]}")
    return "\n".join(rows)
