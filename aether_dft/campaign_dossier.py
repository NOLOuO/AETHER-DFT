from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .project_state import read_scientific_project_state
from .session_store import AetherSessionStore


SHAREABLE_ARTIFACT_NAMES = {
    "INCAR",
    "KPOINTS",
    "POSCAR",
    "CONTCAR",
    "OUTCAR",
    "OSZICAR",
    "job.slurm",
    "run_record.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    licensed = path.name.upper() == "POTCAR"
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "licensed_content_excluded": licensed,
        "share_policy": "metadata-only" if licensed else "hash-and-path",
    }


def _iter_campaign_artifacts(run_roots: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_root in run_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            records.append({"path": str(root), "status": "missing"})
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            lower = path.name.lower()
            if path.name in SHAREABLE_ARTIFACT_NAMES or path.name.upper() == "POTCAR" or lower.startswith("slurm"):
                records.append(_artifact_record(path))
    return records


def _tool_events(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(turns, start=1):
        record = dict(turn.get("record") or turn)
        for tool in record.get("tool_executions") or []:
            if not isinstance(tool, dict):
                continue
            result = tool.get("result") if isinstance(tool.get("result"), dict) else {}
            events.append(
                {
                    "turn": turn_index,
                    "name": str(tool.get("name") or ""),
                    "arguments": tool.get("arguments") or {},
                    "status": str(result.get("status") or "unknown"),
                    "job_id": str(result.get("job_id") or result.get("scheduler_job_id") or ""),
                    "remote_run_root": str(result.get("remote_run_root") or ""),
                    "persisted_output_path": str(tool.get("persisted_output_path") or ""),
                    "human_approval": result.get("human_approval") if isinstance(result.get("human_approval"), dict) else None,
                }
            )
    return events


def build_campaign_dossier(
    *,
    project: str,
    session_id: str,
    run_roots: list[str | Path],
    output_dir: str | Path,
    session_store: AetherSessionStore | None = None,
) -> dict[str, Any]:
    """Build a non-destructive, auditable campaign evidence index.

    The dossier references original files by absolute path and digest. It never
    copies POTCAR content and marks missing evidence instead of inventing it.
    """

    store = session_store or AetherSessionStore()
    turns = store.read_transcript(session_id, limit=10000)
    state = read_scientific_project_state(project)
    artifacts = _iter_campaign_artifacts(run_roots)
    tools = _tool_events(turns)
    submit_events = [
        item
        for item in tools
        if item["name"] == "cluster_remote_submit"
        or (
            item["name"] == "dft_run_task"
            and str((item.get("arguments") or {}).get("execution_mode") or "").lower() == "remote_submit"
        )
    ]
    approved_submit_events = [
        item for item in submit_events if bool((item.get("human_approval") or {}).get("granted"))
    ]
    scheduler_events = [
        item
        for item in tools
        if item["name"]
        in {"cluster_job_status_brief", "cluster_my_jobs", "cluster_remote_monitor", "cluster_remote_fetch"}
    ]
    output_evidence = [
        item for item in artifacts if item.get("name") in {"OUTCAR", "OSZICAR", "CONTCAR"}
    ]
    checklist = {
        "session_exists": bool(turns),
        "scientific_goal_present": bool((state.get("state") or {}).get("research_goal")),
        "structured_decision_present": bool((state.get("state") or {}).get("decisions")),
        "submit_event_present": bool(submit_events),
        "submit_authorization_present": bool(approved_submit_events),
        "scheduler_evidence_present": bool(scheduler_events),
        "vasp_output_evidence_present": bool(output_evidence),
        "all_run_roots_exist": all(item.get("status") != "missing" for item in artifacts),
        "potcar_content_excluded": all(
            bool(item.get("licensed_content_excluded"))
            for item in artifacts
            if str(item.get("name") or "").upper() == "POTCAR"
        ),
    }
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project": project,
        "session_id": session_id,
        "run_roots": [str(Path(item).expanduser().resolve()) for item in run_roots],
        "scientific_state": state,
        "turn_count": len(turns),
        "tool_events": tools,
        "artifacts": artifacts,
        "checklist": checklist,
        "complete": all(checklist.values()),
        "limitations": [
            key for key, value in checklist.items() if not value
        ],
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "campaign_dossier.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    markdown = [
        f"# Campaign dossier: {project}",
        "",
        f"- session: `{session_id}`",
        f"- turns: {len(turns)}",
        f"- complete: `{payload['complete']}`",
        "",
        "## Evidence checklist",
        "",
    ]
    markdown.extend(f"- [{'x' if value else ' '}] {key}" for key, value in checklist.items())
    markdown.extend(
        [
            "",
            "## Artifact policy",
            "",
            "POTCAR content is never copied; only path, size and SHA-256 metadata are recorded.",
        ]
    )
    markdown_path = target / "campaign_dossier.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "complete": payload["complete"],
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "limitations": payload["limitations"],
    }
