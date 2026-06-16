
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .paths import ensure_runtime_dir


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _watch_dir() -> Path:
    path = ensure_runtime_dir("job_watch")
    path.mkdir(parents=True, exist_ok=True)
    return path


def watch_path() -> Path:
    return _watch_dir() / "jobs.json"


def _load() -> list[dict[str, Any]]:
    path = watch_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _save(rows: list[dict[str, Any]]) -> None:
    watch_path().write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def register_run_record(record: Any, *, cluster_alias: str | None = None) -> dict[str, Any] | None:
    """Register a submitted Slurm job for lightweight later resumption.

    This is not a daemon and does not mutate the cluster.  It gives the chat
    harness a durable local index of jobs it submitted so later natural-language
    turns can reconnect to run_root/job_id without the user copy-pasting IDs.
    """

    job_id = str(getattr(record, "scheduler_job_id", "") or "").strip()
    if not job_id:
        return None
    notes = getattr(record, "notes", {}) if isinstance(getattr(record, "notes", {}), dict) else {}
    remote = notes.get("remote") if isinstance(notes.get("remote"), dict) else {}
    entry = {
        "job_id": job_id,
        "task_id": str(getattr(record, "task_id", "") or ""),
        "run_id": str(getattr(record, "run_id", "") or ""),
        "run_root": str(getattr(record, "run_root", "") or ""),
        "remote_run_root": str(remote.get("remote_run_root") or ""),
        "job_script": str(remote.get("job_script") or ""),
        "cluster_alias": str(cluster_alias or remote.get("ssh_host_alias") or "").strip(),
        "last_known_state": "submitted",
        "registered_at": _now(),
        "updated_at": _now(),
    }
    rows = [row for row in _load() if str(row.get("job_id") or "") != job_id]
    rows.insert(0, entry)
    _save(rows[:200])
    return entry


def update_job_state(job_id: str, *, state: str, details: dict[str, Any] | None = None) -> dict[str, Any] | None:
    rows = _load()
    target: dict[str, Any] | None = None
    for row in rows:
        if str(row.get("job_id") or "") == str(job_id):
            row["last_known_state"] = str(state or "")
            row["last_details"] = details or {}
            row["updated_at"] = _now()
            target = row
            break
    if target is not None:
        _save(rows)
    return target


def _job_followup_options(row: dict[str, Any]) -> list[dict[str, Any]]:
    state = str(row.get("last_known_state") or "").strip().upper()
    options: list[dict[str, Any]] = [
        {
            "goal": "refresh_scheduler_state",
            "when": "the user needs current cluster truth rather than the local watcher snapshot",
            "evidence_needed": "live scheduler state for this job id",
            "candidate_tools": ["job_watch_snapshot(live_check=true)", "cluster_job_status_brief"],
        }
    ]
    if state in {"SUBMITTED", "PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "R"} or not state:
        options.append(
            {
                "goal": "inspect_running_or_queued_job",
                "when": "the job appears queued/running or the state is stale/unknown",
                "evidence_needed": "queue state plus recent stdout/stderr or VASP progress if available",
                "candidate_tools": ["cluster_job_status_brief", "cluster_job_tail_log"],
            }
        )
        if row.get("remote_run_root"):
            options.append(
                {
                    "goal": "estimate_vasp_progress",
                    "when": "remote_run_root exists and the user asks whether the calculation is converging or moving",
                    "evidence_needed": "OSZICAR/OUTCAR tail evidence from the remote run directory",
                    "candidate_tools": ["cluster_job_progress_estimate", "cluster_job_partial_outcar"],
                }
            )
    elif state in {"COMPLETED", "COMPLETE", "DONE", "CD"}:
        if row.get("remote_run_root"):
            options.append(
                {
                    "goal": "recover_completed_result",
                    "when": "the user wants the finished calculation interpreted or brought back locally",
                    "evidence_needed": "remote OUTCAR/OSZICAR/CONTCAR plus local run record",
                    "candidate_tools": ["cluster_job_partial_outcar", "cluster_remote_fetch", "result_interpret"],
                }
            )
        else:
            options.append(
                {
                    "goal": "locate_completed_outputs",
                    "when": "the scheduler says complete but the watcher lacks a remote run root",
                    "evidence_needed": "job status and run metadata sufficient to find outputs",
                    "candidate_tools": ["cluster_job_status_brief", "cluster_remote_fetch"],
                }
            )
    elif state in {"FAILED", "CANCELLED", "CANCELED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "F", "CA"}:
        options.append(
            {
                "goal": "diagnose_stopped_job",
                "when": "the job failed, timed out, was cancelled, or ended abnormally",
                "evidence_needed": "scheduler terminal state plus log tail/error messages",
                "candidate_tools": ["cluster_job_status_brief", "cluster_job_tail_log"],
            }
        )
    else:
        options.append(
            {
                "goal": "clarify_unknown_state",
                "when": "the watcher state is not enough to answer confidently",
                "evidence_needed": "fresh scheduler status and any available logs",
                "candidate_tools": ["cluster_job_status_brief"],
            }
        )
    return options


def _with_next_actions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["followup_options"] = _job_followup_options(item)
        enriched.append(item)
    return enriched


def snapshot(*, live_check: bool = False, limit: int = 20) -> dict[str, Any]:
    try:
        limit_int = max(1, min(int(limit or 20), 100))
    except (TypeError, ValueError):
        return {"status": "error", "message": "limit 必须是整数。", "watch_path": str(watch_path()), "jobs": []}
    rows = _load()[:limit_int]
    live_results: list[dict[str, Any]] = []
    if live_check:
        try:
            from dft_app.remote.realtime import job_status_brief
        except Exception as exc:
            return {
                "status": "partial",
                "watch_path": str(watch_path()),
                "jobs": _with_next_actions(rows),
                "live_results": [],
                "message": f"无法加载实时集群查询: {exc}",
            }
        for row in rows:
            job_id = str(row.get("job_id") or "").strip()
            if not job_id:
                continue
            result = job_status_brief(job_id=job_id, cluster_alias=str(row.get("cluster_alias") or "").strip() or None)
            live_results.append({"job_id": job_id, "result": result})
            status = str(result.get("state") or result.get("scheduler_state") or result.get("status") or "")
            if status:
                update_job_state(job_id, state=status, details=result)
        rows = _load()[:limit_int]
    return {
        "status": "ok",
        "watch_path": str(watch_path()),
        "count": len(rows),
        "jobs": _with_next_actions(rows),
        "live_results": live_results,
        "guidance": (
            "本 watcher 只提供 AETHER 已知提交记录和可选 followup_options。"
            "模型应根据用户意图、权限和证据缺口自由选择后续工具；需要最新状态时再 live_check=true。"
        ),
    }
