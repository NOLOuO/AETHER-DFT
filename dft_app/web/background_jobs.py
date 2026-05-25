from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jobs_root(project_root: Path) -> Path:
    root = project_root / ".web_runtime" / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_job(project_root: Path, *, title: str, command: list[str], run_root: str | None) -> dict[str, Any]:
    job_id = f"job_{uuid4().hex[:12]}"
    job_dir = jobs_root(project_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "title": title,
        "command": command,
        "run_root": run_root,
        "status": "queued",
        "created_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "returncode": None,
        "stdout_path": str(job_dir / "stdout.log"),
        "stderr_path": str(job_dir / "stderr.log"),
        "result_path": str(job_dir / "result.json"),
        "worker_pid": None,
    }
    write_job(project_root, payload)
    return payload


def write_job(project_root: Path, payload: dict[str, Any]) -> Path:
    path = jobs_root(project_root) / payload["job_id"] / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_job(project_root: Path, job_id: str) -> dict[str, Any]:
    path = jobs_root(project_root) / job_id / "job.json"
    return json.loads(path.read_text(encoding="utf-8"))


def list_jobs(project_root: Path, *, run_root: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in jobs_root(project_root).glob("*/job.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if run_root and payload.get("run_root") != run_root:
            continue
        items.append(payload)
    items.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return items[:limit]


def spawn_job_worker(project_root: Path, job_id: str) -> int:
    command = [sys.executable, "-m", "dft_app.web.job_worker", "--project-root", str(project_root), "--job-id", job_id]
    proc = subprocess.Popen(
        command,
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return proc.pid

