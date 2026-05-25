from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dft_app.web.background_jobs import load_job, utc_now, write_job


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    job = load_job(project_root, args.job_id)
    job["status"] = "running"
    job["started_at"] = utc_now()
    write_job(project_root, job)

    stdout_path = Path(job["stdout_path"])
    stderr_path = Path(job["stderr_path"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        process = subprocess.run(
            job["command"],
            cwd=project_root,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
        )

    result_payload = {
        "returncode": process.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    Path(job["result_path"]).write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    job["returncode"] = process.returncode
    job["completed_at"] = utc_now()
    job["status"] = "completed" if process.returncode == 0 else "failed"
    write_job(project_root, job)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
