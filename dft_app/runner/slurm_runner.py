from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dft_app.models import ExperimentSpec, PhaseStatus, PipelinePhase, RunRecord, RunStatus


@dataclass
class RunnerResult:
    status: str
    message: str
    details: dict[str, Any]


class SlurmRunner:
    """Submit and monitor Slurm jobs for the first released workflow."""

    def submit(self, spec: ExperimentSpec, run_record: RunRecord) -> RunnerResult:
        run_root = Path(run_record.run_root)
        job_script = run_root / "inputs" / "job.slurm"

        if not job_script.exists():
            message = f"未找到提交脚本: {job_script}"
            run_record.fail_phase(PipelinePhase.SUBMIT, message)
            return RunnerResult("failed", message, {"job_script": str(job_script)})

        if run_record.phases[PipelinePhase.BUILD.value].status != PhaseStatus.COMPLETED:
            message = "build 阶段尚未完成，不能提交作业。"
            run_record.block_phase(PipelinePhase.SUBMIT, message)
            return RunnerResult(
                "blocked",
                message,
                {"build_phase_status": run_record.phases[PipelinePhase.BUILD.value].status.value},
            )

        sbatch = shutil.which("sbatch")
        if not sbatch:
            message = "当前环境未检测到 sbatch，无法提交 Slurm 作业。请在集群登录节点运行此步骤。"
            run_record.block_phase(PipelinePhase.SUBMIT, message)
            return RunnerResult("blocked", message, {"job_script": str(job_script)})

        run_record.start_phase(PipelinePhase.SUBMIT, message="正在提交 Slurm 作业")
        process = subprocess.run(
            [sbatch, str(job_script)],
            cwd=run_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        if process.returncode != 0:
            message = f"sbatch 提交失败: {stderr or stdout or 'unknown error'}"
            run_record.fail_phase(PipelinePhase.SUBMIT, message)
            return RunnerResult(
                "failed",
                message,
                {
                    "command": [sbatch, str(job_script)],
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            )

        job_id = self._parse_job_id(stdout)
        if not job_id:
            message = f"sbatch 输出中未解析到 job id: {stdout or '<empty>'}"
            run_record.fail_phase(PipelinePhase.SUBMIT, message)
            return RunnerResult(
                "failed",
                message,
                {
                    "command": [sbatch, str(job_script)],
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            )

        run_record.scheduler_job_id = job_id
        run_record.complete_phase(
            PipelinePhase.SUBMIT,
            artifacts=[str(job_script)],
            message=f"已提交 Slurm 作业，job_id={job_id}",
        )
        run_record.overall_status = RunStatus.RUNNING
        run_record.touch()
        return RunnerResult(
            "submitted",
            f"已提交 Slurm 作业，job_id={job_id}",
            {
                "job_id": job_id,
                "command": [sbatch, str(job_script)],
                "stdout": stdout,
                "stderr": stderr,
            },
        )

    def monitor(self, run_record: RunRecord) -> RunnerResult:
        if not run_record.scheduler_job_id:
            message = "当前 run 没有 scheduler_job_id，尚未进入可监控状态。"
            return RunnerResult("unavailable", message, {})

        squeue = shutil.which("squeue")
        sacct = shutil.which("sacct")
        if squeue:
            return self._monitor_with_squeue(run_record, squeue)
        if sacct:
            return self._monitor_with_sacct(run_record, sacct)

        message = "当前环境未检测到 squeue 或 sacct，无法查询 Slurm 作业状态。"
        return RunnerResult(
            "unavailable",
            message,
            {"job_id": run_record.scheduler_job_id},
        )

    def _monitor_with_squeue(self, run_record: RunRecord, squeue: str) -> RunnerResult:
        process = subprocess.run(
            [squeue, "-j", run_record.scheduler_job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        if process.returncode != 0:
            message = f"squeue 查询失败: {stderr or stdout or 'unknown error'}"
            return RunnerResult(
                "error",
                message,
                {"returncode": process.returncode, "stdout": stdout, "stderr": stderr},
            )

        state = stdout.splitlines()[0].strip() if stdout else "UNKNOWN"
        return self._apply_monitor_state(
            run_record,
            scheduler_state=state,
            query_tool="squeue",
            raw_stdout=stdout,
            raw_stderr=stderr,
        )

    def _monitor_with_sacct(self, run_record: RunRecord, sacct: str) -> RunnerResult:
        process = subprocess.run(
            [sacct, "-j", run_record.scheduler_job_id, "--format=State", "--noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        if process.returncode != 0:
            message = f"sacct 查询失败: {stderr or stdout or 'unknown error'}"
            return RunnerResult(
                "error",
                message,
                {"returncode": process.returncode, "stdout": stdout, "stderr": stderr},
            )

        first_state = "UNKNOWN"
        for line in stdout.splitlines():
            cleaned = line.strip()
            if cleaned:
                first_state = cleaned.split()[0].split("+")[0]
                break

        return self._apply_monitor_state(
            run_record,
            scheduler_state=first_state,
            query_tool="sacct",
            raw_stdout=stdout,
            raw_stderr=stderr,
        )

    def _apply_monitor_state(
        self,
        run_record: RunRecord,
        *,
        scheduler_state: str,
        query_tool: str,
        raw_stdout: str,
        raw_stderr: str,
    ) -> RunnerResult:
        normalized = scheduler_state.strip().upper()
        details = {
            "job_id": run_record.scheduler_job_id,
            "scheduler_state": normalized,
            "query_tool": query_tool,
            "stdout": raw_stdout,
            "stderr": raw_stderr,
        }

        if normalized in {"PENDING", "CONFIGURING"}:
            run_record.start_phase(
                PipelinePhase.MONITOR,
                message=f"作业仍在等待或配置中: {normalized}",
            )
            return RunnerResult("running", f"作业状态: {normalized}", details)

        if normalized in {"RUNNING", "COMPLETING", "SUSPENDED"}:
            run_record.start_phase(
                PipelinePhase.MONITOR,
                message=f"作业正在运行: {normalized}",
            )
            return RunnerResult("running", f"作业状态: {normalized}", details)

        if normalized in {"COMPLETED"}:
            run_record.complete_phase(
                PipelinePhase.MONITOR,
                message="Slurm 作业已完成，下一步可以进入 parse 阶段。",
            )
            run_record.overall_status = RunStatus.READY
            run_record.touch()
            return RunnerResult("completed", "Slurm 作业已完成。", details)

        if normalized in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}:
            message = f"Slurm 作业异常结束: {normalized}"
            run_record.fail_phase(PipelinePhase.MONITOR, message)
            return RunnerResult("failed", message, details)

        run_record.start_phase(
            PipelinePhase.MONITOR,
            message=f"作业状态未知，保留当前运行记录: {normalized}",
        )
        return RunnerResult("unknown", f"未知 Slurm 状态: {normalized}", details)

    @staticmethod
    def _parse_job_id(stdout: str) -> str | None:
        match = re.search(r"(\d+)", stdout)
        return match.group(1) if match else None
