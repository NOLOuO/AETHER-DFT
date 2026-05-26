from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from dft_app.models import ExperimentSpec, PhaseStatus, PipelinePhase, RunRecord, RunStatus
from dft_app.remote.config import RemoteClusterConfig
from dft_app.runner.slurm_runner import SlurmRunner
from dft_app.submission_gate import verify_submission_evidence

_SAFE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class RemoteExecutionResult:
    status: str
    message: str
    details: dict[str, Any]


@dataclass
class _CommandResult:
    returncode: int
    stdout: str
    stderr: str


class SSHRemoteRunner:
    """Remote cluster runner supporting OpenSSH and WinSCP backends."""

    METADATA_FILES = [
        "experiment_spec.json",
        "planner_summary.json",
        "build_summary.json",
        "structure_resolution.json",
        "run_record.json",
    ]

    FETCH_PATTERNS = ["vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR", "vasp.out", "slurm.out"]
    RESEARCH_EXCLUDE_DIRS = {".aether_backups", ".omx", "__pycache__"}
    RESEARCH_EXCLUDE_SUFFIXES = {".pdf", ".pyc"}

    def __init__(self, config: RemoteClusterConfig | None = None):
        self.config = config
        self._slurm_runner = SlurmRunner()

    def describe_config(self) -> dict[str, Any]:
        config = self._load_config()
        return {
            **config.public_dict(),
            "selected_backend": self._select_backend(config),
        }

    def probe(self) -> RemoteExecutionResult:
        """Run a non-destructive SSH probe on the configured cluster."""
        config = self._load_config()
        backend = self._select_backend(config)
        tools_error = self._ensure_local_tools(config, backend)
        if tools_error is not None:
            return RemoteExecutionResult("blocked", tools_error, {"backend": backend})

        probe_command = (
            "printf 'hostname='; hostname; "
            "printf 'pwd='; pwd; "
            "printf 'sbatch='; command -v sbatch || true; "
            "printf 'squeue='; command -v squeue || true; "
            "printf 'vasp_std='; command -v vasp_std || true"
        )
        process = self._run_remote_command(
            config,
            probe_command,
            timeout=config.connect_timeout + 20,
            backend=backend,
        )
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        if process.returncode != 0:
            return RemoteExecutionResult(
                "failed",
                f"SSH 集群探测失败: {stderr or stdout or 'unknown error'}",
                {
                    "backend": backend,
                    "config": config.public_dict(),
                    "stdout": stdout,
                    "stderr": stderr,
                },
            )
        parsed = self._parse_probe_stdout(stdout)
        has_sbatch = bool(parsed.get("sbatch"))
        message = "SSH 集群连通，检测到 sbatch。" if has_sbatch else "SSH 集群连通，但默认环境未检测到 sbatch。"
        return RemoteExecutionResult(
            "ok" if has_sbatch else "partial",
            message,
            {
                "backend": backend,
                "config": config.public_dict(),
                "probe": parsed,
                "stdout": stdout,
                "stderr": stderr,
            },
        )

    def submit(self, spec: ExperimentSpec, run_record: RunRecord) -> RemoteExecutionResult:
        run_root = Path(run_record.run_root)
        local_gate = verify_submission_evidence(spec, run_record, mode="submit", write_report=False)
        if local_gate["status"] != "ready":
            gate = verify_submission_evidence(spec, run_record, mode="remote_submit", remote_probe=None)
            message = "远程提交前证据核对未通过，已阻止上传和 sbatch。"
            run_record.block_phase(PipelinePhase.SUBMIT, message)
            return RemoteExecutionResult(
                "blocked",
                message,
                {
                    "gate_path": str(run_root / "metadata" / "pre_submit_gate.json"),
                    "blockers": gate["blockers"],
                    "warnings": gate["warnings"],
                },
            )

        probe_result = self.probe()
        probe_payload = {
            "status": probe_result.status,
            "message": probe_result.message,
            "details": probe_result.details,
        }
        gate = verify_submission_evidence(
            spec,
            run_record,
            mode="remote_submit",
            remote_probe=probe_payload,
        )
        gate_path = run_root / "metadata" / "pre_submit_gate.json"
        if gate["status"] != "ready":
            message = "远程提交前集群/输入证据核对未通过，已阻止上传和 sbatch。"
            run_record.block_phase(PipelinePhase.SUBMIT, message)
            return RemoteExecutionResult(
                "blocked",
                message,
                {
                    "gate_path": str(gate_path),
                    "blockers": gate["blockers"],
                    "warnings": gate["warnings"],
                    "probe": probe_payload,
                },
            )

        if run_record.phases[PipelinePhase.BUILD.value].status != PhaseStatus.COMPLETED:
            message = "build 阶段尚未完成，不能执行远程提交。"
            run_record.block_phase(PipelinePhase.SUBMIT, message)
            return RemoteExecutionResult("blocked", message, {})

        config = self._load_config()
        backend = self._select_backend(config)
        tools_error = self._ensure_local_tools(config, backend)
        if tools_error is not None:
            run_record.block_phase(PipelinePhase.SUBMIT, tools_error)
            return RemoteExecutionResult("blocked", tools_error, {})

        remote_run_root = config.remote_run_root(spec.task_id, run_record.run_id)
        run_record.start_phase(
            PipelinePhase.SUBMIT,
            message=f"正在准备远程提交环境，backend={backend}",
        )

        try:
            self._prepare_remote_directories(config, remote_run_root, backend)
            uploaded_files = self._upload_run_artifacts(config, run_root, remote_run_root, backend)
        except Exception as exc:
            message = f"远程目录准备或文件上传失败: {exc}"
            run_record.fail_phase(PipelinePhase.SUBMIT, message)
            return RemoteExecutionResult(
                "failed",
                message,
                {"remote_run_root": remote_run_root, "backend": backend},
            )

        job_script = f"{remote_run_root}/inputs/job.slurm"
        run_record.start_phase(
            PipelinePhase.SUBMIT,
            message=f"正在通过 {backend} 远程提交 Slurm 作业",
        )
        submit_command = (
            f"cd {self._quote(remote_run_root)} && "
            "(sed -i 's/\\r$//' inputs/job.slurm 2>/dev/null || true) && "
            "sbatch inputs/job.slurm"
        )
        process = self._run_remote_command(config, submit_command, timeout=60, backend=backend)

        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        if process.returncode != 0:
            message = f"远程 sbatch 提交失败: {stderr or stdout or 'unknown error'}"
            run_record.fail_phase(PipelinePhase.SUBMIT, message)
            return RemoteExecutionResult(
                "failed",
                message,
                {
                    "backend": backend,
                    "remote_run_root": remote_run_root,
                    "job_script": job_script,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            )

        job_id = self._slurm_runner._parse_job_id(stdout)
        if not job_id:
            message = f"远程 sbatch 输出中未解析到 job id: {stdout or '<empty>'}"
            run_record.fail_phase(PipelinePhase.SUBMIT, message)
            return RemoteExecutionResult(
                "failed",
                message,
                {
                    "backend": backend,
                    "remote_run_root": remote_run_root,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            )

        run_record.scheduler_job_id = job_id
        remote_notes = run_record.notes.setdefault("remote", {})
        remote_notes.update(
            {
                "mode": backend,
                "host": config.host,
                "user": config.user,
                "port": config.port,
                "remote_run_root": remote_run_root,
                "job_script": job_script,
                "uploaded_files": uploaded_files,
            }
        )
        run_record.complete_phase(
            PipelinePhase.SUBMIT,
            artifacts=[str(run_root / "inputs" / "job.slurm"), str(gate_path)],
            message=f"已通过 {backend} 提交远程 Slurm 作业，job_id={job_id}",
        )
        run_record.overall_status = RunStatus.RUNNING
        run_record.touch()
        return RemoteExecutionResult(
            "submitted",
            f"已通过 {backend} 提交远程 Slurm 作业，job_id={job_id}",
            {
                "backend": backend,
                "job_id": job_id,
                "remote_run_root": remote_run_root,
                "stdout": stdout,
                "stderr": stderr,
                "uploaded_files": uploaded_files,
            },
        )

    def monitor(self, run_record: RunRecord, sync_outputs: bool = True) -> RemoteExecutionResult:
        config = self._load_config()
        backend = self._select_backend(config)
        remote_info = run_record.notes.get("remote", {})
        remote_run_root = remote_info.get("remote_run_root")
        if not remote_run_root:
            message = "当前 run 没有远程路径记录，无法执行远程监控。"
            return RemoteExecutionResult("unavailable", message, {})

        if not run_record.scheduler_job_id:
            message = "当前 run 没有 scheduler_job_id，无法执行远程监控。"
            return RemoteExecutionResult("unavailable", message, {})
        job_id = str(run_record.scheduler_job_id).strip()
        if not _SAFE_JOB_ID_RE.fullmatch(job_id):
            return RemoteExecutionResult(
                "blocked",
                "scheduler_job_id 包含不允许的字符，已阻止远程状态查询。",
                {"backend": backend, "remote_run_root": remote_run_root},
            )
        try:
            remote_run_root = self._safe_remote_run_root(str(remote_run_root), config)
        except ValueError as exc:
            return RemoteExecutionResult(
                "blocked",
                f"远程 run 路径不安全，已阻止远程状态查询: {exc}",
                {"backend": backend, "remote_run_root": remote_run_root},
            )

        monitor_command = (
            f"squeue -j {job_id} -h -o %T || "
            f"sacct -j {job_id} --format=State --noheader"
        )
        process = self._run_remote_command(config, monitor_command, timeout=60, backend=backend)
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        if process.returncode != 0:
            message = f"远程作业状态查询失败: {stderr or stdout or 'unknown error'}"
            return RemoteExecutionResult(
                "error",
                message,
                {
                    "backend": backend,
                    "stdout": stdout,
                    "stderr": stderr,
                    "remote_run_root": remote_run_root,
                },
            )

        first_line = next((line.strip() for line in stdout.splitlines() if line.strip()), "UNKNOWN")
        scheduler_state = first_line.split()[0].split("+")[0].upper()

        if scheduler_state in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED"}:
            run_record.start_phase(
                PipelinePhase.MONITOR,
                message=f"远程作业状态: {scheduler_state}",
            )
            return RemoteExecutionResult(
                "running",
                f"远程作业状态: {scheduler_state}",
                {
                    "backend": backend,
                    "scheduler_state": scheduler_state,
                    "remote_run_root": remote_run_root,
                },
            )

        if scheduler_state == "COMPLETED":
            synced_files: list[str] = []
            if sync_outputs:
                fetch_result = self.fetch_outputs(run_record, config=config)
                if fetch_result.status == "failed":
                    message = f"远程作业已完成，但结果同步失败: {fetch_result.message}"
                    run_record.block_phase(PipelinePhase.MONITOR, message)
                    return RemoteExecutionResult(
                        "partial",
                        message,
                        {
                            "backend": backend,
                            "scheduler_state": scheduler_state,
                            "remote_run_root": remote_run_root,
                            "fetch_details": fetch_result.details,
                        },
                    )
                synced_files = fetch_result.details.get("synced_files", [])
            run_record.complete_phase(
                PipelinePhase.MONITOR,
                artifacts=synced_files,
                message="远程 Slurm 作业已完成，结果已同步到本地。",
            )
            run_record.mark_ready()
            return RemoteExecutionResult(
                "completed",
                "远程 Slurm 作业已完成。",
                {
                    "backend": backend,
                    "scheduler_state": scheduler_state,
                    "remote_run_root": remote_run_root,
                    "synced_files": synced_files,
                },
            )

        if scheduler_state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}:
            message = f"远程 Slurm 作业异常结束: {scheduler_state}"
            run_record.fail_phase(PipelinePhase.MONITOR, message)
            return RemoteExecutionResult(
                "failed",
                message,
                {
                    "backend": backend,
                    "scheduler_state": scheduler_state,
                    "remote_run_root": remote_run_root,
                },
            )

        run_record.start_phase(
            PipelinePhase.MONITOR,
            message=f"远程作业状态未知: {scheduler_state}",
        )
        return RemoteExecutionResult(
            "unknown",
            f"远程作业状态未知: {scheduler_state}",
            {
                "backend": backend,
                "scheduler_state": scheduler_state,
                "remote_run_root": remote_run_root,
            },
        )

    def fetch_outputs(
        self, run_record: RunRecord, config: RemoteClusterConfig | None = None
    ) -> RemoteExecutionResult:
        config = config or self._load_config()
        backend = self._select_backend(config)
        remote_info = run_record.notes.get("remote", {})
        remote_run_root = remote_info.get("remote_run_root")
        if not remote_run_root:
            return RemoteExecutionResult("unavailable", "缺少远程路径信息，无法拉取输出。", {})
        try:
            remote_run_root = self._safe_remote_run_root(str(remote_run_root), config)
        except ValueError as exc:
            return RemoteExecutionResult(
                "blocked",
                f"远程 run 路径不安全，已阻止拉取: {exc}",
                {"backend": backend, "remote_run_root": remote_run_root},
            )

        find_expression = " -o ".join(f"-name {self._quote(pattern)}" for pattern in self.FETCH_PATTERNS)
        find_command = (
            f"find {self._quote(remote_run_root)} -type f \\( {find_expression} \\) | sort -u"
        )
        process = self._run_remote_command(config, find_command, timeout=60, backend=backend)
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        if process.returncode != 0:
            return RemoteExecutionResult(
                "failed",
                f"远程查找输出文件失败: {stderr or stdout or 'unknown error'}",
                {"backend": backend, "remote_run_root": remote_run_root},
            )

        remote_files = [line.strip() for line in stdout.splitlines() if line.strip()]
        synced_files: list[str] = []
        outputs_dir = Path(run_record.run_root) / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        try:
            for remote_file in remote_files:
                relative = PurePosixPath(remote_file).relative_to(PurePosixPath(remote_run_root))
                local_target = outputs_dir / Path(*relative.parts)
                local_target.parent.mkdir(parents=True, exist_ok=True)
                self._download_from_remote(config, remote_file, local_target, timeout=120, backend=backend)
                synced_files.append(str(local_target))
        except Exception as exc:
            return RemoteExecutionResult(
                "failed",
                f"远程输出下载失败: {exc}",
                {
                    "backend": backend,
                    "remote_run_root": remote_run_root,
                    "synced_files": synced_files,
                },
            )

        remote_info["last_synced_files"] = synced_files
        run_record.touch()
        return RemoteExecutionResult(
            "synced",
            f"已同步 {len(synced_files)} 个远程输出文件。",
            {"backend": backend, "remote_run_root": remote_run_root, "synced_files": synced_files},
        )

    @staticmethod
    def _normalize_project_prefix(project: str | None) -> str | None:
        if project is None:
            return None
        cleaned = str(project).strip().strip("/")
        if not cleaned:
            return None
        if cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
            raise ValueError(f"project 必须是单一目录名，收到: {project!r}")
        return cleaned

    @staticmethod
    def _filter_manifest_by_project(
        manifest: dict[str, dict[str, Any]], project: str | None
    ) -> dict[str, dict[str, Any]]:
        if not project:
            return manifest
        prefix = f"{project}/"
        return {k: v for k, v in manifest.items() if k == project or k.startswith(prefix)}

    def research_status(
        self,
        local_research_root: Path,
        *,
        remote_research_dir: str | None = None,
        project: str | None = None,
    ) -> RemoteExecutionResult:
        """Compare local research/ with the cluster research directory.

        当 ``project`` 给出时，只比较 ``research/<project>/`` 下的文件，便于按项目同步。
        """

        config = self._load_config()
        backend = self._select_backend(config)
        tools_error = self._ensure_local_tools(config, backend)
        if tools_error is not None:
            return RemoteExecutionResult("blocked", tools_error, {"backend": backend})

        try:
            project_prefix = self._normalize_project_prefix(project)
        except ValueError as exc:
            return RemoteExecutionResult("blocked", str(exc), {"backend": backend})

        local_manifest = self._filter_manifest_by_project(
            self._local_research_manifest(local_research_root), project_prefix
        )
        try:
            remote_dir = self._safe_remote_research_dir(
                remote_research_dir or self._default_remote_research_dir(config), config
            )
        except ValueError as exc:
            return RemoteExecutionResult("blocked", str(exc), {"backend": backend})
        remote_result = self._remote_research_manifest(config, backend, remote_dir)
        if remote_result["status"] == "ok":
            remote_result["manifest"] = self._filter_manifest_by_project(
                remote_result["manifest"], project_prefix
            )
        if remote_result["status"] != "ok":
            return RemoteExecutionResult(
                remote_result["status"],
                remote_result["message"],
                {
                    "backend": backend,
                    "remote_research_dir": remote_dir,
                    "local_count": len(local_manifest),
                    **remote_result,
                },
            )
        remote_manifest = remote_result["manifest"]
        missing_remote = sorted(set(local_manifest) - set(remote_manifest))
        missing_local = sorted(set(remote_manifest) - set(local_manifest))
        differing = sorted(
            rel
            for rel in set(local_manifest).intersection(remote_manifest)
            if local_manifest[rel]["sha256"] != remote_manifest[rel]["sha256"]
        )
        status = "in_sync" if not (missing_remote or missing_local or differing) else "out_of_sync"
        return RemoteExecutionResult(
            "ok",
            "本地 research 与集群 ~/research 已一致。" if status == "in_sync" else "本地 research 与集群 ~/research 存在差异。",
            {
                "backend": backend,
                "remote_research_dir": remote_dir,
                "project": project_prefix,
                "sync_status": status,
                "local_count": len(local_manifest),
                "remote_count": len(remote_manifest),
                "missing_remote": missing_remote,
                "missing_local": missing_local,
                "differing": differing,
                "excluded": {
                    "dirs": sorted(self.RESEARCH_EXCLUDE_DIRS),
                    "suffixes": sorted(self.RESEARCH_EXCLUDE_SUFFIXES),
                },
                "policy": "状态检查只读；同步时默认本地 research 为真源，远端冲突会先备份再覆盖，不删除远端独有文件。",
            },
        )

    def sync_research_to_remote(
        self,
        local_research_root: Path,
        *,
        remote_research_dir: str | None = None,
        dry_run: bool = True,
        project: str | None = None,
    ) -> RemoteExecutionResult:
        """Push local research/ to cluster ~/research without deleting remote-only files.

        ``project`` 给定时只同步 ``research/<project>/`` 子树。
        """

        status_result = self.research_status(
            local_research_root, remote_research_dir=remote_research_dir, project=project
        )
        if status_result.status != "ok":
            return status_result
        details = dict(status_result.details)
        to_upload = sorted(set(details["missing_remote"]) | set(details["differing"]))
        if not to_upload:
            return RemoteExecutionResult(
                "ok",
                "无需同步：集群 ~/research 已包含本地 research 的当前版本。",
                {**details, "dry_run": dry_run, "uploaded": [], "backup_dir": None},
            )
        if dry_run:
            return RemoteExecutionResult(
                "planned",
                f"需要上传/覆盖 {len(to_upload)} 个 research 文件；dry_run=True 未修改集群。",
                {**details, "dry_run": True, "would_upload": to_upload, "backup_dir": None},
            )

        config = self._load_config()
        backend = self._select_backend(config)
        remote_dir = details["remote_research_dir"]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        remote_sync_dir = str(PurePosixPath(remote_dir).parent / ".aether_research_sync")
        remote_archive = f"{remote_sync_dir}/research-{timestamp}.tar.gz"
        backup_dir = f"{remote_dir}/.aether_backups/{timestamp}"
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "research-sync.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                for rel in to_upload:
                    tar.add(local_research_root / Path(*PurePosixPath(rel).parts), arcname=rel)
            mkdir = self._run_remote_command(
                config,
                f"mkdir -p {self._quote(remote_sync_dir)} {self._quote(remote_dir)}",
                timeout=30,
                backend=backend,
            )
            if mkdir.returncode != 0:
                return RemoteExecutionResult("failed", f"远端同步目录创建失败: {mkdir.stderr or mkdir.stdout}", {"backend": backend})
            self._upload_to_remote(config, archive_path, remote_archive, timeout=180, backend=backend)

        backup_commands = []
        for rel in details["differing"]:
            remote_file = str(PurePosixPath(remote_dir) / PurePosixPath(rel))
            backup_file = str(PurePosixPath(backup_dir) / PurePosixPath(rel))
            backup_commands.append(
                "if [ -f {src} ]; then mkdir -p {dst_dir} && cp -p {src} {dst}; fi".format(
                    src=self._quote(remote_file),
                    dst_dir=self._quote(str(PurePosixPath(backup_file).parent)),
                    dst=self._quote(backup_file),
                )
            )
        remote_script = " && ".join(
            [
                *backup_commands,
                f"tar -xzf {self._quote(remote_archive)} -C {self._quote(remote_dir)}",
                f"rm -f {self._quote(remote_archive)}",
            ]
        )
        apply_result = self._run_remote_command(config, remote_script, timeout=180, backend=backend)
        if apply_result.returncode != 0:
            return RemoteExecutionResult(
                "failed",
                f"research 同步到集群失败: {apply_result.stderr or apply_result.stdout}",
                {
                    **details,
                    "dry_run": False,
                    "uploaded": [],
                    "backup_dir": backup_dir,
                    "stderr": apply_result.stderr,
                    "stdout": apply_result.stdout,
                },
            )
        return RemoteExecutionResult(
            "synced",
            f"已将本地 research 同步到集群 {remote_dir}，上传/覆盖 {len(to_upload)} 个文件。",
            {
                **details,
                "dry_run": False,
                "uploaded": to_upload,
                "backup_dir": backup_dir if details["differing"] else None,
            },
        )

    def sync_research_from_remote(
        self,
        local_research_root: Path,
        *,
        remote_research_dir: str | None = None,
        dry_run: bool = True,
        project: str | None = None,
    ) -> RemoteExecutionResult:
        """从集群 ~/research 拉回本地 research/，绝不删除本地独有文件。

        策略：以远端为真源——只下载/覆盖远端有但本地缺失或哈希不同的文件；
        差异覆盖前先把本地原文件备份到 ``research/.aether_local_backups/<ts>/`` 下。
        ``project`` 给定时只拉取 ``research/<project>/`` 子树。
        """
        status_result = self.research_status(
            local_research_root, remote_research_dir=remote_research_dir, project=project
        )
        if status_result.status != "ok":
            return status_result
        details = dict(status_result.details)
        to_download = sorted(set(details["missing_local"]) | set(details["differing"]))
        if not to_download:
            return RemoteExecutionResult(
                "ok",
                "无需拉取：本地 research 已包含集群 ~/research 的当前版本。",
                {**details, "dry_run": dry_run, "downloaded": [], "backup_dir": None, "direction": "pull"},
            )
        if dry_run:
            return RemoteExecutionResult(
                "planned",
                f"需要从集群拉取/覆盖 {len(to_download)} 个 research 文件；dry_run=True 未修改本地。",
                {**details, "dry_run": True, "would_download": to_download, "backup_dir": None, "direction": "pull"},
            )

        config = self._load_config()
        backend = self._select_backend(config)
        remote_dir = details["remote_research_dir"]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        local_backup_dir = local_research_root / ".aether_local_backups" / timestamp
        # 先把本地将被覆盖的差异文件备份
        for rel in details["differing"]:
            src = local_research_root / Path(*PurePosixPath(rel).parts)
            if src.exists() and src.is_file():
                dst = local_backup_dir / Path(*PurePosixPath(rel).parts)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        # 远端打包指定文件清单
        remote_sync_dir = str(PurePosixPath(remote_dir).parent / ".aether_research_sync")
        remote_archive = f"{remote_sync_dir}/research-pull-{timestamp}.tar.gz"
        # 在远端构造 tar 命令；用 \\\\n 写入文件清单避免 argv 太长
        list_lines = "\n".join(to_download)
        list_b64 = self._encode_text_for_remote(list_lines)
        pack_script = " && ".join(
            [
                f"mkdir -p {self._quote(remote_sync_dir)}",
                f"printf '%s' {list_b64} | base64 -d > {self._quote(remote_sync_dir)}/pull-list.txt",
                f"cd {self._quote(remote_dir)} && tar -czf {self._quote(remote_archive)} -T {self._quote(remote_sync_dir)}/pull-list.txt",
            ]
        )
        pack_result = self._run_remote_command(config, pack_script, timeout=180, backend=backend)
        if pack_result.returncode != 0:
            return RemoteExecutionResult(
                "failed",
                f"远端打包 pull 归档失败: {pack_result.stderr or pack_result.stdout}",
                {**details, "dry_run": False, "backup_dir": str(local_backup_dir), "direction": "pull"},
            )

        with tempfile.TemporaryDirectory() as tmp:
            local_archive = Path(tmp) / "pull.tar.gz"
            self._download_from_remote(config, remote_archive, local_archive, timeout=180, backend=backend)
            with tarfile.open(local_archive, "r:gz") as tar:
                try:
                    self._safe_extract_tar(tar, local_research_root, allowed_members=set(to_download))
                except ValueError as exc:
                    return RemoteExecutionResult(
                        "failed",
                        f"远端 pull 归档包含不安全路径: {exc}",
                        {**details, "dry_run": False, "backup_dir": str(local_backup_dir), "direction": "pull"},
                    )
        cleanup = self._run_remote_command(
            config, f"rm -f {self._quote(remote_archive)} {self._quote(remote_sync_dir)}/pull-list.txt", timeout=30, backend=backend
        )
        cleanup_warning = None
        if cleanup.returncode != 0:
            cleanup_warning = cleanup.stderr.strip() or cleanup.stdout.strip()

        return RemoteExecutionResult(
            "synced",
            f"已从集群 {remote_dir} 拉回 {len(to_download)} 个 research 文件。",
            {
                **details,
                "dry_run": False,
                "downloaded": to_download,
                "backup_dir": str(local_backup_dir) if details["differing"] else None,
                "direction": "pull",
                "cleanup_warning": cleanup_warning,
            },
        )

    def pull_remote_run_outputs(
        self,
        remote_run_root: str,
        local_target_dir: Path,
        *,
        patterns: list[str] | None = None,
    ) -> RemoteExecutionResult:
        """轻量拉取某 remote_run_root 下的产出文件到 local_target_dir。

        默认匹配 ``FETCH_PATTERNS``（vasprun.xml / OUTCAR / OSZICAR / CONTCAR / vasp.out / slurm.out）。
        与重量级 ``fetch_outputs`` 区别：不需要 RunRecord，可以独立按路径用，适合
        模型用 ``cluster_my_jobs`` 拿到 job_id → ``research_workspace_pull_logs`` / fetch 工具直接抓产出。
        """
        if not remote_run_root or not str(remote_run_root).strip():
            return RemoteExecutionResult("unavailable", "remote_run_root 不能为空。", {})
        config = self._load_config()
        backend = self._select_backend(config)
        tools_error = self._ensure_local_tools(config, backend)
        if tools_error is not None:
            return RemoteExecutionResult("blocked", tools_error, {"backend": backend})
        try:
            remote_run_root = self._safe_remote_run_root(str(remote_run_root), config)
        except ValueError as exc:
            return RemoteExecutionResult(
                "blocked",
                f"remote_run_root 不安全，已阻止拉取: {exc}",
                {"backend": backend, "remote_run_root": remote_run_root},
            )

        use_patterns = patterns or list(self.FETCH_PATTERNS)
        find_expression = " -o ".join(f"-name {self._quote(p)}" for p in use_patterns)
        find_command = (
            f"find {self._quote(remote_run_root)} -type f \\( {find_expression} \\) | sort -u"
        )
        process = self._run_remote_command(config, find_command, timeout=60, backend=backend)
        if process.returncode != 0:
            return RemoteExecutionResult(
                "failed",
                f"远端查找 run 输出失败: {process.stderr.strip() or process.stdout.strip()}",
                {"backend": backend, "remote_run_root": remote_run_root},
            )
        remote_files = [line.strip() for line in process.stdout.splitlines() if line.strip()]
        if not remote_files:
            return RemoteExecutionResult(
                "missing",
                f"在 {remote_run_root} 下没找到匹配 {use_patterns} 的输出文件。",
                {"backend": backend, "remote_run_root": remote_run_root, "patterns": use_patterns},
            )
        local_target_dir = Path(local_target_dir)
        local_target_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[str] = []
        try:
            for remote_file in remote_files:
                relative = PurePosixPath(remote_file).relative_to(PurePosixPath(remote_run_root))
                local_target = local_target_dir / Path(*relative.parts)
                local_target.parent.mkdir(parents=True, exist_ok=True)
                self._download_from_remote(config, remote_file, local_target, timeout=120, backend=backend)
                downloaded.append(str(local_target))
        except Exception as exc:
            return RemoteExecutionResult(
                "failed",
                f"拉取 run 输出失败: {exc}",
                {
                    "backend": backend,
                    "remote_run_root": remote_run_root,
                    "patterns": use_patterns,
                    "downloaded": downloaded,
                },
            )
        return RemoteExecutionResult(
            "synced",
            f"已从 {remote_run_root} 拉回 {len(downloaded)} 个 run 输出文件到 {local_target_dir}。",
            {
                "backend": backend,
                "remote_run_root": remote_run_root,
                "patterns": use_patterns,
                "downloaded": downloaded,
                "local_target_dir": str(local_target_dir),
            },
        )

    @staticmethod
    def _encode_text_for_remote(text: str) -> str:
        """base64 编码后用单引号包裹，避免在远端解析任何元字符。"""
        import base64

        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return f"'{encoded}'"

    @classmethod
    def _safe_remote_run_root(cls, remote_run_root: str, config: RemoteClusterConfig) -> str:
        """Validate a remote run path before using SSH credentials to inspect it."""

        return cls._safe_remote_path(
            remote_run_root,
            allowed_roots=[config.remote_base_dir, f"/scratch/{config.user}"],
            home_user=config.user,
            label="remote_run_root",
        )

    @classmethod
    def _safe_remote_research_dir(cls, remote_research_dir: str, config: RemoteClusterConfig) -> str:
        """Keep research sync scoped to the user's ``~/research`` tree."""

        return cls._safe_remote_path(
            remote_research_dir,
            allowed_roots=[cls._default_remote_research_dir(config)],
            home_user=config.user,
            label="remote_research_dir",
        )

    @staticmethod
    def _safe_remote_path(path: str, *, allowed_roots: list[str], home_user: str, label: str) -> str:
        value = str(path or "").strip()
        if not value:
            raise ValueError(f"{label} 不能为空。")
        if value == "~":
            value = f"/home/{home_user}"
        elif value.startswith("~/"):
            value = f"/home/{home_user}/{value[2:]}"
        if any(token in value for token in ("\n", "\r", "\x00", "$", "`", ";", "&", "|")):
            raise ValueError(f"{label} 包含不允许的 shell 元字符。")
        posix = PurePosixPath(value)
        if not posix.is_absolute():
            raise ValueError(f"{label} 必须是绝对路径或 ~/ 开头路径。")
        if ".." in posix.parts:
            raise ValueError(f"{label} 不能包含 '..'。")

        normalized = "/" + "/".join(part for part in posix.parts if part not in {"", "/"})
        safe_roots = []
        for root in allowed_roots:
            expanded = str(root or "").strip()
            if expanded == "~":
                expanded = f"/home/{home_user}"
            elif expanded.startswith("~/"):
                expanded = f"/home/{home_user}/{expanded[2:]}"
            if not expanded:
                continue
            root_path = PurePosixPath(expanded)
            if root_path.is_absolute() and ".." not in root_path.parts:
                safe_roots.append("/" + "/".join(part for part in root_path.parts if part not in {"", "/"}))
        if not any(normalized == root or normalized.startswith(f"{root}/") for root in safe_roots):
            raise ValueError(f"{label} 必须位于允许根目录内: {safe_roots}")
        return normalized

    @staticmethod
    def _safe_extract_tar(
        tar: tarfile.TarFile,
        target_dir: Path,
        *,
        allowed_members: set[str] | None = None,
    ) -> list[str]:
        """Extract a tar archive only after path, type, and allow-list checks."""

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        resolved_target = target.resolve()
        members = tar.getmembers()
        extracted: list[str] = []
        for member in members:
            member_name = str(member.name or "").strip()
            rel = PurePosixPath(member_name)
            if not member_name or rel.is_absolute() or ".." in rel.parts or "\\" in member_name:
                raise ValueError(f"不安全成员路径: {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"不安全成员类型: {member.name!r}")
            if allowed_members is not None:
                clean_allowed = {str(PurePosixPath(item)) for item in allowed_members}
                if member.isfile() and member_name not in clean_allowed:
                    raise ValueError(f"归档包含未请求文件: {member_name}")
                if member.isdir() and not any(item.startswith(f"{member_name.rstrip('/')}/") for item in clean_allowed):
                    raise ValueError(f"归档包含未请求目录: {member_name}")
            destination = (target / Path(*rel.parts)).resolve()
            if destination != resolved_target and resolved_target not in destination.parents:
                raise ValueError(f"成员会写出目标目录: {member_name}")
        for member in members:
            tar.extract(member, target)
            extracted.append(member.name)
        return extracted

    def _prepare_remote_directories(
        self, config: RemoteClusterConfig, remote_run_root: str, backend: str
    ) -> None:
        mkdir_command = (
            f"mkdir -p {self._quote(remote_run_root)}/inputs "
            f"{self._quote(remote_run_root)}/metadata "
            f"{self._quote(remote_run_root)}/logs "
            f"{self._quote(remote_run_root)}/outputs"
        )
        process = self._run_remote_command(config, mkdir_command, timeout=30, backend=backend)
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "远程目录创建失败")

    def _upload_run_artifacts(
        self, config: RemoteClusterConfig, run_root: Path, remote_run_root: str, backend: str
    ) -> list[str]:
        uploaded: list[str] = []

        inputs_dir = run_root / "inputs"
        for path in sorted(inputs_dir.glob("*")):
            if path.is_file():
                remote_target = f"{remote_run_root}/inputs/{path.name}"
                self._upload_to_remote(config, path, remote_target, timeout=120, backend=backend)
                uploaded.append(remote_target)

        metadata_dir = run_root / "metadata"
        for name in self.METADATA_FILES:
            path = metadata_dir / name
            if not path.exists():
                continue
            remote_target = f"{remote_run_root}/metadata/{name}"
            self._upload_to_remote(config, path, remote_target, timeout=60, backend=backend)
            uploaded.append(remote_target)

        return uploaded

    def _load_config(self) -> RemoteClusterConfig:
        return self.config or RemoteClusterConfig.from_env()

    def _select_backend(self, config: RemoteClusterConfig) -> str:
        backend = config.backend.strip().lower()
        if backend in {"ssh", "openssh"}:
            return "openssh"
        if backend == "winscp":
            return "winscp"
        if backend != "auto":
            raise ValueError(f"不支持的远程 backend: {config.backend}")

        if os.name == "nt" and self._resolve_winscp_path(config, required=False):
            return "winscp"
        return "openssh"

    def _ensure_local_tools(self, config: RemoteClusterConfig, backend: str) -> str | None:
        if backend == "winscp":
            if self._resolve_winscp_path(config, required=False):
                return None
            return "当前机器缺少 WinSCP.com，无法使用 winscp 远程模式。"

        missing = [command for command in ["ssh", "scp"] if not shutil.which(command)]
        if missing:
            joined = ", ".join(missing)
            return f"当前机器缺少远程所需命令: {joined}"
        return None

    def _run_remote_command(
        self, config: RemoteClusterConfig, remote_command: str, timeout: int, backend: str
    ) -> _CommandResult:
        if backend == "winscp":
            return self._run_winscp_remote_command(config, remote_command, timeout)

        command = self._build_ssh_command(config, remote_command)
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return _CommandResult(
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )

    def _upload_to_remote(
        self,
        config: RemoteClusterConfig,
        local_path: Path,
        remote_path: str,
        timeout: int,
        backend: str,
    ) -> None:
        if backend == "winscp":
            self._run_winscp_transfer_script(
                config,
                [
                    f"put -transfer=binary -nopreservetime {self._winscp_quote(str(local_path))} "
                    f"{self._winscp_quote(remote_path)}"
                ],
                timeout=timeout,
            )
            return

        command = self._build_scp_command(
            config,
            source=str(local_path),
            target=f"{config.ssh_target()}:{remote_path}",
        )
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "scp 上传失败")

    def _download_from_remote(
        self,
        config: RemoteClusterConfig,
        remote_path: str,
        local_path: Path,
        timeout: int,
        backend: str,
    ) -> None:
        if backend == "winscp":
            self._run_winscp_transfer_script(
                config,
                [
                    f"get -transfer=binary -nopreservetime {self._winscp_quote(remote_path)} "
                    f"{self._winscp_quote(str(local_path))}"
                ],
                timeout=timeout,
            )
            return

        command = self._build_scp_command(
            config,
            source=f"{config.ssh_target()}:{remote_path}",
            target=str(local_path),
        )
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "scp 下载失败")

    def _run_winscp_remote_command(
        self, config: RemoteClusterConfig, remote_command: str, timeout: int
    ) -> _CommandResult:
        process = self._run_winscp_script(config, [f"call {remote_command}", "exit"], timeout)
        stdout = self._extract_winscp_command_output(process.stdout)
        stderr = process.stderr.strip()
        if process.returncode != 0 and not stderr:
            stderr = stdout
        return _CommandResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def _run_winscp_transfer_script(
        self, config: RemoteClusterConfig, commands: list[str], timeout: int
    ) -> None:
        process = self._run_winscp_script(config, [*commands, "exit"], timeout)
        if process.returncode != 0:
            message = process.stderr.strip() or self._extract_winscp_error(process.stdout)
            raise RuntimeError(message or "WinSCP 文件传输失败")

    def _run_winscp_script(
        self, config: RemoteClusterConfig, commands: list[str], timeout: int
    ) -> subprocess.CompletedProcess[str]:
        winscp_path = self._resolve_winscp_path(config)
        open_command = self._build_winscp_open_command(config)
        script = "\n".join([open_command, *commands]) + "\n"
        return subprocess.run(
            [winscp_path, f"/ini={config.winscp_ini_path}", "/stdin", "/nointeractiveinput"],
            input=script,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )

    def _resolve_winscp_path(
        self, config: RemoteClusterConfig, required: bool = True
    ) -> str | None:
        path = config.winscp_path or shutil.which("WinSCP.com")
        if path:
            return path
        if required:
            raise RuntimeError("未找到 WinSCP.com，请安装 WinSCP 或设置 SEMI_DFT_REMOTE_WINSCP_PATH。")
        return None

    def _resolve_winscp_private_key(self, config: RemoteClusterConfig) -> str:
        explicit = config.winscp_private_key_path
        if explicit:
            explicit_path = Path(explicit)
            if not explicit_path.exists():
                raise RuntimeError(f"指定的 WinSCP 私钥不存在: {explicit_path}")
            return str(explicit_path)

        if not config.ssh_key_path:
            raise RuntimeError("winscp 模式缺少私钥路径，请设置 SEMI_DFT_REMOTE_SSH_KEY。")

        source_key = Path(config.ssh_key_path)
        if not source_key.exists():
            raise RuntimeError(f"SSH 私钥不存在: {source_key}")

        if source_key.suffix.lower() == ".ppk":
            return str(source_key)

        winscp_path = self._resolve_winscp_path(config)
        secrets_dir = Path(__file__).resolve().parents[2] / ".secrets"
        secrets_dir.mkdir(parents=True, exist_ok=True)
        target_key = secrets_dir / f"{source_key.stem}.auto.ppk"

        if (
            target_key.exists()
            and target_key.stat().st_mtime >= source_key.stat().st_mtime
        ):
            return str(target_key)

        process = subprocess.run(
            [
                winscp_path,
                "/keygen",
                str(source_key),
                f"/output={target_key}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        if process.returncode != 0:
            message = process.stderr.strip() or process.stdout.strip() or "WinSCP 私钥转换失败"
            raise RuntimeError(message)

        return str(target_key)

    def _build_winscp_open_command(self, config: RemoteClusterConfig) -> str:
        private_key = self._resolve_winscp_private_key(config)
        if config.strict_host_key_checking:
            if not config.host_key:
                raise RuntimeError(
                    "winscp 模式启用严格主机检查时，必须设置 SEMI_DFT_REMOTE_HOST_KEY。"
                )
            host_key = config.host_key
        else:
            host_key = "*"

        return (
            f"open sftp://{config.user}@{config.host}:{config.port}/ "
            f"-privatekey={self._winscp_quote(private_key)} "
            f"-hostkey={self._winscp_quote(host_key)}"
        )

    @staticmethod
    def _extract_winscp_command_output(raw_output: str) -> str:
        raw_output = raw_output or ""
        ignored_prefixes = (
            "winscp>",
            "寻找主机",
            "连接到主机",
            "正在验证",
            "使用用户名",
            "使用公钥",
            "已验证",
            "正在开始会话",
            "会话已开始",
            "活动的会话",
            "主机超过",
            "请注意",
            "警告：",
            "中止(",
        )
        lines = []
        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(ignored_prefixes):
                continue
            if stripped in {"已开始。", "已验证。"}:
                continue
            lines.append(stripped)
        return "\n".join(lines).strip()

    @staticmethod
    def _build_ssh_command(config: RemoteClusterConfig, remote_command: str) -> list[str]:
        command = [
            "ssh",
            "-p",
            str(config.port),
            "-o",
            f"ConnectTimeout={config.connect_timeout}",
        ]
        if config.ssh_config_path:
            command.extend(["-F", config.ssh_config_path])
        elif config.ignore_local_ssh_config:
            command.extend(["-F", "NUL"])
        if not config.strict_host_key_checking:
            command.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL"])
        if config.ssh_key_path:
            command.extend(["-i", config.ssh_key_path, "-o", "IdentitiesOnly=yes"])
        command.extend([config.ssh_target(), remote_command])
        return command

    @staticmethod
    def _build_scp_command(config: RemoteClusterConfig, source: str, target: str) -> list[str]:
        command = [
            "scp",
            "-P",
            str(config.port),
        ]
        if config.ssh_config_path:
            command.extend(["-F", config.ssh_config_path])
        elif config.ignore_local_ssh_config:
            command.extend(["-F", "NUL"])
        if not config.strict_host_key_checking:
            command.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL"])
        if config.ssh_key_path:
            command.extend(["-i", config.ssh_key_path, "-o", "IdentitiesOnly=yes"])
        command.extend([source, target])
        return command

    @staticmethod
    def _parse_probe_stdout(stdout: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        current_key: str | None = None
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                parsed[key] = value.strip()
                current_key = key
            elif current_key:
                parsed[current_key] = (parsed.get(current_key, "") + "\n" + line).strip()
        return parsed

    @staticmethod
    def _quote(value: str) -> str:
        escaped = value.replace("'", "'\"'\"'")
        return f"'{escaped}'"

    @staticmethod
    def _winscp_quote(value: str) -> str:
        escaped = value.replace('"', '""')
        return f'"{escaped}"'

    @classmethod
    def _local_research_manifest(cls, root: Path) -> dict[str, dict[str, Any]]:
        manifest: dict[str, dict[str, Any]] = {}
        if not root.exists():
            return manifest
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(root)
            rel = rel_path.as_posix()
            if cls._skip_research_path(rel_path):
                continue
            raw = path.read_bytes()
            manifest[rel] = {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
        return manifest

    @classmethod
    def _skip_research_path(cls, rel_path: Path) -> bool:
        parts = set(rel_path.parts)
        if parts.intersection(cls.RESEARCH_EXCLUDE_DIRS):
            return True
        return rel_path.suffix.lower() in cls.RESEARCH_EXCLUDE_SUFFIXES

    def _remote_research_manifest(self, config: RemoteClusterConfig, backend: str, remote_dir: str) -> dict[str, Any]:
        command = (
            f"if [ ! -d {self._quote(remote_dir)} ]; then echo '__AETHER_RESEARCH_MISSING__'; exit 0; fi; "
            f"cd {self._quote(remote_dir)} && "
            "find . -type f ! -path './.aether_backups/*' ! -path './.omx/*' ! -path './__pycache__/*' ! -name '*.pdf' ! -name '*.pyc' "
            "-print0 | xargs -0 -r sha256sum"
        )
        process = self._run_remote_command(config, command, timeout=120, backend=backend)
        if process.returncode != 0:
            return {"status": "failed", "message": process.stderr.strip() or process.stdout.strip() or "远端 research manifest 读取失败。"}
        stdout = process.stdout or ""
        if "__AETHER_RESEARCH_MISSING__" in stdout:
            return {"status": "ok", "message": "远端 research 目录不存在，将在同步时创建。", "manifest": {}}
        manifest: dict[str, dict[str, Any]] = {}
        for line in stdout.splitlines():
            if not line.strip() or len(line) < 66:
                continue
            sha = line[:64]
            rel = line[66:] if line[64:66].isspace() else line[64:].strip()
            rel = rel[2:] if rel.startswith("./") else rel
            if rel:
                manifest[rel] = {"sha256": sha}
        return {"status": "ok", "message": "远端 research manifest 已读取。", "manifest": manifest}

    @staticmethod
    def _default_remote_research_dir(config: RemoteClusterConfig) -> str:
        return f"/home/{config.user}/research"
