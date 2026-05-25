from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from dft_app.models import ExperimentSpec, PhaseStatus, PipelinePhase, RunRecord, RunStatus
from dft_app.remote.config import RemoteClusterConfig
from dft_app.runner.slurm_runner import SlurmRunner


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

        run_root = Path(run_record.run_root)
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
            artifacts=[str(run_root / "inputs" / "job.slurm")],
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

        monitor_command = (
            f"squeue -j {run_record.scheduler_job_id} -h -o %T || "
            f"sacct -j {run_record.scheduler_job_id} --format=State --noheader"
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
