from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpenSSHConnection:
    host: str
    user: str
    port: int
    ssh_path: Path
    scp_path: Path
    private_key_path: Path
    known_hosts_line: str
    config_path: Path | None = None
    host_alias: str | None = None


@dataclass
class OpenSSHRunResult:
    stdout: str
    stderr: str
    returncode: int


def validate_connection_requirements(config: OpenSSHConnection) -> None:
    missing: list[str] = []
    if not config.ssh_path.exists():
        missing.append(f"ssh.exe 不存在: {config.ssh_path}")
    if not config.scp_path.exists():
        missing.append(f"scp.exe 不存在: {config.scp_path}")
    using_config = bool(config.config_path and config.config_path.exists())
    if not using_config:
        if not config.private_key_path.exists():
            missing.append(f"OpenSSH 私钥不存在: {config.private_key_path}")
        if not config.known_hosts_line.strip():
            missing.append("OpenSSH known_hosts 指纹为空")
    if missing:
        raise RuntimeError("；".join(missing))


def run_ssh_command(config: OpenSSHConnection, command: str, *, timeout: int = 300) -> OpenSSHRunResult:
    prepared_key = prepare_private_key(config.private_key_path)
    known_hosts = prepare_known_hosts_file(config)
    process = subprocess.run(
        [
            str(config.ssh_path),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            "ConnectTimeout=10",
            "-p",
            str(config.port),
            "-i",
            str(prepared_key),
            f"{config.user}@{config.host}",
            command,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return OpenSSHRunResult(
        stdout=process.stdout or "",
        stderr=process.stderr or "",
        returncode=process.returncode,
    )


def scp_download(
    config: OpenSSHConnection,
    remote_path: str,
    local_dir: Path,
    *,
    timeout: int = 300,
) -> OpenSSHRunResult:
    prepared_key = prepare_private_key(config.private_key_path)
    known_hosts = prepare_known_hosts_file(config)
    local_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            str(config.scp_path),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            "ConnectTimeout=10",
            "-P",
            str(config.port),
            "-i",
            str(prepared_key),
            f"{config.user}@{config.host}:{remote_path}",
            str(local_dir),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return OpenSSHRunResult(
        stdout=process.stdout or "",
        stderr=process.stderr or "",
        returncode=process.returncode,
    )


def scp_upload(
    config: OpenSSHConnection,
    local_path: Path,
    remote_path: str,
    *,
    timeout: int = 300,
) -> OpenSSHRunResult:
    prepared_key = prepare_private_key(config.private_key_path)
    known_hosts = prepare_known_hosts_file(config)
    process = subprocess.run(
        [
            str(config.scp_path),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            "ConnectTimeout=10",
            "-P",
            str(config.port),
            "-i",
            str(prepared_key),
            str(local_path),
            f"{config.user}@{config.host}:{remote_path}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return OpenSSHRunResult(
        stdout=process.stdout or "",
        stderr=process.stderr or "",
        returncode=process.returncode,
    )


def prepare_private_key(source_key: Path) -> Path:
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    target = ssh_dir / f"codex_{source_key.stem}_{os.getpid()}"
    if target.exists():
        return target
    shutil.copy2(source_key, target)
    _restrict_private_key_acl(target)
    return target


def prepare_known_hosts_file(config: OpenSSHConnection) -> Path:
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    target = ssh_dir / f"codex_known_hosts_{config.host.replace('.', '_')}"
    target.write_text(config.known_hosts_line.strip() + "\n", encoding="ascii")
    return target


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _restrict_private_key_acl(path: Path) -> None:
    whoami = subprocess.run(
        ["whoami"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    identity = (whoami.stdout or "").strip()
    if not identity:
        raise RuntimeError("无法获取当前 Windows 用户，不能为 OpenSSH 私钥设置权限。")

    commands = [
        ["icacls", str(path), "/inheritance:r"],
        ["icacls", str(path), "/grant:r", f"{identity}:R"],
    ]
    for command in commands:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if process.returncode != 0:
            detail = (process.stderr or process.stdout or "").strip()
            raise RuntimeError(f"设置 OpenSSH 私钥权限失败: {detail}")
