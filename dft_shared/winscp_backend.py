from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IGNORED_PREFIXES = (
    "winscp>",
    "batch",
    "confirm",
    "reconnecttime",
    "寻找主机",
    "连接到主机",
    "正在验证",
    "使用用户名",
    "使用公钥",
    "已验证",
    "正在开始会话",
    "会话已开始",
    "活动的会话",
)


@dataclass
class WinSCPConnection:
    host: str
    user: str
    port: int
    winscp_path: Path
    private_key_path: Path
    host_key: str
    use_default_ini: bool = True


@dataclass
class WinSCPRunResult:
    stdout: str
    stderr: str
    returncode: int


def validate_connection_requirements(config: WinSCPConnection) -> None:
    missing: list[str] = []
    if not config.winscp_path.exists():
        missing.append(f"WinSCP.com 不存在: {config.winscp_path}")
    if not config.private_key_path.exists():
        missing.append(f"私钥不存在: {config.private_key_path}")
    if missing:
        raise RuntimeError("；".join(missing))


def run_winscp_script(config: WinSCPConnection, lines: list[str], *, timeout: int = 300) -> WinSCPRunResult:
    script = "\n".join(lines) + "\n"
    command = [str(config.winscp_path), "/stdin", "/nointeractiveinput"]
    if not config.use_default_ini:
        command.insert(1, "/ini=nul")
    process = subprocess.run(
        command,
        input=script,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return WinSCPRunResult(
        stdout=process.stdout or "",
        stderr=process.stderr or "",
        returncode=process.returncode,
    )


def build_open_command(config: WinSCPConnection) -> str:
    return (
        f"open sftp://{config.user}@{config.host}:{config.port}/ "
        f"-privatekey={winscp_quote(str(config.private_key_path))} "
        f"-hostkey={winscp_quote(config.host_key)}"
    )


def build_remote_sh_call(command: str) -> str:
    return "call sh -lc " + winscp_quote(command)


def extract_winscp_output(raw_output: str) -> str:
    lines: list[str] = []
    for raw_line in (raw_output or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(DEFAULT_IGNORED_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines)


def winscp_quote(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def winscp_single_quoted(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
