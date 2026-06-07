"""统一远程后端：一套接口，两种实现。

业务代码只用 RemoteBackend，不再关心 WinSCP/OpenSSH 的区别。

用法::

    from dft_shared.remote_backend import get_backend

    backend = get_backend()  # 根据 remote.toml / 环境变量自动选择
    backend.run("ls -la ~/aether-dft/relax")
    backend.upload(Path("POSCAR"), "~/aether-dft/relax/")
    backend.download("~/aether-dft/relax/CONTCAR", Path("./output"))
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .remote_config import RemoteConfig


# ==========================================================
# 协议
# ==========================================================

@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def check(self, msg: str = "远程操作失败") -> None:
        if not self.ok:
            detail = self.stderr.strip() or self.stdout.strip() or ""
            raise RuntimeError(f"{msg}: {detail}")


class RemoteBackend(Protocol):
    """远程操作协议——业务代码只依赖这个。"""

    def run(self, command: str, *, timeout: int = 300) -> RunResult: ...
    def upload(self, local: Path, remote: str, *, timeout: int = 300) -> RunResult: ...
    def download(self, remote: str, local_dir: Path, *, timeout: int = 300) -> RunResult: ...
    def mkdir(self, remote_dir: str, *, timeout: int = 300) -> RunResult: ...
    def exists(self, remote_path: str, *, timeout: int = 300) -> bool: ...
    def list_files(self, remote_dir: str, *, timeout: int = 300) -> list[str]: ...


# ==========================================================
# OpenSSH 实现（含 ControlMaster）
# ==========================================================

class OpenSSHBackend:
    """OpenSSH 后端，支持 ControlMaster 连接复用。"""

    def __init__(self, cfg: RemoteConfig):
        self.cfg = cfg
        self._control_path: str | None = None
        if cfg.use_control_master and os.name != "nt":
            import tempfile
            self._control_path = str(Path(tempfile.gettempdir()) / f"dft_ssh_{cfg.host}_{cfg.port}_%r")

    def _ssh_base_args(self) -> list[str]:
        if self.cfg.openssh_config and self.cfg.openssh_config.exists():
            args = [
                "-F", str(self.cfg.openssh_config),
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
            ]
            if self._control_path:
                args.extend([
                    "-o", f"ControlPath={self._control_path}",
                    "-o", "ControlMaster=auto",
                    "-o", "ControlPersist=120",
                ])
            return args
        from .openssh_backend import prepare_known_hosts_file, prepare_private_key

        prepared_key = prepare_private_key(self.cfg.openssh_private_key)
        known_hosts = prepare_known_hosts_file(
            type("_Conn", (), {
                "host": self.cfg.host,
                "known_hosts_line": self.cfg.openssh_known_hosts_line,
            })()
        )
        args = [
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", "ConnectTimeout=10",
            "-p", str(self.cfg.port),
            "-i", str(prepared_key),
        ]
        if self._control_path:
            args.extend([
                "-o", f"ControlPath={self._control_path}",
                "-o", "ControlMaster=auto",
                "-o", "ControlPersist=120",
            ])
        return args

    def run(self, command: str, *, timeout: int = 300) -> RunResult:
        args = [str(self.cfg.ssh_path)] + self._ssh_base_args() + [
            self._ssh_target(), command,
        ]
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, check=False)
        return RunResult(p.stdout or "", p.stderr or "", p.returncode)

    def upload(self, local: Path, remote: str, *, timeout: int = 300) -> RunResult:
        args = [str(self.cfg.scp_path)] + self._scp_opts() + [
            str(local), f"{self._scp_target()}:{remote}",
        ]
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, check=False)
        return RunResult(p.stdout or "", p.stderr or "", p.returncode)

    def download(self, remote: str, local_dir: Path, *, timeout: int = 300) -> RunResult:
        local_dir.mkdir(parents=True, exist_ok=True)
        args = [str(self.cfg.scp_path)] + self._scp_opts() + [
            f"{self._scp_target()}:{remote}", str(local_dir),
        ]
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, check=False)
        return RunResult(p.stdout or "", p.stderr or "", p.returncode)

    def mkdir(self, remote_dir: str, *, timeout: int = 300) -> RunResult:
        return self.run(f"mkdir -p {shell_quote(remote_dir)}", timeout=timeout)

    def exists(self, remote_path: str, *, timeout: int = 300) -> bool:
        result = self.run(f"test -e {shell_quote(remote_path)}", timeout=timeout)
        return result.returncode == 0

    def list_files(self, remote_dir: str, *, timeout: int = 300) -> list[str]:
        result = self.run(f"find {shell_quote(remote_dir)} -maxdepth 1 -type f -printf '%f\\n'", timeout=timeout)
        result.check("列出远端文件失败")
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _scp_opts(self) -> list[str]:
        if self.cfg.openssh_config and self.cfg.openssh_config.exists():
            opts = [
                "-F", str(self.cfg.openssh_config),
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
            ]
            if self._control_path:
                opts.extend([
                    "-o", f"ControlPath={self._control_path}",
                ])
            return opts
        from .openssh_backend import prepare_known_hosts_file, prepare_private_key

        prepared_key = prepare_private_key(self.cfg.openssh_private_key)
        known_hosts = prepare_known_hosts_file(
            type("_Conn", (), {
                "host": self.cfg.host,
                "known_hosts_line": self.cfg.openssh_known_hosts_line,
            })()
        )
        opts = [
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", "ConnectTimeout=10",
            "-P", str(self.cfg.port),
            "-i", str(prepared_key),
        ]
        if self._control_path:
            opts.extend([
                "-o", f"ControlPath={self._control_path}",
            ])
        return opts

    def _ssh_target(self) -> str:
        return self.cfg.openssh_host_alias or f"{self.cfg.user}@{self.cfg.host}"

    def _scp_target(self) -> str:
        return self.cfg.openssh_host_alias or f"{self.cfg.user}@{self.cfg.host}"


# ==========================================================
# WinSCP 实现
# ==========================================================

class WinSCPBackend:
    """WinSCP 后端。"""

    def __init__(self, cfg: RemoteConfig):
        self.cfg = cfg

    def _open_cmd(self) -> str:
        from .winscp_backend import winscp_quote
        return (
            f"open sftp://{self.cfg.user}@{self.cfg.host}:{self.cfg.port}/ "
            f"-privatekey={winscp_quote(str(self.cfg.winscp_private_key))} "
            f"-hostkey={winscp_quote(self.cfg.winscp_host_key)}"
        )

    def _exec_script(self, lines: list[str], timeout: int) -> RunResult:
        script = "\n".join(["option batch abort", "option confirm off", self._open_cmd()] + lines + ["exit"]) + "\n"
        command = [str(self.cfg.winscp_path), "/stdin", "/nointeractiveinput"]
        if not getattr(self.cfg, "winscp_use_default_ini", False):
            command.insert(1, "/ini=nul")
        p = subprocess.run(
            command,
            input=script, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False,
        )
        return RunResult(p.stdout or "", p.stderr or "", p.returncode)

    def run(self, command: str, *, timeout: int = 300) -> RunResult:
        from .winscp_backend import winscp_quote
        return self._exec_script([f"call sh -lc {winscp_quote(command)}"], timeout)

    def upload(self, local: Path, remote: str, *, timeout: int = 300) -> RunResult:
        from .winscp_backend import winscp_quote
        return self._exec_script(
            [f"put -transfer=binary -nopreservetime {winscp_quote(str(local))} {winscp_quote(remote)}"],
            timeout,
        )

    def download(self, remote: str, local_dir: Path, *, timeout: int = 300) -> RunResult:
        from .winscp_backend import winscp_quote
        local_dir.mkdir(parents=True, exist_ok=True)
        return self._exec_script(
            [f"get -transfer=binary -nopreservetime {winscp_quote(remote)} {winscp_quote(str(local_dir) + os.sep)}"],
            timeout,
        )

    def mkdir(self, remote_dir: str, *, timeout: int = 300) -> RunResult:
        from .winscp_backend import winscp_quote
        command = f"mkdir -p {shell_quote(remote_dir)}"
        return self._exec_script([f"call sh -lc {winscp_quote(command)}"], timeout)

    def exists(self, remote_path: str, *, timeout: int = 300) -> bool:
        from .winscp_backend import winscp_quote, extract_winscp_output
        command = f"test -e {shell_quote(remote_path)} && echo YES || echo NO"
        result = self._exec_script([f"call sh -lc {winscp_quote(command)}"], timeout)
        output = extract_winscp_output(result.stdout)
        return "YES" in output.splitlines()

    def list_files(self, remote_dir: str, *, timeout: int = 300) -> list[str]:
        from .winscp_backend import winscp_quote, extract_winscp_output
        command = f"find {shell_quote(remote_dir)} -maxdepth 1 -type f -printf '%f\\n'"
        result = self._exec_script(
            [f"call sh -lc {winscp_quote(command)}"],
            timeout,
        )
        result.check("列出远端文件失败")
        output = extract_winscp_output(result.stdout)
        return [line.strip() for line in output.splitlines() if line.strip()]


# ==========================================================
# 工厂
# ==========================================================


def get_backend(cfg: RemoteConfig | None = None) -> RemoteBackend:
    """根据配置返回对应后端实例。"""
    if cfg is None:
        cfg = RemoteConfig.load()
    if cfg.backend == "winscp":
        from .winscp_backend import WinSCPConnection, validate_connection_requirements

        validate_connection_requirements(
            WinSCPConnection(
                host=cfg.host,
                user=cfg.user,
                port=cfg.port,
                winscp_path=cfg.winscp_path,
                private_key_path=cfg.winscp_private_key,
                host_key=cfg.winscp_host_key,
                use_default_ini=cfg.winscp_use_default_ini,
            )
        )
        return WinSCPBackend(cfg)
    from .openssh_backend import OpenSSHConnection, validate_connection_requirements

    validate_connection_requirements(
        OpenSSHConnection(
            host=cfg.host,
            user=cfg.user,
            port=cfg.port,
            ssh_path=cfg.ssh_path,
            scp_path=cfg.scp_path,
            private_key_path=cfg.openssh_private_key,
            known_hosts_line=cfg.openssh_known_hosts_line,
            config_path=cfg.openssh_config,
            host_alias=cfg.openssh_host_alias,
        )
    )
    return OpenSSHBackend(cfg)


def shell_quote(value: str) -> str:
    """POSIX shell 引用（适用于远程命令拼接）。"""
    return "'" + value.replace("'", "'\"'\"'") + "'"
