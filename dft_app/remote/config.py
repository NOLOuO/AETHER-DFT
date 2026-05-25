from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


@dataclass
class RemoteClusterConfig:
    host: str
    user: str
    remote_base_dir: str
    port: int = 22
    backend: str = "auto"
    ssh_key_path: str | None = None
    host_key: str | None = None
    strict_host_key_checking: bool = True
    connect_timeout: int = 15
    ignore_local_ssh_config: bool = True
    ssh_config_path: str | None = None
    ssh_host_alias: str | None = None
    winscp_path: str | None = None
    winscp_private_key_path: str | None = None
    winscp_ini_path: str = "nul"

    @classmethod
    def from_env(cls) -> "RemoteClusterConfig":
        local_profile = cls._load_local_profile()
        host = os.getenv("SEMI_DFT_REMOTE_HOST") or os.getenv("AETHER_DFT_CLUSTER_HOST")
        user = os.getenv("SEMI_DFT_REMOTE_USER") or os.getenv("AETHER_DFT_CLUSTER_USER")
        remote_base_dir = (
            os.getenv("SEMI_DFT_REMOTE_BASE_DIR")
            or os.getenv("AETHER_DFT_CLUSTER_REMOTE_BASE_DIR")
        )

        if not (host and user):
            alias = (
                os.getenv("SEMI_DFT_REMOTE_SSH_ALIAS")
                or os.getenv("AETHER_DFT_CLUSTER_ALIAS")
                or str(local_profile.get("ssh_host_alias") or "szhang")
            )
            ssh_config_path = (
                os.getenv("SEMI_DFT_REMOTE_SSH_CONFIG")
                or os.getenv("AETHER_DFT_CLUSTER_SSH_CONFIG")
                or str(local_profile.get("ssh_config_path") or "")
                or str(cls._default_ssh_config_path())
            )
            if Path(ssh_config_path).exists():
                parsed = parse_ssh_config_host(Path(ssh_config_path), alias)
                if parsed:
                    host = host or str(parsed.get("hostname") or alias)
                    user = user or str(parsed.get("user") or alias)
                    remote_base_dir = (
                        remote_base_dir
                        or str(local_profile.get("remote_base_dir") or "")
                        or f"/home/{user}/aether-dft-runs"
                    )
                    return cls(
                        host=host,
                        user=user,
                        remote_base_dir=remote_base_dir,
                        port=int(str(parsed.get("port") or os.getenv("SEMI_DFT_REMOTE_PORT", "22"))),
                        backend=(
                            os.getenv("SEMI_DFT_REMOTE_BACKEND")
                            or str(local_profile.get("backend") or "openssh")
                        ).strip().lower(),
                        ssh_key_path=os.getenv("SEMI_DFT_REMOTE_SSH_KEY")
                        or str(parsed.get("identityfile") or "") or None,
                        strict_host_key_checking=cls._env_bool(
                            "SEMI_DFT_REMOTE_STRICT_HOST_KEY_CHECKING",
                            default=bool(local_profile.get("strict_host_key_checking", True)),
                        ),
                        connect_timeout=int(os.getenv("SEMI_DFT_REMOTE_CONNECT_TIMEOUT", "15")),
                        ignore_local_ssh_config=False,
                        ssh_config_path=ssh_config_path,
                        ssh_host_alias=alias,
                        winscp_path=os.getenv("SEMI_DFT_REMOTE_WINSCP_PATH"),
                        winscp_private_key_path=os.getenv("SEMI_DFT_REMOTE_WINSCP_PRIVATE_KEY"),
                        winscp_ini_path=os.getenv("SEMI_DFT_REMOTE_WINSCP_INI_PATH", "nul"),
                    )

        missing = [
            name
            for name, value in (
                ("SEMI_DFT_REMOTE_HOST", host),
                ("SEMI_DFT_REMOTE_USER", user),
                ("SEMI_DFT_REMOTE_BASE_DIR", remote_base_dir),
            )
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"远程模式缺少必要环境变量: {joined}，且未找到可用 SSH alias 配置。"
                "可运行 `aether-dft cluster import-ssh-config --source C:\\Users\\24651\\.ssh\\config --alias szhang`。"
            )

        port = int(os.getenv("SEMI_DFT_REMOTE_PORT", "22"))
        strict = cls._env_bool("SEMI_DFT_REMOTE_STRICT_HOST_KEY_CHECKING", default=True)

        return cls(
            host=host,
            user=user,
            remote_base_dir=remote_base_dir,
            port=port,
            backend=os.getenv("SEMI_DFT_REMOTE_BACKEND", "auto").strip().lower(),
            ssh_key_path=os.getenv("SEMI_DFT_REMOTE_SSH_KEY"),
            host_key=os.getenv("SEMI_DFT_REMOTE_HOST_KEY"),
            strict_host_key_checking=strict,
            connect_timeout=int(os.getenv("SEMI_DFT_REMOTE_CONNECT_TIMEOUT", "15")),
            ignore_local_ssh_config=os.getenv(
                "SEMI_DFT_REMOTE_IGNORE_LOCAL_SSH_CONFIG", "true"
            )
            .strip()
            .lower()
            not in {"0", "false", "no"},
            winscp_path=os.getenv("SEMI_DFT_REMOTE_WINSCP_PATH"),
            winscp_private_key_path=os.getenv("SEMI_DFT_REMOTE_WINSCP_PRIVATE_KEY"),
            winscp_ini_path=os.getenv("SEMI_DFT_REMOTE_WINSCP_INI_PATH", "nul"),
        )

    def ssh_target(self) -> str:
        if self.ssh_host_alias:
            return self.ssh_host_alias
        return f"{self.user}@{self.host}"

    def remote_run_root(self, task_id: str, run_id: str) -> str:
        return str(PurePosixPath(self.remote_base_dir) / task_id / run_id)

    def public_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "user": self.user,
            "port": self.port,
            "remote_base_dir": self.remote_base_dir,
            "backend": self.backend,
            "ssh_host_alias": self.ssh_host_alias,
            "ssh_config_path": self.ssh_config_path,
            "ssh_key_path_configured": bool(self.ssh_key_path),
            "strict_host_key_checking": self.strict_host_key_checking,
            "ignore_local_ssh_config": self.ignore_local_ssh_config,
        }

    @staticmethod
    def _app_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _local_profile_path(cls) -> Path:
        return cls._app_root() / ".secrets" / "cluster.local.json"

    @classmethod
    def _default_ssh_config_path(cls) -> Path:
        project_copy = cls._app_root() / ".secrets" / "ssh_config"
        if project_copy.exists():
            return project_copy
        return Path.home() / ".ssh" / "config"

    @classmethod
    def _load_local_profile(cls) -> dict[str, Any]:
        path = cls._local_profile_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _env_bool(name: str, *, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no"}


def parse_ssh_config_host(path: Path, alias: str) -> dict[str, str] | None:
    """Parse the small subset of OpenSSH config needed by AETHER-DFT."""
    target = alias.lower()
    current_matches = False
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        key = key.strip().lower()
        value = value.strip()
        if key == "host":
            patterns = [item.lower() for item in value.split()]
            current_matches = target in patterns
            if current_matches:
                parsed = {}
            continue
        if not current_matches or not key or not value:
            continue
        parsed[key] = value
    return parsed or None


def write_local_cluster_profile(
    *,
    source_ssh_config: Path,
    alias: str,
    remote_base_dir: str | None = None,
) -> dict[str, Any]:
    """Copy SSH config into .secrets and persist the default cluster alias."""
    app_root = RemoteClusterConfig._app_root()
    secrets_dir = app_root / ".secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    target_config = secrets_dir / "ssh_config"
    target_config.write_text(
        source_ssh_config.read_text(encoding="utf-8", errors="replace"),
        encoding="utf-8",
    )
    parsed = parse_ssh_config_host(target_config, alias)
    if not parsed:
        raise ValueError(f"SSH config 中未找到 Host {alias}")
    user = parsed.get("user") or alias
    payload = {
        "ssh_host_alias": alias,
        "ssh_config_path": str(target_config),
        "remote_base_dir": remote_base_dir or f"/home/{user}/aether-dft-runs",
        "backend": "openssh",
        "strict_host_key_checking": True,
    }
    profile_path = secrets_dir / "cluster.local.json"
    profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "profile_path": str(profile_path),
        "ssh_config_path": str(target_config),
        "ssh_host_alias": alias,
        "remote_base_dir": payload["remote_base_dir"],
        "parsed_host": {
            "hostname": parsed.get("hostname"),
            "user": parsed.get("user"),
            "port": parsed.get("port", "22"),
            "identityfile_configured": bool(parsed.get("identityfile")),
        },
    }
