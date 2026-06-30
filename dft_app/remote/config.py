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
    remote_potcar_roots: tuple[str, ...] = ()

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
                or str(local_profile.get("ssh_host_alias") or "")
            ).strip()
            ssh_config_path = (
                os.getenv("SEMI_DFT_REMOTE_SSH_CONFIG")
                or os.getenv("AETHER_DFT_CLUSTER_SSH_CONFIG")
                or str(local_profile.get("ssh_config_path") or "")
                or str(cls._default_ssh_config_path())
            )
            if alias and Path(ssh_config_path).exists():
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
                        remote_potcar_roots=cls._remote_potcar_roots(local_profile),
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
                "请先运行 `aether cluster import-ssh-config --source <你的 SSH config 路径> --alias <Host 别名>`，"
                "或设置 AETHER_DFT_CLUSTER_HOST / AETHER_DFT_CLUSTER_USER / AETHER_DFT_CLUSTER_REMOTE_BASE_DIR。"
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
            remote_potcar_roots=cls._remote_potcar_roots(local_profile),
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
            "remote_potcar_roots": list(self.remote_potcar_roots),
        }


    @classmethod
    def _remote_potcar_roots(cls, local_profile: dict[str, Any]) -> tuple[str, ...]:
        raw_env = os.getenv("SEMI_DFT_REMOTE_POTCAR_ROOTS") or os.getenv("AETHER_DFT_REMOTE_POTCAR_ROOTS")
        raw_profile = local_profile.get("remote_potcar_roots")
        items: list[str] = []
        if raw_env:
            items.extend(part.strip() for part in str(raw_env).replace(";", ":").split(":"))
        if isinstance(raw_profile, list):
            items.extend(str(item).strip() for item in raw_profile)
        elif isinstance(raw_profile, str):
            items.extend(part.strip() for part in raw_profile.replace(";", ":").split(":"))
        cleaned: list[str] = []
        for item in items:
            if not item or item in cleaned:
                continue
            cleaned.append(item)
        return tuple(cleaned)

    @staticmethod
    def _app_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _local_profile_path(cls) -> Path:
        return cls._app_root() / ".secrets" / "cluster.local.json"

    @classmethod
    def _default_ssh_config_path(cls) -> Path:
        project_copy = cls._app_root() / ".secrets" / "ssh_config"
        return project_copy

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
    """Parse the small subset of OpenSSH config needed by AETHER-DFT.

    OpenSSH uses the first obtained value for each parameter when multiple
    matching Host stanzas exist.  Mirroring that behavior matters for duplicate
    aliases such as a WAN/LAN pair with the same Host name.
    """
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
            continue
        if not current_matches or not key or not value:
            continue
        if key not in parsed:
            parsed[key] = value
    return parsed or None


def parse_ssh_config_hosts(path: Path) -> list[dict[str, Any]]:
    """List concrete Host entries from an OpenSSH config.

    This intentionally handles only the local-user-facing subset AETHER needs:
    aliases, HostName, User, Port and whether an IdentityFile is configured.
    Wildcard patterns are ignored because they are not directly selectable
    cluster aliases.
    """
    entries: list[dict[str, Any]] = []
    current_aliases: list[str] = []
    current: dict[str, str] = {}

    def flush() -> None:
        if not current_aliases:
            return
        for alias in current_aliases:
            if any(ch in alias for ch in "*?[]!"):
                continue
            entries.append(
                {
                    "alias": alias,
                    "hostname": current.get("hostname") or alias,
                    "user": current.get("user") or alias,
                    "port": current.get("port", "22"),
                    "identityfile_configured": bool(current.get("identityfile")),
                }
            )

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        key = key.strip().lower()
        value = value.strip()
        if key == "host":
            flush()
            current_aliases = [item for item in value.split() if item]
            current = {}
            continue
        if current_aliases and key and value and key not in current:
            current[key] = value
    flush()

    counts: dict[str, int] = {}
    for entry in entries:
        alias = str(entry["alias"])
        counts[alias] = counts.get(alias, 0) + 1
        entry["duplicate_alias"] = counts[alias] > 1
        entry["occurrence"] = counts[alias]
    totals: dict[str, int] = {}
    for entry in entries:
        alias = str(entry["alias"])
        totals[alias] = totals.get(alias, 0) + 1
    for entry in entries:
        entry["duplicate_count"] = totals[str(entry["alias"])]
    return entries


def _cluster_profiles_path() -> Path:
    return RemoteClusterConfig._app_root() / ".secrets" / "clusters.local.json"


def list_local_cluster_profiles() -> dict[str, Any]:
    """Return project-local cluster aliases discovered from .secrets/ssh_config."""
    ssh_config_path = RemoteClusterConfig._default_ssh_config_path()
    active = RemoteClusterConfig._load_local_profile()
    hosts = parse_ssh_config_hosts(ssh_config_path) if ssh_config_path.exists() else []
    return {
        "status": "ok" if hosts else "missing",
        "ssh_config_path": str(ssh_config_path) if ssh_config_path.exists() else "",
        "active_alias": active.get("ssh_host_alias"),
        "active_remote_base_dir": active.get("remote_base_dir"),
        "clusters": hosts,
        "message": "已识别项目内 SSH config Host。" if hosts else "项目内尚未导入 SSH config。",
    }


def use_local_cluster_profile(alias: str, *, remote_base_dir: str | None = None) -> dict[str, Any]:
    """Set the active project-local cluster alias."""
    alias = str(alias or "").strip()
    if not alias:
        raise ValueError("cluster alias 不能为空")
    ssh_config_path = RemoteClusterConfig._default_ssh_config_path()
    if not ssh_config_path.exists():
        raise ValueError("项目内尚未导入 SSH config；请先运行 cluster import-ssh-config。")
    parsed = parse_ssh_config_host(ssh_config_path, alias)
    if not parsed:
        raise ValueError(f"项目 SSH config 中未找到 Host {alias}")
    user = parsed.get("user") or alias
    payload = {
        "ssh_host_alias": alias,
        "ssh_config_path": str(ssh_config_path),
        "remote_base_dir": remote_base_dir or f"/home/{user}/aether-dft-runs",
        "backend": "openssh",
        "strict_host_key_checking": True,
    }
    profile_path = RemoteClusterConfig._local_profile_path()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "profile_path": str(profile_path),
        "ssh_config_path": str(ssh_config_path),
        "ssh_host_alias": alias,
        "remote_base_dir": payload["remote_base_dir"],
        "parsed_host": {
            "hostname": parsed.get("hostname"),
            "user": parsed.get("user"),
            "port": parsed.get("port", "22"),
            "identityfile_configured": bool(parsed.get("identityfile")),
        },
    }


def config_for_local_cluster_alias(alias: str) -> RemoteClusterConfig:
    """Build a RemoteClusterConfig for a project-local SSH Host alias.

    Unlike ``use_local_cluster_profile()``, this does not mutate the active
    cluster.  Model-facing tools use it to honor natural language such as
    "用 rxqin 看队列" without requiring the user to switch global state first.
    """
    alias = str(alias or "").strip()
    if not alias:
        return RemoteClusterConfig.from_env()
    ssh_config_path = RemoteClusterConfig._default_ssh_config_path()
    if not ssh_config_path.exists():
        raise ValueError("项目内尚未导入 SSH config；请先运行 cluster import-ssh-config。")
    parsed = parse_ssh_config_host(ssh_config_path, alias)
    if not parsed:
        raise ValueError(f"项目 SSH config 中未找到 Host {alias}")
    user = str(parsed.get("user") or alias)
    local_profile = RemoteClusterConfig._load_local_profile()
    active_alias = str(local_profile.get("ssh_host_alias") or "").strip()
    remote_base_dir = (
        str(local_profile.get("remote_base_dir") or "").strip()
        if active_alias == alias
        else ""
    ) or f"/home/{user}/aether-dft-runs"
    return RemoteClusterConfig(
        host=str(parsed.get("hostname") or alias),
        user=user,
        remote_base_dir=remote_base_dir,
        port=int(str(parsed.get("port") or "22")),
        backend="openssh",
        ssh_key_path=str(parsed.get("identityfile") or "") or None,
        strict_host_key_checking=True,
        ignore_local_ssh_config=False,
        ssh_config_path=str(ssh_config_path),
        ssh_host_alias=alias,
        remote_potcar_roots=RemoteClusterConfig._remote_potcar_roots(local_profile),
    )


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
    hosts = parse_ssh_config_hosts(target_config)
    profiles_path = _cluster_profiles_path()
    profiles_path.write_text(
        json.dumps({"ssh_config_path": str(target_config), "clusters": hosts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected = use_local_cluster_profile(alias, remote_base_dir=remote_base_dir)
    selected["clusters_path"] = str(profiles_path)
    selected["cluster_count"] = len(hosts)
    selected["clusters"] = hosts
    return {
        "status": "ok",
        **selected,
    }
