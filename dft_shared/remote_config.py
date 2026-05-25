"""统一远程连接配置。

所有工具（cluster_submit、remote_fetch、mace_ts_search）共享同一份配置。
优先读 TOML 配置文件，其次环境变量，最后用硬编码默认值。
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "remote.toml"
LOGGER = logging.getLogger(__name__)


@dataclass
class RemoteConfig:
    """集群连接的全部参数——只定义一次。"""

    # 连接
    host: str = "59.77.33.28"
    user: str = "szhang"
    port: int = 22
    backend: str = "winscp"           # "openssh" | "winscp"

    # OpenSSH
    ssh_path: Path = field(default_factory=lambda: _detect_existing_path([shutil.which("ssh"), r"C:\Windows\System32\OpenSSH\ssh.exe"]))
    scp_path: Path = field(default_factory=lambda: _detect_existing_path([shutil.which("scp"), r"C:\Windows\System32\OpenSSH\scp.exe"]))
    openssh_config: Path | None = field(default_factory=lambda: _detect_optional_path([r"C:\Users\24651\.ssh\config", str(Path.home() / ".ssh" / "config")]))
    openssh_host_alias: str | None = None
    openssh_private_key: Path = Path.home() / "Desktop" / "KEY" / "id_rsa_szhang"
    openssh_known_hosts_line: str = (
        "59.77.33.28 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOyuGObXd/M7QNizWKOpcmXxmapvw3j6Ms+GeKSfE68D"
    )

    # WinSCP
    winscp_path: Path = field(default_factory=lambda: _detect_existing_path([r"D:\WinSCP\WinSCP.com", r"C:\Program Files (x86)\WinSCP\WinSCP.com", r"C:\Program Files\WinSCP\WinSCP.com"]))
    winscp_private_key: Path = _PROJECT_ROOT.parent / "auto_dft" / ".secrets" / "id_rsa_szhang.auto.ppk"
    winscp_host_key: str = "ssh-ed25519 255 bOiJB7GAYDPeBA4MX/H0wM/pGZS+H5G01TZtYMYiGVM="
    winscp_use_default_ini: bool = True
    winscp_session_name: str | None = None

    # 远程目录预设
    remote_dirs: dict[str, str] = field(default_factory=lambda: {
        "relax": "/home/szhang/DFTauto/relax",
        "ts": "/home/szhang/DFTauto/ts",
        "clean": "/home/szhang/clean",
    })

    # 远程工具
    pos2file_path: str = "/home/szhang/bin/pos2file"
    qvasp_path: str = "/usr/local/bin/qvasp"
    remote_timeout_seconds: int = 300

    # SSH ControlMaster
    use_control_master: bool = True

    # 本地环境
    conda_activate: str = "D:/miniconda3/Scripts/activate"
    mace_env: str = "mace"
    analysis_env: str = "p312env"

    @classmethod
    def load(cls) -> RemoteConfig:
        """加载配置：TOML → 环境变量 → 默认值。"""
        cfg = cls()

        # 尝试从 TOML 加载
        if _CONFIG_PATH.exists():
            cfg = _load_from_toml(cfg)

        # 环境变量覆盖
        cfg = _apply_env_overrides(cfg)

        if cfg.openssh_config and cfg.openssh_config.exists() and not cfg.openssh_host_alias:
            inferred_alias = _infer_host_alias_from_config(cfg.openssh_config, host=cfg.host, user=cfg.user)
            if inferred_alias:
                cfg.openssh_host_alias = inferred_alias

        return cfg

    def get_remote_dir(self, preset: str = "relax") -> str:
        return self.remote_dirs.get(preset, self.remote_dirs.get("relax", "/home/szhang/DFTauto/relax"))


def _load_from_toml(cfg: RemoteConfig) -> RemoteConfig:
    """从 TOML 文件加载配置。"""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return cfg

    try:
        data = tomllib.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("读取 remote.toml 失败，将回退到默认配置: %s", exc)
        return cfg

    conn = data.get("connection", {})
    if conn.get("host"):
        cfg.host = conn["host"]
    if conn.get("user"):
        cfg.user = conn["user"]
    if conn.get("port"):
        cfg.port = int(conn["port"])
    if conn.get("backend"):
        cfg.backend = conn["backend"]

    openssh = data.get("openssh", {})
    if openssh.get("ssh_path"):
        cfg.ssh_path = Path(openssh["ssh_path"])
    if openssh.get("scp_path"):
        cfg.scp_path = Path(openssh["scp_path"])
    if openssh.get("config"):
        cfg.openssh_config = Path(openssh["config"])
    if openssh.get("host_alias"):
        cfg.openssh_host_alias = openssh["host_alias"]
    if openssh.get("private_key"):
        cfg.openssh_private_key = Path(openssh["private_key"])
    if openssh.get("known_hosts_line"):
        cfg.openssh_known_hosts_line = openssh["known_hosts_line"]

    winscp = data.get("winscp", {})
    if winscp.get("path"):
        cfg.winscp_path = Path(winscp["path"])
    if winscp.get("private_key"):
        cfg.winscp_private_key = Path(winscp["private_key"])
    if winscp.get("host_key"):
        cfg.winscp_host_key = winscp["host_key"]
    if "use_default_ini" in winscp:
        cfg.winscp_use_default_ini = bool(winscp["use_default_ini"])
    if winscp.get("session_name"):
        cfg.winscp_session_name = str(winscp["session_name"]).strip() or None

    dirs = data.get("remote_dirs", {})
    if dirs:
        cfg.remote_dirs.update(dirs)

    tools = data.get("tools", {})
    if tools.get("pos2file"):
        cfg.pos2file_path = tools["pos2file"]
    if tools.get("qvasp"):
        cfg.qvasp_path = tools["qvasp"]

    local = data.get("local", {})
    if local.get("conda_activate"):
        cfg.conda_activate = local["conda_activate"]
    if local.get("mace_env"):
        cfg.mace_env = local["mace_env"]
    if local.get("analysis_env"):
        cfg.analysis_env = local["analysis_env"]

    return cfg


def _apply_env_overrides(cfg: RemoteConfig) -> RemoteConfig:
    """环境变量覆盖（兼容旧环境变量名）。"""
    env_map = {
        "DFT_REMOTE_HOST": "host",
        "DFT_REMOTE_USER": "user",
        "DFT_REMOTE_PORT": "port",
        "DFT_REMOTE_BACKEND": "backend",
        # 旧变量名兼容
        "XSD_PREOPT_REMOTE_HOST": "host",
        "XSD_PREOPT_REMOTE_USER": "user",
        "VASP_RESULT_REMOTE_HOST": "host",
        "VASP_RESULT_REMOTE_USER": "user",
    }
    for env_key, attr in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            if attr == "port":
                cfg.port = int(val)
            else:
                setattr(cfg, attr, val)

    for env_key, attr in [
        ("DFT_REMOTE_SSH_PATH", "ssh_path"),
        ("DFT_REMOTE_SCP_PATH", "scp_path"),
        ("DFT_REMOTE_OPENSSH_CONFIG", "openssh_config"),
        ("DFT_REMOTE_OPENSSH_KEY", "openssh_private_key"),
        ("DFT_REMOTE_WINSCP_PATH", "winscp_path"),
        ("DFT_REMOTE_WINSCP_KEY", "winscp_private_key"),
    ]:
        val = os.getenv(env_key)
        if val:
            setattr(cfg, attr, Path(val))

    alias = os.getenv("DFT_REMOTE_OPENSSH_ALIAS")
    if alias:
        cfg.openssh_host_alias = alias
    winscp_session = os.getenv("DFT_REMOTE_WINSCP_SESSION")
    if winscp_session:
        cfg.winscp_session_name = winscp_session.strip() or None

    # 本地环境覆盖
    for env_key, attr in [
        ("DFT_LOCAL_CONDA_ACTIVATE", "conda_activate"),
        ("DFT_LOCAL_MACE_ENV", "mace_env"),
        ("DFT_LOCAL_ANALYSIS_ENV", "analysis_env"),
    ]:
        val = os.getenv(env_key)
        if val:
            setattr(cfg, attr, val)

    remote_timeout = os.getenv("DFT_REMOTE_TIMEOUT_SECONDS")
    if remote_timeout:
        cfg.remote_timeout_seconds = int(remote_timeout)

    return cfg


def _detect_existing_path(candidates: list[str | None]) -> Path:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    for candidate in candidates:
        if candidate:
            return Path(candidate)
    return Path()


def _detect_optional_path(candidates: list[str | None]) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _infer_host_alias_from_config(config_path: Path, *, host: str, user: str) -> str | None:
    try:
        lines = config_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    current_hosts: list[str] = []
    current_hostname: str | None = None
    current_user: str | None = None

    def _match_alias() -> str | None:
        if not current_hosts:
            return None
        if current_hostname and current_hostname != host:
            return None
        if current_user and current_user != user:
            return None
        for candidate in current_hosts:
            if candidate not in {"*", "?"} and "*" not in candidate and "?" not in candidate:
                return candidate
        return None

    for raw_line in lines + ["Host __END__"]:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].lower()
        value = " ".join(parts[1:])
        if key == "host":
            matched = _match_alias()
            if matched:
                return matched
            current_hosts = parts[1:]
            current_hostname = None
            current_user = None
        elif key == "hostname":
            current_hostname = value
        elif key == "user":
            current_user = value
    return None
