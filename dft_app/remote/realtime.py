"""集群实时轻量查询：用户问"看看怎么样了"时秒回的工具集。

设计原则：
- 每个函数单独可调用、立即返回；不依赖前置工具。
- 重量级 `SSHRemoteRunner.monitor`/`fetch_outputs` 留作完整闭环；本模块只做"瞥一眼"。
- 集群不可达 / 命令缺失 / job 不存在等情况下返回 ``status`` ≠ ``ok`` 的诚实结果，
  不抛异常、不假装数据。
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

from dft_app.remote.config import RemoteClusterConfig, config_for_local_cluster_alias
from dft_app.remote.ssh_remote_runner import SSHRemoteRunner


_ACTIVE_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED"}
_SAFE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# log_name 只接受 basename，禁止 '/'；上层会自动尝试 logs/<name>、outputs/<name> 等前缀
_SAFE_LOG_BASENAME_RE = re.compile(r"^[A-Za-z0-9_.@+:-]+$")
# remote_run_root 必须是合理的远端绝对路径（或 ~）；允许科研目录里的 Unicode，
# 但禁止 shell 元字符、引号、反斜杠、空白与路径穿越。
_UNSAFE_REMOTE_PATH_CHARS_RE = re.compile(r"[\s$;&|><`'\"\\\r\n]")


def _runner(config: RemoteClusterConfig | None = None) -> SSHRemoteRunner:
    return SSHRemoteRunner(config=config)


def _exec(runner: SSHRemoteRunner, command: str, timeout: int = 30) -> dict[str, Any]:
    config = runner._load_config()
    backend = runner._select_backend(config)
    tools_error = runner._ensure_local_tools(config, backend)
    if tools_error is not None:
        return {"ok": False, "stdout": "", "stderr": tools_error, "returncode": -1, "backend": backend}
    result = runner._run_remote_command(config, command, timeout=timeout, backend=backend)
    return {
        "ok": result.returncode == 0,
        "stdout": (result.stdout or "").strip(),
        "stderr": (result.stderr or "").strip(),
        "returncode": result.returncode,
        "backend": backend,
    }


def _safe_job_id(job_id: str | None) -> str:
    value = str(job_id or "").strip()
    if not value:
        return ""
    if not _SAFE_JOB_ID_RE.fullmatch(value):
        raise ValueError("job_id 只能包含字母、数字、下划线、点和短横线。")
    return value


def _safe_log_name(log_name: str | None) -> str:
    """log_name 只接受基名（无 '/'）；上层负责加 logs/ 或 outputs/ 前缀。"""
    value = str(log_name or "vasp.out").strip() or "vasp.out"
    if not _SAFE_LOG_BASENAME_RE.fullmatch(value):
        raise ValueError("log_name 必须是单一文件基名（字母/数字/_./@+:- ），不能含路径分隔符。")
    if value in {".", ".."}:
        raise ValueError("log_name 不能是 '.' 或 '..'。")
    return value


def _safe_remote_path(path: str | None, config: RemoteClusterConfig) -> str:
    """校验 remote_run_root：阻止 shell 元字符、路径穿越和越权读取。

    Shell quoting 只能防命令注入，不能防模型拿当前 SSH 凭证去试探
    ``/etc``、``/home/otheruser`` 或当前用户 home 下的敏感文件。因此这里把
    路径限制在当前配置的 ``remote_base_dir``、``/scratch/<user>/`` 或
    ``/home/<user>/research`` 下。最后一项用于 AETHER 的 research 工作区
    与集群 ``~/research`` 对齐，只开放该目录，不开放整个 home。
    """
    value = str(path or "").strip()
    if not value:
        raise ValueError("remote_run_root 不能为空。")
    if _UNSAFE_REMOTE_PATH_CHARS_RE.search(value):
        raise ValueError(
            "remote_run_root 含非法字符；禁止 shell 元字符（$ ; & | > < ` ' \" \\ 等）、空白和换行。"
        )
    parts = [p for p in value.split("/") if p]
    if any(p == ".." for p in parts):
        raise ValueError("remote_run_root 不能包含 '..' 路径段。")
    if value == "~":
        value = f"/home/{config.user}"
    elif value.startswith("~/"):
        value = f"/home/{config.user}/{value[2:]}"
    if not value.startswith("/"):
        raise ValueError("remote_run_root 必须是绝对路径或 ~/ 开头路径。")

    def clean(raw: str) -> str:
        return str(PurePosixPath(str(raw).rstrip("/") or "/"))

    normalized = clean(value)
    allowed_roots = {
        clean(config.remote_base_dir),
        clean(f"/scratch/{config.user}"),
        clean(f"/home/{config.user}/research"),
    }
    if not any(normalized == root or normalized.startswith(root + "/") for root in allowed_roots if root != "/"):
        raise ValueError(
            "remote_run_root 超出允许范围；只允许配置的 remote_base_dir 或 /scratch/<user>/ 下的路径。"
        )
    return normalized


def _safe_positive_int(value: Any, *, default: int, lo: int, hi: int, name: str) -> int:
    """把模型/调用方给的 limit/lines 这类整数安全地归一化；非法值直接抛 ValueError。"""
    if value is None or value == "":
        return default
    try:
        cast = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正整数，收到 {value!r}。") from exc
    return max(lo, min(cast, hi))


def _quote_shell(value: str) -> str:
    escaped = str(value).replace("'", "'\"'\"'")
    return f"'{escaped}'"


def _resolve_remote_run_root(job_id: str | None, project_root: Path | None) -> str | None:
    """从本地 RecordStore 按 scheduler_job_id 反查 remote_run_root。"""
    if not job_id:
        return None
    try:
        from dft_app.storage import RecordStore
    except Exception:
        return None
    root = project_root or Path.cwd()
    try:
        records = RecordStore(root).list_runs(limit=200)
    except Exception:
        return None
    target = str(job_id).strip()
    for rec in records:
        if str(rec.get("scheduler_job_id") or "").strip() == target:
            remote_root = (rec.get("notes") or {}).get("remote", {}).get("remote_run_root")
            if remote_root:
                return str(remote_root)
            run_root = rec.get("run_root")
            if run_root:
                # 退路：读完整 run_record.json
                try:
                    record = RecordStore(root).load_run_record(Path(run_root))
                    return (record.notes.get("remote") or {}).get("remote_run_root")
                except Exception:
                    continue
    return None


def _config_for_alias(cluster_alias: str | None) -> RemoteClusterConfig | None:
    alias = str(cluster_alias or "").strip()
    return config_for_local_cluster_alias(alias) if alias else None


def job_status_brief(job_id: str, *, cluster_alias: str | None = None) -> dict[str, Any]:
    """单 job 状态 / 已运行时长 / 节点；< 2s。"""
    try:
        job_id = _safe_job_id(job_id)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    if not job_id:
        return {"status": "error", "message": "job_id 不能为空。"}
    try:
        runner = _runner(_config_for_alias(cluster_alias))
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "cluster_alias": cluster_alias}
    squeue_cmd = f"squeue -j {job_id} -h -o '%T|%M|%N|%R'"
    result = _exec(runner, squeue_cmd, timeout=20)
    if not result["ok"] and "Invalid job id" not in result["stderr"]:
        return {
            "status": "error",
            "job_id": job_id,
            "message": result["stderr"] or result["stdout"] or "squeue 调用失败",
            "backend": result["backend"],
        }
    stdout = result["stdout"]
    if stdout:
        parts = stdout.split("\n", 1)[0].split("|")
        state = parts[0].strip().split("+")[0].upper() if parts else "UNKNOWN"
        elapsed = parts[1].strip() if len(parts) > 1 else ""
        node = parts[2].strip() if len(parts) > 2 else ""
        reason = parts[3].strip() if len(parts) > 3 else ""
        return {
            "status": "ok",
            "job_id": job_id,
            "scheduler_state": state,
            "active": state in _ACTIVE_STATES,
            "elapsed": elapsed,
            "node": node,
            "reason": reason,
            "source": "squeue",
            "backend": result["backend"],
        }
    sacct_cmd = (
        f"sacct -j {job_id} --format=JobID,State,Elapsed,NodeList,ExitCode --noheader --parsable2 | head -n 1"
    )
    sacct = _exec(runner, sacct_cmd, timeout=20)
    if not sacct["ok"] or not sacct["stdout"]:
        return {
            "status": "unknown",
            "job_id": job_id,
            "message": "squeue 与 sacct 均未返回结果；该 job 可能已被清理。",
            "backend": result["backend"],
        }
    parts = sacct["stdout"].split("|")
    state = parts[1].split("+")[0].upper() if len(parts) > 1 else "UNKNOWN"
    elapsed = parts[2] if len(parts) > 2 else ""
    node = parts[3] if len(parts) > 3 else ""
    exit_code = parts[4] if len(parts) > 4 else ""
    return {
        "status": "ok",
        "job_id": job_id,
        "scheduler_state": state,
        "active": state in _ACTIVE_STATES,
        "elapsed": elapsed,
        "node": node,
        "exit_code": exit_code,
        "source": "sacct",
        "backend": result["backend"],
    }


def job_cancel(job_id: str, *, cluster_alias: str | None = None) -> dict[str, Any]:
    """精确取消单个 SLURM job，并回读同一 job_id 验证。

    这是一个有副作用的操作，但安全边界必须窄：只接受一个经过 allow-list
    校验的 job_id，不支持通配符、范围、``--me`` 或批量取消。
    """
    try:
        job_id = _safe_job_id(job_id)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    if not job_id:
        return {"status": "error", "message": "job_id 不能为空。"}
    try:
        runner = _runner(_config_for_alias(cluster_alias))
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "cluster_alias": cluster_alias}
    cancel = _exec(runner, f"scancel {job_id}", timeout=20)
    if not cancel["ok"]:
        return {
            "status": "error",
            "job_id": job_id,
            "message": cancel["stderr"] or cancel["stdout"] or "scancel 调用失败",
            "backend": cancel["backend"],
        }
    check = _exec(runner, f"squeue -j {job_id} -h -o '%i|%T|%R'", timeout=20)
    if check["ok"] and not check["stdout"]:
        return {
            "status": "canceled",
            "job_id": job_id,
            "verified_absent_from_squeue": True,
            "backend": cancel["backend"],
        }
    if check["ok"]:
        return {
            "status": "pending_verification",
            "job_id": job_id,
            "verified_absent_from_squeue": False,
            "squeue": check["stdout"],
            "backend": cancel["backend"],
        }
    return {
        "status": "submitted_cancel",
        "job_id": job_id,
        "verified_absent_from_squeue": None,
        "message": check["stderr"] or check["stdout"] or "已调用 scancel，但 squeue 验证失败。",
        "backend": cancel["backend"],
    }


def my_jobs(*, limit: int = 20, cluster_alias: str | None = None) -> dict[str, Any]:
    """squeue --me 简化；< 2s。"""
    try:
        safe_limit = _safe_positive_int(limit, default=20, lo=1, hi=200, name="limit")
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    try:
        runner = _runner(_config_for_alias(cluster_alias))
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "cluster_alias": cluster_alias}
    cmd = f"squeue --me -h -o '%i|%j|%T|%M|%N|%R' | head -n {safe_limit}"
    result = _exec(runner, cmd, timeout=20)
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["stderr"] or result["stdout"] or "squeue --me 调用失败",
            "backend": result["backend"],
        }
    jobs: list[dict[str, Any]] = []
    for line in result["stdout"].splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jobs.append(
            {
                "job_id": parts[0].strip(),
                "name": parts[1].strip() if len(parts) > 1 else "",
                "scheduler_state": parts[2].strip().split("+")[0].upper(),
                "elapsed": parts[3].strip() if len(parts) > 3 else "",
                "node": parts[4].strip() if len(parts) > 4 else "",
                "reason": parts[5].strip() if len(parts) > 5 else "",
            }
        )
    return {
        "status": "ok",
        "count": len(jobs),
        "jobs": jobs,
        "backend": result["backend"],
    }


def job_tail_log(
    job_id: str | None = None,
    *,
    remote_run_root: str | None = None,
    log_name: str = "vasp.out",
    lines: int = 50,
    project_root: str | None = None,
    cluster_alias: str | None = None,
) -> dict[str, Any]:
    """tail -n <lines> 某个 job 的指定日志（默认 vasp.out）；< 2s。"""
    try:
        safe_lines = _safe_positive_int(lines, default=50, lo=1, hi=500, name="lines")
        safe_job_id = _safe_job_id(job_id)
        safe_log = _safe_log_name(log_name)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    resolved_root_raw = remote_run_root or _resolve_remote_run_root(
        safe_job_id, Path(project_root) if project_root else None
    )
    if not resolved_root_raw:
        return {
            "status": "unavailable",
            "message": "找不到 remote_run_root；请提供 remote_run_root，或确认 job_id 对应的本地 run 记录里有 notes.remote.remote_run_root。",
            "job_id": safe_job_id or job_id,
        }
    try:
        runner = _runner(_config_for_alias(cluster_alias))
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "cluster_alias": cluster_alias}
    config = runner._load_config()
    try:
        resolved_root = _safe_remote_path(resolved_root_raw, config)
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "remote_run_root": resolved_root_raw}
    quoted_root = _quote_shell(resolved_root)
    candidates = [safe_log, f"logs/{safe_log}", f"outputs/{safe_log}", "slurm.out", "OSZICAR"]
    quoted_candidates = " ".join(_quote_shell(item) for item in dict.fromkeys(candidates))
    cmd = (
        f"for fname in {quoted_candidates}; do "
        f"  if [ -f {quoted_root}/\"$fname\" ]; then "
        f"    echo \"__AETHER_LOG_PATH__=$fname\"; "
        f"    tail -n {safe_lines} {quoted_root}/\"$fname\"; break; "
        f"  fi; "
        f"done"
    )
    result = _exec(runner, cmd, timeout=30)
    if not result["ok"]:
        return {
            "status": "error",
            "message": result["stderr"] or "tail 调用失败",
            "remote_run_root": resolved_root,
            "log_name": log_name,
        }
    stdout = result["stdout"]
    if not stdout:
        return {
            "status": "missing",
            "message": f"在 {resolved_root} 下没找到 {log_name} 或常见 fallback 日志文件。",
            "remote_run_root": resolved_root,
        }
    log_path = ""
    body_lines: list[str] = []
    for raw_line in stdout.splitlines():
        if raw_line.startswith("__AETHER_LOG_PATH__="):
            log_path = raw_line.split("=", 1)[1].strip()
            continue
        body_lines.append(raw_line)
    return {
        "status": "ok",
        "remote_run_root": resolved_root,
        "log_path_relative": log_path,
        "lines_requested": safe_lines,
        "lines_returned": len(body_lines),
        "tail": "\n".join(body_lines),
    }


def job_partial_outcar(
    job_id: str | None = None,
    *,
    remote_run_root: str | None = None,
    project_root: str | None = None,
    cluster_alias: str | None = None,
) -> dict[str, Any]:
    """解析当前 OUTCAR 的最后一步：能量 / 力 / ionic step / SCF iter；< 3s。"""
    try:
        safe_job_id = _safe_job_id(job_id)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    resolved_root_raw = remote_run_root or _resolve_remote_run_root(
        safe_job_id, Path(project_root) if project_root else None
    )
    if not resolved_root_raw:
        return {
            "status": "unavailable",
            "message": "找不到 remote_run_root；请提供。",
            "job_id": safe_job_id or job_id,
        }
    try:
        runner = _runner(_config_for_alias(cluster_alias))
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "cluster_alias": cluster_alias}
    config = runner._load_config()
    try:
        resolved_root = _safe_remote_path(resolved_root_raw, config)
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "remote_run_root": resolved_root_raw}
    quoted_outcar = _quote_shell(f"{resolved_root}/OUTCAR")
    # 取 OUTCAR 尾段 + 全文件关键行的最后一小段。已完成的大 OUTCAR 末尾常是
    # timing/memory accounting，单纯 tail 可能看不到最后一次能量/力。
    cmd = (
        f"if [ -f {quoted_outcar} ]; then "
        f"{{ tail -n 400 {quoted_outcar}; "
        f"grep -E \"TOTEN|FORCES: max atom|Iteration|reached required accuracy\" {quoted_outcar} | tail -n 160; }}; "
        f"else echo __AETHER_NO_OUTCAR__; fi"
    )
    result = _exec(runner, cmd, timeout=30)
    if not result["ok"]:
        return {"status": "error", "message": result["stderr"] or "OUTCAR tail 失败", "remote_run_root": resolved_root}
    if "__AETHER_NO_OUTCAR__" in result["stdout"]:
        return {
            "status": "missing",
            "message": "远端没有 OUTCAR，作业可能尚未开始 SCF。",
            "remote_run_root": resolved_root,
        }
    text = result["stdout"]
    toten_matches = re.findall(r"TOTEN\s*=\s*(-?\d+\.\d+)", text)
    free_matches = re.findall(r"free\s+energy\s+TOTEN\s*=\s*(-?\d+\.\d+)", text)
    force_matches = re.findall(r"FORCES: max atom, RMS\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", text)
    ionic_matches = re.findall(r"-+\s*Iteration\s*(\d+)\s*\(\s*(\d+)\s*\)", text)
    force_match = force_matches[-1] if force_matches else None
    ionic_match = ionic_matches[-1] if ionic_matches else None
    accuracy_reached = "reached required accuracy" in text.lower()
    return {
        "status": "ok",
        "remote_run_root": resolved_root,
        "last_toten_ev": float(toten_matches[-1]) if toten_matches else None,
        "last_free_energy_ev": float(free_matches[-1]) if free_matches else None,
        "max_force_ev_a": float(force_match[0]) if force_match else None,
        "rms_force_ev_a": float(force_match[1]) if force_match else None,
        "last_ionic_step": int(ionic_match[0]) if ionic_match else None,
        "last_scf_iter_within_step": int(ionic_match[1]) if ionic_match else None,
        "accuracy_reached": accuracy_reached,
        "guidance": (
            "accuracy_reached=True 表示已收敛；False + max_force_ev_a > EDIFFG 阈值通常还需要更多 ionic step。"
            "TOTEN 在多步之间震荡 → 可能 SCF 难收敛或 ISMEAR/SIGMA 不合适。"
        ),
    }


def job_progress_estimate(
    job_id: str | None = None,
    *,
    remote_run_root: str | None = None,
    project_root: str | None = None,
    cluster_alias: str | None = None,
) -> dict[str, Any]:
    """收敛轨迹分析：能量是否震荡？力是否在下降？给"剩余步数估计"。< 5s。"""
    try:
        safe_job_id = _safe_job_id(job_id)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    resolved_root_raw = remote_run_root or _resolve_remote_run_root(
        safe_job_id, Path(project_root) if project_root else None
    )
    if not resolved_root_raw:
        return {
            "status": "unavailable",
            "message": "找不到 remote_run_root；请提供。",
            "job_id": safe_job_id or job_id,
        }
    try:
        runner = _runner(_config_for_alias(cluster_alias))
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "cluster_alias": cluster_alias}
    config = runner._load_config()
    try:
        resolved_root = _safe_remote_path(resolved_root_raw, config)
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "remote_run_root": resolved_root_raw}
    quoted_oszicar = _quote_shell(f"{resolved_root}/OSZICAR")
    # OSZICAR 简短，常驻 ionic step trajectory
    cmd = (
        f"if [ -f {quoted_oszicar} ]; then tail -n 200 {quoted_oszicar}; "
        f"else echo __AETHER_NO_OSZICAR__; fi"
    )
    result = _exec(runner, cmd, timeout=30)
    if not result["ok"]:
        return {"status": "error", "message": result["stderr"] or "OSZICAR tail 失败", "remote_run_root": resolved_root}
    if "__AETHER_NO_OSZICAR__" in result["stdout"]:
        return {
            "status": "missing",
            "message": "远端没有 OSZICAR，作业可能尚未开始 SCF。",
            "remote_run_root": resolved_root,
        }
    text = result["stdout"]
    # 解析 ionic step 摘要行（带 F= / E0= 的行）
    energies: list[float] = []
    for line in text.splitlines():
        match = re.search(r"F=\s*(-?\d+\.\d+E?[+-]?\d*)", line)
        if match:
            try:
                energies.append(float(match.group(1)))
            except ValueError:
                continue
    if len(energies) < 2:
        return {
            "status": "partial",
            "remote_run_root": resolved_root,
            "ionic_steps_seen": len(energies),
            "message": "ionic step 数据不足，无法估计趋势。",
            "energies_tail": energies,
        }
    deltas = [energies[i] - energies[i - 1] for i in range(1, len(energies))]
    abs_deltas = [abs(d) for d in deltas]
    monotonic_down = all(d <= 1e-4 for d in deltas[-5:]) if len(deltas) >= 5 else False
    oscillating = any(deltas[i] * deltas[i - 1] < 0 for i in range(1, len(deltas))) if len(deltas) >= 2 else False
    convergence_score = max(0.0, 1.0 - min(1.0, (abs_deltas[-1] if abs_deltas else 1.0) / 1e-3))
    guidance_bits: list[str] = []
    if monotonic_down:
        guidance_bits.append("最近 5 个 ionic step 能量单调下降，正在正常收敛。")
    if oscillating:
        guidance_bits.append("能量序列有震荡，可能 ISMEAR/SIGMA 或 mixing 需要调整。")
    if not guidance_bits:
        guidance_bits.append("能量趋势不明显，建议再等几步再判断。")
    return {
        "status": "ok",
        "remote_run_root": resolved_root,
        "ionic_steps_seen": len(energies),
        "last_energy_ev": energies[-1],
        "last_delta_ev": deltas[-1] if deltas else None,
        "abs_delta_trend_tail": abs_deltas[-5:],
        "monotonic_decreasing_tail": monotonic_down,
        "oscillating": oscillating,
        "convergence_score": round(convergence_score, 3),
        "guidance": " ".join(guidance_bits),
    }
