from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .chat import ask_once
from .paths import ensure_runtime_dir


AskFn = Callable[..., dict[str, Any]]
ProgressFn = Callable[[dict[str, Any]], None]
PermissionFn = Callable[[dict[str, Any]], bool]
StreamFn = Callable[[dict[str, Any]], None]


def _tool_names(record: dict[str, Any]) -> list[str]:
    return [str(item.get("name") or "") for item in record.get("tool_executions") or []]


def _contains_any(names: list[str], candidates: set[str]) -> bool:
    return bool(set(names) & candidates)


def _default_readonly_permission(details: dict[str, Any]) -> bool:
    """Deny every side-effect request during validation.

    The validation command is deliberately real for model/API/SSH reads, but it
    must never submit, cancel, sync, or write research state unless a future
    explicit validation mode says so.
    """

    return False


def _validation_prompt(*, project: str, cluster_alias: str | None, include_outcar: bool) -> str:
    alias_clause = f"如果需要连接集群，优先使用 cluster_alias={cluster_alias}。" if cluster_alias else (
        "如果需要连接集群，先读取项目内 cluster profiles，再自己选择最合适的 alias。"
    )
    outcar_clause = (
        "还要尝试只读查找/解析一个已有 OUTCAR 或当前作业 OUTCAR 片段；如果没有可用路径，说明缺口，不要编造。"
        if include_outcar
        else "不要求解析 OUTCAR；只验证项目上下文、集群配置、连通性和当前队列。"
    )
    return (
        "真实课题验收（只读模式）：你是 AETHER-DFT 计算化学科研合伙人，"
        f"当前项目是 {project}。"
        "不要写文件，不要同步 research，不要提交、取消或修改任何集群任务。"
        f"{alias_clause}"
        "请你自己判断需要调用哪些只读工具来取得证据；至少应确认项目上下文、可用集群配置、SSH/SLURM 连通性或当前队列之一。"
        f"{outcar_clause}"
        "最后用自然语言给出：调用了哪些工具、拿到什么证据、哪些环节还没真实验证。"
    )


def run_real_case_validation(
    *,
    project: str,
    model_id: str | None = None,
    cluster_alias: str | None = None,
    include_outcar: bool = False,
    max_steps: int = 6,
    max_tokens: int = 1400,
    ask_fn: AskFn = ask_once,
    progress_callback: ProgressFn | None = None,
    permission_prompt_callback: PermissionFn | None = None,
    stream_callback: StreamFn | None = None,
) -> dict[str, Any]:
    """Run a model-led, read-only real-case validation turn and persist evidence.

    This is not a user-facing fixed workflow. It is an acceptance-test harness:
    the model receives a natural-language validation goal and chooses tools; the
    runner only denies side effects and scores whether key evidence classes were
    actually touched.
    """

    project = str(project or "").strip()
    if not project:
        raise ValueError("project 不能为空。")

    prompt = _validation_prompt(project=project, cluster_alias=cluster_alias, include_outcar=include_outcar)
    started_at = datetime.now().astimezone()
    denied_permissions: list[dict[str, Any]] = []

    def permission_guard(details: dict[str, Any]) -> bool:
        denied_permissions.append(
            {
                "tool_name": details.get("tool_name"),
                "permission_label": details.get("permission_label"),
                "message": details.get("message"),
            }
        )
        if permission_prompt_callback is not None:
            # The validation harness remains safe: caller can observe details,
            # but side effects are still denied for this read-only command.
            try:
                permission_prompt_callback(details)
            except Exception:
                pass
        return _default_readonly_permission(details)

    try:
        record = ask_fn(
            prompt,
            project=project,
            model_id=model_id,
            max_steps=max_steps,
            max_tokens=max_tokens,
            permission_mode="ask",
            progress_callback=progress_callback,
            permission_prompt_callback=permission_guard,
            stream_callback=stream_callback,
        )
        error = ""
    except Exception as exc:
        record = {}
        error = str(exc)

    names = _tool_names(record)
    evidence = {
        "project_context": _contains_any(names, {"project_state_read", "project_continuity_digest", "research_onboarding_context"}),
        "cluster_profile": _contains_any(names, {"cluster_profile_list", "cluster_config"}),
        "cluster_live": _contains_any(names, {"cluster_probe", "cluster_my_jobs", "cluster_job_status_brief"}),
        "outcar_read": _contains_any(names, {"cluster_job_partial_outcar", "cluster_job_progress_estimate", "vasp_output_scan", "result_interpret"}),
        "side_effect_blocked": bool(denied_permissions),
    }
    required = ["project_context", "cluster_profile", "cluster_live"]
    if include_outcar:
        required.append("outcar_read")
    missing = [key for key in required if not evidence.get(key)]
    status = "error" if error else ("ok" if not missing else "incomplete")

    result: dict[str, Any] = {
        "status": status,
        "project": project,
        "model_id": model_id or record.get("model_id") or "",
        "cluster_alias": cluster_alias or "",
        "include_outcar": include_outcar,
        "started_at": started_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round((datetime.now().astimezone() - started_at).total_seconds(), 3),
        "prompt": prompt,
        "tool_names": names,
        "evidence": evidence,
        "missing_evidence": missing,
        "denied_permissions": denied_permissions,
        "record_path": record.get("record_path", ""),
        "finish_reason": record.get("finish_reason", ""),
        "response": record.get("response", ""),
        "error": error,
    }
    report_dir = ensure_runtime_dir("real_case_validations")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_project = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in project)
    report_path = report_dir / f"{stamp}-{safe_project}.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    result["report_path"] = str(report_path)
    return result

