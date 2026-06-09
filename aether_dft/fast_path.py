from __future__ import annotations

"""Natural-language CLI fast paths.

These paths intentionally bypass the LLM for high-frequency, read-only status
questions.  They are not a fixed research workflow; they are a latency shortcut
for intents where the correct tool is obvious and safe.
"""

from dataclasses import dataclass
import json
import re
from typing import Any, Protocol


class RegistryLike(Protocol):
    def run_tool(self, name: str, arguments: dict[str, Any] | str | None = None) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class FastPathResponse:
    handled: bool
    text: str = ""
    exit_code: int = 0
    route: str = ""


JOB_ID_RE = re.compile(r"(?<![\w.-])(\d{4,})(?![\w.-])")


def dispatch_fast_path(query: str, *, registry: RegistryLike | None = None) -> FastPathResponse:
    """Return a fast response for obvious read-only CLI intents.

    ``handled=False`` means the caller should fall through to the normal LLM
    harness.  Fast paths must stay conservative: if an intent is ambiguous or
    would write/submit/cancel, miss and let the model/user-confirmation path
    handle it.
    """

    raw = " ".join(str(query or "").split())
    if not raw:
        return FastPathResponse(False)
    text = raw.lower()
    registry = registry or _default_registry()

    if _matches_projects(text):
        return FastPathResponse(True, _format_projects(), route="project_list")

    model = _extract_model_switch(text)
    if model:
        return FastPathResponse(True, _switch_model(model), route="model_switch")

    project = _extract_project_lookup(raw)
    if project:
        return FastPathResponse(True, _format_project_lookup(project), route="project_lookup")

    job_id = _extract_job_id(text)
    if job_id and _matches_convergence(text):
        return FastPathResponse(True, _format_job_convergence(registry, job_id), route="job_convergence")
    if job_id and _matches_job_status(text):
        return FastPathResponse(True, _format_job_brief(registry, job_id), route="job_status")

    if _matches_status_overview(text):
        return FastPathResponse(True, _format_my_jobs(registry), route="my_jobs")

    if _matches_last_results(text):
        return FastPathResponse(True, _format_recent_runs(registry), route="recent_runs")

    return FastPathResponse(False)


def _default_registry() -> RegistryLike:
    from aether_dft.runtime_harness.tool_registry import ToolRegistry

    return ToolRegistry(permission_mode="dev")


def _run_tool(registry: RegistryLike, name: str, args: dict[str, Any]) -> dict[str, Any]:
    wrapper = registry.run_tool(name, args)
    result = wrapper.get("result", wrapper) if isinstance(wrapper, dict) else {}
    return result if isinstance(result, dict) else {"status": "error", "message": str(result)}


def _matches_projects(text: str) -> bool:
    return bool(re.search(r"(有哪些|列出|list|show).*(项目|projects?)|项目列表|我的项目", text))


def _matches_status_overview(text: str) -> bool:
    return bool(
        re.search(
            r"^(status|jobs?|queue)$|"
            r"^(队列|作业列表|作业状态)$",
            text,
        )
    )


def _matches_job_status(text: str) -> bool:
    return bool(re.search(r"(job|作业|任务|状态|怎么样|进度|看看|看下|queue|squeue)", text))


def _matches_convergence(text: str) -> bool:
    return bool(re.search(r"(收敛|算到哪|到哪|outcar|oszicar|能量|converg|progress)", text))


def _matches_last_results(text: str) -> bool:
    return bool(re.search(r"(上次|最近|latest|last).*(结果|run|计算|任务)|最近run|最近计算", text))


def _extract_job_id(text: str) -> str | None:
    match = JOB_ID_RE.search(text)
    return match.group(1) if match else None


def _extract_model_switch(text: str) -> str | None:
    if not re.search(r"(切到|切换|使用|换成|set|switch).*(deepseek|qwen|bailian)", text):
        return None
    if "deepseek" in text:
        return "deepseek:deepseek-v4-pro"
    if "qwen" in text or "bailian" in text or "百炼" in text:
        return "bailian:qwen3.7-max"
    return None


def _extract_project_lookup(raw: str) -> str | None:
    match = re.search(r"(?:切到|进入|打开|查看|使用)\s*([\w\u4e00-\u9fff.-]+)\s*(?:项目)?", raw, flags=re.I)
    if not match or "模型" in raw or "deepseek" in raw.lower() or "qwen" in raw.lower():
        return None
    value = match.group(1).strip()
    if value in {"项目", "project"}:
        return None
    return value


def _format_projects() -> str:
    from aether_dft.project_state import list_projects

    projects = list_projects()
    if not projects:
        return "暂无项目。可以先运行：aether-dft project init <name>"
    lines = ["项目列表："]
    for item in projects:
        slug = str(item.get("slug") or item.get("name") or "")
        desc = str(item.get("description") or "").strip()
        status = str(item.get("status") or "").strip()
        suffix = f" — {desc}" if desc else ""
        state = f" [{status}]" if status else ""
        lines.append(f"- {slug}{state}{suffix}")
    return "\n".join(lines)


def _format_project_lookup(project: str) -> str:
    from aether_dft.project_state import load_project, project_paths

    try:
        data = load_project(project)
    except FileNotFoundError:
        return f"找不到项目：{project}\n可以先看：aether-dft project list"
    paths = project_paths(str(data.get("slug") or project))
    return "\n".join(
        [
            f"项目：{data.get('slug') or project}",
            f"状态：{data.get('status') or 'unknown'}",
            f"描述：{data.get('description') or '（无）'}",
            f"状态文件：{paths.state_md}",
            f"继续对话：aether-dft chat --project {data.get('slug') or project} \"继续\"",
        ]
    )


def _switch_model(model_id: str) -> str:
    from aether_dft.model_catalog import set_default_model

    preferences = set_default_model(model_id)
    return f"默认模型已切换到：{model_id}\npreferences: {json.dumps(preferences, ensure_ascii=False)}"


def _format_my_jobs(registry: RegistryLike) -> str:
    result = _run_tool(registry, "cluster_my_jobs", {"limit": 20})
    if result.get("status") != "ok":
        return _format_error("cluster_my_jobs", result)
    jobs = result.get("jobs") or []
    if not jobs:
        return "当前 squeue --me 没有 running/pending 作业。"
    lines = [f"当前队列：{len(jobs)} 个作业（fast-path，未调用 LLM）", ""]
    lines.append(f"{'JOBID':<10} {'STATE':<10} {'ELAPSED':<10} {'NODE':<12} NAME / REASON")
    for job in jobs:
        lines.append(
            f"{str(job.get('job_id') or ''):<10} "
            f"{str(job.get('scheduler_state') or ''):<10} "
            f"{str(job.get('elapsed') or ''):<10} "
            f"{str(job.get('node') or ''):<12} "
            f"{job.get('name') or ''} {job.get('reason') or ''}".rstrip()
        )
    lines.append("")
    lines.append("要看单个作业：aether-dft job <JOBID> 怎么样")
    return "\n".join(lines)


def _format_job_brief(registry: RegistryLike, job_id: str) -> str:
    status = _run_tool(registry, "cluster_job_status_brief", {"job_id": job_id})
    lines = [f"Job {job_id} 状态（fast-path）：", _format_dict_compact(status)]
    tail = _run_tool(registry, "cluster_job_tail_log", {"job_id": job_id, "lines": 20})
    if tail.get("status") == "ok":
        lines.extend(["", f"日志尾部：{tail.get('log_path_relative') or ''}", str(tail.get("tail") or "").strip()])
    elif tail.get("status") not in {"unavailable", "missing"}:
        lines.extend(["", "日志读取：", _format_dict_compact(tail)])
    return "\n".join(item for item in lines if item is not None)


def _format_job_convergence(registry: RegistryLike, job_id: str) -> str:
    status = _run_tool(registry, "cluster_job_status_brief", {"job_id": job_id})
    outcar = _run_tool(registry, "cluster_job_partial_outcar", {"job_id": job_id})
    progress = _run_tool(registry, "cluster_job_progress_estimate", {"job_id": job_id})
    lines = [f"Job {job_id} 收敛快照（fast-path）：", ""]
    lines.append("队列：" + _format_dict_compact(status))
    lines.append("OUTCAR：" + _format_dict_compact(outcar))
    lines.append("趋势：" + _format_dict_compact(progress))
    if outcar.get("status") in {"unavailable", "missing"}:
        lines.append("提示：没有本地 remote_run_root 映射或远端输出尚未生成时，只能给队列状态。")
    return "\n".join(lines)


def _format_recent_runs(registry: RegistryLike) -> str:
    result = _run_tool(registry, "dft_run_list", {"limit": 3})
    if result.get("status") not in {"ok", "empty"}:
        return _format_error("dft_run_list", result)
    runs = result.get("runs") or result.get("items") or []
    if not runs:
        return "没有找到最近 DFT run 记录。"
    lines = ["最近 run："]
    for item in runs[:3]:
        lines.append(
            f"- {item.get('run_id') or ''} task={item.get('task_id') or ''} "
            f"status={item.get('overall_status') or item.get('status') or ''} root={item.get('run_root') or ''}"
        )
    return "\n".join(lines)


def _format_error(tool_name: str, result: dict[str, Any]) -> str:
    return f"{tool_name} 返回 {result.get('status') or 'error'}：{result.get('message') or json.dumps(result, ensure_ascii=False)}"


def _format_dict_compact(data: dict[str, Any]) -> str:
    keys = [
        "status",
        "scheduler_state",
        "active",
        "elapsed",
        "node",
        "reason",
        "last_toten_ev",
        "last_free_energy_ev",
        "max_force_ev_a",
        "accuracy_reached",
        "ionic_steps_seen",
        "last_energy_ev",
        "last_delta_ev",
        "convergence_score",
        "message",
        "source",
    ]
    parts = []
    for key in keys:
        if key in data and data.get(key) not in {None, ""}:
            parts.append(f"{key}={data.get(key)}")
    return ", ".join(parts) if parts else json.dumps(data, ensure_ascii=False, default=str)
