from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from dft_app.remote import SSHRemoteRunner


@dataclass(frozen=True)
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


class AetherToolRunner:
    """Safe tool surface exposed to qwen/openai-compatible chat models."""

    def __init__(self, *, allow_cluster_submit: bool = False):
        self.allow_cluster_submit = allow_cluster_submit

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            _schema(
                "cluster_probe",
                "通过项目配置的 SSH alias 进行非破坏性集群连通性探测，检查 hostname/pwd/sbatch/squeue/vasp_std。",
                {},
            ),
            _schema(
                "cluster_config",
                "读取当前集群 SSH 配置摘要；不会返回 API key 或私钥内容。",
                {},
            ),
            _schema(
                "adsorption_workflow_status",
                "读取已准备 adsorption workflow bundle 的三子任务状态和下一步建议。",
                {"run_root": {"type": "string", "description": "包含 adsorption_workflow_bundle.json 的 run_root。"}},
                ["run_root"],
            ),
            _schema(
                "adsorption_workflow_remote_submit",
                "通过 SSH/SLURM 远程提交 clean_slab、isolated_adsorbate、adsorbed_system 三个子任务；只有 CLI 启用 --allow-cluster-submit 时才会真正提交。",
                {"run_root": {"type": "string", "description": "包含 adsorption_workflow_bundle.json 的 run_root。"}},
                ["run_root"],
            ),
            _schema(
                "adsorption_workflow_monitor",
                "通过远程或本地 runner 轮询 adsorption workflow 子任务状态。",
                {"run_root": {"type": "string", "description": "包含 adsorption_workflow_bundle.json 的 run_root。"}},
                ["run_root"],
            ),
            _schema(
                "adsorption_workflow_fetch",
                "同步已提交远程 adsorption workflow 子任务的 VASP 输出文件。",
                {"run_root": {"type": "string", "description": "包含 adsorption_workflow_bundle.json 的 run_root。"}},
                ["run_root"],
            ),
            _schema(
                "adsorption_workflow_parse_analyze",
                "解析已完成并同步的三子任务输出，汇总吸附能。",
                {"run_root": {"type": "string", "description": "包含 adsorption_workflow_bundle.json 的 run_root。"}},
                ["run_root"],
            ),
            _schema(
                "recommend_next_tasks",
                "基于项目状态、任务记录和知识库推荐下一步科研任务。",
                {
                    "project": {"type": "string", "description": "项目 slug，可为空字符串。"},
                    "focus": {"type": "string", "description": "关注点，例如 adsorption/cluster/analysis，可为空字符串。"},
                },
            ),
        ]

    def run(self, name: str, arguments: dict[str, Any] | None = None) -> ToolExecution:
        args = arguments or {}
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "cluster_probe": self._cluster_probe,
            "cluster_config": self._cluster_config,
            "adsorption_workflow_status": lambda payload: self._adsorption_workflow(payload, status=True),
            "adsorption_workflow_remote_submit": self._adsorption_workflow_remote_submit,
            "adsorption_workflow_monitor": lambda payload: self._adsorption_workflow(payload, monitor=True),
            "adsorption_workflow_fetch": lambda payload: self._adsorption_workflow(payload, fetch=True),
            "adsorption_workflow_parse_analyze": lambda payload: self._adsorption_workflow(payload, parse_analyze=True),
            "recommend_next_tasks": self._recommend_next_tasks,
        }
        if name not in handlers:
            result = {"status": "error", "message": f"未知工具: {name}"}
        else:
            try:
                result = handlers[name](args)
            except Exception as exc:
                result = {"status": "error", "message": str(exc)}
        return ToolExecution(name=name, arguments=args, result=result)

    @staticmethod
    def _cluster_probe(_: dict[str, Any]) -> dict[str, Any]:
        result = SSHRemoteRunner().probe()
        return {"status": result.status, "message": result.message, "details": result.details}

    @staticmethod
    def _cluster_config(_: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "config": SSHRemoteRunner().describe_config()}

    def _adsorption_workflow_remote_submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.allow_cluster_submit:
            return {
                "status": "blocked",
                "message": "模型请求了远程提交，但当前 agent 未启用 --allow-cluster-submit；不会假装提交。",
            }
        return self._adsorption_workflow(payload, submit=True, remote=True)

    @staticmethod
    def _adsorption_workflow(
        payload: dict[str, Any],
        *,
        submit: bool = False,
        status: bool = False,
        monitor: bool = False,
        fetch: bool = False,
        parse_analyze: bool = False,
        remote: bool = False,
    ) -> dict[str, Any]:
        from dft_app.cli.main import execute_adsorption_workflow

        run_root = str(payload.get("run_root") or "").strip()
        if not run_root:
            return {"status": "error", "message": "缺少 run_root。"}
        result = execute_adsorption_workflow(
            run_root=Path(run_root),
            submit=submit,
            status=status,
            monitor=monitor,
            fetch=fetch,
            parse_analyze=parse_analyze,
            remote=remote,
        )
        return {"status": "ok", "result": result}

    @staticmethod
    def _recommend_next_tasks(payload: dict[str, Any]) -> dict[str, Any]:
        from .recommendations import recommend_next_tasks

        project = str(payload.get("project") or "").strip() or None
        focus = str(payload.get("focus") or "").strip() or None
        return {"status": "ok", "recommendations": recommend_next_tasks(project, focus=focus)}


def parse_tool_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"工具参数不是合法 JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("工具参数 JSON 顶层必须是 object。")
    return payload
