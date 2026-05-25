from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from dft_app.models import TaskType
from dft_app.planner.auto_planner import AutoPlanner
from dft_app.planner.rule_based_planner import RuleBasedPlanner

from .project_state import append_progress, project_paths
from .paths import ensure_runtime_dir

PlannerMode = Literal["rule", "auto"]
ExecutionMode = Literal["dry_run", "build", "submit", "remote_submit"]


@dataclass(frozen=True)
class DftTaskEnvelope:
    task_id: str
    prompt: str
    project: str | None
    planner_mode: PlannerMode
    plan: dict[str, Any]
    spec: dict[str, Any] | None
    readiness: str
    dft_command: list[str]
    execution_hint: str
    created_at: str
    task_record_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _task_dir(project: str | None) -> Path:
    if project:
        paths = project_paths(project)
        path = paths.root / "tasks"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return ensure_runtime_dir("tasks")


def _coerce_task_type(value: str | None) -> TaskType | None:
    if not value:
        return None
    return TaskType(value)


def _planner(mode: PlannerMode):
    if mode == "rule":
        return RuleBasedPlanner()
    if mode == "auto":
        return AutoPlanner()
    raise ValueError(f"未知 planner mode: {mode}")


def _planning_result_to_payload(result: Any, *, planner_mode: PlannerMode) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if planner_mode == "rule":
        plan, spec = result
        return plan.to_dict(), spec.to_dict() if spec is not None else None
    return result.plan.to_dict(), result.spec.to_dict() if result.spec is not None else None


def build_dft_run_args(
    *,
    prompt: str,
    material: str | None = None,
    structure_path: str | None = None,
    task_type: str | None = None,
    submit_profile: str | None = None,
    execution_mode: ExecutionMode = "dry_run",
) -> list[str]:
    args = ["run", prompt]
    if task_type:
        args.extend(["--task-type", task_type])
    if material:
        args.extend(["--material", material])
    if structure_path:
        args.extend(["--structure-path", structure_path])
    if submit_profile:
        args.extend(["--submit-profile", submit_profile])
    if execution_mode == "dry_run":
        args.append("--dry-run")
    elif execution_mode == "submit":
        args.append("--submit")
    elif execution_mode == "remote_submit":
        args.extend(["--submit", "--remote"])
    elif execution_mode == "build":
        pass
    else:
        raise ValueError(f"未知执行模式: {execution_mode}")
    return args


def create_task_plan(
    prompt: str,
    *,
    project: str | None = None,
    material: str | None = None,
    structure_path: str | None = None,
    task_type: str | None = None,
    submit_profile: str | None = None,
    planner_mode: PlannerMode = "rule",
    persist: bool = True,
) -> DftTaskEnvelope:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("任务 prompt 不能为空")
    task_id = f"task_{uuid4().hex[:8]}"
    forced_task_type = _coerce_task_type(task_type)
    planner = _planner(planner_mode)
    if planner_mode == "rule":
        result = planner.build_planning_artifacts(
            prompt=normalized_prompt,
            task_id=task_id,
            material_name=material,
            structure_path=structure_path,
            forced_task_type=forced_task_type,
            submit_profile=submit_profile,
        )
    else:
        result = planner.plan(
            prompt=normalized_prompt,
            task_id=task_id,
            material_name=material,
            structure_path=structure_path,
            forced_task_type=forced_task_type,
            submit_profile=submit_profile,
        )
    plan_payload, spec_payload = _planning_result_to_payload(result, planner_mode=planner_mode)
    ready = "ready" if spec_payload is not None else "needs_confirmation"
    dft_command = build_dft_run_args(
        prompt=normalized_prompt,
        material=material,
        structure_path=structure_path,
        task_type=task_type or (spec_payload or {}).get("task_type"),
        submit_profile=submit_profile or (spec_payload or {}).get("submit_profile"),
        execution_mode="dry_run",
    )
    envelope = DftTaskEnvelope(
        task_id=task_id,
        prompt=normalized_prompt,
        project=project,
        planner_mode=planner_mode,
        plan=plan_payload,
        spec=spec_payload,
        readiness=ready,
        dft_command=["aether-dft", "dft", *dft_command],
        execution_hint=(
            "可执行 dry-run/build/submit。默认只 dry-run；提交集群必须显式 --submit。"
            if spec_payload is not None
            else "复杂任务需要先确认缺失信息或拆分子任务。"
        ),
        created_at=_now(),
    )
    if persist:
        task_path = save_task_envelope(envelope)
        envelope = DftTaskEnvelope(**{**envelope.to_dict(), "task_record_path": str(task_path)})
        if project:
            append_progress(
                project,
                completed=[f"已生成结构化 DFT 任务 `{task_id}`。"],
                blockers=[] if spec_payload is not None else ["任务仍需补充信息或人工确认后才能执行。"],
                next_steps=["运行 `aether-dft task run ... --project <slug>` 进入 dry-run/build/submit。"],
            )
    return envelope


def save_task_envelope(envelope: DftTaskEnvelope) -> Path:
    path = _task_dir(envelope.project) / f"{envelope.task_id}.json"
    payload = envelope.to_dict()
    payload["task_record_path"] = str(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_task_records(project: str | None = None) -> list[dict[str, Any]]:
    directory = _task_dir(project)
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("task_*.json"), reverse=True):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


def run_dft_task(
    prompt: str,
    *,
    project: str | None = None,
    material: str | None = None,
    structure_path: str | None = None,
    task_type: str | None = None,
    submit_profile: str | None = None,
    planner_mode: PlannerMode = "rule",
    execution_mode: ExecutionMode = "dry_run",
) -> dict[str, Any]:
    envelope = create_task_plan(
        prompt,
        project=project,
        material=material,
        structure_path=structure_path,
        task_type=task_type,
        submit_profile=submit_profile,
        planner_mode=planner_mode,
        persist=True,
    )
    if envelope.spec is None:
        return {
            "status": "needs_confirmation",
            "task": envelope.to_dict(),
            "message": "复杂任务已建档，但还不能直接执行。",
        }

    from dft_app.cli.main import main as dft_main

    dft_args = build_dft_run_args(
        prompt=prompt,
        material=material,
        structure_path=structure_path,
        task_type=task_type or str(envelope.spec.get("task_type") or ""),
        submit_profile=submit_profile or envelope.spec.get("submit_profile"),
        execution_mode=execution_mode,
    )
    exit_code = dft_main(dft_args)
    status = "ok" if exit_code == 0 else "failed"
    if project:
        append_progress(
            project,
            completed=[f"DFT 任务 `{envelope.task_id}` 已执行 `{execution_mode}`，退出码 {exit_code}。"],
            blockers=[] if exit_code == 0 else ["DFT 主线执行失败，需查看命令输出。"],
            next_steps=["检查任务记录、run 目录或继续提交/解析结果。"],
        )
    return {
        "status": status,
        "exit_code": exit_code,
        "execution_mode": execution_mode,
        "dft_args": dft_args,
        "task": envelope.to_dict(),
    }
