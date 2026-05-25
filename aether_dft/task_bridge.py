from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from dft_app.models import (
    ConfirmationItem,
    ConvergenceSettings,
    EncutStrategy,
    ExecutionReadiness,
    ExperimentPlan,
    ExperimentSpec,
    JobSettings,
    KpointsStrategy,
    PlanComplexity,
    PlanSubtask,
    SmearingSettings,
    SpinSettings,
    StructureConstraint,
    StructureSource,
    TaskType,
)
from dft_app.planner.auto_planner import AutoPlanner
from dft_app.planner.rule_based_planner import RuleBasedPlanner

from .project_state import append_progress, project_paths
from .paths import ensure_runtime_dir
from .research_vasp_templates import resolve_research_vasp_template

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


def _extract_spec_object(result: Any, *, planner_mode: PlannerMode) -> ExperimentSpec | None:
    if planner_mode == "rule":
        return result[1]
    return result.spec


def _apply_research_vasp_template(
    spec: ExperimentSpec | None,
    *,
    project: str | None,
    prompt: str,
    material: str | None = None,
) -> dict[str, Any] | None:
    if spec is None:
        return None
    template = resolve_research_vasp_template(
        project,
        spec.task_type.value,
        prompt=prompt,
        material=material or spec.material_name,
    )
    overrides = dict(template.get("incar_overrides") or {})
    if overrides:
        spec.incar_overrides.update(overrides)
    if template.get("submit_profile") and not spec.submit_profile:
        spec.submit_profile = str(template["submit_profile"])
    notes = spec.notes if isinstance(spec.notes, dict) else {}
    notes["research_template"] = {
        "template_found": template.get("template_found", False),
        "template_id": template.get("template_id"),
        "template_scope": template.get("template_scope"),
        "source_paths": template.get("source_paths", []),
        "expected_incar": template.get("expected_incar", {}),
        "required_incar": template.get("required_incar", []),
        "severity_by_key": template.get("severity_by_key", {}),
        "free_atom_policy": template.get("free_atom_policy"),
        "blocked_method_rules": template.get("blocked_method_rules", []),
    }
    spec.notes = notes
    return template


def _experiment_spec_from_payload(data: dict[str, Any]) -> ExperimentSpec:
    kpoints_value = data["kpoints_strategy"].get("value")
    if isinstance(kpoints_value, list):
        kpoints_value = tuple(kpoints_value)
    return ExperimentSpec(
        task_id=data["task_id"],
        task_type=TaskType(data["task_type"]),
        material_name=data["material_name"],
        source_prompt=data["source_prompt"],
        created_at=data["created_at"],
        chemical_formula=data.get("chemical_formula"),
        description=data.get("description"),
        structure_source=StructureSource(data["structure_source"]),
        structure_path=data.get("structure_path"),
        structure_id=data.get("structure_id"),
        structure_constraints=StructureConstraint(**data["structure_constraints"]),
        workflow=data.get("workflow", []),
        code=data.get("code", "vasp"),
        functional=data.get("functional", "PBE"),
        task_goal=data.get("task_goal"),
        incar_overrides=data.get("incar_overrides", {}),
        kpoints_strategy=KpointsStrategy(mode=data["kpoints_strategy"]["mode"], value=kpoints_value),
        encut_strategy=EncutStrategy(**data["encut_strategy"]),
        smearing=SmearingSettings(**data["smearing"]),
        spin_settings=SpinSettings(**data["spin_settings"]),
        convergence_settings=ConvergenceSettings(**data["convergence_settings"]),
        workflow_parameters=data.get("workflow_parameters", {}),
        submit_profile=data.get("submit_profile"),
        scheduler=data.get("scheduler", "slurm"),
        job_overrides=JobSettings(**data["job_overrides"]),
        requires_confirmation=data.get("requires_confirmation", True),
        confirmation_items=[ConfirmationItem(item) for item in data.get("confirmation_items", [])],
        allow_reuse_previous_results=data.get("allow_reuse_previous_results", True),
        restart_from_task_id=data.get("restart_from_task_id"),
        tags=data.get("tags", []),
        notes=data.get("notes", {}),
    )


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
        if forced_task_type is not None:
            spec = planner.plan(
                prompt=normalized_prompt,
                task_id=task_id,
                material_name=material,
                structure_path=structure_path,
                forced_task_type=forced_task_type,
                submit_profile=submit_profile,
            )
            plan = ExperimentPlan(
                task_id=task_id,
                source_prompt=normalized_prompt,
                experiment_type="single_calculation",
                summary="用户显式指定 task_type；按单个 VASP 任务建模。",
                complexity=PlanComplexity.SIMPLE,
                readiness=ExecutionReadiness.READY,
                requires_confirmation=spec.requires_confirmation,
                missing_information=[],
                assumptions=["显式 task_type 优先于复杂意图拆分；research 模板仍会单独解析并核对。"],
                subtasks=[
                    PlanSubtask(
                        name="single_task",
                        goal=spec.task_goal or "执行单个 VASP 任务",
                        system_role="primary_system",
                        task_type=spec.task_type.value,
                    )
                ],
                recommended_submit_profile=spec.submit_profile,
                raw_plan={"forced_task_type": forced_task_type.value},
            )
            result = (plan, spec)
        else:
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
    spec_obj = _extract_spec_object(result, planner_mode=planner_mode)
    research_template = _apply_research_vasp_template(
        spec_obj,
        project=project,
        prompt=normalized_prompt,
        material=material,
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
            + (f" 已应用 research 模板 `{research_template.get('template_id')}`。" if research_template and research_template.get("template_found") else "")
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

    dft_args = build_dft_run_args(
        prompt=prompt,
        material=material,
        structure_path=structure_path,
        task_type=task_type or str(envelope.spec.get("task_type") or ""),
        submit_profile=submit_profile or envelope.spec.get("submit_profile"),
        execution_mode=execution_mode,
    )
    if execution_mode == "dry_run":
        exit_code = 0
        status = "ok"
        build_result = None
        submit_result = None
        run_record_payload = None
    else:
        from dft_app.cli.main import STORE, create_demo_run_record, get_builder, get_remote_runner, get_runner
        from dft_app.models import PipelinePhase, RunStatus

        spec = _experiment_spec_from_payload(envelope.spec)
        run_record = create_demo_run_record(spec)
        build_result = get_builder().build_initial_workspace(spec, run_record)
        submit_result = None
        if execution_mode in {"submit", "remote_submit"}:
            if run_record.overall_status == RunStatus.READY:
                runner_result = get_remote_runner().submit(spec, run_record) if execution_mode == "remote_submit" else get_runner().submit(spec, run_record)
                summary_name = "remote_submit_summary.json" if execution_mode == "remote_submit" else "submit_summary.json"
                submit_summary_path = STORE.write_metadata(
                    Path(run_record.run_root),
                    summary_name,
                    {
                        "status": runner_result.status,
                        "message": runner_result.message,
                        "details": runner_result.details,
                    },
                )
                run_record.phases[PipelinePhase.SUBMIT.value].artifacts.append(str(submit_summary_path))
                STORE.save_run_record(run_record)
                submit_result = {"status": runner_result.status, "message": runner_result.message, "details": runner_result.details}
            else:
                submit_result = {"status": "blocked", "message": "build 未达到 READY，未提交。"}
        exit_code = 0 if build_result and build_result.get("status") in {"ready", "success"} else 1
        if execution_mode in {"submit", "remote_submit"} and submit_result and submit_result.get("status") not in {"submitted", "ok"}:
            exit_code = 1
        status = "ok" if exit_code == 0 else "failed"
        run_record_payload = run_record.to_dict()
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
        "build_result": build_result,
        "submit_result": submit_result,
        "run_record": run_record_payload,
    }
