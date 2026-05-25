from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dft_app.builder.structure_resolver import StructureResolver
from dft_app.modeling import (
    AdsorptionCandidateGenerator,
    AdsorptionGenerationRequest,
    CandidateManifest,
    CandidateManifestWriter,
    ModelSpec,
)
from dft_app.models import ExperimentPlan, ExperimentSpec, PipelinePhase, RunRecord, TaskType
from dft_app.planner.rule_based_planner import RuleBasedPlanner
from dft_app.storage import RecordStore
from rich.console import Console
from dft_app.workflow import (
    AdsorptionEnergyWorkflow,
    WorkflowScaffold,
    WorkflowTaskScaffold,
)

_console = Console(stderr=True)


@dataclass
class OrchestrationResult:
    status: str
    message: str
    details: dict[str, Any]


class ComplexWorkflowOrchestrator:
    """Create a durable scaffold for complex multi-step DFT tasks."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.store = RecordStore(project_root)
        self.structure_resolver = StructureResolver()
        self._builders = {
            "adsorption_energy": AdsorptionEnergyWorkflow(),
        }

    def scaffold(
        self,
        plan: ExperimentPlan,
        run_record: RunRecord,
        model_spec: ModelSpec | None = None,
        structure_path: str | None = None,
    ) -> OrchestrationResult:
        builder = self._builders.get(plan.experiment_type)
        if builder is None:
            workflow_scaffold = self._build_generic_scaffold(plan, model_spec)
            generic_mode = True
        else:
            workflow_scaffold = builder.build(plan)
            generic_mode = False

        run_root = Path(run_record.run_root)
        metadata_dir = run_root / "metadata"
        scaffold_dir = run_root / "scaffold"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        scaffold_dir.mkdir(parents=True, exist_ok=True)

        task_dirs = self._materialize_scaffold(scaffold_dir, workflow_scaffold)
        adsorption_candidate_details = self._maybe_materialize_adsorption_candidates(
            run_root=run_root,
            plan=plan,
            model_spec=model_spec,
            structure_path=structure_path,
        )
        if adsorption_candidate_details is not None:
            workflow_scaffold.metadata["adsorption_candidates"] = adsorption_candidate_details

        plan_path = self.store.write_metadata(run_root, "experiment_plan.json", plan.to_dict())
        model_path = None
        modeling_summary_path = None
        if model_spec is not None:
            model_path = self.store.write_metadata(
                run_root,
                "model_spec.json",
                model_spec.to_dict(),
            )
            modeling_summary_path = self.store.write_metadata(
                run_root,
                "modeling_summary.json",
                {
                    "task_id": model_spec.task_id,
                    "model_type": model_spec.model_type,
                    "readiness": model_spec.readiness,
                    "system_count": len(model_spec.systems),
                    "confirmation_fields": [
                        item["field"]
                        for item in model_spec.to_dict().get("confirmation_summary", [])
                    ],
                },
            )
        scaffold_path = self.store.write_metadata(
            run_root,
            "workflow_scaffold.json",
            workflow_scaffold.to_dict(),
        )
        summary_path = self.store.write_metadata(
            run_root,
            "complex_workflow_summary.json",
            {
                "experiment_type": plan.experiment_type,
                "status": "scaffolded",
                "message": (
                    "复杂任务骨架已生成，等待补充关键信息并实现子任务执行。"
                    if not generic_mode
                    else "已生成通用复杂任务骨架，请人工确认特殊位点/子任务定义。"
                ),
                "task_dirs": task_dirs,
                "missing_information": plan.missing_information,
                "confirmation_items": workflow_scaffold.confirmation_items,
                "generic_scaffold": generic_mode,
                "adsorption_candidates": adsorption_candidate_details,
            },
        )

        run_record.start_phase(PipelinePhase.PLAN, message="已识别复杂任务，正在生成 workflow 骨架")
        run_record.complete_phase(
            PipelinePhase.PLAN,
            artifacts=[
                str(plan_path),
                *([str(model_path)] if model_path is not None else []),
                *([str(modeling_summary_path)] if modeling_summary_path is not None else []),
                str(scaffold_path),
                str(summary_path),
                *task_dirs,
                *(
                    [str(adsorption_candidate_details["summary_path"])]
                    if adsorption_candidate_details is not None
                    else []
                ),
            ],
            message="复杂任务骨架已生成。",
        )
        run_record.block_phase(
            PipelinePhase.BUILD,
            "复杂任务执行层尚未实现，当前已生成子任务目录骨架，请先补充缺失参数并确认。",
        )
        run_record.notes["complex_workflow"] = {
            "experiment_type": plan.experiment_type,
            "workflow_scaffold_path": str(scaffold_path),
            "task_dirs": task_dirs,
            "missing_information": plan.missing_information,
            "adsorption_candidates": adsorption_candidate_details,
        }
        self.store.save_run_record(run_record)
        return OrchestrationResult(
            status="scaffolded",
            message="复杂任务骨架已生成。",
            details={
                "plan_path": str(plan_path),
                "workflow_scaffold_path": str(scaffold_path),
                "task_dirs": task_dirs,
                "missing_information": plan.missing_information,
                "generic_scaffold": generic_mode,
                "adsorption_candidates": adsorption_candidate_details,
            },
        )

    def _maybe_materialize_adsorption_candidates(
        self,
        *,
        run_root: Path,
        plan: ExperimentPlan,
        model_spec: ModelSpec | None,
        structure_path: str | None,
    ) -> dict[str, Any] | None:
        if model_spec is None:
            _console.print("[yellow]⚠ 未检测到 model_spec，跳过吸附候选生成。[/yellow]")
            return None
        if model_spec.model_type != "adsorption_energy":
            return None
        if not structure_path:
            _console.print(
                "[yellow]⚠ 未提供 --structure-path，无法生成吸附候选。"
                "请通过 --structure-path 指定 slab 结构文件。[/yellow]"
            )
            return None

        adsorbate_source = self._infer_adsorbate_source(plan)
        if not adsorbate_source:
            _console.print(
                "[yellow]⚠ 无法从 prompt 推断吸附物，跳过候选生成。"
                "请通过 --adsorbate 显式指定吸附物名称或 SMILES。[/yellow]"
            )
            return None

        structure_spec = ExperimentSpec(
            task_id=plan.task_id,
            task_type=TaskType.RELAX,
            material_name=plan.raw_plan.get("material_name") or plan.summary or "adsorption_system",
            source_prompt=plan.source_prompt,
            structure_path=structure_path,
        )
        structure_resolution = self.structure_resolver.resolve(structure_spec)
        if structure_resolution.structure is None:
            _console.print(
                f"[yellow]⚠ 结构解析失败（{structure_path}），无法生成吸附候选。[/yellow]"
            )
            return None

        defect_recipe = self._normalize_defect_recipe(plan)
        output_dir = run_root / "scaffold" / "adsorption_candidates"
        generator = AdsorptionCandidateGenerator()
        candidates = generator.generate(
            AdsorptionGenerationRequest(
                slab_structure=structure_resolution.structure,
                adsorbate_source=adsorbate_source,
                task_id=plan.task_id,
                material_name=plan.raw_plan.get("material_name") or plan.summary or "adsorption_system",
                source_prompt=plan.source_prompt,
                defect_recipe=defect_recipe,
            )
        )
        manifest = CandidateManifest(
            task_id=plan.task_id,
            material_name=plan.raw_plan.get("material_name") or plan.summary or "adsorption_system",
            source_prompt=plan.source_prompt,
            slab_source=structure_path,
            adsorbate_source=adsorbate_source,
            candidates=candidates,
            metadata={
                "structure_resolution": structure_resolution.metadata,
                "defect_recipe": defect_recipe,
                "generation_mode": "mainline_scaffold",
            },
        )
        writer = CandidateManifestWriter()
        paths = writer.write(manifest, output_dir)
        summary_path = self.store.write_metadata(
            run_root,
            "adsorption_candidate_generation.json",
            {
                "status": "generated",
                "task_id": plan.task_id,
                "candidate_count": len(candidates),
                "manifest": paths,
                "adsorbate_source": adsorbate_source,
                "defect_recipe": defect_recipe,
            },
        )
        return {
            "status": "generated",
            "candidate_count": len(candidates),
            "manifest": paths,
            "adsorbate_source": adsorbate_source,
            "defect_recipe": defect_recipe,
            "summary_path": str(summary_path),
        }

    @staticmethod
    def _infer_adsorbate_source(plan: ExperimentPlan) -> str | None:
        adsorbate_hint = RuleBasedPlanner._extract_adsorbate_hint(plan.source_prompt)
        if adsorbate_hint:
            return ComplexWorkflowOrchestrator._normalize_adsorbate_hint(adsorbate_hint)
        material_name = plan.raw_plan.get("material_name")
        if isinstance(material_name, str) and "/" in material_name:
            return ComplexWorkflowOrchestrator._normalize_adsorbate_hint(
                material_name.split("/", 1)[0].strip()
            )
        return None

    @staticmethod
    def _normalize_adsorbate_hint(value: str | None) -> str | None:
        if not value:
            return None
        text = value.strip()
        generic_match = re.search(r"([A-Z][a-z]?\d?(?:[A-Z][a-z]?\d?)*)", text)
        if generic_match:
            return generic_match.group(1)
        text = re.sub(r"^(计算|研究|评估|分析|构建|生成)\s*", "", text)
        text = re.sub(r"\s*(吸附|adsorption|adsorb).*$", "", text, flags=re.IGNORECASE)
        return text.strip() or None

    @staticmethod
    def _normalize_defect_recipe(plan: ExperimentPlan) -> dict[str, Any] | None:
        defect_hint = RuleBasedPlanner._extract_defect_hint(plan.source_prompt)
        if not defect_hint:
            return None
        if str(defect_hint.get("mode") or "").lower() == "vacancy":
            species = defect_hint.get("species") or defect_hint.get("site")
            if species:
                return {
                    "mode": "vacancy",
                    "species": species,
                    "surface_only": True,
                }
        return None

    def _build_generic_scaffold(
        self,
        plan: ExperimentPlan,
        model_spec: ModelSpec | None,
    ) -> WorkflowScaffold:
        tasks: list[WorkflowTaskScaffold] = []
        if plan.subtasks:
            for index, subtask in enumerate(plan.subtasks, start=1):
                tasks.append(
                    WorkflowTaskScaffold(
                        name=subtask.name,
                        system_role=subtask.system_role,
                        goal=subtask.goal,
                        task_type=subtask.task_type,
                        relative_dir=f"{index:02d}_{subtask.name}",
                        blockers=[
                            "需要人工确认该子任务的结构构筑与输入参数。"
                        ],
                        suggested_inputs=subtask.notes or {},
                        notes={
                            "source": "generic_complex_scaffold",
                            "task_type": subtask.task_type,
                        },
                    )
                )
        elif model_spec is not None:
            for index, system in enumerate(model_spec.systems, start=1):
                tasks.append(
                    WorkflowTaskScaffold(
                        name=system.name,
                        system_role=system.role,
                        goal=system.summary,
                        task_type=system.calc.task_type,
                        relative_dir=f"{index:02d}_{system.name}",
                        blockers=[
                            "需要人工确认该体系的特殊位点、初始结构和构筑策略。"
                        ],
                        suggested_inputs={
                            "build_parameters": system.build.parameters,
                            "template_hints": system.build.template_hints,
                        },
                        notes={
                            "source": "model_spec",
                            "operations": system.build.operations,
                        },
                    )
                )

        if not tasks:
            tasks.append(
                WorkflowTaskScaffold(
                    name="manual_definition",
                    system_role="manual_build",
                    goal="当前复杂任务尚未细化，需人工补充体系和步骤定义。",
                    task_type=None,
                    relative_dir="01_manual_definition",
                    blockers=["需要先明确任务包含哪些体系与中间步骤。"],
                )
            )

        return WorkflowScaffold(
            task_id=plan.task_id,
            workflow_type=plan.experiment_type,
            summary=plan.summary,
            readiness=plan.readiness.value,
            requires_confirmation=True,
            confirmation_items=plan.missing_information or [
                "确认复杂任务的特殊位点、参考体系和初始构型。"
            ],
            shared_assumptions=plan.assumptions,
            missing_information=plan.missing_information,
            tasks=tasks,
            analysis_steps=["整理人工确认后的子任务定义，再进入 builder/runner 主线。"],
            metadata={
                "source": "generic_complex_scaffold",
                "raw_plan": plan.raw_plan,
                "model_type": model_spec.model_type if model_spec is not None else None,
            },
        )

    def _materialize_scaffold(
        self, scaffold_root: Path, workflow_scaffold: WorkflowScaffold
    ) -> list[str]:
        created_dirs: list[str] = []
        for task in workflow_scaffold.tasks:
            task_dir = scaffold_root / Path(task.relative_dir)
            task_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir = task_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "inputs").mkdir(parents=True, exist_ok=True)
            (task_dir / "outputs").mkdir(parents=True, exist_ok=True)
            (task_dir / "report").mkdir(parents=True, exist_ok=True)
            self._write_json(metadata_dir / "subtask_scaffold.json", task.__dict__)
            created_dirs.append(str(task_dir))
        return created_dirs

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
