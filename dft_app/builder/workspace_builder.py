from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import textwrap
from pathlib import Path
from typing import Any

from pymatgen.core import Element, Structure
from pymatgen.io.vasp import Incar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from dft_app.builder.structure_resolver import StructureResolver
from dft_app.builder.vasp_input_generator import VaspInputGenerator
from dft_app.cluster_profiles import SubmitProfile, resolve_submit_profile
from dft_app.modeling import ModelSpec
from dft_app.models import PipelinePhase, RunRecord
from dft_app.models.experiment_spec import ExperimentSpec


class WorkspaceBuilder:
    """Build the initial task workspace and draft input artifacts."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.structure_resolver = StructureResolver()
        self.input_generator = VaspInputGenerator()

    def build_initial_workspace(
        self,
        spec: ExperimentSpec,
        run_record: RunRecord,
        model_spec: ModelSpec | None = None,
    ) -> dict[str, Any]:
        run_root = self.project_root / ".aether" / "runs" / spec.task_id / run_record.run_id
        inputs_dir = run_root / "inputs"
        outputs_dir = run_root / "outputs"
        report_dir = run_root / "report"
        logs_dir = run_root / "logs"
        metadata_dir = run_root / "metadata"

        for path in (inputs_dir, outputs_dir, report_dir, logs_dir, metadata_dir):
            path.mkdir(parents=True, exist_ok=True)

        run_record.run_root = str(run_root)
        run_record.checkpoint_path = str(outputs_dir / ".pipeline_checkpoint.json")

        run_record.start_phase(PipelinePhase.PLAN, message="planner 已生成任务对象")
        self._write_json(metadata_dir / "experiment_spec.json", spec.to_dict())
        self._write_json(metadata_dir / "planner_summary.json", self._planner_summary(spec))
        if model_spec is not None:
            self._write_json(metadata_dir / "model_spec.json", model_spec.to_dict())
            self._write_json(
                metadata_dir / "modeling_summary.json",
                self._modeling_summary(model_spec),
            )
        run_record.complete_phase(
            PipelinePhase.PLAN,
            artifacts=[
                str(metadata_dir / "experiment_spec.json"),
                str(metadata_dir / "planner_summary.json"),
                *(
                    [
                        str(metadata_dir / "model_spec.json"),
                        str(metadata_dir / "modeling_summary.json"),
                    ]
                    if model_spec is not None
                    else []
                ),
            ],
            message="planner 输出已写入 metadata",
        )

        run_record.start_phase(PipelinePhase.BUILD, message="开始生成 workspace 与输入草稿")
        incar_preview_path = inputs_dir / "INCAR.preview.json"
        kpoints_preview_path = inputs_dir / "KPOINTS.preview.json"
        poscar_notes_path = inputs_dir / "POSCAR.notes.txt"
        slurm_path = inputs_dir / "job.slurm"
        structure_resolution_path = metadata_dir / "structure_resolution.json"
        build_summary_path = metadata_dir / "build_summary.json"
        pre_submit_checklist_path = report_dir / "pre_submit_checklist.md"
        generated_artifacts: list[str] = []
        generated_artifacts.extend(
            self._materialize_model_scaffold(run_root, spec, model_spec)
        )
        structure_resolution = self.structure_resolver.resolve(spec)

        self._write_json(incar_preview_path, self._build_incar_preview(spec))
        self._write_json(kpoints_preview_path, self._build_kpoints_preview(spec))
        self._write_json(
            structure_resolution_path,
            self._structure_resolution_summary(structure_resolution),
        )
        self._write_text(slurm_path, self._build_slurm_template(spec))
        self._write_text(
            pre_submit_checklist_path,
            self._build_pre_submit_checklist(spec, model_spec),
        )
        generated_artifacts.extend(
            [
                str(incar_preview_path),
                str(kpoints_preview_path),
                str(structure_resolution_path),
                str(slurm_path),
                str(pre_submit_checklist_path),
            ]
        )

        if structure_resolution.status == "resolved" and structure_resolution.structure is not None:
            generated_inputs = self.input_generator.generate(
                spec,
                structure_resolution.structure,
                inputs_dir,
                structure_resolution.metadata,
            )
            generated_artifacts.extend(
                [
                    generated_inputs["poscar_path"],
                    generated_inputs["incar_path"],
                    generated_inputs["kpoints_path"],
                    generated_inputs["potcar_map_path"],
                ]
            )
            if generated_inputs.get("potcar_path"):
                generated_artifacts.append(generated_inputs["potcar_path"])
            poscar_notes_path.write_text(
                self._build_poscar_notes(spec, structure_resolution),
                encoding="utf-8",
            )
            generated_artifacts.append(str(poscar_notes_path))
            generated_artifacts.extend(
                self._write_model_validation_artifacts(
                    run_root,
                    spec,
                    model_spec,
                    structure_resolution,
                )
            )
            generated_artifacts.extend(
                self._materialize_resolved_model_scaffold(
                    run_root,
                    spec,
                    model_spec,
                    generated_inputs,
                )
            )

            build_message = (
                "workspace 已创建，结构已解析，已生成真实 VASP 输入文件，"
                "提交前仍建议人工确认参数和作业脚本。"
            )
            run_record.complete_phase(
                PipelinePhase.BUILD,
                artifacts=generated_artifacts,
                message=build_message,
            )
            run_record.mark_ready()
        elif structure_resolution.status in {"needs_confirmation", "missing_api_key"}:
            poscar_notes_path.write_text(
                self._build_poscar_notes(spec, structure_resolution),
                encoding="utf-8",
            )
            generated_artifacts.append(str(poscar_notes_path))
            build_message = structure_resolution.message
            run_record.block_phase(PipelinePhase.BUILD, build_message)
        else:
            poscar_notes_path.write_text(
                self._build_poscar_notes(spec, structure_resolution),
                encoding="utf-8",
            )
            generated_artifacts.append(str(poscar_notes_path))
            build_message = structure_resolution.message
            run_record.fail_phase(PipelinePhase.BUILD, build_message)

        generated_artifacts.append(str(build_summary_path))
        self._write_json(
            build_summary_path,
            self._build_summary(
                spec,
                run_record,
                structure_resolution,
                generated_artifacts,
                model_spec,
            ),
        )
        self._append_phase_artifacts(run_record, PipelinePhase.BUILD, generated_artifacts)
        self._write_json(Path(run_record.checkpoint_path), run_record.to_dict())
        self._write_json(metadata_dir / "run_record.json", run_record.to_dict())

        return {
            "run_root": str(run_root),
            "inputs_dir": str(inputs_dir),
            "metadata_dir": str(metadata_dir),
            "checkpoint_path": run_record.checkpoint_path,
            "status": run_record.overall_status.value,
            "structure_resolution_status": structure_resolution.status,
            "structure_resolution_message": structure_resolution.message,
        }

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8", newline="\n")

    @staticmethod
    def _planner_summary(spec: ExperimentSpec) -> dict[str, Any]:
        return {
            "task_id": spec.task_id,
            "task_type": spec.task_type.value,
            "material_name": spec.material_name,
            "workflow": spec.workflow,
            "functional": spec.functional,
            "structure_source": spec.structure_source.value,
            "structure_id": spec.structure_id,
            "structure_path": spec.structure_path,
            "incar_overrides": spec.incar_overrides,
            "submit_profile": spec.submit_profile,
        }

    @staticmethod
    def _modeling_summary(model_spec: ModelSpec) -> dict[str, Any]:
        return {
            "task_id": model_spec.task_id,
            "model_type": model_spec.model_type,
            "source_kind": model_spec.source_kind.value,
            "readiness": model_spec.readiness,
            "requires_confirmation": model_spec.requires_confirmation,
            "system_count": len(model_spec.systems),
            "systems": [
                {
                    "name": system.name,
                    "role": system.role,
                    "operations": system.build.operations,
                    "dependencies": system.dependencies,
                }
                for system in model_spec.systems
            ],
            "confirmation_fields": [
                item.field for item in model_spec.confirmation_summary
            ],
        }

    @staticmethod
    def _build_incar_preview(spec: ExperimentSpec) -> dict[str, Any]:
        preview = {
            "functional": spec.functional,
            "task_type": spec.task_type.value,
            "defaults": {
                "EDIFF": spec.convergence_settings.ediff,
                "EDIFFG": spec.convergence_settings.ediffg,
                "NSW": spec.convergence_settings.nsw,
                "ISMEAR": spec.smearing.ismear,
                "SIGMA": spec.smearing.sigma,
            },
            "overrides": spec.incar_overrides,
        }
        return preview

    @staticmethod
    def _build_kpoints_preview(spec: ExperimentSpec) -> dict[str, Any]:
        return {
            "mode": spec.kpoints_strategy.mode,
            "value": spec.kpoints_strategy.value,
        }

    @staticmethod
    def _build_poscar_notes(spec: ExperimentSpec, resolution: Any) -> str:
        metadata = resolution.metadata or {}
        fixed_atom_count = metadata.get("fixed_atom_count")
        template_incar_path = metadata.get("template_incar_path")
        template_kpoints_path = metadata.get("template_kpoints_path")
        if resolution.status == "resolved":
            lines = [
                "POSCAR 已真实生成。\n"
                f"结构来源: {resolution.source}\n",
                f"来源详情: {resolution.source_detail or '未记录'}\n",
            ]
            if metadata.get("input_format"):
                lines.append(f"输入格式: {metadata['input_format']}\n")
            if fixed_atom_count is not None:
                lines.append(f"固定原子数: {fixed_atom_count}\n")
            if template_incar_path:
                lines.append(f"INCAR 模板: {template_incar_path}\n")
            if template_kpoints_path:
                lines.append(f"KPOINTS 模板: {template_kpoints_path}\n")
            lines.append("请在提交前确认原子排序、Selective Dynamics、磁性和赝势映射是否符合预期。\n")
            return "".join(lines)
        if spec.structure_source.value == "local_file" and spec.structure_path:
            return (
                "POSCAR 尚未生成。\n"
                f"本地结构文件预期路径: {spec.structure_path}\n"
                f"当前状态: {resolution.message}\n"
            )
        if spec.structure_source.value == "materials_project":
            return (
                "POSCAR 尚未生成。\n"
                f"Materials Project 结构标识: {spec.structure_id}\n"
                f"当前状态: {resolution.message}\n"
            )
        return (
            "POSCAR 尚未生成。\n"
            f"当前状态: {resolution.message}\n"
            "需要人工确认结构来源或补充结构文件。\n"
        )

    @staticmethod
    def _build_slurm_template(spec: ExperimentSpec) -> str:
        profile = resolve_submit_profile(spec)
        partition = spec.job_overrides.partition or profile.partition
        nodes = spec.job_overrides.nodes or profile.nodes
        ntasks_per_node = spec.job_overrides.ntasks_per_node or profile.ntasks_per_node
        walltime = spec.job_overrides.walltime or profile.walltime
        memory_per_cpu = spec.job_overrides.memory_per_cpu or profile.memory_per_cpu
        vasp_variant = spec.job_overrides.vasp_variant or profile.vasp_variant
        job_name = f"{spec.material_name}_{spec.task_type.value}"
        lines = [
            "#!/bin/bash",
            f"#SBATCH -p {partition}",
            f"#SBATCH -J {job_name}",
            "#SBATCH -o logs/slurm.out",
            "#SBATCH -e logs/slurm.err",
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --ntasks-per-node {ntasks_per_node}",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --mem-per-cpu {memory_per_cpu}",
            "",
            "ulimit -s unlimited",
            profile.oneapi_source,
        ]
        lines.extend(profile.module_load_commands)
        lines.append(profile.run_command_template.format(vasp_variant=vasp_variant))
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_summary(
        spec: ExperimentSpec,
        run_record: RunRecord,
        structure_resolution: Any,
        generated_artifacts: list[str],
        model_spec: ModelSpec | None = None,
    ) -> dict[str, Any]:
        return {
            "task_id": spec.task_id,
            "run_id": run_record.run_id,
            "workflow": spec.workflow,
            "structure_source": spec.structure_source.value,
            "structure_path": spec.structure_path,
            "structure_id": spec.structure_id,
            "requires_confirmation": spec.requires_confirmation,
            "current_status": run_record.overall_status.value,
            "submit_profile": spec.submit_profile,
            "scheduler_profile": WorkspaceBuilder._scheduler_profile_summary(spec),
            "modeling": (
                {
                    "model_type": model_spec.model_type,
                    "readiness": model_spec.readiness,
                    "system_count": len(model_spec.systems),
                    "confirmation_fields": [
                        item.field for item in model_spec.confirmation_summary
                    ],
                }
                if model_spec is not None
                else None
            ),
            "structure_resolution": {
                "status": structure_resolution.status,
                "source": structure_resolution.source,
                "source_detail": structure_resolution.source_detail,
                "message": structure_resolution.message,
                "metadata": structure_resolution.metadata or {},
            },
            "generated_artifacts": generated_artifacts,
        }

    @staticmethod
    def _structure_resolution_summary(structure_resolution: Any) -> dict[str, Any]:
        return {
            "status": structure_resolution.status,
            "source": structure_resolution.source,
            "source_detail": structure_resolution.source_detail,
            "message": structure_resolution.message,
            "metadata": structure_resolution.metadata or {},
        }

    @staticmethod
    def _scheduler_profile_summary(spec: ExperimentSpec) -> dict[str, Any]:
        profile = resolve_submit_profile(spec)
        return {
            "profile_name": profile.name,
            "partition": spec.job_overrides.partition or profile.partition,
            "nodes": spec.job_overrides.nodes or profile.nodes,
            "ntasks_per_node": spec.job_overrides.ntasks_per_node
            or profile.ntasks_per_node,
            "walltime": spec.job_overrides.walltime or profile.walltime,
            "memory_per_cpu": spec.job_overrides.memory_per_cpu
            or profile.memory_per_cpu,
            "vasp_variant": spec.job_overrides.vasp_variant or profile.vasp_variant,
            "module_load_commands": profile.module_load_commands,
        }

    @staticmethod
    def _append_phase_artifacts(
        run_record: RunRecord, phase: PipelinePhase, artifacts: list[str]
    ) -> None:
        phase_artifacts = run_record.phases[phase.value].artifacts
        for artifact in artifacts:
            if artifact not in phase_artifacts:
                phase_artifacts.append(artifact)

    def _materialize_model_scaffold(
        self,
        run_root: Path,
        spec: ExperimentSpec,
        model_spec: ModelSpec | None,
    ) -> list[str]:
        if model_spec is None:
            return []

        scaffold_root = run_root / "scaffold"
        artifacts: list[str] = []

        if model_spec.model_type == "defect_doping":
            defect_hint = spec.structure_constraints.defect or {}
            supercell_hint = spec.structure_constraints.supercell
            for name, role, todo in [
                (
                    "defect_system",
                    "defect_or_doped_supercell",
                    [
                        "确认缺陷类型或掺杂元素。",
                        "确认目标位点与超胞尺寸。",
                        "准备缺陷体系的起始结构。",
                    ],
                ),
                (
                    "pristine_reference",
                    "pristine_supercell_reference",
                    [
                        "准备与缺陷体系同尺寸的本征超胞。",
                        "确保计算参数与缺陷体系保持可比。",
                    ],
                ),
            ]:
                task_dir = scaffold_root / name
                metadata_dir = task_dir / "metadata"
                for path in (task_dir, metadata_dir, task_dir / "inputs", task_dir / "outputs"):
                    path.mkdir(parents=True, exist_ok=True)
                scaffold_payload = {
                    "name": name,
                    "role": role,
                    "status": "pending_confirmation",
                    "todo": todo,
                    "defect_hint": defect_hint,
                    "supercell_hint": supercell_hint,
                }
                scaffold_path = metadata_dir / "subtask_scaffold.json"
                self._write_json(scaffold_path, scaffold_payload)
                artifacts.append(str(scaffold_path))

        if model_spec.model_type == "work_function":
            task_dir = scaffold_root / "surface_checks"
            metadata_dir = task_dir / "metadata"
            task_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)
            checklist_path = metadata_dir / "surface_checklist.json"
            self._write_json(
                checklist_path,
                {
                    "name": "surface_checks",
                    "status": "pending_confirmation",
                    "surface_hint": spec.structure_constraints.surface,
                    "items": [
                        "确认输入结构确实是表面 slab 而非 bulk。",
                        "确认真空层厚度是否足够。",
                        "确认是否需要偶极修正。",
                        "确认表面法向与功函数提取方向一致。",
                    ],
                },
            )
            artifacts.append(str(checklist_path))

        return artifacts

    def _build_pre_submit_checklist(
        self,
        spec: ExperimentSpec,
        model_spec: ModelSpec | None,
    ) -> str:
        lines = [
            f"# 预提交检查单: {spec.material_name}",
            "",
            f"- task_type: `{spec.task_type.value}`",
            f"- workflow: `{', '.join(spec.workflow)}`",
            f"- structure_source: `{spec.structure_source.value}`",
            "",
            "## 必查项",
        ]

        if model_spec is not None and model_spec.confirmation_summary:
            for item in model_spec.confirmation_summary:
                lines.append(
                    f"- [{item.level.value}] `{item.field}`: {item.reason}"
                )
        else:
            lines.append("- [required] `structure`: 确认输入结构、计算参数与提交脚本。")

        if model_spec is not None and model_spec.assumptions:
            lines.extend(["", "## 当前默认假设"])
            for assumption in model_spec.assumptions:
                lines.append(f"- {assumption}")

        lines.extend(["", "## 提交前提醒"])
        lines.append("- 检查 `inputs/INCAR`、`inputs/KPOINTS`、`inputs/POSCAR` 与 `inputs/job.slurm`。")
        lines.append("- 若任务依赖参考体系或后处理文件，请确认对应目录骨架与输出保留策略。")
        return "\n".join(lines) + "\n"

    def _write_model_validation_artifacts(
        self,
        run_root: Path,
        spec: ExperimentSpec,
        model_spec: ModelSpec | None,
        structure_resolution: Any,
    ) -> list[str]:
        if model_spec is None or structure_resolution.structure is None:
            return []

        metadata_dir = run_root / "metadata"
        artifacts: list[str] = []
        structure = structure_resolution.structure

        if model_spec.model_type == "work_function":
            vacuum_summary = self._estimate_surface_vacuum(structure)
            vacuum_summary["task_type"] = spec.task_type.value
            vacuum_summary["note"] = "该估计基于三个晶格方向的占据跨度近似，仅作为预检查提示。"
            path = metadata_dir / "surface_model_validation.json"
            self._write_json(path, vacuum_summary)
            artifacts.append(str(path))

        if model_spec.model_type == "defect_doping":
            defect_hint = spec.structure_constraints.defect or {}
            supercell_hint = spec.structure_constraints.supercell
            defect_summary = {
                "task_type": spec.task_type.value,
                "atom_count": len(structure),
                "species": sorted({site.specie.symbol for site in structure}),
                "requires_pristine_reference": True,
                "defect_hint": defect_hint,
                "supercell_hint": supercell_hint,
                "note": "缺陷与掺杂计算建议与同尺寸本征超胞配对使用。",
            }
            path = metadata_dir / "defect_model_validation.json"
            self._write_json(path, defect_summary)
            artifacts.append(str(path))

        return artifacts

    @staticmethod
    def _estimate_surface_vacuum(structure: Any) -> dict[str, Any]:
        axis_labels = ["a", "b", "c"]
        lattice_lengths = [float(structure.lattice.a), float(structure.lattice.b), float(structure.lattice.c)]
        axis_summaries: list[dict[str, Any]] = []
        for index, label in enumerate(axis_labels):
            fractions = [float(site.frac_coords[index]) for site in structure]
            axis_min = min(fractions)
            axis_max = max(fractions)
            span = max(axis_max - axis_min, 0.0)
            thickness = span * lattice_lengths[index]
            vacuum = max(lattice_lengths[index] - thickness, 0.0)
            axis_summaries.append(
                {
                    "axis": label,
                    "idipol": index + 1,
                    "lattice_length": lattice_lengths[index],
                    "fraction_span": span,
                    "estimated_slab_thickness": thickness,
                    "estimated_vacuum": vacuum,
                }
            )

        best_axis = max(axis_summaries, key=lambda item: item["estimated_vacuum"])
        return {
            "axes": axis_summaries,
            "recommended_vacuum_axis": best_axis["axis"],
            "recommended_idipol": best_axis["idipol"],
            "estimated_vacuum_along_recommended_axis": best_axis["estimated_vacuum"],
            "estimated_slab_thickness_along_recommended_axis": best_axis["estimated_slab_thickness"],
            "vacuum_ok_guess": best_axis["estimated_vacuum"] >= 12.0,
            "surface_like_guess": best_axis["estimated_vacuum"] >= 8.0
            and not math.isclose(best_axis["fraction_span"], 1.0),
        }

    def _materialize_resolved_model_scaffold(
        self,
        run_root: Path,
        spec: ExperimentSpec,
        model_spec: ModelSpec | None,
        generated_inputs: dict[str, Any],
    ) -> list[str]:
        if model_spec is None:
            return []

        artifacts: list[str] = []
        poscar_path = Path(generated_inputs["poscar_path"])
        poscar_text = poscar_path.read_text(encoding="utf-8")
        scaffold_root = run_root / "scaffold"

        if model_spec.model_type == "defect_doping":
            defect_hint = spec.structure_constraints.defect or {}
            supercell_hint = spec.structure_constraints.supercell
            seed_text = poscar_text
            seed_structure = Structure.from_file(poscar_path)
            if supercell_hint:
                seed_structure = self._build_supercell_seed_structure(seed_structure, supercell_hint)
                seed_text = seed_structure.to(fmt="poscar")
            for name, note in [
                (
                    "defect_system",
                    "请在该 seed 结构基础上引入目标缺陷或掺杂，并与本征参考保持相同超胞尺寸。",
                ),
                (
                    "pristine_reference",
                    "请保留该 seed 作为本征参考，确保计算参数与缺陷体系一致。",
                ),
            ]:
                inputs_dir = scaffold_root / name / "inputs"
                inputs_dir.mkdir(parents=True, exist_ok=True)
                seed_path = inputs_dir / "POSCAR.seed"
                note_path = inputs_dir / "README.txt"
                recipe_path = inputs_dir / "defect_recipe.json"
                self._write_text(seed_path, seed_text)
                self._write_text(
                    note_path,
                    note
                    + (
                        f"\n建议缺陷/掺杂信息: {json.dumps(defect_hint, ensure_ascii=False)}"
                        if defect_hint
                        else ""
                    )
                    + (
                        f"\n建议超胞: {json.dumps(supercell_hint, ensure_ascii=False)}"
                        if supercell_hint
                        else ""
                    )
                    + "\n",
                )
                self._write_json(
                    recipe_path,
                    {
                        "task_type": spec.task_type.value,
                        "role": name,
                        "defect_hint": defect_hint,
                        "supercell_hint": supercell_hint,
                        "source_seed": str(seed_path),
                    },
                )
                artifacts.extend([str(seed_path), str(note_path), str(recipe_path)])
                if name == "defect_system":
                    artifacts.extend(
                        self._materialize_defect_candidates(
                            inputs_dir,
                            seed_structure,
                            defect_hint,
                            generated_inputs,
                        )
                    )

        if model_spec.model_type == "work_function":
            metadata_dir = scaffold_root / "surface_checks" / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            vacuum_summary = self._estimate_surface_vacuum(Structure.from_file(poscar_path))
            recommended_path = metadata_dir / "recommended_overrides.json"
            self._write_json(
                recommended_path,
                {
                    "task_type": spec.task_type.value,
                    "surface_hint": spec.structure_constraints.surface,
                    "optional_incar_overrides": {
                        "LVHAR": True,
                        "LDIPOL": True,
                        "IDIPOL": vacuum_summary["recommended_idipol"],
                    },
                    "vacuum_analysis": vacuum_summary,
                    "note": "功函数表面任务常见的可选设置，IDIPOL 已按估计的真空主方向给出建议，是否启用仍需人工确认。",
                },
            )
            artifacts.append(str(recommended_path))
            bundle_dir = scaffold_root / "surface_checks" / "suggested_bundle"
            artifacts.extend(
                self._write_suggested_input_bundle(
                    bundle_dir=bundle_dir,
                    generated_inputs=generated_inputs,
                    poscar_text=poscar_text,
                    incar_overrides={
                        "LVHAR": True,
                        "LDIPOL": True,
                        "IDIPOL": vacuum_summary["recommended_idipol"],
                    },
                    title="功函数任务建议输入包",
                    notes=[
                        "该目录可作为人工确认后的起点，已保留当前 POSCAR 并给出功函数任务常见的 INCAR 建议。",
                        f"估计真空主方向: {vacuum_summary['recommended_vacuum_axis']} 轴",
                        f"建议 IDIPOL: {vacuum_summary['recommended_idipol']}",
                        "请重点确认 slab 法向、真空厚度以及是否确实需要偶极修正。",
                    ],
                )
            )

        return artifacts

    @staticmethod
    def _build_supercell_seed_structure(
        structure: Structure,
        supercell_hint: list[list[int]],
    ) -> Structure:
        structure = structure.copy()
        structure.make_supercell(supercell_hint)
        return structure

    def _materialize_defect_candidates(
        self,
        inputs_dir: Path,
        seed_structure: Structure,
        defect_hint: dict[str, Any],
        generated_inputs: dict[str, Any],
    ) -> list[str]:
        if not defect_hint:
            return []

        candidates_dir = inputs_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[str] = []
        mode = str(defect_hint.get("mode") or "")
        dopant = defect_hint.get("dopant")
        site_hint = defect_hint.get("site")
        site_role_hint = defect_hint.get("site_role")
        requested_geometry_hint = defect_hint.get("geometry_hint")
        site_mode = defect_hint.get("site_mode")
        layer_region_hint = defect_hint.get("layer_region_hint")
        wyckoff = defect_hint.get("wyckoff")
        surface_site_hint = defect_hint.get("surface_site_hint")
        host_anion_family = self._infer_host_anion_family(seed_structure)
        host_framework_label = self._host_framework_label(host_anion_family)
        normalized_site_hint = self._normalize_site_hint(site_hint)
        candidate_summaries: list[dict[str, Any]] = []

        unique_species = list(dict.fromkeys(site.specie.symbol for site in seed_structure))
        enforce_site_filter = bool(normalized_site_hint and normalized_site_hint in unique_species)
        if normalized_site_hint and normalized_site_hint not in unique_species:
            warning_path = candidates_dir / "site_hint_warning.txt"
            self._write_text(
                warning_path,
                (
                    f"提示中的目标位点元素 `{normalized_site_hint}` 未在当前结构中找到。\n"
                    f"当前结构包含的元素: {', '.join(unique_species)}\n"
                    "系统已保留所有宿主元素的候选结构作为备选，请优先检查输入结构是否正确。\n"
                ),
            )
            artifacts.append(str(warning_path))
        allowed_species = [
            species
            for species in unique_species
            if (not dopant or species != dopant)
            and (not enforce_site_filter or normalized_site_hint == species)
        ]
        if site_mode == "interstitial":
            candidate_sites = self._enumerate_interstitial_sites(seed_structure)
        else:
            candidate_sites = self._enumerate_candidate_sites(seed_structure, allowed_species)
        if site_mode == "interstitial":
            layer_axis_diagnostics = self._collect_layer_axis_diagnostics(seed_structure)
            diagnostics_json_path = candidates_dir / "layer_axis_diagnostics.json"
            diagnostics_md_path = candidates_dir / "layer_axis_diagnostics.md"
            self._write_json(diagnostics_json_path, layer_axis_diagnostics)
            self._write_text(
                diagnostics_md_path,
                self._build_layer_axis_diagnostics_markdown(layer_axis_diagnostics),
            )
            artifacts.extend([str(diagnostics_json_path), str(diagnostics_md_path)])
        if site_role_hint and not any(
            str(candidate_site["site_role"]) == str(site_role_hint)
            for candidate_site in candidate_sites
        ):
            warning_path = candidates_dir / "site_role_warning.txt"
            self._write_text(
                warning_path,
                (
                    f"提示中的位点角色 `{site_role_hint}` 未在当前候选中找到匹配项。\n"
                    "请确认输入结构、目标位点类型或提示语是否正确。\n"
                ),
            )
            artifacts.append(str(warning_path))
        if requested_geometry_hint and not any(
            str(candidate_site["geometry_hint"]) == str(requested_geometry_hint)
            for candidate_site in candidate_sites
        ):
            warning_path = candidates_dir / "geometry_hint_warning.txt"
            self._write_text(
                warning_path,
                (
                    f"提示中的局部几何 `{requested_geometry_hint}` 未在当前候选中找到匹配项。\n"
                    "当前结构可能与预期构型不一致，建议优先核对结构来源。\n"
                ),
            )
            artifacts.append(str(warning_path))
        if site_mode == "interstitial":
            warning_path = candidates_dir / "site_mode_warning.txt"
            self._write_text(
                warning_path,
                (
                    "当前请求更像是间隙位构筑，而不是普通替位缺陷。\n"
                    "当前版本会自动给出一批粗略间隙位候选结构，但仍需要人工确认真实位点坐标、Wyckoff 对应关系和初始放置策略。\n"
                ),
            )
            artifacts.append(str(warning_path))
        if layer_region_hint and not any(
            str(candidate_site.get("layer_region")) == str(layer_region_hint)
            for candidate_site in candidate_sites
        ):
            detected_axis = next(
                (
                    str(candidate_site.get("layer_axis"))
                    for candidate_site in candidate_sites
                    if candidate_site.get("layer_axis") is not None
                ),
                None,
            )
            detected_axis_score = next(
                (
                    candidate_site.get("layer_axis_score")
                    for candidate_site in candidate_sites
                    if candidate_site.get("layer_axis_score") is not None
                ),
                None,
            )
            warning_path = candidates_dir / "layer_region_warning.txt"
            self._write_text(
                warning_path,
                (
                    f"提示中的区域偏好 `{layer_region_hint}` 未在当前候选中找到匹配项。\n"
                    + (
                        f"系统当前更像是沿 `{detected_axis}` 轴识别分层，评分约为 {float(detected_axis_score):.3f}。\n"
                        if detected_axis is not None and detected_axis_score is not None
                        else ""
                    )
                    +
                    "建议优先核对输入结构是否与预期的层间/层内位点环境一致。\n"
                ),
            )
            artifacts.append(str(warning_path))
        if wyckoff or surface_site_hint:
            request_path = candidates_dir / "special_site_request.json"
            self._write_json(
                request_path,
                {
                    "site_mode": site_mode,
                    "layer_region_hint": layer_region_hint,
                    "wyckoff": wyckoff,
                    "surface_site_hint": surface_site_hint,
                    "site_hint": site_hint,
                    "site_role_hint": site_role_hint,
                    "geometry_hint": requested_geometry_hint,
                    "note": "这些特殊位点提示已被结构化记录，当前版本仍需人工确认具体构筑方式。",
                },
            )
            artifacts.append(str(request_path))
        for candidate_site in candidate_sites:
            species = candidate_site["host_species"]
            first_index = candidate_site["site_index"]
            candidate_name = self._candidate_name_for_site(
                mode=mode,
                site_mode=site_mode,
                candidate_site=candidate_site,
            )
            candidate_dir = candidates_dir / candidate_name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_structure = seed_structure.copy()
            if site_mode == "interstitial" and dopant:
                candidate_structure.append(
                    dopant,
                    candidate_site["frac_coords"],
                    coords_are_cartesian=False,
                )
            elif mode == "doping" and dopant:
                candidate_structure.replace(first_index, dopant)
            elif mode == "vacancy":
                candidate_structure.remove_sites([first_index])
            else:
                continue

            poscar_path = candidate_dir / "POSCAR"
            note_path = candidate_dir / "candidate_summary.txt"
            recipe_path = candidate_dir / "candidate_recipe.json"
            self._write_text(poscar_path, candidate_structure.to(fmt="poscar"))
            if site_mode == "interstitial":
                score, reason = self._score_interstitial_candidate(
                    dopant=dopant,
                    host_anion_family=host_anion_family,
                    multiplicity=int(candidate_site["multiplicity"]),
                    candidate_site_role=str(candidate_site["site_role"]),
                    requested_site_role=site_role_hint,
                    candidate_geometry=str(candidate_site["geometry_hint"]),
                    requested_geometry=requested_geometry_hint,
                    candidate_layer_region=str(candidate_site.get("layer_region")),
                    requested_layer_region=layer_region_hint,
                    candidate_gap_rank=candidate_site.get("layer_gap_rank"),
                    candidate_gap_count=candidate_site.get("layer_gap_count"),
                    placement_source=candidate_site.get("placement_source"),
                    environment_label=candidate_site.get("environment_label"),
                    environment_quality_score=candidate_site.get("environment_quality_score"),
                    environment_preference_match=(
                        candidate_site.get("environment_host_preference")
                        == self._preferred_host_environment_for_dopant(dopant)
                        if self._preferred_host_environment_for_dopant(dopant) is not None
                        else None
                    ),
                    dominant_anion_family=candidate_site.get("environment_dominant_anion_family"),
                    dopant_anion_family_preference=self._preferred_anion_family_for_dopant(dopant),
                    dopant_anion_family_match=(
                        (
                            candidate_site.get("environment_dominant_anion_family")
                            in {"oxide", "chalcogen"}
                        )
                        if self._preferred_anion_family_for_dopant(dopant) == "oxide_or_chalcogen"
                        else (
                            candidate_site.get("environment_dominant_anion_family")
                            in {"chalcogen", "halide"}
                        )
                        if self._preferred_anion_family_for_dopant(dopant) == "chalcogen_or_halide"
                        else (
                            candidate_site.get("environment_dominant_anion_family")
                            in {"oxide", "halide"}
                        )
                        if self._preferred_anion_family_for_dopant(dopant) == "oxide_or_halide"
                        else (
                            candidate_site.get("environment_dominant_anion_family") == "metal"
                        )
                        if self._preferred_anion_family_for_dopant(dopant) == "metal_surrounded"
                        else None
                    ),
                    shell_quality_label=candidate_site.get("shell_quality_label"),
                    shell_quality_score=candidate_site.get("shell_quality_score"),
                    clearance=float(candidate_site.get("clearance", 0.0)),
                    wyckoff=candidate_site.get("wyckoff"),
                    requested_wyckoff=wyckoff,
                    family_size=int(candidate_site.get("family_size", 1)),
                )
            else:
                score, reason = self._score_defect_candidate(
                    host_species=species,
                    site_hint=normalized_site_hint,
                    multiplicity=int(candidate_site["multiplicity"]),
                    candidate_site_role=str(candidate_site["site_role"]),
                    requested_site_role=site_role_hint,
                    candidate_geometry=str(candidate_site["geometry_hint"]),
                    requested_geometry=requested_geometry_hint,
                )
            self._write_text(
                note_path,
                (
                    f"自动生成的候选结构: {candidate_name}\n"
                    f"原子种类: {species}\n"
                    f"位点索引: {first_index}\n"
                    f"等价位点数: {candidate_site['multiplicity']}\n"
                    f"代表分数坐标: {candidate_site['frac_coords']}\n"
                    f"局部环境摘要: {candidate_site['coordination_hint']}\n"
                    f"位点角色判断: {candidate_site['site_role']}\n"
                    f"几何判断: {candidate_site['geometry_hint']}\n"
                    f"区域判断: {candidate_site.get('layer_region', 'unknown_region')}\n"
                    + (
                        f"局部环境类型: {candidate_site['environment_label']}\n"
                        if candidate_site.get("environment_label") is not None
                        else ""
                    )
                    + (
                        f"局部环境评分: {candidate_site['environment_quality_score']}\n"
                        if candidate_site.get("environment_quality_score") is not None
                        else ""
                    )
                    + (
                        "局部环境角色统计: "
                        + ", ".join(
                            f"{key}={value}"
                            for key, value in (candidate_site.get("environment_role_counts") or {}).items()
                        )
                        + "\n"
                        if candidate_site.get("environment_role_counts")
                        else ""
                    )
                    + (
                        "局部环境家族统计: "
                        + ", ".join(
                            f"{key}={value}"
                            for key, value in (candidate_site.get("environment_family_counts") or {}).items()
                        )
                        + "\n"
                        if candidate_site.get("environment_family_counts")
                        else ""
                    )
                    + (
                        f"主导阴离子家族: {candidate_site['environment_dominant_anion_family']}\n"
                        if candidate_site.get("environment_dominant_anion_family") is not None
                        else ""
                    )
                    + (
                        f"局部环境阴离子占比: {candidate_site['environment_anion_fraction']:.3f}\n"
                        if candidate_site.get("environment_anion_fraction") is not None
                        else ""
                    )
                    + (
                        f"局部环境偏好: {candidate_site['environment_host_preference']}\n"
                        if candidate_site.get("environment_host_preference") is not None
                        else ""
                    )
                    + (
                        f"局部环境距离离散度: {candidate_site['environment_distance_spread']:.3f} A\n"
                        if candidate_site.get("environment_distance_spread") is not None
                        else ""
                    )
                    + (
                        f"第一配位壳层质量: {candidate_site['shell_quality_label']}\n"
                        if candidate_site.get("shell_quality_label") is not None
                        else ""
                    )
                    + (
                        f"第一配位壳层评分: {candidate_site['shell_quality_score']}\n"
                        if candidate_site.get("shell_quality_score") is not None
                        else ""
                    )
                    + (
                        f"第一配位壳层数: {candidate_site['shell_first_shell_count']}\n"
                        if candidate_site.get("shell_first_shell_count") is not None
                        else ""
                    )
                    + (
                        f"第一配位平均距离: {candidate_site['shell_mean_distance']:.3f} A\n"
                        if candidate_site.get("shell_mean_distance") is not None
                        else ""
                    )
                    + (
                        f"第一配位距离离散度: {candidate_site['shell_distance_spread']:.3f} A\n"
                        if candidate_site.get("shell_distance_spread") is not None
                        else ""
                    )
                    + (
                        f"分层轴判断: {candidate_site['layer_axis']}\n"
                        if candidate_site.get("layer_axis") is not None
                        else ""
                    )
                    + (
                        f"分层轴评分: {candidate_site['layer_axis_score']:.3f}\n"
                        if candidate_site.get("layer_axis_score") is not None
                        else ""
                    )
                    + (
                        f"沿分层轴最大间隙: {candidate_site['layer_axis_max_gap']:.3f} A\n"
                        if candidate_site.get("layer_axis_max_gap") is not None
                        else ""
                    )
                    + (
                        f"层间隙估计: {candidate_site['layer_gap_size']:.3f} A\n"
                        if candidate_site.get("layer_gap_size") is not None
                        else ""
                    )
                    + (
                        f"层间隙排名: 第 {candidate_site['layer_gap_rank']} / {candidate_site['layer_gap_count']} 大\n"
                        if candidate_site.get("layer_gap_rank") is not None
                        and candidate_site.get("layer_gap_count") is not None
                        else ""
                    )
                    + (
                        f"放置来源: {candidate_site['placement_source']}\n"
                        if candidate_site.get("placement_source") is not None
                        else ""
                    )
                    + (
                        f"距层间隙中点偏移: {candidate_site['layer_midpoint_offset']:.3f} A\n"
                        if candidate_site.get("layer_midpoint_offset") is not None
                        else ""
                    )
                    +
                    f"候选家族: {candidate_site.get('family_label', 'unknown_family')}\n"
                    f"候选家族规模: {candidate_site.get('family_size', 1)}\n"
                    + (
                        f"间隙位净空估计: {candidate_site['clearance']:.3f} A\n"
                        if candidate_site.get("clearance") is not None
                        else ""
                    )
                    +
                    f"推荐分数: {score}\n"
                    f"推荐原因: {reason}\n"
                    + (
                        f"替换为掺杂元素: {dopant}\n" if mode == "doping" and dopant else ""
                    )
                    + (f"位点提示: {site_hint}\n" if site_hint else "")
                    + (f"位点角色提示: {site_role_hint}\n" if site_role_hint else "")
                    + (f"几何提示: {requested_geometry_hint}\n" if requested_geometry_hint else "")
                    + (f"位点模式: {site_mode}\n" if site_mode else "")
                    + (f"区域偏好: {layer_region_hint}\n" if layer_region_hint else "")
                    + (f"Wyckoff 提示: {wyckoff}\n" if wyckoff else "")
                    + (f"宿主阴离子主族: {host_anion_family}\n" if host_anion_family else "")
                    + (f"宿主框架判断: {host_framework_label}\n" if host_framework_label else "")
                    + (f"表面位点提示: {surface_site_hint}\n" if surface_site_hint else "")
                ),
            )
            self._write_json(
                recipe_path,
                {
                    "mode": mode,
                    "dopant": dopant,
                    "host_species": species,
                    "site_index": first_index,
                    "equivalent_site_indices": candidate_site["equivalent_site_indices"],
                    "multiplicity": candidate_site["multiplicity"],
                    "frac_coords": candidate_site["frac_coords"],
                    "coordination_hint": candidate_site["coordination_hint"],
                    "site_role": candidate_site["site_role"],
                    "geometry_hint": candidate_site["geometry_hint"],
                    "layer_region": candidate_site.get("layer_region"),
                    "layer_axis": candidate_site.get("layer_axis"),
                    "layer_axis_score": candidate_site.get("layer_axis_score"),
                    "layer_axis_max_gap": candidate_site.get("layer_axis_max_gap"),
                    "layer_gap_size": candidate_site.get("layer_gap_size"),
                    "layer_gap_rank": candidate_site.get("layer_gap_rank"),
                    "layer_gap_count": candidate_site.get("layer_gap_count"),
                    "placement_source": candidate_site.get("placement_source"),
                    "layer_midpoint_offset": candidate_site.get("layer_midpoint_offset"),
                    "neighbor_fingerprint": candidate_site.get("neighbor_fingerprint"),
                    "environment_label": candidate_site.get("environment_label"),
                    "environment_quality_score": candidate_site.get("environment_quality_score"),
                    "environment_role_counts": candidate_site.get("environment_role_counts"),
                    "environment_family_counts": candidate_site.get("environment_family_counts"),
                    "environment_dominant_anion_family": candidate_site.get("environment_dominant_anion_family"),
                    "environment_host_preference": candidate_site.get("environment_host_preference"),
                    "environment_anion_fraction": candidate_site.get("environment_anion_fraction"),
                    "environment_distance_spread": candidate_site.get("environment_distance_spread"),
                    "shell_quality_label": candidate_site.get("shell_quality_label"),
                    "shell_quality_score": candidate_site.get("shell_quality_score"),
                    "shell_first_shell_count": candidate_site.get("shell_first_shell_count"),
                    "shell_mean_distance": candidate_site.get("shell_mean_distance"),
                    "shell_distance_spread": candidate_site.get("shell_distance_spread"),
                    "family_label": candidate_site.get("family_label"),
                    "family_size": candidate_site.get("family_size"),
                    "clearance": candidate_site.get("clearance"),
                    "wyckoff_candidate": candidate_site.get("wyckoff"),
                    "site_hint": site_hint,
                    "site_role_hint": site_role_hint,
                    "requested_geometry_hint": requested_geometry_hint,
                    "site_mode": site_mode,
                    "layer_region_hint": layer_region_hint,
                    "wyckoff": wyckoff,
                    "host_anion_family": host_anion_family,
                    "host_framework_label": host_framework_label,
                    "surface_site_hint": surface_site_hint,
                    "recommendation_score": score,
                    "recommendation_reason": reason,
                },
            )
            bundle_overrides = {"ISYM": 0}
            artifacts.extend(
                self._write_suggested_input_bundle(
                    bundle_dir=candidate_dir,
                    generated_inputs=generated_inputs,
                    poscar_text=candidate_structure.to(fmt="poscar"),
                    incar_overrides=bundle_overrides,
                    title=f"缺陷候选输入包: {candidate_name}",
                    notes=[
                        "该候选目录已包含 POSCAR 与建议输入卡，可在人工确认后直接继续修改。",
                        "缺陷/掺杂计算常建议关闭对称性自动化，因此已在 INCAR.suggested 中加入 ISYM = 0。",
                        f"候选说明: {reason}",
                    ],
                )
            )
            artifacts.extend([str(poscar_path), str(note_path), str(recipe_path)])
            candidate_summaries.append(
                {
                    "candidate": candidate_name,
                    "host_species": species,
                    "site_index": first_index,
                    "equivalent_site_indices": candidate_site["equivalent_site_indices"],
                    "multiplicity": candidate_site["multiplicity"],
                    "frac_coords": candidate_site["frac_coords"],
                    "coordination_hint": candidate_site["coordination_hint"],
                    "site_role": candidate_site["site_role"],
                    "geometry_hint": candidate_site["geometry_hint"],
                    "layer_region": candidate_site.get("layer_region"),
                    "layer_axis": candidate_site.get("layer_axis"),
                    "layer_axis_score": candidate_site.get("layer_axis_score"),
                    "layer_axis_max_gap": candidate_site.get("layer_axis_max_gap"),
                    "layer_gap_size": candidate_site.get("layer_gap_size"),
                    "layer_gap_rank": candidate_site.get("layer_gap_rank"),
                    "layer_gap_count": candidate_site.get("layer_gap_count"),
                    "placement_source": candidate_site.get("placement_source"),
                    "layer_midpoint_offset": candidate_site.get("layer_midpoint_offset"),
                    "neighbor_fingerprint": candidate_site.get("neighbor_fingerprint"),
                    "environment_label": candidate_site.get("environment_label"),
                    "environment_quality_score": candidate_site.get("environment_quality_score"),
                    "environment_role_counts": candidate_site.get("environment_role_counts"),
                    "environment_family_counts": candidate_site.get("environment_family_counts"),
                    "environment_dominant_anion_family": candidate_site.get("environment_dominant_anion_family"),
                    "environment_host_preference": candidate_site.get("environment_host_preference"),
                    "environment_anion_fraction": candidate_site.get("environment_anion_fraction"),
                    "environment_distance_spread": candidate_site.get("environment_distance_spread"),
                    "shell_quality_label": candidate_site.get("shell_quality_label"),
                    "shell_quality_score": candidate_site.get("shell_quality_score"),
                    "shell_first_shell_count": candidate_site.get("shell_first_shell_count"),
                    "shell_mean_distance": candidate_site.get("shell_mean_distance"),
                    "shell_distance_spread": candidate_site.get("shell_distance_spread"),
                    "family_label": candidate_site.get("family_label"),
                    "family_size": candidate_site.get("family_size"),
                    "clearance": candidate_site.get("clearance"),
                    "wyckoff_candidate": candidate_site.get("wyckoff"),
                    "host_anion_family": host_anion_family,
                    "host_framework_label": host_framework_label,
                    "score": score,
                    "reason": reason,
                }
            )

        if candidate_summaries:
            candidate_summaries.sort(
                key=self._candidate_summary_sort_key,
                reverse=True,
            )
            ranking_path = candidates_dir / "recommended_candidates.json"
            self._write_json(
                ranking_path,
                {
                    "mode": mode,
                    "dopant": dopant,
                    "site_hint": site_hint,
                    "site_role_hint": site_role_hint,
                    "geometry_hint": requested_geometry_hint,
                    "site_mode": site_mode,
                    "layer_region_hint": layer_region_hint,
                    "wyckoff": wyckoff,
                    "surface_site_hint": surface_site_hint,
                    "host_anion_family": host_anion_family,
                    "host_framework_label": host_framework_label,
                    "preferred_species_from_prompt": normalized_site_hint,
                    "site_hint_present_in_structure": bool(
                        normalized_site_hint and normalized_site_hint in unique_species
                    ),
                    "available_species": unique_species,
                    "candidates": candidate_summaries,
                },
            )
            artifacts.append(str(ranking_path))
            overview_path = candidates_dir / "candidate_overview.md"
            self._write_text(
                overview_path,
                self._build_candidate_overview(
                    candidate_summaries=candidate_summaries,
                    dopant=dopant,
                    site_mode=site_mode,
                    layer_region_hint=layer_region_hint,
                    requested_wyckoff=wyckoff,
                    dopant_environment_preference=self._preferred_host_environment_for_dopant(dopant),
                    host_anion_family=host_anion_family,
                    host_framework_label=host_framework_label,
                ),
            )
            artifacts.append(str(overview_path))
            comparison_path = candidates_dir / "top_candidate_comparison.md"
            self._write_text(
                comparison_path,
                self._build_top_candidate_comparison(
                    candidate_summaries=candidate_summaries,
                    host_anion_family=host_anion_family,
                    host_framework_label=host_framework_label,
                ),
            )
            artifacts.append(str(comparison_path))
            best_bundle_dir = candidates_dir / "best_guess_bundle"
            artifacts.extend(
                self._materialize_best_candidate_bundle(
                    candidates_dir=candidates_dir,
                    best_bundle_dir=best_bundle_dir,
                    best_candidate=candidate_summaries[0],
                    dopant_environment_preference=self._preferred_host_environment_for_dopant(dopant),
                    host_anion_family=host_anion_family,
                    host_framework_label=host_framework_label,
                )
            )
            shortlist = self._select_parallel_shortlist(candidate_summaries)
            artifacts.extend(
                self._materialize_parallel_shortlist(
                    candidates_dir=candidates_dir,
                    shortlist_dir=candidates_dir / "parallel_shortlist",
                    shortlist=shortlist,
                )
            )
        return artifacts

    def _materialize_best_candidate_bundle(
        self,
        *,
        candidates_dir: Path,
        best_bundle_dir: Path,
        best_candidate: dict[str, Any],
        dopant_environment_preference: str | None = None,
        host_anion_family: str | None = None,
        host_framework_label: str | None = None,
    ) -> list[str]:
        candidate_name = str(best_candidate["candidate"])
        source_dir = candidates_dir / candidate_name
        if not source_dir.exists():
            return []
        if best_bundle_dir.exists():
            shutil.rmtree(best_bundle_dir)
        shutil.copytree(source_dir, best_bundle_dir)
        summary_path = best_bundle_dir / "BEST_GUESS.txt"
        self._write_text(
            summary_path,
            (
                f"当前推荐的最佳候选: {candidate_name}\n"
                f"推荐分数: {best_candidate.get('score')}\n"
                + (
                    f"区域判断: {best_candidate['layer_region']}\n"
                    if best_candidate.get("layer_region") is not None
                    else ""
                )
                + (
                    f"分层轴判断: {best_candidate['layer_axis']}\n"
                    if best_candidate.get("layer_axis") is not None
                    else ""
                )
                + (
                    f"掺杂元素环境偏好: {dopant_environment_preference}\n"
                    if dopant_environment_preference
                    else ""
                )
                + (
                    f"宿主阴离子主族: {host_anion_family}\n"
                    if host_anion_family
                    else ""
                )
                + (
                    f"宿主框架判断: {host_framework_label}\n"
                    if host_framework_label
                    else ""
                )
                + (
                    f"当前候选局部环境: {best_candidate['environment_label']}\n"
                    if best_candidate.get("environment_label") is not None
                    else ""
                )
                + (
                    f"当前候选主导阴离子家族: {best_candidate['environment_dominant_anion_family']}\n"
                    if best_candidate.get("environment_dominant_anion_family") is not None
                    else ""
                )
                + (
                    f"当前候选配位壳层: {best_candidate['shell_quality_label']}\n"
                    if best_candidate.get("shell_quality_label") is not None
                    else ""
                )
                +
                f"原因: {best_candidate.get('reason')}\n"
                "该目录是从候选目录复制出来的快捷入口，提交前仍需人工确认。\n"
            ),
        )
        return [str(best_bundle_dir), str(summary_path)]

    @staticmethod
    def _select_parallel_shortlist(
        candidate_summaries: list[dict[str, Any]],
        *,
        max_candidates: int = 3,
        score_window: int = 5,
    ) -> list[dict[str, Any]]:
        if not candidate_summaries:
            return []
        best_score = int(candidate_summaries[0].get("score", 0))
        shortlist = [
            candidate
            for candidate in candidate_summaries
            if int(candidate.get("score", 0)) >= best_score - score_window
        ]
        return shortlist[:max_candidates]

    def _materialize_parallel_shortlist(
        self,
        *,
        candidates_dir: Path,
        shortlist_dir: Path,
        shortlist: list[dict[str, Any]],
    ) -> list[str]:
        if not shortlist:
            return []
        if shortlist_dir.exists():
            shutil.rmtree(shortlist_dir)
        shortlist_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[str] = []

        summary_json_path = shortlist_dir / "shortlist.json"
        summary_md_path = shortlist_dir / "shortlist.md"
        self._write_json(
            summary_json_path,
            {
                "recommended_parallel_count": len(shortlist),
                "candidates": shortlist,
            },
        )
        artifacts.append(str(summary_json_path))

        lines = [
            "# 并行候选 shortlist",
            "",
            "以下候选分数接近，适合并行保留继续做弛豫或单点比较。",
            "",
        ]
        for index, candidate in enumerate(shortlist, start=1):
            candidate_name = str(candidate["candidate"])
            source_dir = candidates_dir / candidate_name
            target_dir = shortlist_dir / f"rank{index:02d}_{candidate_name}"
            if source_dir.exists():
                shutil.copytree(source_dir, target_dir)
                artifacts.append(str(target_dir))
            lines.extend(
                [
                    f"## Rank {index}: {candidate_name}",
                    f"- 分数: `{candidate.get('score')}`",
                    f"- 区域判断: `{candidate.get('layer_region')}`",
                    f"- 局部环境: `{candidate.get('environment_label')}`",
                    f"- 局部环境评分: `{candidate.get('environment_quality_score')}`",
                    f"- 主导阴离子家族: `{candidate.get('environment_dominant_anion_family')}`",
                    f"- 配位壳层: `{candidate.get('shell_quality_label')}`",
                    f"- 配位壳层评分: `{candidate.get('shell_quality_score')}`",
                    f"- 分层轴: `{candidate.get('layer_axis')}`",
                    f"- 层缝排名: `{candidate.get('layer_gap_rank')}/{candidate.get('layer_gap_count')}`",
                    f"- 放置来源: `{candidate.get('placement_source')}`",
                    f"- 净空(A): `{candidate.get('clearance')}`",
                    f"- 候选目录副本: `{target_dir.name}`",
                    f"- 推荐理由: {candidate.get('reason')}",
                    "",
                ]
            )
        lines.extend(
            [
                "## 使用建议",
                "- `best_guess_bundle/` 适合先看单个最优候选。",
                "- `parallel_shortlist/` 适合把分数接近的候选并行提交。", 
                "- 若后续总能量接近，建议再结合频率、磁矩和局部构型稳定性做二次筛选。",
                "",
            ]
        )
        self._write_text(summary_md_path, "\n".join(lines))
        artifacts.append(str(summary_md_path))
        return artifacts

    @staticmethod
    def _build_candidate_overview(
        *,
        candidate_summaries: list[dict[str, Any]],
        dopant: Any,
        site_mode: Any,
        layer_region_hint: Any,
        requested_wyckoff: Any,
        dopant_environment_preference: str | None = None,
        host_anion_family: str | None = None,
        host_framework_label: str | None = None,
    ) -> str:
        lines = [
            "# 候选总览",
            "",
            f"- 掺杂元素: `{dopant or '未识别'}`",
            f"- 位点模式: `{site_mode or '未指定'}`",
            f"- 区域偏好: `{layer_region_hint or '未指定'}`",
            f"- 请求 Wyckoff: `{requested_wyckoff or '未指定'}`",
            f"- 掺杂元素环境偏好: `{dopant_environment_preference or '未推断'}`",
            f"- 宿主阴离子主族: `{host_anion_family or '未识别'}`",
            f"- 宿主框架判断: `{host_framework_label or '未识别'}`",
            "",
            "| 排名 | 候选 | 分数 | 几何 | 区域 | 环境 | 主导阴离子 | 壳层 | 分层轴 | 层缝排名 | 家族规模 | 净空(A) | 说明 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for index, candidate in enumerate(candidate_summaries, start=1):
            gap_rank_text = ""
            if candidate.get("layer_gap_rank") is not None and candidate.get("layer_gap_count") is not None:
                gap_rank_text = (
                    f"{candidate.get('layer_gap_rank')}/{candidate.get('layer_gap_count')}"
                )
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        str(candidate.get("candidate", "")),
                        str(candidate.get("score", "")),
                        str(candidate.get("geometry_hint", "")),
                        str(candidate.get("layer_region", "")),
                        str(candidate.get("environment_label", "")),
                        str(candidate.get("environment_dominant_anion_family", "")),
                        str(candidate.get("shell_quality_label", "")),
                        str(candidate.get("layer_axis", "")),
                        gap_rank_text,
                        str(candidate.get("family_size", "")),
                        str(candidate.get("clearance", "")),
                        str(candidate.get("reason", "")).replace("|", "/"),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## 使用建议",
                "- 优先检查 `best_guess_bundle/`，它是当前排序最高的候选快捷入口。",
                "- 若请求里带有 Wyckoff 标签，请确认候选摘要中的说明是否支持该标签。",
                "- 若多个候选分数接近，建议并行保留前 2-3 个候选继续做弛豫。",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _build_top_candidate_comparison(
        *,
        candidate_summaries: list[dict[str, Any]],
        host_anion_family: str | None = None,
        host_framework_label: str | None = None,
    ) -> str:
        lines = [
            "# 顶部候选对比",
            "",
        ]
        if not candidate_summaries:
            lines.extend(
                [
                    "当前没有可比较的候选。",
                    "",
                ]
            )
            return "\n".join(lines)
        best = candidate_summaries[0]
        second = candidate_summaries[1] if len(candidate_summaries) > 1 else None
        lines.extend(
            [
                f"- 当前最佳候选: `{best.get('candidate')}`",
                f"- 当前最佳分数: `{best.get('score')}`",
                f"- 宿主阴离子主族: `{host_anion_family or '未识别'}`",
                f"- 宿主框架判断: `{host_framework_label or '未识别'}`",
                "",
            ]
        )
        if second is None:
            lines.extend(
                [
                    "当前只有 1 个候选，无需做顶部候选对比。",
                    "",
                ]
            )
            return "\n".join(lines)

        lines.extend(
            [
                f"- 次优候选: `{second.get('candidate')}`",
                f"- 次优分数: `{second.get('score')}`",
                f"- 分差: `{int(best.get('score', 0)) - int(second.get('score', 0))}`",
                "",
                "| 项目 | 最佳候选 | 次优候选 |",
                "| --- | --- | --- |",
                f"| 放置来源 | `{best.get('placement_source')}` | `{second.get('placement_source')}` |",
                f"| 区域判断 | `{best.get('layer_region')}` | `{second.get('layer_region')}` |",
                f"| 主导阴离子家族 | `{best.get('environment_dominant_anion_family')}` | `{second.get('environment_dominant_anion_family')}` |",
                f"| 局部环境 | `{best.get('environment_label')}` | `{second.get('environment_label')}` |",
                f"| 环境评分 | `{best.get('environment_quality_score')}` | `{second.get('environment_quality_score')}` |",
                f"| 配位壳层 | `{best.get('shell_quality_label')}` | `{second.get('shell_quality_label')}` |",
                f"| 壳层评分 | `{best.get('shell_quality_score')}` | `{second.get('shell_quality_score')}` |",
                f"| 净空(A) | `{best.get('clearance')}` | `{second.get('clearance')}` |",
                "",
                "## 取舍说明",
                f"- 最佳候选理由: {best.get('reason')}",
                f"- 次优候选理由: {second.get('reason')}",
                "",
                "## 使用建议",
                "- 如果你更重视化学环境匹配和后续弛豫稳定性，优先看最佳候选。",
                "- 如果你更重视特殊分数坐标或高对称初猜，也建议同时保留次优候选并行比较。",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _build_layer_axis_diagnostics_markdown(
        diagnostics: dict[str, Any],
    ) -> str:
        selected_axis = diagnostics.get("selected_axis")
        lines = [
            "# 分层轴诊断",
            "",
            f"- 选中的分层轴: `{selected_axis or '未识别'}`",
            "",
            "| 晶轴 | 层簇数 | 最大非回绕层缝(A) | 平均非回绕层缝(A) | 分层评分 | 是否选中 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for axis_entry in diagnostics.get("axes", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(axis_entry.get("axis_label", "")),
                        str(axis_entry.get("layer_count", "")),
                        str(axis_entry.get("max_gap", "")),
                        str(axis_entry.get("avg_gap", "")),
                        str(axis_entry.get("layering_score", "")),
                        "是" if axis_entry.get("selected") else "",
                    ]
                )
                + " |"
            )
            gaps = axis_entry.get("non_wrap_gaps") or []
            if gaps:
                lines.append("")
                lines.append(
                    f"轴 `{axis_entry.get('axis_label')}` 的非回绕层缝: "
                    + ", ".join(
                        f"#{gap.get('rank')}={gap.get('gap_size')} A"
                        for gap in gaps
                    )
                )
        lines.extend(
            [
                "",
                "## 说明",
                "- 分层评分越高，表示该轴上存在更明显的层簇与层缝分离。",
                "- 这里只统计非回绕层缝，避免周期边界造成的假大间隙干扰判断。",
                "- 该诊断用于人工确认层间/层内偏好，不直接等价于真实 Wyckoff 判定。",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _normalize_site_hint(site_hint: Any) -> str | None:
        if not isinstance(site_hint, str):
            return None
        match = re.search(r"([A-Z][a-z]?)", site_hint)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _score_defect_candidate(
        *,
        host_species: str,
        site_hint: str | None,
        multiplicity: int,
        candidate_site_role: str,
        requested_site_role: Any,
        candidate_geometry: str,
        requested_geometry: Any,
    ) -> tuple[int, str]:
        reasons: list[str] = []
        score = min(60 + max(multiplicity - 1, 0), 80)

        if site_hint and host_species == site_hint:
            score = min(100 + max(multiplicity - 1, 0), 120)
            reasons.append(
                f"与提示中的目标位点 {site_hint} 完全匹配，"
                f"该代表位点对应 {multiplicity} 个对称等价位点。"
            )
        if site_hint and host_species != site_hint:
            return 20, f"提示更像是 {site_hint} 位点，该候选仅保留作备选。"
        if not site_hint:
            reasons.append(
                f"未提供明确元素位点提示，按对称不等价位点生成候选；"
                f"该代表位点对应 {multiplicity} 个等价位点。"
            )

        if requested_site_role:
            if candidate_site_role == requested_site_role:
                score += 15
                reasons.append(f"位点角色与提示 `{requested_site_role}` 一致。")
            else:
                score -= 15
                reasons.append(
                    f"位点角色更像 `{candidate_site_role}`，与提示 `{requested_site_role}` 不一致。"
                )

        if requested_geometry:
            if candidate_geometry == requested_geometry:
                score += 10
                reasons.append(f"局部几何与提示 `{requested_geometry}` 一致。")
            elif candidate_geometry != "unknown":
                score -= 10
                reasons.append(
                    f"局部几何更像 `{candidate_geometry}`，与提示 `{requested_geometry}` 不一致。"
                )

        if not reasons:
            reasons.append("按对称不等价位点生成候选。")
        return score, " ".join(reasons)

    @staticmethod
    def _score_interstitial_candidate(
        *,
        dopant: Any,
        host_anion_family: Any,
        multiplicity: int,
        candidate_site_role: str,
        requested_site_role: Any,
        candidate_geometry: str,
        requested_geometry: Any,
        candidate_layer_region: str,
        requested_layer_region: Any,
        candidate_gap_rank: Any,
        candidate_gap_count: Any,
        placement_source: Any,
        environment_label: Any,
        environment_quality_score: Any,
        environment_preference_match: Any,
        dominant_anion_family: Any,
        dopant_anion_family_preference: Any,
        dopant_anion_family_match: Any,
        shell_quality_label: Any,
        shell_quality_score: Any,
        clearance: float,
        wyckoff: Any,
        requested_wyckoff: Any,
        family_size: int,
    ) -> tuple[int, str]:
        score = min(50 + max(multiplicity - 1, 0), 70)
        reasons = [
            f"按粗网格采样与局部净空筛选生成间隙位候选；该代表位点对应 {multiplicity} 个等价点。"
        ]
        if family_size > 1:
            reasons.append(f"同类候选家族共识别到 {family_size} 个粗候选点。")
        if placement_source == "largest_gap_midpoint":
            reasons.append("该候选来自最大非回绕层缝中点的定向补充采样。")

        if clearance > 0:
            bonus = min(int(clearance * 10), 20)
            score += bonus
            reasons.append(f"局部净空约 {clearance:.2f} A，适合作为初始插入候选。")

        if environment_quality_score is not None:
            environment_bonus = int(environment_quality_score)
            score += environment_bonus
            if environment_bonus > 0:
                reasons.append(
                    f"局部近邻环境更像 `{environment_label}`，为该间隙位提供额外支持。"
                )
            elif environment_bonus < 0:
                reasons.append(
                    f"局部近邻环境更像 `{environment_label}`，说明该位置仍需谨慎确认。"
                )
        if environment_preference_match is True:
            score += 4
            reasons.append("该候选的近邻角色分布与掺杂元素更偏好的间隙环境一致。")
        elif environment_preference_match is False:
            score -= 4
            reasons.append("该候选的近邻角色分布与掺杂元素更偏好的间隙环境不完全一致。")
        if dopant_anion_family_preference and dominant_anion_family:
            if dopant_anion_family_match is True:
                score += 4
                reasons.append(
                    f"该候选更接近 `{dominant_anion_family}` 型阴离子笼位，与掺杂元素偏好 `{dopant_anion_family_preference}` 一致。"
                )
            elif dopant_anion_family_match is False:
                score -= 2
                reasons.append(
                    f"该候选当前更像 `{dominant_anion_family}` 型阴离子环境，与掺杂元素偏好 `{dopant_anion_family_preference}` 不完全一致。"
                )
        if host_anion_family and dominant_anion_family:
            if dominant_anion_family == host_anion_family:
                score += 3
                reasons.append(
                    f"该候选主导阴离子家族 `{dominant_anion_family}` 与宿主框架一致，更像体系内自然层间位环境。"
                )
            else:
                score -= 2
                reasons.append(
                    f"该候选主导阴离子家族 `{dominant_anion_family}` 与宿主框架 `{host_anion_family}` 不一致，需额外核对。"
                )
        if shell_quality_score is not None:
            shell_bonus = int(shell_quality_score)
            score += shell_bonus
            if shell_bonus > 0:
                reasons.append(f"第一配位壳层质量更像 `{shell_quality_label}`，有利于作为稳定初猜。")
            elif shell_bonus < 0:
                reasons.append(f"第一配位壳层质量更像 `{shell_quality_label}`，说明局部构型仍偏粗糙。")

        if requested_site_role:
            if candidate_site_role == requested_site_role:
                score += 10
                reasons.append(f"位点角色与提示 `{requested_site_role}` 一致。")
            else:
                score -= 10
                reasons.append(
                    f"位点角色更像 `{candidate_site_role}`，与提示 `{requested_site_role}` 不一致。"
                )

        if requested_geometry:
            if candidate_geometry == requested_geometry:
                score += 10
                reasons.append(f"局部几何与提示 `{requested_geometry}` 一致。")
            elif candidate_geometry != "unknown":
                score -= 10
                reasons.append(
                    f"局部几何更像 `{candidate_geometry}`，与提示 `{requested_geometry}` 不一致。"
                )

        if requested_layer_region:
            if candidate_layer_region == requested_layer_region:
                score += 12
                reasons.append(f"区域判断与提示 `{requested_layer_region}` 一致。")
            elif candidate_layer_region:
                score -= 12
                reasons.append(
                    f"区域判断更像 `{candidate_layer_region}`，与提示 `{requested_layer_region}` 不一致。"
                )

        if candidate_gap_rank is not None and candidate_gap_count is not None:
            reasons.append(
                f"当前候选落在选定分层轴的第 {candidate_gap_rank}/{candidate_gap_count} 大非回绕层缝中。"
            )
            if requested_layer_region == "interlayer_gap_like":
                if int(candidate_gap_rank) == 1:
                    score += 6
                    reasons.append("该候选位于最大的非回绕层缝中，更接近层间位初猜。")
                else:
                    score -= 6
                    reasons.append("该候选不在最大层缝中，因此更像层内空腔而非首选层间位。")
            elif requested_layer_region == "intralayer_cavity_like":
                if int(candidate_gap_rank) > 1:
                    score += 4
                    reasons.append("该候选不在最大层缝中，更接近层内空腔候选。")
        if requested_layer_region == "interlayer_gap_like" and placement_source == "largest_gap_midpoint":
            score += 4
            reasons.append("由于任务明确偏向层间位，定向层缝中点采样获得额外优先级。")
        if (
            requested_layer_region == "interlayer_gap_like"
            and placement_source == "largest_gap_midpoint_special_fraction"
        ):
            score += 6
            reasons.append("该候选同时满足最大层缝中点与特殊分数坐标优先规则，更适合作为高对称层间位初猜。")

        if requested_wyckoff:
            requested_mult = WorkspaceBuilder._parse_wyckoff_multiplicity(requested_wyckoff)
            if requested_mult is not None and family_size == requested_mult:
                score += 5
                reasons.append(
                    f"候选家族规模与请求的 Wyckoff 重数 `{requested_wyckoff}` 近似一致。"
                )
            if wyckoff == requested_wyckoff:
                score += 5
                reasons.append(f"候选 Wyckoff 标签与提示 `{requested_wyckoff}` 一致。")
            else:
                reasons.append(
                    f"当前粗候选未能可靠映射到请求的 Wyckoff `{requested_wyckoff}`，需人工确认。"
                )

        return score, " ".join(reasons)

    def _preferred_host_environment_for_dopant(
        self,
        dopant: Any,
    ) -> str | None:
        if not isinstance(dopant, str) or not dopant:
            return None
        dopant_role = self._classify_site_role(dopant)
        if dopant_role == "cation_like":
            return "anion_rich"
        if dopant_role == "anion_like":
            return "cation_rich"
        return None

    @staticmethod
    def _classify_species_family(species: str) -> str:
        oxide = {"O"}
        chalcogen = {"S", "Se", "Te", "Po"}
        pnictogen = {"N", "P", "As", "Sb", "Bi"}
        halide = {"F", "Cl", "Br", "I", "At"}
        if species in oxide:
            return "oxide"
        if species in chalcogen:
            return "chalcogen"
        if species in pnictogen:
            return "pnictogen"
        if species in halide:
            return "halide"
        try:
            element = Element(species)
            if element.is_metal:
                return "metal"
        except Exception:
            pass
        return "other"

    def _preferred_anion_family_for_dopant(
        self,
        dopant: Any,
    ) -> str | None:
        if not isinstance(dopant, str) or not dopant:
            return None
        try:
            element = Element(dopant)
        except Exception:
            return None
        if element.symbol in {"Fe", "Co", "Ni", "Mn", "Cr", "V", "Ti"}:
            return "oxide_or_chalcogen"
        if element.symbol in {"Cu", "Ag", "Au", "Zn", "Cd", "Hg"}:
            return "chalcogen_or_halide"
        if element.symbol in {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba"}:
            return "oxide_or_halide"
        if element.symbol in {"F", "Cl", "Br", "I"}:
            return "metal_surrounded"
        if element.is_metal:
            return "oxide_or_chalcogen"
        return None

    def _infer_host_anion_family(
        self,
        structure: Structure,
    ) -> str | None:
        family_counts: dict[str, int] = {}
        for site in structure:
            species = site.specie.symbol
            if self._classify_site_role(species) != "anion_like":
                continue
            family = self._classify_species_family(species)
            family_counts[family] = family_counts.get(family, 0) + 1
        if not family_counts:
            return None
        return max(family_counts.items(), key=lambda item: (item[1], item[0]))[0]

    @staticmethod
    def _host_framework_label(host_anion_family: str | None) -> str | None:
        if host_anion_family == "oxide":
            return "oxide_framework"
        if host_anion_family == "chalcogen":
            return "chalcogen_framework"
        if host_anion_family == "halide":
            return "halide_framework"
        if host_anion_family == "pnictogen":
            return "pnictogen_framework"
        return None

    @staticmethod
    def _candidate_name_for_site(
        *,
        mode: str,
        site_mode: Any,
        candidate_site: dict[str, Any],
    ) -> str:
        if site_mode == "interstitial":
            geometry = str(candidate_site.get("geometry_hint") or "unknown")
            site_index = int(candidate_site["site_index"])
            if candidate_site.get("placement_source") == "largest_gap_midpoint_special_fraction":
                return f"interlayer_symmetry_priority_{geometry}_site{site_index}"
            if candidate_site.get("placement_source") == "largest_gap_midpoint":
                return f"interlayer_priority_{geometry}_site{site_index}"
            return f"interstitial_{geometry}_site{site_index}"
        species = str(candidate_site["host_species"])
        site_index = int(candidate_site["site_index"])
        return (
            f"replace_{species}_site{site_index}"
            if mode == "doping"
            else f"vacancy_{species}_site{site_index}"
        )

    def _enumerate_candidate_sites(
        self,
        structure: Structure,
        allowed_species: list[str],
    ) -> list[dict[str, Any]]:
        try:
            symmetrized = SpacegroupAnalyzer(structure, symprec=0.1).get_symmetrized_structure()
            groups = symmetrized.equivalent_indices
            results: list[dict[str, Any]] = []
            for indices in groups:
                representative_index = int(indices[0])
                species = structure[representative_index].specie.symbol
                if species not in allowed_species:
                    continue
                results.append(
                    {
                        "host_species": species,
                        "site_index": representative_index,
                        "equivalent_site_indices": [int(index) for index in indices],
                        "multiplicity": len(indices),
                        "frac_coords": [
                            round(float(value), 6)
                            for value in structure[representative_index].frac_coords
                        ],
                        "coordination_hint": self._coordination_hint(
                            structure,
                            representative_index,
                        ),
                        "site_role": self._classify_site_role(species),
                        "geometry_hint": self._estimate_site_geometry(
                            structure,
                            representative_index,
                        ),
                    }
                )
            if results:
                return results
        except Exception:
            pass

        results = []
        for species in allowed_species:
            first_index = next(
                index
                for index, site in enumerate(structure)
                if site.specie.symbol == species
            )
            results.append(
                {
                    "host_species": species,
                    "site_index": int(first_index),
                    "equivalent_site_indices": [int(first_index)],
                    "multiplicity": 1,
                    "frac_coords": [
                        round(float(value), 6)
                        for value in structure[first_index].frac_coords
                    ],
                    "coordination_hint": self._coordination_hint(structure, first_index),
                    "site_role": self._classify_site_role(species),
                    "geometry_hint": self._estimate_site_geometry(
                        structure,
                        first_index,
                    ),
                }
            )
        return results

    def _enumerate_interstitial_sites(
        self,
        structure: Structure,
    ) -> list[dict[str, Any]]:
        fractions = [0.125, 0.375, 0.625, 0.875]
        raw_candidates: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int]] = set()
        distance_matrix = structure.lattice.get_all_distances

        for fx in fractions:
            for fy in fractions:
                for fz in fractions:
                    self._append_interstitial_candidate(
                        structure=structure,
                        frac=[fx, fy, fz],
                        raw_candidates=raw_candidates,
                        seen=seen,
                        distance_matrix=distance_matrix,
                        placement_source="coarse_grid",
                    )

        for priority_point in self._interlayer_priority_fraction_points(structure, fractions):
            self._append_interstitial_candidate(
                structure=structure,
                frac=priority_point["frac"],
                raw_candidates=raw_candidates,
                seen=seen,
                distance_matrix=distance_matrix,
                placement_source=str(priority_point["placement_source"]),
            )

        raw_candidates.sort(
            key=self._interstitial_candidate_sort_key,
            reverse=True,
        )
        family_sizes: dict[str, int] = {}
        for candidate in raw_candidates:
            family_label = str(candidate.get("family_label") or "unknown")
            family_sizes[family_label] = family_sizes.get(family_label, 0) + 1
        candidates: list[dict[str, Any]] = []
        selected_families: set[str] = set()
        for candidate in raw_candidates:
            family_label = str(candidate.get("family_label") or "unknown")
            if family_label in selected_families:
                continue
            selected_families.add(family_label)
            candidate["site_index"] = len(candidates)
            candidate["family_size"] = family_sizes.get(family_label, 1)
            candidate["wyckoff"] = self._approximate_wyckoff_label(
                candidate["family_size"],
                candidate.get("geometry_hint"),
            )
            candidates.append(candidate)
            if len(candidates) >= 12:
                break
        return candidates

    @staticmethod
    def _placement_priority(placement_source: Any) -> int:
        if placement_source == "largest_gap_midpoint_special_fraction":
            return 3
        if placement_source == "largest_gap_midpoint":
            return 2
        if placement_source == "coarse_grid":
            return 1
        return 0

    def _interstitial_candidate_sort_key(
        self,
        candidate: dict[str, Any],
    ) -> tuple[float, ...]:
        gap_rank = candidate.get("layer_gap_rank")
        midpoint_offset = candidate.get("layer_midpoint_offset")
        return (
            float(candidate.get("clearance") or 0.0),
            float(candidate.get("environment_quality_score") or 0.0),
            float(candidate.get("shell_quality_score") or 0.0),
            float(self._placement_priority(candidate.get("placement_source"))),
            1.0 if gap_rank == 1 else 0.0,
            -float(midpoint_offset) if midpoint_offset is not None else -999.0,
        )

    def _candidate_summary_sort_key(
        self,
        candidate: dict[str, Any],
    ) -> tuple[float, ...]:
        gap_rank = candidate.get("layer_gap_rank")
        midpoint_offset = candidate.get("layer_midpoint_offset")
        return (
            float(candidate.get("score") or 0.0),
            float(self._placement_priority(candidate.get("placement_source"))),
            float(candidate.get("environment_quality_score") or 0.0),
            float(candidate.get("shell_quality_score") or 0.0),
            1.0 if gap_rank == 1 else 0.0,
            float(candidate.get("clearance") or 0.0),
            -float(midpoint_offset) if midpoint_offset is not None else -999.0,
        )

    def _append_interstitial_candidate(
        self,
        *,
        structure: Structure,
        frac: list[float],
        raw_candidates: list[dict[str, Any]],
        seen: set[tuple[int, int, int]],
        distance_matrix: Any,
        placement_source: str,
    ) -> None:
        normalized_frac = [float(value) % 1.0 for value in frac]
        key = tuple(int(round(value * 1000)) for value in normalized_frac)
        if key in seen:
            return
        seen.add(key)
        all_distances = distance_matrix([normalized_frac], structure.frac_coords)[0]
        clearance = float(min(all_distances))
        if clearance < 1.3:
            return

        site_role = self._classify_interstitial_role(structure, normalized_frac)
        geometry_hint = self._estimate_interstitial_geometry(structure, normalized_frac)
        coordination_hint = self._coordination_hint_for_position(structure, normalized_frac)
        layer_analysis = self._interstitial_layer_analysis(structure, normalized_frac)
        neighbor_fingerprint = self._interstitial_neighbor_fingerprint(
            structure,
            normalized_frac,
        )
        environment_analysis = self._evaluate_interstitial_environment(
            structure,
            normalized_frac,
        )
        shell_quality = self._evaluate_interstitial_shell_quality(
            structure,
            normalized_frac,
            geometry_hint,
        )
        raw_candidates.append(
            {
                "host_species": "interstitial",
                "site_index": len(raw_candidates),
                "equivalent_site_indices": [],
                "multiplicity": 1,
                "frac_coords": [round(float(value), 6) for value in normalized_frac],
                "coordination_hint": coordination_hint,
                "site_role": site_role,
                "geometry_hint": geometry_hint,
                "layer_region": layer_analysis["region"],
                "layer_axis": layer_analysis["axis_label"],
                "layer_axis_score": layer_analysis["axis_score"],
                "layer_axis_max_gap": layer_analysis["axis_max_gap"],
                "layer_gap_size": layer_analysis["gap_size"],
                "layer_gap_rank": layer_analysis["gap_rank"],
                "layer_gap_count": layer_analysis["gap_count"],
                "layer_midpoint_offset": layer_analysis["midpoint_offset"],
                "neighbor_fingerprint": neighbor_fingerprint,
                "environment_label": environment_analysis["label"],
                "environment_quality_score": environment_analysis["quality_score"],
                "environment_role_counts": environment_analysis["role_counts"],
                "environment_family_counts": environment_analysis["family_counts"],
                "environment_dominant_anion_family": environment_analysis["dominant_anion_family"],
                "environment_host_preference": environment_analysis["host_environment_preference"],
                "environment_anion_fraction": environment_analysis["anion_fraction"],
                "environment_distance_spread": environment_analysis["distance_spread"],
                "shell_quality_label": shell_quality["label"],
                "shell_quality_score": shell_quality["score"],
                "shell_first_shell_count": shell_quality["first_shell_count"],
                "shell_mean_distance": shell_quality["mean_distance"],
                "shell_distance_spread": shell_quality["distance_spread"],
                "family_label": (
                    f"{geometry_hint}_{layer_analysis['region']}_{neighbor_fingerprint}"
                ),
                "clearance": round(clearance, 6),
                "wyckoff": None,
                "placement_source": placement_source,
            }
        )

    def _interlayer_priority_fraction_points(
        self,
        structure: Structure,
        fractions: list[float],
    ) -> list[dict[str, Any]]:
        axis_summary = self._detect_layer_axis(structure)
        if axis_summary is None:
            return []
        non_wrap_gaps = sorted(
            axis_summary.get("non_wrap_gaps", []),
            key=lambda item: float(item["gap_size"]),
            reverse=True,
        )
        if not non_wrap_gaps:
            return []
        top_gap = non_wrap_gaps[0]
        gap_index = int(top_gap["index"])
        layers = axis_summary.get("layers", [])
        if gap_index >= len(layers):
            return []
        next_layer = layers[gap_index + 1]
        current_layer = layers[gap_index]
        lower = float(current_layer["center"])
        upper = float(next_layer["center"])
        midpoint = (lower + upper) / 2.0
        axis = int(axis_summary["axis"])

        points: list[dict[str, Any]] = []
        special_fractions = [0.0, 1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 5.0 / 6.0]
        for first in special_fractions:
            for second in special_fractions:
                frac = [0.5, 0.5, 0.5]
                remaining_axes = [
                    candidate_axis for candidate_axis in range(3) if candidate_axis != axis
                ]
                frac[axis] = midpoint % 1.0
                frac[remaining_axes[0]] = first
                frac[remaining_axes[1]] = second
                points.append(
                    {
                        "frac": frac,
                        "placement_source": "largest_gap_midpoint_special_fraction",
                    }
                )
        for first in fractions:
            for second in fractions:
                frac = [0.5, 0.5, 0.5]
                remaining_axes = [
                    candidate_axis for candidate_axis in range(3) if candidate_axis != axis
                ]
                frac[axis] = midpoint % 1.0
                frac[remaining_axes[0]] = first
                frac[remaining_axes[1]] = second
                points.append(
                    {
                        "frac": frac,
                        "placement_source": "largest_gap_midpoint",
                    }
                )
        return points

    @staticmethod
    def _parse_wyckoff_multiplicity(wyckoff: Any) -> int | None:
        if not isinstance(wyckoff, str):
            return None
        match = re.match(r"(\d+)", wyckoff.strip())
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _approximate_wyckoff_label(
        family_size: int,
        geometry_hint: Any,
    ) -> str | None:
        if family_size <= 0:
            return None
        suffix = "i"
        if geometry_hint == "octahedral_like":
            suffix = "o"
        elif geometry_hint == "tetrahedral_like":
            suffix = "t"
        return f"{family_size}{suffix}"

    @staticmethod
    def _coordination_hint(
        structure: Structure,
        site_index: int,
    ) -> str:
        site = structure[site_index]
        neighbors = structure.get_neighbors(site, 3.0)
        if not neighbors:
            return "3.0 A 内未找到近邻。"

        species_counts: dict[str, int] = {}
        for neighbor in neighbors:
            symbol = neighbor.specie.symbol
            species_counts[symbol] = species_counts.get(symbol, 0) + 1
        summary = ", ".join(
            f"{species}:{count}" for species, count in sorted(species_counts.items())
        )
        return f"3.0 A 内近邻统计: {summary}"

    def _coordination_hint_for_position(
        self,
        structure: Structure,
        frac_coords: list[float],
    ) -> str:
        cart_coords = structure.lattice.get_cartesian_coords(frac_coords)
        neighbors = structure.get_sites_in_sphere(cart_coords, 3.0, include_index=True)
        if not neighbors:
            return "3.0 A 内未找到近邻。"
        species_counts: dict[str, int] = {}
        for neighbor in neighbors:
            species = neighbor.specie.symbol
            species_counts[species] = species_counts.get(species, 0) + 1
        summary = ", ".join(
            f"{species}:{count}" for species, count in sorted(species_counts.items())
        )
        return f"3.0 A 内近邻统计: {summary}"

    def _interstitial_neighbor_fingerprint(
        self,
        structure: Structure,
        frac_coords: list[float],
    ) -> str:
        cart_coords = structure.lattice.get_cartesian_coords(frac_coords)
        neighbors = sorted(
            structure.get_sites_in_sphere(cart_coords, 3.0, include_index=True),
            key=lambda item: float(item.nn_distance),
        )[:6]
        if not neighbors:
            return "empty"
        species_counts: dict[str, int] = {}
        for neighbor in neighbors:
            symbol = neighbor.specie.symbol
            species_counts[symbol] = species_counts.get(symbol, 0) + 1
        parts = [f"{species}{count}" for species, count in sorted(species_counts.items())]
        return "_".join(parts)

    @staticmethod
    def _classify_site_role(species: str) -> str:
        try:
            element = Element(species)
            if element.is_metal:
                return "cation_like"
            return "anion_like"
        except Exception:
            return "unknown"

    @staticmethod
    def _estimate_site_geometry(
        structure: Structure,
        site_index: int,
    ) -> str:
        coordination_number = len(structure.get_neighbors(structure[site_index], 3.0))
        if coordination_number == 6:
            return "octahedral_like"
        if coordination_number == 4:
            return "tetrahedral_like"
        if coordination_number in {5, 7}:
            return "distorted_octahedral_like"
        return "unknown"

    def _estimate_interstitial_geometry(
        self,
        structure: Structure,
        frac_coords: list[float],
    ) -> str:
        cart_coords = structure.lattice.get_cartesian_coords(frac_coords)
        neighbors = sorted(
            structure.get_sites_in_sphere(cart_coords, 3.0, include_index=True),
            key=lambda item: float(item.nn_distance),
        )
        coordination_number = len(neighbors[:6])
        if coordination_number >= 6 and float(neighbors[5].nn_distance) <= 2.8:
            return "octahedral_like"
        if coordination_number >= 4 and float(neighbors[3].nn_distance) <= 2.6:
            return "tetrahedral_like"
        return "unknown"

    def _interstitial_layer_analysis(
        self,
        structure: Structure,
        frac_coords: list[float],
    ) -> dict[str, Any]:
        axis_summary = self._detect_layer_axis(structure)
        if axis_summary is None:
            return {
                "region": "unknown_region",
                "axis_label": None,
                "axis_score": None,
                "axis_max_gap": None,
                "gap_size": None,
                "gap_rank": None,
                "gap_count": None,
                "midpoint_offset": None,
            }

        axis = int(axis_summary["axis"])
        axis_length = float(axis_summary["axis_length"])
        coordinate_value = float(frac_coords[axis]) % 1.0
        layers = axis_summary["layers"]
        if len(layers) < 2:
            return {
                "region": "unknown_region",
                "axis_label": self._axis_label(axis),
                "axis_score": round(float(axis_summary["layering_score"]), 6),
                "axis_max_gap": round(float(axis_summary["max_gap"]), 6),
                "gap_size": None,
                "gap_rank": None,
                "gap_count": None,
                "midpoint_offset": None,
            }

        best_gap_size = None
        best_midpoint_offset = None
        best_gap_rank = None
        non_wrap_gaps_sorted = sorted(
            axis_summary.get("non_wrap_gaps", []),
            key=lambda item: float(item["gap_size"]),
            reverse=True,
        )
        for index, layer in enumerate(layers):
            next_layer = layers[(index + 1) % len(layers)]
            lower = float(layer["center"])
            upper = float(next_layer["center"])
            is_wrap_gap = upper <= lower
            if is_wrap_gap:
                upper += 1.0
            candidate_value = coordinate_value
            if candidate_value <= lower:
                candidate_value += 1.0
            if candidate_value > upper:
                continue

            gap_fraction = upper - lower
            midpoint = lower + gap_fraction / 2.0
            midpoint_offset = abs(candidate_value - midpoint) * axis_length
            gap_size = gap_fraction * axis_length
            if best_gap_size is None or midpoint_offset < best_midpoint_offset:
                best_gap_size = gap_size
                best_midpoint_offset = midpoint_offset
                if not is_wrap_gap:
                    best_gap_rank = next(
                        (
                            rank
                            for rank, gap_entry in enumerate(non_wrap_gaps_sorted, start=1)
                            if int(gap_entry["index"]) == index
                        ),
                        None,
                    )

        if best_gap_size is None or best_midpoint_offset is None:
            return {
                "region": "unknown_region",
                "axis_label": self._axis_label(axis),
                "axis_score": round(float(axis_summary["layering_score"]), 6),
                "axis_max_gap": round(float(axis_summary["max_gap"]), 6),
                "gap_size": None,
                "gap_rank": None,
                "gap_count": len(non_wrap_gaps_sorted) or None,
                "midpoint_offset": None,
            }
        layered_axis = (
            float(axis_summary["layering_score"]) >= 1.15
            and float(axis_summary["max_gap"]) >= 1.1
        )
        if not layered_axis:
            region = "unknown_region"
        elif (
            best_gap_rank == 1
            and best_gap_size >= 1.8
            and best_midpoint_offset <= min(0.9, max(best_gap_size * 0.35, 0.5))
        ):
            region = "interlayer_gap_like"
        else:
            region = "intralayer_cavity_like"
        return {
            "region": region,
            "axis_label": self._axis_label(axis),
            "axis_score": round(float(axis_summary["layering_score"]), 6),
            "axis_max_gap": round(float(axis_summary["max_gap"]), 6),
            "gap_size": round(best_gap_size, 6),
            "gap_rank": best_gap_rank,
            "gap_count": len(non_wrap_gaps_sorted) or None,
            "midpoint_offset": round(best_midpoint_offset, 6),
        }

    def _detect_layer_axis(
        self,
        structure: Structure,
    ) -> dict[str, Any] | None:
        axis_summaries = self._layer_axis_summaries(structure)
        if not axis_summaries:
            return None
        return axis_summaries[0]

    def _collect_layer_axis_diagnostics(
        self,
        structure: Structure,
    ) -> dict[str, Any]:
        axis_summaries = self._layer_axis_summaries(structure)
        selected_axis = axis_summaries[0]["axis"] if axis_summaries else None
        payload_axes: list[dict[str, Any]] = []
        for summary in axis_summaries:
            payload_axes.append(
                {
                    "axis": summary["axis"],
                    "axis_label": self._axis_label(int(summary["axis"])),
                    "axis_length": round(float(summary["axis_length"]), 6),
                    "layer_count": len(summary.get("layers", [])),
                    "max_gap": round(float(summary["max_gap"]), 6),
                    "avg_gap": round(float(summary["avg_gap"]), 6),
                    "layering_score": round(float(summary["layering_score"]), 6),
                    "selected": int(summary["axis"]) == int(selected_axis)
                    if selected_axis is not None
                    else False,
                    "non_wrap_gaps": [
                        {
                            "rank": rank,
                            "index": int(gap["index"]),
                            "gap_size": round(float(gap["gap_size"]), 6),
                        }
                        for rank, gap in enumerate(
                            sorted(
                                summary.get("non_wrap_gaps", []),
                                key=lambda item: float(item["gap_size"]),
                                reverse=True,
                            ),
                            start=1,
                        )
                    ],
                }
            )
        return {
            "selected_axis": self._axis_label(int(selected_axis))
            if selected_axis is not None
            else None,
            "axes": payload_axes,
        }

    def _layer_axis_summaries(
        self,
        structure: Structure,
    ) -> list[dict[str, Any]]:
        axis_summaries: list[dict[str, Any]] = []
        for axis in range(3):
            layers = self._atomic_layer_clusters_along_axis(structure, axis)
            axis_length = float(structure.lattice.abc[axis])
            if len(layers) < 2:
                axis_summaries.append(
                    {
                        "axis": axis,
                        "layers": layers,
                        "axis_length": axis_length,
                        "max_gap": 0.0,
                        "avg_gap": 0.0,
                        "non_wrap_gaps": [],
                        "layering_score": 0.0,
                    }
                )
                continue
            gaps: list[float] = []
            non_wrap_gaps: list[dict[str, Any]] = []
            for index, layer in enumerate(layers):
                next_layer = layers[(index + 1) % len(layers)]
                lower = float(layer["center"])
                upper = float(next_layer["center"])
                is_wrap_gap = upper <= lower
                if upper <= lower:
                    upper += 1.0
                gap_size = (upper - lower) * axis_length
                gaps.append(gap_size)
                if not is_wrap_gap:
                    non_wrap_gaps.append(
                        {
                            "index": index,
                            "gap_size": gap_size,
                        }
                    )
            analysis_gaps = [float(item["gap_size"]) for item in non_wrap_gaps] or gaps
            max_gap = max(analysis_gaps)
            avg_gap = sum(analysis_gaps) / len(analysis_gaps)
            layering_score = max_gap / max(avg_gap, 1e-6)
            axis_summaries.append(
                {
                    "axis": axis,
                    "layers": layers,
                    "axis_length": axis_length,
                    "max_gap": max_gap,
                    "avg_gap": avg_gap,
                    "non_wrap_gaps": non_wrap_gaps,
                    "layering_score": layering_score,
                }
            )
        axis_summaries.sort(
            key=lambda item: (
                float(item["layering_score"]),
                float(item["max_gap"]),
                -int(item["axis"]),
            ),
            reverse=True,
        )
        return axis_summaries

    @staticmethod
    def _axis_label(axis: int) -> str:
        return ["a", "b", "c"][axis]

    @staticmethod
    def _atomic_layer_clusters_along_axis(
        structure: Structure,
        axis: int,
    ) -> list[dict[str, Any]]:
        axis_values = sorted(float(site.frac_coords[axis]) % 1.0 for site in structure)
        if not axis_values:
            return []

        axis_length = float(structure.lattice.abc[axis])
        merge_threshold = min(1.2 / max(axis_length, 1e-6), 0.08)
        layers: list[list[float]] = [[axis_values[0]]]
        for value in axis_values[1:]:
            if value - layers[-1][-1] <= merge_threshold:
                layers[-1].append(value)
            else:
                layers.append([value])

        if len(layers) > 1 and (axis_values[0] + 1.0 - axis_values[-1]) <= merge_threshold:
            merged = [value - 1.0 for value in layers[-1]] + layers[0]
            layers = [merged] + layers[1:-1]

        normalized_layers: list[dict[str, Any]] = []
        for layer in layers:
            adjusted = [value + 1.0 if value < 0 else value for value in layer]
            center = sum(adjusted) / len(adjusted)
            normalized_layers.append(
                {
                    "center": center % 1.0,
                    "count": len(layer),
                }
            )
        normalized_layers.sort(key=lambda item: float(item["center"]))
        return normalized_layers

    def _classify_interstitial_role(
        self,
        structure: Structure,
        frac_coords: list[float],
    ) -> str:
        cart_coords = structure.lattice.get_cartesian_coords(frac_coords)
        neighbors = sorted(
            structure.get_sites_in_sphere(cart_coords, 3.0, include_index=True),
            key=lambda item: float(item.nn_distance),
        )[:6]
        if not neighbors:
            return "unknown"

        role_counts: dict[str, int] = {"cation_like": 0, "anion_like": 0}
        for neighbor in neighbors:
            neighbor_role = self._classify_site_role(neighbor.specie.symbol)
            if neighbor_role in role_counts:
                role_counts[neighbor_role] += 1
        if role_counts["anion_like"] > role_counts["cation_like"]:
            return "cation_like"
        if role_counts["cation_like"] > role_counts["anion_like"]:
            return "anion_like"
        if role_counts["anion_like"] >= 4 and role_counts["cation_like"] <= 2:
            return "cation_like"
        if role_counts["cation_like"] >= 4 and role_counts["anion_like"] <= 2:
            return "anion_like"
        return "unknown"

    def _evaluate_interstitial_environment(
        self,
        structure: Structure,
        frac_coords: list[float],
    ) -> dict[str, Any]:
        cart_coords = structure.lattice.get_cartesian_coords(frac_coords)
        neighbors = sorted(
            structure.get_sites_in_sphere(cart_coords, 3.2, include_index=True),
            key=lambda item: float(item.nn_distance),
        )[:8]
        if not neighbors:
            return {
                "label": "unknown_environment",
                "quality_score": 0,
                "role_counts": {},
                "family_counts": {},
                "dominant_anion_family": None,
                "anion_fraction": None,
                "distance_spread": None,
            }

        role_counts: dict[str, int] = {"cation_like": 0, "anion_like": 0}
        family_counts: dict[str, int] = {}
        distances = [float(neighbor.nn_distance) for neighbor in neighbors[:6]]
        for neighbor in neighbors:
            neighbor_role = self._classify_site_role(neighbor.specie.symbol)
            if neighbor_role in role_counts:
                role_counts[neighbor_role] += 1
            family = self._classify_species_family(neighbor.specie.symbol)
            family_counts[family] = family_counts.get(family, 0) + 1

        known_role_total = role_counts["cation_like"] + role_counts["anion_like"]
        anion_fraction = (
            role_counts["anion_like"] / known_role_total if known_role_total else None
        )
        distance_spread = (
            statistics.pstdev(distances) if len(distances) > 1 else 0.0
        )

        label = "mixed_cage_like"
        quality_score = 1
        if anion_fraction is None:
            label = "unknown_environment"
            quality_score = 0
        elif anion_fraction >= 0.7:
            label = "anion_cage_like"
            quality_score = 8
        elif anion_fraction >= 0.55:
            label = "anion_leaning_mixed"
            quality_score = 4
        elif anion_fraction <= 0.3:
            label = "cation_crowded"
            quality_score = -8
        elif anion_fraction <= 0.45:
            label = "cation_leaning_mixed"
            quality_score = -4

        if distance_spread is not None:
            if distance_spread <= 0.18:
                quality_score += 2
            elif distance_spread >= 0.45:
                quality_score -= 2

        anion_family_counts = {
            family: count
            for family, count in family_counts.items()
            if family in {"oxide", "chalcogen", "pnictogen", "halide"}
        }
        dominant_anion_family = None
        if anion_family_counts:
            dominant_anion_family = max(
                anion_family_counts.items(),
                key=lambda item: (item[1], item[0]),
            )[0]

        return {
            "label": label,
            "quality_score": quality_score,
            "role_counts": role_counts,
            "family_counts": family_counts,
            "dominant_anion_family": dominant_anion_family,
            "host_environment_preference": (
                "anion_rich"
                if anion_fraction is not None and anion_fraction >= 0.55
                else "cation_rich"
                if anion_fraction is not None and anion_fraction <= 0.45
                else "mixed"
            ),
            "anion_fraction": round(float(anion_fraction), 6)
            if anion_fraction is not None
            else None,
            "distance_spread": round(float(distance_spread), 6)
            if distance_spread is not None
            else None,
        }

    def _evaluate_interstitial_shell_quality(
        self,
        structure: Structure,
        frac_coords: list[float],
        geometry_hint: str,
    ) -> dict[str, Any]:
        cart_coords = structure.lattice.get_cartesian_coords(frac_coords)
        neighbors = sorted(
            structure.get_sites_in_sphere(cart_coords, 3.2, include_index=True),
            key=lambda item: float(item.nn_distance),
        )
        if not neighbors:
            return {
                "label": "empty_shell",
                "score": -4,
                "first_shell_count": 0,
                "mean_distance": None,
                "distance_spread": None,
            }

        if geometry_hint == "octahedral_like":
            target_count = 6
        elif geometry_hint == "tetrahedral_like":
            target_count = 4
        else:
            target_count = min(6, len(neighbors))
        shell_neighbors = neighbors[:target_count]
        distances = [float(neighbor.nn_distance) for neighbor in shell_neighbors]
        mean_distance = sum(distances) / len(distances)
        spread = statistics.pstdev(distances) if len(distances) > 1 else 0.0
        score = 0
        label = "rough_shell"
        if len(shell_neighbors) == target_count:
            score += 2
        else:
            score -= 2
        if spread <= 0.18:
            score += 3
            label = "uniform_shell"
        elif spread <= 0.3:
            score += 1
            label = "acceptable_shell"
        elif spread >= 0.5:
            score -= 3
            label = "distorted_shell"
        if mean_distance <= 1.7 or mean_distance >= 3.0:
            score -= 2
        return {
            "label": label,
            "score": score,
            "first_shell_count": len(shell_neighbors),
            "mean_distance": round(float(mean_distance), 6),
            "distance_spread": round(float(spread), 6),
        }

    def _write_suggested_input_bundle(
        self,
        *,
        bundle_dir: Path,
        generated_inputs: dict[str, Any],
        poscar_text: str,
        incar_overrides: dict[str, Any] | None,
        title: str,
        notes: list[str],
    ) -> list[str]:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[str] = []

        poscar_path = bundle_dir / "POSCAR"
        self._write_text(poscar_path, poscar_text)
        artifacts.append(str(poscar_path))

        base_incar_path = Path(generated_inputs["incar_path"])
        if base_incar_path.exists():
            shutil.copyfile(base_incar_path, bundle_dir / "INCAR.base")
            artifacts.append(str(bundle_dir / "INCAR.base"))
            incar = Incar.from_file(base_incar_path)
            if incar_overrides:
                incar.update(incar_overrides)
            incar.write_file(bundle_dir / "INCAR.suggested")
            artifacts.append(str(bundle_dir / "INCAR.suggested"))

        for key, target_name in [
            ("kpoints_path", "KPOINTS"),
            ("potcar_map_path", "POTCAR.mapping.json"),
        ]:
            source_path = Path(generated_inputs[key])
            if source_path.exists():
                shutil.copyfile(source_path, bundle_dir / target_name)
                artifacts.append(str(bundle_dir / target_name))

        slurm_path = Path(generated_inputs["poscar_path"]).parent / "job.slurm"
        if slurm_path.exists():
            shutil.copyfile(slurm_path, bundle_dir / "job.slurm")
            artifacts.append(str(bundle_dir / "job.slurm"))

        readme_path = bundle_dir / "README.txt"
        readme_lines = [title, ""]
        readme_lines.extend(f"- {note}" for note in notes)
        readme_lines.append("- `INCAR.base` 保留原始自动生成版本。")
        readme_lines.append("- `INCAR.suggested` 合并了当前任务的建议覆盖项。")
        self._write_text(readme_path, "\n".join(readme_lines) + "\n")
        artifacts.append(str(readme_path))
        return artifacts
