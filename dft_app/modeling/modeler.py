from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from dft_app.models import ConfirmationItem, ExperimentPlan, ExperimentSpec, PlanSubtask
from dft_app.modeling.models import (
    BuildOperation,
    BuildSpec,
    CalcSpec,
    ConfirmationEntry,
    ConfirmationLevel,
    ModelSourceKind,
    ModelSpec,
    SystemSpec,
    WorkflowSpec,
    WorkflowStepSpec,
)


@dataclass
class ModelingResult:
    model_spec: ModelSpec


class TaskModeler:
    """Translate planner outputs into explicit semi-automatic modeling objects."""

    def build(
        self,
        *,
        spec: ExperimentSpec | None,
        plan: ExperimentPlan,
    ) -> ModelingResult:
        if spec is not None:
            model_spec = self._build_from_spec(spec)
        else:
            model_spec = self._build_from_plan(plan)
        return ModelingResult(model_spec=model_spec)

    def _build_from_spec(self, spec: ExperimentSpec) -> ModelSpec:
        system = SystemSpec(
            name="primary_system",
            role=self._infer_single_system_role(spec),
            summary=spec.description or spec.task_goal or "单体系标准任务",
            build=BuildSpec(
                source_type=spec.structure_source.value,
                source_ref=spec.structure_path or spec.structure_id,
                operations=self._build_operations_for_spec(spec),
                parameters=self._build_parameters_for_spec(spec),
                template_hints=self._template_hints_for_spec(spec),
                notes=self._build_notes_for_spec(spec),
            ),
            calc=self._calc_from_spec(spec),
            confirmation_items=self._confirmations_from_spec(spec),
            metadata={
                "task_type": spec.task_type.value,
                "material_name": spec.material_name,
            },
        )
        workflow = WorkflowSpec(
            workflow_type=spec.task_type.value,
            steps=[
                WorkflowStepSpec(
                    name=step,
                    goal=self._goal_for_workflow_step(spec, step),
                    system="primary_system",
                    task_type=spec.task_type.value,
                )
                for step in spec.workflow
            ],
            metadata={"source": "experiment_spec"},
        )
        return ModelSpec(
            task_id=spec.task_id,
            model_type=spec.task_type.value,
            source_kind=ModelSourceKind.SIMPLE_SPEC,
            source_prompt=spec.source_prompt,
            readiness="ready" if not spec.requires_confirmation else "needs_confirmation",
            requires_confirmation=spec.requires_confirmation,
            summary="已将简单任务映射为单体系建模对象。",
            systems=[system],
            workflow=workflow,
            confirmation_summary=list(system.confirmation_items),
            missing_information=[],
            assumptions=self._assumptions_for_spec(spec),
            metadata={
                "material_name": spec.material_name,
                "structure_source": spec.structure_source.value,
                "builder_mode": "single_system",
            },
        )

    def _build_from_plan(self, plan: ExperimentPlan) -> ModelSpec:
        if plan.experiment_type == "adsorption_energy":
            return self._build_adsorption_model(plan)
        if plan.experiment_type in {"transition_state_search", "ts_search"}:
            return self._build_ts_model(plan)
        return self._build_generic_complex_model(plan)

    def _build_adsorption_model(self, plan: ExperimentPlan) -> ModelSpec:
        slab = SystemSpec(
            name="clean_slab",
            role="slab",
            summary="洁净表面体系，用于获得 E_slab。",
            build=BuildSpec(
                source_type="surface_parent",
                operations=[BuildOperation.BUILD_SLAB.value],
                parameters={
                    "surface": None,
                    "layers": None,
                    "vacuum": None,
                    "supercell": None,
                    "fixed_bottom_layers": None,
                },
                template_hints=["surface_relax", "surface_static"],
                notes=["需要从 bulk 或已有 slab 模板派生表面模型。"],
            ),
            calc=CalcSpec(
                task_type="relax_scf",
                workflow=["relax", "scf"],
                functional=self._payload_str(plan.raw_plan, "functional"),
                submit_profile=plan.recommended_submit_profile,
            ),
            confirmation_items=[
                self._confirmation("surface", ConfirmationLevel.REQUIRED, "需要人工确认晶面取向。"),
                self._confirmation("layers", ConfirmationLevel.REQUIRED, "需要人工确认 slab 层数。"),
                self._confirmation("vacuum", ConfirmationLevel.REQUIRED, "需要人工确认真空层厚度。"),
                self._confirmation(
                    "fixed_bottom_layers",
                    ConfirmationLevel.REQUIRED,
                    "需要人工确认固定底层数。",
                ),
            ],
        )
        molecule = SystemSpec(
            name="isolated_adsorbate",
            role="molecule",
            summary="孤立分子体系，用于获得 E_molecule。",
            build=BuildSpec(
                source_type="adsorbate_source",
                operations=[BuildOperation.BUILD_ISOLATED_BOX.value],
                parameters={"box": None, "conformation": None},
                template_hints=["molecule_relax", "molecule_static"],
                notes=["需要准备吸附物分子的起始结构和盒子大小。"],
            ),
            calc=CalcSpec(
                task_type="relax_scf",
                workflow=["relax", "scf"],
                functional=self._payload_str(plan.raw_plan, "functional"),
                submit_profile=plan.recommended_submit_profile,
            ),
            confirmation_items=[
                self._confirmation(
                    "adsorbate_conformation",
                    ConfirmationLevel.REQUIRED,
                    "需要人工确认分子初始构型。",
                ),
                self._confirmation("box", ConfirmationLevel.RECOMMENDED, "建议确认孤立分子盒子大小。"),
            ],
        )
        adsorbed = SystemSpec(
            name="adsorbed_system",
            role="adsorbate_slab",
            summary="吸附体系，用于获得 E_adsorbate_slab。",
            build=BuildSpec(
                source_type="derived",
                operations=[
                    BuildOperation.DERIVE_FROM_PREVIOUS_RESULT.value,
                    BuildOperation.PLACE_ADSORBATE.value,
                    BuildOperation.ENUMERATE_ADSORPTION_CANDIDATES.value,
                ],
                parameters={
                    "parent_slab": "clean_slab",
                    "adsorbate": "isolated_adsorbate",
                    "adsorption_site": None,
                    "orientation": None,
                    "enumerate_candidates": True,
                },
                template_hints=["adsorption_relax", "adsorption_static"],
                notes=["需要把分子放置到表面并决定是否枚举多个位点/取向。"],
            ),
            calc=CalcSpec(
                task_type="relax_scf",
                workflow=["relax", "scf"],
                functional=self._payload_str(plan.raw_plan, "functional"),
                submit_profile=plan.recommended_submit_profile,
            ),
            dependencies=["clean_slab", "isolated_adsorbate"],
            confirmation_items=[
                self._confirmation("adsorption_site", ConfirmationLevel.REQUIRED, "需要人工确认吸附位点。"),
                self._confirmation("orientation", ConfirmationLevel.REQUIRED, "需要人工确认吸附初始取向。"),
                self._confirmation(
                    "enumerate_candidates",
                    ConfirmationLevel.RECOMMENDED,
                    "建议确认是否枚举多个吸附初猜构型。",
                    current_value=True,
                ),
            ],
        )
        systems = [slab, molecule, adsorbed]
        workflow = WorkflowSpec(
            workflow_type="adsorption_energy",
            steps=[
                WorkflowStepSpec(
                    name="build_clean_slab",
                    goal="构筑洁净表面并完成 relax/scf",
                    system="clean_slab",
                    task_type="relax_scf",
                ),
                WorkflowStepSpec(
                    name="build_isolated_adsorbate",
                    goal="构筑孤立分子并完成 relax/scf",
                    system="isolated_adsorbate",
                    task_type="relax_scf",
                ),
                WorkflowStepSpec(
                    name="build_adsorbed_system",
                    goal="构筑吸附体系并完成 relax/scf",
                    system="adsorbed_system",
                    depends_on=["build_clean_slab", "build_isolated_adsorbate"],
                    task_type="relax_scf",
                ),
                WorkflowStepSpec(
                    name="adsorption_energy_analysis",
                    goal="计算 E_ads = E_adsorbate_slab - E_slab - E_molecule",
                    depends_on=[
                        "build_clean_slab",
                        "build_isolated_adsorbate",
                        "build_adsorbed_system",
                    ],
                    metadata={"formula": "E_ads = E_adsorbate_slab - E_slab - E_molecule"},
                ),
            ],
            analysis_formula="E_ads = E_adsorbate_slab - E_slab - E_molecule",
            metadata={"builder_mode": "multi_system_adsorption"},
        )
        return ModelSpec(
            task_id=plan.task_id,
            model_type="adsorption_energy",
            source_kind=ModelSourceKind.COMPLEX_PLAN,
            source_prompt=plan.source_prompt,
            readiness=plan.readiness.value,
            requires_confirmation=True,
            summary="已将吸附能任务映射为三体系半自动建模对象。",
            systems=systems,
            workflow=workflow,
            confirmation_summary=self._merge_confirmations(systems, plan.missing_information),
            missing_information=plan.missing_information,
            assumptions=plan.assumptions,
            metadata={
                "experiment_type": plan.experiment_type,
                "recommended_submit_profile": plan.recommended_submit_profile,
                "current_stage": "model_defined_execution_pending",
            },
        )

    def _build_ts_model(self, plan: ExperimentPlan) -> ModelSpec:
        systems = [
            SystemSpec(
                name="initial_state",
                role="reference_state",
                summary="过渡态搜索的参考初态结构。",
                build=BuildSpec(
                    source_type="reaction_endpoint",
                    operations=[BuildOperation.DIRECT_USE.value],
                    parameters={"structure": None},
                    template_hints=["surface_relax", "molecule_relax"],
                ),
                calc=CalcSpec(
                    task_type="relax",
                    workflow=["relax"],
                    functional=self._payload_str(plan.raw_plan, "functional"),
                    submit_profile=plan.recommended_submit_profile,
                ),
                confirmation_items=[
                    self._confirmation(
                        "initial_state_structure",
                        ConfirmationLevel.REQUIRED,
                        "需要人工确认初态结构。",
                    )
                ],
            ),
            SystemSpec(
                name="transition_state_guess",
                role="transition_state_guess",
                summary="过渡态初猜结构。",
                build=BuildSpec(
                    source_type="derived",
                    operations=[BuildOperation.BUILD_TS_GUESS.value],
                    parameters={"guess_strategy": None},
                    template_hints=["vtst_dimer"],
                    notes=["模板信号提示优先采用 VTST dimer。"],
                ),
                calc=CalcSpec(
                    task_type="transition_state_search",
                    workflow=["dimer_search"],
                    functional=self._payload_str(plan.raw_plan, "functional"),
                    incar_overrides={"IBRION": 3, "ICHAIN": 2, "POTIM": 0},
                    submit_profile=plan.recommended_submit_profile,
                ),
                dependencies=["initial_state"],
                confirmation_items=[
                    self._confirmation(
                        "ts_guess_strategy",
                        ConfirmationLevel.REQUIRED,
                        "需要人工确认过渡态初猜策略。",
                    ),
                    self._confirmation(
                        "vtst_template",
                        ConfirmationLevel.RECOMMENDED,
                        "建议确认 VTST dimer 模板是否适用。",
                        current_value="vtst_dimer",
                    ),
                ],
            ),
        ]
        workflow = WorkflowSpec(
            workflow_type="transition_state_search",
            steps=[
                WorkflowStepSpec(
                    name="prepare_initial_state",
                    goal="准备或优化初态参考结构",
                    system="initial_state",
                    task_type="relax",
                ),
                WorkflowStepSpec(
                    name="search_transition_state",
                    goal="执行 VTST dimer 过渡态搜索",
                    system="transition_state_guess",
                    depends_on=["prepare_initial_state"],
                    task_type="transition_state_search",
                ),
                WorkflowStepSpec(
                    name="frequency_check",
                    goal="对过渡态候选结构进行频率验证",
                    depends_on=["search_transition_state"],
                ),
            ],
            metadata={"builder_mode": "ts_scaffold_only"},
        )
        return ModelSpec(
            task_id=plan.task_id,
            model_type="transition_state_search",
            source_kind=ModelSourceKind.COMPLEX_PLAN,
            source_prompt=plan.source_prompt,
            readiness=plan.readiness.value,
            requires_confirmation=True,
            summary="已将过渡态任务映射为参考态 + TS 初猜的半自动建模对象。",
            systems=systems,
            workflow=workflow,
            confirmation_summary=self._merge_confirmations(systems, plan.missing_information),
            missing_information=plan.missing_information,
            assumptions=plan.assumptions,
            metadata={
                "experiment_type": plan.experiment_type,
                "recommended_submit_profile": plan.recommended_submit_profile,
                "current_stage": "model_defined_execution_pending",
            },
        )

    def _build_generic_complex_model(self, plan: ExperimentPlan) -> ModelSpec:
        systems = [
            self._system_from_subtask(subtask, plan.recommended_submit_profile)
            for subtask in (plan.subtasks or [])
        ]
        if not systems:
            systems = [
                SystemSpec(
                    name="primary_system",
                    role="unknown",
                    summary="复杂任务尚未细化为正式体系。",
                    build=BuildSpec(
                        source_type="manual_build",
                        operations=[BuildOperation.DIRECT_USE.value],
                    ),
                    calc=CalcSpec(submit_profile=plan.recommended_submit_profile),
                    confirmation_items=[
                        self._confirmation(
                            "system_definition",
                            ConfirmationLevel.REQUIRED,
                            "需要先明确复杂任务包含哪些体系。",
                        )
                    ],
                )
            ]
        workflow = WorkflowSpec(
            workflow_type=plan.experiment_type,
            steps=[
                WorkflowStepSpec(
                    name=subtask.name,
                    goal=subtask.goal,
                    system=subtask.name,
                    task_type=subtask.task_type,
                )
                for subtask in (plan.subtasks or [])
            ],
            metadata={"builder_mode": "generic_complex_scaffold"},
        )
        return ModelSpec(
            task_id=plan.task_id,
            model_type=plan.experiment_type,
            source_kind=ModelSourceKind.COMPLEX_PLAN,
            source_prompt=plan.source_prompt,
            readiness=plan.readiness.value,
            requires_confirmation=True,
            summary="已为复杂任务生成通用建模骨架，等待进一步细化。",
            systems=systems,
            workflow=workflow,
            confirmation_summary=self._merge_confirmations(systems, plan.missing_information),
            missing_information=plan.missing_information,
            assumptions=plan.assumptions,
            metadata={
                "experiment_type": plan.experiment_type,
                "recommended_submit_profile": plan.recommended_submit_profile,
                "current_stage": "generic_model_scaffold",
            },
        )

    def _system_from_subtask(
        self, subtask: PlanSubtask, submit_profile: str | None
    ) -> SystemSpec:
        return SystemSpec(
            name=subtask.name,
            role=subtask.system_role,
            summary=subtask.goal,
            build=BuildSpec(
                source_type="manual_build",
                operations=[BuildOperation.DIRECT_USE.value],
                notes=["该复杂子任务尚未映射到专用构筑器。"],
            ),
            calc=CalcSpec(
                task_type=subtask.task_type,
                workflow=[subtask.task_type] if subtask.task_type else [],
                submit_profile=submit_profile,
            ),
            confirmation_items=[
                self._confirmation(
                    f"{subtask.name}_model",
                    ConfirmationLevel.REQUIRED,
                    "需要人工确认该子任务的体系定义和构筑方式。",
                )
            ],
            metadata={"goal": subtask.goal},
        )

    def _infer_single_system_role(self, spec: ExperimentSpec) -> str:
        task_type = spec.task_type.value
        if self._is_surface_like_spec(spec):
            return "slab"
        if task_type == "defect_doping":
            return "defect_supercell"
        if task_type == "spin_related":
            return "magnetic_system"
        return "primary_system"

    def _build_operations_for_spec(self, spec: ExperimentSpec) -> list[str]:
        task_type = spec.task_type.value
        if self._is_surface_like_spec(spec) and spec.structure_constraints.surface:
            return [BuildOperation.BUILD_SLAB.value]
        if task_type == "defect_doping":
            return [BuildOperation.BUILD_DEFECT_SUPERCELL.value]
        if task_type == "spin_related":
            return [
                BuildOperation.DIRECT_USE.value,
                BuildOperation.SET_SPIN_CONFIGURATION.value,
            ]
        return [BuildOperation.DIRECT_USE.value]

    def _build_parameters_for_spec(self, spec: ExperimentSpec) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        task_type = spec.task_type.value
        if spec.structure_constraints.phase:
            parameters["phase"] = spec.structure_constraints.phase
        if spec.structure_constraints.space_group:
            parameters["space_group"] = spec.structure_constraints.space_group
        if spec.structure_constraints.supercell:
            parameters["supercell"] = spec.structure_constraints.supercell
        if spec.structure_constraints.surface:
            parameters["surface"] = spec.structure_constraints.surface
        if spec.structure_constraints.defect:
            parameters["defect"] = spec.structure_constraints.defect
        if task_type in {"dos", "pdos", "band_structure"}:
            parameters["requires_reference_charge_density"] = True
        if task_type == "dos":
            parameters["analysis_target"] = "total_density_of_states"
        if task_type == "pdos":
            parameters["analysis_target"] = "projected_density_of_states"
        if task_type == "charge_analysis":
            parameters["analysis_target"] = "charge_density_and_bader_partition"
        if task_type == "work_function":
            parameters["analysis_target"] = "vacuum_level_minus_fermi_level"
            parameters["requires_vacuum_plateau"] = True
        if task_type == "vibrational_frequency":
            parameters["analysis_target"] = "normal_modes_and_imaginary_frequencies"
        if task_type == "molecular_dynamics":
            parameters["analysis_target"] = "trajectory_sampling"
            parameters["default_ensemble"] = "nvt_like"
        if task_type == "spin_related":
            parameters["spin_mode"] = {
                "is_spin_polarized": spec.spin_settings.is_spin_polarized,
                "is_soc": spec.spin_settings.is_soc,
            }
        if task_type == "defect_doping":
            parameters["requires_pristine_reference"] = True
        return parameters

    def _template_hints_for_spec(self, spec: ExperimentSpec) -> list[str]:
        task_type = spec.task_type.value
        if task_type == "work_function":
            return ["surface_static", "work_function_surface"]
        if task_type == "charge_analysis":
            return ["charge_density", "bader_analysis"]
        if task_type == "dos":
            return ["dos_static"]
        if task_type == "pdos":
            return ["pdos_static"]
        if task_type == "vibrational_frequency":
            return ["frequency_analysis"]
        if task_type == "molecular_dynamics":
            return ["md_nvt"]
        if task_type == "spin_related":
            return ["magnetic_relax"]
        if task_type == "defect_doping":
            return ["defect_supercell"]
        if self._is_surface_like_spec(spec):
            return ["surface_relax"]
        if "band" in spec.workflow:
            return ["band_structure"]
        return ["bulk_relax"]

    def _build_notes_for_spec(self, spec: ExperimentSpec) -> list[str]:
        task_type = spec.task_type.value
        notes = ["builder 将基于该建模对象生成 POSCAR / INCAR / KPOINTS / job.slurm。"]
        if spec.structure_source.value == "local_file":
            notes.append("优先复用结构目录旁已有的 INCAR / KPOINTS 模板。")
        if task_type in {"dos", "pdos", "band_structure"}:
            notes.append("建议基于已收敛的 SCF 电荷密度执行后处理型电子结构计算。")
        if task_type == "charge_analysis":
            notes.append("建议保留 CHGCAR / AECCAR* 等文件，便于后续 Bader 或差分电荷分析。")
        if task_type == "work_function":
            notes.append("表面功函数任务通常需要足够真空层，并建议检查是否需要偶极修正。")
        if task_type == "vibrational_frequency":
            notes.append("频率任务前通常需要较充分的结构优化，以减少伪虚频。")
        if task_type == "molecular_dynamics":
            notes.append("MD 任务需在建模阶段明确温度、时间步长、总步数和采样策略。")
        if task_type == "spin_related":
            notes.append("自旋相关任务建议显式确认初始磁矩、磁构型及是否启用 SOC。")
        if task_type == "defect_doping":
            notes.append("缺陷与掺杂任务通常需要同时准备本征超胞作为能量参考。")
            defect = spec.structure_constraints.defect or {}
            if defect.get("site_mode") == "interstitial":
                notes.append("当前任务包含间隙位构筑，自动生成的候选仅作为初猜，仍需人工确认真实插入位点。")
            if defect.get("layer_region_hint"):
                notes.append("当前任务带有层间/层内区域偏好，建议优先检查候选是否满足该几何环境。")
            if defect.get("wyckoff"):
                notes.append("当前任务带有 Wyckoff 位点提示，需人工确认候选结构与该标签的真实对应关系。")
        return notes

    def _calc_from_spec(self, spec: ExperimentSpec) -> CalcSpec:
        return CalcSpec(
            code=spec.code,
            task_type=spec.task_type.value,
            workflow=spec.workflow.copy(),
            functional=spec.functional,
            incar_overrides=spec.incar_overrides.copy(),
            kpoints={
                "mode": spec.kpoints_strategy.mode,
                "value": spec.kpoints_strategy.value,
            },
            encut={
                "mode": spec.encut_strategy.mode,
                "value": spec.encut_strategy.value,
            },
            smearing={
                "ismear": spec.smearing.ismear,
                "sigma": spec.smearing.sigma,
            },
            spin={
                "is_spin_polarized": spec.spin_settings.is_spin_polarized,
                "is_soc": spec.spin_settings.is_soc,
            },
            convergence={
                "ediff": spec.convergence_settings.ediff,
                "ediffg": spec.convergence_settings.ediffg,
                "nsw": spec.convergence_settings.nsw,
            },
            submit_profile=spec.submit_profile,
            job={
                "partition": spec.job_overrides.partition,
                "nodes": spec.job_overrides.nodes,
                "ntasks": spec.job_overrides.ntasks,
                "ntasks_per_node": spec.job_overrides.ntasks_per_node,
                "cpus_per_task": spec.job_overrides.cpus_per_task,
                "walltime": spec.job_overrides.walltime,
                "memory": spec.job_overrides.memory,
                "memory_per_cpu": spec.job_overrides.memory_per_cpu,
                "vasp_variant": spec.job_overrides.vasp_variant,
            },
        )

    def _confirmations_from_spec(self, spec: ExperimentSpec) -> list[ConfirmationEntry]:
        confirmation_map = {
            ConfirmationItem.STRUCTURE: self._confirmation(
                "structure",
                ConfirmationLevel.REQUIRED,
                "需要确认结构来源、原子排序和 Selective Dynamics。",
                current_value=spec.structure_path or spec.structure_id,
            ),
            ConfirmationItem.PARAMETERS: self._confirmation(
                "parameters",
                ConfirmationLevel.RECOMMENDED,
                "建议确认 functional、KPOINTS、ENCUT 和 INCAR 覆盖项。",
                current_value={"functional": spec.functional, "workflow": spec.workflow},
            ),
            ConfirmationItem.SUBMISSION: self._confirmation(
                "submission",
                ConfirmationLevel.RECOMMENDED,
                "建议确认提交队列、资源配置和 job.slurm。",
                current_value=spec.submit_profile,
            ),
        }
        items = [
            confirmation_map[item]
            for item in spec.confirmation_items
            if item in confirmation_map
        ]
        if self._is_surface_like_spec(spec):
            items.append(
                self._confirmation(
                    "surface_model",
                    ConfirmationLevel.REQUIRED,
                    "当前任务包含表面约束，需确认晶面、层数和真空层。",
                    current_value=spec.structure_constraints.surface,
                )
            )
        items.extend(self._task_specific_confirmations(spec))
        return items

    def _assumptions_for_spec(self, spec: ExperimentSpec) -> list[str]:
        task_type = spec.task_type.value
        assumptions = ["当前任务按单体系标准工作流处理。"]
        if spec.structure_source.value == "local_file":
            assumptions.append("默认以本地结构文件作为建模起点。")
        if task_type in {"dos", "pdos", "band_structure"}:
            assumptions.append("默认该任务将基于可复用的 SCF 电荷密度继续执行电子结构后处理。")
        if task_type == "work_function":
            assumptions.append("默认输入结构已具备表面模型和足够真空层。")
        if task_type == "charge_analysis":
            assumptions.append("默认需要保留电荷密度相关输出文件，供后续分析使用。")
        if task_type == "vibrational_frequency":
            assumptions.append("默认当前结构已经接近稳定构型，可直接进行有限位移频率分析。")
        if task_type == "molecular_dynamics":
            assumptions.append("默认后续会继续补充热浴、采样时长和轨迹分析设置。")
        if task_type == "spin_related":
            assumptions.append("默认需要对磁构型或 SOC 开关进行人工确认。")
        if task_type == "defect_doping":
            assumptions.append("默认需要与本征超胞结果对比，后续才能进行缺陷形成能分析。")
        return assumptions

    def _task_specific_confirmations(self, spec: ExperimentSpec) -> list[ConfirmationEntry]:
        task_type = spec.task_type.value
        items: list[ConfirmationEntry] = []

        if task_type in {"dos", "pdos"}:
            items.append(
                self._confirmation(
                    "dos_kpoints",
                    ConfirmationLevel.RECOMMENDED,
                    "建议确认 DOS/PDOS 使用更密的 KPOINTS 网格和合适的 NEDOS 设置。",
                )
            )
        if task_type == "band_structure":
            items.append(
                self._confirmation(
                    "band_path",
                    ConfirmationLevel.REQUIRED,
                    "需要确认高对称点路径和非自洽能带计算路径设置。",
                )
            )
        if task_type == "charge_analysis":
            items.append(
                self._confirmation(
                    "charge_outputs",
                    ConfirmationLevel.RECOMMENDED,
                    "建议确认是否保留 CHGCAR、AECCAR0、AECCAR2 等电荷分析所需文件。",
                )
            )
        if task_type == "work_function":
            items.append(
                self._confirmation(
                    "vacuum_and_dipole",
                    ConfirmationLevel.REQUIRED,
                    "需要确认表面模型的真空层是否足够，以及是否启用偶极修正。",
                )
            )
        if task_type == "vibrational_frequency":
            items.append(
                self._confirmation(
                    "frequency_scope",
                    ConfirmationLevel.REQUIRED,
                    "需要确认频率计算对象、位移策略，以及是否只对部分原子做振动分析。",
                )
            )
        if task_type == "molecular_dynamics":
            items.append(
                self._confirmation(
                    "md_protocol",
                    ConfirmationLevel.REQUIRED,
                    "需要确认温度、时间步长、总步数和系综控制方式。",
                )
            )
        if task_type == "spin_related":
            items.append(
                self._confirmation(
                    "spin_configuration",
                    ConfirmationLevel.REQUIRED,
                    "需要确认初始磁矩、磁有序类型，以及是否启用 SOC / 非共线设置。",
                )
            )
        if task_type == "defect_doping":
            defect = spec.structure_constraints.defect or {}
            items.append(
                self._confirmation(
                    "defect_model",
                    ConfirmationLevel.REQUIRED,
                    "需要确认缺陷类型、掺杂位点、超胞尺寸以及是否计算本征参考体系。",
                    current_value=defect,
                )
            )
            if defect.get("site_mode") == "interstitial":
                items.append(
                    self._confirmation(
                        "interstitial_site",
                        ConfirmationLevel.REQUIRED,
                        "需要人工确认间隙位候选、初始插入坐标以及是否采用多个候选并行比较。",
                        current_value={
                            "site_mode": defect.get("site_mode"),
                            "wyckoff": defect.get("wyckoff"),
                            "geometry_hint": defect.get("geometry_hint"),
                            "layer_region_hint": defect.get("layer_region_hint"),
                        },
                    )
                )
            if defect.get("layer_region_hint"):
                items.append(
                    self._confirmation(
                        "layer_region_preference",
                        ConfirmationLevel.RECOMMENDED,
                        "建议确认当前候选是否满足层间/层内区域偏好。",
                        current_value=defect.get("layer_region_hint"),
                    )
                )
            if defect.get("wyckoff"):
                items.append(
                    self._confirmation(
                        "wyckoff_mapping",
                        ConfirmationLevel.RECOMMENDED,
                        "建议确认当前结构中的候选位点是否真的对应请求的 Wyckoff 标签。",
                        current_value=defect.get("wyckoff"),
                    )
                )
        return items

    def _goal_for_workflow_step(self, spec: ExperimentSpec, step: str) -> str:
        task_type = spec.task_type.value
        goal_map = {
            ("dos", "scf"): "生成 DOS 所需的自洽电荷密度",
            ("dos", "dos"): "计算总态密度并输出 DOS 数据",
            ("pdos", "scf"): "生成 PDOS 所需的自洽电荷密度",
            ("pdos", "pdos"): "计算分波态密度并输出投影结果",
            ("band_structure", "scf"): "生成能带所需的参考电荷密度",
            ("band_structure", "band"): "沿高对称路径执行非自洽能带计算",
            ("charge_analysis", "scf"): "生成电荷分析所需的参考电荷密度",
            ("charge_analysis", "charge_analysis"): "输出电荷密度并准备 Bader 或差分电荷分析",
            ("work_function", "scf"): "计算表面体系静态电荷密度与费米能级",
            ("work_function", "work_function"): "分析平面平均静电势并提取功函数",
            ("vibrational_frequency", "relax"): "预优化结构以降低伪虚频风险",
            ("vibrational_frequency", "frequency"): "执行有限位移频率计算并分析虚频",
            ("molecular_dynamics", "molecular_dynamics"): "执行分子动力学采样并输出轨迹",
        }
        if (task_type, step) in goal_map:
            return goal_map[(task_type, step)]
        return f"执行 {step} 计算"

    def _is_surface_like_spec(self, spec: ExperimentSpec) -> bool:
        if spec.structure_constraints.surface:
            return True
        if spec.task_type.value == "work_function":
            return True
        surface_text = f"{spec.material_name} {spec.source_prompt}".lower()
        return bool(
            re.search(r"\(\s*\d+\s*\d+\s*\d+\s*\)", surface_text)
            or "surface" in surface_text
            or "表面" in surface_text
        )

    def _merge_confirmations(
        self, systems: list[SystemSpec], missing_information: list[str]
    ) -> list[ConfirmationEntry]:
        merged: list[ConfirmationEntry] = []
        seen: set[tuple[str, str]] = set()
        for system in systems:
            for item in system.confirmation_items:
                key = (item.field, item.level.value)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        for missing in missing_information:
            key = (missing, ConfirmationLevel.REQUIRED.value)
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                self._confirmation(
                    missing,
                    ConfirmationLevel.REQUIRED,
                    "planner 识别到该信息缺失，建模前需要人工补齐。",
                )
            )
        return merged

    @staticmethod
    def _confirmation(
        field: str,
        level: ConfirmationLevel,
        reason: str,
        current_value: Any = None,
    ) -> ConfirmationEntry:
        return ConfirmationEntry(
            field=field,
            level=level,
            reason=reason,
            current_value=current_value,
        )

    @staticmethod
    def _payload_str(payload: dict[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None
