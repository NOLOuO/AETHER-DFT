from __future__ import annotations

from dft_app.models import ExperimentPlan
from dft_app.workflow.base import (
    ComplexWorkflowBuilder,
    WorkflowScaffold,
    WorkflowTaskScaffold,
)


class AdsorptionEnergyWorkflow(ComplexWorkflowBuilder):
    workflow_type = "adsorption_energy"

    def build(self, plan: ExperimentPlan) -> WorkflowScaffold:
        role_aliases = {
            "slab": "slab",
            "substrate": "slab",
            "molecule": "molecule",
            "adsorbate": "molecule",
            "adsorbate_slab": "adsorbate_slab",
            "adsorption_system": "adsorbate_slab",
            "analysis": "analysis",
        }
        task_map = {
            "slab": WorkflowTaskScaffold(
                name="clean_slab",
                system_role="slab",
                goal="构建并优化洁净表面，得到 E_slab。",
                task_type="relax_scf",
                relative_dir="subtasks/01_clean_slab",
                blockers=[
                    "需要确认 Pt(111) slab 层数",
                    "需要确认表面超胞尺寸",
                    "需要确认真空层厚度",
                    "需要确认固定底层数",
                ],
                suggested_inputs={
                    "structure_kind": "surface_slab",
                    "recommended_templates": ["surface_relax", "surface_static"],
                },
            ),
            "molecule": WorkflowTaskScaffold(
                name="isolated_adsorbate",
                system_role="molecule",
                goal="构建并优化孤立分子，得到 E_molecule。",
                task_type="relax_scf",
                relative_dir="subtasks/02_isolated_adsorbate",
                blockers=[
                    "需要确认分子初始构型来源",
                    "需要确认孤立分子盒子大小",
                ],
                suggested_inputs={
                    "structure_kind": "isolated_molecule",
                    "recommended_templates": ["molecule_relax", "molecule_static"],
                },
            ),
            "adsorbate_slab": WorkflowTaskScaffold(
                name="adsorbed_system",
                system_role="adsorbate_slab",
                goal="构建吸附构型并优化，得到 E_adsorbate_slab。",
                task_type="relax_scf",
                relative_dir="subtasks/03_adsorbed_system",
                blockers=[
                    "需要确认吸附位点",
                    "需要确认吸附初始取向",
                    "需要确认是否枚举多个构型",
                ],
                suggested_inputs={
                    "structure_kind": "adsorption_complex",
                    "recommended_templates": ["adsorption_relax", "adsorption_static"],
                },
            ),
            "analysis": WorkflowTaskScaffold(
                name="adsorption_energy_analysis",
                system_role="analysis",
                goal="汇总三个体系能量并计算吸附能。",
                task_type=None,
                relative_dir="subtasks/04_analysis",
                blockers=[
                    "需要 clean_slab、isolated_adsorbate、adsorbed_system 三个子任务完成",
                ],
                suggested_inputs={
                    "formula": "E_ads = E_adsorbate_slab - E_slab - E_molecule",
                },
            ),
        }

        scaffold_tasks: list[WorkflowTaskScaffold] = []
        for subtask in plan.subtasks:
            mapped = task_map.get(role_aliases.get(subtask.system_role, subtask.system_role))
            if mapped is not None:
                scaffold_tasks.append(mapped)
            else:
                scaffold_tasks.append(
                    WorkflowTaskScaffold(
                        name=subtask.name,
                        system_role=subtask.system_role,
                        goal=subtask.goal,
                        task_type=subtask.task_type,
                        relative_dir=f"subtasks/{subtask.name}",
                        blockers=["该子任务尚未映射到正式 builder。"],
                    )
                )

        if not scaffold_tasks:
            scaffold_tasks = list(task_map.values())

        return WorkflowScaffold(
            task_id=plan.task_id,
            workflow_type=self.workflow_type,
            summary=plan.summary,
            readiness=plan.readiness.value,
            requires_confirmation=plan.requires_confirmation,
            confirmation_items=[
                "surface_model",
                "adsorbate_model",
                "adsorption_site",
                "templates",
                "submission",
            ],
            shared_assumptions=plan.assumptions,
            missing_information=plan.missing_information,
            tasks=scaffold_tasks,
            analysis_steps=[
                "检查三个子任务都已完成并收敛",
                "提取 E_slab、E_molecule、E_adsorbate_slab",
                "计算 E_ads = E_adsorbate_slab - E_slab - E_molecule",
                "输出吸附能报告",
            ],
            metadata={
                "requires_multi_system_execution": True,
                "current_stage": "scaffold_only",
            },
        )
