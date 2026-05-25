from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dft_app.cluster_profiles import SUBMIT_PROFILES, infer_submit_profile_from_prompt
from dft_app.llm import DomesticCopilotLLM
from dft_app.models import (
    ExecutionReadiness,
    ExperimentPlan,
    ExperimentSpec,
    PlanComplexity,
    PlanSubtask,
    StructureConstraint,
    StructureSource,
    TaskType,
)
from dft_app.planner.rule_based_planner import RuleBasedPlanner
from dft_app.planner.template_knowledge import PlannerTemplateKnowledge

WORKFLOW_STEP_ALIASES = {
    "structure_optimization": "relax",
    "optimization": "relax",
    "geometry_optimization": "relax",
    "static_refinement": "scf",
    "static_scf": "scf",
    "static_self_consistent_calculation": "scf",
    "self_consistent_field": "scf",
    "scf_calculation": "scf",
    "single_point": "scf",
    "single_point_scf": "scf",
    "band_structure": "band",
    "band_structure_calculation": "band",
    "non_self_consistent_band": "band",
    "dos_analysis": "dos",
    "extract_dos_data": "dos",
    "density_of_states": "dos",
    "projected_density_of_states": "pdos",
    "extract_pdos_data": "pdos",
    "bader_analysis": "charge_analysis",
    "charge_density_analysis": "charge_analysis",
    "work_function_analysis": "work_function",
    "extract_work_function": "work_function",
    "work_function_extraction": "work_function",
    "potential_analysis": "work_function",
    "frequency_analysis": "frequency",
    "phonon_analysis": "frequency",
    "md_sampling": "molecular_dynamics",
    "molecular_dynamics_simulation": "molecular_dynamics",
    "ts_search": "transition_state_search",
}

TASK_TYPE_WORKFLOW_ALLOWLIST: dict[str, set[str]] = {
    "single_point": {"single_point", "scf"},
    "geometry_optimization": {"relax"},
    "static_refinement": {"scf"},
    "dos": {"scf", "dos"},
    "pdos": {"scf", "pdos"},
    "band_structure": {"scf", "band"},
    "charge_analysis": {"scf", "charge_analysis"},
    "work_function": {"scf", "work_function"},
    "vibrational_frequency": {"relax", "frequency"},
    "transition_state_search": {"transition_state_search"},
    "molecular_dynamics": {"molecular_dynamics"},
    "spin_related": {"relax", "scf"},
    "defect_doping": {"relax", "scf"},
    "relax": {"relax"},
    "relax_scf": {"relax", "scf"},
    "relax_scf_band": {"relax", "scf", "band"},
    "encut_convergence": {"encut_convergence"},
    "kpoints_convergence": {"kpoints_convergence"},
    "eos": {"eos"},
}


@dataclass
class PlanningResult:
    plan: ExperimentPlan
    spec: ExperimentSpec | None


class LLMPlanner:
    """LLM-first planner that uses the project's built-in LLM config."""

    def __init__(
        self,
        llm: DomesticCopilotLLM | None = None,
        fallback: RuleBasedPlanner | None = None,
        template_knowledge: PlannerTemplateKnowledge | None = None,
    ):
        self.llm = llm or DomesticCopilotLLM()
        self.fallback = fallback or RuleBasedPlanner()
        self.template_knowledge = template_knowledge or PlannerTemplateKnowledge()

    def plan(
        self,
        *,
        prompt: str,
        task_id: str,
        material_name: str | None = None,
        structure_path: str | None = None,
        forced_task_type: TaskType | None = None,
        submit_profile: str | None = None,
    ) -> PlanningResult:
        if not self.llm.is_available():
            return self._fallback_result(
                prompt=prompt,
                task_id=task_id,
                material_name=material_name,
                structure_path=structure_path,
                forced_task_type=forced_task_type,
                submit_profile=submit_profile,
                message="未找到可用的大模型配置，已回退到规则解析。",
            )

        template_context = self.template_knowledge.build_context(
            prompt=prompt,
            material_name=material_name,
        )
        rule_hints = self.fallback.build_llm_hints(
            prompt=prompt,
            material_name=material_name,
            structure_path=structure_path,
            forced_task_type=forced_task_type,
            submit_profile=submit_profile,
        )
        try:
            result = self.llm.call_messages(
                self._build_messages(
                    prompt=prompt,
                    task_id=task_id,
                    material_name=material_name,
                    structure_path=structure_path,
                    forced_task_type=forced_task_type,
                    submit_profile=submit_profile,
                    template_context=template_context,
                    rule_hints=rule_hints,
                ),
                max_tokens=2200,
            )
            payload = self._parse_json_payload(str(result["content"]))
            plan = self._build_plan(
                payload,
                prompt=prompt,
                task_id=task_id,
                provider_id=str(result["provider_id"]),
                model_id=str(result["model_id"]),
                submit_profile=submit_profile,
            )
            spec = self._try_build_spec(
                plan=plan,
                payload=payload,
                prompt=prompt,
                task_id=task_id,
                material_name=material_name,
                structure_path=structure_path,
                forced_task_type=forced_task_type,
                submit_profile=submit_profile,
            )
            return PlanningResult(plan=plan, spec=spec)
        except Exception as exc:
            return self._fallback_result(
                prompt=prompt,
                task_id=task_id,
                material_name=material_name,
                structure_path=structure_path,
                forced_task_type=forced_task_type,
                submit_profile=submit_profile,
                message=f"LLM planner 调用失败，已回退到规则解析。原因: {exc}",
            )

    def explain(self, result: PlanningResult) -> dict[str, Any]:
        return {
            "plan": result.plan.to_dict(),
            "spec": result.spec.to_dict() if result.spec is not None else None,
        }

    def _fallback_result(
        self,
        *,
        prompt: str,
        task_id: str,
        material_name: str | None,
        structure_path: str | None,
        forced_task_type: TaskType | None,
        submit_profile: str | None,
        message: str,
    ) -> PlanningResult:
        plan, spec = self.fallback.build_planning_artifacts(
            prompt=prompt,
            task_id=task_id,
            material_name=material_name,
            structure_path=structure_path,
            forced_task_type=forced_task_type,
            submit_profile=submit_profile,
            fallback_message=message,
        )
        return PlanningResult(plan=plan, spec=spec)

    def _build_messages(
        self,
        *,
        prompt: str,
        task_id: str,
        material_name: str | None,
        structure_path: str | None,
        forced_task_type: TaskType | None,
        submit_profile: str | None,
        template_context: dict[str, Any],
        rule_hints: dict[str, Any],
    ) -> list[dict[str, str]]:
        system = (
            "你是一个面向 VASP 自动化的任务规划器。"
            "你的职责是把自然语言 DFT 请求转换为严格 JSON。"
            "不要输出解释，不要输出 Markdown，只输出一个 JSON 对象。"
            "优先判断这是单个标准任务还是复杂组合任务。"
            "如果是复杂任务，例如吸附能、反应能、表面构型搜索、多步机理、NEB 等，"
            "必须拆成 subtasks，并把 readiness 设为 needs_implementation 或 needs_confirmation。"
            "如果是简单单任务，只在能明确映射到标准任务时才给出 canonical_task_type。"
            "若给出 retrieved_template_context，请把它当作历史任务卡与参数模板参考。"
            "若给出 rule_hints，请把它当作启发式先验，用来减少误判，但不要机械照抄。"
            "如果模板信号显示 VTST、ICHAIN=2、IBRION=3 或路径明显是 TS，"
            "优先视为过渡态搜索，而不是普通 relax。"
        )
        user = json.dumps(
            {
                "task_id": task_id,
                "prompt": prompt,
                "material_name_hint": material_name,
                "structure_path": structure_path,
                "forced_task_type": forced_task_type.value if forced_task_type else None,
                "submit_profile_hint": submit_profile,
                "supported_single_task_types": [
                    "single_point",
                    "geometry_optimization",
                    "static_refinement",
                    "dos",
                    "pdos",
                    "band_structure",
                    "charge_analysis",
                    "work_function",
                    "vibrational_frequency",
                    "molecular_dynamics",
                    "spin_related",
                    "defect_doping",
                    "relax",
                    "relax_scf",
                    "relax_scf_band",
                    "encut_convergence",
                    "kpoints_convergence",
                    "eos",
                ],
                "required_json_schema": {
                    "experiment_type": "string",
                    "summary": "string",
                    "complexity": "simple|complex",
                    "readiness": "ready|needs_confirmation|needs_implementation",
                    "canonical_task_type": "string|null",
                    "material_name": "string|null",
                    "functional": "string|null",
                    "structure_source": "local_file|materials_project|manual_build|derived|null",
                    "structure_id": "string|null",
                    "submit_profile": "string|null",
                    "workflow": ["string"],
                    "incar_overrides": {},
                    "missing_information": ["string"],
                    "assumptions": ["string"],
                    "subtasks": [
                        {
                            "name": "string",
                            "goal": "string",
                            "system_role": "string",
                            "task_type": "string|null",
                        }
                    ],
                },
                "rule_hints": rule_hints,
                "retrieved_template_context": template_context,
            },
            ensure_ascii=False,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _parse_json_payload(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(stripped[start : end + 1])
            raise RuntimeError("LLM planner 未返回合法 JSON。")

    @staticmethod
    def _build_plan(
        payload: dict[str, Any],
        *,
        prompt: str,
        task_id: str,
        provider_id: str,
        model_id: str,
        submit_profile: str | None,
    ) -> ExperimentPlan:
        subtasks = [
            PlanSubtask(
                name=str(item.get("name", f"task_{index + 1}")),
                goal=str(item.get("goal", "")),
                system_role=str(item.get("system_role", "unknown")),
                task_type=str(item["task_type"]) if item.get("task_type") else None,
            )
            for index, item in enumerate(payload.get("subtasks") or [])
            if isinstance(item, dict)
        ]

        complexity = PlanComplexity(
            str(payload.get("complexity", "complex")).strip().lower()
        )
        readiness = ExecutionReadiness(
            str(payload.get("readiness", "needs_confirmation")).strip().lower()
        )
        normalized_experiment_type = LLMPlanner._normalize_experiment_type(
            payload=payload,
            complexity=complexity,
        )

        return ExperimentPlan(
            task_id=task_id,
            source_prompt=prompt,
            experiment_type=normalized_experiment_type,
            summary=str(payload.get("summary") or ""),
            complexity=complexity,
            readiness=readiness,
            requires_confirmation=bool(payload.get("missing_information") or complexity == PlanComplexity.COMPLEX),
            missing_information=[str(item) for item in (payload.get("missing_information") or [])],
            assumptions=[str(item) for item in (payload.get("assumptions") or [])],
            subtasks=subtasks,
            recommended_submit_profile=LLMPlanner._normalize_submit_profile(
                (
                    str(payload.get("submit_profile")).strip()
                    if payload.get("submit_profile")
                    else submit_profile or infer_submit_profile_from_prompt(prompt)
                )
            ),
            llm_provider=provider_id,
            llm_model=model_id,
            raw_plan=payload,
        )

    def _try_build_spec(
        self,
        *,
        plan: ExperimentPlan,
        payload: dict[str, Any],
        prompt: str,
        task_id: str,
        material_name: str | None,
        structure_path: str | None,
        forced_task_type: TaskType | None,
        submit_profile: str | None,
    ) -> ExperimentSpec | None:
        canonical_task_type = payload.get("canonical_task_type")
        if not canonical_task_type:
            return None

        blocked_experiment_types = {
            "adsorption_energy",
            "reaction_energy",
            "surface_reaction",
            "free_energy_correction",
            "neb",
            "mechanism",
            "transition_state_search",
            "ts_search",
        }
        if plan.experiment_type in blocked_experiment_types:
            return None

        if (
            plan.complexity == PlanComplexity.COMPLEX
            and len(plan.subtasks) > 1
            and any(subtask.task_type is None for subtask in plan.subtasks)
        ):
            return None

        try:
            task_type = forced_task_type or TaskType(str(canonical_task_type))
        except Exception:
            return None

        structure_source_raw = str(payload.get("structure_source") or "").strip()
        try:
            structure_source = (
                StructureSource(structure_source_raw)
                if structure_source_raw
                else (StructureSource.LOCAL_FILE if structure_path else StructureSource.MANUAL_BUILD)
            )
        except Exception:
            structure_source = StructureSource.LOCAL_FILE if structure_path else StructureSource.MANUAL_BUILD

        resolved_material_name = (
            material_name
            or str(payload.get("material_name") or "").strip()
            or self.fallback._infer_material_name(prompt)
        )
        structure_constraints = self.fallback._infer_structure_constraints(
            prompt=prompt,
            material_name=resolved_material_name,
            task_type=task_type,
        )
        payload_surface = payload.get("surface_hint")
        payload_defect = payload.get("defect_hint")
        if isinstance(payload_surface, dict):
            structure_constraints.surface = {
                "substrate": payload_surface.get("substrate"),
                "miller_index": payload_surface.get("miller_index"),
                "vacuum": payload_surface.get("vacuum"),
                "layers": payload_surface.get("layers"),
            }
        if isinstance(payload_defect, dict):
            structure_constraints.defect = dict(payload_defect)
        workflow_steps = [
            normalized
            for step in (payload.get("workflow") or [])
            if (normalized := self._normalize_workflow_step(str(step))) is not None
        ]
        workflow_steps = self._merge_with_default_workflow(task_type, workflow_steps)
        workflow_steps = self._filter_workflow_for_task_type(task_type, workflow_steps)

        return ExperimentSpec(
            task_id=task_id,
            task_type=task_type,
            material_name=resolved_material_name,
            source_prompt=prompt,
            description=str(payload.get("summary") or prompt),
            structure_source=structure_source,
            structure_path=structure_path,
            structure_id=str(payload.get("structure_id")) if payload.get("structure_id") else None,
            structure_constraints=structure_constraints,
            workflow=workflow_steps,
            functional=str(payload.get("functional") or self.fallback._infer_functional(prompt)),
            task_goal=self.fallback._build_task_goal(task_type),
            incar_overrides=dict(payload.get("incar_overrides") or {}),
            submit_profile=(
                self._normalize_submit_profile(submit_profile)
                or self._normalize_submit_profile(plan.recommended_submit_profile)
                or self._normalize_submit_profile(infer_submit_profile_from_prompt(prompt))
            ),
        )

    @staticmethod
    def _normalize_workflow_step(step: str) -> str | None:
        normalized = step.strip().lower()
        if normalized in WORKFLOW_STEP_ALIASES:
            return WORKFLOW_STEP_ALIASES[normalized]
        if any(
            token in normalized
            for token in [
                "structure preparation",
                "prepare structure",
                "structure setup",
                "preprocess",
                "pre-processing",
                "structure import",
                "model setup",
                "slab_model_preparation",
                "结构准备",
                "准备结构",
                "读取结构",
                "导入结构",
                "建模",
                "模型准备",
                "准备 pt(111) slab 结构文件",
            ]
        ):
            return None
        if "single point" in normalized or "单点" in normalized:
            return "single_point"
        if any(token in normalized for token in ["relax", "optimiz", "优化", "弛豫"]):
            return "relax"
        if "scf" in normalized or "自洽" in normalized:
            return "scf"
        if "band" in normalized or "能带" in normalized:
            return "band"
        if normalized == "dos" or "态密度" in normalized:
            return "dos"
        if normalized == "pdos" or "分波态密度" in normalized:
            return "pdos"
        if "charge" in normalized or "电荷" in normalized or "bader" in normalized:
            return "charge_analysis"
        if "work function" in normalized or "功函数" in normalized:
            return "work_function"
        if "locpot" in normalized or "真空区电势" in normalized or "真空能级" in normalized:
            return "work_function"
        if "frequency" in normalized or "振动频率" in normalized or "phonon" in normalized:
            return "frequency"
        if "molecular dynamics" in normalized or normalized == "md" or "分子动力学" in normalized:
            return "molecular_dynamics"
        if "transition state" in normalized or "过渡态" in normalized or normalized == "ts":
            return "transition_state_search"
        if "encut" in normalized:
            return "encut_convergence"
        if "kpoint" in normalized or "k 点" in normalized or "kpoints" in normalized:
            return "kpoints_convergence"
        if normalized == "eos" or "状态方程" in normalized:
            return "eos"
        return normalized

    @staticmethod
    def _merge_with_default_workflow(
        task_type: TaskType,
        workflow_steps: list[str],
    ) -> list[str]:
        default_steps = ExperimentSpec._default_workflow_for_task_type(task_type)
        if not workflow_steps:
            return default_steps

        merged: list[str] = []
        seen: set[str] = set()
        for step in default_steps:
            if step not in seen:
                merged.append(step)
                seen.add(step)
        for step in workflow_steps:
            if step not in seen:
                merged.append(step)
                seen.add(step)
        return merged

    @staticmethod
    def _normalize_submit_profile(profile_name: str | None) -> str | None:
        if profile_name is None:
            return None
        normalized = str(profile_name).strip()
        if not normalized:
            return None
        if normalized in SUBMIT_PROFILES:
            return normalized
        return None

    @staticmethod
    def _filter_workflow_for_task_type(
        task_type: TaskType,
        workflow_steps: list[str],
    ) -> list[str]:
        allowlist = TASK_TYPE_WORKFLOW_ALLOWLIST.get(task_type.value)
        if not allowlist:
            return workflow_steps
        filtered = [step for step in workflow_steps if step in allowlist]
        return filtered or ExperimentSpec._default_workflow_for_task_type(task_type)

    @staticmethod
    def _normalize_experiment_type(
        *,
        payload: dict[str, Any],
        complexity: PlanComplexity,
    ) -> str:
        raw_experiment_type = str(payload.get("experiment_type") or "unknown").strip()
        canonical_task_type = str(payload.get("canonical_task_type") or "").strip()
        normalized = raw_experiment_type.lower()

        if complexity == PlanComplexity.SIMPLE and canonical_task_type:
            return "single_calculation"

        alias_map = {
            "surface_work_function_calculation": "single_calculation",
            "work_function_calculation": "single_calculation",
            "dos_calculation": "single_calculation",
            "pdos_calculation": "single_calculation",
            "band_structure_calculation": "single_calculation",
            "charge_analysis_calculation": "single_calculation",
            "frequency_calculation": "single_calculation",
            "single_task": "single_calculation",
        }
        return alias_map.get(normalized, raw_experiment_type)
