from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

from dft_app.cluster_profiles import infer_submit_profile_from_prompt
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


TASK_PATTERNS: dict[TaskType, tuple[str, ...]] = {
    TaskType.SINGLE_POINT: ("single point", "单点", "单点能", "single-point"),
    TaskType.GEOMETRY_OPTIMIZATION: ("geometry optimization", "几何优化", "结构优化"),
    TaskType.STATIC_REFINEMENT: ("static", "静态精修", "静态计算", "static refinement"),
    TaskType.DOS: ("dos", "density of states", "态密度"),
    TaskType.PDOS: ("pdos", "projected dos", "partial dos", "分波态密度"),
    TaskType.BAND_STRUCTURE: ("band structure", "能带"),
    TaskType.CHARGE_ANALYSIS: ("charge analysis", "bader", "电荷分析", "差分电荷"),
    TaskType.WORK_FUNCTION: ("work function", "功函数"),
    TaskType.VIBRATIONAL_FREQUENCY: ("frequency", "振动频率", "频率计算", "phonon"),
    TaskType.TRANSITION_STATE_SEARCH: ("transition state", "过渡态", "ts", "vtst", "dimer"),
    TaskType.MOLECULAR_DYNAMICS: ("molecular dynamics", "md", "分子动力学"),
    TaskType.SPIN_RELATED: ("spin", "磁性", "自旋", "soc", "noncollinear"),
    TaskType.DEFECT_DOPING: ("defect", "doping", "掺杂", "缺陷"),
    TaskType.RELAX_SCF_BAND: ("band", "band structure", "能带"),
    TaskType.ENCUT_CONVERGENCE: (
        "encut convergence",
        "encut test",
        "encut 收敛",
    ),
    TaskType.KPOINTS_CONVERGENCE: (
        "kpoints convergence",
        "kpoint convergence",
        "k 点收敛",
    ),
    TaskType.EOS: ("equation of state", "eos", "状态方程"),
    TaskType.RELAX_SCF: ("scf", "静态", "自洽"),
    TaskType.RELAX: ("relax", "optimize", "优化", "弛豫"),
}

FUNCTIONAL_KEYWORDS = {
    "r2scan": "r2SCAN",
    "scan": "SCAN",
    "hse06": "HSE06",
    "pbesol": "PBEsol",
    "pbe": "PBE",
    "lda": "LDA",
}

COMPLEX_EXPERIMENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "adsorption_energy": (
        "吸附能",
        "adsorption energy",
        "adsorption",
        "adsorb",
    ),
    "transition_state_search": (
        "过渡态",
        "transition state",
        " ts ",
        "ts-",
        " ts",
        "vtst",
        "dimer",
        "爬山",
    ),
    "neb": (
        "neb",
        "ci-neb",
        "弹性带",
        "反应路径",
        "minimum energy path",
    ),
    "reaction_energy": (
        "reaction energy",
        "反应能",
        "反应热",
    ),
    "surface_reaction": (
        "表面反应",
        "surface reaction",
        "机理",
        "mechanism",
        "脱氢",
        "加氢",
    ),
    "free_energy_correction": (
        "free energy",
        "gibbs",
        "热力学修正",
        "自由能",
        "zero point energy",
        "zpe",
    ),
}


class RuleBasedPlanner:
    """Rule-first planner that converts prompts into ExperimentSpec."""

    def __init__(self, default_code: str = "vasp", default_scheduler: str = "slurm"):
        self.default_code = default_code
        self.default_scheduler = default_scheduler

    def plan(
        self,
        *,
        prompt: str,
        task_id: str,
        material_name: str | None = None,
        structure_path: str | None = None,
        forced_task_type: TaskType | None = None,
        submit_profile: str | None = None,
    ) -> ExperimentSpec:
        normalized_prompt = prompt.strip()
        inferred_material = material_name or self._infer_material_name(normalized_prompt)
        task_type = forced_task_type or self._infer_task_type(normalized_prompt)
        functional = self._infer_functional(normalized_prompt)
        incar_overrides = self._infer_incar_overrides(normalized_prompt)
        structure_constraints = self._infer_structure_constraints(
            prompt=normalized_prompt,
            material_name=inferred_material,
            task_type=task_type,
        )
        structure_source, structure_id = self._infer_structure_source(
            normalized_prompt, structure_path
        )
        inferred_submit_profile = submit_profile or infer_submit_profile_from_prompt(
            normalized_prompt
        )

        return ExperimentSpec(
            task_id=task_id,
            task_type=task_type,
            material_name=inferred_material,
            source_prompt=normalized_prompt,
            description=normalized_prompt,
            structure_source=structure_source,
            structure_path=structure_path,
            structure_id=structure_id,
            structure_constraints=structure_constraints,
            functional=functional,
            task_goal=self._build_task_goal(task_type),
            incar_overrides=incar_overrides,
            submit_profile=inferred_submit_profile,
            code=self.default_code,
            scheduler=self.default_scheduler,
        )

    def build_planning_artifacts(
        self,
        *,
        prompt: str,
        task_id: str,
        material_name: str | None = None,
        structure_path: str | None = None,
        forced_task_type: TaskType | None = None,
        submit_profile: str | None = None,
        fallback_message: str | None = None,
    ) -> tuple[ExperimentPlan, ExperimentSpec | None]:
        hints = self.build_llm_hints(
            prompt=prompt,
            material_name=material_name,
            structure_path=structure_path,
            forced_task_type=forced_task_type,
            submit_profile=submit_profile,
        )
        experiment_type = str(hints["experiment_type_guess"])
        message = fallback_message or "已使用规则解析。"

        if experiment_type == "single_calculation":
            spec = self.plan(
                prompt=prompt,
                task_id=task_id,
                material_name=material_name,
                structure_path=structure_path,
                forced_task_type=forced_task_type,
                submit_profile=submit_profile,
            )
            plan = ExperimentPlan(
                task_id=task_id,
                source_prompt=prompt,
                experiment_type="single_calculation",
                summary=message,
                complexity=PlanComplexity.SIMPLE,
                readiness=ExecutionReadiness.READY,
                requires_confirmation=spec.requires_confirmation,
                missing_information=[],
                assumptions=["已使用规则解析回退模式。"],
                subtasks=[
                    PlanSubtask(
                        name="single_task",
                        goal=spec.task_goal or "执行单个 VASP 任务",
                        system_role="primary_system",
                        task_type=spec.task_type.value,
                    )
                ],
                recommended_submit_profile=spec.submit_profile,
                raw_plan={
                    "fallback": True,
                    "rule_hints": hints,
                },
            )
            return plan, spec

        plan = self._build_complex_plan(
            prompt=prompt,
            task_id=task_id,
            hints=hints,
            fallback_message=message,
        )
        return plan, None

    def explain(self, spec: ExperimentSpec) -> dict[str, Any]:
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
            "requires_confirmation": spec.requires_confirmation,
            "confirmation_items": [item.value for item in spec.confirmation_items],
        }

    def build_llm_hints(
        self,
        *,
        prompt: str,
        material_name: str | None = None,
        structure_path: str | None = None,
        forced_task_type: TaskType | None = None,
        submit_profile: str | None = None,
    ) -> dict[str, Any]:
        normalized_prompt = prompt.strip()
        inferred_material = material_name or self._infer_material_name(normalized_prompt)
        inferred_submit_profile = submit_profile or infer_submit_profile_from_prompt(
            normalized_prompt
        )
        complex_experiment = self._infer_complex_experiment_type(normalized_prompt)
        structure_source, structure_id = self._infer_structure_source(
            normalized_prompt, structure_path
        )
        surface_hint = self._extract_surface_hint(normalized_prompt)
        adsorbate_hint = self._extract_adsorbate_hint(normalized_prompt)
        reaction_hint = self._extract_reaction_hint(normalized_prompt)

        if complex_experiment is None:
            return {
                "complexity_guess": "simple",
                "experiment_type_guess": "single_calculation",
                "canonical_task_type_guess": (
                    forced_task_type.value
                    if forced_task_type is not None
                    else self._infer_task_type(normalized_prompt).value
                ),
                "material_name_guess": inferred_material,
                "functional_guess": self._infer_functional(normalized_prompt),
                "structure_source_guess": structure_source.value,
                "structure_id_guess": structure_id,
                "submit_profile_guess": inferred_submit_profile,
                "surface_hint": surface_hint,
                "adsorbate_hint": adsorbate_hint,
                "reaction_hint": reaction_hint,
                "missing_information_guess": [],
            }

        return {
            "complexity_guess": "complex",
            "experiment_type_guess": complex_experiment,
            "canonical_task_type_guess": None,
            "material_name_guess": inferred_material,
            "functional_guess": self._infer_functional(normalized_prompt),
            "structure_source_guess": structure_source.value,
            "structure_id_guess": structure_id,
            "submit_profile_guess": inferred_submit_profile,
            "surface_hint": surface_hint,
            "adsorbate_hint": adsorbate_hint,
            "reaction_hint": reaction_hint,
            "missing_information_guess": self._missing_information_for_experiment(
                complex_experiment,
                normalized_prompt,
                surface_hint=surface_hint,
                adsorbate_hint=adsorbate_hint,
                reaction_hint=reaction_hint,
            ),
        }

    def _infer_task_type(self, prompt: str) -> TaskType:
        prompt_lower = prompt.lower()

        if "band" in prompt_lower or "能带" in prompt_lower:
            return TaskType.BAND_STRUCTURE
        if "pdos" in prompt_lower or "分波态密度" in prompt_lower:
            return TaskType.PDOS
        if "dos" in prompt_lower or "态密度" in prompt_lower:
            return TaskType.DOS
        if "单点" in prompt_lower or "single point" in prompt_lower:
            return TaskType.SINGLE_POINT
        if "静态精修" in prompt_lower or "static refinement" in prompt_lower:
            return TaskType.STATIC_REFINEMENT
        if "功函数" in prompt_lower or "work function" in prompt_lower:
            return TaskType.WORK_FUNCTION
        if "电荷分析" in prompt_lower or "bader" in prompt_lower or "差分电荷" in prompt_lower:
            return TaskType.CHARGE_ANALYSIS
        if "频率" in prompt_lower or "frequency" in prompt_lower or "phonon" in prompt_lower:
            return TaskType.VIBRATIONAL_FREQUENCY
        if "分子动力学" in prompt_lower or "molecular dynamics" in prompt_lower or re.search(r"\bmd\b", prompt_lower):
            return TaskType.MOLECULAR_DYNAMICS
        if "掺杂" in prompt_lower or "缺陷" in prompt_lower or "doping" in prompt_lower or "defect" in prompt_lower:
            return TaskType.DEFECT_DOPING
        if "自旋" in prompt_lower or "磁性" in prompt_lower or re.search(r"\bsoc\b", prompt_lower):
            return TaskType.SPIN_RELATED
        if "eos" in prompt_lower or "状态方程" in prompt_lower:
            return TaskType.EOS
        if "encut" in prompt_lower and ("convergence" in prompt_lower or "收敛" in prompt_lower):
            return TaskType.ENCUT_CONVERGENCE
        if (
            ("kpoint" in prompt_lower or "kpoints" in prompt_lower or "k 点" in prompt_lower)
            and ("convergence" in prompt_lower or "收敛" in prompt_lower)
        ):
            return TaskType.KPOINTS_CONVERGENCE
        if "几何优化" in prompt_lower or "geometry optimization" in prompt_lower or "结构优化" in prompt_lower:
            return TaskType.GEOMETRY_OPTIMIZATION
        if "scf" in prompt_lower or "自洽" in prompt_lower:
            return TaskType.STATIC_REFINEMENT
        if self._looks_like_ts_request(f" {prompt_lower} "):
            return TaskType.TRANSITION_STATE_SEARCH

        for task_type, keywords in TASK_PATTERNS.items():
            if any(keyword in prompt_lower for keyword in keywords):
                if task_type == TaskType.STATIC_REFINEMENT and (
                    "band" in prompt_lower or "能带" in prompt_lower
                ):
                    continue
                return task_type

        best_task = TaskType.GEOMETRY_OPTIMIZATION
        best_score = -1.0
        for task_type, keywords in TASK_PATTERNS.items():
            score = max(fuzz.partial_ratio(prompt_lower, keyword) for keyword in keywords)
            if score > best_score:
                best_score = score
                best_task = task_type
        return best_task

    def _infer_complex_experiment_type(self, prompt: str) -> str | None:
        prompt_lower = f" {prompt.lower()} "
        matched: list[tuple[str, int]] = []
        for experiment_type, keywords in COMPLEX_EXPERIMENT_PATTERNS.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in prompt_lower:
                    score += len(keyword)
            if score > 0:
                matched.append((experiment_type, score))

        if self._looks_like_adsorption_request(prompt_lower):
            matched.append(("adsorption_energy", 100))
        if self._looks_like_ts_request(prompt_lower):
            matched.append(("transition_state_search", 120))
        if self._looks_like_neb_request(prompt_lower):
            matched.append(("neb", 110))

        if not matched:
            return None

        matched.sort(key=lambda item: item[1], reverse=True)
        return matched[0][0]

    def _infer_material_name(self, prompt: str) -> str:
        mp_match = re.search(r"\bmp-\d+\b", prompt, flags=re.IGNORECASE)
        if mp_match:
            return mp_match.group(0)

        formula_candidates = re.findall(
            r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*){0,5})\b", prompt
        )
        blacklist = {"SOC", "PBE", "LDA", "SCAN", "HSE", "DOS", "EOS"}
        for candidate in formula_candidates:
            if candidate.upper() not in blacklist and any(ch.isalpha() for ch in candidate):
                return candidate
        return "UNKNOWN_MATERIAL"

    def _infer_functional(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        for key, value in FUNCTIONAL_KEYWORDS.items():
            if key in prompt_lower:
                return value
        return "PBE"

    def _infer_incar_overrides(self, prompt: str) -> dict[str, Any]:
        overrides: dict[str, Any] = {}

        ivdw_match = re.search(r"\bIVDW\s*=\s*(\d+)\b", prompt, flags=re.IGNORECASE)
        if ivdw_match:
            overrides["IVDW"] = int(ivdw_match.group(1))

        encut_match = re.search(r"\bENCUT\s*=\s*(\d+)\b", prompt, flags=re.IGNORECASE)
        if encut_match:
            overrides["ENCUT"] = int(encut_match.group(1))

        if re.search(r"\bSOC\b|spin[- ]?orbit", prompt, flags=re.IGNORECASE):
            overrides["LSORBIT"] = True

        return overrides

    def _infer_structure_source(
        self, prompt: str, structure_path: str | None
    ) -> tuple[StructureSource, str | None]:
        if structure_path:
            return StructureSource.LOCAL_FILE, None

        mp_match = re.search(r"\bmp-\d+\b", prompt, flags=re.IGNORECASE)
        if mp_match:
            return StructureSource.MATERIALS_PROJECT, mp_match.group(0)

        if re.search(r"materials project|mp api", prompt, flags=re.IGNORECASE):
            return StructureSource.MATERIALS_PROJECT, "TO_BE_CONFIRMED"

        return StructureSource.MANUAL_BUILD, None

    def _infer_structure_constraints(
        self,
        *,
        prompt: str,
        material_name: str,
        task_type: TaskType,
    ) -> StructureConstraint:
        combined_text = f"{material_name} {prompt}"
        surface_hint = self._extract_surface_hint(combined_text)
        defect_hint = self._extract_defect_hint(prompt)
        supercell = self._extract_supercell_hint(prompt)

        surface_constraint = None
        if surface_hint is not None and (
            task_type == TaskType.WORK_FUNCTION
            or surface_hint.get("miller_index") is not None
            or re.search(r"surface|表面|slab", combined_text, flags=re.IGNORECASE)
        ):
            surface_constraint = {
                "substrate": surface_hint.get("substrate"),
                "miller_index": surface_hint.get("miller_index"),
                "vacuum": None,
                "layers": None,
            }

        return StructureConstraint(
            supercell=supercell,
            surface=surface_constraint,
            defect=defect_hint,
        )

    @staticmethod
    def _extract_defect_hint(prompt: str) -> dict[str, Any] | None:
        prompt = prompt.strip()
        if not re.search(r"掺杂|缺陷|doping|defect|vacancy|空位", prompt, flags=re.IGNORECASE):
            return None

        site_role = None
        if re.search(r"金属位|metal site|cation site|阳离子位", prompt, flags=re.IGNORECASE):
            site_role = "cation_like"
        elif re.search(r"阴离子位|anion site", prompt, flags=re.IGNORECASE):
            site_role = "anion_like"

        geometry_hint = None
        if re.search(r"八面体|octa", prompt, flags=re.IGNORECASE):
            geometry_hint = "octahedral_like"
        elif re.search(r"四面体|tetra", prompt, flags=re.IGNORECASE):
            geometry_hint = "tetrahedral_like"
        elif re.search(r"三角棱柱|trigonal prismatic|trigonal prism", prompt, flags=re.IGNORECASE):
            geometry_hint = "trigonal_prismatic_like"
        site_mode = None
        if re.search(r"间隙|interstitial|间隙位", prompt, flags=re.IGNORECASE):
            site_mode = "interstitial"
        elif re.search(r"替位|取代位|substitution|substitutional", prompt, flags=re.IGNORECASE):
            site_mode = "substitutional"
        layer_region_hint = None
        if re.search(r"层间|interlayer", prompt, flags=re.IGNORECASE):
            layer_region_hint = "interlayer_gap_like"
        elif re.search(r"层内|intralayer", prompt, flags=re.IGNORECASE):
            layer_region_hint = "intralayer_cavity_like"

        wyckoff_match = re.search(
            r"(?:wyckoff\s*[:：]?\s*|wyckoff位\s*|位于\s*)(\d+[A-Za-z])\b",
            prompt,
            flags=re.IGNORECASE,
        ) or re.search(r"\b(\d+[A-Za-z])\s*(?:位|位点)\b", prompt)
        wyckoff = wyckoff_match.group(1) if wyckoff_match else None

        surface_site_hint = None
        if re.search(r"\btop\b|顶位", prompt, flags=re.IGNORECASE):
            surface_site_hint = "top"
        elif re.search(r"\bbridge\b|桥位", prompt, flags=re.IGNORECASE):
            surface_site_hint = "bridge"
        elif re.search(r"\bhollow\b|空心位", prompt, flags=re.IGNORECASE):
            surface_site_hint = "hollow"
        elif re.search(r"\bfcc\b", prompt, flags=re.IGNORECASE):
            surface_site_hint = "fcc"
        elif re.search(r"\bhcp\b", prompt, flags=re.IGNORECASE):
            surface_site_hint = "hcp"

        vacancy_match = re.search(r"([A-Z][a-z]?)\s*(?:vacancy|空位)", prompt, flags=re.IGNORECASE)
        if vacancy_match:
            return {
                "mode": "vacancy",
                "species": vacancy_match.group(1),
                "site": vacancy_match.group(1),
                "site_role": site_role,
                "geometry_hint": geometry_hint,
                "site_mode": site_mode,
                "layer_region_hint": layer_region_hint,
                "wyckoff": wyckoff,
                "surface_site_hint": surface_site_hint,
            }

        zh_match = re.search(r"([A-Z][a-z]?)\s*(?:[^，。；;,.]{0,16})?\s*掺杂", prompt)
        en_match = re.search(r"([A-Z][a-z]?)\s*(?:doping|doped)", prompt, flags=re.IGNORECASE)
        dopant = zh_match.group(1) if zh_match else (en_match.group(1) if en_match else None)
        if dopant in {"Wy"}:
            dopant = None
        if dopant is None:
            element_tokens = [
                token
                for token in re.findall(r"\b([A-Z][a-z]?)\b", prompt)
                if token not in {"Wy"}
            ]
            if element_tokens:
                dopant = element_tokens[0]
        site_match = (
            re.search(r"(?:替代|取代|掺杂(?:到|在)?)\s*([A-Z][a-z]?)\s*(?:位|位点)?", prompt)
            or re.search(r"([A-Z][a-z]?)\s*(?:位|位点)", prompt)
            or re.search(
                r"(?:substitut(?:e|ing)\s*(?:for)?|at|on)\s*([A-Z][a-z]?)",
                prompt,
                flags=re.IGNORECASE,
            )
        )
        site = site_match.group(1) if site_match else None
        if site in {"Wy"}:
            site = None
        if dopant:
            return {
                "mode": "doping",
                "dopant": dopant,
                "site": site,
                "site_role": site_role,
                "geometry_hint": geometry_hint,
                "site_mode": site_mode,
                "layer_region_hint": layer_region_hint,
                "wyckoff": wyckoff,
                "surface_site_hint": surface_site_hint,
            }
        return {
            "mode": "defect_or_doping",
            "dopant": None,
            "site": site,
            "site_role": site_role,
            "geometry_hint": geometry_hint,
            "site_mode": site_mode,
            "layer_region_hint": layer_region_hint,
            "wyckoff": wyckoff,
            "surface_site_hint": surface_site_hint,
        }

    @staticmethod
    def _extract_supercell_hint(prompt: str) -> list[list[int]] | None:
        matrix_match = re.search(
            r"(\d+)\s*[xX×]\s*(\d+)\s*[xX×]\s*(\d+)",
            prompt,
        )
        if not matrix_match:
            return None
        a, b, c = (int(matrix_match.group(index)) for index in range(1, 4))
        return [[a, 0, 0], [0, b, 0], [0, 0, c]]

    def _build_task_goal(self, task_type: TaskType) -> str:
        mapping = {
            TaskType.SINGLE_POINT: "完成单点能计算",
            TaskType.GEOMETRY_OPTIMIZATION: "完成几何优化",
            TaskType.STATIC_REFINEMENT: "完成静态精修计算",
            TaskType.DOS: "完成态密度 DOS 计算",
            TaskType.PDOS: "完成分波态密度 PDOS 计算",
            TaskType.BAND_STRUCTURE: "完成能带计算",
            TaskType.CHARGE_ANALYSIS: "完成电荷分析",
            TaskType.WORK_FUNCTION: "完成功函数计算",
            TaskType.VIBRATIONAL_FREQUENCY: "完成振动频率计算",
            TaskType.TRANSITION_STATE_SEARCH: "完成过渡态搜索准备与计算",
            TaskType.MOLECULAR_DYNAMICS: "完成分子动力学计算",
            TaskType.SPIN_RELATED: "完成自旋相关计算",
            TaskType.DEFECT_DOPING: "完成缺陷与掺杂计算",
            TaskType.RELAX: "完成结构弛豫",
            TaskType.RELAX_SCF: "完成结构弛豫并进行自洽静态计算",
            TaskType.RELAX_SCF_BAND: "完成结构弛豫、自洽静态计算和能带计算",
            TaskType.ENCUT_CONVERGENCE: "完成 ENCUT 收敛测试",
            TaskType.KPOINTS_CONVERGENCE: "完成 KPOINTS 收敛测试",
            TaskType.EOS: "生成 EOS 数据并拟合状态方程",
        }
        return mapping[task_type]

    def _build_complex_plan(
        self,
        *,
        prompt: str,
        task_id: str,
        hints: dict[str, Any],
        fallback_message: str,
    ) -> ExperimentPlan:
        experiment_type = str(hints["experiment_type_guess"])
        subtasks = self._subtasks_for_experiment(experiment_type)
        return ExperimentPlan(
            task_id=task_id,
            source_prompt=prompt,
            experiment_type=experiment_type,
            summary=fallback_message,
            complexity=PlanComplexity.COMPLEX,
            readiness=ExecutionReadiness.NEEDS_CONFIRMATION,
            requires_confirmation=True,
            missing_information=[
                str(item) for item in (hints.get("missing_information_guess") or [])
            ],
            assumptions=["已使用规则解析回退模式。"],
            subtasks=subtasks,
            recommended_submit_profile=(
                str(hints.get("submit_profile_guess"))
                if hints.get("submit_profile_guess")
                else None
            ),
            raw_plan={
                "fallback": True,
                "rule_hints": hints,
            },
        )

    def _subtasks_for_experiment(self, experiment_type: str) -> list[PlanSubtask]:
        if experiment_type == "adsorption_energy":
            return [
                PlanSubtask(
                    name="clean_slab",
                    goal="构建并优化洁净表面，得到 E_slab。",
                    system_role="slab",
                    task_type="relax_scf",
                ),
                PlanSubtask(
                    name="isolated_adsorbate",
                    goal="构建并优化孤立分子，得到 E_molecule。",
                    system_role="molecule",
                    task_type="relax_scf",
                ),
                PlanSubtask(
                    name="adsorbed_system",
                    goal="构建并优化吸附体系，得到 E_adsorbate_slab。",
                    system_role="adsorbate_slab",
                    task_type="relax_scf",
                ),
                PlanSubtask(
                    name="adsorption_energy_analysis",
                    goal="汇总能量并计算吸附能。",
                    system_role="analysis",
                    task_type=None,
                ),
            ]
        if experiment_type == "transition_state_search":
            return [
                PlanSubtask(
                    name="initial_state",
                    goal="准备或确认初态参考结构。",
                    system_role="reference_state",
                    task_type="relax",
                ),
                PlanSubtask(
                    name="transition_state_guess",
                    goal="构造并确认过渡态初猜。",
                    system_role="transition_state_guess",
                    task_type="transition_state_search",
                ),
                PlanSubtask(
                    name="frequency_check",
                    goal="验证过渡态候选结构是否只有一个虚频。",
                    system_role="analysis",
                    task_type=None,
                ),
            ]
        if experiment_type == "neb":
            return [
                PlanSubtask(
                    name="initial_state",
                    goal="准备初态端点结构。",
                    system_role="initial_state",
                    task_type="relax",
                ),
                PlanSubtask(
                    name="final_state",
                    goal="准备终态端点结构。",
                    system_role="final_state",
                    task_type="relax",
                ),
                PlanSubtask(
                    name="neb_images",
                    goal="生成并优化 NEB 中间 images。",
                    system_role="reaction_path",
                    task_type="neb",
                ),
            ]
        return [
            PlanSubtask(
                name="complex_task",
                goal="复杂任务需要先拆分体系并确认参数。",
                system_role="primary_system",
                task_type=None,
            )
        ]

    def _missing_information_for_experiment(
        self,
        experiment_type: str,
        prompt: str,
        *,
        surface_hint: dict[str, Any] | None,
        adsorbate_hint: str | None,
        reaction_hint: dict[str, Any] | None,
    ) -> list[str]:
        missing: list[str] = []
        prompt_lower = prompt.lower()
        if experiment_type == "adsorption_energy":
            if not surface_hint:
                missing.append("surface")
            if not adsorbate_hint:
                missing.append("adsorbate")
            missing.extend(
                [
                    "layers",
                    "vacuum",
                    "fixed_bottom_layers",
                    "adsorption_site",
                    "orientation",
                ]
            )
        elif experiment_type == "transition_state_search":
            missing.extend(
                [
                    "initial_state_structure",
                    "ts_guess_strategy",
                    "frequency_validation",
                ]
            )
        elif experiment_type == "neb":
            missing.extend(
                [
                    "initial_state_structure",
                    "final_state_structure",
                    "image_count",
                ]
            )
        elif experiment_type in {"reaction_energy", "surface_reaction"}:
            if reaction_hint is None:
                missing.extend(["reactants", "products"])
            if "surface" in experiment_type and not surface_hint:
                missing.append("surface")

        if "pbe" not in prompt_lower and "scan" not in prompt_lower and "hse" not in prompt_lower:
            missing.append("functional")

        deduped: list[str] = []
        for item in missing:
            if item not in deduped:
                deduped.append(item)
        return deduped

    @staticmethod
    def _extract_surface_hint(prompt: str) -> dict[str, Any] | None:
        match = re.search(r"([A-Z][a-z]?)\((\d{3})\)", prompt)
        if match:
            return {
                "substrate": match.group(1),
                "miller_index": [int(ch) for ch in match.group(2)],
            }
        if re.search(r"surface|表面|slab", prompt, flags=re.IGNORECASE):
            return {"substrate": None, "miller_index": None}
        return None

    @staticmethod
    def _extract_adsorbate_hint(prompt: str) -> str | None:
        zh_match = re.search(r"(.+?)在.+?(?:表面|面)上.*吸附", prompt)
        if zh_match:
            return zh_match.group(1).strip(" ，,。")
        en_match = re.search(
            r"adsorption(?: energy)? of (.+?) on .+",
            prompt,
            flags=re.IGNORECASE,
        )
        if en_match:
            return en_match.group(1).strip()
        return None

    @staticmethod
    def _extract_reaction_hint(prompt: str) -> dict[str, Any] | None:
        if "->" in prompt:
            reactant_text, product_text = prompt.split("->", 1)
            return {
                "reactants": reactant_text.strip(),
                "products": product_text.strip(),
            }
        return None

    @staticmethod
    def _looks_like_adsorption_request(prompt_text: str) -> bool:
        keywords = ["吸附", "adsorption", "adsorb"]
        return any(keyword in prompt_text for keyword in keywords)

    @staticmethod
    def _looks_like_ts_request(prompt_text: str) -> bool:
        keywords = [
            "过渡态",
            "transition state",
            " ts ",
            "ts-",
            " vtst",
            "dimer",
            "爬山",
        ]
        return any(keyword in prompt_text for keyword in keywords)

    @staticmethod
    def _looks_like_neb_request(prompt_text: str) -> bool:
        keywords = [
            "neb",
            "ci-neb",
            "弹性带",
            "反应路径",
            "minimum energy path",
        ]
        return any(keyword in prompt_text for keyword in keywords)
