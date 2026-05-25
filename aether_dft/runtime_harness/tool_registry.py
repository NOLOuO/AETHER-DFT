from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ase.io import read
from dft_app.cli.main import collect_adsorption_workflow_status
from dft_app.remote import SSHRemoteRunner
from dft_shared.structure_analyzer.bond_analyzer import analyze_bonds
from dft_shared.structure_analyzer.comparator import compare_structures
from dft_shared.structure_analyzer.io import convert_structure
from dft_shared.structure_analyzer.operations import (
    add_adsorbate,
    add_dopant,
    add_vacancy,
    candidate_quality_score,
    enumerate_defect_sites,
    enumerate_adsorption_sites,
    inspect_slab_surface,
    interpolate_ts_midpoint_candidates,
    make_supercell,
    resolve_structure,
    sanity_check,
    structure_relax_short,
)
from dft_shared.chemistry_hints import get_adsorbate_chemistry_hint
from dft_app.modeling.candidate_manifest import compose_manifest_from_authored_candidates
from dft_shared.workflow_config import load_workflow_config

from aether_dft.adsorption import (
    build_adsorption_slab,
    generate_adsorption_candidates,
    plan_adsorption_task,
    run_adsorption_full_workflow,
)
from aether_dft.prompt_engine import load_architecture_live_doc_snapshot
from aether_dft.permissions import get_permission_mode, permission_mode_label, should_allow_tool
from aether_dft.project_state import append_progress, project_paths, read_project_context, write_project_state
from aether_dft.adsorption_authoring import (
    AdsorptionCandidatePlan,
    create_candidate_plan,
    list_candidate_plans,
    load_candidate_plan,
)
from aether_dft.candidate_outcomes import record_candidate_outcome
from aether_dft.convergence import compose_convergence_plan
from aether_dft.evaluation import list_adsorption_eval_cases, render_model_comparison_report, score_adsorption_plan_against_eval
from aether_dft.knowledge import add_note, list_notes, search_for_system, search_notes, show_note
from aether_dft.recommendations import recommend_next_tasks
from aether_dft.research_workspace import append_research_progress, build_research_proposal, read_research_onboarding_context
from aether_dft.task_bridge import create_task_plan, run_dft_task


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    read_only: bool = True
    required: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False},
        },
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__") and not isinstance(value, (dict, list, str, int, float, bool, type(None))):
        try:
            return asdict(value)  # type: ignore[arg-type]
        except Exception:
            return dict(value.__dict__)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


class ToolRegistry:
    def __init__(self, *, allow_cluster_submit: bool = False, permission_mode: str | None = None):
        self.allow_cluster_submit = allow_cluster_submit
        self.permission_mode = permission_mode or get_permission_mode()
        self._tools: dict[str, tuple[ToolSpec, Callable[[dict[str, Any]], dict[str, Any]]]] = {}
        self._register_all()

    def _register(self, spec: ToolSpec, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._tools[spec.name] = (spec, handler)

    def _register_all(self) -> None:
        self._register(ToolSpec("computational_chemistry_workflow_map", "列出 AETHER-DFT 两步主线与工作流阶段。", {}), self._workflow_map)
        self._register(ToolSpec("structure_modeling_tool_status", "报告 Step 2 结构建模工具能力、适用任务类型、证据门槛与当前完成度。", {}, True), self._structure_modeling_tool_status)
        self._register(ToolSpec("structure_modeling_intent_plan", "把自然语言 Step 2 建模意图转成非强制的工具选择建议、缺失输入和证据门槛；这是导航，不是固定流水线。", {"intent": {"type": "string"}, "available_inputs": {"type": "object"}, "project": {"type": "string"}, "allow_writes": {"type": "boolean"}}, True, ("intent",)), self._structure_modeling_intent_plan)
        self._register(ToolSpec("research_onboarding_context", "读取 research 入职上下文：AGENTS、避坑清单、项目研究进展。", {"project": {"type": "string"}, "max_chars": {"type": "integer"}}, True), self._research_onboarding_context)
        self._register(ToolSpec("research_proposal_plan", "把自然语言课题讨论整理成科学问题、结构需求、证据需求和下一步。", {"prompt": {"type": "string"}, "project": {"type": "string"}}, True, ("prompt",)), self._research_proposal_plan)
        self._register(ToolSpec("research_progress_append", "按研究工作区格式倒序追加 research/<项目>/研究进展.md。", {"project": {"type": "string"}, "completed": {"type": "array", "items": {"type": "string"}}, "blockers": {"type": "array", "items": {"type": "string"}}, "next_steps": {"type": "array", "items": {"type": "string"}}}, False, ("project",)), self._research_progress_append)
        self._register(ToolSpec("project_state_read", "读取项目 state 与 progress。", {"project": {"type": "string"}, "max_chars": {"type": "integer"}}, True, ("project",)), self._project_state_read)
        self._register(ToolSpec("project_progress_append", "追加研究进展。", {"project": {"type": "string"}, "completed": {"type": "array", "items": {"type": "string"}}, "blockers": {"type": "array", "items": {"type": "string"}}, "next_steps": {"type": "array", "items": {"type": "string"}}}, False, ("project",)), self._project_progress_append)
        self._register(ToolSpec("knowledge_note_add", "把重要结论/参数经验写入项目知识库。", {"project": {"type": "string"}, "title": {"type": "string"}, "content": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}, False, ("project", "title", "content")), self._knowledge_note_add)
        self._register(ToolSpec("knowledge_note_list", "列出项目知识库笔记。", {"project": {"type": "string"}}, True, ("project",)), self._knowledge_note_list)
        self._register(ToolSpec("knowledge_note_search", "在项目知识库中搜索笔记。", {"project": {"type": "string"}, "query": {"type": "string"}}, True, ("project", "query")), self._knowledge_note_search)
        self._register(ToolSpec("knowledge_note_show", "读取单条知识笔记。", {"note": {"type": "string"}, "project": {"type": "string"}}, True, ("note",)), self._knowledge_note_show)
        self._register(ToolSpec("architecture_live_doc_snapshot", "读取智能体架构.md 作为 volatile context。", {"max_chars": {"type": "integer"}}, True), self._architecture_live_doc_snapshot)
        self._register(ToolSpec("architecture_live_doc_update", "向智能体架构.md 追加块。", {"title": {"type": "string"}, "content": {"type": "string"}}, False), self._architecture_live_doc_update)
        self._register(ToolSpec("structure_convert", "结构格式转换。", {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "fmt": {"type": "string"}}, False, ("input_path", "output_path")), self._structure_convert)
        self._register(ToolSpec("structure_resolve", "只读解析本地结构、Materials Project mp-id 或 ASE 单元素 bulk，返回 summary/metadata。", {"input_path": {"type": "string"}, "material": {"type": "string"}, "mp_id": {"type": "string"}, "source": {"type": "string"}}, True), self._structure_resolve)
        self._register(ToolSpec("structure_supercell", "对结构做 supercell 扩胞；不等同任意切胞。", {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "scaling_matrix": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}}, False, ("input_path", "output_path", "scaling_matrix")), self._structure_supercell)
        self._register(ToolSpec("structure_build_slab", "按 material/miller/supercell/vacuum/fixed-layer 参数构建 slab POSCAR。", {"material": {"type": "string"}, "output_dir": {"type": "string"}, "miller_index": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}, "supercell": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}, "structure_path": {"type": "string"}, "mp_id": {"type": "string"}, "source": {"type": "string"}, "min_slab_size": {"type": "number"}, "min_vacuum_size": {"type": "number"}, "fixed_bottom_layers": {"type": "integer"}, "center_slab": {"type": "boolean"}}, False, ("material", "output_dir")), self._structure_build_slab)
        self._register(ToolSpec("structure_enumerate_sites", "枚举 slab 上的吸附位点（ontop/bridge/hollow 等），返回每个位点的笛卡尔坐标、site_family 与最近邻顶层原子；模型自主选择候选时优先用它。", {"slab_path": {"type": "string"}, "max_sites_per_family": {"type": "integer"}, "top_layer_tolerance": {"type": "number"}, "nearest_neighbors": {"type": "integer"}}, True, ("slab_path",)), self._structure_enumerate_sites)
        self._register(ToolSpec("slab_surface_inspect", "报告 slab 表面化学环境：顶层 / 第二层原子的配位数 / 最近邻 / 对称等价分组，让模型在 enumerate_sites 之前就能识别对称冗余、合金分布与特殊原子。", {"slab_path": {"type": "string"}, "top_layer_tolerance": {"type": "number"}, "second_layer_tolerance": {"type": "number"}, "neighbor_radius": {"type": "number"}, "symprec": {"type": "number"}}, True, ("slab_path",)), self._slab_surface_inspect)
        self._register(ToolSpec("adsorbate_chemistry_hint", "返回吸附物的化学先验：候选 anchor 原子、典型 binding motif、几何尺寸与典型吸附高度；查 curated 表，回退到 ASE / RDKit 启发式。生成候选前优先调它。", {"adsorbate": {"type": "string"}}, True, ("adsorbate",)), self._adsorbate_chemistry_hint)
        self._register(ToolSpec("knowledge_search_for_system", "跨项目 KB + research workspace 搜索与给定 material+adsorbate 相关的先验笔记。生成候选前优先调它确认是否有过去经验。", {"material": {"type": "string"}, "adsorbate": {"type": "string"}, "extra_terms": {"type": "array", "items": {"type": "string"}}, "project_priority": {"type": "string"}, "max_results": {"type": "integer"}}, True), self._knowledge_search_for_system)
        self._register(ToolSpec("structure_add_adsorbate", "在 slab 指定位点或顶层原子附近添加单个吸附物初猜并默认冻结底层 N 层（与黑盒生成器一致）。site_index 为 0-based 顶层原子序号；如要精确放置请用 cart_coords（可从 structure_enumerate_sites 获取）。", {"slab_path": {"type": "string"}, "adsorbate": {"type": "string"}, "output_path": {"type": "string"}, "height": {"type": "number"}, "site_index": {"type": "integer"}, "cart_coords": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "orientation": {"type": "string"}, "anchor_atom_index": {"type": "integer"}, "anchor_symbol": {"type": "string"}, "coords_mode": {"type": "string"}, "fixed_bottom_layers": {"type": "integer"}}, False, ("slab_path", "adsorbate", "output_path")), self._structure_add_adsorbate)
        self._register(ToolSpec("candidate_quality_score", "对模型自主生成的吸附候选做几何自检：anchor-surface 距离、吸附物完整性、重叠/floating 风险，低分需重试。", {"slab_path": {"type": "string"}, "candidate_path": {"type": "string"}, "adsorbate": {"type": "string"}, "anchor_symbol": {"type": "string"}, "top_layer_tolerance": {"type": "number"}, "min_anchor_surface_distance": {"type": "number"}, "max_anchor_surface_distance": {"type": "number"}, "min_adsorbate_slab_distance": {"type": "number"}}, True, ("slab_path", "candidate_path")), self._candidate_quality_score)
        self._register(ToolSpec("structure_relax_short", "用本地 ASE calculator 做真实短程预优化筛查；默认 EMT，失败/不支持会明确返回，不假装成功。", {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "calculator": {"type": "string"}, "max_steps": {"type": "integer"}, "fmax": {"type": "number"}, "trajectory_path": {"type": "string"}}, False, ("input_path", "output_path")), self._structure_relax_short)
        self._register(ToolSpec("structure_defect", "统一缺陷入口：vacancy 或 substitution dopant。", {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "mode": {"type": "string"}, "species": {"type": "string"}, "index": {"type": "integer"}, "dopant": {"type": "string"}, "surface_only": {"type": "boolean"}}, False, ("input_path", "output_path", "mode")), self._structure_defect)
        self._register(ToolSpec("defect_site_enumerate", "枚举 vacancy/substitution 可操作原子位点，供模型写缺陷候选推理。", {"structure_path": {"type": "string"}, "species": {"type": "string"}, "surface_only": {"type": "boolean"}, "top_layer_tolerance": {"type": "number"}, "max_sites": {"type": "integer"}}, True, ("structure_path",)), self._defect_site_enumerate)
        self._register(ToolSpec("structure_add_vacancy", "删除指定元素原子以生成 vacancy/氧空位等缺陷结构。", {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "species": {"type": "string"}, "index": {"type": "integer"}, "surface_only": {"type": "boolean"}}, False, ("input_path", "output_path", "species")), self._structure_add_vacancy)
        self._register(ToolSpec("structure_add_dopant", "替换指定原子为 dopant，生成掺杂结构。", {"input_path": {"type": "string"}, "output_path": {"type": "string"}, "dopant": {"type": "string"}, "species": {"type": "string"}, "index": {"type": "integer"}, "surface_only": {"type": "boolean"}}, False, ("input_path", "output_path", "dopant")), self._structure_add_dopant)
        self._register(ToolSpec("structure_sanity_check", "检查结构最短距离、真空估算、物种和原子数。", {"structure_path": {"type": "string"}, "min_distance": {"type": "number"}, "min_vacuum": {"type": "number"}, "vacuum_axis": {"type": "string"}}, True, ("structure_path",)), self._structure_sanity_check)
        self._register(ToolSpec("structure_bond_analyze", "键连与配位分析。", {"structure_path": {"type": "string"}}, True, ("structure_path",)), self._structure_bond_analyze)
        self._register(ToolSpec("structure_displacement_compare", "结构位移对比。", {"initial_path": {"type": "string"}, "final_path": {"type": "string"}, "top_n": {"type": "integer"}}, True, ("initial_path", "final_path")), self._structure_displacement_compare)
        self._register(ToolSpec("adsorption_plan", "吸附任务规划。", {"prompt": {"type": "string"}, "project": {"type": "string"}, "adsorbate": {"type": "string"}, "material": {"type": "string"}, "slab_path": {"type": "string"}, "preferred_site": {"type": "string"}, "preferred_orientation": {"type": "string"}, "persist": {"type": "boolean"}}, True, ("prompt",)), self._adsorption_plan)
        self._register(ToolSpec("adsorption_build_slab", "构建 slab（structure_build_slab 的兼容别名）。", {"material": {"type": "string"}, "output_dir": {"type": "string"}, "miller_index": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}, "supercell": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}, "structure_path": {"type": "string"}, "mp_id": {"type": "string"}, "source": {"type": "string"}, "min_slab_size": {"type": "number"}, "min_vacuum_size": {"type": "number"}, "fixed_bottom_layers": {"type": "integer"}}, False, ("material", "output_dir")), self._adsorption_build_slab)
        self._register(ToolSpec("adsorption_candidates", "批量自动枚举所有 site×orientation 的兜底生成器，不做领域判断；优先用 structure_enumerate_sites + structure_add_adsorbate + adsorption_candidate_manifest_compose 走模型自主路径，仅在 slab/吸附物特别陌生或想要快速 baseline 时回退到本工具。", {"slab_path": {"type": "string"}, "adsorbate": {"type": "string"}, "material": {"type": "string"}, "prompt": {"type": "string"}, "project": {"type": "string"}, "output_dir": {"type": "string"}, "task_id": {"type": "string"}, "candidate_height": {"type": "number"}, "max_sites_per_family": {"type": "integer"}, "preferred_site": {"type": "string"}, "preferred_orientation": {"type": "string"}, "vacancy_species": {"type": "string"}}, False, ("slab_path", "adsorbate", "material")), self._adsorption_candidates)
        self._register(ToolSpec("adsorption_candidate_plan", "创建结构化推理 plan：rationale / expected_binding_motif / anchor_atom / target_sites(含 reason) / target_orientations / 排除位点。compose_manifest 之前必须先调它。", {"material": {"type": "string"}, "adsorbate": {"type": "string"}, "rationale": {"type": "string"}, "expected_binding_motif": {"type": "string"}, "anchor_atom": {"type": "string"}, "target_sites": {"type": "array", "items": {"type": "object"}}, "target_orientations": {"type": "array", "items": {"type": "string"}}, "excluded_sites_with_reason": {"type": "array", "items": {"type": "object"}}, "symmetry_pruning_applied": {"type": "boolean"}, "priors_consulted": {"type": "object"}, "project": {"type": "string"}, "task_id": {"type": "string"}, "notes": {"type": "string"}}, False, ("material", "adsorbate", "rationale", "expected_binding_motif", "anchor_atom", "target_sites", "target_orientations")), self._adsorption_candidate_plan)
        self._register(ToolSpec("adsorption_candidate_plan_list", "列出某项目（或 runtime）下已创建的 adsorption candidate plans。", {"project": {"type": "string"}}, True), self._adsorption_candidate_plan_list)
        self._register(ToolSpec("adsorption_candidate_manifest_compose", "把模型自己生成的 POSCAR 收编成 manifest.json；必须传 plan_id（由 adsorption_candidate_plan 产生），每个 candidate 的 reason ≥ 20 字，site_label 需与 plan.target_sites 对齐。", {"task_id": {"type": "string"}, "material_name": {"type": "string"}, "source_prompt": {"type": "string"}, "slab_source": {"type": "string"}, "adsorbate_source": {"type": "string"}, "output_dir": {"type": "string"}, "candidates": {"type": "array", "items": {"type": "object"}}, "metadata": {"type": "object"}, "plan_id": {"type": "string"}, "project": {"type": "string"}, "prune_rationale": {"type": "string"}}, False, ("task_id", "material_name", "source_prompt", "slab_source", "adsorbate_source", "output_dir", "candidates", "plan_id")), self._adsorption_candidate_manifest_compose)
        self._register(ToolSpec("candidate_outcome_record", "把已完成候选的 DFT 结果复盘写回 KB：E_ads、verdict、初末态位移/漂移、可复用经验。只记录证据，不假装执行。", {"project": {"type": "string"}, "material": {"type": "string"}, "adsorbate": {"type": "string"}, "candidate_id": {"type": "string"}, "verdict": {"type": "string"}, "adsorption_energy_ev": {"type": "number"}, "initial_path": {"type": "string"}, "final_path": {"type": "string"}, "manifest_path": {"type": "string"}, "calculation_summary": {"type": "string"}, "notes": {"type": "string"}}, False, ("project", "material", "adsorbate", "candidate_id", "verdict")), self._candidate_outcome_record)
        self._register(ToolSpec("adsorption_full_workflow", "生成吸附全流程工作区。", {"material": {"type": "string"}, "adsorbate": {"type": "string"}, "output_dir": {"type": "string"}}, False, ("material", "adsorbate", "output_dir")), self._adsorption_full_workflow)
        self._register(ToolSpec("transition_state_plan", "TS 任务规划。", {"prompt": {"type": "string"}, "material": {"type": "string"}}, True), self._transition_state_plan)
        self._register(ToolSpec("transition_state_dry_run", "TS dry-run。", {"prompt": {"type": "string"}, "material": {"type": "string"}}, True), self._transition_state_dry_run)
        self._register(ToolSpec("ts_midpoint_candidates_enumerate", "基于 IS/FS 线性插值生成 TS/NEB 中间构型初猜；不执行 NEB/Dimer。", {"initial_path": {"type": "string"}, "final_path": {"type": "string"}, "output_dir": {"type": "string"}, "n_images": {"type": "integer"}}, False, ("initial_path", "final_path", "output_dir")), self._ts_midpoint_candidates_enumerate)
        self._register(ToolSpec("convergence_plan_compose", "组合 ENCUT/KPOINTS 收敛性测试矩阵；只生成计划，不提交计算。", {"material": {"type": "string"}, "property_target": {"type": "string"}, "encut_values": {"type": "array", "items": {"type": "integer"}}, "kpoint_grids": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}, "force_threshold_ev_a": {"type": "number"}, "energy_tolerance_mev_atom": {"type": "number"}, "project": {"type": "string"}, "output_dir": {"type": "string"}}, False, ("material",)), self._convergence_plan_compose)
        self._register(ToolSpec("adsorption_eval_case_list", "列出小型文献先验 eval set，用于验证模型候选计划是否化学合理。", {}, True), self._adsorption_eval_case_list)
        self._register(ToolSpec("adsorption_eval_score_plan", "把 adsorption_candidate_plan 与 eval case 对比评分；只做行为评估，不代表 DFT 正确。", {"plan": {"type": "object"}, "case_id": {"type": "string"}, "material": {"type": "string"}, "adsorbate": {"type": "string"}}, True, ("plan",)), self._adsorption_eval_score_plan)
        self._register(ToolSpec("adsorption_eval_model_comparison_report", "生成 deepseek/qwen 模型候选计划评估对比 Markdown；无 live 结果时生成明确未运行模板。", {"output_path": {"type": "string"}, "model_results": {"type": "array", "items": {"type": "object"}}}, False, ("output_path",)), self._adsorption_eval_model_comparison_report)
        self._register(ToolSpec("ts_workflow_config", "读取 TS workflow 配置。", {}, True), self._ts_workflow_config)
        self._register(ToolSpec("neb_input_check", "检查 NEB 输入。", {"n_images": {"type": "integer"}}, True), self._neb_input_check)
        self._register(ToolSpec("dimer_input_check", "检查 Dimer 输入。", {"work_dir": {"type": "string"}}, True), self._dimer_input_check)
        self._register(ToolSpec("task_type_catalog", "列出任务类型。", {}, True), self._task_type_catalog)
        self._register(ToolSpec("dft_run_step", "执行单步 DFT 主线。", {"phase": {"type": "string"}}, False), self._dft_run_step)
        self._register(ToolSpec("dft_run_task", "创建并执行真实 DFT 任务。", {"prompt": {"type": "string"}, "project": {"type": "string"}, "material": {"type": "string"}, "structure_path": {"type": "string"}, "task_type": {"type": "string"}, "submit_profile": {"type": "string"}, "execution_mode": {"type": "string"}}, False, ("prompt",)), self._dft_run_task)
        self._register(ToolSpec("dft_run_report", "读取 run 报告。", {"run_id": {"type": "string"}, "run_root": {"type": "string"}}, True), self._dft_run_report)
        self._register(ToolSpec("dft_run_list", "列出 run。", {"limit": {"type": "integer"}}, True), self._dft_run_list)
        self._register(ToolSpec("vasp_output_scan", "扫描 VASP 输出。", {"run_root": {"type": "string"}}, True), self._vasp_output_scan)
        self._register(ToolSpec("vasp_input_summary", "总结 VASP 输入。", {"run_root": {"type": "string"}}, True), self._vasp_input_summary)
        self._register(ToolSpec("dft_task_plan", "创建 DFT task plan。", {"prompt": {"type": "string"}, "project": {"type": "string"}, "material": {"type": "string"}, "structure_path": {"type": "string"}, "task_type": {"type": "string"}}, False, ("prompt",)), self._dft_task_plan)
        self._register(ToolSpec("cluster_probe", "探测 SSH/SLURM 集群。", {}, True), self._cluster_probe)
        self._register(ToolSpec("cluster_config", "读取集群配置。", {}, True), self._cluster_config)
        self._register(ToolSpec("cluster_remote_submit", "通过 SSH/SLURM 远程提交已建好的 run。", {"run_root": {"type": "string"}, "run_id": {"type": "string"}}, False), self._cluster_remote_submit)
        self._register(ToolSpec("cluster_remote_monitor", "轮询远程 run 状态并在完成时同步输出。", {"run_root": {"type": "string"}, "run_id": {"type": "string"}, "sync_outputs": {"type": "boolean"}}, False), self._cluster_remote_monitor)
        self._register(ToolSpec("cluster_remote_fetch", "同步远程 run 输出到本地。", {"run_root": {"type": "string"}, "run_id": {"type": "string"}}, False), self._cluster_remote_fetch)
        self._register(ToolSpec("adsorption_workflow_status", "读取 adsorption workflow 状态。", {"run_root": {"type": "string"}}, True), self._adsorption_workflow_status)
        self._register(ToolSpec("recommend_next_tasks", "推荐下一步科研任务。", {"project": {"type": "string"}, "focus": {"type": "string"}}, True), self._recommend_next_tasks)

    def list_tools(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec, _ in self._tools.values()]

    def openai_tool_schemas(self) -> list[dict[str, Any]]:
        return [_schema(spec.name, spec.description, spec.parameters, list(spec.required)) for spec, _ in self._tools.values()]

    def run_tool(self, name: str, arguments: dict[str, Any] | str | None = None) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {}
        payload = arguments or {}
        if name not in self._tools:
            result = {"status": "error", "message": f"未知工具: {name}"}
        else:
            spec, handler = self._tools[name]
            allowed, reason = should_allow_tool(
                read_only=spec.read_only,
                mode=self.permission_mode,
                explicit_permission=bool(payload.pop("_permission_granted", False)),
            )
            if not allowed:
                result = {
                    "status": "permission_required",
                    "message": "当前为“需要用户同意”模式；此工具会修改状态/文件或产生副作用，必须先得到用户明确同意。",
                    "permission_mode": self.permission_mode,
                    "permission_label": permission_mode_label(self.permission_mode),
                    "tool": name,
                    "read_only": spec.read_only,
                    "reason": reason,
                }
                return {"name": name, "arguments": payload, "result": result}
            try:
                result = handler(payload)
            except Exception as exc:
                result = {"status": "error", "message": str(exc)}
        return {"name": name, "arguments": payload, "result": result}

    def _workflow_map(self, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "mainline": [
                {"step": 1, "title": "discussion -> plan", "tools": ["research_onboarding_context", "research_proposal_plan", "architecture_live_doc_snapshot", "architecture_live_doc_update", "project_state_read", "research_progress_append", "project_progress_append", "recommend_next_tasks"]},
                {"step": 2, "title": "structure -> model", "tools": ["structure_modeling_tool_status", "structure_modeling_intent_plan", "structure_convert", "structure_resolve", "structure_sanity_check", "structure_build_slab", "slab_surface_inspect", "adsorbate_chemistry_hint", "knowledge_search_for_system", "structure_enumerate_sites", "adsorption_candidate_plan", "structure_add_adsorbate", "candidate_quality_score", "structure_relax_short", "structure_defect", "defect_site_enumerate", "ts_midpoint_candidates_enumerate", "convergence_plan_compose", "adsorption_plan", "adsorption_build_slab", "adsorption_candidate_manifest_compose", "adsorption_candidates"]},
                {"step": 3, "title": "execute -> explain -> write_back", "tools": ["dft_run_task", "dft_run_report", "dft_run_list", "cluster_probe", "cluster_config", "cluster_remote_submit", "cluster_remote_monitor", "cluster_remote_fetch", "candidate_outcome_record", "knowledge_note_add", "knowledge_note_search", "knowledge_note_show"]},
            ],
            "workflow": [
                {"phase": "project_context"},
                {"phase": "structure_io"},
                {"phase": "adsorption_modeling"},
                {"phase": "dft_tasking"},
                {"phase": "cluster_execution"},
                {"phase": "parse"},
                {"phase": "knowledge_backflow"},
            ],
            "evaluation_tools": ["adsorption_eval_case_list", "adsorption_eval_score_plan"],
        }

    def _structure_modeling_intent_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        intent = str(payload.get("intent") or "").strip()
        if not intent:
            return {"status": "error", "message": "intent 不能为空。"}
        available = payload.get("available_inputs") or {}
        if not isinstance(available, dict):
            available = {}
        text = " ".join([intent, json.dumps(available, ensure_ascii=False)]).lower()

        def has_any(*needles: str) -> bool:
            return any(needle.lower() in text for needle in needles)

        if has_any("吸附", "adsorption", "adsorbate", "candidate", "候选", "h2o", "co2", "co ", "oh", "ooh"):
            task_type = "adsorption"
        elif has_any("slab", "surface", "表面", "晶面", "miller", "(111)", "(100)", "(110)"):
            task_type = "slab"
        elif has_any("缺陷", "vacancy", "空位", "dopant", "掺杂", "substitution", "替换"):
            task_type = "defect"
        elif has_any("neb", "ts", "过渡态", "transition state", "反应路径", "插值"):
            task_type = "ts_neb"
        elif has_any("收敛", "convergence", "encut", "kpoint", "k-point"):
            task_type = "convergence"
        elif has_any("转换", "convert", "poscar", "cif", "xsd", "格式"):
            task_type = "conversion"
        else:
            task_type = "unknown"

        project = str(payload.get("project") or "").strip() or None
        allow_writes = bool(payload.get("allow_writes", True))
        missing: list[str] = []
        groups: list[dict[str, Any]] = []
        quality_gates: list[str] = []

        def input_present(*keys: str) -> bool:
            return any(str(available.get(key) or "").strip() for key in keys)

        if task_type == "adsorption":
            if not input_present("adsorbate"):
                missing.append("adsorbate")
            if not input_present("slab_path", "structure_path", "material", "mp_id"):
                missing.append("slab_path_or_material_source")
            if allow_writes and not input_present("output_dir"):
                missing.append("output_dir")
            groups = [
                {
                    "purpose": "判断吸附物如何结合",
                    "candidate_tools": ["adsorbate_chemistry_hint"],
                    "call_when": "adsorbate 已知且需要选择 anchor / motif / 初始高度。",
                },
                {
                    "purpose": "查项目/体系先验",
                    "candidate_tools": ["knowledge_search_for_system"],
                    "call_when": "material 或 adsorbate 已知；命中为 prior，未命中也要记录 no project prior found。",
                },
                {
                    "purpose": "理解当前表面和可选位点",
                    "candidate_tools": ["structure_build_slab", "slab_surface_inspect", "structure_enumerate_sites"],
                    "call_when": "没有 slab_path 时先建 slab；有 slab_path 时优先 inspect/enumerate。",
                },
                {
                    "purpose": "模型写出少量有理由候选",
                    "candidate_tools": ["adsorption_candidate_plan"],
                    "call_when": "已有 anchor/motif、prior 状态和表面/位点证据；候选数量由 rationale 决定。",
                },
                {
                    "purpose": "生成并检查候选结构",
                    "candidate_tools": ["structure_add_adsorbate", "structure_sanity_check", "candidate_quality_score"],
                    "call_when": "用户要实际建模且 output_path 明确；warning/failed 不能说成成功。",
                },
                {
                    "purpose": "收编候选并沉淀知识",
                    "candidate_tools": ["adsorption_candidate_manifest_compose", "knowledge_note_add"],
                    "call_when": "已有 plan_id、候选 POSCAR、每个 candidate.reason 和质量检查结果。",
                },
            ]
            quality_gates = [
                "compose 前必须有 adsorption_candidate_plan.plan_id。",
                "每个 candidate 需要 site_label、orientation、anchor_symbol、reason、POSCAR 路径。",
                "adsorption_candidates 是 fallback_only，不作为默认主路径。",
            ]
        elif task_type == "slab":
            if not input_present("material", "structure_path", "mp_id"):
                missing.append("material_or_structure_source")
            if not input_present("miller_index"):
                missing.append("miller_index")
            if allow_writes and not input_present("output_dir"):
                missing.append("output_dir")
            groups = [
                {"purpose": "确认材料来源", "candidate_tools": ["structure_resolve"], "call_when": "来源可能是本地结构、mp-id 或 ASE bulk。"},
                {"purpose": "构建表面模型", "candidate_tools": ["structure_build_slab"], "call_when": "miller/supercell/vacuum/fixed layer 已明确。"},
                {"purpose": "检查表面边界", "candidate_tools": ["slab_surface_inspect", "structure_sanity_check"], "call_when": "slab 生成后立即检查真空、原子数、表面对称。"},
            ]
            quality_gates = ["不要默认 (111)；除非用户给出或材料常识明确且要说明。", "固定层和真空厚度要写入回答。"]
        elif task_type == "defect":
            if not input_present("structure_path", "slab_path"):
                missing.append("structure_path")
            if not input_present("species"):
                missing.append("species")
            if allow_writes and not input_present("output_path"):
                missing.append("output_path")
            groups = [
                {"purpose": "枚举可操作位点", "candidate_tools": ["defect_site_enumerate"], "call_when": "需要选择 vacancy/substitution 原子 index。"},
                {"purpose": "生成缺陷结构", "candidate_tools": ["structure_defect"], "call_when": "已说明 atom_index 选择依据；不能默认第一个原子。"},
                {"purpose": "检查缺陷结构", "candidate_tools": ["structure_sanity_check"], "call_when": "写出 POSCAR 后检查最短距离和物种。"},
            ]
            quality_gates = ["必须解释 index 选择依据。", "surface_only 与体相缺陷要区分。"]
        elif task_type == "ts_neb":
            missing.extend([key for key in ("initial_path", "final_path", "output_dir") if not input_present(key)])
            groups = [
                {"purpose": "检查 IS/FS 是否可插值", "candidate_tools": ["neb_input_check"], "call_when": "有初态/终态路径时先检查原子数和元素顺序。"},
                {"purpose": "生成 NEB/TS 初猜", "candidate_tools": ["ts_midpoint_candidates_enumerate"], "call_when": "检查通过后才插值；这不是 TS 结果。"},
            ]
            quality_gates = ["只声称生成初猜，不声称找到过渡态。"]
        elif task_type == "convergence":
            missing.extend([key for key in ("target_property", "tolerance") if not input_present(key)])
            groups = [
                {"purpose": "生成收敛测试矩阵", "candidate_tools": ["convergence_plan_compose"], "call_when": "目标性质、误差阈值和预算明确。"},
            ]
            quality_gates = ["只生成计划，不提交作业或声称已收敛。"]
        elif task_type == "conversion":
            missing.extend([key for key in ("input_path", "output_path") if not input_present(key)])
            groups = [
                {"purpose": "读取/检查输入结构", "candidate_tools": ["structure_resolve", "structure_sanity_check"], "call_when": "先确认输入存在且可解析。"},
                {"purpose": "转换格式", "candidate_tools": ["structure_convert"], "call_when": "输出路径和目标格式明确。"},
            ]
            quality_gates = ["转换后应再 sanity_check，不能只说文件已写。"]
        else:
            groups = [
                {"purpose": "先识别建模任务类型", "candidate_tools": ["structure_modeling_tool_status", "research_proposal_plan"], "call_when": "用户意图不清或缺少结构来源/目标。"},
            ]
            quality_gates = ["先追问或读取项目上下文；不要猜测并写结构。"]

        return {
            "status": "ok",
            "task_type": task_type,
            "project": project,
            "principle": "这是工具选择导航，不是固定程序；模型应根据已有证据跳过不必要工具，并解释取舍。",
            "available_inputs": available,
            "missing_inputs": missing,
            "tool_groups": groups,
            "quality_gates": quality_gates,
            "stop_conditions": [
                "关键输入缺失且无法从项目状态/文件系统推断时，先说明缺口。",
                "工具返回 warning/failed/unavailable 时，先修正建模假设或降低声明强度。",
                "用户只要讨论/规划时，不要写结构文件。",
            ],
        }

    def _structure_modeling_tool_status(self, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "principle": "Step 2 提供结构建模原语；模型根据科研意图选择工具并写明判断，不执行固定流水线。",
            "completion": {
                "structure_io": "ready",
                "slab_build_and_inspect": "ready",
                "adsorption_model_authored_candidates": "ready",
                "candidate_quality_and_short_relax": "ready",
                "defect_primitives": "ready_minimal",
                "ts_midpoint_primitives": "ready_minimal",
                "convergence_plan_primitives": "ready_minimal",
                "black_box_adsorption_baseline": "fallback_only",
            },
            "decision_matrix": [
                {
                    "intent": "已有结构读取/格式转换",
                    "evidence": ["input_path/source", "format", "sanity boundary"],
                    "tools": ["structure_modeling_intent_plan", "structure_resolve", "structure_convert", "structure_sanity_check"],
                    "not_a_fixed_program": "只在需要转换或检查时调用；不要为聊天式讨论无意义写文件。",
                },
                {
                    "intent": "slab 建模",
                    "evidence": ["material or structure_path/mp_id", "miller_index", "vacuum/fixed layers"],
                    "tools": ["structure_build_slab", "slab_surface_inspect", "structure_sanity_check"],
                    "not_a_fixed_program": "slab_surface_inspect 用于解释表面环境，不替模型决定研究方向。",
                },
                {
                    "intent": "吸附候选",
                    "evidence": ["adsorbate anchor/motif", "system prior", "surface symmetry/coordination"],
                    "tools": [
                        "structure_modeling_intent_plan",
                        "adsorbate_chemistry_hint",
                        "knowledge_search_for_system",
                        "slab_surface_inspect",
                        "structure_enumerate_sites",
                        "adsorption_candidate_plan",
                        "structure_add_adsorbate",
                        "candidate_quality_score",
                        "adsorption_candidate_manifest_compose",
                    ],
                    "not_a_fixed_program": "这些是证据门槛；候选数量、位点、取向由 plan.rationale 决定，adsorption_candidates 只作兜底。",
                },
                {
                    "intent": "缺陷/掺杂",
                    "evidence": ["candidate atom_index", "surface_only or bulk", "vacancy/substitution reason"],
                    "tools": ["defect_site_enumerate", "structure_defect", "structure_sanity_check"],
                    "not_a_fixed_program": "不要默认删除第一个原子；必须解释 atom_index 选择依据。",
                },
                {
                    "intent": "TS/NEB 初猜",
                    "evidence": ["IS/FS atom count", "element order", "reaction coordinate rationale"],
                    "tools": ["ts_midpoint_candidates_enumerate", "neb_input_check"],
                    "not_a_fixed_program": "插值只是初猜，不声称找到 TS。",
                },
                {
                    "intent": "收敛性测试",
                    "evidence": ["target property", "tolerance", "budget"],
                    "tools": ["convergence_plan_compose"],
                    "not_a_fixed_program": "只生成测试矩阵，不提交或声称收敛。",
                },
            ],
            "quality_gates": [
                "写结构前确认输入/输出路径和科研目标。",
                "compose manifest 前必须有 adsorption_candidate_plan.plan_id。",
                "warning/failed/unavailable 不能包装成成功。",
                "候选结构必须带 reason、sanity/quality 结果和下一步建议。",
            ],
        }

    def _research_onboarding_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        return read_research_onboarding_context(
            str(payload.get("project") or "").strip() or None,
            max_chars=int(payload.get("max_chars") or 14000),
        )

    def _research_proposal_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        return build_research_proposal(
            str(payload.get("prompt") or ""),
            project=str(payload.get("project") or "").strip() or None,
        )

    def _research_progress_append(self, payload: dict[str, Any]) -> dict[str, Any]:
        return append_research_progress(
            str(payload.get("project") or "").strip(),
            completed=[str(item) for item in payload.get("completed") or []],
            blockers=[str(item) for item in payload.get("blockers") or []],
            next_steps=[str(item) for item in payload.get("next_steps") or []],
        )

    def _project_state_read(self, payload: dict[str, Any]) -> dict[str, Any]:
        from aether_dft.project_state import project_paths, read_project_context, read_project_context_digest

        project = str(payload.get("project") or "").strip()
        if not project:
            return {"status": "error", "message": "缺少 project。"}
        max_chars = int(payload.get("max_chars") or 8000)
        paths = project_paths(project)
        context = read_project_context(project, max_chars=max_chars)
        research = read_research_onboarding_context(project, max_chars=max_chars)
        if research.get("context"):
            context = (context + "\n\n## Research workspace onboarding\n" + str(research["context"])).strip()
        return {
            "status": "ok",
            "project": paths.slug,
            "state_md_path": str(paths.state_md),
            "progress_path": str(paths.progress),
            "current_state_path": str(paths.state),
            "state_md_exists": paths.state_md.exists(),
            "context": context,
            "digest": read_project_context_digest(project),
            "research_onboarding": {
                "project_found": research.get("project_found"),
                "files_read": research.get("files_read"),
                "available_projects": research.get("available_projects"),
            },
        }

    def _project_progress_append(self, payload: dict[str, Any]) -> dict[str, Any]:
        project = str(payload.get("project") or "").strip()
        if not project:
            return {"status": "error", "message": "缺少 project。"}
        path = append_progress(
            project,
            completed=[str(item) for item in payload.get("completed") or []],
            blockers=[str(item) for item in payload.get("blockers") or []],
            next_steps=[str(item) for item in payload.get("next_steps") or []],
        )
        return {"status": "ok", "project": project, "progress_path": str(path), "state_md_path": str(project_paths(project).state_md)}

    def _knowledge_note_add(self, payload: dict[str, Any]) -> dict[str, Any]:
        project = str(payload.get("project") or "").strip()
        if not project:
            return {"status": "error", "message": "缺少 project。"}
        note = add_note(
            project,
            str(payload.get("title") or "").strip(),
            str(payload.get("content") or "").strip(),
            tags=[str(item) for item in payload.get("tags") or []],
        )
        return {"status": "ok", "note": note.to_dict()}

    def _knowledge_note_list(self, payload: dict[str, Any]) -> dict[str, Any]:
        project = str(payload.get("project") or "").strip()
        if not project:
            return {"status": "error", "message": "缺少 project。"}
        return {"status": "ok", "notes": list_notes(project)}

    def _knowledge_note_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        project = str(payload.get("project") or "").strip()
        query = str(payload.get("query") or "").strip()
        if not project:
            return {"status": "error", "message": "缺少 project。"}
        return {"status": "ok", "matches": search_notes(project, query)}

    def _knowledge_note_show(self, payload: dict[str, Any]) -> dict[str, Any]:
        note = str(payload.get("note") or "").strip()
        if not note:
            return {"status": "error", "message": "缺少 note。"}
        project = str(payload.get("project") or "").strip() or None
        return {"status": "ok", "note": show_note(note, project=project)}

    def _architecture_live_doc_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        max_chars = int(payload.get("max_chars") or 2400)
        snapshot = load_architecture_live_doc_snapshot(max_chars=max_chars)
        return {"status": "ok", "snapshot": snapshot}

    def _architecture_live_doc_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        title = str(payload.get("title") or "").strip()
        content = str(payload.get("content") or "").strip()
        from aether_dft.paths import PROJECT_ROOT

        path = PROJECT_ROOT / "智能体架构.md"
        existing = path.read_text(encoding="utf-8") if path.exists() else "# AETHER-DFT 智能体架构\n\n"
        block = f"\n\n## {title or '未命名块'}\n\n{content}\n"
        path.write_text(existing.rstrip() + block, encoding="utf-8")
        return {"status": "ok", "path": str(path)}

    def _structure_convert(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = convert_structure(payload["input_path"], payload["output_path"], fmt=payload.get("fmt"))
        return {"status": "ok", "result": _jsonable(result)}

    def _structure_resolve(self, payload: dict[str, Any]) -> dict[str, Any]:
        return resolve_structure(
            input_path=payload.get("input_path"),
            material=payload.get("material"),
            mp_id=payload.get("mp_id"),
            output_path=None,
            source=str(payload.get("source") or "auto"),
        )

    def _structure_supercell(self, payload: dict[str, Any]) -> dict[str, Any]:
        return make_supercell(payload["input_path"], payload["output_path"], payload.get("scaling_matrix") or [1, 1, 1])

    def _structure_build_slab(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = build_adsorption_slab(
            material=payload["material"],
            output_dir=payload["output_dir"],
            structure_path=payload.get("structure_path"),
            mp_id=payload.get("mp_id"),
            source=str(payload.get("source") or "auto"),
            miller_index=payload.get("miller_index"),
            supercell=payload.get("supercell") or [2, 2, 1],
            min_slab_size=float(payload.get("min_slab_size") or 8.0),
            min_vacuum_size=float(payload.get("min_vacuum_size") or 12.0),
            fixed_bottom_layers=int(payload.get("fixed_bottom_layers") or 2),
            center_slab=bool(payload.get("center_slab", True)),
        )
        return {"status": "ok", "result": result.to_dict()}

    def _structure_add_adsorbate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return add_adsorbate(
            slab_path=payload["slab_path"],
            adsorbate=payload["adsorbate"],
            output_path=payload["output_path"],
            height=float(payload.get("height") or 2.0),
            site_index=payload.get("site_index"),
            cart_coords=payload.get("cart_coords"),
            orientation=str(payload.get("orientation") or "upright"),
            anchor_atom_index=payload.get("anchor_atom_index"),
            anchor_symbol=payload.get("anchor_symbol"),
            coords_mode=str(payload.get("coords_mode") or "cartesian"),
            fixed_bottom_layers=int(payload.get("fixed_bottom_layers") if payload.get("fixed_bottom_layers") is not None else 2),
        )

    def _structure_enumerate_sites(self, payload: dict[str, Any]) -> dict[str, Any]:
        return enumerate_adsorption_sites(
            payload["slab_path"],
            max_sites_per_family=int(payload.get("max_sites_per_family") or 4),
            top_layer_tolerance=float(payload.get("top_layer_tolerance") or 0.75),
            nearest_neighbors=int(payload.get("nearest_neighbors") or 3),
        )

    def _candidate_quality_score(self, payload: dict[str, Any]) -> dict[str, Any]:
        return candidate_quality_score(
            slab_path=payload["slab_path"],
            candidate_path=payload["candidate_path"],
            adsorbate=str(payload.get("adsorbate") or "").strip() or None,
            anchor_symbol=str(payload.get("anchor_symbol") or "").strip() or None,
            top_layer_tolerance=float(payload.get("top_layer_tolerance") or 0.75),
            min_anchor_surface_distance=float(payload.get("min_anchor_surface_distance") or 1.1),
            max_anchor_surface_distance=float(payload.get("max_anchor_surface_distance") or 3.5),
            min_adsorbate_slab_distance=float(payload.get("min_adsorbate_slab_distance") or 0.65),
        )

    def _structure_relax_short(self, payload: dict[str, Any]) -> dict[str, Any]:
        return structure_relax_short(
            input_path=payload["input_path"],
            output_path=payload["output_path"],
            calculator=str(payload.get("calculator") or "emt"),
            max_steps=int(payload.get("max_steps") or 20),
            fmax=float(payload.get("fmax") or 0.2),
            trajectory_path=str(payload.get("trajectory_path") or "").strip() or None,
        )

    def _slab_surface_inspect(self, payload: dict[str, Any]) -> dict[str, Any]:
        return inspect_slab_surface(
            payload["slab_path"],
            top_layer_tolerance=float(payload.get("top_layer_tolerance") or 0.75),
            second_layer_tolerance=float(payload.get("second_layer_tolerance") or 2.5),
            neighbor_radius=float(payload.get("neighbor_radius") or 3.5),
            symprec=float(payload.get("symprec") or 0.05),
        )

    def _adsorbate_chemistry_hint(self, payload: dict[str, Any]) -> dict[str, Any]:
        return get_adsorbate_chemistry_hint(str(payload.get("adsorbate") or "").strip())

    def _knowledge_search_for_system(self, payload: dict[str, Any]) -> dict[str, Any]:
        material = str(payload.get("material") or "").strip() or None
        adsorbate = str(payload.get("adsorbate") or "").strip() or None
        extra_terms_raw = payload.get("extra_terms") or []
        extra_terms = [str(term) for term in extra_terms_raw if str(term).strip()]
        project_priority = str(payload.get("project_priority") or "").strip() or None
        if not (material or adsorbate or extra_terms):
            return {"status": "error", "message": "至少提供 material、adsorbate 或 extra_terms 之一。"}
        return search_for_system(
            material=material,
            adsorbate=adsorbate,
            extra_terms=extra_terms,
            project_priority=project_priority,
            max_results=int(payload.get("max_results") or 12),
        )

    def _structure_defect(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode") or "").strip().lower()
        if mode in {"vacancy", "remove"}:
            return self._structure_add_vacancy(payload)
        if mode in {"substitution", "dopant", "dope"}:
            return self._structure_add_dopant(payload)
        return {"status": "error", "message": "mode 必须是 vacancy 或 substitution。"}

    def _defect_site_enumerate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return enumerate_defect_sites(
            structure_path=payload["structure_path"],
            species=str(payload.get("species") or "").strip() or None,
            surface_only=bool(payload.get("surface_only", True)),
            top_layer_tolerance=float(payload.get("top_layer_tolerance") or 0.75),
            max_sites=int(payload.get("max_sites") or 12),
        )

    def _structure_add_vacancy(self, payload: dict[str, Any]) -> dict[str, Any]:
        return add_vacancy(
            input_path=payload["input_path"],
            output_path=payload["output_path"],
            species=payload["species"],
            index=payload.get("index"),
            surface_only=bool(payload.get("surface_only", True)),
        )

    def _structure_add_dopant(self, payload: dict[str, Any]) -> dict[str, Any]:
        return add_dopant(
            input_path=payload["input_path"],
            output_path=payload["output_path"],
            dopant=payload["dopant"],
            species=payload.get("species"),
            index=payload.get("index"),
            surface_only=bool(payload.get("surface_only", False)),
        )

    def _structure_sanity_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        return sanity_check(
            payload["structure_path"],
            min_distance=float(payload.get("min_distance") or 0.65),
            min_vacuum=float(payload.get("min_vacuum") or 6.0),
            vacuum_axis=str(payload.get("vacuum_axis") or "c"),
        )

    def _structure_bond_analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        atoms = read(payload["structure_path"])
        report = analyze_bonds(atoms)
        return {"status": "ok", "report": _jsonable(report)}

    def _structure_displacement_compare(self, payload: dict[str, Any]) -> dict[str, Any]:
        report = compare_structures(payload["initial_path"], payload["final_path"], top_n=int(payload.get("top_n") or 10))
        return {"status": "ok", "report": _jsonable(report)}

    def _adsorption_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan = plan_adsorption_task(
            payload["prompt"],
            project=payload.get("project"),
            adsorbate=payload.get("adsorbate"),
            material=payload.get("material"),
            slab_path=payload.get("slab_path"),
            preferred_site=payload.get("preferred_site"),
            preferred_orientation=payload.get("preferred_orientation"),
            persist=bool(payload.get("persist", False)),
        )
        return {"status": "ok", "plan": plan.to_dict()}

    def _adsorption_build_slab(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = build_adsorption_slab(
            material=payload["material"],
            output_dir=payload["output_dir"],
            structure_path=payload.get("structure_path"),
            mp_id=payload.get("mp_id"),
            source=str(payload.get("source") or "auto"),
            miller_index=payload.get("miller_index"),
            supercell=payload.get("supercell") or [2, 2, 1],
            min_slab_size=float(payload.get("min_slab_size") or 8.0),
            min_vacuum_size=float(payload.get("min_vacuum_size") or 12.0),
            fixed_bottom_layers=int(payload.get("fixed_bottom_layers") or 2),
        )
        return {"status": "ok", "result": result.to_dict()}

    def _adsorption_candidates(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip() or f"{payload['adsorbate']} adsorption on {payload['material']}"
        result = generate_adsorption_candidates(
            slab_path=payload["slab_path"],
            adsorbate=payload["adsorbate"],
            material=payload["material"],
            prompt=prompt,
            project=payload.get("project"),
            output_dir=payload.get("output_dir"),
            task_id=payload.get("task_id"),
            candidate_height=float(payload.get("candidate_height") or 2.1),
            max_sites_per_family=int(payload.get("max_sites_per_family") or 2),
            preferred_site=payload.get("preferred_site"),
            preferred_orientation=payload.get("preferred_orientation"),
            vacancy_species=payload.get("vacancy_species"),
        )
        return {
            "status": result.get("status", "ok"),
            "result": result.get("result") or {},
            "task_id": result.get("task_id"),
            "output_dir": result.get("output_dir"),
            "exit_code": result.get("exit_code"),
            "stdout": result.get("stdout"),
            "next_research_tasks": result.get("next_research_tasks", []),
        }

    def _adsorption_candidate_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan = create_candidate_plan(
            material=str(payload.get("material") or ""),
            adsorbate=str(payload.get("adsorbate") or ""),
            rationale=str(payload.get("rationale") or ""),
            expected_binding_motif=str(payload.get("expected_binding_motif") or ""),
            anchor_atom=str(payload.get("anchor_atom") or ""),
            target_sites=payload.get("target_sites") or [],
            target_orientations=payload.get("target_orientations") or [],
            excluded_sites_with_reason=payload.get("excluded_sites_with_reason"),
            symmetry_pruning_applied=bool(payload.get("symmetry_pruning_applied", False)),
            priors_consulted=payload.get("priors_consulted") or {},
            project=str(payload.get("project") or "").strip() or None,
            task_id=str(payload.get("task_id") or "").strip() or None,
            notes=str(payload.get("notes") or ""),
        )
        return {
            "status": "ok",
            "plan_id": plan.plan_id,
            "plan_path": plan.plan_path,
            "plan": plan.to_dict(),
            "guidance": (
                "把 plan_id 传给 adsorption_candidate_manifest_compose；"
                "candidates 的 site_label 必须与 plan.target_sites[*].site_id 对齐；"
                "每个 candidate.reason ≥ 20 字，写明科学依据。"
            ),
        }

    def _adsorption_candidate_plan_list(self, payload: dict[str, Any]) -> dict[str, Any]:
        project = str(payload.get("project") or "").strip() or None
        return {"status": "ok", "plans": list_candidate_plans(project)}

    def _adsorption_candidate_manifest_compose(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidates = payload.get("candidates") or []
        if not isinstance(candidates, list):
            return {"status": "error", "message": "candidates 必须是数组。"}
        plan_id = str(payload.get("plan_id") or "").strip()
        if not plan_id:
            return {
                "status": "error",
                "message": "compose 必须传 plan_id；先调 adsorption_candidate_plan 创建结构化推理。",
            }
        project = str(payload.get("project") or "").strip() or None
        try:
            plan = load_candidate_plan(plan_id, project=project)
        except FileNotFoundError as exc:
            return {"status": "error", "message": str(exc)}
        result = compose_manifest_from_authored_candidates(
            task_id=str(payload["task_id"]),
            material_name=str(payload["material_name"]),
            source_prompt=str(payload["source_prompt"]),
            slab_source=str(payload["slab_source"]),
            adsorbate_source=str(payload["adsorbate_source"]),
            output_dir=str(payload["output_dir"]),
            candidates=[dict(item) for item in candidates],
            metadata=payload.get("metadata") or {},
            plan_payload=plan.to_dict(),
            prune_rationale=str(payload.get("prune_rationale") or "").strip() or None,
        )
        return result

    def _candidate_outcome_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        energy = payload.get("adsorption_energy_ev")
        return record_candidate_outcome(
            project=str(payload.get("project") or ""),
            material=str(payload.get("material") or ""),
            adsorbate=str(payload.get("adsorbate") or ""),
            candidate_id=str(payload.get("candidate_id") or ""),
            verdict=str(payload.get("verdict") or ""),
            adsorption_energy_ev=float(energy) if energy is not None else None,
            initial_path=str(payload.get("initial_path") or "").strip() or None,
            final_path=str(payload.get("final_path") or "").strip() or None,
            manifest_path=str(payload.get("manifest_path") or "").strip() or None,
            calculation_summary=str(payload.get("calculation_summary") or "").strip() or None,
            notes=str(payload.get("notes") or "").strip() or None,
        )

    def _adsorption_full_workflow(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = run_adsorption_full_workflow(
            material=payload["material"],
            adsorbate=payload["adsorbate"],
            output_dir=payload["output_dir"],
            prompt=payload.get("prompt"),
            project=payload.get("project"),
            structure_path=payload.get("structure_path"),
            mp_id=payload.get("mp_id"),
            source=str(payload.get("source") or "auto"),
        )
        return {"status": "ok", "result": result}

    def _transition_state_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        material = str(payload.get("material") or "").strip()
        task = {
            "plan": {"experiment_type": "transition_state_search", "prompt": prompt, "material": material},
            "dft_command": ["aether-dft", "dft", "run", prompt, "--task-type", "transition_state_search", "--dry-run"],
        }
        return {"status": "ok", "task": task}

    def _transition_state_dry_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._transition_state_plan(payload)
        result["dry_run"] = True
        return result

    def _ts_midpoint_candidates_enumerate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return interpolate_ts_midpoint_candidates(
            initial_path=payload["initial_path"],
            final_path=payload["final_path"],
            output_dir=payload["output_dir"],
            n_images=int(payload.get("n_images") or 3),
        )

    def _convergence_plan_compose(self, payload: dict[str, Any]) -> dict[str, Any]:
        return compose_convergence_plan(
            material=str(payload.get("material") or ""),
            property_target=str(payload.get("property_target") or "energy"),
            encut_values=[int(v) for v in payload.get("encut_values") or []] or None,
            kpoint_grids=[[int(v) for v in grid] for grid in payload.get("kpoint_grids") or []] or None,
            force_threshold_ev_a=float(payload.get("force_threshold_ev_a") or 0.03),
            energy_tolerance_mev_atom=float(payload.get("energy_tolerance_mev_atom") or 5.0),
            project=str(payload.get("project") or "").strip() or None,
            output_dir=str(payload.get("output_dir") or "").strip() or None,
        )

    def _adsorption_eval_case_list(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "cases": list_adsorption_eval_cases()}

    def _adsorption_eval_score_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        return score_adsorption_plan_against_eval(
            dict(payload.get("plan") or {}),
            case_id=str(payload.get("case_id") or "").strip() or None,
            material=str(payload.get("material") or "").strip() or None,
            adsorbate=str(payload.get("adsorbate") or "").strip() or None,
        )

    def _adsorption_eval_model_comparison_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        return render_model_comparison_report(
            output_path=payload["output_path"],
            model_results=[dict(item) for item in payload.get("model_results") or []],
        )

    def _ts_workflow_config(self, _: dict[str, Any]) -> dict[str, Any]:
        cfg = load_workflow_config()
        return {"status": "ok", "config": cfg, "boundary": "不会假装已完成 TS / NEB / Dimer；这里只返回配置和边界。"}

    def _neb_input_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        missing = []
        for key in ("initial_path", "final_path"):
            if not payload.get(key):
                missing.append(key)
        return {"status": "needs_inputs" if missing else "ok", "missing": missing, "boundary": "不执行 MACE / NEB，只检查输入是否齐全。"}

    def _dimer_input_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        missing = []
        work_dir = str(payload.get("work_dir") or "").strip()
        if not work_dir:
            missing.append("work_dir")
        else:
            path = Path(work_dir)
            for name in ("POSCAR", "MODECAR"):
                if not (path / name).exists():
                    missing.append(name)
        return {"status": "needs_inputs" if missing else "ok", "missing": missing, "boundary": "不执行远程提交，只检查 Dimer 输入。"}

    def _task_type_catalog(self, _: dict[str, Any]) -> dict[str, Any]:
        from dft_app.models import TaskType

        return {"status": "ok", "task_types": [{"task_type": item.value} for item in TaskType]}

    def _dft_run_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        phase = str(payload.get("phase") or "").strip()
        if not phase:
            return {
                "status": "needs_inputs",
                "missing": ["phase"],
                "message": "dft_run_step 只是兼容入口，不会伪造执行；要走真实主线请提供完整任务信息并调用 run_dft_task / CLI main。",
                "required_inputs": ["prompt", "material", "structure_path", "task_type"],
            }
        return {
            "status": "needs_inputs",
            "phase": phase,
            "message": f"dft_run_step 不会伪造 `{phase}` 执行；请用真实任务参数走 run_dft_task / CLI main。",
            "required_inputs": ["prompt", "material", "structure_path", "task_type"],
            "supported_phases": ["build", "dry_run", "fetch", "monitor", "parse", "remote_submit", "submit"],
        }

    def _load_run_bundle(self, payload: dict[str, Any]) -> tuple[Any, Any, Any, Path] | dict[str, Any]:
        from dft_app.storage import RecordStore

        store = RecordStore(Path.cwd())
        try:
            run_root = store.resolve_run_root(run_root=payload.get("run_root"), run_id=payload.get("run_id"))
            spec = store.load_experiment_spec(run_root)
            record = store.load_run_record(run_root)
            return store, spec, record, run_root
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def _dft_run_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = run_dft_task(
            str(payload.get("prompt") or ""),
            project=str(payload.get("project") or "").strip() or None,
            material=str(payload.get("material") or "").strip() or None,
            structure_path=str(payload.get("structure_path") or "").strip() or None,
            task_type=str(payload.get("task_type") or "").strip() or None,
            submit_profile=str(payload.get("submit_profile") or "").strip() or None,
            execution_mode=str(payload.get("execution_mode") or "dry_run").strip() or "dry_run",
        )
        return {"status": result.get("status", "ok"), "result": result}

    def _dft_run_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        from dft_app.storage import RecordStore

        store = RecordStore(Path.cwd())
        try:
            run_root = store.resolve_run_root(run_root=payload.get("run_root"), run_id=payload.get("run_id"))
            record = store.load_run_record(run_root)
            return {"status": "ok", "run": record.to_dict()}
        except Exception as exc:
            return {"status": "failed", "message": str(exc)}

    def _dft_run_list(self, payload: dict[str, Any]) -> dict[str, Any]:
        from dft_app.storage import RecordStore

        try:
            runs = RecordStore(Path.cwd()).list_runs(limit=int(payload.get("limit") or 20))
            return {"status": "ok", "runs": runs}
        except Exception as exc:
            return {"status": "failed", "message": str(exc), "runs": []}

    def _vasp_output_scan(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_root = Path(payload["run_root"])
        outcar = run_root / "OUTCAR"
        oszicar = run_root / "OSZICAR"
        if not run_root.exists():
            return {
                "status": "missing",
                "run_root": str(run_root),
                "outcar": {"exists": False, "last_toten": None, "has_required_accuracy": False},
                "oszicar_exists": False,
                "message": "run_root 不存在，无法判断 VASP 输出状态。",
            }
        text = outcar.read_text(encoding="utf-8", errors="replace") if outcar.exists() else ""
        toten_matches = re.findall(r"TOTEN\s*=\s*([-0-9.]+)", text)
        last_toten = float(toten_matches[-1]) if toten_matches else None
        has_required_accuracy = "reached required accuracy" in text.lower()
        has_energy = last_toten is not None
        if not outcar.exists():
            status = "missing"
        elif has_required_accuracy and has_energy:
            status = "completed"
        else:
            status = "incomplete"
        return {
            "status": status,
            "run_root": str(run_root),
            "outcar": {
                "exists": outcar.exists(),
                "last_toten": last_toten,
                "has_required_accuracy": has_required_accuracy,
                "has_energy": has_energy,
            },
            "oszicar_exists": oszicar.exists(),
        }

    def _vasp_input_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_root = Path(payload["run_root"])
        incar = run_root / "INCAR"
        poscar = run_root / "POSCAR"
        incar_data: dict[str, str] = {}
        if incar.exists():
            for line in incar.read_text(encoding="utf-8", errors="replace").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    incar_data[key.strip()] = value.strip()
        poscar_data = {"n_sites": 0}
        if poscar.exists():
            try:
                poscar_data["n_sites"] = len(read(poscar))
            except Exception:
                pass
        return {"status": "ok", "incar": incar_data, "poscar": poscar_data}

    def _dft_task_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        envelope = create_task_plan(payload["prompt"], project=payload.get("project"), material=payload.get("material"), structure_path=payload.get("structure_path"), task_type=payload.get("task_type"))
        return {"status": "ok", "task": envelope.to_dict()}

    def _cluster_probe(self, _: dict[str, Any]) -> dict[str, Any]:
        result = SSHRemoteRunner().probe()
        return {"status": result.status, "message": result.message, "details": result.details}

    def _cluster_config(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "config": SSHRemoteRunner().describe_config()}

    def _cluster_remote_submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.allow_cluster_submit:
            return {"status": "blocked", "message": "当前运行未启用 allow_cluster_submit。"}
        bundle = self._load_run_bundle(payload)
        if isinstance(bundle, dict):
            return bundle
        store, spec, record, _run_root = bundle
        result = SSHRemoteRunner().submit(spec, record)
        store.save_run_record(record)
        return {"status": result.status, "message": result.message, "details": result.details}

    def _cluster_remote_monitor(self, payload: dict[str, Any]) -> dict[str, Any]:
        bundle = self._load_run_bundle(payload)
        if isinstance(bundle, dict):
            return bundle
        store, _spec, record, _run_root = bundle
        sync_outputs = bool(payload.get("sync_outputs", True))
        result = SSHRemoteRunner().monitor(record, sync_outputs=sync_outputs)
        store.save_run_record(record)
        return {"status": result.status, "message": result.message, "details": result.details}

    def _cluster_remote_fetch(self, payload: dict[str, Any]) -> dict[str, Any]:
        bundle = self._load_run_bundle(payload)
        if isinstance(bundle, dict):
            return bundle
        store, _spec, record, _run_root = bundle
        result = SSHRemoteRunner().fetch_outputs(record)
        store.save_run_record(record)
        return {"status": result.status, "message": result.message, "details": result.details}

    def _adsorption_workflow_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "result": collect_adsorption_workflow_status(Path(payload["run_root"]))}

    def _recommend_next_tasks(self, payload: dict[str, Any]) -> dict[str, Any]:
        project = str(payload.get("project") or "").strip() or None
        focus = str(payload.get("focus") or "").strip() or None
        return {"status": "ok", "recommendations": recommend_next_tasks(project, focus=focus)}


def list_registered_tools() -> list[dict[str, Any]]:
    return ToolRegistry().list_tools()
