from __future__ import annotations

import hashlib
from typing import Any

from .paths import PROJECT_ROOT


_TASK_ALIASES = {
    "freq": "vibrational_frequency",
    "frequency": "vibrational_frequency",
    "vibrational": "vibrational_frequency",
    "vibrational_frequency": "vibrational_frequency",
    "zpe": "vibrational_frequency",
    "free_energy": "vibrational_frequency",
    "ts": "transition_state_search",
    "dimer": "transition_state_search",
    "transition_state": "transition_state_search",
    "transition_state_search": "transition_state_search",
    "relax": "relax",
    "geometry_optimization": "relax",
    "optimization": "relax",
    "opt": "relax",
    "single": "single_point",
    "single_point": "single_point",
    "scf": "single_point",
}


def normalize_research_task_type(task_type: str | None, prompt: str = "") -> str | None:
    raw = (task_type or "").strip().lower()
    if raw:
        return _TASK_ALIASES.get(raw, raw)
    text = prompt.lower()
    if any(token in text for token in ("freq", "frequency", "频率", "zpe", "自由能")):
        return "vibrational_frequency"
    if any(token in text for token in ("dimer", "ts", "过渡态")):
        return "transition_state_search"
    if any(token in text for token in ("relax", "优化", "结构优化")):
        return "relax"
    if any(token in text for token in ("scf", "single", "单点", "静态")):
        return "single_point"
    return raw or None


def _source(label: str, *parts: str) -> dict[str, Any]:
    path = PROJECT_ROOT.joinpath(*parts)
    return _source_from_path(label, path)


def _source_from_path(label: str, path: Any) -> dict[str, Any]:
    source_path = PROJECT_ROOT.joinpath(path) if isinstance(path, str) else path
    payload: dict[str, Any] = {
        "label": label,
        "path": str(source_path),
        "exists": source_path.exists(),
    }
    if source_path.exists() and source_path.is_file():
        raw = source_path.read_bytes()
        payload["sha256"] = hashlib.sha256(raw).hexdigest()
        payload["mtime_ns"] = source_path.stat().st_mtime_ns
    return payload


def _base_sources(project: str | None) -> list[dict[str, Any]]:
    sources = [
        _source("research_agents", "research", "AGENTS.md"),
        _source("common_pitfalls", "research", "Common", "避坑清单.md"),
    ]
    if project:
        sources.append(_source("project_progress", "research", project, "研究进展.md"))
        project_common = PROJECT_ROOT / "research" / project / "common"
        if project_common.exists():
            for path in sorted(project_common.glob("*.md")):
                sources.append(_source_from_path(f"project_common:{path.name}", path))
    return sources


def _mch_frequency_template(project: str, task_type: str) -> dict[str, Any]:
    overrides = {
        "IBRION": 5,
        "NFREE": 2,
        "POTIM": 0.015,
        "NSW": 1,
        "ISYM": 0,
        "PREC": "Normal",
        "LREAL": "Auto",
        "EDIFF": 1e-5,
        "EDIFFG": -0.03,
    }
    return {
        "status": "ok",
        "template_found": True,
        "project": project,
        "task_type": task_type,
        "template_id": "mch_pt_br_stable_intermediate_frequency",
        "template_scope": "MCH-Pt-Br 已优化稳定中间体的有限差分频率 / ZPE / 自由能校正",
        "source_paths": _base_sources(project),
        "incar_overrides": overrides,
        "expected_incar": dict(overrides),
        "required_incar": list(overrides),
        "severity_by_key": {key: "blocker" for key in overrides},
        "free_atom_policy": "只放开吸附物 + 参与反应/位移较大的局部 Pt；其余 Pt 固定。",
        "submit_profile": None,
        "notes": [
            "该模板来自 research/MCH-Pt-Br/common/DFT任务与自由能校正规则.md 的频率校正规则。",
            "未优化稳定中间体需先 relax，再做频率；已优化中间体不要再 relax。",
            "小而软的虚频模式不能单独作为失败判据，需结合振动模式判断。",
        ],
        "blocked_method_rules": [],
    }


def _mch_dimer_template(project: str, task_type: str) -> dict[str, Any]:
    overrides = {
        "PREC": "Normal",
        "EDIFF": 1e-5,
        "IOPT": 1,
        "IBRION": 3,
        "ICHAIN": 2,
        "POTIM": 0,
        "ISIF": 2,
    }
    return {
        "status": "ok",
        "template_found": True,
        "project": project,
        "task_type": task_type,
        "template_id": "mch_pt_br_vasp_dimer_refinement",
        "template_scope": "MCH-Pt-Br 过渡态在 MACE NEB/Hessian/MODECAR 后的 VASP Dimer 精修",
        "source_paths": _base_sources(project),
        "incar_overrides": overrides,
        "expected_incar": dict(overrides),
        "required_incar": list(overrides),
        "severity_by_key": {
            "PREC": "blocker",
            "EDIFF": "blocker",
            "IOPT": "blocker",
            "ICHAIN": "blocker",
            "IBRION": "warning",
            "POTIM": "warning",
            "ISIF": "warning",
        },
        "free_atom_policy": "TS 频率只放开吸附物 + 参与反应/大位移局部 Pt；其余固定。",
        "submit_profile": None,
        "notes": [
            "TS 主路线是 MACE NEB(FIRE→CI-NEB) → MACE Hessian → MODECAR → VASP Dimer。",
            "MCH TS rescue 只复用模板并自动更新 MAGMOM，不靠随意改 INCAR 抢救。",
        ],
        "blocked_method_rules": [
            "不要建议纯 VASP NEB 作为主路线。",
            "不要把 PREC 改成 Accurate；research 固定为 PREC=Normal。",
            "不要把 EDIFF 收紧到 1E-6/1E-7；research 固定为 EDIFF=1E-5。",
            "不要把 IOPT 改成非 1；research 固定 Dimer IOPT=1。",
        ],
    }


def _mch_relax_template(project: str, task_type: str) -> dict[str, Any]:
    overrides = {
        "PREC": "Normal",
        "LREAL": "Auto",
        "EDIFF": 1e-5,
        "EDIFFG": -0.03,
    }
    return {
        "status": "ok",
        "template_found": True,
        "project": project,
        "task_type": task_type,
        "template_id": "mch_pt_br_local_relax_alignment",
        "template_scope": "MCH-Pt-Br 本地优化 / 频率前优化与项目口径对齐",
        "source_paths": _base_sources(project),
        "incar_overrides": overrides,
        "expected_incar": dict(overrides),
        "required_incar": list(overrides),
        "severity_by_key": {key: "warning" for key in overrides},
        "free_atom_policy": "按结构约束/Selective Dynamics 放开吸附物和局部 Pt；避免全 slab 无约束漂移。",
        "submit_profile": None,
        "notes": [
            "稳定中间体若尚未优化，先 relax，再按频率模板做自由能校正。",
            "该模板只对齐项目公共精度口径；具体局部放松范围仍需模型按结构与 research 证据判断。",
        ],
        "blocked_method_rules": [],
    }


def resolve_research_vasp_template(
    project: str | None = None,
    task_type: str | None = None,
    *,
    prompt: str = "",
    material: str | None = None,
) -> dict[str, Any]:
    """Return machine-readable VASP template rules distilled from research docs.

    This is deliberately a rule resolver, not a fixed workflow runner: it gives the
    model the project-specific constraints it must apply before build/preflight.
    """

    project_clean = (project or "").strip() or None
    normalized_task = normalize_research_task_type(task_type, prompt)
    if project_clean == "MCH-Pt-Br" and normalized_task == "vibrational_frequency":
        template = _mch_frequency_template(project_clean, normalized_task)
    elif project_clean == "MCH-Pt-Br" and normalized_task == "transition_state_search":
        template = _mch_dimer_template(project_clean, normalized_task)
    elif project_clean == "MCH-Pt-Br" and normalized_task == "relax":
        template = _mch_relax_template(project_clean, normalized_task)
    else:
        template = {
            "status": "ok",
            "template_found": False,
            "project": project_clean,
            "task_type": normalized_task,
            "template_id": "generic_vasp_no_project_override",
            "template_scope": "未找到可安全自动应用的项目专属模板；模型必须继续读取 research 并显式核对。",
            "source_paths": _base_sources(project_clean),
            "incar_overrides": {},
            "expected_incar": {},
            "required_incar": [],
            "severity_by_key": {},
            "free_atom_policy": None,
            "submit_profile": None,
            "notes": ["没有项目级硬模板时，不要编造 INCAR；优先使用结构同目录模板或要求补充 project 规则证据。"],
            "blocked_method_rules": [],
        }
    template["material"] = material
    return template
