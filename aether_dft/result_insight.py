from __future__ import annotations

"""Lightweight scientific interpretation of finished or partial VASP runs."""

import json
import re
from pathlib import Path
from typing import Any

from ase.io import read

from .research_workspace import read_research_onboarding_context


def _read(path: Path, limit: int = 500_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def detect_synthetic_vasp_output(text: str) -> dict[str, Any]:
    """Detect smoke-test / synthetic markers that must not be treated as science.

    Real VASP output can contain many generic words, so keep this deliberately
    narrow: markers must indicate AETHER validation, synthetic output, or an
    explicit smoke-test banner.  This protects the agent from turning a harness
    fixture into a scientific conclusion.
    """

    lowered = text.lower()
    markers = [
        "aether synthetic",
        "synthetic vasp",
        "synthetic vasp-like",
        "cluster smoke outcar",
        "smoke outcar",
        "smoke-test outcar",
        "smoke test outcar",
    ]
    matched = [marker for marker in markers if marker in lowered]
    return {
        "detected": bool(matched),
        "markers": matched[:5],
        "warning": (
            "输出包含 synthetic/smoke-test 标记；这只能用于验证管道可用性，"
            "不能作为真实 VASP 科学结果。"
            if matched
            else ""
        ),
    }


def _parse_float_tail(pattern: str, text: str) -> list[float]:
    values: list[float] = []
    for match in re.findall(pattern, text):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def _frequency_summary(outcar_text: str) -> dict[str, Any]:
    real = _parse_float_tail(r"\bf\s*=\s*(-?\d+(?:\.\d+)?)\s+THz", outcar_text)
    imaginary = _parse_float_tail(r"\bf/i=\s*(-?\d+(?:\.\d+)?)\s+THz", outcar_text)
    has_frequency_section = bool(real or imaginary or "Eigenvectors and eigenvalues of the dynamical matrix" in outcar_text)
    if not has_frequency_section:
        return {"detected": False}
    return {
        "detected": True,
        "real_mode_count": len(real),
        "imaginary_mode_count": len(imaginary),
        "min_real_thz": min(real) if real else None,
        "max_real_thz": max(real) if real else None,
        "max_imaginary_thz": max(imaginary) if imaginary else None,
        "imaginary_modes_thz": imaginary[:12],
    }


def _first_existing(root: Path, *names: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / names[0]


def _bond_to_dict(bond: Any) -> dict[str, Any]:
    return {
        "atom_i": int(getattr(bond, "atom_i", -1)),
        "atom_j": int(getattr(bond, "atom_j", -1)),
        "element_i": str(getattr(bond, "element_i", "")),
        "element_j": str(getattr(bond, "element_j", "")),
        "distance": round(float(getattr(bond, "distance", 0.0)), 4),
        "type": str(getattr(bond, "anomaly_type", "")),
    }


def _structure_change_summary(poscar: Path, contcar: Path) -> dict[str, Any]:
    if not (poscar.exists() and contcar.exists()):
        return {
            "status": "unavailable",
            "message": "缺 POSCAR 或 CONTCAR，无法判断吸附保持/解离/迁移。",
        }
    try:
        initial = read(str(poscar))
        final = read(str(contcar))
        if len(initial) != len(final) or initial.get_chemical_symbols() != final.get_chemical_symbols():
            return {
                "status": "incompatible",
                "message": "POSCAR/CONTCAR 原子数或元素顺序不同，不能直接做位移/键变化比较。",
            }
        from dft_shared.structure_analyzer.bond_analyzer import compare_bonds
        from dft_shared.structure_analyzer.comparator import compare_structures

        displacement = compare_structures(initial, final, top_n=5)
        formed, broken = compare_bonds(initial, final)
        drift = displacement.adsorbate_drift or {}
        broken_dicts = [_bond_to_dict(item) for item in broken[:8]]
        formed_dicts = [_bond_to_dict(item) for item in formed[:8]]
        has_adsorbate_drift = bool(drift.get("drifted"))
        has_broken_light_bonds = any(
            item["element_i"] in {"H", "C", "N", "O", "S", "P", "Cl", "Br"} and item["element_j"] in {"H", "C", "N", "O", "S", "P", "Cl", "Br"}
            for item in broken_dicts
        )
        if has_adsorbate_drift:
            adsorption_verdict = "possible_desorption_or_large_migration"
        elif has_broken_light_bonds:
            adsorption_verdict = "possible_adsorbate_dissociation"
        elif broken_dicts or formed_dicts:
            adsorption_verdict = "bonding_changed_review_geometry"
        elif displacement.max_displacement < 0.5:
            adsorption_verdict = "adsorption_geometry_stable"
        else:
            adsorption_verdict = "adsorption_geometry_changed_review_needed"
        return {
            "status": "ok",
            "adsorption_verdict": adsorption_verdict,
            "max_displacement_a": round(displacement.max_displacement, 4),
            "mean_displacement_a": round(displacement.mean_displacement, 4),
            "adsorbate_drift": drift,
            "formed_bonds": formed_dicts,
            "broken_bonds": broken_dicts,
            "anomalies": displacement.anomalies,
            "guidance": "这是启发式结构/键变化判断；最终 adsorption/dissociation 结论仍需结合体系定义、可视检查和能量对照。",
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def interpret_result(run_root: str | Path) -> dict[str, Any]:
    """Interpret a run without pretending to know chemistry that is not in files."""

    root = Path(run_root)
    if not root.exists():
        return {"status": "missing", "run_root": str(root), "message": "run_root 不存在。"}
    outcar_path = _first_existing(root, "OUTCAR", "outputs/OUTCAR")
    oszicar_path = _first_existing(root, "OSZICAR", "outputs/OSZICAR")
    contcar = _first_existing(root, "CONTCAR", "outputs/CONTCAR")
    poscar = _first_existing(root, "POSCAR", "inputs/POSCAR")
    outcar_text = _read(outcar_path)
    oszicar_text = _read(oszicar_path, limit=120_000)
    synthetic = detect_synthetic_vasp_output(outcar_text + "\n" + oszicar_text)
    if not outcar_text and not oszicar_text:
        return {
            "status": "no_outputs",
            "run_root": str(root),
            "files": {"OUTCAR": False, "OSZICAR": False, "CONTCAR": contcar.exists(), "POSCAR": poscar.exists()},
            "interpretation": "还没有可解释的 VASP 输出；先确认作业是否启动、日志是否回拉。",
            "suggestions": ["cluster_job_status_brief", "cluster_job_tail_log", "cluster_remote_fetch"],
        }
    toten = _parse_float_tail(r"TOTEN\s*=\s*(-?\d+(?:\.\d+)?(?:E[+-]?\d+)?)", outcar_text)
    osz_energies = _parse_float_tail(r"F=\s*(-?\d+(?:\.\d+)?(?:E[+-]?\d+)?)", oszicar_text)
    reached = "reached required accuracy" in outcar_text.lower()
    stopped = "Voluntary context switches" in outcar_text or "General timing and accounting informations" in outcar_text
    frequency = _frequency_summary(outcar_text)
    warnings: list[str] = []
    suggestions: list[str] = []
    if synthetic["detected"]:
        warnings.append(synthetic["warning"])
        suggestions.append("若需要科研结论，重新提交真实 VASP 任务并等待真实 OUTCAR/OSZICAR 回拉后再解释。")
    if frequency.get("detected") and frequency.get("imaginary_mode_count", 0):
        warnings.append("频率输出包含虚频；若目标是稳定中间体，需要检查构型是否为极小点。")
        suggestions.append("对虚频模式做位移可视化；稳定中间体需重新优化，TS 则需确认是否只有目标反应坐标一个虚频。")
    if not reached and not frequency.get("detected"):
        warnings.append("OUTCAR 未出现 reached required accuracy；不能把结构优化当作已收敛。")
        suggestions.append("若 job 已结束，检查 NSW/EDIFFG/SCF 收敛；若仍在跑，用 progress_estimate 继续观察。")
    if len(osz_energies) >= 3:
        deltas = [osz_energies[i] - osz_energies[i - 1] for i in range(1, len(osz_energies))]
        oscillating = any(deltas[i] * deltas[i - 1] < 0 for i in range(1, len(deltas)))
        if oscillating:
            warnings.append("OSZICAR 能量轨迹有震荡；可能需要检查 smearing/mixing 或初始构型。")
            suggestions.append("对金属表面核对 ISMEAR/SIGMA；对分子/绝缘体系避免不合适 smearing。")
    else:
        oscillating = False
    if contcar.exists():
        suggestions.append("对比 POSCAR/CONTCAR 位移与键连，判断吸附是否保持、是否解离或迁移。")
    structure_change = _structure_change_summary(poscar, contcar)
    if structure_change.get("status") == "ok":
        verdict_hint = structure_change.get("adsorption_verdict")
        if verdict_hint == "possible_adsorbate_dissociation":
            warnings.append("POSCAR/CONTCAR 键变化提示吸附物可能解离；需要可视化和能量对照确认。")
        elif verdict_hint == "possible_desorption_or_large_migration":
            warnings.append("吸附物整体漂移较大，可能脱附或迁移；不能直接当作原目标构型的吸附能。")
        elif verdict_hint == "bonding_changed_review_geometry":
            suggestions.append("存在成键/断键变化，建议先做 bond/位移 review 再记录 outcome。")
    if synthetic["detected"]:
        verdict = "test_output_detected"
        headline = "输出含 synthetic/smoke-test 标记；管道解析成功，但不能当作真实 VASP 科学结果。"
    elif frequency.get("detected") and stopped and toten:
        if frequency.get("imaginary_mode_count", 0):
            verdict = "frequency_finished_with_imaginary_modes"
            headline = "频率任务已正常结束并产生频率，但包含虚频；需要结合任务目标判断是 TS 还是未稳定构型。"
        else:
            verdict = "frequency_finished_no_imaginary_modes"
            headline = "频率任务已正常结束，未检出虚频；可进入 ZPE/热校正/自由能记录，但仍需核对任务模板和参考态。"
    elif reached and toten:
        verdict = "finished_converged"
        headline = "计算输出显示电子/离子收敛，可进入结构对比、能量归一化和项目规则复核。"
    elif stopped and toten:
        verdict = "finished_not_converged"
        headline = "计算产生能量但未达收敛标志；结果只能作为诊断，不能作为最终能量。"
    else:
        verdict = "running_or_partial"
        headline = "当前输出不完整，适合做实时诊断，不适合给最终科学结论。"
    return {
        "status": "ok",
        "run_root": str(root),
        "verdict": verdict,
        "headline": headline,
        "energy": {
            "last_toten_ev": toten[-1] if toten else None,
            "last_oszicar_f_ev": osz_energies[-1] if osz_energies else None,
            "ionic_steps_seen": len(osz_energies),
            "oscillating": oscillating,
        },
        "frequency": frequency,
        "synthetic_output": synthetic,
        "files": {"OUTCAR": bool(outcar_text), "OSZICAR": bool(oszicar_text), "CONTCAR": contcar.exists(), "POSCAR": poscar.exists()},
        "structure_change": structure_change,
        "adsorption_interpretation": structure_change.get("adsorption_verdict") if structure_change.get("status") == "ok" else "unavailable",
        "warnings": warnings,
        "suggestions": suggestions,
        "next_tools": ["structure_displacement_compare", "structure_bond_analyze", "candidate_outcome_record", "research_learning_capture"],
        "guidance": "这是文件证据解读，不替代具体体系的能量定义/零点能/覆盖度/构型对比。",
    }


def propose_next_experiments(project: str | None = None, recent_results: list[dict[str, Any]] | str | None = None) -> dict[str, Any]:
    """Propose a small set of next actions from recent evidence."""

    parsed: list[dict[str, Any]]
    if isinstance(recent_results, str) and recent_results.strip():
        try:
            raw = json.loads(recent_results)
            parsed = raw if isinstance(raw, list) else [raw]
        except Exception:
            parsed = [{"note": recent_results}]
    else:
        parsed = recent_results if isinstance(recent_results, list) else []
    onboarding = read_research_onboarding_context(project, max_chars=2500)
    any_failed = any(str(item.get("verdict") or item.get("status") or "").lower() in {"failed", "finished_not_converged", "incomplete"} for item in parsed if isinstance(item, dict))
    proposals: list[dict[str, Any]] = []
    if any_failed:
        proposals.append(
            {
                "title": "先做失败/未收敛诊断",
                "why": "已有结果未形成可用科学证据，贸然扩展构型会放大错误。",
                "actions": ["tail 日志定位 SCF/离子步问题", "按 research 模板复核 INCAR", "必要时重建更稳初猜或调整 smearing/mixing"],
            }
        )
    proposals.extend(
        [
            {
                "title": "补一个对照构型或基准位点",
                "why": "单个构型很难支持吸附强弱/机理判断，需要至少一个近邻 site 或 clean slab reference。",
                "actions": ["用 slab_surface_inspect 找非等价位点", "只生成 1-3 个有理由候选", "记录排除位点理由"],
            },
            {
                "title": "把已获得结果沉淀为项目 prior",
                "why": "长期科研合伙人需要把位点偏好、失败参数和模板规则变成下轮上下文。",
                "actions": ["research_learning_capture 写 Learning", "candidate_outcome_record 记录候选 outcome", "research_workspace_sync_to_cluster 统一 ~/research"],
            },
            {
                "title": "按项目目标选择下一类证据",
                "why": "如果结构已稳，下一步才考虑吸附能、频率、NEB/Dimer 或微观动力学。",
                "actions": ["根据 research 进展确认任务类型", "resolve_research_vasp_template", "生成并 preflight 新输入包"],
            },
        ]
    )
    return {
        "status": "ok",
        "project": onboarding.get("project") or project or "",
        "recent_result_count": len(parsed),
        "proposals": proposals[:3],
        "research_context_digest": onboarding.get("context", "")[:900],
    }
