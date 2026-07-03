"""Adsorption relaxation feedback for model-led candidate refinement.

Recent agentic catalyst/adsorption systems commonly use a closed loop:
generate diverse adsorption candidates, run a cheap relaxation or pre-check,
inspect drift/failure/energy feedback, then refine the next candidate batch.

This module is deliberately a decision aid, not a fixed workflow engine.  It
does not mutate project state; the model decides whether to call campaign
update, candidate outcome writeback, structure generation, or DFT submission.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dft_shared.structure_analyzer.comparator import compare_structures


def _maybe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _path_exists(value: str | None) -> bool:
    return bool(value and Path(value).exists())


def _compare_if_possible(initial_path: str | None, relaxed_path: str | None) -> dict[str, Any] | None:
    if not (_path_exists(initial_path) and _path_exists(relaxed_path)):
        return None
    report = compare_structures(str(initial_path), str(relaxed_path), top_n=8)
    return {
        "max_displacement": report.max_displacement,
        "mean_displacement": report.mean_displacement,
        "top_movers": report.top_movers,
        "adsorbate_drift": report.adsorbate_drift,
        "anomalies": report.anomalies,
    }


def _quality_payload(payload: dict[str, Any]) -> dict[str, Any]:
    quality = payload.get("quality_report")
    if not isinstance(quality, dict):
        quality = payload.get("quality")
    return quality if isinstance(quality, dict) else {}


def _displacement_payload(payload: dict[str, Any]) -> dict[str, Any]:
    explicit = payload.get("displacement_report")
    if isinstance(explicit, dict):
        return explicit
    compared = _compare_if_possible(
        str(payload.get("initial_path") or "").strip() or None,
        str(payload.get("relaxed_path") or payload.get("final_path") or "").strip() or None,
    )
    return compared or {}


def _issue_texts(*, quality: dict[str, Any], displacement: dict[str, Any], notes: str) -> list[str]:
    issues: list[str] = []
    issues.extend(_string_list(quality.get("issues")))
    issues.extend(_string_list(displacement.get("anomalies")))
    if notes:
        lowered = notes.lower()
        for marker in ("dissociate", "dissociated", "desorb", "floating", "overlap", "解离", "脱附", "漂移", "重叠"):
            if marker in lowered:
                issues.append(f"note_mentions:{marker}")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for item in issues:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def adsorption_relaxation_feedback(
    *,
    candidate_id: str | None = None,
    material: str | None = None,
    adsorbate: str | None = None,
    initial_path: str | None = None,
    relaxed_path: str | None = None,
    quality_report: dict[str, Any] | None = None,
    displacement_report: dict[str, Any] | None = None,
    adsorption_energy_ev: float | None = None,
    energy_change_ev: float | None = None,
    outcome: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Turn relaxation/result feedback into next modeling actions.

    The thresholds are intentionally conservative heuristics.  They are meant
    to help the model decide what evidence/tool to collect next, not to make a
    final publication-grade thermodynamic claim.
    """

    payload = {
        "quality_report": quality_report or {},
        "displacement_report": displacement_report or {},
        "initial_path": initial_path,
        "relaxed_path": relaxed_path,
    }
    quality = _quality_payload(payload)
    displacement = _displacement_payload(payload)
    notes_text = str(notes or "").strip()
    issues = _issue_texts(quality=quality, displacement=displacement, notes=notes_text)

    q_verdict = str(quality.get("verdict") or quality.get("status") or "").strip().lower()
    q_score = _maybe_float((quality.get("score") or {}).get("total") if isinstance(quality.get("score"), dict) else quality.get("score"))
    max_disp = _maybe_float(displacement.get("max_displacement"))
    ads_drift = _maybe_float(displacement.get("adsorbate_drift"))
    e_ads = _maybe_float(adsorption_energy_ev)
    d_e = _maybe_float(energy_change_ev)
    outcome_l = str(outcome or "").strip().lower()

    findings: list[dict[str, Any]] = []

    def add(level: str, code: str, message: str) -> None:
        findings.append({"level": level, "code": code, "message": message})

    if not quality and not displacement and e_ads is None and not outcome_l:
        add("warning", "insufficient_feedback", "缺少 quality/displacement/energy/outcome 证据；不能判断候选是否应进入正式 DFT。")

    if q_verdict in {"reject", "failed", "error"}:
        add("error", "quality_rejected", f"候选质量检查 verdict={q_verdict}，不应直接提交。")
    elif q_verdict in {"warning", "ambiguous"}:
        add("warning", "quality_warning", f"候选质量检查 verdict={q_verdict}，建议先修正或补证据。")
    if q_score is not None and q_score < 0.45:
        add("warning", "low_quality_score", f"候选质量分数 {q_score:.2f} 偏低。")

    if max_disp is not None and max_disp >= 2.0:
        add("warning", "large_relaxation_displacement", f"relax 后最大位移 {max_disp:.2f} Å，初猜可能不稳定。")
    if ads_drift is not None and ads_drift >= 1.2:
        add("warning", "adsorbate_drift", f"吸附物漂移 {ads_drift:.2f} Å，位点/取向可能不合理。")
    if any(any(marker in issue.lower() for marker in ("overlap", "重叠", "too close")) for issue in issues):
        add("error", "geometry_overlap", "检测到原子重叠/过近问题，需要重建候选。")
    if any(any(marker in issue.lower() for marker in ("floating", "desorb", "脱附", "漂移")) for issue in issues):
        add("warning", "weak_or_unbound_candidate", "候选可能漂浮或脱附，需要比较邻近位点/不同 anchor。")
    if any(any(marker in issue.lower() for marker in ("dissociate", "解离")) for issue in issues):
        add("info", "dissociation_channel", "候选出现解离信号；如果目标包含反应路径，可能应转入中间体/TS 候选，而不是简单丢弃。")

    if e_ads is not None:
        if e_ads > 0.15:
            add("warning", "unfavorable_adsorption_energy", f"E_ads={e_ads:.3f} eV，吸附热力学不利。")
        elif e_ads < -0.25:
            add("info", "promising_adsorption_energy", f"E_ads={e_ads:.3f} eV，候选值得保留或进一步精修。")
    if d_e is not None and d_e > 0.05:
        add("warning", "relaxation_energy_increased", f"relax 后能量升高 {d_e:.3f} eV，需检查输入/结构。")
    if outcome_l in {"desorbed", "dissociated", "failed", "unbound"}:
        add("warning", f"outcome_{outcome_l}", f"已有 outcome={outcome_l}，应避免盲目继续提交同类候选。")

    severe_codes = {item["code"] for item in findings if item["level"] == "error"}
    warning_codes = {item["code"] for item in findings if item["level"] == "warning"}
    if severe_codes:
        decision = "rebuild_candidate"
    elif {"adsorbate_drift", "weak_or_unbound_candidate", "large_relaxation_displacement"} & warning_codes:
        decision = "refine_candidate_family"
    elif {"unfavorable_adsorption_energy", "outcome_desorbed", "outcome_unbound", "outcome_failed"} & warning_codes:
        decision = "prune_or_deprioritize"
    elif not findings or all(item["level"] == "info" for item in findings):
        decision = "promote_or_submit"
    else:
        decision = "needs_more_evidence"

    next_actions = {
        "rebuild_candidate": [
            "call structure_enumerate_sites or slab_surface_inspect to choose a physically distinct site",
            "call adsorption_candidate_plan to revise anchor/site/orientation rationale",
            "call structure_add_adsorbate with safer height/orientation, then candidate_quality_score",
        ],
        "refine_candidate_family": [
            "generate 2-4 nearby site/orientation variants instead of hand-perfecting one geometry",
            "cheap-relax variants with structure_relax_short or submit a small DFT batch if allowed",
            "record drift/outcome via candidate_outcome_record after final evidence exists",
        ],
        "prune_or_deprioritize": [
            "update campaign candidate as discarded/retry with reason if campaign is active",
            "prefer stronger neighboring site families or different anchor atom/orientation",
            "keep this result as a negative prior with candidate_outcome_record",
        ],
        "promote_or_submit": [
            "if only cheap relaxation was used, run vasp_input_preflight_check then dft_run_task/cluster_remote_submit when allowed",
            "if DFT already completed, call result_interpret and candidate_outcome_record",
            "write reusable lesson with research_learning_capture if the motif is scientifically useful",
        ],
        "needs_more_evidence": [
            "collect candidate_quality_score and displacement comparison before deciding",
            "if calculation ran, inspect OUTCAR/OSZICAR with cluster_job_partial_outcar or result_interpret",
        ],
    }[decision]

    return {
        "status": "ok",
        "candidate_id": str(candidate_id or "").strip(),
        "system": {"material": material, "adsorbate": adsorbate},
        "decision": decision,
        "findings": findings,
        "issues": issues,
        "evidence": {
            "quality_verdict": q_verdict,
            "quality_score": q_score,
            "max_displacement": max_disp,
            "adsorbate_drift": ads_drift,
            "adsorption_energy_ev": e_ads,
            "energy_change_ev": d_e,
            "outcome": outcome_l,
            "initial_path": initial_path,
            "relaxed_path": relaxed_path,
        },
        "next_actions": next_actions,
        "model_instruction": (
            "Use this as relaxation-feedback evidence. Do not declare the research goal complete from this tool alone; "
            "choose the next modeling/submission/writeback tool based on decision and remaining evidence gaps."
        ),
    }
