"""Small adsorption-candidate evaluation set and scoring helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AdsorptionEvalCase:
    case_id: str
    material: str
    adsorbate: str
    expected_anchor_atom: str
    expected_site_family: str
    expected_motif_keywords: list[str]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ADSORPTION_EVAL_SET = [
    AdsorptionEvalCase(
        case_id="h2o_pt111_o_down",
        material="Pt(111)",
        adsorbate="H2O",
        expected_anchor_atom="O",
        expected_site_family="ontop",
        expected_motif_keywords=["O-down", "upright", "atop"],
        note="Canonical water/Pt(111) sanity case: oxygen lone pair toward atop Pt.",
    ),
    AdsorptionEvalCase(
        case_id="co_pt111_c_down",
        material="Pt(111)",
        adsorbate="CO",
        expected_anchor_atom="C",
        expected_site_family="ontop",
        expected_motif_keywords=["C-down", "CO", "atop"],
        note="CO often binds through carbon; site-family preference is functional/coverage sensitive, so score is advisory.",
    ),
    AdsorptionEvalCase(
        case_id="oh_pt111_o_down",
        material="Pt(111)",
        adsorbate="OH",
        expected_anchor_atom="O",
        expected_site_family="ontop",
        expected_motif_keywords=["O-down", "OH"],
        note="Hydroxyl should anchor through O; exact bridge/atop can vary, but O anchor is non-negotiable.",
    ),
    AdsorptionEvalCase(
        case_id="nh3_cu111_n_down",
        material="Cu(111)",
        adsorbate="NH3",
        expected_anchor_atom="N",
        expected_site_family="ontop",
        expected_motif_keywords=["N-down", "lone pair", "upright"],
        note="Ammonia binds through the N lone pair on close-packed metal surfaces.",
    ),
    AdsorptionEvalCase(
        case_id="o_pt111_hollow",
        material="Pt(111)",
        adsorbate="O",
        expected_anchor_atom="O",
        expected_site_family="hollow",
        expected_motif_keywords=["hollow", "O"],
        note="Atomic oxygen high-coordination hollow case for checking non-molecular adsorbates.",
    ),
    AdsorptionEvalCase(
        case_id="co_cu111_c_down",
        material="Cu(111)",
        adsorbate="CO",
        expected_anchor_atom="C",
        expected_site_family="ontop",
        expected_motif_keywords=["C-down", "CO", "atop"],
        note="Simple CO/Cu(111) transfer case; anchor choice should remain carbon even when site ranking is uncertain.",
    ),
    AdsorptionEvalCase(
        case_id="h_pt111_hollow",
        material="Pt(111)",
        adsorbate="H",
        expected_anchor_atom="H",
        expected_site_family="hollow",
        expected_motif_keywords=["hollow", "H"],
        note="Atomic hydrogen prefers high-coordination sites on close-packed Pt in many low-coverage models.",
    ),
]


def list_adsorption_eval_cases() -> list[dict[str, Any]]:
    return [case.to_dict() for case in ADSORPTION_EVAL_SET]


def _find_case(case_id: str | None, material: str | None, adsorbate: str | None) -> AdsorptionEvalCase:
    if case_id:
        for case in ADSORPTION_EVAL_SET:
            if case.case_id == case_id:
                return case
        raise ValueError(f"未知 eval case: {case_id}")
    material_l = (material or "").lower()
    adsorbate_l = (adsorbate or "").lower()
    for case in ADSORPTION_EVAL_SET:
        if case.material.lower() == material_l and case.adsorbate.lower() == adsorbate_l:
            return case
    raise ValueError("未找到匹配 eval case；请提供 case_id 或 material+adsorbate。")


def score_adsorption_plan_against_eval(
    plan_payload: dict[str, Any],
    *,
    case_id: str | None = None,
    material: str | None = None,
    adsorbate: str | None = None,
) -> dict[str, Any]:
    """Score an adsorption_candidate_plan against a tiny literature-prior eval set.

    This is not a substitute for DFT validation; it is a behavior check that the
    model used chemically plausible anchor/site/motif choices.
    """
    case = _find_case(case_id, material, adsorbate)
    anchor = str(plan_payload.get("anchor_atom") or "").lower()
    motif = str(plan_payload.get("expected_binding_motif") or "").lower()
    target_sites = plan_payload.get("target_sites") or []
    site_text = " ".join(
        [
            str(item.get("site_id", "")) + " " + str(item.get("site_family", "")) + " " + str(item.get("reason", ""))
            for item in target_sites
            if isinstance(item, dict)
        ]
    ).lower()
    anchor_ok = anchor == case.expected_anchor_atom.lower()
    site_ok = case.expected_site_family.lower() in site_text
    keyword_hits = [
        keyword for keyword in case.expected_motif_keywords
        if keyword.lower() in motif or keyword.lower() in site_text
    ]
    score = round(0.45 * float(anchor_ok) + 0.35 * float(site_ok) + 0.20 * (len(keyword_hits) / max(1, len(case.expected_motif_keywords))), 3)
    return {
        "status": "ok",
        "case": case.to_dict(),
        "score": score,
        "passed": score >= 0.7,
        "checks": {
            "anchor_ok": anchor_ok,
            "site_family_ok": site_ok,
            "motif_keyword_hits": keyword_hits,
        },
        "boundary": "小型文献先验行为评估；不代表真实 DFT 能量或最终结构正确。",
    }


def render_model_comparison_report(
    *,
    output_path: str,
    model_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a markdown report template for deepseek/qwen eval comparison.

    ``model_results`` may contain rows like
    ``{"model_id": "...", "case_id": "...", "score": 0.8, "passed": true}``.
    If omitted, the report is an explicit not-yet-run template.
    """
    from collections import defaultdict
    from pathlib import Path

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = model_results or []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row.get("model_id") or "unknown")].append(row)

    lines = [
        "# AETHER-DFT adsorption candidate model comparison",
        "",
        "## Scope",
        "",
        "This report compares model-authored adsorption plans against the built-in small literature-prior eval set.",
        "Scores are behavior checks only; they do not replace DFT validation or literature review.",
        "",
        "## Eval cases",
        "",
        "| Case | Material | Adsorbate | Expected anchor | Expected site | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in ADSORPTION_EVAL_SET:
        lines.append(
            f"| `{case.case_id}` | {case.material} | {case.adsorbate} | {case.expected_anchor_atom} | {case.expected_site_family} | {case.note} |"
        )
    lines.extend(["", "## Model results", ""])
    if not rows:
        lines.extend(
            [
                "_No live model results recorded yet._",
                "",
                "Run `AETHER_RUN_LLM_TESTS=1 pytest tests/test_llm_authored_adsorption_e2e.py -q`",
                "and collect `adsorption_eval_score_plan` outputs for deepseek/qwen before filling this table.",
            ]
        )
    else:
        lines.extend(["| Model | Cases | Pass rate | Mean score |", "| --- | ---: | ---: | ---: |"])
        for model_id, model_rows in sorted(by_model.items()):
            pass_rate = sum(1 for row in model_rows if row.get("passed")) / max(1, len(model_rows))
            mean_score = sum(float(row.get("score") or 0.0) for row in model_rows) / max(1, len(model_rows))
            lines.append(f"| `{model_id}` | {len(model_rows)} | {pass_rate:.2%} | {mean_score:.3f} |")
    lines.extend(
        [
            "",
            "## Known gaps",
            "",
            "- Live API runs are opt-in and may be skipped in CI.",
            "- The eval set is intentionally small and should be expanded with project-specific literature priors.",
            "- A high score means the model followed expected chemistry priors, not that the generated structure is globally optimal.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "report_path": str(target),
        "case_count": len(ADSORPTION_EVAL_SET),
        "model_result_count": len(rows),
    }
