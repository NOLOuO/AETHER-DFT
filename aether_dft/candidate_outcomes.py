"""Candidate outcome write-back helpers.

Phase 5 of the model-authored candidate roadmap turns finished calculations
back into project priors.  The helpers here intentionally do not claim a DFT
run is complete; they only persist evidence supplied by the caller.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dft_shared.structure_analyzer.comparator import compare_structures

from .knowledge import add_note
from .project_state import append_progress


def _exists(value: str | None) -> bool:
    return bool(value and Path(value).exists())


def record_candidate_outcome(
    *,
    project: str,
    material: str,
    adsorbate: str,
    candidate_id: str,
    verdict: str,
    adsorption_energy_ev: float | None = None,
    initial_path: str | None = None,
    final_path: str | None = None,
    manifest_path: str | None = None,
    calculation_summary: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Persist one completed candidate outcome as a searchable KB note.

    Parameters are deliberately explicit so the model must cite where the
    evidence came from.  If initial/final structures exist, a displacement
    comparison is included; otherwise the note records the missing evidence.
    """

    project_clean = project.strip()
    if not project_clean:
        raise ValueError("project 不能为空。")
    candidate_clean = candidate_id.strip()
    if not candidate_clean:
        raise ValueError("candidate_id 不能为空。")
    verdict_clean = verdict.strip().lower()
    if verdict_clean not in {"success", "failed", "ambiguous", "retry", "discard", "promising"}:
        raise ValueError("verdict 必须是 success/failed/ambiguous/retry/discard/promising 之一。")

    displacement_payload: dict[str, Any] | None = None
    if _exists(initial_path) and _exists(final_path):
        report = compare_structures(str(initial_path), str(final_path), top_n=8)
        displacement_payload = {
            "max_displacement": report.max_displacement,
            "mean_displacement": report.mean_displacement,
            "top_movers": report.top_movers,
            "adsorbate_drift": report.adsorbate_drift,
            "anomalies": report.anomalies,
        }

    title = f"{adsorbate} on {material} candidate {candidate_clean} outcome"
    energy_line = (
        f"- Adsorption energy: {adsorption_energy_ev:.6f} eV"
        if adsorption_energy_ev is not None
        else "- Adsorption energy: not provided"
    )
    evidence_lines = [
        f"- Material: {material}",
        f"- Adsorbate: {adsorbate}",
        f"- Candidate ID: {candidate_clean}",
        f"- Verdict: {verdict_clean}",
        energy_line,
        f"- Manifest: {manifest_path or 'not provided'}",
        f"- Initial structure: {initial_path or 'not provided'}",
        f"- Final structure: {final_path or 'not provided'}",
    ]
    if displacement_payload is None:
        evidence_lines.append("- Displacement comparison: unavailable (missing initial_path or final_path)")
    else:
        evidence_lines.extend(
            [
                f"- Max displacement: {displacement_payload['max_displacement']:.4f} Å",
                f"- Mean displacement: {displacement_payload['mean_displacement']:.4f} Å",
                "- Structural anomalies: "
                + ("; ".join(displacement_payload["anomalies"]) if displacement_payload["anomalies"] else "none detected"),
            ]
        )

    content = "\n".join(
        [
            "## Outcome evidence",
            "",
            *evidence_lines,
            "",
            "## Calculation summary",
            "",
            (calculation_summary or "No calculation summary provided.").strip(),
            "",
            "## Reuse guidance",
            "",
            (notes or "Use this outcome as a prior only when material, surface model, adsorbate, and candidate motif are comparable.").strip(),
            "",
            "## Machine-readable payload",
            "",
            "```json",
            json.dumps(
                {
                    "kind": "candidate_outcome",
                    "project": project_clean,
                    "material": material,
                    "adsorbate": adsorbate,
                    "candidate_id": candidate_clean,
                    "verdict": verdict_clean,
                    "adsorption_energy_ev": adsorption_energy_ev,
                    "initial_path": initial_path,
                    "final_path": final_path,
                    "manifest_path": manifest_path,
                    "displacement": displacement_payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
        ]
    )

    tags = [
        "adsorption",
        "candidate_outcome",
        f"material:{material}",
        f"adsorbate:{adsorbate}",
        f"verdict:{verdict_clean}",
    ]
    note = add_note(project_clean, title, content, tags=tags)
    progress_path = append_progress(
        project_clean,
        completed=[f"记录候选 `{candidate_clean}` 的计算复盘，verdict={verdict_clean}。"],
        blockers=[],
        next_steps=["下次同类候选生成前先调用 knowledge_search_for_system 复用该 outcome。"],
    )
    return {
        "status": "ok",
        "note": note.to_dict(),
        "progress_path": str(progress_path),
        "outcome": {
            "material": material,
            "adsorbate": adsorbate,
            "candidate_id": candidate_clean,
            "verdict": verdict_clean,
            "adsorption_energy_ev": adsorption_energy_ev,
            "displacement": displacement_payload,
        },
        "guidance": "后续生成同体系候选前调用 knowledge_search_for_system(material, adsorbate) 检索该 outcome。",
    }
