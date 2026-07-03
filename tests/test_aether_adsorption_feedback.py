from __future__ import annotations

from aether_dft.adsorption_feedback import adsorption_relaxation_feedback
from aether_dft.runtime_harness.tool_registry import ToolRegistry


def test_adsorption_relaxation_feedback_flags_drift_for_refinement():
    result = adsorption_relaxation_feedback(
        candidate_id="top-001",
        material="Pt(111)",
        adsorbate="CO",
        quality_report={"verdict": "warning", "score": {"total": 0.62}, "issues": []},
        displacement_report={"max_displacement": 2.4, "adsorbate_drift": 1.6, "anomalies": []},
        adsorption_energy_ev=-0.35,
    )

    assert result["status"] == "ok"
    assert result["decision"] == "refine_candidate_family"
    assert any(item["code"] == "adsorbate_drift" for item in result["findings"])
    assert any("structure_relax_short" in action or "generate" in action for action in result["next_actions"])


def test_adsorption_relaxation_feedback_promotes_good_candidate():
    result = adsorption_relaxation_feedback(
        candidate_id="hollow-001",
        material="Pt(111)",
        adsorbate="CO",
        quality_report={"verdict": "pass", "score": {"total": 0.9}},
        displacement_report={"max_displacement": 0.4, "adsorbate_drift": 0.2, "anomalies": []},
        adsorption_energy_ev=-0.55,
    )

    assert result["decision"] == "promote_or_submit"
    assert any(item["code"] == "promising_adsorption_energy" for item in result["findings"])


def test_adsorption_relaxation_feedback_tool_is_discoverable_and_read_only():
    registry = ToolRegistry()
    discovered = registry.run_tool(
        "aether_discover_tools",
        {"query": "relax 后吸附物漂移，怎么反馈到下一轮候选", "max_tools": 20},
    )["result"]

    assert "adsorption_relaxation_feedback" in discovered["tool_names"]
    assert registry.is_read_only_tool("adsorption_relaxation_feedback")

    result = registry.run_tool(
        "adsorption_relaxation_feedback",
        {
            "candidate_id": "bridge-001",
            "material": "Pt(111)",
            "adsorbate": "H2O",
            "quality_report": {"verdict": "reject", "issues": ["floating adsorbate"]},
        },
    )["result"]

    assert result["status"] == "ok"
    assert result["decision"] == "rebuild_candidate"
