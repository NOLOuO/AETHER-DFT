from __future__ import annotations

from pathlib import Path

from aether_dft.evaluation import (
    list_adsorption_eval_cases,
    render_model_comparison_report,
    score_adsorption_plan_against_eval,
)
from aether_dft.runtime_harness.tool_registry import ToolRegistry


def test_adsorption_eval_scores_plausible_h2o_plan():
    cases = list_adsorption_eval_cases()
    assert any(case["case_id"] == "h2o_pt111_o_down" for case in cases)
    plan = {
        "material": "Pt(111)",
        "adsorbate": "H2O",
        "anchor_atom": "O",
        "expected_binding_motif": "atop O-down upright",
        "target_sites": [
            {"site_id": "ontop-01", "site_family": "ontop", "reason": "O lone pair binds atop Pt"}
        ],
    }
    result = score_adsorption_plan_against_eval(plan, case_id="h2o_pt111_o_down")
    assert result["status"] == "ok"
    assert result["passed"] is True
    assert result["checks"]["anchor_ok"] is True


def test_adsorption_eval_tools_are_registered():
    registry = ToolRegistry()
    listed = registry.run_tool("adsorption_eval_case_list", {})
    assert listed["result"]["status"] == "ok"
    plan = {
        "anchor_atom": "O",
        "expected_binding_motif": "hollow O",
        "target_sites": [{"site_id": "hollow-01", "reason": "atomic oxygen high coordination"}],
    }
    scored = registry.run_tool(
        "adsorption_eval_score_plan",
        {"plan": plan, "case_id": "o_pt111_hollow"},
    )
    assert scored["result"]["status"] == "ok"
    assert scored["result"]["passed"] is True


def test_model_comparison_report_template(tmp_path):
    report = render_model_comparison_report(output_path=str(tmp_path / "model_compare.md"))
    assert report["status"] == "ok"
    assert report["case_count"] >= 7
    text = Path(report["report_path"]).read_text(encoding="utf-8")
    assert "No live model results recorded yet" in text
    assert "deepseek/qwen" in text


def test_model_comparison_report_tool_with_results(tmp_path):
    result = ToolRegistry().run_tool(
        "adsorption_eval_model_comparison_report",
        {
            "output_path": str(tmp_path / "report.md"),
            "model_results": [
                {"model_id": "deepseek:deepseek-v4-pro", "case_id": "h2o_pt111_o_down", "score": 0.9, "passed": True},
                {"model_id": "bailian:qwen3.7-max", "case_id": "h2o_pt111_o_down", "score": 0.7, "passed": True},
            ],
        },
    )
    assert result["result"]["status"] == "ok"
    text = Path(result["result"]["report_path"]).read_text(encoding="utf-8")
    assert "deepseek:deepseek-v4-pro" in text
