from __future__ import annotations

from aether_dft.runtime_harness.tool_catalog import (
    FALLBACK_TOOL_NAMES,
    PRIMARY_TOOL_NAMES,
    describe_tool_route,
    list_tool_catalog,
)


def test_n3_tool_catalog_marks_primary_and_fallback_routes():
    catalog = list_tool_catalog()
    names = {item["name"] for item in catalog}

    assert "adsorption_candidate_plan" in PRIMARY_TOOL_NAMES
    assert "structure_add_adsorbate" in PRIMARY_TOOL_NAMES
    assert "adsorption_candidate_manifest_compose" in PRIMARY_TOOL_NAMES
    assert "cluster_execution_intent_plan" in PRIMARY_TOOL_NAMES
    assert "dft_run_task" in PRIMARY_TOOL_NAMES
    assert "cluster_remote_submit" in PRIMARY_TOOL_NAMES

    assert "adsorption_candidates" in FALLBACK_TOOL_NAMES
    assert "adsorption_full_workflow" in FALLBACK_TOOL_NAMES
    assert "dft_task_plan" in FALLBACK_TOOL_NAMES
    assert "dft_run_step" in FALLBACK_TOOL_NAMES
    assert "cluster_job_tail_log" in FALLBACK_TOOL_NAMES

    assert names.issuperset(set(PRIMARY_TOOL_NAMES))
    assert names.issuperset(set(FALLBACK_TOOL_NAMES))


def test_n3_tool_catalog_documents_replacements():
    adsorption_black_box = describe_tool_route("adsorption_candidates")
    assert adsorption_black_box["tier"] == "fallback"
    assert "adsorption_candidate_plan" in str(adsorption_black_box["use_instead"])

    dft_task_plan = describe_tool_route("dft_task_plan")
    assert dft_task_plan["tier"] == "compat"
    assert "dft_run_task" in str(dft_task_plan["use_instead"])

    unknown = describe_tool_route("not_registered")
    assert unknown["tier"] is None
    assert unknown["note"] == "unclassified"
