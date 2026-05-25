from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aether_dft.agent import run_agent_once
from dft_app.llm.key_store import resolve_api_key


def _llm_tests_enabled() -> bool:
    return os.getenv("AETHER_RUN_LLM_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


pytestmark = [
    pytest.mark.requires_llm,
    pytest.mark.skipif(not _llm_tests_enabled(), reason="Set AETHER_RUN_LLM_TESTS=1 to run live LLM E2E test."),
]


@pytest.fixture(autouse=True)
def isolated_aether_state(tmp_path, monkeypatch):
    import aether_dft.knowledge as knowledge
    import aether_dft.paths as paths
    import aether_dft.project_state as project_state

    projects_dir = tmp_path / "projects"
    knowledge_dir = tmp_path / "knowledge_base"
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", knowledge_dir)
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", knowledge_dir)
    monkeypatch.setattr(knowledge, "KNOWLEDGE_BASE_DIR", knowledge_dir)


def test_live_model_uses_reasoning_tools_for_adsorption_candidate_plan(tmp_path):
    """Live smoke test for the roadmap's "taught the model" claim.

    Default CI skips it.  To run locally:
      AETHER_RUN_LLM_TESTS=1 pytest tests/test_llm_authored_adsorption_e2e.py -q
    Requires the configured provider API key (DeepSeek by default).
    """
    api_key = resolve_api_key(Path.cwd(), aliases=("deepseek",), env_names=("DEEPSEEK_API_KEY",))
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY or local api_keys.local.json entry is required.")

    prompt = (
        "只做规划，不生成文件：为 H2O 在 Pt(111) 上生成吸附候选前，"
        "请先调用吸附物化学先验、体系知识搜索，并创建 adsorption_candidate_plan。"
    )
    record = run_agent_once(
        prompt,
        project="llm-e2e-adsorption",
        max_steps=8,
        max_tokens=2000,
        permission_mode="dev",
    )
    tool_names = [item["name"] for item in record.get("tool_executions", [])]
    assert "adsorbate_chemistry_hint" in tool_names
    assert "knowledge_search_for_system" in tool_names
    assert "adsorption_candidate_plan" in tool_names

    plan_results = [
        item.get("result", {})
        for item in record.get("tool_executions", [])
        if item.get("name") == "adsorption_candidate_plan"
    ]
    assert plan_results
    plan = plan_results[-1].get("plan") or {}
    assert len(str(plan.get("rationale") or "")) >= 30
    assert plan.get("target_sites")

    events_path = Path(record["record_path"]).with_name("llm-e2e-tool-trace.json")
    events_path.write_text(json.dumps({"tool_names": tool_names, "record": record}, ensure_ascii=False, indent=2), encoding="utf-8")
