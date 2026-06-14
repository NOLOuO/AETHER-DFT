from __future__ import annotations

import json
from pathlib import Path

from aether_dft import cli
from aether_dft.session_store import AetherSessionStore, SESSION_CONTEXT_MAX_CHARS
from aether_dft.tool_registry import AetherToolRegistry, list_registered_tools
from aether_dft.runtime_harness.tool_registry import ToolRegistry


def test_session_store_persists_and_resumes(tmp_path: Path):
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo", first_prompt="first")
    transcript_path = store.append_turn(
        session_id,
        {
            "project": "demo",
            "prompt": "计算 H2O 吸附",
            "response": "下一步生成 slab。",
        },
    )
    assert transcript_path.exists()
    payload = store.resume_payload(session_id=session_id)
    assert payload["status"] == "ok"
    assert payload["state"]["turn_count"] == 1
    assert payload["state"]["title"] == "first"
    assert payload["recent_turns"][0]["record"]["prompt"] == "计算 H2O 吸附"


def test_session_title_uses_first_real_prompt_when_session_starts_empty(tmp_path: Path):
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")
    store.append_turn(
        session_id,
        {
            "project": "demo",
            "prompt": "讨论一下 H2O 在 Pt(111) 上应该先算哪些吸附构型",
            "response": "先比较 top/bridge/hollow。",
        },
    )

    payload = store.resume_payload(session_id=session_id)
    assert payload["state"]["title"].startswith("H2O 在 Pt(111)")
    assert store.list_sessions(project="demo")[0].title.startswith("H2O 在 Pt(111)")


def test_session_store_mirrors_project_session_reference(tmp_path: Path, monkeypatch):
    import aether_dft.research_workspace as research_workspace

    research_root = tmp_path / "research"
    project_root = research_root / "MCH-Pt-Br"
    project_root.mkdir(parents=True)
    (research_root / "Common").mkdir()
    monkeypatch.setattr(research_workspace, "RESEARCH_ROOT", research_root)
    monkeypatch.setattr(research_workspace, "COMMON_DIR", research_root / "Common")

    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="MCH-Pt-Br", first_prompt="first")
    transcript_path = store.append_turn(
        session_id,
        {"project": "MCH-Pt-Br", "prompt": "分析 MCH/Pt", "response": "先查 OUTCAR。"},
    )

    ref_path = project_root / ".aether" / "sessions" / f"{session_id}.json"
    index_path = project_root / ".aether" / "sessions" / "sessions.json"
    assert ref_path.exists()
    assert index_path.exists()
    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    assert reference["session_id"] == session_id
    assert reference["project"] == "MCH-Pt-Br"
    assert reference["title"] == "first"
    assert reference["turn_count"] == 1
    assert reference["canonical_transcript"] == str(transcript_path)
    assert store.project_session_reference_path(session_id) == ref_path
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index[0]["session_id"] == session_id


def test_tool_registry_exposes_builtin_agent_tools():
    tools = list_registered_tools()
    names = {item["name"] for item in tools}
    assert "cluster_probe" in names
    assert "recommend_next_tasks" in names
    assert AetherToolRegistry().run_tool("recommend_next_tasks", {"focus": "adsorption"})["result"]["status"] == "ok"


def test_session_context_auto_compacts_before_prompt_budget(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AETHER_DFT_CONTEXT_MAX_CHARS", "12000")
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")
    for index in range(90):
        store.append_turn(
            session_id,
            {
                "project": "demo",
                "prompt": f"用户第 {index} 轮 " + "x" * 1200,
                "response": f"助手第 {index} 轮 " + "y" * 1200,
                "tool_executions": [{"name": "project_state_read"}],
            },
        )
    state = store.load_state(session_id)
    context = store.build_session_context(session_id)
    assert state["compacted_turn_count"] > 0
    assert "Compacted Session Summary" in context
    assert len(context) <= SESSION_CONTEXT_MAX_CHARS


def test_session_context_includes_compact_tool_trail_without_large_results(tmp_path: Path):
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")
    store.append_turn(
        session_id,
        {
            "project": "demo",
            "prompt": "只读检查 H2O/Pt(111)",
            "response": "已查到 H2O O-down 先验。",
            "tool_executions": [
                {
                    "name": "adsorbate_chemistry_hint",
                    "arguments": {"adsorbate": "H2O"},
                    "result": {"status": "ok", "payload": "x" * 5000},
                },
                {
                    "name": "aether_discover_tools",
                    "arguments": {"category": "structure_modeling", "query": "Pt slab H2O"},
                    "result": {"status": "ok", "schemas": [{"large": "y" * 5000}]},
                },
            ],
        },
    )

    context = store.build_session_context(session_id)

    assert "tool_trail:" in context
    assert "adsorbate_chemistry_hint(ok; adsorbate=H2O)" in context
    assert "aether_discover_tools(ok; category=structure_modeling" in context
    assert "x" * 200 not in context
    assert "y" * 200 not in context


def test_permission_mode_blocks_write_tools_in_ask_mode():
    result = ToolRegistry(permission_mode="ask").run_tool(
        "project_progress_append",
        {"project": "demo", "completed": ["x"]},
    )
    assert result["result"]["status"] == "permission_required"


def test_cli_session_and_tools_smoke(tmp_path: Path, monkeypatch, capsys):
    import aether_dft.paths as paths
    import aether_dft.session_store as session_store

    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)
    # ensure_runtime_dir reads paths.RUNTIME_DIR dynamically; the class import is enough.
    store = session_store.AetherSessionStore()
    session_id = store.start_session(project="demo", first_prompt="hello")
    store.append_turn(session_id, {"project": "demo", "prompt": "hello", "response": "world"})

    assert cli.main(["session", "list", "--project", "demo"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sessions"][0]["session_id"] == session_id

    assert cli.main(["session", "resume", session_id]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["session_id"] == session_id

    assert cli.main(["tools", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(item["name"] == "recommend_next_tasks" for item in payload["tools"])

    assert cli.main(["tools", "run", "recommend_next_tasks", "--arguments", "{\"focus\":\"adsorption\"}"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["status"] == "ok"
