from __future__ import annotations

import json
from pathlib import Path

from aether_dft import cli
from aether_dft.session_store import AetherSessionStore, SESSION_CONTEXT_MAX_CHARS
from aether_dft.session_search import rank_session_summaries
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
    assert payload["state"]["title"].startswith("讨论一下 H2O 在 Pt(111)")
    assert store.list_sessions(project="demo")[0].title.startswith("讨论一下 H2O 在 Pt(111)")


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


def test_session_manual_compact_keeps_recent_turns_without_deleting_transcript(tmp_path: Path):
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")
    for index in range(6):
        store.append_turn(
            session_id,
            {
                "project": "demo",
                "prompt": f"prompt {index}",
                "response": f"response {index}",
            },
        )

    result = store.compact_session(session_id, keep_recent=2)
    context = store.build_session_context(session_id)
    transcript = store.read_transcript(session_id, limit=10)

    assert result["status"] == "ok"
    assert result["compacted_turn_count"] == 4
    assert len(transcript) == 6
    assert "Compacted Session Summary" in context
    assert "prompt 0" in context
    assert "turn 5 user: prompt 4" in context
    assert "turn 6 user: prompt 5" in context
    assert "turn 3 user: prompt 2" not in context


def test_session_pending_turn_is_resume_metadata_not_transcript(tmp_path: Path):
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")

    pending = store.record_pending_turn(session_id, prompt="继续分析最新 OUTCAR", project="demo", model_id="bailian:qwen")
    context = store.build_session_context(session_id)

    assert pending["status"] == "in_progress"
    assert store.read_transcript(session_id, limit=10) == []
    assert "## Pending Turn" in context
    assert "继续分析最新 OUTCAR" in context

    store.append_turn(session_id, {"project": "demo", "prompt": "继续分析最新 OUTCAR", "response": "已经完成。"})

    assert store.pending_turn(session_id) is None
    assert "Pending Turn" not in store.build_session_context(session_id)
    assert len(store.read_transcript(session_id, limit=10)) == 1


def test_session_search_and_rename(tmp_path: Path):
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo", first_prompt="初始讨论")
    store.append_turn(session_id, {"project": "demo", "prompt": "讨论 Pt slab", "response": "需要检查表面。"})
    store.append_turn(session_id, {"project": "demo", "prompt": "分析 OUTCAR 收敛", "response": "电子步已收敛。"})

    matches = store.search_transcript(session_id, query="OUTCAR", limit=5)
    renamed = store.rename_session(session_id, "Pt slab OUTCAR follow-up")

    assert len(matches) == 1
    assert matches[0]["record"]["prompt"] == "分析 OUTCAR 收敛"
    assert renamed["title"] == "Pt slab OUTCAR follow-up"
    assert store.list_sessions(project="demo")[0].title == "Pt slab OUTCAR follow-up"


def test_semantic_session_search_uses_metadata_catalog_not_fixed_phrases(tmp_path: Path):
    store = AetherSessionStore(tmp_path / "sessions")
    water = store.start_session(project="Pt-H2O", first_prompt="讨论 H2O 在 Pt(111) 的吸附构型")
    store.append_turn(
        water,
        {
            "project": "Pt-H2O",
            "prompt": "比较 top bridge hollow 的初始构型",
            "response": "优先 O-down hollow 和 atop，后续看吸附能。",
        },
    )
    barrier = store.start_session(project="MCH-Pt-Br", first_prompt="MCH 脱氢 NEB 势垒")
    store.append_turn(
        barrier,
        {
            "project": "MCH-Pt-Br",
            "prompt": "看 Br 修饰 Pt 上 C-H 活化过渡态",
            "response": "下一步检查 NEB images。",
        },
    )

    sessions = store.list_sessions(limit=10)

    def selector(query: str, catalog: list[dict], max_results: int) -> list[dict]:
        assert "water adsorption on platinum" in query
        assert any(item["session_id"] == water for item in catalog)
        assert any("O-down" in item["transcript_excerpt"] for item in catalog)
        selected = next(item for item in catalog if item["session_id"] == water)
        return [{"rank": selected["rank"], "reason": "语义匹配 water/Pt adsorption"}]

    result = rank_session_summaries(
        "water adsorption on platinum",
        sessions,
        transcript_loader=lambda sid: store.read_transcript(sid, limit=8),
        selector=selector,
    )

    assert result["selection_method"] == "semantic"
    assert result["matches"][0].summary.session_id == water
    assert result["matches"][0].reason == "语义匹配 water/Pt adsorption"


def test_session_resume_recovers_malformed_transcript_rows(tmp_path: Path):
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo", first_prompt="resume recovery")
    transcript = store._transcript_path(session_id)
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(
        "\n".join(
            [
                "{bad json",
                json.dumps({"type": "turn", "record": "not a dict"}, ensure_ascii=False),
                json.dumps(
                    {
                        "type": "turn",
                        "record": {
                            "project": "demo",
                            "prompt": "检查 OUTCAR",
                            "response": "电子步收敛。",
                            "tool_executions": [
                                {"arguments": {"job_id": "1"}, "result": {"status": "ok"}},
                                {"name": "cluster_job_partial_outcar", "arguments": {"job_id": "1"}, "result": {"status": "ok"}},
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"type": "turn", "record": {"prompt": "   ", "response": "  "}}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = store.resume_payload(session_id=session_id)
    recovery = payload["recovery"]

    assert payload["status"] == "ok"
    assert recovery["status"] == "recovered"
    assert recovery["invalid_json_rows"] == 1
    assert recovery["malformed_rows"] == 2
    assert recovery["skipped_tool_records"] == 1
    assert len(payload["recent_turns"]) == 1
    tools = payload["recent_turns"][0]["record"]["tool_executions"]
    assert [tool["name"] for tool in tools] == ["cluster_job_partial_outcar"]


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


def test_session_context_analysis_identifies_large_tool_results(tmp_path: Path):
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo", first_prompt="context analysis")
    store.append_turn(
        session_id,
        {
            "project": "demo",
            "prompt": "分析最新 OUTCAR",
            "response": "需要看收敛和能量。",
            "tool_executions": [
                {
                    "name": "cluster_job_partial_outcar",
                    "arguments": {"job_id": "123"},
                    "result": {"status": "ok", "outcar": "ENERGY\n" + "x" * 5000},
                }
            ],
        },
    )

    analysis = store.analyze_context(session_id)

    assert analysis["status"] == "ok"
    assert analysis["top_buckets"][0]["name"] == "tool_result_chars"
    assert analysis["top_tool_results"][0]["name"] == "cluster_job_partial_outcar"
    assert analysis["large_turns"][0]["turn"] == 1


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

    args_file = tmp_path / "tool-args.json"
    args_file.write_text(json.dumps({"focus": "cluster"}), encoding="utf-8")
    assert cli.main(["tools", "run", "recommend_next_tasks", "--arguments-file", str(args_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["arguments"]["focus"] == "cluster"
    assert payload["result"]["status"] == "ok"


def test_tools_run_reports_invalid_json_arguments(capsys):
    assert cli.main(["tools", "run", "recommend_next_tasks", "--arguments", "{bad json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "合法 JSON" in payload["message"]
