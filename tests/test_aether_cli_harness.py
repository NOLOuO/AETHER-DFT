from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from aether_dft import cli
from aether_dft.context import build_context_payload, render_context_markdown
from aether_dft.harness import preflight, require_permission
from dft_shared.structure_analyzer.tool_registry import list_structure_tools


def test_context_payload_contains_openai_compatible_model():
    payload = build_context_payload()
    assert payload["model"]["protocol"] == "openai-compatible"
    assert payload["entrypoints"]["dft_mainline"]
    rendered = render_context_markdown(payload)
    assert "AETHER-DFT Context Snapshot" in rendered
    assert "OpenAI-compatible" in rendered or "openai-compatible" in rendered


def test_harness_preflight_has_required_surface():
    payload = preflight()
    assert payload["checks"]["config/system_prompt.md"] is True
    assert payload["checks"]["dft_app"] is True
    assert payload["checks"]["dft_shared"] is True


def test_cli_harness_real_case_uses_validation_runner(monkeypatch, capsys):
    captured = {}

    def fake_validation(**kwargs):
        captured.update(kwargs)
        return {
            "status": "ok",
            "project": kwargs["project"],
            "model_id": kwargs["model_id"],
            "tool_names": ["project_state_read", "cluster_profile_list", "cluster_probe"],
            "missing_evidence": [],
            "record_path": "transcript.jsonl",
            "report_path": "report.json",
            "response": "真实验收完成。",
        }

    monkeypatch.setattr("aether_dft.real_case_validation.run_real_case_validation", fake_validation)

    assert cli.main(["harness", "real-case", "--project", "MCH-Pt-Br", "--model", "deepseek", "--cluster-alias", "szhang"]) == 0

    out = capsys.readouterr().out
    assert "real-case validation: ok" in out
    assert "project_state_read" in out
    assert captured["project"] == "MCH-Pt-Br"
    assert captured["cluster_alias"] == "szhang"
    assert captured["include_outcar"] is False


def test_permission_guard_blocks_destructive_operations():
    payload = require_permission("delete-test", destructive=True)
    assert payload["allowed"] is False


def test_cli_progress_printer_renders_parallel_heartbeat_and_context_guard(capsys):
    printer = cli.make_chat_progress_printer()

    printer({"event": "turn_start", "model_id": "deepseek:deepseek-v4-pro"})
    printer({"event": "model_request", "step": 1, "max_steps": 4})
    printer({"event": "tool_parallel_start", "step": 1, "count": 2, "names": ["cluster_my_jobs", "cluster_job_status_brief"]})
    printer({"event": "tool_start", "step": 1, "name": "cluster_my_jobs", "arguments": "{}", "parallel": True})
    printer({"event": "tool_progress", "step": 1, "name": "cluster_my_jobs", "elapsed_seconds": 2.1, "message": "仍在等待 SSH"})
    printer({"event": "tool_finish", "step": 1, "name": "cluster_my_jobs", "status": "ok", "microcompacted": True, "persisted_output_path": "trace.json"})
    printer({"event": "token_guard_finalize", "step": 2, "usage_ratio": 0.91})

    out = capsys.readouterr().out
    assert "parallel tools" in out
    assert "still running" in out
    assert "microcompacted" in out
    assert "context guard" in out


def test_structure_tool_registry_has_ten_project_relevant_tools():
    tools = list_structure_tools()
    assert len(tools) >= 10
    names = {item["name"] for item in tools}
    assert "xsd_to_poscar" in names
    assert "adsorption_candidates" in names
    assert "result_explain_bridge" in names


def test_cli_model_current_smoke(capsys):
    assert cli.main(["model", "current"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["model_id"] in {"deepseek:deepseek-v4-pro", "bailian:qwen3.7-max"}


def test_cli_doctor_json_is_machine_readable(capsys):
    assert cli.main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["program"]["name"] == "AETHER-DFT"
    assert payload["python"]["required"] == "3.12.x or 3.13.x"


def test_launcher_keeps_project_venv_dependency_isolated():
    launcher = Path("aether.ps1").read_text(encoding="utf-8")
    assert '+= "--system-site-packages"' not in launcher
    assert "Test-VenvUsesSharedSitePackages" in launcher
    assert "安装 AETHER-DFT 运行依赖到项目 .venv" in launcher


def test_launcher_accepts_312_or_313_without_forcing_313():
    launcher = Path("aether.ps1").read_text(encoding="utf-8")
    assert "$SupportedMinors = @(12, 13)" in launcher
    assert "$env:AETHER_PYTHON" in launcher
    assert launcher.index('@{ exe = "python"; args = @() }') < launcher.index('@{ exe = "py"; args = @("-3.13") }')


def test_cli_model_list_alias(capsys):
    assert cli.main(["model", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["current_model_id"]
    assert any(item["provider_id"] in {"deepseek", "bailian"} for item in payload["models"])


def test_key_store_does_not_read_personal_workspace_fallbacks(tmp_path, monkeypatch):
    from dft_app.llm.key_store import load_api_keys

    monkeypatch.delenv("AETHER_DFT_API_KEYS_PATHS", raising=False)
    assert load_api_keys(tmp_path) == {}


def test_cli_model_smoke_summarizes_required_tool(monkeypatch, capsys):
    captured = {}

    def fake_run_agent_once(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {
            "finish_reason": "stop",
            "response": "ok",
            "record_path": "trace.jsonl",
            "tool_executions": [{"name": "project_state_read", "result": {"status": "ok"}}],
        }

    monkeypatch.setattr("aether_dft.agent.run_agent_once", fake_run_agent_once)

    assert cli.main(["model", "smoke", "--model", "bailian:qwen3.7-max", "--project", "demo"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["model_id"] == "bailian:qwen3.7-max"
    assert payload["tool_names"] == ["project_state_read"]
    assert captured["kwargs"]["model_id"] == "bailian:qwen3.7-max"
    assert "project=demo" in captured["prompt"]


def test_cli_preload_reports_context_sources(capsys):
    assert cli.main(["preload", "--project", "MCH-Pt-Br", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["model"]["model_id"]
    assert payload["project"]["slug"] == "MCH-Pt-Br"
    assert payload["prompt_preload"]["discussion_tool_count"] > 0
    assert payload["prompt_preload"]["execution_tool_count"] >= payload["prompt_preload"]["discussion_tool_count"]
    assert payload["prompt_preload"]["job_watch_digest_loaded"] is True
    assert "job_watch_digest" in payload["cluster"]
    assert "aether-dft chat --project MCH-Pt-Br" in payload["next_user_entrypoints"]


def test_cli_preload_human_output_mentions_files(capsys):
    assert cli.main(["preload", "--project", "MCH-Pt-Br"]) == 0
    out = capsys.readouterr().out
    assert "AETHER preload ready" in out
    assert "research files preloaded" in out
    assert "job_watch_digest" in out
    assert "tools discussion/execution" in out


def test_cli_outcar_find_lists_remote_outcars(monkeypatch, capsys):
    from dft_app.remote import RemoteExecutionResult

    class FakeRunner:
        def find_remote_outcars(self, *, search_root=None, limit=20, max_depth=8):
            assert search_root == "~/research"
            assert limit == 2
            return RemoteExecutionResult(
                "ok",
                "找到 1 个 OUTCAR。",
                {
                    "outcars": [
                        {
                            "modified": "2026-06-06 10:00",
                            "size": 123,
                            "path": "/home/szhang/research/demo/run/OUTCAR",
                            "run_root": "/home/szhang/research/demo/run",
                        }
                    ]
                },
            )

    monkeypatch.setattr("dft_app.remote.SSHRemoteRunner", FakeRunner)

    assert cli.main(["outcar", "find", "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "最近 OUTCAR" in out
    assert "/home/szhang/research/demo/run/OUTCAR" in out


def test_cli_outcar_analyze_pulls_and_interprets_latest(monkeypatch, tmp_path, capsys):
    from dft_app.remote import RemoteExecutionResult

    class FakeRunner:
        def find_remote_outcars(self, *, search_root=None, limit=20, max_depth=8):
            return RemoteExecutionResult(
                "ok",
                "找到 1 个 OUTCAR。",
                {"outcars": [{"path": "/home/szhang/research/demo/freq/OUTCAR"}]},
            )

        def pull_remote_outcar_context(self, remote_outcar_path, local_target_dir):
            local = Path(local_target_dir)
            local.mkdir(parents=True, exist_ok=True)
            (local / "OUTCAR").write_text(
                "\n".join(
                    [
                        " free  energy   TOTEN  =      -640.0 eV",
                        " Eigenvectors and eigenvalues of the dynamical matrix",
                        " 1 f  =   17.0 THz",
                        " General timing and accounting informations for this job",
                    ]
                ),
                encoding="utf-8",
            )
            return RemoteExecutionResult(
                "synced",
                "ok",
                {
                    "remote_outcar_path": remote_outcar_path,
                    "local_target_dir": str(local),
                    "downloaded": [str(local / "OUTCAR")],
                },
            )

    monkeypatch.setattr("dft_app.remote.SSHRemoteRunner", FakeRunner)

    assert cli.main(["outcar", "analyze", "--output-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "frequency_finished_no_imaginary_modes" in out
    assert "last TOTEN: -640.0 eV" in out


def test_cli_structure_tools_smoke(capsys):
    assert cli.main(["structure", "tools"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert len(payload["structure_tools"]) >= 10


def test_cli_quick_start_names_program_model_and_version(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "AETHER-DFT" in out
    assert "0.1.0" in out
    assert "Model:" in out
    assert "deepseek:deepseek-v4-pro" in out
    assert "直接输入自然语言即可" in out
    assert "/model" in out
    assert "/project" in out
    assert "/resume" in out


def test_cli_no_args_enters_interactive_when_tty(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "Session Info" in out
    assert "Program: " in out
    assert "Resume this session with:" in out
    assert "aether chat --resume --session-id" in out
    assert "resumed:" not in out


def test_cli_demo_is_display_only_not_second_repl(capsys):
    assert cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Session Info" in out
    assert "display-only" in out
    assert "aether" in out


def test_cli_chat_help_is_product_friendly(capsys):
    assert cli.main(["chat", "--help"]) == 0
    out = capsys.readouterr().out
    assert "Just type natural language" in out
    assert "/auto" in out
    assert "--material" not in out
    assert "--preferred-site" not in out


def test_cli_natural_language_resume_inherits_session_project(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths
    import aether_dft.session_store as session_store

    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)

    store = session_store.AetherSessionStore()
    session_id = store.start_session(project="MCH-Pt-Br", first_prompt="first")
    store.append_turn(session_id, {"project": "MCH-Pt-Br", "prompt": "first", "response": "ok"})

    captured = {}

    def fake_ask_once(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {
            "response": "已续接项目上下文。",
            "record_path": "trace.jsonl",
            "tool_executions": [],
            "progress": {"next_steps": []},
        }

    monkeypatch.setattr(cli, "ask_once", fake_ask_once)

    assert cli.main(["这个体系下一步怎么做"]) == 0

    assert captured["kwargs"]["session_id"] == session_id
    assert captured["kwargs"]["project"] == "MCH-Pt-Br"
    assert "已续接项目上下文" in capsys.readouterr().out


def test_cli_natural_language_model_failure_is_user_readable(monkeypatch, capsys):
    def fake_ask_once(prompt, **kwargs):
        raise RuntimeError("DeepSeek 接口调用失败: Request timed out.")

    monkeypatch.setattr(cli, "ask_once", fake_ask_once)

    assert cli.main(["看看", "怎么样了"]) == 1
    out = capsys.readouterr().out
    assert "模型调用失败" in out
    assert "Request timed out" in out
    assert "Traceback" not in out


def test_cli_interactive_status_and_context_shortcuts(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/status", "/context", "/help", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert '"program": "AETHER-DFT"' in out
    assert '"usable_context_tokens": 936000' in out
    assert '"context_usage_percent"' in out
    assert "上下文充足" in out
    assert "AETHER interactive chat" in out


def test_cli_compact_command_compacts_current_session(tmp_path, capsys):
    from aether_dft.session_store import AetherSessionStore

    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")
    for index in range(5):
        store.append_turn(session_id, {"project": "demo", "prompt": f"p{index}", "response": f"r{index}"})

    cli.handle_chat_compact_command("/compact 2", session_store=store, session_id=session_id)

    out = capsys.readouterr().out
    state = store.load_state(session_id)
    assert "compact complete" in out
    assert state["compacted_turn_count"] == 3
    assert state["compact_keep_recent_turns"] == 2


def test_cli_interactive_model_command_opens_selector(monkeypatch, tmp_path, capsys):
    import aether_dft.model_catalog as model_catalog
    import aether_dft.paths as paths

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(model_catalog, "PREFERENCES_PATH", tmp_path / "runtime" / "model-preferences.json")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/model", "1", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat"]) == 0
    out = capsys.readouterr().out
    assert "select model" in out
    assert "model switched" in out
    assert "bailian:qwen3.7-max" in out
    assert '"model": "bailian:qwen3.7-max"' in out


def test_cli_interactive_slash_opens_command_palette(monkeypatch, tmp_path, capsys):
    import aether_dft.model_catalog as model_catalog
    import aether_dft.paths as paths

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(model_catalog, "PREFERENCES_PATH", tmp_path / "runtime" / "model-preferences.json")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/", "1", "1", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat"]) == 0
    out = capsys.readouterr().out
    assert "slash commands" in out
    assert "/model" in out
    assert "select model" in out
    assert "model switched" in out
    assert '"model": "bailian:qwen3.7-max"' in out


def test_cli_interactive_sessions_and_resume_command(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths
    import aether_dft.session_store as session_store

    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)

    store = session_store.AetherSessionStore()
    older = store.start_session(project="MCH-Pt-Br", first_prompt="old")
    store.append_turn(older, {"project": "MCH-Pt-Br", "prompt": "old", "response": "old response"})
    newer = store.start_session(project="Other", first_prompt="new")
    store.append_turn(newer, {"project": "Other", "prompt": "new", "response": "new response"})

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/sessions", f"/resume {older}", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat"]) == 0
    out = capsys.readouterr().out
    assert "recent sessions" in out
    assert older in out
    assert newer in out
    assert f"resumed" in out
    assert f'"id": "{older}"' in out
    assert '"project": "MCH-Pt-Br"' in out


def test_cli_resume_all_search_can_cross_project(tmp_path, capsys):
    from aether_dft.session_store import AetherSessionStore

    store = AetherSessionStore(tmp_path / "sessions")
    current = store.start_session(project="MCH-Pt-Br", first_prompt="current")
    other = store.start_session(project="Other", first_prompt="other surface discussion")
    store.append_turn(other, {"project": "Other", "prompt": "other surface discussion", "response": "other response"})
    args = argparse.Namespace(project="MCH-Pt-Br")

    resumed = cli.handle_chat_resume_command("/resume all surface", args, store, current)

    out = capsys.readouterr().out
    assert resumed == other
    assert "resumed" in out
    assert args.project == "Other"


def test_cli_interactive_resume_command_opens_selector(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths
    import aether_dft.session_store as session_store

    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)

    store = session_store.AetherSessionStore()
    older = store.start_session(project="MCH-Pt-Br", first_prompt="old")
    store.append_turn(older, {"project": "MCH-Pt-Br", "prompt": "old", "response": "old response"})
    newer = store.start_session(project="Other", first_prompt="new")
    store.append_turn(newer, {"project": "Other", "prompt": "new", "response": "new response"})

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/resume", "2", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat"]) == 0
    out = capsys.readouterr().out
    assert "resume session" in out
    assert older in out
    assert newer in out
    assert f'"id": "{older}"' in out
    assert '"project": "MCH-Pt-Br"' in out


def test_cli_failed_turn_can_continue_from_pending(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["继续处理这个体系", "/status", "/continue", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    calls = {"count": 0}

    def fake_ask_once(prompt, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary provider timeout")
        return {
            "prompt": prompt,
            "response": "继续完成了。",
            "record_path": str(tmp_path / "record.jsonl"),
            "elapsed_seconds": 0.1,
            "tool_executions": [],
            "progress": {"next_steps": []},
        }

    monkeypatch.setattr(cli, "ask_once", fake_ask_once)

    assert cli.main(["chat"]) == 0
    out = capsys.readouterr().out
    assert "模型调用失败" in out
    assert '"pending_turn"' in out
    assert "continue>" in out
    assert "继续完成了。" in out
    assert calls["count"] == 2


def test_cli_history_and_rename_commands(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths
    import aether_dft.session_store as session_store

    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)

    store = session_store.AetherSessionStore()
    session_id = store.start_session(project="demo", first_prompt="old title")
    store.append_turn(session_id, {"project": "demo", "prompt": "分析 OUTCAR", "response": "收敛正常。"})

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter([f"/resume {session_id}", "/history OUTCAR", "/rename 新的科研会话标题", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat", "--project", "demo"]) == 0
    out = capsys.readouterr().out
    assert "history matches for 'OUTCAR'" in out
    assert "分析 OUTCAR" in out
    assert "renamed" in out
    assert '"title": "新的科研会话标题"' in out


def test_cli_project_command_fuzzy_matches_research_workspace(monkeypatch, tmp_path, capsys):
    import aether_dft.research_workspace as research_workspace
    from aether_dft.session_store import AetherSessionStore

    research_root = tmp_path / "research"
    (research_root / "Common").mkdir(parents=True)
    (research_root / "MCH-Pt-Br").mkdir()
    monkeypatch.setattr(research_workspace, "RESEARCH_ROOT", research_root)
    monkeypatch.setattr(research_workspace, "COMMON_DIR", research_root / "Common")

    store = AetherSessionStore(tmp_path / "sessions")
    current = store.start_session(project=None)
    args = argparse.Namespace(project=None)

    session_id = cli.handle_chat_project_command("/project mch", args, store, current)

    out = capsys.readouterr().out
    assert args.project == "MCH-Pt-Br"
    assert session_id != current
    assert "project switched" in out


def test_cli_interactive_resume_is_scoped_to_current_project(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths
    import aether_dft.session_store as session_store

    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)

    store = session_store.AetherSessionStore()
    target = store.start_session(project="MCH-Pt-Br", first_prompt="target")
    store.append_turn(target, {"project": "MCH-Pt-Br", "prompt": "target", "response": "target response"})
    other = store.start_session(project="Other", first_prompt="other")
    store.append_turn(other, {"project": "Other", "prompt": "other", "response": "other response"})

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/resume", "1", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat", "--project", "MCH-Pt-Br"]) == 0
    out = capsys.readouterr().out
    assert "resume session" in out
    assert target in out
    assert other not in out
    assert f'"id": "{target}"' in out
    assert '"project": "MCH-Pt-Br"' in out


def test_cli_interactive_resume_searches_current_project_sessions(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths
    import aether_dft.session_store as session_store

    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)

    store = session_store.AetherSessionStore()
    target = store.start_session(project="MCH-Pt-Br", first_prompt="TS barrier discussion")
    store.append_turn(target, {"project": "MCH-Pt-Br", "prompt": "TS barrier discussion", "response": "barrier response"})
    other = store.start_session(project="Other", first_prompt="TS barrier discussion")
    store.append_turn(other, {"project": "Other", "prompt": "TS barrier discussion", "response": "other response"})

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/resume barrier", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat", "--project", "MCH-Pt-Br"]) == 0
    out = capsys.readouterr().out
    assert f'"id": "{target}"' in out
    assert other not in out
    assert '"project": "MCH-Pt-Br"' in out


def test_cli_interactive_project_command_opens_selector(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(cli, "list_projects", lambda: [{"slug": "MCH-Pt-Br", "title": "MCH on Pt"}])
    monkeypatch.setattr(cli, "load_project", lambda slug: {"slug": slug, "title": "MCH on Pt"})
    monkeypatch.setattr(cli, "read_project_context", lambda slug: f"context for {slug}")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/project", "1", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat"]) == 0
    out = capsys.readouterr().out
    assert "select project" in out
    assert "project switched" in out
    assert "MCH-Pt-Br" in out
    assert '"project": "MCH-Pt-Br"' in out


def test_cli_interactive_permission_command_opens_selector(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths
    import aether_dft.permissions as permissions

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(permissions, "PERMISSIONS_PATH", tmp_path / "runtime" / "permissions.json")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/permission", "2", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat"]) == 0
    out = capsys.readouterr().out
    assert "select permission" in out
    assert "permission switched" in out
    assert '"mode": "ask"' in out


def test_cli_interactive_new_command_starts_fresh_session(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/new", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat", "--project", "MCH-Pt-Br"]) == 0
    out = capsys.readouterr().out
    assert "new session" in out
    assert '"project": "MCH-Pt-Br"' in out
    assert '"turn_count": 0' in out


def test_cli_mainline_prints_explicit_workflow(capsys):
    assert cli.main(["mainline"]) == 0
    out = capsys.readouterr().out
    assert "AETHER-DFT mainline" in out
    assert "evidence-led research chat" in out


def test_package_module_entry_prints_help():
    result = subprocess.run(
        [sys.executable, "-m", "aether_dft", "--help"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    assert "mainline" in output
    assert "chat" in output


def test_doctor_payload_names_program_model_and_version(capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    json_text = out[out.index("{") : out.rindex("}") + 1]
    payload = json.loads(json_text)
    assert payload["program"]["name"] == "AETHER-DFT"
    assert payload["program"]["command"] == "aether"
    assert payload["program"]["version"] == "0.1.0"
    assert payload["effective_model"]["model_id"] == "deepseek:deepseek-v4-pro"
    assert payload["effective_model"]["context_window"] == 1000000
    assert payload["python"]["required"] == "3.12.x or 3.13.x"
    assert payload["python"]["ok"] is True
    assert "executable" not in payload["python"]
    assert "conda" not in out.lower()
    assert "uv" not in out.lower()
