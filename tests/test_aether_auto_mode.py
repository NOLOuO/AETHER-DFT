from __future__ import annotations

import json
from pathlib import Path

from aether_dft import cli
from aether_dft.auto_mode import build_auto_mode_digest, configure_auto_mode, auto_mode_status, infer_research_goal
from aether_dft.runtime_harness.tool_registry import ToolRegistry


def _redirect_dirs(monkeypatch, tmp_path: Path) -> None:
    import aether_dft.paths as paths
    import aether_dft.project_state as project_state

    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_state, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")


def test_auto_mode_configure_persists_goal_and_schedules_followups(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)

    result = configure_auto_mode(
        project="demo",
        enabled=True,
        research_goal="找到 H2O 在 Pt(111) 上最稳定吸附构型并完成 DFT 验证",
        monitor_interval_hours=3,
        allow_cluster_submit=True,
    )

    assert result["status"] == "ok"
    state = auto_mode_status(project="demo")["state"]
    assert state["enabled"] is True
    assert state["monitor_interval_hours"] == 3
    assert state["allow_cluster_submit"] is True
    assert "H2O" in state["research_goal"]

    scheduled = auto_mode_status(project="demo")["scheduled_followups"]["followups"]
    auto_kinds = {(item.get("metadata") or {}).get("auto_kind") for item in scheduled}
    assert {"monitor", "daily_report"}.issubset(auto_kinds)

    digest = build_auto_mode_digest(project="demo")
    assert "Auto mode is ON" in digest
    assert "Autonomy contract" in digest
    assert "H2O" in digest


def test_auto_mode_requires_goal_when_enabled(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)

    result = configure_auto_mode(project="demo", enabled=True)

    assert result["status"] == "needs_goal"
    assert result["state"]["enabled"] is False


def test_auto_goal_can_be_inferred_from_existing_session(tmp_path: Path, monkeypatch):
    from aether_dft.session_store import AetherSessionStore

    _redirect_dirs(monkeypatch, tmp_path)
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo", first_prompt="研究目标：阐明 Br 修饰 Pt 上 MCH 脱氢最可能路径")
    store.append_turn(session_id, {"project": "demo", "prompt": "先比较 TS1/TS4", "response": "当前重点是势垒和中间体稳定性。"})

    inferred = infer_research_goal(project="demo", session_store=store, session_id=session_id)

    assert inferred["status"] == "ok"
    assert "MCH 脱氢" in inferred["goal"]
    assert inferred["source"].startswith(("current_session_state", "current_session_context", "session_summary"))


def test_auto_mode_tools_are_model_visible_and_permissioned(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)

    registry = ToolRegistry(permission_mode="never")
    names = {item["name"] for item in registry.list_tools()}

    assert {"auto_mode_status", "auto_mode_configure", "auto_mode_checkpoint"}.issubset(names)
    discussion_names = {item["function"]["name"] for item in registry.openai_tool_schemas(interaction_mode="discussion")}
    assert "auto_mode_status" in discussion_names
    assert "auto_mode_checkpoint" in discussion_names

    blocked = ToolRegistry(permission_mode="ask").run_tool(
        "auto_mode_configure",
        {"project": "demo", "enabled": True, "research_goal": "test"},
    )
    assert blocked["result"]["status"] == "permission_required"


def test_auto_cli_on_status_off(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)

    assert cli.main(["auto", "on", "研究 MCH 在 Br/Pt 上脱氢路径", "--project", "demo", "--allow-cluster-submit"]) == 0
    enabled = json.loads(capsys.readouterr().out)
    assert enabled["state"]["enabled"] is True

    assert cli.main(["auto", "status", "--project", "demo"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["state"]["research_goal"] == "研究 MCH 在 Br/Pt 上脱氢路径"
    assert status["state"]["allow_cluster_submit"] is True

    assert cli.main(["auto", "off", "--project", "demo"]) == 0
    disabled = json.loads(capsys.readouterr().out)
    assert disabled["state"]["enabled"] is False


def test_interactive_slash_auto_enables_goal(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths
    import aether_dft.project_state as project_state

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_state, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/auto 研究 CO 在 Pt(111) 的吸附与扩散", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat", "--project", "demo"]) == 0
    out = capsys.readouterr().out
    assert "auto mode" in out
    assert "ON" in out
    assert '"enabled": true' in out
    assert "CO 在 Pt(111)" in out


def test_interactive_slash_auto_toggles_and_infers_goal(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths
    import aether_dft.project_state as project_state
    from aether_dft.session_store import AetherSessionStore

    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_state, "PROJECTS_DIR", tmp_path / "projects")
    store = AetherSessionStore()
    existing = store.start_session(project="demo", first_prompt="研究目标：确定 H2O/Pt(111) 最稳定吸附构型")
    store.append_turn(existing, {"project": "demo", "prompt": "比较 top/hollow", "response": "继续算吸附能。"})

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/auto", "/status", "/auto", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat", "--project", "demo"]) == 0
    out = capsys.readouterr().out
    assert "auto goal inferred" in out
    assert "H2O/Pt(111)" in out
    assert '"enabled": true' in out
    assert '"enabled": false' in out
