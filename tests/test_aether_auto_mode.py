from __future__ import annotations

import json
from pathlib import Path

from aether_dft import cli
from aether_dft.auto_mode import (
    build_auto_mode_digest,
    collect_due_auto_intents,
    complete_due_auto_intents,
    configure_auto_mode,
    auto_mode_status,
    infer_research_goal,
)
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
    policy = auto_mode_status(project="demo")["policy"]
    assert state["enabled"] is True
    assert state["monitor_interval_hours"] == 3
    assert state["allow_cluster_submit"] is True
    assert "H2O" in state["research_goal"]
    assert policy["computational_strategy"]["default_bias"].startswith("Enumerate diverse plausible candidates")

    scheduled = auto_mode_status(project="demo")["scheduled_followups"]["followups"]
    auto_kinds = {(item.get("metadata") or {}).get("auto_kind") for item in scheduled}
    assert {"initial_advance", "monitor", "daily_report"}.issubset(auto_kinds)
    initial = next(item for item in scheduled if (item.get("metadata") or {}).get("auto_kind") == "initial_advance")
    assert initial["interval_minutes"] is None
    assert "Do not wait for a manual tick" in initial["prompt"]

    digest = build_auto_mode_digest(project="demo")
    assert "Auto mode is ON" in digest
    assert "Autonomy contract" in digest
    assert "Human time is scarce; compute is the lever" in digest
    assert "enumerate candidates" in digest
    assert "H2O" in digest


def test_auto_mode_collects_due_work_for_background_loop(tmp_path: Path, monkeypatch):
    from aether_dft.followups import schedule_followup

    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(
        project="demo",
        enabled=True,
        research_goal="确定 H2O/Pt(111) 最稳定吸附构型并完成吸附能验证",
        monitor_interval_hours=4,
    )
    due_at = "2000-01-01T00:00:00+08:00"
    followup = schedule_followup(
        project="demo",
        title="Check H2O adsorption evidence",
        prompt="检查已有 H2O/Pt(111) 计算、缺口和下一步。",
        due_at=due_at,
        interval_minutes=240,
        metadata={"auto_mode": True, "auto_kind": "monitor"},
    )["followup"]

    plan = collect_due_auto_intents(project="demo", now="2000-01-01T01:00:00+08:00")

    assert plan["should_run"] is True
    assert followup["id"] in plan["followup_ids"]
    assert "AUTO MODE DUE WORK" in plan["prompt"]
    assert "Do not follow a fixed pipeline" in plan["prompt"]
    assert "enumerate a diverse candidate set" in plan["prompt"]
    assert "batch-submit" in plan["prompt"]
    assert "hand-perfecting a single model" in plan["prompt"]
    assert "auto_campaign_status/list" in plan["prompt"]
    assert "Register generated candidates" in plan["prompt"]
    assert "source_manifest_path" in plan["prompt"]
    assert "Completion condition" in plan["prompt"]
    assert "H2O/Pt(111)" in plan["prompt"]

    completed = complete_due_auto_intents(project="demo", followup_ids=plan["followup_ids"], note="tested")
    assert completed["status"] == "ok"
    assert any(item["status"] == "rescheduled" for item in completed["results"])


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


def test_interactive_slash_auto_run_words_do_not_trigger_manual_model_turn(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths
    import aether_dft.project_state as project_state
    from aether_dft.session_store import AetherSessionStore

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_state, "PROJECTS_DIR", tmp_path / "projects")
    store = AetherSessionStore()
    session_id = store.start_session(project="demo", first_prompt="研究目标：验证 CO/Pt 吸附构型")
    calls: list[str] = []

    def fake_turn(prompt, **kwargs):
        calls.append(prompt)
        return True, session_id

    monkeypatch.setattr(cli, "run_chat_model_turn", fake_turn)
    args = cli.argparse.Namespace(project="demo")

    ok, returned = cli.handle_chat_auto_command("/auto run", args, store, session_id)

    assert ok is True
    assert returned == session_id
    assert calls == []
    assert "不用手动推进" in capsys.readouterr().out


def test_auto_due_runner_invokes_model_and_reschedules_due_work(monkeypatch, tmp_path):
    from aether_dft.followups import schedule_followup
    from aether_dft.session_store import AetherSessionStore
    from aether_dft.auto_mode import checkpoint_auto_mode

    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="研究 MCH 在 Br/Pt 上脱氢路径")
    followup = schedule_followup(
        project="demo",
        title="Auto monitor now",
        prompt="检查集群任务和研究进展。",
        due_at="2000-01-01T00:00:00+08:00",
        interval_minutes=240,
        metadata={"auto_mode": True, "auto_kind": "monitor"},
    )["followup"]
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")
    captured: list[str] = []
    captured_args: list[cli.argparse.Namespace] = []

    def fake_runner(prompt, **kwargs):
        captured.append(prompt)
        captured_args.append(kwargs["args"])
        store.append_turn(
            session_id,
            {
                "project": "demo",
                "prompt": prompt,
                "response": "已检查集群状态并记录 checkpoint。",
                "tool_executions": [{"name": "cluster_remote_monitor", "result": {"status": "ok"}}],
            },
        )
        checkpoint_auto_mode(
            project="demo",
            observation="checked current evidence",
            decision="continue campaign",
            next_focus="build candidates",
        )
        return True, session_id

    args = cli.argparse.Namespace(project="demo")
    result = cli.run_auto_due_once(
        args=args,
        session_store=store,
        session_ref={"id": session_id},
        now="2000-01-01T01:00:00+08:00",
        turn_runner=fake_runner,
        quiet=True,
    )

    assert result["status"] == "ok"
    assert result["ran"] is True
    assert captured and "AUTO MODE DUE WORK" in captured[0]
    assert captured_args and captured_args[0].max_steps >= cli.AUTO_TURN_MIN_STEPS
    assert captured_args[0].max_tokens >= cli.AUTO_TURN_MIN_TOKENS
    assert followup["id"] in result["plan"]["followup_ids"]
    assert any(item["status"] == "rescheduled" for item in result["completion"]["results"])


def test_auto_due_runner_keeps_due_open_without_checkpoint(monkeypatch, tmp_path):
    from aether_dft.followups import due_followups
    from aether_dft.session_store import AetherSessionStore

    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")

    def fake_runner(prompt, **kwargs):
        return True, session_id

    args = cli.argparse.Namespace(project="demo")
    result = cli.run_auto_due_once(
        args=args,
        session_store=store,
        session_ref={"id": session_id},
        now="2999-01-01T00:00:00+08:00",
        turn_runner=fake_runner,
        quiet=True,
    )

    assert result["status"] == "needs_checkpoint"
    assert result["completed"] is False
    assert due_followups(project="demo", now="2999-01-01T00:00:00+08:00")["count"] >= 1


def test_auto_due_runner_partial_fallback_keeps_due_open(monkeypatch, tmp_path):
    from aether_dft.auto_mode import auto_mode_status
    from aether_dft.followups import due_followups
    from aether_dft.session_store import AetherSessionStore

    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")

    def fake_runner(prompt, **kwargs):
        store.append_turn(
            session_id,
            {
                "project": "demo",
                "prompt": prompt,
                "response": "工具已经推进，但模型忘记 checkpoint。",
                "record_path": str(tmp_path / "record.jsonl"),
                "tool_executions": [
                    {"name": "auto_campaign_start", "result": {"status": "ok"}},
                    {"name": "structure_build_slab", "result": {"status": "ok"}},
                ],
            },
        )
        return True, session_id

    args = cli.argparse.Namespace(project="demo", max_steps=2, max_tokens=500)
    result = cli.run_auto_due_once(
        args=args,
        session_store=store,
        session_ref={"id": session_id},
        now="2999-01-01T00:00:00+08:00",
        turn_runner=fake_runner,
        quiet=True,
    )

    assert result["status"] == "partial_checkpoint"
    assert result["fallback_checkpoint"]["status"] == "ok"
    assert "auto_campaign_start" in auto_mode_status(project="demo")["state"]["last_checkpoint"]["observation"]
    due_titles = {item["title"] for item in due_followups(project="demo", now="2999-01-01T00:00:00+08:00")["followups"]}
    assert "Auto initial advance" in due_titles


def test_auto_due_runner_model_written_partial_checkpoint_keeps_due_open(monkeypatch, tmp_path):
    from aether_dft.auto_mode import checkpoint_auto_mode
    from aether_dft.followups import due_followups
    from aether_dft.session_store import AetherSessionStore

    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")

    def fake_runner(prompt, **kwargs):
        store.append_turn(
            session_id,
            {
                "project": "demo",
                "prompt": prompt,
                "response": "只完成了中间建模 checkpoint。",
                "tool_executions": [
                    {"name": "auto_campaign_start", "result": {"status": "ok"}},
                    {"name": "structure_build_slab", "result": {"status": "ok"}},
                    {"name": "auto_mode_checkpoint", "result": {"status": "ok"}},
                ],
            },
        )
        checkpoint_auto_mode(
            project="demo",
            observation="built an intermediate slab only",
            decision="continue",
            next_focus="register candidates",
        )
        return True, session_id

    result = cli.run_auto_due_once(
        args=cli.argparse.Namespace(project="demo", max_steps=2, max_tokens=500),
        session_store=store,
        session_ref={"id": session_id},
        now="2999-01-01T00:00:00+08:00",
        turn_runner=fake_runner,
        quiet=True,
    )

    assert result["status"] == "partial_checkpoint"
    assert result["completed"] is False
    due_titles = {item["title"] for item in due_followups(project="demo", now="2999-01-01T00:00:00+08:00")["followups"]}
    assert "Auto initial advance" in due_titles


def test_auto_due_runner_fallback_consumes_due_after_candidate_registration(monkeypatch, tmp_path):
    from aether_dft.auto_mode import auto_mode_status
    from aether_dft.followups import due_followups
    from aether_dft.session_store import AetherSessionStore

    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")

    def fake_runner(prompt, **kwargs):
        store.append_turn(
            session_id,
            {
                "project": "demo",
                "prompt": prompt,
                "response": "候选已注册，但模型忘记 checkpoint。",
                "record_path": str(tmp_path / "record.jsonl"),
                "tool_executions": [
                    {"name": "auto_campaign_start", "result": {"status": "ok"}},
                    {"name": "auto_campaign_register_candidates", "result": {"status": "ok", "added": 4}},
                ],
            },
        )
        return True, session_id

    args = cli.argparse.Namespace(project="demo", max_steps=2, max_tokens=500)
    result = cli.run_auto_due_once(
        args=args,
        session_store=store,
        session_ref={"id": session_id},
        now="2999-01-01T00:00:00+08:00",
        turn_runner=fake_runner,
        quiet=True,
    )

    assert result["status"] == "ok"
    assert result["fallback_checkpoint"]["status"] == "ok"
    assert "auto_campaign_register_candidates" in auto_mode_status(project="demo")["state"]["last_checkpoint"]["observation"]
    due_titles = {item["title"] for item in due_followups(project="demo", now="2999-01-01T00:00:00+08:00")["followups"]}
    assert "Auto initial advance" not in due_titles


def test_auto_registers_manifest_before_fallback_checkpoint(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []

    def fake_list_campaigns(**kwargs):
        return {"status": "ok", "campaigns": [{"campaign_id": "campaign-1"}]}

    def fake_register(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "added": [{"candidate_id": "c1"}], "campaign": {"campaign_id": kwargs["campaign_id"]}}

    monkeypatch.setattr("aether_dft.auto_campaign.list_campaigns", fake_list_campaigns)
    monkeypatch.setattr("aether_dft.auto_campaign.register_candidates", fake_register)
    record = {
        "tool_executions": [
            {"name": "auto_campaign_start", "result": {"status": "ok", "campaign": {"campaign_id": "campaign-1"}}},
            {"name": "adsorption_candidate_manifest_compose", "result": {"status": "composed", "manifest_json": str(tmp_path / "manifest.json")}},
        ]
    }

    result = cli._auto_register_manifest_if_present(project="demo", record=record)

    assert result["status"] == "ok"
    assert calls[0]["project"] == "demo"
    assert calls[0]["campaign_id"] == "campaign-1"
    assert calls[0]["source_manifest_path"].endswith("manifest.json")


def test_auto_due_runner_registers_manifest_even_when_model_wrote_checkpoint(monkeypatch, tmp_path):
    from aether_dft.auto_mode import checkpoint_auto_mode
    from aether_dft.followups import due_followups
    from aether_dft.session_store import AetherSessionStore

    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")
    calls: list[dict[str, object]] = []

    def fake_list_campaigns(**kwargs):
        return {"status": "ok", "campaigns": [{"campaign_id": "campaign-1"}]}

    def fake_register(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "added": [{"candidate_id": "c1"}], "campaign": {"campaign_id": kwargs["campaign_id"]}}

    monkeypatch.setattr("aether_dft.auto_campaign.list_campaigns", fake_list_campaigns)
    monkeypatch.setattr("aether_dft.auto_campaign.register_candidates", fake_register)

    def fake_runner(prompt, **kwargs):
        store.append_turn(
            session_id,
            {
                "project": "demo",
                "prompt": prompt,
                "response": "manifest 已生成并 checkpoint，但模型忘了注册 campaign。",
                "tool_executions": [
                    {"name": "auto_campaign_start", "result": {"status": "ok", "campaign": {"campaign_id": "campaign-1"}}},
                    {"name": "adsorption_candidate_manifest_compose", "result": {"status": "composed", "manifest_json": str(tmp_path / "manifest.json")}},
                    {"name": "auto_mode_checkpoint", "result": {"status": "ok"}},
                ],
            },
        )
        checkpoint_auto_mode(
            project="demo",
            observation="composed a candidate manifest",
            decision="continue",
            next_focus="submit ready candidates",
        )
        return True, session_id

    result = cli.run_auto_due_once(
        args=cli.argparse.Namespace(project="demo", max_steps=2, max_tokens=500),
        session_store=store,
        session_ref={"id": session_id},
        now="2999-01-01T00:00:00+08:00",
        turn_runner=fake_runner,
        quiet=True,
    )

    assert result["status"] == "ok"
    assert result["completed"] is True
    assert calls and calls[0]["source_manifest_path"].endswith("manifest.json")
    due_titles = {item["title"] for item in due_followups(project="demo", now="2999-01-01T00:00:00+08:00")["followups"]}
    assert "Auto initial advance" not in due_titles


def test_interactive_slash_auto_enables_goal(monkeypatch, tmp_path, capsys):
    import aether_dft.paths as paths
    import aether_dft.project_state as project_state
    from aether_dft.auto_mode import auto_mode_status

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_state, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    auto_prompts: list[str] = []

    def fake_auto_turn(prompt, **kwargs):
        auto_prompts.append(prompt)
        auto_mode_status(project="demo")
        from aether_dft.auto_mode import checkpoint_auto_mode

        checkpoint_auto_mode(project="demo", observation="started", decision="continue", next_focus="candidate build")
        return True, kwargs.get("session_id")

    monkeypatch.setattr(cli, "run_chat_model_turn", fake_auto_turn)
    inputs = iter(["/auto 研究 CO 在 Pt(111) 的吸附与扩散", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat", "--project", "demo"]) == 0
    out = capsys.readouterr().out
    assert "auto mode" in out
    assert "ON" in out
    assert '"enabled": true' in out
    assert "CO 在 Pt(111)" in out
    state = auto_mode_status(project="demo")["state"]
    assert state["allow_cluster_submit"] is False
    assert auto_prompts and "AUTO MODE DUE WORK" in auto_prompts[0]


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
    auto_prompts: list[str] = []

    def fake_auto_turn(prompt, **kwargs):
        auto_prompts.append(prompt)
        from aether_dft.auto_mode import checkpoint_auto_mode

        checkpoint_auto_mode(project="demo", observation="started", decision="continue", next_focus="candidate build")
        return True, kwargs.get("session_id")

    monkeypatch.setattr(cli, "run_chat_model_turn", fake_auto_turn)
    inputs = iter(["/auto", "/status", "/auto", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat", "--project", "demo"]) == 0
    out = capsys.readouterr().out
    assert "auto goal inferred" in out
    assert "H2O/Pt(111)" in out
    assert '"enabled": true' in out
    assert '"enabled": false' in out
    assert auto_prompts and "AUTO MODE DUE WORK" in auto_prompts[0]
