from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from aether_dft import cli
from aether_dft.auto_mode import (
    build_auto_mode_digest,
    collect_due_auto_intents,
    complete_due_auto_intents,
    configure_auto_mode,
    auto_mode_status,
    answer_auto_human_question,
    audit_auto_research_progress,
    infer_research_goal,
    latest_pending_auto_human_question,
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
    assert state["current_phase"] == "goal_driven_loop"
    assert state["iteration_count"] == 0
    assert "H2O" in state["research_goal"]
    assert policy["computational_strategy"]["default_bias"].startswith("Enumerate diverse plausible candidates")
    assert policy["completion_contract"]["required_audit_tool"] == "auto_mode_convergence_audit"

    scheduled = auto_mode_status(project="demo")["scheduled_followups"]["followups"]
    auto_kinds = {(item.get("metadata") or {}).get("auto_kind") for item in scheduled}
    assert {"initial_advance", "monitor", "daily_report"}.issubset(auto_kinds)
    initial = next(item for item in scheduled if (item.get("metadata") or {}).get("auto_kind") == "initial_advance")
    assert initial["interval_minutes"] is None
    assert "Do not wait for a manual tick" in initial["prompt"]

    digest = build_auto_mode_digest(project="demo")
    assert "Auto mode is ON" in digest
    assert "Autonomy contract" in digest
    assert "auto_mode_convergence_audit" in digest
    assert "Human time is scarce; compute is the lever" in digest
    assert "enumerate candidates" in digest
    assert "H2O" in digest


def test_auto_human_question_uses_current_project_when_model_omits_project(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="验证 CO/Pt 吸附构型")

    registry = ToolRegistry()
    registry.default_project = "demo"
    result = registry.run_tool(
        "auto_human_question",
        {
            "question": "请确认优先研究 CO 吸附还是扩散？",
            "why_needed": "两个方向都会产生不同的候选空间。",
        },
    )

    assert result["result"]["status"] == "pending_human_answer"
    pending = latest_pending_auto_human_question(project="demo")
    assert pending is not None
    assert pending["project"] == "demo"
    assert "CO 吸附" in pending["question"]


def test_auto_convergence_audit_accepts_completed_synonym_with_evidence(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="验证 /auto 行为")

    result = audit_auto_research_progress(
        project="demo",
        verdict="completed",
        success_criteria=["向人类提问并记录答案"],
        evidence_refs=["auto_human_question:q1"],
        completed_items=["auto_human_question 已记录答案"],
        missing_evidence=[],
    )

    state = result["state"]
    assert state["convergence_audit"]["verdict"] == "converged"
    assert state["status"] == "converged"
    assert state["current_phase"] == "complete_with_evidence"


def test_auto_convergence_audit_demotes_completion_without_evidence(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="验证 MCH 吸附能")

    result = audit_auto_research_progress(
        project="demo",
        verdict="converged",
        success_criteria=["得到吸附能"],
        evidence_refs=[],
        completed_items=["模型认为已完成"],
        missing_evidence=[],
    )

    state = result["state"]
    audit = state["convergence_audit"]
    assert audit["verdict"] == "needs_more_evidence"
    assert state["status"] == "active"
    assert state["current_phase"] == "needs_more_evidence"
    assert audit["convergence_blockers"]
    assert any("evidence_refs" in item for item in audit["missing_evidence"])


def test_auto_convergence_audit_requires_computational_evidence_for_dft_goals(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="计算 H2O/Pt(111) 吸附能")

    result = audit_auto_research_progress(
        project="demo",
        verdict="converged",
        success_criteria=["吸附能计算完成"],
        evidence_refs=["literature:water-pt"],
        completed_items=["读了文献"],
        missing_evidence=[],
    )

    audit = result["state"]["convergence_audit"]
    assert audit["verdict"] == "needs_more_evidence"
    assert any("DFT" in item or "结构" in item for item in audit["missing_evidence"])


def test_auto_convergence_audit_rejects_running_calculation_status_for_dft_goals(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="计算 H2O/Pt(111) 吸附能")

    result = audit_auto_research_progress(
        project="demo",
        verdict="converged",
        success_criteria=["吸附能计算完成"],
        evidence_refs=["run:demo-001"],
        completed_items=["候选结构已提交"],
        missing_evidence=[],
        calculation_status={"status": "running", "completed": 0},
    )

    audit = result["state"]["convergence_audit"]
    assert audit["verdict"] == "needs_more_evidence"
    assert audit["convergence_blockers"]


def test_auto_convergence_audit_accepts_completed_calculation_status_for_dft_goals(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="计算 H2O/Pt(111) 吸附能")

    result = audit_auto_research_progress(
        project="demo",
        verdict="converged",
        success_criteria=["吸附能计算完成"],
        evidence_refs=["run:demo-001"],
        completed_items=["OUTCAR 已解析并得到吸附能"],
        missing_evidence=[],
        calculation_status={"status": "completed", "final_energy": -123.4},
    )

    audit = result["state"]["convergence_audit"]
    assert audit["verdict"] == "converged"
    assert result["state"]["status"] == "converged"


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
    assert "auto_mode_convergence_audit" in plan["prompt"]
    assert "Persistent research-loop discipline" in plan["prompt"]
    assert "auto_human_question" in plan["prompt"]
    assert "exactly one concise question" in plan["prompt"]
    assert "H2O/Pt(111)" in plan["prompt"]

    completed = complete_due_auto_intents(project="demo", followup_ids=plan["followup_ids"], note="tested")
    assert completed["status"] == "ok"
    assert any(item["status"] == "rescheduled" for item in completed["results"])


def test_auto_mode_requires_goal_when_enabled(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)

    result = configure_auto_mode(project="demo", enabled=True)

    assert result["status"] == "needs_goal"
    assert result["state"]["enabled"] is False


def test_auto_mode_convergence_audit_persists_professional_evidence(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="验证 H2O/Pt(111) 最稳定吸附构型")

    audit = audit_auto_research_progress(
        project="demo",
        verdict="needs_more_evidence",
        success_criteria=[
            "至少 3 个吸附位点候选完成结构质量检查",
            "最优候选完成 DFT 收敛并解析吸附能",
        ],
        evidence_refs=["campaign:h2o-pt", "run:demo-001"],
        completed_items=["top/hollow/bridge 候选已登记"],
        missing_evidence=["缺少收敛 OUTCAR 和吸附能"],
        calculation_status={"running": 1, "completed": 0},
        literature_status={"checked": False},
        uncertainty="H-up/H-down 取向尚未比较",
        next_focus="提交或监控最小候选批次",
        confidence=0.55,
    )

    assert audit["status"] == "ok"
    state = auto_mode_status(project="demo", include_due=False)["state"]
    assert state["status"] == "active"
    assert state["current_phase"] == "needs_more_evidence"
    assert state["success_criteria"][0].startswith("至少 3 个")
    assert state["convergence_audit"]["verdict"] == "needs_more_evidence"
    assert "缺少收敛 OUTCAR" in state["convergence_audit"]["missing_evidence"][0]
    digest = build_auto_mode_digest(project="demo")
    assert "Last convergence audit" in digest
    assert "needs_more_evidence" in digest


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

    assert {
        "auto_mode_status",
        "auto_mode_configure",
        "auto_mode_checkpoint",
        "auto_mode_convergence_audit",
        "auto_human_question",
    }.issubset(names)
    discussion_names = {item["function"]["name"] for item in registry.openai_tool_schemas(interaction_mode="discussion")}
    assert "auto_mode_status" in discussion_names
    assert "auto_mode_checkpoint" in discussion_names
    assert "auto_mode_convergence_audit" in discussion_names
    assert "auto_human_question" in discussion_names

    blocked = ToolRegistry(permission_mode="ask").run_tool(
        "auto_mode_configure",
        {"project": "demo", "enabled": True, "research_goal": "test"},
    )
    assert blocked["result"]["status"] == "permission_required"


def test_model_facing_auto_configure_cannot_enable_cluster_submit(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)

    registry = ToolRegistry(permission_mode="dev")
    result = registry.run_tool(
        "auto_mode_configure",
        {
            "project": "demo",
            "enabled": True,
            "research_goal": "筛选 CO/Pt(111) 吸附构型",
            "allow_cluster_submit": True,
        },
    )

    assert result["result"]["status"] == "ok"
    state = auto_mode_status(project="demo")["state"]
    assert state["enabled"] is True
    assert state["allow_cluster_submit"] is False
    schema = next(
        item
        for item in registry.openai_tool_schemas(interaction_mode="execution")
        if item["function"]["name"] == "auto_mode_configure"
    )
    assert "allow_cluster_submit" not in schema["function"]["parameters"]["properties"]


def test_auto_human_question_tool_records_pending_and_answer(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 H2O/Pt(111) 吸附候选")

    result = ToolRegistry(permission_mode="ask").run_tool(
        "auto_human_question",
        {
            "project": "demo",
            "question": "优先目标是最稳定吸附能，还是同时筛选扩散路径？",
            "why_needed": "两个目标会改变候选空间和计算批次。",
            "decision_boundary": "稳定构型 vs 扩散路径",
            "options": ["先稳定吸附", "同时看扩散"],
            "evidence_refs": ["auto:goal"],
        },
    )

    assert result["result"]["status"] == "pending_human_answer"
    pending = latest_pending_auto_human_question(project="demo")
    assert pending and "优先目标" in pending["question"]
    duplicate = ToolRegistry(permission_mode="ask").run_tool(
        "auto_human_question",
        {"project": "demo", "question": "另一个问题应该被延后吗？"},
    )
    assert duplicate["result"]["question"]["id"] == pending["id"]
    state = auto_mode_status(project="demo")["state"]
    assert state["status"] == "waiting_for_human"
    assert state["human_questions"]

    answered = answer_auto_human_question(project="demo", question_id=pending["id"], answer="先稳定吸附")

    assert answered["status"] == "answered"
    state = auto_mode_status(project="demo")["state"]
    assert state["status"] == "active"
    assert state["human_questions"] == []
    assert state["human_answers"][-1]["answer"] == "先稳定吸附"
    repeated = answer_auto_human_question(project="demo", question_id=pending["id"], answer="覆盖答案不应生效")
    assert repeated["status"] == "already_answered"
    assert auto_mode_status(project="demo")["state"]["human_answers"][-1]["answer"] == "先稳定吸附"


def test_auto_checkpoint_human_questions_create_answerable_pending_record(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 H2O/Pt(111) 吸附候选")

    result = ToolRegistry(permission_mode="dev").run_tool(
        "auto_mode_checkpoint",
        {
            "project": "demo",
            "status": "waiting_for_human",
            "observation": "候选分支存在目标歧义。",
            "human_questions": ["优先最稳定吸附能，还是同时筛扩散路径？"],
            "evidence_refs": ["auto:goal"],
        },
    )

    assert result["result"]["status"] == "ok"
    pending = latest_pending_auto_human_question(project="demo")
    assert pending is not None
    assert "最稳定吸附能" in pending["question"]
    state = auto_mode_status(project="demo")["state"]
    assert state["status"] == "waiting_for_human"
    assert state["human_questions"] == [pending["question"]]


def test_auto_human_question_tool_uses_cli_handler(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    seen: list[dict[str, object]] = []

    def fake_handler(payload):
        seen.append(payload)
        return answer_auto_human_question(
            project="demo",
            question_id=str(payload["question_id"]),
            answer="先用小批量候选验证",
            source="test",
        )

    result = ToolRegistry(permission_mode="ask", human_question_handler=fake_handler).run_tool(
        "auto_human_question",
        {"project": "demo", "question": "候选空间很大，要先小批量还是全量？"},
    )

    assert result["result"]["status"] == "answered"
    assert seen and seen[0]["question_id"]
    state = auto_mode_status(project="demo")["state"]
    assert state["human_answers"][-1]["answer"] == "先用小批量候选验证"


def test_auto_human_question_is_not_parallel_safe(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)

    registry = ToolRegistry(permission_mode="ask")
    specs = {item["name"]: item for item in registry.list_tools()}

    assert specs["auto_human_question"]["read_only"] is True
    assert specs["auto_human_question"]["parallel_safe"] is False
    assert registry.is_parallel_safe_tool("auto_human_question") is False


def test_background_auto_turn_disables_cli_prompts(monkeypatch, tmp_path):
    _redirect_dirs(monkeypatch, tmp_path)
    from aether_dft.session_store import AetherSessionStore

    store = AetherSessionStore(tmp_path / "sessions")
    session_id = store.start_session(project="demo")
    callbacks: list[object] = []

    def fake_ask_once(prompt, **kwargs):
        callbacks.append(kwargs.get("permission_prompt_callback"))
        callbacks.append(kwargs.get("human_question_callback"))
        return {"response": "ok", "tool_executions": [], "progress": {}, "record_path": str(tmp_path / "r.jsonl")}

    monkeypatch.setattr(cli, "ask_once", fake_ask_once)
    args = cli.argparse.Namespace(
        project="demo",
        model=None,
        max_tokens=100,
        max_steps=2,
        auto_interactive_questions=False,
    )

    ok, returned = cli.run_chat_model_turn(
        "background",
        args=args,
        session_store=store,
        session_id=session_id,
        failure_hint="x",
    )

    assert ok is True
    assert returned == session_id
    assert callbacks == [None, None]


def test_cli_human_question_handler_reads_answer(monkeypatch, tmp_path, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    pending = ToolRegistry(permission_mode="ask").run_tool(
        "auto_human_question",
        {
            "project": "demo",
            "question": "先筛选吸附构型还是扩散路径？",
            "options": ["吸附构型", "扩散路径"],
        },
    )["result"]["question"]
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")

    result = cli.answer_auto_human_question_from_cli(
        {"project": "demo", "question": pending, "question_id": pending["id"]}
    )

    assert result["status"] == "answered"
    assert auto_mode_status(project="demo")["state"]["human_answers"][-1]["answer"] == "吸附构型"
    assert "[auto question]" in capsys.readouterr().out


def test_auto_cli_on_status_off(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)

    assert cli.main(["auto", "on", "研究 MCH 在 Br/Pt 上脱氢路径", "--project", "demo", "--allow-cluster-submit", "--json"]) == 0
    enabled = json.loads(capsys.readouterr().out)
    assert enabled["state"]["enabled"] is True

    assert cli.main(["auto", "status", "--project", "demo", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["state"]["research_goal"] == "研究 MCH 在 Br/Pt 上脱氢路径"
    assert status["state"]["allow_cluster_submit"] is True

    assert cli.main(["auto", "off", "--project", "demo", "--json"]) == 0
    disabled = json.loads(capsys.readouterr().out)
    assert disabled["state"]["enabled"] is False


def test_auto_cli_accepts_goal_without_on_keyword(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)

    assert cli.normalize_auto_argv(["auto", "研究", "CO/Pt(111)", "吸附构型", "--project", "demo"]) == [
        "auto",
        "on",
        "研究 CO/Pt(111) 吸附构型",
        "--project",
        "demo",
    ]

    assert cli.main(["auto", "研究", "CO/Pt(111)", "吸附构型", "--project", "demo", "--json"]) == 0
    enabled = json.loads(capsys.readouterr().out)

    assert enabled["state"]["enabled"] is True
    assert enabled["state"]["research_goal"] == "研究 CO/Pt(111) 吸附构型"
    assert enabled["state"]["allow_cluster_submit"] is False


def test_auto_cli_no_subcommand_behaves_as_switch(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="验证 H2O/Pt(111) 吸附能")

    assert cli.main(["auto", "--project", "demo", "--json"]) == 0
    disabled = json.loads(capsys.readouterr().out)

    assert disabled["state"]["enabled"] is False


def test_auto_cli_goal_starts_initial_due_pass_unless_json_or_no_start(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    calls: list[dict[str, object]] = []

    def fake_run_auto_due_once(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "ran": True, "completed": False}

    monkeypatch.setattr(cli, "run_auto_due_once", fake_run_auto_due_once)

    assert cli.main(["auto", "研究", "CO/Pt(111)", "吸附构型", "--project", "demo"]) == 0
    out = capsys.readouterr().out
    assert "[auto]" in out
    assert calls
    assert calls[0]["args"].project == "demo"
    assert calls[0]["args"].max_steps >= cli.AUTO_TURN_MIN_STEPS

    calls.clear()
    assert cli.main(["auto", "研究", "H2O/Pt(111)", "吸附构型", "--project", "demo-json", "--json"]) == 0
    capsys.readouterr()
    assert calls == []

    assert cli.main(["auto", "研究", "NH3/Pt(111)", "吸附构型", "--project", "demo-nostart", "--no-start"]) == 0
    out = capsys.readouterr().out
    assert "--no-start" in out
    assert calls == []


def test_auto_cli_default_status_is_human_card(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="验证 CO/Pt(111) 吸附构型")
    audit_auto_research_progress(
        project="demo",
        verdict="needs_more_evidence",
        success_criteria=["至少两个候选结构通过质量检查"],
        missing_evidence=["还没有 OUTCAR 解析证据"],
        next_focus="提交最小候选批次",
    )

    assert cli.main(["auto", "status", "--project", "demo"]) == 0

    out = capsys.readouterr().out
    assert "/auto" in out
    assert "goal" in out
    assert "convergence audit" in out
    assert "missing evidence" in out
    assert "OUTCAR" in out


def test_auto_daemon_exits_cleanly_when_auto_is_off(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)

    assert cli.main(["auto", "daemon", "--project", "demo", "--once", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "auto_disabled"
    assert payload["ran"] is False


def test_auto_daemon_runs_due_worker_without_new_workflow(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    calls: list[dict[str, object]] = []

    def fake_run_auto_due_once(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "ran": True, "completed": True}

    monkeypatch.setattr(cli, "run_auto_due_once", fake_run_auto_due_once)

    assert cli.main(["auto", "daemon", "--project", "demo", "--max-cycles", "1", "--interval-seconds", "1"]) == 0

    out = capsys.readouterr().out
    assert "[auto daemon]" in out
    assert calls
    assert calls[0]["args"].project == "demo"
    assert calls[0]["args"].max_steps >= cli.AUTO_TURN_MIN_STEPS
    log_path = cli._auto_daemon_paths("demo")["log"]
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "daemon_start" for event in events)
    assert any(event["event"] == "cycle_result" for event in events)
    assert any(event["event"] == "daemon_stop" for event in events)


def test_auto_daemon_refuses_duplicate_lock(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    paths = cli._auto_daemon_paths("demo")
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["lock"].write_text(json.dumps({"pid": 12345, "project": "demo"}), encoding="utf-8")
    monkeypatch.setattr(cli, "_daemon_pid_status", lambda pid: {"status": "running", "pid": int(pid)})
    calls: list[dict[str, object]] = []

    def fake_run_auto_due_once(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "ran": True}

    monkeypatch.setattr(cli, "run_auto_due_once", fake_run_auto_due_once)

    assert cli.main(["auto", "daemon", "--project", "demo", "--once", "--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "locked"
    assert payload["existing"]["pid"] == 12345
    assert payload["lock_process"]["status"] == "running"
    assert calls == []
    assert paths["lock"].exists()


def test_auto_daemon_status_reads_lock_and_recent_events(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    paths = cli._auto_daemon_paths("demo")
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["lock"].write_text(json.dumps({"pid": 12345, "project": "demo", "started_at": "2026-01-01T00:00:00+08:00"}), encoding="utf-8")
    monkeypatch.setattr(cli, "_daemon_pid_status", lambda pid: {"status": "running", "pid": int(pid)})
    cli._append_daemon_event(paths["log"], {"event": "daemon_start", "project": "demo"})
    cli._append_daemon_event(paths["log"], {"event": "cycle_result", "result": {"status": "idle"}})
    calls: list[dict[str, object]] = []

    def fake_run_auto_due_once(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "ran": True}

    monkeypatch.setattr(cli, "run_auto_due_once", fake_run_auto_due_once)

    assert cli.main(["auto", "daemon", "--project", "demo", "--status", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "locked"
    assert payload["lock"]["pid"] == 12345
    assert payload["lock_process"]["status"] == "running"
    assert payload["log_exists"] is True
    assert [event["event"] for event in payload["recent_events"][-2:]] == ["daemon_start", "cycle_result"]
    assert calls == []


def test_auto_daemon_status_marks_stale_lock(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    paths = cli._auto_daemon_paths("demo")
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["lock"].write_text(json.dumps({"pid": 12345, "project": "demo"}), encoding="utf-8")
    monkeypatch.setattr(cli, "_daemon_pid_status", lambda pid: {"status": "stale", "pid": int(pid)})

    assert cli.main(["auto", "daemon", "--project", "demo", "--status", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "stale_lock"
    assert payload["lock_process"]["status"] == "stale"
    assert paths["lock"].exists()


def test_auto_status_includes_daemon_health(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")

    monkeypatch.setattr(
        cli,
        "_auto_daemon_status",
        lambda project, event_limit=3: {
            "status": "stale_lock",
            "project": project,
            "lock_path": "demo.lock.json",
            "lock_process": {"status": "stale"},
        },
    )

    assert cli.main(["auto", "status", "--project", "demo", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["state"]["enabled"] is True
    assert payload["daemon"]["status"] == "stale_lock"
    assert payload["daemon"]["lock_process"]["status"] == "stale"


def test_auto_status_preview_prints_daemon_health(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")

    monkeypatch.setattr(
        cli,
        "_auto_daemon_status",
        lambda project, event_limit=3: {
            "status": "stale_lock",
            "project": project,
            "lock_path": "demo.lock.json",
            "lock_process": {"status": "stale"},
        },
    )

    assert cli.main(["auto", "status", "--project", "demo"]) == 0

    out = capsys.readouterr().out
    assert "daemon" in out
    assert "stale_lock" in out
    assert "demo.lock.json" in out


def test_chat_auto_status_prints_human_readable_panel(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")

    monkeypatch.setattr(
        cli,
        "_auto_daemon_status",
        lambda project, event_limit=3: {
            "status": "stopped",
            "project": project,
            "lock_path": "demo.lock.json",
            "lock_process": {},
        },
    )

    handled, session_id = cli.handle_chat_auto_command(
        "/auto status",
        SimpleNamespace(project="demo"),
        session_store=None,
        session_id="session-1",
    )

    out = capsys.readouterr().out
    assert handled is True
    assert session_id == "session-1"
    assert "/auto" in out
    assert "daemon" in out
    assert "stopped" in out


def test_auto_daemon_refuses_stale_lock_without_force(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    paths = cli._auto_daemon_paths("demo")
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["lock"].write_text(json.dumps({"pid": 12345, "project": "demo"}), encoding="utf-8")
    monkeypatch.setattr(cli, "_daemon_pid_status", lambda pid: {"status": "stale", "pid": int(pid)})

    assert cli.main(["auto", "daemon", "--project", "demo", "--once", "--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "stale_lock"
    assert payload["lock_process"]["status"] == "stale"
    assert paths["lock"].exists()


def test_auto_daemon_force_lock_replaces_stale_lock(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    paths = cli._auto_daemon_paths("demo")
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["lock"].write_text(json.dumps({"pid": 12345, "project": "demo"}), encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run_auto_due_once(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "ran": False}

    monkeypatch.setattr(cli, "run_auto_due_once", fake_run_auto_due_once)

    assert cli.main(["auto", "daemon", "--project", "demo", "--once", "--force-lock", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert calls
    assert not paths["lock"].exists()


def test_auto_daemon_once_logs_runtime_error(tmp_path: Path, monkeypatch, capsys):
    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")

    def broken_run_auto_due_once(**kwargs):
        raise RuntimeError("simulated scheduler failure")

    monkeypatch.setattr(cli, "run_auto_due_once", broken_run_auto_due_once)

    assert cli.main(["auto", "daemon", "--project", "demo", "--once", "--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "simulated scheduler failure" in payload["message"]
    log_path = cli._auto_daemon_paths("demo")["log"]
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "cycle_error" for event in events)
    assert any(event["event"] == "daemon_stop" for event in events)


def test_auto_daemon_waits_for_pending_human_question(tmp_path: Path, monkeypatch, capsys):
    from aether_dft.auto_mode import request_auto_human_question

    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="筛选 CO/Pt(111) 吸附构型")
    request_auto_human_question(project="demo", question="优先筛选吸附构型还是扩散路径？")
    calls: list[dict[str, object]] = []

    def fake_run_auto_due_once(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "ran": True}

    monkeypatch.setattr(cli, "run_auto_due_once", fake_run_auto_due_once)

    assert cli.main(["auto", "daemon", "--project", "demo", "--once"]) == 0

    out = capsys.readouterr().out
    assert "waiting for human answer" in out
    assert "优先筛选吸附构型" in out
    assert calls == []


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

    def fake_background_loop(*, args, session_store, session_ref):
        args._auto_wake_event = cli.threading.Event()
        return cli.threading.Event()

    monkeypatch.setattr(cli, "start_auto_background_loop", fake_background_loop)
    inputs = iter(["/auto 研究 CO 在 Pt(111) 的吸附与扩散", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat", "--project", "demo"]) == 0
    out = capsys.readouterr().out
    assert "/auto" in out
    assert "ON" in out
    assert "goal" in out
    assert "CO 在 Pt(111)" in out
    state = auto_mode_status(project="demo")["state"]
    assert state["allow_cluster_submit"] is False
    assert "后台已被唤醒" in out


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

    def fake_background_loop(*, args, session_store, session_ref):
        args._auto_wake_event = cli.threading.Event()
        return cli.threading.Event()

    monkeypatch.setattr(cli, "start_auto_background_loop", fake_background_loop)
    inputs = iter(["/auto", "/status", "/auto", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert cli.main(["chat", "--project", "demo"]) == 0
    out = capsys.readouterr().out
    assert "auto goal inferred" in out
    assert "H2O/Pt(111)" in out
    assert "ON" in out
    assert "OFF" in out
