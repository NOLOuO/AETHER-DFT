from __future__ import annotations

import json
from pathlib import Path

from aether_dft import cli
from aether_dft.followups import complete_followup, due_followups, list_followups, schedule_followup
from aether_dft.runtime_harness.tool_registry import ToolRegistry


def _redirect_project_dirs(monkeypatch, tmp_path: Path) -> None:
    import aether_dft.paths as paths
    import aether_dft.project_state as project_state

    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_state, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")


def test_followup_schedule_due_and_complete_are_project_local(tmp_path: Path, monkeypatch):
    _redirect_project_dirs(monkeypatch, tmp_path)

    created = schedule_followup(
        project="demo",
        prompt="检查 job 123 的 OUTCAR 是否收敛",
        due_at="2026-06-18T08:00:00+08:00",
        related_job_id="123",
    )

    assert created["status"] == "ok"
    assert "projects" in created["path"]
    assert Path(created["path"]).exists()

    due = due_followups(project="demo", now="2026-06-18T09:00:00+08:00")
    assert due["count"] == 1
    assert due["followups"][0]["related_job_id"] == "123"
    assert due["followups"][0]["evidence_goals"][0]["goal"] == "refresh_related_job"

    done = complete_followup(due["followups"][0]["id"], project="demo", note="已检查")
    assert done["status"] == "ok"
    assert list_followups(project="demo")["count"] == 0
    assert list_followups(project="demo", include_done=True)["followups"][0]["completion_note"] == "已检查"


def test_followup_interval_reschedules(tmp_path: Path, monkeypatch):
    _redirect_project_dirs(monkeypatch, tmp_path)
    created = schedule_followup(project="demo", prompt="每小时检查一次队列", interval_minutes=60)
    fid = created["followup"]["id"]

    result = complete_followup(fid, project="demo", note="checked", reschedule=True)

    assert result["status"] == "rescheduled"
    assert result["followup"]["status"] == "scheduled"
    assert result["followup"]["last_note"] == "checked"


def test_followup_tools_are_discoverable_and_respect_permissions(tmp_path: Path, monkeypatch):
    _redirect_project_dirs(monkeypatch, tmp_path)

    registry = ToolRegistry(permission_mode="never")
    names = {tool["name"] for tool in registry.list_tools()}

    assert {"research_followup_schedule", "research_followup_list", "research_followup_due", "research_followup_complete"}.issubset(names)
    assert "research_followup_due" in {
        item["function"]["name"]
        for item in registry.openai_tool_schemas(interaction_mode="discussion")
    }
    blocked = ToolRegistry(permission_mode="ask").run_tool(
        "research_followup_schedule",
        {"project": "demo", "prompt": "检查任务", "interval_minutes": 1},
    )
    assert blocked["result"]["status"] == "permission_required"


def test_followup_cli_smoke(tmp_path: Path, monkeypatch, capsys):
    _redirect_project_dirs(monkeypatch, tmp_path)

    assert cli.main(["followup", "schedule", "检查 job 7", "--project", "demo", "--due-at", "2026-06-18T08:00:00+08:00", "--job-id", "7"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["status"] == "ok"

    assert cli.main(["followup", "due", "--project", "demo", "--now", "2026-06-18T09:00:00+08:00"]) == 0
    due = json.loads(capsys.readouterr().out)
    assert due["count"] == 1
    assert due["followups"][0]["id"] == created["followup"]["id"]
