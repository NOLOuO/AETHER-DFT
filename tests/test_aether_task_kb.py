from __future__ import annotations

import json
from pathlib import Path

import pytest

from aether_dft import cli
from aether_dft.knowledge import add_note, search_notes, show_note
from aether_dft.project_state import init_project
from aether_dft.task_bridge import create_task_plan, list_task_records


@pytest.fixture(autouse=True)
def isolated_aether_state(tmp_path, monkeypatch):
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


def test_task_plan_creates_ready_single_calculation(tmp_path):
    envelope = create_task_plan(
        "计算 Si 的 DOS",
        project=None,
        material="Si",
        task_type="dos",
        planner_mode="rule",
        persist=False,
    )
    assert envelope.readiness == "ready"
    assert envelope.spec is not None
    assert envelope.spec["task_type"] == "dos"
    assert "aether-dft" in envelope.dft_command[0]


def test_project_task_plan_persists_and_lists():
    init_project("pytest-task-project", description="test project", overwrite=True)
    envelope = create_task_plan(
        "优化 Pt slab",
        project="pytest-task-project",
        material="Pt",
        task_type="relax",
        planner_mode="rule",
        persist=True,
    )
    records = list_task_records("pytest-task-project")
    assert any(item["task_id"] == envelope.task_id for item in records)
    assert Path(envelope.task_record_path).exists()


def test_knowledge_base_add_search_show():
    init_project("pytest-kb-project", description="test kb", overwrite=True)
    note = add_note("pytest-kb-project", "DFT 参数经验", "ENCUT 和 KPOINTS 需要先做收敛。", tags=["dft"])
    matches = search_notes("pytest-kb-project", "ENCUT")
    assert matches
    shown = show_note(note.note_id, project="pytest-kb-project")
    assert "ENCUT" in shown["content"]


def test_cli_task_plan_smoke(capsys):
    assert cli.main([
        "task",
        "plan",
        "计算",
        "Si",
        "的",
        "DOS",
        "--material",
        "Si",
        "--task-type",
        "dos",
        "--no-persist",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["readiness"] == "ready"
    assert payload["spec"]["task_type"] == "dos"


def test_cli_kb_add_and_search_smoke(capsys):
    init_project("pytest-cli-kb", description="test kb cli", overwrite=True)
    assert cli.main([
        "kb",
        "add",
        "pytest-cli-kb",
        "--title",
        "吸附能公式",
        "--text",
        "E_ads = E_adsorbate_slab - E_slab - E_molecule",
        "--tag",
        "adsorption",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert cli.main(["kb", "search", "pytest-cli-kb", "E_ads"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matches"]


def test_cli_chat_task_plan_mode(capsys):
    assert cli.main([
        "chat",
        "计算",
        "Si",
        "的",
        "DOS",
        "--task-plan",
        "--material",
        "Si",
        "--task-type",
        "dos",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["readiness"] == "ready"
    assert payload["spec"]["task_type"] == "dos"
