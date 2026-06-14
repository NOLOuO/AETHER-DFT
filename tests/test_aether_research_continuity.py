from __future__ import annotations

import json
from pathlib import Path

from aether_dft import paths, project_state
from aether_dft.runtime_harness.tool_registry import ToolRegistry


def _patch_project_dirs(tmp_path: Path, monkeypatch):
    projects = tmp_path / "projects"
    kb = tmp_path / "knowledge_base"
    monkeypatch.setattr(paths, "PROJECTS_DIR", projects)
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", kb)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", projects)
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", kb)
    return projects, kb


def test_continuity_tools_are_registered_and_in_workflow_map():
    registry = ToolRegistry()
    names = {tool["name"] for tool in registry.list_tools()}

    assert {"project_continuity_digest", "research_cycle_checkpoint", "evidence_claim_audit"}.issubset(names)
    workflow = registry.run_tool("computational_chemistry_workflow_map", {})["result"]
    categories = {item["category"]: set(item["tools"]) for item in workflow["capability_stages"]}
    assert "project_continuity_digest" in categories["project_context"]
    assert "research_cycle_checkpoint" in categories["writeback_learning"]


def test_project_continuity_digest_reads_runs_without_forcing_workflow(tmp_path: Path, monkeypatch):
    _patch_project_dirs(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    project_state.init_project("continuity-demo", description="demo", overwrite=True)
    run_root = tmp_path / ".aether" / "runs" / "continuity-demo-task" / "run-001"
    metadata = run_root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "run_record.json").write_text(
        json.dumps(
            {
                "task_id": "continuity-demo-task",
                "run_id": "run-001",
                "run_root": str(run_root),
                "checkpoint_path": str(run_root / "checkpoint.json"),
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "overall_status": "running",
                "current_phase": "submit",
                "scheduler_job_id": "12345",
                "tags": [],
                "notes": {},
                "phases": {},
            }
        ),
        encoding="utf-8",
    )

    result = ToolRegistry().run_tool(
        "project_continuity_digest",
        {
            "project": "continuity-demo",
            "recent_results": [{"status": "no_outputs"}],
        },
    )["result"]

    assert result["status"] == "ok"
    assert result["not_a_fixed_program"]
    assert any(loop["kind"] == "cluster" for loop in result["open_loops"])
    assert any(loop["kind"] == "outputs" for loop in result["open_loops"])
    assert "cluster_job_status_brief" in result["suggested_tools"]


def test_research_cycle_checkpoint_updates_project_state_and_progress(tmp_path: Path, monkeypatch):
    _patch_project_dirs(tmp_path, monkeypatch)
    project_state.init_project("cycle-demo", description="demo", overwrite=True)

    result = ToolRegistry(permission_mode="dev").run_tool(
        "research_cycle_checkpoint",
        {
            "project": "cycle-demo",
            "goal": "Compare two H2O/Pt(111) candidates",
            "current_decision": "Atop candidate needs full VASP relax before claiming adsorption energy.",
            "evidence_refs": ["run-001/OUTCAR"],
            "open_questions": ["Need clean slab reference?"],
            "blockers": ["No converged OUTCAR yet."],
            "next_steps": ["Fetch job logs", "Run result_interpret after OUTCAR exists"],
            "run_ids": ["run-001"],
            "candidate_ids": ["h2o_atop"],
        },
    )["result"]

    assert result["status"] == "ok"
    checkpoint_path = Path(result["checkpoint_path"])
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["evidence_refs"] == ["run-001/OUTCAR"]
    state = json.loads(project_state.project_paths("cycle-demo").state.read_text(encoding="utf-8"))
    assert state["latest_checkpoint_id"] == result["checkpoint_id"]
    assert "No converged OUTCAR yet." in project_state.project_paths("cycle-demo").progress.read_text(encoding="utf-8")


def test_evidence_claim_audit_demotes_unsupported_claims():
    result = ToolRegistry().run_tool(
        "evidence_claim_audit",
        {
            "evidence_items": [{"id": "outcar-1", "path": "OUTCAR"}],
            "claims": [
                {"claim": "The optimization converged.", "evidence_refs": ["outcar-1"], "confidence": "high"},
                {"claim": "Bridge site is definitely best.", "evidence_refs": []},
            ],
        },
    )["result"]

    assert result["status"] == "ok"
    assert result["verdict"] == "needs_evidence"
    assert result["unsupported_count"] == 1
    assert result["claims"][0]["supported"] is True
    assert result["claims"][1]["supported"] is False
