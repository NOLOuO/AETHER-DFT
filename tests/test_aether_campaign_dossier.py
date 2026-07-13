from __future__ import annotations

import json
from pathlib import Path

from aether_dft import paths, project_state
from aether_dft.campaign_dossier import build_campaign_dossier
from aether_dft.session_store import AetherSessionStore


def test_campaign_dossier_hashes_outputs_and_never_copies_potcar(tmp_path: Path, monkeypatch):
    projects = tmp_path / "projects"
    monkeypatch.setattr(paths, "PROJECTS_DIR", projects)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", projects)
    project_state.init_project("paper-demo", description="demo", overwrite=True)
    project_state.write_project_state(
        "paper-demo",
        {
            "research_goal": "Validate a converged adsorption structure.",
            "latest_decision": "Run one final frequency calculation.",
        },
    )

    sessions = AetherSessionStore(tmp_path / "sessions")
    session_id = sessions.start_session(project="paper-demo")
    sessions.append_turn(
        session_id,
        {
            "prompt": "submit after approval",
            "response": "submitted and monitored",
            "tool_executions": [
                {
                    "name": "cluster_remote_submit",
                    "arguments": {"run_id": "run-1"},
                    "result": {
                        "status": "submitted",
                        "human_approval": {"granted": True, "scope_digest": "abc"},
                    },
                },
                {"name": "cluster_job_status_brief", "arguments": {"job_id": "123"}, "result": {"status": "ok", "job_id": "123"}},
            ],
        },
    )

    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "OUTCAR").write_text("vasp evidence", encoding="utf-8")
    (run_root / "CONTCAR").write_text("structure", encoding="utf-8")
    (run_root / "POTCAR").write_text("licensed potential", encoding="utf-8")
    output = tmp_path / "dossier"

    result = build_campaign_dossier(
        project="paper-demo",
        session_id=session_id,
        run_roots=[run_root],
        output_dir=output,
        session_store=sessions,
    )

    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    potcar = next(item for item in payload["artifacts"] if item.get("name") == "POTCAR")
    assert potcar["licensed_content_excluded"] is True
    assert potcar["share_policy"] == "metadata-only"
    assert potcar["sha256"]
    assert not (output / "POTCAR").exists()
    assert payload["checklist"]["session_exists"] is True
    assert payload["checklist"]["scheduler_evidence_present"] is True
    assert payload["checklist"]["vasp_output_evidence_present"] is True
    assert payload["checklist"]["submit_authorization_present"] is True


def test_campaign_dossier_marks_missing_evidence_instead_of_claiming_complete(tmp_path: Path, monkeypatch):
    projects = tmp_path / "projects"
    monkeypatch.setattr(paths, "PROJECTS_DIR", projects)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", projects)
    project_state.init_project("incomplete-demo", description="demo", overwrite=True)
    sessions = AetherSessionStore(tmp_path / "sessions")

    result = build_campaign_dossier(
        project="incomplete-demo",
        session_id="missing-session",
        run_roots=[tmp_path / "missing-run"],
        output_dir=tmp_path / "dossier",
        session_store=sessions,
    )

    assert result["complete"] is False
    assert "session_exists" in result["limitations"]
    assert "all_run_roots_exist" in result["limitations"]
