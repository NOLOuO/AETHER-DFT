from __future__ import annotations

from pathlib import Path
import json

from aether_dft.auto_campaign import (
    list_campaigns,
    next_batch,
    prune_plan,
    register_candidates,
    start_campaign,
    update_candidate,
)
from aether_dft.runtime_harness.tool_registry import ToolRegistry


def _redirect_dirs(monkeypatch, tmp_path: Path) -> None:
    import aether_dft.paths as paths
    import aether_dft.project_state as project_state

    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_state, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")


def test_auto_campaign_tracks_candidates_batch_and_pruning(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)

    created = start_campaign(project="demo", goal="Search H2O/Pt(111) adsorption candidates", campaign_id="h2o-pt")

    assert created["status"] == "ok"
    assert created["summary"]["next_focus"] == "enumerate_or_register_candidates"

    registered = register_candidates(
        project="demo",
        campaign_id="h2o-pt",
        candidates=[
            {"candidate_id": "top-o-down", "structure_path": "top.POSCAR", "motif": "ontop", "quality_score": 0.91},
            {"candidate_id": "bridge-o-down", "structure_path": "bridge.POSCAR", "motif": "bridge", "quality_score": 0.72},
            {"candidate_id": "bad-overlap", "structure_path": "bad.POSCAR", "motif": "hollow", "quality_score": 0.25},
        ],
        source_manifest_path="manifest.json",
    )

    assert registered["summary"]["candidate_count"] == 3
    assert registered["summary"]["ready_count"] == 3

    batch = next_batch(project="demo", campaign_id="h2o-pt", max_candidates=2, min_quality_score=0.5)
    assert [item["candidate_id"] for item in batch["candidates"]] == ["top-o-down", "bridge-o-down"]

    updated = update_candidate(
        project="demo",
        campaign_id="h2o-pt",
        candidate_id="top-o-down",
        status="submitted",
        run_id="run_001",
        job_id="99160",
        remote_run_root="~/research/demo/run_001",
    )
    assert updated["candidate"]["job_id"] == "99160"
    assert updated["summary"]["running_count"] == 1

    update_candidate(
        project="demo",
        campaign_id="h2o-pt",
        candidate_id="bridge-o-down",
        status="completed",
        result={"adsorption_energy_ev": -0.42},
    )
    pruned = prune_plan(project="demo", campaign_id="h2o-pt", keep_top=1, min_quality_score=0.5, apply=True)

    assert pruned["status"] == "ok"
    assert pruned["apply"] is True
    assert pruned["keepers"]
    listed = list_campaigns(project="demo")
    assert listed["campaigns"][0]["campaign_id"] == "h2o-pt"


def test_auto_campaign_imports_candidate_fields_from_manifest(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)
    start_campaign(project="demo", goal="Screen CO/Pt candidates", campaign_id="co-pt")
    poscar = tmp_path / "ontop.POSCAR"
    poscar.write_text("POSCAR placeholder", encoding="utf-8")
    manifest = tmp_path / "candidate_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task_id": "co-screen",
                "material_name": "Pt(111)",
                "adsorbate_source": "CO",
                "candidates": [
                    {
                        "candidate_id": "co-ontop",
                        "site_family": "ontop",
                        "site_label": "ontop-01",
                        "orientation_label": "C-down",
                        "anchor_symbol": "C",
                        "height": 1.9,
                        "metadata": {"rank": 1, "model_reason": "CO C-down atop is a strong baseline."},
                        "score": {"total": 0.88, "reason": "good distance and motif"},
                        "exported_files": {"poscar_path": str(poscar)},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    registered = register_candidates(
        project="demo",
        campaign_id="co-pt",
        source_manifest_path=str(manifest),
        candidates=[{"candidate_id": "co-ontop", "status": "quality_pass"}],
    )

    candidate = registered["campaign"]["candidates"][0]
    assert registered["manifest_imported"] == 1
    assert candidate["candidate_id"] == "co-ontop"
    assert candidate["status"] == "quality_pass"
    assert candidate["structure_path"] == str(poscar)
    assert candidate["material"] == "Pt(111)"
    assert candidate["adsorbate"] == "CO"
    assert candidate["quality_score"] == 0.88


def test_auto_campaign_tools_are_registered_and_discoverable(tmp_path: Path, monkeypatch):
    _redirect_dirs(monkeypatch, tmp_path)

    registry = ToolRegistry(permission_mode="dev")
    names = {item["name"] for item in registry.list_tools()}

    expected = {
        "auto_campaign_start",
        "auto_campaign_list",
        "auto_campaign_status",
        "auto_campaign_register_candidates",
        "auto_campaign_update_candidate",
        "auto_campaign_next_batch",
        "auto_campaign_prune_plan",
    }
    assert expected.issubset(names)

    discussion_names = {item["function"]["name"] for item in registry.openai_tool_schemas(interaction_mode="discussion")}
    assert {"auto_campaign_list", "auto_campaign_status", "auto_campaign_next_batch"}.issubset(discussion_names)

    started = registry.run_tool(
        "auto_campaign_start",
        {"project": "demo", "goal": "Batch screen MCH/Pt-Br intermediates", "campaign_id": "mch"},
    )
    assert started["result"]["status"] == "ok"

    status = registry.run_tool("auto_campaign_status", {"project": "demo", "campaign_id": "mch"})
    assert status["result"]["summary"]["next_focus"] == "enumerate_or_register_candidates"


def test_auto_mode_status_surfaces_active_campaigns(tmp_path: Path, monkeypatch):
    from aether_dft.auto_mode import auto_mode_status, configure_auto_mode

    _redirect_dirs(monkeypatch, tmp_path)
    configure_auto_mode(project="demo", enabled=True, research_goal="批量筛选 CO/Pt(111) 候选")
    start_campaign(project="demo", goal="批量筛选 CO/Pt(111) 候选", campaign_id="co-screen")

    status = auto_mode_status(project="demo", include_due=True)

    assert status["active_campaigns"]["status"] == "ok"
    assert status["active_campaigns"]["campaigns"][0]["campaign_id"] == "co-screen"
