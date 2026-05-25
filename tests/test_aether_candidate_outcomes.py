from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import fcc111
from ase.io import write

from aether_dft.candidate_outcomes import record_candidate_outcome
from aether_dft.knowledge import search_for_system, search_notes
from aether_dft.project_state import init_project
from aether_dft.recommendations import recommend_next_tasks
from aether_dft.runtime_harness.tool_registry import ToolRegistry


@pytest.fixture(autouse=True)
def isolated_aether_state(tmp_path, monkeypatch):
    import aether_dft.knowledge as knowledge
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
    monkeypatch.setattr(knowledge, "KNOWLEDGE_BASE_DIR", knowledge_dir)


def _write_initial_final(tmp_path: Path) -> tuple[Path, Path]:
    atoms = fcc111("Pt", size=(2, 2, 2), vacuum=8.0)
    initial = tmp_path / "POSCAR"
    final = tmp_path / "CONTCAR"
    write(initial, atoms, format="vasp")
    relaxed = atoms.copy()
    positions = relaxed.get_positions()
    positions[-1, 2] += 0.05
    relaxed.set_positions(positions)
    write(final, relaxed, format="vasp")
    return initial, final


def test_candidate_outcome_record_writes_searchable_prior(tmp_path):
    init_project("pytest-outcomes", description="candidate outcome test", overwrite=True)
    initial, final = _write_initial_final(tmp_path)

    result = record_candidate_outcome(
        project="pytest-outcomes",
        material="Pt(111)",
        adsorbate="H2O",
        candidate_id="ontop_01_upright",
        verdict="success",
        adsorption_energy_ev=-0.42,
        initial_path=str(initial),
        final_path=str(final),
        calculation_summary="VASP converged; no large drift.",
        notes="H2O prefers atop O-down in this small Pt(111) test.",
    )

    assert result["status"] == "ok"
    assert result["outcome"]["displacement"] is not None
    assert Path(result["note"]["path"]).exists()

    matches = search_for_system(material="Pt(111)", adsorbate="H2O", project_priority="pytest-outcomes")
    assert matches["returned"] >= 1
    assert any(item["source"] == "knowledge_base" for item in matches["matches"])
    assert search_notes("pytest-outcomes", "candidate_outcome")
    recs = recommend_next_tasks("pytest-outcomes", focus="H2O Pt(111) adsorption")
    assert any("outcome" in item["title"].lower() for item in recs)


def test_candidate_outcome_record_tool_surface(tmp_path):
    init_project("pytest-outcomes-tool", description="candidate outcome tool test", overwrite=True)
    initial, final = _write_initial_final(tmp_path)
    result = ToolRegistry().run_tool(
        "candidate_outcome_record",
        {
            "project": "pytest-outcomes-tool",
            "material": "Pt(111)",
            "adsorbate": "H2O",
            "candidate_id": "ontop_01",
            "verdict": "promising",
            "adsorption_energy_ev": -0.35,
            "initial_path": str(initial),
            "final_path": str(final),
            "calculation_summary": "mock completed evidence",
        },
    )
    assert result["result"]["status"] == "ok"
    assert result["result"]["note"]["note_id"].startswith("note_")
