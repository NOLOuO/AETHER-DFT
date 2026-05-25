from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import fcc111
from pymatgen.io.ase import AseAtomsAdaptor

from aether_dft.convergence import compose_convergence_plan
from aether_dft.runtime_harness.tool_registry import ToolRegistry
from dft_shared.structure_analyzer.operations import (
    enumerate_defect_sites,
    interpolate_ts_midpoint_candidates,
    structure_relax_short,
)


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


def _write_pt111(path: Path) -> Path:
    atoms = fcc111("Pt", size=(2, 2, 2), vacuum=8.0)
    structure = AseAtomsAdaptor.get_structure(atoms)
    path.write_text(structure.to(fmt="poscar"), encoding="utf-8")
    return path


def test_defect_site_enumerate_lists_surface_sites(tmp_path):
    poscar = _write_pt111(tmp_path / "POSCAR")
    result = enumerate_defect_sites(str(poscar), species="Pt", surface_only=True)
    assert result["status"] == "ok"
    assert result["candidate_count"] == 4
    assert all(item["element"] == "Pt" for item in result["candidates"])
    assert all(item["is_surface"] for item in result["candidates"])


def test_ts_midpoint_candidates_write_interpolated_poscars(tmp_path):
    initial = _write_pt111(tmp_path / "IS.POSCAR")
    final = _write_pt111(tmp_path / "FS.POSCAR")
    # Move one atom slightly so interpolation is non-trivial.
    from pymatgen.core import Structure

    structure = Structure.from_file(final)
    structure.translate_sites([0], [0.0, 0.0, 0.2], frac_coords=False)
    structure.to(fmt="poscar", filename=str(final))

    result = interpolate_ts_midpoint_candidates(
        initial_path=str(initial),
        final_path=str(final),
        output_dir=str(tmp_path / "ts"),
        n_images=2,
    )
    assert result["status"] == "ok"
    assert len(result["images"]) == 2
    for image in result["images"]:
        assert Path(image["poscar_path"]).exists()
    assert (tmp_path / "ts" / "ts_midpoint_manifest.json").exists()


def test_convergence_plan_compose_persists_matrix(tmp_path):
    plan = compose_convergence_plan(
        material="Pt(111)",
        encut_values=[400, 450],
        kpoint_grids=[[3, 3, 1]],
        output_dir=str(tmp_path / "conv"),
    )
    assert plan["status"] == "planned"
    assert len(plan["matrix"]) == 2
    assert Path(plan["plan_path"]).exists()


def test_phase6_tools_are_registered(tmp_path):
    poscar = _write_pt111(tmp_path / "POSCAR")
    registry = ToolRegistry()
    defect = registry.run_tool("defect_site_enumerate", {"structure_path": str(poscar), "species": "Pt"})
    assert defect["result"]["status"] == "ok"

    conv = registry.run_tool(
        "convergence_plan_compose",
        {"material": "Pt(111)", "output_dir": str(tmp_path / "conv_tool")},
    )
    assert conv["result"]["status"] == "planned"


def test_structure_relax_short_runs_real_emt_when_supported(tmp_path):
    poscar = _write_pt111(tmp_path / "POSCAR")
    relaxed = tmp_path / "RELAXED.POSCAR"
    result = structure_relax_short(
        input_path=str(poscar),
        output_path=str(relaxed),
        max_steps=2,
        fmax=1.0,
    )
    assert result["status"] == "ok"
    assert result["calculator"] == "emt"
    assert relaxed.exists()
    assert "final_energy_ev" in result


def test_structure_relax_short_reports_unsupported_calculator(tmp_path):
    poscar = _write_pt111(tmp_path / "POSCAR")
    result = ToolRegistry().run_tool(
        "structure_relax_short",
        {
            "input_path": str(poscar),
            "output_path": str(tmp_path / "RELAXED.POSCAR"),
            "calculator": "mace",
        },
    )
    assert result["result"]["status"] == "unavailable"
