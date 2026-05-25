from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import fcc111
from pymatgen.io.ase import AseAtomsAdaptor

from aether_dft.knowledge import add_note, search_for_system
from aether_dft.project_state import init_project
from dft_shared.chemistry_hints import get_adsorbate_chemistry_hint, list_curated_adsorbates
from dft_shared.structure_analyzer.operations import inspect_slab_surface


@pytest.fixture(autouse=True)
def isolated_aether_state(tmp_path, monkeypatch):
    import aether_dft.paths as paths
    import aether_dft.knowledge as knowledge
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


def write_pt111_poscar(path: Path) -> Path:
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=8.0)
    structure = AseAtomsAdaptor.get_structure(atoms)
    path.write_text(structure.to(fmt="poscar"), encoding="utf-8")
    return path


def test_inspect_slab_surface_groups_symmetric_pt111(tmp_path):
    slab_path = write_pt111_poscar(tmp_path / "POSCAR")
    result = inspect_slab_surface(str(slab_path))
    assert result["status"] == "ok"
    assert result["spacegroup"], "Pt(111) 应该有可识别空间群"
    assert result["surface_composition"]["top_layer"] == {"Pt": 4}
    # Pt(111) 4 个顶层原子应该全部对称等价
    assert len(result["symmetry_groups"]) == 1
    group = result["symmetry_groups"][0]
    assert group["multiplicity"] == 4
    for atom in result["top_layer"]:
        assert atom["coordination_number"] > 0
        assert atom["nearest_neighbors"]


def test_chemistry_hint_curated_h2o_returns_o_anchor():
    hint = get_adsorbate_chemistry_hint("H2O")
    assert hint["status"] == "ok"
    assert hint["source"] == "curated"
    assert hint["anchor_candidates"][0]["element"] == "O"
    assert any(motif["preferred_site_family"] == "ontop" for motif in hint["binding_motifs"])
    assert hint["typical_height_angstrom"] > 1.0


def test_chemistry_hint_alias_water_maps_to_h2o():
    hint = get_adsorbate_chemistry_hint("water")
    assert hint["status"] == "ok"
    assert hint["adsorbate"] == "H2O"


def test_chemistry_hint_falls_back_for_unknown_smiles_like():
    # 用一个 RDKit 能解析的 SMILES（甲醇）但不在 curated 表 alias 内
    hint = get_adsorbate_chemistry_hint("CCO")
    assert hint["status"] == "ok"
    assert hint["source"] in {"ase_inferred", "rdkit_smiles"}
    assert hint["anchor_candidates"][0]["element"] in {"O", "C", "H"}


def test_chemistry_hint_unknown_returns_status_unknown():
    hint = get_adsorbate_chemistry_hint("ZZZZ-not-a-molecule")
    assert hint["status"] == "unknown"


def test_curated_list_includes_core_species():
    names = list_curated_adsorbates()
    for required in ("H2O", "CO", "OH", "NH3"):
        assert required in names


def test_knowledge_search_for_system_finds_matching_note():
    init_project("pytest-system-search", description="search by system test", overwrite=True)
    add_note(
        "pytest-system-search",
        "H2O on Pt(111) 经验",
        "我们之前在 Pt(111) 上算过 H2O 的 atop O-down，ENCUT 450 收敛良好。",
        tags=["adsorption", "pt", "water"],
    )
    add_note(
        "pytest-system-search",
        "无关条目",
        "CO 在 Pd 上的吸附能",
    )
    result = search_for_system(material="Pt(111)", adsorbate="H2O")
    assert result["status"] == "ok"
    assert result["returned"] >= 1
    titles = [match["title"] for match in result["matches"]]
    assert any("H2O on Pt(111)" in title or "Pt(111)" in title for title in titles)


def test_knowledge_search_for_system_rejects_empty_query():
    with pytest.raises(ValueError):
        search_for_system()
