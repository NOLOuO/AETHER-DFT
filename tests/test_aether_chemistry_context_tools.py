from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import fcc111
from pymatgen.io.ase import AseAtomsAdaptor

from aether_dft.knowledge import add_note, search_for_system
from aether_dft.context_digests import build_relevant_priors_digest
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
    result = search_for_system(material="Pt(111)", adsorbate="H2O", semantic=False)
    assert result["status"] == "ok"
    assert result["returned"] >= 1
    titles = [match["title"] for match in result["matches"]]
    assert any("H2O on Pt(111)" in title or "Pt(111)" in title for title in titles)


def test_knowledge_search_for_system_rejects_empty_query():
    with pytest.raises(ValueError):
        search_for_system()



def test_knowledge_search_for_system_can_use_semantic_selector():
    init_project("pytest-semantic-search", description="semantic search", overwrite=True)
    add_note(
        "pytest-semantic-search",
        "水在铂表面的避坑",
        "摘要：水分子在铂表面容易出现悬浮初态；应优先检查氧端朝下和顶位吸附构型。",
        tags=["避坑"],
    )
    add_note(
        "pytest-semantic-search",
        "普通 API 文档",
        "Description: Tool API docs for a builder; no chemistry warning here.",
        tags=["api"],
    )

    def selector(query, catalog, max_results):
        assert "H2O" in query
        assert all("content" not in item for item in catalog)
        chosen = next(item for item in catalog if item.get("project") == "pytest-semantic-search" and "避坑" in item["title"])
        return [{"rank": chosen["rank"], "semantic_reason": "gotcha beats API docs"}]

    result = search_for_system(
        material="Pt(111)",
        adsorbate="H2O",
        project_priority="pytest-semantic-search",
        selector=selector,
    )

    assert result["selection_method"] == "semantic"
    assert result["semantic_candidates_considered"] >= 2
    assert result["returned"] == 1
    assert result["matches"][0]["title"] == "水在铂表面的避坑"
    assert result["matches"][0]["score"] == 0
    assert result["matches"][0]["semantic_reason"] == "gotcha beats API docs"


def test_knowledge_search_for_system_falls_back_when_semantic_selector_fails():
    init_project("pytest-semantic-fallback", description="semantic fallback", overwrite=True)
    add_note("pytest-semantic-fallback", "H2O Pt prior", "H2O on Pt(111) lexical prior")

    def broken_selector(query, catalog, max_results):
        raise RuntimeError("selector unavailable")

    result = search_for_system(
        material="Pt(111)",
        adsorbate="H2O",
        project_priority="pytest-semantic-fallback",
        selector=broken_selector,
    )

    assert result["selection_method"] == "lexical_fallback"
    assert "selector unavailable" in result["selection_error"]
    assert result["returned"] >= 1


def test_relevant_priors_digest_uses_unified_knowledge_search_without_live_selector(monkeypatch):
    init_project("pytest-prior-digest", description="prompt prior digest", overwrite=True)
    add_note(
        "pytest-prior-digest",
        "Pt water warning",
        "H2O on Pt(111) prior: compare atop and bridge first; avoid floating O-up initial states.",
        tags=["warning"],
    )

    def should_not_call_live_selector(*args, **kwargs):
        raise AssertionError("prompt preload should not call the live semantic selector by default")

    import aether_dft.knowledge as knowledge

    monkeypatch.setattr(knowledge, "_default_memory_selector", should_not_call_live_selector)
    digest = build_relevant_priors_digest(project="pytest-prior-digest", query="H2O Pt(111)", max_items=2)

    assert "Relevant project/research priors (lexical preload" in digest
    assert "Pt water warning" in digest
    assert "knowledge_search_for_system" in digest


def test_relevant_priors_digest_can_opt_into_semantic_preload(monkeypatch):
    init_project("pytest-prior-semantic", description="prompt prior semantic digest", overwrite=True)
    add_note(
        "pytest-prior-semantic",
        "水在铂表面的避坑",
        "摘要：水分子在铂表面容易悬浮；应优先检查氧端朝下和顶位吸附构型。",
        tags=["避坑"],
    )

    import aether_dft.knowledge as knowledge

    def selector(query, catalog, max_results):
        chosen = next(item for item in catalog if "避坑" in item["title"])
        return [{"rank": chosen["rank"], "semantic_reason": "中文语义命中避坑经验"}]

    monkeypatch.setenv("AETHER_DFT_PRELOAD_SEMANTIC_PRIORS", "1")
    monkeypatch.setattr(knowledge, "_default_memory_selector", selector)
    digest = build_relevant_priors_digest(project="pytest-prior-semantic", query="H2O Pt111", max_items=2)

    assert "Relevant project/research priors (semantic preload" in digest
    assert "水在铂表面的避坑" in digest
    assert "中文语义命中避坑经验" in digest
