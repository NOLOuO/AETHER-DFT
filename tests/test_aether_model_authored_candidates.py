from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase.build import fcc111
from pymatgen.io.ase import AseAtomsAdaptor

from aether_dft.adsorption_authoring import create_candidate_plan
from dft_app.modeling import compose_manifest_from_authored_candidates
from dft_app.modeling.confirmed_candidate_handoff import ConfirmedCandidateHandoff
from dft_shared.structure_analyzer.operations import (
    add_adsorbate,
    candidate_quality_score,
    enumerate_adsorption_sites,
    sanity_check,
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


def write_pt111_poscar(path: Path) -> Path:
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=8.0)
    structure = AseAtomsAdaptor.get_structure(atoms)
    path.write_text(structure.to(fmt="poscar"), encoding="utf-8")
    return path


def test_enumerate_sites_returns_multiple_families_and_top_layer(tmp_path):
    slab_path = write_pt111_poscar(tmp_path / "POSCAR")
    result = enumerate_adsorption_sites(str(slab_path), max_sites_per_family=3)

    assert result["status"] == "ok"
    assert result["sites"], "Pt(111) slab 应该至少有一些位点"
    families = {site["site_family"].lower() for site in result["sites"]}
    # pymatgen 在 Pt(111) 上至少能给出 ontop 和 hollow
    assert "ontop" in families
    for site in result["sites"]:
        assert len(site["cart_coords"]) == 3
        assert site["nearest_top_atoms"], "每个位点至少要带最近顶层原子"
    assert result["top_layer_atoms"], "顶层原子摘要不能为空"
    assert all(item["element"] == "Pt" for item in result["top_layer_atoms"])


def test_add_adsorbate_applies_selective_dynamics(tmp_path):
    slab_path = write_pt111_poscar(tmp_path / "POSCAR")
    sites = enumerate_adsorption_sites(str(slab_path), max_sites_per_family=2)["sites"]
    chosen = sites[0]
    output = tmp_path / "candidate.POSCAR"
    result = add_adsorbate(
        slab_path=str(slab_path),
        adsorbate="H2O",
        output_path=str(output),
        cart_coords=chosen["cart_coords"],
        orientation="upright",
        fixed_bottom_layers=2,
    )
    assert result["status"] == "ok"
    poscar_text = output.read_text(encoding="utf-8")
    assert "Selective dynamics" in poscar_text or "selective" in poscar_text.lower()
    assert " F " in poscar_text or " F\n" in poscar_text, "至少有原子被冻结"
    assert " T " in poscar_text or " T\n" in poscar_text, "吸附物层应保持可动"


def test_model_authored_pipeline_produces_consumable_manifest(tmp_path):
    slab_path = write_pt111_poscar(tmp_path / "POSCAR")
    enum = enumerate_adsorption_sites(str(slab_path), max_sites_per_family=2)
    chosen = enum["sites"][:2]
    assert len(chosen) >= 2, "Pt(111) 至少要有 2 个候选位点"

    plan = create_candidate_plan(
        material="Pt(111)",
        adsorbate="H2O",
        rationale="H2O 上 O 有 lone pair 给体，Pt(111) 顶层 4 个原子对称等价，选 ontop 主测 + 第二个位点对照。",
        expected_binding_motif="atop O-down upright",
        anchor_atom="O",
        target_sites=[
            {"site_id": chosen[0]["site_id"], "reason": "对称等价代表，经典 atop O-down"},
            {"site_id": chosen[1]["site_id"], "reason": "对照位点，验证位点敏感性"},
        ],
        target_orientations=["upright"],
        symmetry_pruning_applied=True,
        priors_consulted={"chemistry_hint_source": "curated", "knowledge_search_hits": 0},
    )

    authored_entries = []
    for index, site in enumerate(chosen, start=1):
        candidate_id = f"model_pick_{index:02d}_{site['site_family'].lower()}"
        out_poscar = tmp_path / f"{candidate_id}.POSCAR"
        added = add_adsorbate(
            slab_path=str(slab_path),
            adsorbate="H2O",
            output_path=str(out_poscar),
            cart_coords=site["cart_coords"],
            orientation="upright",
            fixed_bottom_layers=2,
        )
        assert added["status"] == "ok"
        sanity = sanity_check(str(out_poscar))
        assert sanity["status"] in {"ok", "warning"}
        authored_entries.append(
            {
                "candidate_id": candidate_id,
                "poscar_path": str(out_poscar),
                "site_family": site["site_family"],
                "site_label": site["site_id"],
                "orientation_label": "upright",
                "anchor_symbol": added["anchor_symbol"],
                "height": 2.0,
                "reason": f"基于 curated H2O hint 选 atop O-down，对应 {site['site_id']} 位点。",
                "metadata": {"site_cart_coords": site["cart_coords"]},
            }
        )

    output_dir = tmp_path / "authored_candidates"
    composed = compose_manifest_from_authored_candidates(
        task_id="ads_model_authored_test",
        material_name="Pt(111)",
        source_prompt="H2O 在 Pt(111) 上吸附（model-authored test）",
        slab_source=str(slab_path),
        adsorbate_source="H2O",
        output_dir=str(output_dir),
        candidates=authored_entries,
        plan_payload=plan.to_dict(),
    )
    assert composed["status"] == "composed"
    assert composed["candidate_count"] == 2
    assert composed["plan_id"] == plan.plan_id

    manifest_json = Path(composed["manifest_json"])
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["candidate_count"] == 2
    candidate_ids = {item["candidate_id"] for item in manifest["candidates"]}
    assert {entry["candidate_id"] for entry in authored_entries} == candidate_ids
    for candidate in manifest["candidates"]:
        files = candidate["exported_files"]
        assert Path(files["poscar_path"]).exists()
        assert Path(files["cif_path"]).exists()
        assert Path(files["summary_path"]).exists()

    selection_dir = tmp_path / "selection"
    selected_id = authored_entries[0]["candidate_id"]
    selection = ConfirmedCandidateHandoff().materialize_selection(
        manifest_path=manifest_json,
        candidate_id=selected_id,
        output_dir=selection_dir,
    )
    assert selection.candidate_id == selected_id
    assert Path(selection.selected_poscar_path).exists()
    assert Path(selection.selected_cif_path).exists()


def test_candidate_quality_score_flags_good_and_floating_candidates(tmp_path):
    slab_path = write_pt111_poscar(tmp_path / "POSCAR")
    site = enumerate_adsorption_sites(str(slab_path), max_sites_per_family=1)["sites"][0]

    good_poscar = tmp_path / "good.POSCAR"
    added = add_adsorbate(
        slab_path=str(slab_path),
        adsorbate="H2O",
        output_path=str(good_poscar),
        cart_coords=site["cart_coords"],
        anchor_symbol="O",
        orientation="upright",
        fixed_bottom_layers=2,
    )
    assert added["status"] == "ok"
    good = candidate_quality_score(
        slab_path=str(slab_path),
        candidate_path=str(good_poscar),
        adsorbate="H2O",
        anchor_symbol="O",
    )
    assert good["status"] == "ok"
    assert good["verdict"] == "pass"
    assert good["score"]["total"] >= 0.75
    assert good["measurements"]["anchor_surface_distance"] is not None

    floating_poscar = tmp_path / "floating.POSCAR"
    high_site = list(site["cart_coords"])
    high_site[2] += 5.0
    added = add_adsorbate(
        slab_path=str(slab_path),
        adsorbate="H2O",
        output_path=str(floating_poscar),
        cart_coords=high_site,
        anchor_symbol="O",
        orientation="upright",
        fixed_bottom_layers=2,
    )
    assert added["status"] == "ok"
    floating = candidate_quality_score(
        slab_path=str(slab_path),
        candidate_path=str(floating_poscar),
        adsorbate="H2O",
        anchor_symbol="O",
    )
    assert floating["status"] == "warning"
    assert floating["verdict"] in {"retry", "reject"}
    assert any("过远" in issue or "floating" in issue.lower() for issue in floating["issues"])


def test_compose_rejects_empty_or_dup(tmp_path):
    with pytest.raises(ValueError):
        compose_manifest_from_authored_candidates(
            task_id="t",
            material_name="m",
            source_prompt="p",
            slab_source="s",
            adsorbate_source="H2O",
            output_dir=str(tmp_path / "empty"),
            candidates=[],
        )
    fake_poscar = tmp_path / "fake.POSCAR"
    fake_poscar.write_text("garbage", encoding="utf-8")
    with pytest.raises(ValueError):
        compose_manifest_from_authored_candidates(
            task_id="t",
            material_name="m",
            source_prompt="p",
            slab_source="s",
            adsorbate_source="H2O",
            output_dir=str(tmp_path / "shortreason"),
            candidates=[
                {
                    "candidate_id": "x",
                    "poscar_path": str(fake_poscar),
                    "site_label": "ontop-01",
                    "reason": "too short",
                }
            ],
        )
