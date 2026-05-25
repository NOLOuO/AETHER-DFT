from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from ase.build import bulk, fcc111
from pymatgen.io.ase import AseAtomsAdaptor

from aether_dft import cli
from aether_dft.adsorption import (
    build_adsorption_slab,
    generate_adsorption_candidates,
    plan_adsorption_task,
    run_adsorption_full_workflow,
    run_adsorption_pipeline,
)
from aether_dft.project_state import init_project
from aether_dft.recommendations import recommend_next_tasks
from dft_app.llm.key_store import resolve_api_key


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


def test_adsorption_plan_missing_slab_recommends_next_steps():
    plan = plan_adsorption_task(
        "计算 H2O 在 Pt(111) 上的吸附",
        adsorbate="H2O",
        material="Pt(111)",
        persist=False,
    )
    assert plan.readiness == "needs_confirmation"
    assert "slab_path" in plan.missing_inputs
    assert any("build-slab" in item or "slab" in item.lower() for item in plan.next_research_tasks)
    assert "build_slab" in plan.dft_entrypoints


def test_build_adsorption_slab_from_ase_element(tmp_path):
    result = build_adsorption_slab(
        material="Pt(111)",
        output_dir=str(tmp_path / "slab"),
        source="ase",
        supercell=(2, 2, 1),
    )
    assert result.status == "ok"
    assert result.source == "ase"
    assert result.miller_index == (1, 1, 1)
    assert result.atom_count > 0
    assert Path(result.slab_path).exists()
    assert Path(result.metadata_path).exists()


def test_build_adsorption_slab_from_local_bulk(tmp_path):
    bulk_path = tmp_path / "bulk_pt.cif"
    atoms = bulk("Pt")
    structure = AseAtomsAdaptor.get_structure(atoms)
    bulk_path.write_text(structure.to(fmt="cif"), encoding="utf-8")
    result = build_adsorption_slab(
        material="Pt(111)",
        output_dir=str(tmp_path / "local_slab"),
        structure_path=str(bulk_path),
        supercell=(1, 1, 1),
    )
    assert result.status == "ok"
    assert result.source == "local"
    assert Path(result.slab_path).exists()


def test_build_adsorption_slab_mp_without_key_is_honest(tmp_path, monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.delenv("MATERIALS_PROJECT_API_KEY", raising=False)
    monkeypatch.setattr("aether_dft.adsorption.resolve_api_key", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="API key"):
        build_adsorption_slab(
            material="Pt",
            output_dir=str(tmp_path / "mp_slab"),
            source="mp",
            mp_id="mp-126",
        )


def test_resolve_api_key_from_local_file(tmp_path):
    (tmp_path / "api_keys.local.json").write_text(
        json.dumps({"materials_project": "local-secret"}),
        encoding="utf-8",
    )
    assert resolve_api_key(tmp_path, aliases=("materials_project", "mp")) == "local-secret"


def test_adsorption_candidates_generates_manifest(tmp_path):
    slab = write_pt111_poscar(tmp_path / "POSCAR")
    result = generate_adsorption_candidates(
        slab_path=str(slab),
        adsorbate="H2O",
        material="Pt",
        prompt="H2O adsorption on Pt(111)",
        output_dir=str(tmp_path / "candidates"),
        max_sites_per_family=1,
    )
    assert result["status"] == "ok"
    assert result["result"]["candidate_count"] > 0
    manifest = Path(result["result"]["manifest"]["manifest_json"])
    assert manifest.exists()
    assert any(Path(item["exported_files"]["poscar_path"]).exists() for item in result["result"]["top_candidates"])


def test_cli_adsorption_plan_and_recommend(capsys):
    init_project("adsorption-demo", description="adsorption demo", overwrite=True)
    assert cli.main([
        "adsorption",
        "plan",
        "计算",
        "H2O",
        "在",
        "Pt(111)",
        "上吸附",
        "--project",
        "adsorption-demo",
        "--adsorbate",
        "H2O",
        "--material",
        "Pt(111)",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["readiness"] == "needs_confirmation"
    assert cli.main(["recommend", "--project", "adsorption-demo", "--focus", "adsorption"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommendations"]
    assert any("候选" in item["title"] or "slab" in item["title"].lower() for item in payload["recommendations"])


def test_cli_adsorption_candidates_smoke(tmp_path, capsys):
    slab = write_pt111_poscar(tmp_path / "POSCAR")
    output = tmp_path / "out"
    assert cli.main([
        "adsorption",
        "candidates",
        "--slab-path",
        str(slab),
        "--adsorbate",
        "H2O",
        "--material",
        "Pt",
        "--output-dir",
        str(output),
        "--max-sites-per-family",
        "1",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["result"]["candidate_count"] > 0


def test_cli_adsorption_build_slab_and_pipeline(tmp_path, capsys):
    slab_dir = tmp_path / "cli_slab"
    assert cli.main([
        "adsorption",
        "build-slab",
        "--material",
        "Pt(111)",
        "--source",
        "ase",
        "--output-dir",
        str(slab_dir),
        "--supercell",
        "1",
        "1",
        "1",
    ]) == 0
    slab_payload = json.loads(capsys.readouterr().out)
    assert slab_payload["status"] == "ok"
    assert Path(slab_payload["slab_path"]).exists()

    pipeline_dir = tmp_path / "pipeline"
    assert cli.main([
        "adsorption",
        "pipeline",
        "--material",
        "Pt(111)",
        "--adsorbate",
        "H2O",
        "--source",
        "ase",
        "--output-dir",
        str(pipeline_dir),
        "--supercell",
        "1",
        "1",
        "1",
        "--max-sites-per-family",
        "1",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert Path(payload["slab"]["slab_path"]).exists()
    assert payload["candidates"]["result"]["candidate_count"] > 0


def test_run_adsorption_pipeline_programmatic(tmp_path):
    result = run_adsorption_pipeline(
        material="Pt(111)",
        adsorbate="H2O",
        output_dir=str(tmp_path / "programmatic_pipeline"),
        source="ase",
        supercell=(1, 1, 1),
        max_sites_per_family=1,
    )
    assert result["status"] == "ok"
    assert Path(result["slab"]["slab_path"]).exists()
    assert result["candidates"]["result"]["candidate_count"] > 0


def test_run_adsorption_full_workflow_materializes_three_vasp_workspaces(tmp_path):
    result = run_adsorption_full_workflow(
        material="Pt(111)",
        adsorbate="H2O",
        output_dir=str(tmp_path / "full_workflow"),
        source="ase",
        supercell=(1, 1, 1),
        max_sites_per_family=1,
    )
    try:
        assert result["status"] == "prepared"
        assert result["workflow_status"]["status"] == "prepared"
        subtasks = result["workflow_bundle"]["subtasks"]
        for name in ("clean_slab", "isolated_adsorbate", "adsorbed_system"):
            assert Path(subtasks[name]["inputs"]["poscar_path"]).exists()
            assert Path(subtasks[name]["inputs"]["incar_path"]).exists()
            assert Path(subtasks[name]["job_slurm"]).exists()
        assert "尚未运行 VASP" in result["honest_boundary"]
    finally:
        run_root = Path(result.get("run_root", ""))
        if run_root.exists() and "AETHER-DFT" in str(run_root):
            shutil.rmtree(run_root.parent, ignore_errors=True)
