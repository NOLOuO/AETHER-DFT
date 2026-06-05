from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase.build import fcc111
from pymatgen.io.ase import AseAtomsAdaptor

from aether_dft.adsorption_authoring import (
    create_candidate_plan,
    list_candidate_plans,
    load_candidate_plan,
)
from aether_dft.project_state import init_project
from aether_dft.prompt_engine import render_compiled_system_prompt
from aether_dft.runtime_harness.tool_registry import ToolRegistry
from dft_app.modeling import compose_manifest_from_authored_candidates
from dft_shared.structure_analyzer.operations import add_adsorbate, enumerate_adsorption_sites


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


def _good_plan_kwargs(material="Pt(111)", adsorbate="H2O"):
    return {
        "material": material,
        "adsorbate": adsorbate,
        "rationale": (
            "Pt(111) 顶层原子全等价；H2O 经典 atop O-down，"
            "选 ontop-01 主测、ontop-02 对照，剩余 hollow 不在 motif 内排除。"
        ),
        "expected_binding_motif": "atop O-down upright",
        "anchor_atom": "O",
        "target_sites": [
            {"site_id": "ontop-01", "reason": "对称代表 + O-down 主测"},
            {"site_id": "ontop-02", "reason": "邻位对照，验证 site dependence"},
        ],
        "target_orientations": ["upright"],
        "excluded_sites_with_reason": [
            {"site_id": "hollow-01", "reason": "lone pair 几何不匹配"},
        ],
        "symmetry_pruning_applied": True,
        "priors_consulted": {"chemistry_hint_source": "curated"},
    }


def test_create_candidate_plan_persists_and_loads():
    init_project("pytest-plan-roundtrip", overwrite=True)
    plan = create_candidate_plan(project="pytest-plan-roundtrip", **_good_plan_kwargs())
    assert plan.plan_id.startswith("plan_")
    assert Path(plan.plan_path).exists()
    loaded = load_candidate_plan(plan.plan_id, project="pytest-plan-roundtrip")
    assert loaded.plan_id == plan.plan_id
    assert loaded.target_site_ids() == ["ontop-01", "ontop-02"]
    plans = list_candidate_plans("pytest-plan-roundtrip")
    assert any(item["plan_id"] == plan.plan_id for item in plans)


def test_plan_rejects_short_rationale():
    kwargs = _good_plan_kwargs()
    kwargs["rationale"] = "太短"
    with pytest.raises(ValueError, match="rationale"):
        create_candidate_plan(**kwargs)


def test_create_candidate_plan_non_strict_records_quality_warnings():
    kwargs = _good_plan_kwargs()
    kwargs["rationale"] = "太短"
    kwargs["target_sites"] = [{"site_id": "ontop-01", "reason": "x"}]
    kwargs["target_orientations"] = []

    plan = create_candidate_plan(**kwargs, strict=False)

    assert plan.plan_id.startswith("plan_")
    assert Path(plan.plan_path).exists()
    assert plan.quality_warnings
    assert any("rationale" in item for item in plan.quality_warnings)
    assert any("target_sites" in item for item in plan.quality_warnings)
    assert any("target_orientations" in item for item in plan.quality_warnings)
    assert plan.target_orientations == ["unspecified"]


def test_registry_plan_handler_returns_soft_warning_on_short_rationale(tmp_path):
    """guided-not-enforced：tool 调用面不再向模型抛 raise，而是 status=needs_revision。"""
    registry = ToolRegistry()
    raw = registry.run_tool(
        "adsorption_candidate_plan",
        {
            "material": "Pt(111)",
            "adsorbate": "H2O",
            "rationale": "太短",
            "expected_binding_motif": "atop O-down",
            "anchor_atom": "O",
            "target_sites": [{"site_id": "ontop-01", "reason": "对称代表"}],
            "target_orientations": ["upright"],
        },
    )
    result = raw["result"]
    assert result["status"] == "needs_revision"
    assert "rationale" in result["warning"]
    assert result["plan"]["quality_warnings"]
    assert result["echo"]["material"] == "Pt(111)"
    # 不应该把 raise 暴露成 tool registry 的 error status
    assert "error" not in result["status"]


def test_registry_plan_handler_returns_soft_warning_on_short_site_reason():
    registry = ToolRegistry()
    raw = registry.run_tool(
        "adsorption_candidate_plan",
        {
            "material": "Pt(111)",
            "adsorbate": "H2O",
            "rationale": "Pt(111) 顶层原子全等价，H2O O-down 是经典 motif；选 ontop-01 主测、ontop-02 对照。",
            "expected_binding_motif": "atop O-down",
            "anchor_atom": "O",
            "target_sites": [{"site_id": "ontop-01", "reason": "x"}],
            "target_orientations": ["upright"],
        },
    )
    result = raw["result"]
    assert result["status"] == "needs_revision"
    assert "reason" in result["warning"]
    assert result["min_chars_hint"]["target_site_reason"] == 10


def test_plan_rejects_empty_target_sites():
    kwargs = _good_plan_kwargs()
    kwargs["target_sites"] = []
    with pytest.raises(ValueError, match="target_sites"):
        create_candidate_plan(**kwargs)


def test_plan_rejects_short_site_reason():
    kwargs = _good_plan_kwargs()
    kwargs["target_sites"] = [{"site_id": "ontop-01", "reason": "x"}]
    with pytest.raises(ValueError, match="reason"):
        create_candidate_plan(**kwargs)


def test_plan_rejects_dup_site_ids():
    kwargs = _good_plan_kwargs()
    kwargs["target_sites"] = [
        {"site_id": "ontop-01", "reason": "对称代表"},
        {"site_id": "ontop-01", "reason": "重复测试"},
    ]
    with pytest.raises(ValueError, match="重复"):
        create_candidate_plan(**kwargs)


def test_plan_rejects_empty_orientations():
    kwargs = _good_plan_kwargs()
    kwargs["target_orientations"] = []
    with pytest.raises(ValueError, match="target_orientations"):
        create_candidate_plan(**kwargs)


def _write_pt111_poscar(path: Path) -> Path:
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=8.0)
    structure = AseAtomsAdaptor.get_structure(atoms)
    path.write_text(structure.to(fmt="poscar"), encoding="utf-8")
    return path


def test_compose_missing_plan_id_returns_warning_via_registry(tmp_path):
    """guided-not-enforced：缺 plan_id 不再 error；compose 通过 + quality_warnings 包含 plan_missing。"""
    slab_path = _write_pt111_poscar(tmp_path / "POSCAR")
    sites = enumerate_adsorption_sites(str(slab_path))["sites"][:1]
    added = add_adsorbate(
        slab_path=str(slab_path),
        adsorbate="H2O",
        output_path=str(tmp_path / "c.POSCAR"),
        cart_coords=sites[0]["cart_coords"],
    )
    assert added["status"] == "ok"
    registry = ToolRegistry()
    raw = registry.run_tool(
        "adsorption_candidate_manifest_compose",
        {
            "task_id": "t",
            "material_name": "Pt(111)",
            "source_prompt": "p",
            "slab_source": str(slab_path),
            "adsorbate_source": "H2O",
            "output_dir": str(tmp_path / "out"),
            "candidates": [
                {
                    "candidate_id": "c1",
                    "poscar_path": str(tmp_path / "c.POSCAR"),
                    "site_label": sites[0]["site_id"],
                    "reason": "atop O-down 主测，符合 chemistry hint 与对称依据",
                }
            ],
        },
    )
    result = raw["result"]
    assert result["status"] == "composed"
    assert result["has_warnings"] is True
    codes = {w["code"] for w in result["quality_warnings"]}
    assert "plan_missing" in codes
    # 自动 audit 也应跟随
    assert "audit" in result
    assert result["audit"]["status"] == "ok"


def test_compose_site_label_not_in_plan_returns_warning(tmp_path):
    """guided-not-enforced：site_label 不在 plan 不再 raise；改为 warning。"""
    slab_path = _write_pt111_poscar(tmp_path / "POSCAR")
    sites = enumerate_adsorption_sites(str(slab_path))["sites"][:1]
    added = add_adsorbate(
        slab_path=str(slab_path),
        adsorbate="H2O",
        output_path=str(tmp_path / "c.POSCAR"),
        cart_coords=sites[0]["cart_coords"],
    )
    assert added["status"] == "ok"
    plan = create_candidate_plan(**_good_plan_kwargs())
    result = compose_manifest_from_authored_candidates(
        task_id="t",
        material_name="Pt(111)",
        source_prompt="p",
        slab_source=str(slab_path),
        adsorbate_source="H2O",
        output_dir=str(tmp_path / "out"),
        candidates=[
            {
                "candidate_id": "c1",
                "poscar_path": str(tmp_path / "c.POSCAR"),
                "site_label": "ghost-99",
                "reason": "atop O-down 主测，符合 chemistry hint 与对称依据",
            }
        ],
        plan_payload=plan.to_dict(),
    )
    assert result["status"] == "composed"
    assert result["has_warnings"] is True
    codes = {w["code"] for w in result["quality_warnings"]}
    assert "site_label_not_in_plan" in codes


def test_compose_too_many_candidates_returns_warning(tmp_path):
    """guided-not-enforced：候选数超阈值无 prune_rationale 不再 raise；改为 warning。"""
    slab_path = _write_pt111_poscar(tmp_path / "POSCAR")
    sites = enumerate_adsorption_sites(str(slab_path), max_sites_per_family=4)["sites"]
    assert len(sites) >= 7, "Pt(111) 需要足够多位点触发阈值"
    used = sites[:7]

    plan_kwargs = _good_plan_kwargs()
    plan_kwargs["target_sites"] = [
        {"site_id": site["site_id"], "reason": f"plan 覆盖 {site['site_family']} 测试"}
        for site in used
    ]
    plan = create_candidate_plan(**plan_kwargs)

    entries = []
    for index, site in enumerate(used, start=1):
        poscar = tmp_path / f"c_{index:02d}.POSCAR"
        added = add_adsorbate(
            slab_path=str(slab_path),
            adsorbate="H2O",
            output_path=str(poscar),
            cart_coords=site["cart_coords"],
        )
        assert added["status"] == "ok"
        entries.append(
            {
                "candidate_id": f"c_{index:02d}",
                "poscar_path": str(poscar),
                "site_label": site["site_id"],
                "reason": "broad 覆盖测试，超阈值 + 无 prune_rationale 应该出 warning",
            }
        )
    result = compose_manifest_from_authored_candidates(
        task_id="broad",
        material_name="Pt(111)",
        source_prompt="broad coverage test",
        slab_source=str(slab_path),
        adsorbate_source="H2O",
        output_dir=str(tmp_path / "broad"),
        candidates=entries,
        plan_payload=plan.to_dict(),
    )
    assert result["status"] == "composed"
    assert result["has_warnings"] is True
    codes = {w["code"] for w in result["quality_warnings"]}
    assert "prune_rationale_missing" in codes


def test_compose_succeeds_with_prune_rationale(tmp_path):
    slab_path = _write_pt111_poscar(tmp_path / "POSCAR")
    sites = enumerate_adsorption_sites(str(slab_path), max_sites_per_family=4)["sites"][:7]
    plan_kwargs = _good_plan_kwargs()
    plan_kwargs["target_sites"] = [
        {"site_id": site["site_id"], "reason": "覆盖测试合法位点理由"} for site in sites
    ]
    plan = create_candidate_plan(**plan_kwargs)
    entries = []
    for index, site in enumerate(sites, start=1):
        poscar = tmp_path / f"c_{index:02d}.POSCAR"
        added = add_adsorbate(
            slab_path=str(slab_path),
            adsorbate="H2O",
            output_path=str(poscar),
            cart_coords=site["cart_coords"],
        )
        assert added["status"] == "ok"
        entries.append(
            {
                "candidate_id": f"c_{index:02d}",
                "poscar_path": str(poscar),
                "site_label": site["site_id"],
                "reason": "broad 覆盖测试，验证 prune_rationale 通过路径",
            }
        )
    composed = compose_manifest_from_authored_candidates(
        task_id="broad",
        material_name="Pt(111)",
        source_prompt="broad coverage test",
        slab_source=str(slab_path),
        adsorbate_source="H2O",
        output_dir=str(tmp_path / "broad_ok"),
        candidates=entries,
        plan_payload=plan.to_dict(),
        prune_rationale="保留 7 个对称等价代表位点，已合并 hollow 系列",
    )
    assert composed["status"] == "composed"
    manifest = json.loads(Path(composed["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["metadata"]["prune_rationale"].startswith("保留")
    assert manifest["metadata"]["plan_id"] == plan.plan_id


def test_prompt_includes_adsorption_authoring_section():
    rendered = render_compiled_system_prompt()
    assert "一个有经验的同组合作者通常这样想" in rendered
    assert "adsorption_candidate_plan" in rendered
    assert "manifest_audit" in rendered
    # 不再期望存在硬性"红线"/"必须"段落
    assert "### 红线" not in rendered


def test_manifest_audit_returns_scored_findings(tmp_path):
    """新增 manifest_audit 工具：读已存盘 manifest 给行为画像评分。"""
    slab_path = _write_pt111_poscar(tmp_path / "POSCAR")
    sites = enumerate_adsorption_sites(str(slab_path))["sites"][:2]
    plan_kwargs = _good_plan_kwargs()
    plan_kwargs["target_sites"] = [
        {"site_id": site["site_id"], "reason": f"{site['site_family']} 位点科学依据"}
        for site in sites
    ]
    plan = create_candidate_plan(**plan_kwargs)
    entries = []
    for index, site in enumerate(sites, start=1):
        poscar = tmp_path / f"c_{index}.POSCAR"
        add_adsorbate(
            slab_path=str(slab_path),
            adsorbate="H2O",
            output_path=str(poscar),
            cart_coords=site["cart_coords"],
        )
        entries.append(
            {
                "candidate_id": f"c_{index}",
                "poscar_path": str(poscar),
                "site_label": site["site_id"],
                "reason": "atop O-down 主测，基于 curated chemistry hint + 对称等价合并",
            }
        )
    result = compose_manifest_from_authored_candidates(
        task_id="audit_test",
        material_name="Pt(111)",
        source_prompt="audit",
        slab_source=str(slab_path),
        adsorbate_source="H2O",
        output_dir=str(tmp_path / "audit_out"),
        candidates=entries,
        plan_payload=plan.to_dict(),
    )
    registry = ToolRegistry()
    raw = registry.run_tool("manifest_audit", {"manifest_path": result["manifest_json"]})
    audit = raw["result"]
    assert audit["status"] == "ok"
    assert 0.0 <= audit["total_score"] <= 1.0
    assert set(audit["score_breakdown"].keys()) == {
        "plan_link", "reason_quality", "site_alignment", "prior_consultation", "candidate_size",
    }
    assert isinstance(audit["findings"], list)
    assert isinstance(audit["suggestions"], list)
