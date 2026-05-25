from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from dft_app.models import (
    ExperimentSpec,
    PhaseStatus,
    PipelinePhase,
    RunRecord,
    RunStatus,
    TaskType,
)
from dft_app.storage import RecordStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def copy_sample_run_fixture(*, relative_run_root: str, destination_root: Path) -> Path:
    """Copy one checked-in run tree into a temp workspace for Web UI tests."""

    source_root = PROJECT_ROOT / ".aether" / "runs" / relative_run_root
    target_root = destination_root / ".aether" / "runs" / relative_run_root
    shutil.copytree(source_root, target_root)
    return target_root


def create_minimal_run_fixture(
    *,
    project_root: Path,
    task_id: str = "task_web",
    run_id: str = "run_ready",
    material_name: str = "Cu slab",
    prompt: str = "测试 Web UI run 详情页",
    task_type: TaskType = TaskType.RELAX_SCF,
    overall_status: RunStatus = RunStatus.READY,
    current_phase: PipelinePhase = PipelinePhase.BUILD,
    scheduler_job_id: str | None = None,
    notes: dict[str, Any] | None = None,
    build_summary: dict[str, Any] | None = None,
    planner_summary: dict[str, Any] | None = None,
) -> Path:
    """Create the smallest RecordStore-backed run fixture for list/detail pages."""

    store = RecordStore(project_root)
    run_root = project_root / ".aether" / "runs" / task_id / run_id
    structure_path = run_root / "inputs" / "POSCAR"
    structure_path.parent.mkdir(parents=True, exist_ok=True)
    structure_path.write_text("fixture-poscar\n", encoding="utf-8")

    spec = ExperimentSpec(
        task_id=task_id,
        task_type=task_type,
        material_name=material_name,
        source_prompt=prompt,
        structure_path=str(structure_path),
    )
    store.write_metadata(run_root, "experiment_spec.json", spec.to_dict())

    run_record = RunRecord(
        task_id=task_id,
        run_id=run_id,
        run_root=str(run_root),
        checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
        overall_status=overall_status,
        current_phase=current_phase,
        scheduler_job_id=scheduler_job_id,
        notes=notes or {},
    )
    run_record.phases[PipelinePhase.PLAN.value].status = PhaseStatus.COMPLETED
    run_record.phases[current_phase.value].status = (
        PhaseStatus.COMPLETED
        if overall_status in {RunStatus.READY, RunStatus.COMPLETED}
        else PhaseStatus.RUNNING
    )
    store.save_run_record(run_record)

    if planner_summary is not None:
        store.write_metadata(run_root, "planner_summary.json", planner_summary)
    if build_summary is not None:
        store.write_metadata(run_root, "build_summary.json", build_summary)

    return run_root


def create_adsorption_workflow_fixture(
    *,
    project_root: Path,
    task_id: str = "task_ads_web",
    run_id: str = "run_status_web",
) -> Path:
    """Create a minimal adsorption workflow tree for Web UI workflow tests."""

    store = RecordStore(project_root)
    parent_root = create_minimal_run_fixture(
        project_root=project_root,
        task_id=task_id,
        run_id=run_id,
        material_name="Cu slab",
        prompt="计算 H2O 在 Cu(111) 上的吸附能",
    )

    subtask_roots = {
        "clean_slab": parent_root / "scaffold" / "subtasks" / "01_clean_slab",
        "isolated_adsorbate": parent_root / "scaffold" / "subtasks" / "02_isolated_adsorbate",
        "adsorbed_system": parent_root / "scaffold" / "subtasks" / "03_adsorbed_system",
    }
    subtask_materials = {
        "clean_slab": "Cu slab-clean-slab",
        "isolated_adsorbate": "H2O-isolated",
        "adsorbed_system": "Cu slab-adsorbed",
    }

    for name, subtask_root in subtask_roots.items():
        structure_path = subtask_root / "inputs" / "POSCAR"
        structure_path.parent.mkdir(parents=True, exist_ok=True)
        structure_path.write_text(f"{name}-poscar\n", encoding="utf-8")
        spec = ExperimentSpec(
            task_id=f"{task_id}_{name}",
            task_type=TaskType.RELAX_SCF,
            material_name=subtask_materials[name],
            source_prompt=f"{name} fixture",
            structure_path=str(structure_path),
        )
        store.write_metadata(subtask_root, "experiment_spec.json", spec.to_dict())
        run_record = RunRecord(
            task_id=f"{task_id}_{name}",
            run_id=subtask_root.name,
            run_root=str(subtask_root),
            checkpoint_path=str(subtask_root / "outputs" / ".pipeline_checkpoint.json"),
            overall_status=RunStatus.READY,
            current_phase=PipelinePhase.BUILD,
        )
        run_record.phases[PipelinePhase.PLAN.value].status = PhaseStatus.COMPLETED
        run_record.phases[PipelinePhase.BUILD.value].status = PhaseStatus.COMPLETED
        store.save_run_record(run_record)

    store.write_metadata(
        parent_root,
        "adsorption_workflow_bundle.json",
        {
            "status": "prepared",
            "selected_candidate_id": "ontop_01_upright",
            "subtasks": {
                name: {"bundle_root": str(subtask_root)}
                for name, subtask_root in subtask_roots.items()
            },
        },
    )
    candidate_manifest_dir = parent_root / "scaffold" / "adsorption_candidates"
    candidate_manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = candidate_manifest_dir / "candidate_manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "candidate_count": 2,
                "candidates": [
                    {"candidate_id": "ontop_01_upright"},
                    {"candidate_id": "bridge_01_flat"},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_md = candidate_manifest_dir / "candidate_manifest.md"
    manifest_md.write_text("# candidates\n", encoding="utf-8")
    store.write_metadata(
        parent_root,
        "adsorption_candidate_generation.json",
        {
            "status": "generated",
            "candidate_count": 2,
            "manifest": {
                "manifest_json": str(manifest_json),
                "manifest_md": str(manifest_md),
            },
        },
    )
    return parent_root

