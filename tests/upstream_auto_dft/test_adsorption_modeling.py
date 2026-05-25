from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ase.build import fcc111
from pymatgen.io.ase import AseAtomsAdaptor

from dft_app.cli.main import (
    collect_adsorption_workflow_status,
    execute_adsorption_workflow,
    maybe_generate_adsorption_candidates,
    maybe_materialize_adsorption_workflow_bundle,
    maybe_select_adsorption_candidate,
)
from dft_app.modeling import (
    AdsorptionCandidateGenerator,
    AdsorptionGenerationRequest,
    CandidateManifest,
    CandidateManifestWriter,
    ConfirmedCandidateHandoff,
    TaskModeler,
)
from dft_app.orchestrator import ComplexWorkflowOrchestrator
from dft_app.models import ExperimentPlan, ExecutionReadiness, PlanComplexity, RunRecord, RunStatus
from dft_app.storage import RecordStore
from tests.temp_workspace import workspace_tempdir


class AdsorptionModelingTests(unittest.TestCase):
    def test_generate_adsorption_candidates_for_h2o(self) -> None:
        slab = AseAtomsAdaptor.get_structure(fcc111("Cu", size=(2, 2, 3), vacuum=12.0))
        generator = AdsorptionCandidateGenerator()
        candidates = generator.generate(
            AdsorptionGenerationRequest(
                slab_structure=slab,
                adsorbate_source="H2O",
                task_id="task_test",
                material_name="Cu slab",
                source_prompt="test adsorption",
                max_sites_per_family=1,
            )
        )
        self.assertGreaterEqual(len(candidates), 1)
        top_candidate = candidates[0]
        self.assertIsNotNone(top_candidate.structure)
        self.assertIn("minimum_clearance", top_candidate.metadata)
        self.assertIsNotNone(top_candidate.score)

    def test_manifest_writer_and_selection_handoff(self) -> None:
        slab = AseAtomsAdaptor.get_structure(fcc111("Cu", size=(2, 2, 3), vacuum=12.0))
        generator = AdsorptionCandidateGenerator()
        candidates = generator.generate(
            AdsorptionGenerationRequest(
                slab_structure=slab,
                adsorbate_source="H2O",
                task_id="task_test",
                material_name="Cu slab",
                source_prompt="test adsorption",
                max_sites_per_family=1,
            )
        )
        manifest = CandidateManifest(
            task_id="task_test",
            material_name="Cu slab",
            source_prompt="test adsorption",
            slab_source="slab.vasp",
            adsorbate_source="H2O",
            candidates=candidates,
        )
        writer = CandidateManifestWriter()
        handoff = ConfirmedCandidateHandoff()
        with workspace_tempdir("ads_manifest_") as out:
            paths = writer.write(manifest, out)
            manifest_path = Path(paths["manifest_json"])
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["candidate_count"], len(candidates))
            selection = handoff.materialize_selection(
                manifest_path=manifest_path,
                candidate_id=candidates[0].candidate_id,
                output_dir=out / "selected",
            )
            self.assertTrue(Path(selection.selected_poscar_path).exists())
            self.assertTrue(Path(selection.selected_summary_path).exists())

    def test_mainline_adsorption_candidate_generation_helper(self) -> None:
        with workspace_tempdir("ads_mainline_") as tmp_path:
            slab_path = tmp_path / "slab.vasp"
            slab_atoms = fcc111("Cu", size=(2, 2, 3), vacuum=12.0)
            slab_path.write_text(AseAtomsAdaptor.get_structure(slab_atoms).to(fmt="poscar"), encoding="utf-8")
            run_root = tmp_path / ".aether" / "runs" / "task_ads" / "run_demo"
            run_record = RunRecord(
                task_id="task_ads",
                run_id="run_demo",
                run_root=str(run_root),
                checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
            )
            plan = ExperimentPlan(
                task_id="task_ads",
                source_prompt="计算 H2O 在 Cu(111) 面上的吸附能",
                experiment_type="adsorption_energy",
                summary="Cu(111)+H2O adsorption",
                complexity=PlanComplexity.COMPLEX,
                readiness=ExecutionReadiness.NEEDS_CONFIRMATION,
                requires_confirmation=True,
                raw_plan={},
            )

            class Args:
                structure_path = str(slab_path)

            result = maybe_generate_adsorption_candidates(
                args=Args(),
                plan=plan,
                run_record=run_record,
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result["status"], "generated")
            self.assertGreaterEqual(result["candidate_count"], 1)
            self.assertTrue(Path(result["manifest"]["manifest_json"]).exists())

    def test_orchestrator_scaffold_includes_adsorption_candidates(self) -> None:
        with workspace_tempdir("ads_orchestrator_") as tmp_path:
            slab_path = tmp_path / "slab.vasp"
            slab_atoms = fcc111("Cu", size=(2, 2, 3), vacuum=12.0)
            slab_path.write_text(AseAtomsAdaptor.get_structure(slab_atoms).to(fmt="poscar"), encoding="utf-8")
            run_root = tmp_path / ".aether" / "runs" / "task_ads" / "run_orchestrator"
            run_record = RunRecord(
                task_id="task_ads",
                run_id="run_orchestrator",
                run_root=str(run_root),
                checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
            )
            plan = ExperimentPlan(
                task_id="task_ads",
                source_prompt="计算 H2O 在 Cu(111) 面上的吸附能",
                experiment_type="adsorption_energy",
                summary="Cu(111)+H2O adsorption",
                complexity=PlanComplexity.COMPLEX,
                readiness=ExecutionReadiness.NEEDS_CONFIRMATION,
                requires_confirmation=True,
                raw_plan={"material_name": "H2O/Cu(111)"},
            )
            model_spec = TaskModeler().build(spec=None, plan=plan).model_spec
            orchestrator = ComplexWorkflowOrchestrator(Path(__file__).resolve().parents[1])
            result = orchestrator.scaffold(
                plan,
                run_record,
                model_spec=model_spec,
                structure_path=str(slab_path),
            )
            self.assertEqual(result.status, "scaffolded")
            self.assertIn("adsorption_candidates", result.details)
            self.assertTrue(
                Path(result.details["adsorption_candidates"]["manifest"]["manifest_json"]).exists()
            )

    def test_mainline_adsorption_candidate_selection_helper(self) -> None:
        with workspace_tempdir("ads_select_") as tmp_path:
            slab_path = tmp_path / "slab.vasp"
            slab_atoms = fcc111("Cu", size=(2, 2, 3), vacuum=12.0)
            slab_path.write_text(AseAtomsAdaptor.get_structure(slab_atoms).to(fmt="poscar"), encoding="utf-8")
            run_root = tmp_path / ".aether" / "runs" / "task_ads" / "run_select"
            run_record = RunRecord(
                task_id="task_ads",
                run_id="run_select",
                run_root=str(run_root),
                checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
            )
            plan = ExperimentPlan(
                task_id="task_ads",
                source_prompt="计算 H2O 在 Cu(111) 面上的吸附能",
                experiment_type="adsorption_energy",
                summary="Cu(111)+H2O adsorption",
                complexity=PlanComplexity.COMPLEX,
                readiness=ExecutionReadiness.NEEDS_CONFIRMATION,
                requires_confirmation=True,
                raw_plan={},
            )

            class Args:
                structure_path = str(slab_path)
                selected_candidate_id = "ontop_01_upright"
                material = "Cu slab"
                submit_profile = None
                submit = False
                remote = False

            candidate_result = maybe_generate_adsorption_candidates(
                args=Args(),
                plan=plan,
                run_record=run_record,
            )
            selection_result = maybe_select_adsorption_candidate(
                args=Args(),
                plan=plan,
                run_record=run_record,
                adsorption_candidate_result=candidate_result,
            )
            self.assertIsNotNone(selection_result)
            assert selection_result is not None
            self.assertEqual(selection_result["status"], "selected")
            self.assertEqual(selection_result["build_result"]["status"], "ready")

    def test_mainline_adsorption_workflow_bundle_helper(self) -> None:
        with workspace_tempdir("ads_bundle_") as tmp_path:
            slab_path = tmp_path / "slab.vasp"
            slab_atoms = fcc111("Cu", size=(2, 2, 3), vacuum=12.0)
            slab_path.write_text(AseAtomsAdaptor.get_structure(slab_atoms).to(fmt="poscar"), encoding="utf-8")
            run_root = tmp_path / ".aether" / "runs" / "task_ads" / "run_bundle"
            run_record = RunRecord(
                task_id="task_ads",
                run_id="run_bundle",
                run_root=str(run_root),
                checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
            )
            plan = ExperimentPlan(
                task_id="task_ads",
                source_prompt="计算 H2O 在 Cu(111) 面上的吸附能",
                experiment_type="adsorption_energy",
                summary="Cu(111)+H2O adsorption",
                complexity=PlanComplexity.COMPLEX,
                readiness=ExecutionReadiness.NEEDS_CONFIRMATION,
                requires_confirmation=True,
                raw_plan={},
            )

            class Args:
                structure_path = str(slab_path)
                selected_candidate_id = "ontop_01_upright"
                material = "Cu slab"
                submit_profile = None
                submit = False
                remote = False

            candidate_result = maybe_generate_adsorption_candidates(
                args=Args(),
                plan=plan,
                run_record=run_record,
            )
            selection_result = maybe_select_adsorption_candidate(
                args=Args(),
                plan=plan,
                run_record=run_record,
                adsorption_candidate_result=candidate_result,
            )
            bundle_result = maybe_materialize_adsorption_workflow_bundle(
                args=Args(),
                plan=plan,
                run_record=run_record,
                selected_candidate_result=selection_result,
            )
            self.assertIsNotNone(bundle_result)
            assert bundle_result is not None
            self.assertEqual(bundle_result["status"], "prepared")
            self.assertTrue(Path(bundle_result["subtasks"]["clean_slab"]["job_slurm"]).exists())
            self.assertTrue(Path(bundle_result["subtasks"]["isolated_adsorbate"]["inputs"]["poscar_path"]).exists())
            self.assertTrue(Path(bundle_result["subtasks"]["adsorbed_system"]["inputs"]["poscar_path"]).exists())
            self.assertTrue(Path(bundle_result["subtasks"]["clean_slab"]["bundle_root"], "metadata", "run_record.json").exists())

    def test_collect_adsorption_workflow_status_reports_submit_readiness(self) -> None:
        with workspace_tempdir("ads_status_") as tmp_path:
            slab_path = tmp_path / "slab.vasp"
            slab_atoms = fcc111("Cu", size=(2, 2, 3), vacuum=12.0)
            slab_path.write_text(AseAtomsAdaptor.get_structure(slab_atoms).to(fmt="poscar"), encoding="utf-8")
            run_root = tmp_path / ".aether" / "runs" / "task_ads" / "run_status"
            run_record = RunRecord(
                task_id="task_ads",
                run_id="run_status",
                run_root=str(run_root),
                checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
            )
            plan = ExperimentPlan(
                task_id="task_ads",
                source_prompt="计算 H2O 在 Cu(111) 面上的吸附能",
                experiment_type="adsorption_energy",
                summary="Cu(111)+H2O adsorption",
                complexity=PlanComplexity.COMPLEX,
                readiness=ExecutionReadiness.NEEDS_CONFIRMATION,
                requires_confirmation=True,
                raw_plan={},
            )

            class Args:
                structure_path = str(slab_path)
                selected_candidate_id = "ontop_01_upright"
                material = "Cu slab"
                submit_profile = None
                submit = False
                remote = False

            candidate_result = maybe_generate_adsorption_candidates(args=Args(), plan=plan, run_record=run_record)
            selection_result = maybe_select_adsorption_candidate(
                args=Args(), plan=plan, run_record=run_record, adsorption_candidate_result=candidate_result
            )
            bundle_result = maybe_materialize_adsorption_workflow_bundle(
                args=Args(), plan=plan, run_record=run_record, selected_candidate_result=selection_result
            )
            assert bundle_result is not None
            summary_root = Path(bundle_result["summary_path"]).parent.parent

            status = collect_adsorption_workflow_status(summary_root)
            self.assertEqual(status["status"], "prepared")
            self.assertEqual(set(status["submit_ready"]), {"clean_slab", "isolated_adsorbate", "adsorbed_system"})
            self.assertIn("提交尚未启动的子任务（adsorption-workflow --submit）。", status["recommended_next_steps"])
            self.assertTrue((summary_root / "metadata" / "adsorption_workflow_status.json").exists())

    def test_execute_adsorption_workflow_monitor_and_fetch_persist_remote_summaries(self) -> None:
        with workspace_tempdir("ads_monitor_") as tmp_path:
            slab_path = tmp_path / "slab.vasp"
            slab_atoms = fcc111("Cu", size=(2, 2, 3), vacuum=12.0)
            slab_path.write_text(AseAtomsAdaptor.get_structure(slab_atoms).to(fmt="poscar"), encoding="utf-8")
            run_root = tmp_path / ".aether" / "runs" / "task_ads" / "run_monitor_fetch"
            run_record = RunRecord(
                task_id="task_ads",
                run_id="run_monitor_fetch",
                run_root=str(run_root),
                checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
            )
            plan = ExperimentPlan(
                task_id="task_ads",
                source_prompt="计算 H2O 在 Cu(111) 面上的吸附能",
                experiment_type="adsorption_energy",
                summary="Cu(111)+H2O adsorption",
                complexity=PlanComplexity.COMPLEX,
                readiness=ExecutionReadiness.NEEDS_CONFIRMATION,
                requires_confirmation=True,
                raw_plan={},
            )

            class Args:
                structure_path = str(slab_path)
                selected_candidate_id = "ontop_01_upright"
                material = "Cu slab"
                submit_profile = None
                submit = False
                remote = False

            candidate_result = maybe_generate_adsorption_candidates(args=Args(), plan=plan, run_record=run_record)
            selection_result = maybe_select_adsorption_candidate(
                args=Args(), plan=plan, run_record=run_record, adsorption_candidate_result=candidate_result
            )
            bundle_result = maybe_materialize_adsorption_workflow_bundle(
                args=Args(), plan=plan, run_record=run_record, selected_candidate_result=selection_result
            )
            assert bundle_result is not None
            summary_root = Path(bundle_result["summary_path"]).parent.parent
            store = RecordStore(tmp_path)

            for name in ("clean_slab", "isolated_adsorbate", "adsorbed_system"):
                subtask_root = Path(bundle_result["subtasks"][name]["bundle_root"])
                subtask_record = store.load_run_record(subtask_root)
                subtask_record.scheduler_job_id = f"job-{name}"
                subtask_record.overall_status = RunStatus.RUNNING
                subtask_record.notes.setdefault(
                    "remote",
                    {
                        "remote_run_root": f"/remote/{name}",
                        "mode": "winscp",
                    },
                )
                store.save_run_record(subtask_record)

            class FakeRemoteRunner:
                def monitor(self, run_record: RunRecord) -> SimpleNamespace:
                    return SimpleNamespace(
                        status="running",
                        message=f"{run_record.run_id} monitored",
                        details={"job_id": run_record.scheduler_job_id},
                    )

                def fetch_outputs(self, run_record: RunRecord) -> SimpleNamespace:
                    return SimpleNamespace(
                        status="synced",
                        message=f"{run_record.run_id} fetched",
                        details={"synced_files": [f"{run_record.run_root}/outputs/OUTCAR"]},
                    )

            with patch("dft_app.cli.main.get_remote_runner", return_value=FakeRemoteRunner()):
                result = execute_adsorption_workflow(
                    run_root=summary_root,
                    monitor=True,
                    fetch=True,
                    remote=True,
                )

            for name in ("clean_slab", "isolated_adsorbate", "adsorbed_system"):
                subtask_root = Path(bundle_result["subtasks"][name]["bundle_root"])
                self.assertEqual(result["subtasks"][name]["monitor"]["status"], "running")
                self.assertEqual(result["subtasks"][name]["fetch"]["status"], "synced")
                self.assertTrue((subtask_root / "metadata" / "remote_monitor_summary.json").exists())
                self.assertTrue((subtask_root / "metadata" / "remote_fetch_summary.json").exists())
            self.assertIn("monitor_pending", result["workflow_status"])
            self.assertEqual(
                set(result["workflow_status"]["fetch_ready"]),
                {"clean_slab", "isolated_adsorbate", "adsorbed_system"},
            )

    def test_execute_adsorption_workflow_defaults_to_status_summary(self) -> None:
        with workspace_tempdir("ads_default_") as tmp_path:
            slab_path = tmp_path / "slab.vasp"
            slab_atoms = fcc111("Cu", size=(2, 2, 3), vacuum=12.0)
            slab_path.write_text(AseAtomsAdaptor.get_structure(slab_atoms).to(fmt="poscar"), encoding="utf-8")
            run_root = tmp_path / ".aether" / "runs" / "task_ads" / "run_status_default"
            run_record = RunRecord(
                task_id="task_ads",
                run_id="run_status_default",
                run_root=str(run_root),
                checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
            )
            plan = ExperimentPlan(
                task_id="task_ads",
                source_prompt="计算 H2O 在 Cu(111) 面上的吸附能",
                experiment_type="adsorption_energy",
                summary="Cu(111)+H2O adsorption",
                complexity=PlanComplexity.COMPLEX,
                readiness=ExecutionReadiness.NEEDS_CONFIRMATION,
                requires_confirmation=True,
                raw_plan={},
            )

            class Args:
                structure_path = str(slab_path)
                selected_candidate_id = "ontop_01_upright"
                material = "Cu slab"
                submit_profile = None
                submit = False
                remote = False

            candidate_result = maybe_generate_adsorption_candidates(args=Args(), plan=plan, run_record=run_record)
            selection_result = maybe_select_adsorption_candidate(
                args=Args(), plan=plan, run_record=run_record, adsorption_candidate_result=candidate_result
            )
            bundle_result = maybe_materialize_adsorption_workflow_bundle(
                args=Args(), plan=plan, run_record=run_record, selected_candidate_result=selection_result
            )
            assert bundle_result is not None
            summary_root = Path(bundle_result["summary_path"]).parent.parent

            result = execute_adsorption_workflow(run_root=summary_root)
            self.assertTrue(result["actions"]["status"])
            self.assertEqual(result["workflow_status"]["status"], "prepared")
            self.assertEqual(
                set(result["workflow_status"]["submit_ready"]),
                {"clean_slab", "isolated_adsorbate", "adsorbed_system"},
            )

    def test_adsorption_workflow_aggregate_energy(self) -> None:
        with workspace_tempdir("ads_aggregate_") as tmp_path:
            slab_path = tmp_path / "slab.vasp"
            slab_atoms = fcc111("Cu", size=(2, 2, 3), vacuum=12.0)
            slab_path.write_text(AseAtomsAdaptor.get_structure(slab_atoms).to(fmt="poscar"), encoding="utf-8")
            run_root = tmp_path / ".aether" / "runs" / "task_ads" / "run_aggregate"
            run_record = RunRecord(
                task_id="task_ads",
                run_id="run_aggregate",
                run_root=str(run_root),
                checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
            )
            plan = ExperimentPlan(
                task_id="task_ads",
                source_prompt="计算 H2O 在 Cu(111) 面上的吸附能",
                experiment_type="adsorption_energy",
                summary="Cu(111)+H2O adsorption",
                complexity=PlanComplexity.COMPLEX,
                readiness=ExecutionReadiness.NEEDS_CONFIRMATION,
                requires_confirmation=True,
                raw_plan={},
            )

            class Args:
                structure_path = str(slab_path)
                selected_candidate_id = "ontop_01_upright"
                material = "Cu slab"
                submit_profile = None
                submit = False
                remote = False

            candidate_result = maybe_generate_adsorption_candidates(args=Args(), plan=plan, run_record=run_record)
            selection_result = maybe_select_adsorption_candidate(
                args=Args(), plan=plan, run_record=run_record, adsorption_candidate_result=candidate_result
            )
            bundle_result = maybe_materialize_adsorption_workflow_bundle(
                args=Args(), plan=plan, run_record=run_record, selected_candidate_result=selection_result
            )
            assert bundle_result is not None
            summary_root = Path(bundle_result["summary_path"]).parent.parent
            self.assertTrue(Path(bundle_result["summary_path"]).exists())
            energies = {
                "clean_slab": -10.0,
                "isolated_adsorbate": -2.0,
                "adsorbed_system": -13.5,
            }
            for name, energy in energies.items():
                subtask_root = Path(bundle_result["subtasks"][name]["bundle_root"])
                outputs_dir = subtask_root / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)
                poscar_source = Path(bundle_result["subtasks"][name]["inputs"]["poscar_path"])
                (outputs_dir / "CONTCAR").write_text(poscar_source.read_text(encoding="utf-8"), encoding="utf-8")
                (outputs_dir / "OUTCAR").write_text(
                    f"reached required accuracy\nfree  energy   TOTEN  =      {energy} eV\nGeneral timing and accounting informations for this job\n",
                    encoding="utf-8",
                )

            result = execute_adsorption_workflow(run_root=summary_root, parse_analyze=True)
            self.assertAlmostEqual(result["aggregate"]["adsorption_energy"], -1.5)
            self.assertTrue(Path(result["aggregate"]["summary_path"]).exists())
            self.assertTrue(Path(result["aggregate"]["report_path"]).exists())


if __name__ == "__main__":
    unittest.main()

