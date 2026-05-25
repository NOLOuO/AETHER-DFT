from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from dft_app.integrations.dft_tools_bridge import (
    build_dft_tools_kb_ingest_payload,
    build_dft_tools_manual_payload,
    run_dft_tools_explain_bridge,
)
from dft_app.storage import RecordStore
from tests.temp_workspace import workspace_tempdir


class DftToolsBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_cm = workspace_tempdir("bridge_")
        self.tmp_dir = self._tmp_cm.__enter__()
        self.addCleanup(self._tmp_cm.__exit__, None, None, None)
        self.project_root = self.tmp_dir / "semi_auto_dft"
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.run_root = self.project_root / ".aether" / "runs" / "task_demo" / "run_demo"
        (self.run_root / "metadata").mkdir(parents=True, exist_ok=True)
        self.store = RecordStore(self.project_root)

        self._write_json(
            self.run_root / "metadata" / "experiment_spec.json",
            {
                "task_id": "task_demo",
                "task_type": "relax",
                "material_name": "Si",
                "source_prompt": "计算 Si 的结构优化",
                "task_goal": "完成结构弛豫",
                "structure_source": "local_file",
                "structure_path": "F:/path/POSCAR",
                "workflow": ["relax"],
                "functional": "PBE",
                "kpoints_strategy": {"mode": "auto_density", "value": 40},
                "encut_strategy": {"mode": "auto", "value": None},
                "smearing": {"ismear": 0, "sigma": 0.05},
                "spin_settings": {"is_spin_polarized": False, "is_soc": False},
                "convergence_settings": {"ediff": 1e-6, "ediffg": -0.01, "nsw": 100},
                "submit_profile": None,
                "job_overrides": {"nodes": 1},
                "description": "计算 Si 的结构优化",
            },
        )
        self._write_json(
            self.run_root / "metadata" / "parsed_result.json",
            {
                "task_id": "task_demo",
                "run_id": "run_demo",
                "calc_type": "relax",
                "completed": True,
                "converged": True,
                "total_energy": -3.74,
                "energy_per_atom": -3.74,
                "band_gap": 0.0,
                "efermi": 8.0,
                "max_force": 0.0,
                "ionic_steps": 1,
                "electronic_steps": 10,
                "warnings": [],
                "raw_summary": {},
            },
        )
        self._write_json(
            self.run_root / "metadata" / "analysis_summary.json",
            {
                "status": "analyzed",
                "analysis_summary": {
                    "convergence_assessment": "good",
                    "recommended_actions": ["继续做 DOS"],
                },
            },
        )
        self._write_json(
            self.run_root / "metadata" / "run_record.json",
            {
                "task_id": "task_demo",
                "run_id": "run_demo",
            },
        )

    def test_build_manual_payload_uses_run_metadata(self) -> None:
        payload = build_dft_tools_manual_payload(self.store, self.run_root)
        self.assertEqual(payload["task_name"], "task_demo")
        self.assertEqual(payload["task_goal_text"], "完成结构弛豫")
        self.assertEqual(payload["structure_context"]["material_name"], "Si")
        self.assertEqual(payload["input_context"]["functional"], "PBE")
        self.assertEqual(payload["result_summary"]["total_energy"], -3.74)
        self.assertEqual(payload["result_summary"]["recommended_actions"], ["继续做 DOS"])

    def test_build_kb_ingest_payload_maps_explain_and_run_context(self) -> None:
        explain_request = build_dft_tools_manual_payload(self.store, self.run_root)
        knowledge_payload = {
            "task_name": "task_demo",
            "status_judgement": "状态判断：已完成。",
            "next_actions": ["继续做 DOS。"],
        }
        payload = build_dft_tools_kb_ingest_payload(explain_request, knowledge_payload)
        self.assertEqual(payload["task_name"], "task_demo")
        self.assertEqual(payload["task_type"], "relax")
        self.assertEqual(payload["source_tool"], "auto_dft")
        self.assertTrue(payload["completed"])
        self.assertTrue(payload["converged"])
        self.assertEqual(payload["status"], "success")
        self.assertIn("workflow:relax", payload["tags"])
        self.assertEqual(payload["recommended_actions"], ["继续做 DOS。"])
        self.assertEqual(payload["input_context"]["dft_tools_explain_status"], "状态判断：已完成。")

    @patch("dft_app.integrations.dft_tools_bridge.request_dft_tools_explain")
    @patch("dft_app.integrations.dft_tools_bridge.request_dft_tools_kb_ingest")
    def test_run_bridge_persists_explain_and_ingest_artifacts(self, mock_ingest, mock_request) -> None:
        mock_request.return_value = {
            "status_judgement": "状态判断：已完成。",
            "likely_causes": ["结构已收敛。"],
            "next_actions": ["可继续做 DOS。"],
            "evidence_used": [{"source": "result_summary", "detail": "total_energy=-3.74"}],
            "missing_evidence": [],
            "raw_llm_text": "ok",
            "provider": "deepseek",
            "model": "deepseek-reasoner",
            "knowledge_backflow_payload": {"task_name": "task_demo"},
        }
        mock_ingest.return_value = {"ok": True, "history_path": "kb.jsonl"}

        result = run_dft_tools_explain_bridge(self.store, self.run_root, base_url="http://127.0.0.1:8016")
        self.assertEqual(result["base_url"], "http://127.0.0.1:8016")
        self.assertTrue((self.run_root / "metadata" / "dft_tools_explain_result.json").exists())
        self.assertTrue((self.run_root / "metadata" / "dft_tools_knowledge_backflow_payload.json").exists())
        self.assertTrue((self.run_root / "metadata" / "dft_tools_kb_ingest_result.json").exists())
        self.assertTrue((self.run_root / "report" / "dft_tools_explain_summary.md").exists())
        self.assertEqual(result["kb_ingest_result"]["status"], "ingested")

    @patch("dft_app.integrations.dft_tools_bridge.request_dft_tools_explain")
    @patch("dft_app.integrations.dft_tools_bridge.request_dft_tools_kb_ingest")
    def test_run_bridge_can_skip_kb_ingest(self, mock_ingest, mock_request) -> None:
        mock_request.return_value = {
            "status_judgement": "状态判断：已完成。",
            "likely_causes": [],
            "next_actions": [],
            "evidence_used": [],
            "missing_evidence": [],
            "raw_llm_text": "ok",
            "provider": "deepseek",
            "model": "deepseek-reasoner",
            "knowledge_backflow_payload": {"task_name": "task_demo"},
        }
        result = run_dft_tools_explain_bridge(self.store, self.run_root, base_url="http://127.0.0.1:8016", ingest_kb=False)
        self.assertFalse(result["ingest_kb"])
        self.assertEqual(result["kb_ingest_result"]["status"], "skipped")
        mock_ingest.assert_not_called()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

