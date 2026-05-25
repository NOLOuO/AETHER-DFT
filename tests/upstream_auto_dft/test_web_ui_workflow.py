from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from dft_app.web import SemiAutoDFTWebApp

from tests.web_ui_test_support import create_adsorption_workflow_fixture
from tests.web_ui_test_support import create_minimal_run_fixture
from tests.temp_workspace import workspace_tempdir


class WebUIWorkflowTests(unittest.TestCase):
    def _build_workflow_app(self) -> tuple[SemiAutoDFTWebApp, Path]:
        cm = workspace_tempdir("web_workflow_")
        tmp_dir = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        project_root = tmp_dir / "semi_auto_dft"
        project_root.mkdir(parents=True, exist_ok=True)
        run_root = create_adsorption_workflow_fixture(project_root=project_root)
        return SemiAutoDFTWebApp(project_root), run_root

    def _call(
        self,
        app: SemiAutoDFTWebApp,
        path: str,
        *,
        method: str = "GET",
        body: str = "",
    ) -> tuple[str, str]:
        captured: dict[str, str] = {}

        def start_response(status: str, _headers: list[tuple[str, str]]) -> None:
            captured["status"] = status

        environ = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path.split("?", 1)[0],
            "QUERY_STRING": path.split("?", 1)[1] if "?" in path else "",
            "CONTENT_LENGTH": str(len(body.encode("utf-8"))),
            "wsgi.input": io.BytesIO(body.encode("utf-8")),
        }
        payload = b"".join(app(environ, start_response)).decode("utf-8")
        return captured["status"], payload

    def test_run_detail_renders_adsorption_workflow_section(self) -> None:
        app, run_root = self._build_workflow_app()

        status, payload = self._call(app, f"/run-detail?run_root={run_root}")

        self.assertEqual(status, "200 OK")
        self.assertIn("Adsorption Workflow", payload)
        self.assertIn("workflow 状态", payload)
        self.assertIn("提交尚未启动的子任务", payload)
        self.assertIn("workflow 提交", payload)
        self.assertIn("Candidate 阶段", payload)
        self.assertIn("当前 selected candidate", payload)
        self.assertIn("候选已选定，可继续进入 workflow bundle / submit 阶段。", payload)

    def test_workflow_status_action_invokes_cli(self) -> None:
        app, run_root = self._build_workflow_app()

        with patch("dft_app.web.app.spawn_job_worker", return_value=11111):
            status, payload = self._call(
                app,
                "/actions/workflow",
                method="POST",
                body=f"run_root={run_root}&action=status",
            )

        self.assertEqual(status, "200 OK")
        self.assertIn("后台任务状态", payload)
        self.assertIn("-m dft_app.cli.main adsorption-workflow", payload)
        self.assertIn("--status", payload)

    def test_workflow_parse_analyze_action_invokes_cli(self) -> None:
        app, run_root = self._build_workflow_app()

        with patch("dft_app.web.app.spawn_job_worker", return_value=22222):
            status, _payload = self._call(
                app,
                "/actions/workflow",
                method="POST",
                body=f"run_root={run_root}&action=parse-analyze",
            )

        self.assertEqual(status, "200 OK")
        self.assertIn("-m dft_app.cli.main adsorption-workflow", _payload)
        self.assertIn("--parse-analyze", _payload)

    def test_dft_tools_explain_action_invokes_cli(self) -> None:
        app, run_root = self._build_workflow_app()

        with patch("dft_app.web.app.spawn_job_worker", return_value=33333):
            status, payload = self._call(
                app,
                "/actions/dft-tools-explain",
                method="POST",
                body=f"run_root={run_root}",
            )

        self.assertEqual(status, "200 OK")
        self.assertIn("后台任务状态", payload)
        self.assertIn("-m dft_app.cli.main dft-tools-explain", payload)

    def test_run_detail_renders_dft_tools_explain_section_when_metadata_exists(self) -> None:
        with workspace_tempdir("web_explain_") as tmp_dir:
            project_root = tmp_dir / "semi_auto_dft"
            project_root.mkdir(parents=True, exist_ok=True)
            run_root = create_minimal_run_fixture(project_root=project_root)
            explain_payload = {
                "status_judgement": "状态判断：已完成。",
                "likely_causes": ["结构已收敛。"],
                "next_actions": ["继续做 DOS。"],
                "provider": "deepseek",
                "model": "deepseek-reasoner",
            }
            kb_ingest_payload = {
                "enabled": True,
                "status": "ingested",
                "response": {"ok": True},
                "error": None,
            }
            metadata_dir = run_root / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (metadata_dir / "dft_tools_explain_result.json").write_text(
                json.dumps(explain_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (metadata_dir / "dft_tools_kb_ingest_result.json").write_text(
                json.dumps(kb_ingest_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            app = SemiAutoDFTWebApp(project_root)

            status, payload = self._call(app, f"/run-detail?run_root={run_root}")

            self.assertEqual(status, "200 OK")
            self.assertIn("dft_tools 结果解释", payload)
            self.assertIn("dft_tools 知识库回流", payload)
            self.assertIn("状态判断：已完成。", payload)
            self.assertIn("继续做 DOS。", payload)
            self.assertIn("结果入口", payload)


if __name__ == "__main__":
    unittest.main()
