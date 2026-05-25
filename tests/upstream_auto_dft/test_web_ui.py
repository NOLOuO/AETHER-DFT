from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from urllib.parse import quote
from unittest.mock import patch

from dft_app.models import RunRecord
from dft_app.storage import RecordStore
from dft_app.web import SemiAutoDFTWebApp
from tests.temp_workspace import workspace_tempdir


class WebUITests(unittest.TestCase):
    def _build_app(self) -> tuple[SemiAutoDFTWebApp, Path, Path]:
        cm = workspace_tempdir("web_ui_")
        tmp_dir = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        project_root = tmp_dir / "semi_auto_dft"
        project_root.mkdir(parents=True, exist_ok=True)
        store = RecordStore(project_root)
        run_root = project_root / ".aether" / "runs" / "task_demo" / "run_demo"
        record = RunRecord(
            task_id="task_demo",
            run_id="run_demo",
            run_root=str(run_root),
            checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
        )
        store.save_run_record(record)
        store.write_metadata(run_root, "experiment_spec.json", {"task_id": "task_demo"})
        app = SemiAutoDFTWebApp(project_root)
        return app, project_root, run_root

    def _call(self, app: SemiAutoDFTWebApp, path: str, method: str = "GET", body: str = "") -> tuple[str, str]:
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

    def test_home_page_renders(self) -> None:
        app, _project_root, _run_root = self._build_app()
        status, payload = self._call(app, "/")
        self.assertEqual(status, "200 OK")
        self.assertIn("semi_auto_dft", payload)
        self.assertIn("执行 dft run", payload)
        self.assertIn("adsorption 主线步骤概览", payload)

    def test_runs_page_and_detail_render(self) -> None:
        app, _project_root, run_root = self._build_app()
        status, runs_payload = self._call(app, "/runs")
        self.assertEqual(status, "200 OK")
        self.assertIn("run_demo", runs_payload)
        detail_status, detail_payload = self._call(app, f"/run-detail?run_root={run_root}")
        self.assertEqual(detail_status, "200 OK")
        self.assertIn("Run 概览", detail_payload)
        self.assertIn("task_demo", detail_payload)

    def test_runs_page_uses_url_encoded_run_root(self) -> None:
        app, _project_root, run_root = self._build_app()
        encoded_root = quote(str(run_root), safe="")
        status, payload = self._call(app, "/runs")
        self.assertEqual(status, "200 OK")
        self.assertIn(encoded_root, payload)

    def test_run_action_invokes_cli(self) -> None:
        app, _project_root, _run_root = self._build_app()
        with patch("dft_app.web.app.spawn_job_worker", return_value=12345):
            status, payload = self._call(
                app,
                "/actions/run",
                method="POST",
                body="prompt=%E6%B5%8B%E8%AF%95&structure_path=F%3A%5Cslab.vasp&dry_run=1",
            )
        self.assertEqual(status, "200 OK")
        self.assertIn("后台任务状态", payload)
        self.assertIn("-m dft_app.cli.main run", payload)

    def test_fetch_action_uses_run_root(self) -> None:
        app, _project_root, run_root = self._build_app()
        with patch("dft_app.web.app.spawn_job_worker", return_value=54321):
            status, payload = self._call(
                app,
                "/actions/fetch",
                method="POST",
                body=f"run_root={run_root}",
            )
        self.assertEqual(status, "200 OK")
        self.assertIn("后台任务状态", payload)
        self.assertIn("-m dft_app.cli.main fetch", payload)


if __name__ == "__main__":
    unittest.main()

