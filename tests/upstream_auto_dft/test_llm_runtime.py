from __future__ import annotations

import unittest
from pathlib import Path

from dft_app.llm import DomesticCopilotLLM


class LocalLlmRuntimeTests(unittest.TestCase):
    def test_builtin_runtime_uses_project_root_and_local_keys(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        runtime = DomesticCopilotLLM().describe_runtime()

        self.assertEqual(project_root, Path(runtime["app_root"]))
        self.assertTrue(runtime["local_key_file_exists"])
        self.assertIn("bailian", runtime["configured_providers"])
        self.assertIn("deepseek", runtime["configured_providers"])
        self.assertEqual("deepseek", runtime["default_model"]["provider"])
        self.assertEqual("deepseek-v4-pro", runtime["default_model"]["model"])


if __name__ == "__main__":
    unittest.main()
