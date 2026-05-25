from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from aether_dft import cli
from aether_dft.context import build_context_payload, render_context_markdown
from aether_dft.harness import preflight, require_permission
from dft_shared.structure_analyzer.tool_registry import list_structure_tools


def test_context_payload_contains_openai_compatible_model():
    payload = build_context_payload()
    assert payload["model"]["protocol"] == "openai-compatible"
    assert payload["entrypoints"]["dft_mainline"]
    rendered = render_context_markdown(payload)
    assert "AETHER-DFT Context Snapshot" in rendered
    assert "OpenAI-compatible" in rendered or "openai-compatible" in rendered


def test_harness_preflight_has_required_surface():
    payload = preflight()
    assert payload["checks"]["config/system_prompt.md"] is True
    assert payload["checks"]["dft_app"] is True
    assert payload["checks"]["dft_shared"] is True


def test_permission_guard_blocks_destructive_operations():
    payload = require_permission("delete-test", destructive=True)
    assert payload["allowed"] is False


def test_structure_tool_registry_has_ten_project_relevant_tools():
    tools = list_structure_tools()
    assert len(tools) >= 10
    names = {item["name"] for item in tools}
    assert "xsd_to_poscar" in names
    assert "adsorption_candidates" in names
    assert "result_explain_bridge" in names


def test_cli_model_current_smoke(capsys):
    assert cli.main(["model", "current"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["model_id"] in {"deepseek:deepseek-v4-pro", "bailian:qwen3.7-max"}


def test_cli_structure_tools_smoke(capsys):
    assert cli.main(["structure", "tools"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert len(payload["structure_tools"]) >= 10


def test_cli_quick_start_names_program_model_and_version(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "AETHER-DFT v0.1.0" in out
    assert "model: deepseek:deepseek-v4-pro" in out
    assert "aether mainline" in out
    assert "aether chat" in out


def test_cli_no_args_enters_interactive_when_tty(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "Session Info" in out
    assert "Program: " in out


def test_cli_interactive_status_and_context_shortcuts(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    inputs = iter(["/status", "/context", "/help", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert '"program": "AETHER-DFT"' in out
    assert '"usable_context_tokens": 936000' in out
    assert "AETHER interactive chat" in out


def test_cli_mainline_prints_explicit_workflow(capsys):
    assert cli.main(["mainline"]) == 0
    out = capsys.readouterr().out
    assert "AETHER-DFT mainline" in out
    assert "discussion -> plan -> structure -> recommend" in out


def test_package_module_entry_prints_help():
    result = subprocess.run(
        [sys.executable, "-m", "aether_dft", "--help"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    assert "mainline" in output
    assert "chat" in output


def test_doctor_payload_names_program_model_and_version(capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    json_text = out[out.index("{") : out.rindex("}") + 1]
    payload = json.loads(json_text)
    assert payload["program"]["name"] == "AETHER-DFT"
    assert payload["program"]["command"] == "aether"
    assert payload["program"]["version"] == "0.1.0"
    assert payload["effective_model"]["model_id"] == "deepseek:deepseek-v4-pro"
    assert payload["effective_model"]["context_window"] == 1000000
