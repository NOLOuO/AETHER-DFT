from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aether_dft import paths
from aether_dft.prompt_sections import PromptSectionCompiler
from aether_dft.runtime_harness.core import AgentHarness
from aether_dft.runtime_harness.tool_registry import ToolRegistry


def test_m14_chemistry_compute_supports_enhanced_modes_and_legacy_operation():
    registry = ToolRegistry(permission_mode="dev")

    legacy = registry.run_tool(
        "chemistry_compute",
        {"operation": "boltzmann_population", "energies_ev": [0.0, 0.08], "temperature_k": 300},
    )["result"]
    assert legacy["status"] == "ok"
    assert legacy["populations"][0] > legacy["populations"][1]

    converted = registry.run_tool(
        "chemistry_compute",
        {"mode": "convert", "value": 96.48533212331, "from_unit": "kJ/mol", "to_unit": "eV"},
    )["result"]
    assert converted["status"] == "ok"
    assert abs(converted["result"] - 1.0) < 1e-9

    kbt = registry.run_tool("chemistry_compute", {"mode": "kBT", "temperature_k": 300, "unit": "eV"})["result"]
    assert kbt["status"] == "ok"
    assert 0.025 < kbt["result"] < 0.027

    rate = registry.run_tool(
        "chemistry_compute",
        {"mode": "tst_rate", "activation_energy": 0.5, "temperature_k": 300, "transmission_coefficient": 1.0},
    )["result"]
    assert rate["status"] == "ok"
    assert rate["result"] > 0
    assert "half_life_s" in rate

    gibbs = registry.run_tool(
        "chemistry_compute",
        {"mode": "gibbs", "enthalpy": 0.2, "entropy": 1e-4, "temperature_k": 300},
    )["result"]
    assert gibbs["status"] == "ok"
    assert abs(gibbs["result"] - 0.17) < 1e-12


def test_m14_discussion_snapshot_writes_markdown_json_without_forcing_workflow(tmp_path: Path, monkeypatch):
    import aether_dft.discussion_snapshot as snapshot_module

    def fake_runtime_dir(name: str) -> Path:
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(snapshot_module, "ensure_runtime_dir", fake_runtime_dir)
    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path)
    result = ToolRegistry(permission_mode="dev").run_tool(
        "discussion_state_snapshot",
        {
            "title": "Pt water adsorption discussion",
            "summary": "We agreed to compare atop and bridge candidates before cluster submission.",
            "consensus": ["Start from project research templates, not generic INCAR defaults."],
            "open_questions": ["Which coverage is closest to the target paper?"],
            "next_steps": ["Build 1-3 reasoned candidates, then preflight VASP inputs."],
            "tags": ["M14", "discussion"],
            "persist_path": str(tmp_path / "discussion_snapshots" / "explicit-snapshot-copy.json"),
        },
    )["result"]

    assert result["status"] == "ok"
    md_path = Path(result["snapshot_path"])
    json_path = Path(result["snapshot_json"])
    assert md_path.exists()
    assert json_path.exists()
    assert Path(result["persisted_path"]).exists()
    assert "Pt water adsorption discussion" in md_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"].startswith("We agreed")


def test_m14_discussion_snapshot_rejects_out_of_scope_persist_path(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    result = ToolRegistry(permission_mode="dev").run_tool(
        "discussion_state_snapshot",
        {
            "title": "unsafe path",
            "summary": "persist_path should be scoped.",
            "persist_path": str(tmp_path / "outside.json"),
        },
    )["result"]

    assert result["status"] == "warning"
    assert "persist_path 被拒绝" in result["message"]
    assert not (tmp_path / "outside.json").exists()


def test_m15_result_interpret_no_outputs_is_honest_and_actionable(tmp_path: Path):
    result = ToolRegistry().run_tool("result_interpret", {"run_root": str(tmp_path)})["result"]

    assert result["status"] == "no_outputs"
    assert "还没有可解释的 VASP 输出" in result["interpretation"]
    assert {"cluster_job_status_brief", "cluster_job_tail_log", "cluster_remote_fetch"}.issubset(result["suggestions"])


def test_m16_prompt_sections_keep_agent_flexible_and_state_aware():
    runtime_data = {
        "created_at": "2026-05-26T00:00:00+08:00",
        "workspace": "F:/AETHER-DFT",
        "project": "M17-demo",
        "model_id": "fake:model",
        "permission_policy": "dev",
        "cluster_runtime_digest": "- active job 123 under this project",
        "research_workspace_digest": "- research/M17-demo is in sync with ~/research/M17-demo",
        "relevant_priors_digest": "- Pt water prior: compare atop/bridge first",
        "tool_discovery_digest": "- `chemistry_compute` — calculator",
        "response_contract": "不要固定流程；由模型按证据选择工具。",
    }
    compiled = PromptSectionCompiler().build(runtime_data)
    included = {layer["name"] for layer in compiled["layers"] if layer["included"]}

    assert {"general_agent_voice", "research_workspace_habit", "cluster_runtime_digest", "research_workspace_digest", "relevant_priors_digest"}.issubset(included)
    prompt = compiled["prompt"]
    assert "工具是独立原语。你可以自由组合、跳过、回退" in prompt
    assert "research_workspace_diff" in prompt
    assert "active job 123" in prompt
    assert "Pt water prior" in prompt


@dataclass
class _MemorySessionStore:
    root: Path

    def ensure_session(self, session_id=None, project=None, first_prompt=None):
        return session_id or "m17-session"

    def build_session_context(self, session_id):
        return "Previous turn agreed: no fixed pipeline; choose tools based on evidence."

    def append_turn(self, session_id, record):
        path = self.root / f"{session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class _M17ScriptedAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:m17-e2e"})()

    def __init__(self, run_root: Path):
        self.run_root = run_root
        self.calls = 0

    @staticmethod
    def _call(name: str, arguments: dict[str, Any], idx: int) -> dict[str, Any]:
        return {
            "id": f"call_{idx}_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            tool_names = {tool["function"]["name"] for tool in tools}
            assert {"web_search", "literature_search", "chemistry_compute", "discussion_state_snapshot"}.issubset(tool_names)
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    self._call("web_search", {"query": "Pt(111) water adsorption DFT", "max_results": 2}, 1),
                    self._call("literature_search", {"query": "water adsorption Pt(111) DFT", "max_results": 2}, 2),
                    self._call("chemistry_compute", {"mode": "kBT", "temperature_k": 300}, 3),
                ],
            }
        if self.calls == 2:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    self._call(
                        "discussion_state_snapshot",
                        {
                            "title": "M17 guided e2e checkpoint",
                            "summary": "Discussion evidence collected; now choose modeling/execution tools by intent.",
                            "consensus": ["Do not claim external facts without connector evidence."],
                            "next_steps": ["Plan Step 2 modeling", "Plan Step 3 cluster execution"],
                        },
                        4,
                    )
                ],
            }
        if self.calls == 3:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    self._call(
                        "structure_modeling_intent_plan",
                        {
                            "intent": "Build 1-3 H2O adsorption candidates on Pt(111), guided by chemistry priors.",
                            "available_inputs": {"material": "Pt", "adsorbate": "H2O", "miller_index": [1, 1, 1], "output_dir": "candidates"},
                            "allow_writes": False,
                        },
                        5,
                    )
                ],
            }
        if self.calls == 4:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    self._call(
                        "cluster_execution_intent_plan",
                        {
                            "intent": "Put the selected Step 2 structure on the cluster using research templates, preflight, submit only after checks.",
                            "available_inputs": {"structure_path": "candidate.POSCAR", "project": "M17-demo", "task_type": "relax"},
                            "project": "M17-demo",
                            "allow_submit": False,
                        },
                        6,
                    )
                ],
            }
        if self.calls == 5:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    self._call("result_interpret", {"run_root": str(self.run_root)}, 7),
                    self._call("next_experiment_propose", {"project": "M17-demo", "recent_results": []}, 8),
                ],
            }
        if self.calls == 6:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    self._call(
                        "behavior_audit",
                        {
                            "goal": "M17 guided conversation",
                            "proposed_actions": ["answer user", "research_learning_capture if reusable"],
                            "tool_results": [{"status": "ok"}, {"status": "no_outputs"}],
                            "proposed_reply": "已完成一轮证据驱动的讨论、建模计划、集群计划和结果解释；没有输出时明确说明无证据。",
                        },
                        9,
                    )
                ],
            }
        return {
            "content": "M17 E2E 已按证据走完：讨论工具、建模导航、集群导航、结果解释、行为审计均被调用；没有把它写成固定程序。",
            "finish_reason": "stop",
            "tool_calls": [],
        }


def test_m17_scripted_agent_harness_exercises_six_step_research_conversation(tmp_path: Path, monkeypatch):
    import aether_dft.discussion_snapshot as snapshot_module

    def fake_runtime_dir(name: str) -> Path:
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(snapshot_module, "ensure_runtime_dir", fake_runtime_dir)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "OSZICAR").write_text("1 F= -10.0\n2 F= -10.2\n3 F= -10.25\n", encoding="utf-8")

    adapter = _M17ScriptedAdapter(run_root)
    harness = AgentHarness(
        adapter=adapter,
        registry=ToolRegistry(permission_mode="dev"),
        sessions=_MemorySessionStore(tmp_path / "sessions"),
    )
    record = harness.run_turn(
        "继续这个 H2O/Pt(111) 课题：先讨论证据，再决定建模与集群下一步，不要固定流程。",
        project="M17-demo",
        max_steps=10,
    )

    names = [item["name"] for item in record["tool_executions"]]
    assert names == [
        "web_search",
        "literature_search",
        "chemistry_compute",
        "discussion_state_snapshot",
        "structure_modeling_intent_plan",
        "cluster_execution_intent_plan",
        "result_interpret",
        "next_experiment_propose",
        "behavior_audit",
    ]
    assert record["finish_reason"] == "stop"
    assert "没有把它写成固定程序" in record["response"]
    assert record["tool_executions"][3]["result"]["status"] == "ok"
    assert Path(record["tool_executions"][3]["result"]["snapshot_path"]).exists()
    assert record["tool_executions"][6]["result"]["verdict"] == "running_or_partial"
    assert any(item["result"]["status"] == "ok" for item in record["tool_executions"] if item["name"] == "behavior_audit")


class _AuditStopAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:audit-stop"})()

    def __init__(self):
        self.calls = 0
        self.second_call_tools: list[dict[str, Any]] | None = None
        self.second_call_tool_choice: str | dict[str, Any] | None = None

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "audit",
                        "type": "function",
                        "function": {
                            "name": "behavior_audit",
                            "arguments": json.dumps(
                                {
                                    "goal": "stop after audit",
                                    "proposed_actions": ["reply"],
                                    "tool_results": [{"status": "ok"}],
                                    "proposed_reply": "有证据，准备回复。",
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
        self.second_call_tools = tools
        self.second_call_tool_choice = tool_choice
        return {"content": "审计完成后已停止工具调用，只给结论。", "finish_reason": "stop", "tool_calls": []}


def test_m17_harness_forces_natural_reply_after_behavior_audit(tmp_path: Path):
    adapter = _AuditStopAdapter()
    harness = AgentHarness(
        adapter=adapter,
        registry=ToolRegistry(permission_mode="dev"),
        sessions=_MemorySessionStore(tmp_path / "sessions"),
    )

    record = harness.run_turn("做完审计后不要继续调工具", project="audit-stop-demo", max_steps=4)

    assert record["finish_reason"] == "stop"
    assert [item["name"] for item in record["tool_executions"]] == ["behavior_audit"]
    assert adapter.second_call_tools == []
    assert adapter.second_call_tool_choice == "none"
    assert "停止工具调用" in record["response"]


class _TooManyToolsAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:too-many-tools"})()

    def __init__(self):
        self.calls = 0

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": f"tool_{idx}",
                        "type": "function",
                        "function": {
                            "name": "chemistry_compute",
                            "arguments": json.dumps({"mode": "kBT", "temperature_k": 300 + idx}),
                        },
                    }
                    for idx in range(10)
                ],
            }
        return {"content": "已按上限执行工具。", "finish_reason": "stop", "tool_calls": []}


def test_harness_caps_tool_calls_per_model_response(tmp_path: Path):
    harness = AgentHarness(
        adapter=_TooManyToolsAdapter(),
        registry=ToolRegistry(permission_mode="dev"),
        sessions=_MemorySessionStore(tmp_path / "sessions"),
    )

    record = harness.run_turn("测试单轮工具调用上限", project="cap-demo", max_steps=2)

    blocked = [item for item in record["tool_executions"] if item["result"].get("status") == "blocked"]
    assert len(record["tool_executions"]) == 10
    assert len(blocked) == 2
    assert "单轮工具调用超过上限" in blocked[0]["result"]["message"]
