from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.build import fcc111
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

from aether_dft import paths, project_state
from aether_dft.prompt_engine import load_base_system_prompt
import aether_dft.runtime_harness.core as harness_core
from aether_dft.runtime_harness.core import AgentHarness, infer_turn_mode
from aether_dft.runtime_harness.session import HarnessSessionStore
from aether_dft.runtime_harness.tool_registry import ToolRegistry


class FakeToolCallingAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:qwen3.7-max"})()

    def __init__(self):
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls.append(messages)
        if len(self.calls) == 1:
            assert any(tool["function"]["name"] == "project_state_read" for tool in tools)
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_state",
                        "type": "function",
                        "function": {
                            "name": "project_state_read",
                            "arguments": "{\"project\":\"chem-demo\"}",
                        },
                    }
                ],
            }
        assert any(message.get("role") == "tool" and message.get("name") == "project_state_read" for message in messages)
        return {"content": "已读取项目状态，可以继续推进。", "finish_reason": "stop", "tool_calls": []}


class TextToolCallingAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:deepseek-dsml"})()

    def __init__(self):
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return {
                "content": (
                    "我需要读取项目状态。\n"
                    "<｜｜DSML｜｜tool_calls>\n"
                    "<｜｜DSML｜｜invoke name=\"project_state_read\">\n"
                    "<｜｜DSML｜｜parameter name=\"project\" string=\"true\">chem-demo</｜｜DSML｜｜parameter>\n"
                    "</｜｜DSML｜｜invoke>\n"
                ),
                "finish_reason": "stop",
                "tool_calls": [],
            }
        assert any(message.get("role") == "tool" and message.get("name") == "project_state_read" for message in messages)
        return {"content": "文本工具调用已执行。", "finish_reason": "stop", "tool_calls": []}


class TextToolExampleAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:deepseek-dsml"})()

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        return {
            "content": (
                "示例格式如下，不要执行它："
                "<invoke name=\"project_state_read\">"
                "<parameter name=\"project\">chem-demo</parameter>"
                "</invoke>"
            ),
            "finish_reason": "stop",
            "tool_calls": [],
        }


class InterruptingAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:interrupt"})()

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        raise KeyboardInterrupt()


class StreamAwareAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:stream"})()

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None, stream_callback=None):
        if stream_callback:
            stream_callback({"type": "content_delta", "delta": "流"})
            stream_callback({"type": "content_delta", "delta": "式"})
        return {"content": "流式", "finish_reason": "stop", "tool_calls": []}


class EmptyRegistry:
    def openai_tool_schemas(self, interaction_mode=None):
        return []


def test_root_prompt_file_is_primary_system_prompt():
    prompt_path = paths.PROJECT_ROOT / "aether_dft" / "prompt_assets" / "system_chemistry.md"
    assert prompt_path.exists()
    prompt = load_base_system_prompt()
    assert "AETHER-DFT" in prompt
    assert "agent harness" in prompt


def test_root_tool_registry_discovers_domain_tools():
    registry = ToolRegistry()
    tools = registry.list_tools()
    names = {item["name"] for item in tools}
    assert "computational_chemistry_workflow_map" in names
    assert "structure_modeling_tool_status" in names
    assert "structure_modeling_intent_plan" in names
    assert "cluster_execution_intent_plan" in names
    assert "research_vasp_template_resolve" in names
    assert "vasp_input_preflight_check" in names
    assert "cluster_research_status" not in names
    assert "cluster_research_sync" not in names
    assert "research_workspace_diff" in names
    assert "research_workspace_sync_to_cluster" in names
    assert "research_workspace_sync_from_cluster" in names
    assert "research_onboarding_context" in names
    assert "research_proposal_plan" in names
    assert "research_progress_append" in names
    assert "project_state_read" in names
    assert "project_progress_append" in names
    assert "knowledge_note_add" in names
    assert "knowledge_note_list" in names
    assert "knowledge_note_search" in names
    assert "knowledge_note_show" in names
    assert "architecture_live_doc_snapshot" in names
    assert "architecture_live_doc_update" in names
    assert "structure_convert" in names
    assert "structure_resolve" in names
    assert "structure_supercell" in names
    assert "structure_build_slab" in names
    assert "structure_add_adsorbate" in names
    assert "structure_defect" in names
    assert "structure_add_vacancy" in names
    assert "structure_add_dopant" in names
    assert "structure_sanity_check" in names
    assert "structure_bond_analyze" in names
    assert "structure_displacement_compare" in names
    assert "adsorption_plan" in names
    assert "adsorption_build_slab" in names
    assert "adsorption_candidates" in names
    assert "adsorption_full_workflow" in names
    assert "transition_state_plan" in names
    assert "transition_state_dry_run" not in names
    assert "ts_workflow_config" in names
    assert "neb_input_check" in names
    assert "dimer_input_check" in names
    assert "task_type_catalog" in names
    assert "dft_run_step" in names
    assert "dft_run_task" in names
    assert "dft_run_report" in names
    assert "dft_run_list" in names
    assert "vasp_output_scan" in names
    assert "vasp_input_summary" in names
    assert "dft_task_plan" not in names
    assert "cluster_probe" in names
    assert "cluster_remote_submit" in names
    assert "cluster_remote_monitor" in names
    assert "cluster_remote_fetch" in names
    assert "adsorption_workflow_status" in names
    by_name = {item["name"]: item for item in tools}
    for side_effect_tool in {
        "structure_convert",
        "structure_supercell",
        "structure_build_slab",
        "structure_add_adsorbate",
        "structure_defect",
        "structure_add_vacancy",
        "structure_add_dopant",
        "adsorption_build_slab",
        "adsorption_candidates",
        "adsorption_full_workflow",
        "research_progress_append",
        "knowledge_note_add",
        "knowledge_note_search",
        "knowledge_note_show",
        "dft_run_task",
        "cluster_remote_submit",
        "cluster_remote_monitor",
        "cluster_remote_fetch",
    }:
        if side_effect_tool in {"knowledge_note_search", "knowledge_note_show"}:
            assert by_name[side_effect_tool]["read_only"] is True
        else:
            assert by_name[side_effect_tool]["read_only"] is False


def test_agent_harness_executes_tool_loop_and_persists_session(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "projects"
    knowledge_dir = tmp_path / "knowledge_base"
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", knowledge_dir)
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", knowledge_dir)

    project_state.init_project("chem-demo", description="demo", overwrite=True)
    sessions = HarnessSessionStore(tmp_path / "sessions")
    harness = AgentHarness(adapter=FakeToolCallingAdapter(), registry=ToolRegistry(), sessions=sessions)

    record = harness.run_turn("继续推进这个课题", project="chem-demo", max_steps=3)

    assert record["response"] == "已读取项目状态，可以继续推进。"
    assert record["tool_executions"][0]["name"] == "project_state_read"
    assert Path(record["record_path"]).exists()
    resumed = sessions.store.resume_payload(session_id=record["session_id"])
    assert resumed["status"] == "ok"
    assert resumed["state"]["turn_count"] == 1


def test_agent_harness_executes_dsml_text_tool_calls(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "projects"
    knowledge_dir = tmp_path / "knowledge_base"
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", knowledge_dir)
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", knowledge_dir)

    project_state.init_project("chem-demo", description="demo", overwrite=True)
    sessions = HarnessSessionStore(tmp_path / "sessions")
    harness = AgentHarness(adapter=TextToolCallingAdapter(), registry=ToolRegistry(), sessions=sessions)

    record = harness.run_turn("继续推进这个课题", project="chem-demo", max_steps=3)

    assert record["response"] == "文本工具调用已执行。"
    assert record["finish_reason"] == "stop"
    assert [item["name"] for item in record["tool_executions"]] == ["project_state_read"]


def test_agent_harness_does_not_execute_dsml_examples_without_tool_calls_marker(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "projects"
    knowledge_dir = tmp_path / "knowledge_base"
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", knowledge_dir)
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", knowledge_dir)

    project_state.init_project("chem-demo", description="demo", overwrite=True)
    sessions = HarnessSessionStore(tmp_path / "sessions")
    harness = AgentHarness(adapter=TextToolExampleAdapter(), registry=ToolRegistry(), sessions=sessions)

    record = harness.run_turn("解释一下工具调用格式", project="chem-demo", max_steps=2)

    assert record["tool_executions"] == []
    assert record["finish_reason"] in {"stop", "tool_markup_finalized"}


def test_agent_harness_saves_partial_trace_on_keyboard_interrupt(tmp_path: Path):
    sessions = HarnessSessionStore(tmp_path / "sessions")
    events: list[dict[str, Any]] = []
    harness = AgentHarness(adapter=InterruptingAdapter(), registry=ToolRegistry(), sessions=sessions)

    record = harness.run_turn("用户中途按 Ctrl+C", max_steps=3, progress_callback=events.append)

    assert record["finish_reason"] == "user_interrupted"
    assert "partial trace" in record["response"]
    assert Path(record["record_path"]).exists()
    assert any(event.get("event") == "turn_interrupted" for event in events)


def test_agent_harness_forwards_stream_callback_when_no_tools(tmp_path: Path):
    sessions = HarnessSessionStore(tmp_path / "sessions")
    events: list[dict[str, Any]] = []
    harness = AgentHarness(adapter=StreamAwareAdapter(), registry=EmptyRegistry(), sessions=sessions)

    record = harness.run_turn("直接回答", max_steps=1, stream_callback=events.append)

    assert record["response"] == "流式"
    assert [event["delta"] for event in events] == ["流", "式"]


class OneToolRegistry:
    def openai_tool_schemas(self, interaction_mode=None):
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_context",
                    "description": "read context",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]


def test_agent_harness_streams_even_when_tools_are_available(tmp_path: Path):
    sessions = HarnessSessionStore(tmp_path / "sessions")
    events: list[dict[str, Any]] = []
    harness = AgentHarness(adapter=StreamAwareAdapter(), registry=OneToolRegistry(), sessions=sessions)

    record = harness.run_turn("直接回答但工具可用", max_steps=1, stream_callback=events.append)

    assert record["response"] == "流式"
    assert [event["delta"] for event in events] == ["流", "式"]


def test_knowledge_note_tools_round_trip(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "projects"
    knowledge_dir = tmp_path / "knowledge_base"
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", knowledge_dir)
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", knowledge_dir)

    project_state.init_project("chem-demo", description="demo", overwrite=True)
    registry = ToolRegistry()

    added = registry.run_tool(
        "knowledge_note_add",
        {
            "project": "chem-demo",
            "title": "adsorption site heuristic",
            "content": "Prefer top-layer bridge-like sites before hollow guesses.",
            "tags": ["adsorption", "heuristic"],
        },
    )
    assert added["result"]["status"] == "ok"

    listed = registry.run_tool("knowledge_note_list", {"project": "chem-demo"})
    assert listed["result"]["status"] == "ok"
    assert listed["result"]["notes"]

    searched = registry.run_tool("knowledge_note_search", {"project": "chem-demo", "query": "bridge"})
    assert searched["result"]["status"] == "ok"
    assert searched["result"]["matches"]

    note_id = added["result"]["note"]["note_id"]
    shown = registry.run_tool("knowledge_note_show", {"project": "chem-demo", "note": note_id})
    assert shown["result"]["status"] == "ok"
    assert "adsorption site heuristic" in shown["result"]["note"]["content"]


def test_dft_run_task_tool_invokes_bridge(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_run_dft_task(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {"status": "ok", "exit_code": 0, "execution_mode": kwargs.get("execution_mode"), "task": {"task_id": "task_1"}}

    monkeypatch.setattr("aether_dft.runtime_harness.tool_registry.run_dft_task", fake_run_dft_task)
    registry = ToolRegistry()
    result = registry.run_tool(
        "dft_run_task",
        {
            "prompt": "继续吸附任务",
            "project": "chem-demo",
            "material": "Cu(111)",
            "structure_path": "POSCAR",
            "task_type": "relax",
            "execution_mode": "build",
        },
    )
    assert result["result"]["status"] == "ok"
    assert captured["prompt"] == "继续吸附任务"
    assert captured["kwargs"]["execution_mode"] == "build"


def test_dft_run_task_tool_preserves_step2_lineage_arguments(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_run_dft_task(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {"status": "ok", "exit_code": 0, "execution_mode": kwargs.get("execution_mode"), "task": {"task_id": "task_1"}}

    monkeypatch.setattr("aether_dft.runtime_harness.tool_registry.run_dft_task", fake_run_dft_task)
    result = ToolRegistry().run_tool(
        "dft_run_task",
        {
            "prompt": "把 Step2 候选放到集群前 build",
            "project": "chem-demo",
            "material": "Pt slab",
            "structure_path": "candidate.POSCAR",
            "task_type": "relax",
            "model_spec_path": "model_spec.json",
            "step2_manifest_path": "manifest.json",
            "candidate_id": "model_pick_01",
            "execution_mode": "build",
        },
    )
    assert result["result"]["status"] == "ok"
    assert captured["kwargs"]["model_spec_path"] == "model_spec.json"
    assert captured["kwargs"]["step2_manifest_path"] == "manifest.json"
    assert captured["kwargs"]["candidate_id"] == "model_pick_01"


def test_cluster_remote_tools_route_to_runner_and_store(tmp_path: Path, monkeypatch):
    class FakeRunRecord:
        def __init__(self):
            self.saved = False

    class FakeSpec:
        pass

    class FakeStore:
        def __init__(self, root):
            self.root = root
            self.saved = []

        def resolve_run_root(self, run_root=None, run_id=None):
            return Path(run_root or tmp_path / "run")

        def load_experiment_spec(self, run_root):
            return FakeSpec()

        def load_run_record(self, run_root):
            return FakeRunRecord()

        def save_run_record(self, run_record):
            run_record.saved = True
            self.saved.append(run_record)

    class FakeResult:
        def __init__(self, status, message):
            self.status = status
            self.message = message
            self.details = {"ok": True}

    class FakeRunner:
        def probe(self):
            return FakeResult("ok", "probe")

        def describe_config(self):
            return {"host": "fake"}

        def submit(self, spec, run_record):
            run_record.submitted = True
            return FakeResult("submitted", "submit")

        def monitor(self, run_record, sync_outputs=True):
            run_record.monitored = sync_outputs
            return FakeResult("completed", "monitor")

        def fetch_outputs(self, run_record):
            run_record.fetched = True
            return FakeResult("synced", "fetch")

    monkeypatch.setattr("dft_app.storage.RecordStore", FakeStore)
    monkeypatch.setattr("aether_dft.runtime_harness.tool_registry.SSHRemoteRunner", FakeRunner)
    registry = ToolRegistry(allow_cluster_submit=True)

    submit = registry.run_tool("cluster_remote_submit", {"run_root": str(tmp_path / "run")})
    assert submit["result"]["status"] == "submitted"

    monitor = registry.run_tool("cluster_remote_monitor", {"run_root": str(tmp_path / "run"), "sync_outputs": False})
    assert monitor["result"]["status"] == "completed"

    fetch = registry.run_tool("cluster_remote_fetch", {"run_root": str(tmp_path / "run")})
    assert fetch["result"]["status"] == "synced"


def test_research_workspace_tools_status_and_sync_dry_run(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeConfig:
        user = "szhang"
        remote_base_dir = "/home/szhang/aether-dft-runs"

    class FakeResult:
        def __init__(self, status, message, details):
            self.status = status
            self.message = message
            self.details = details

    class FakeRunner:
        def _load_config(self):
            return FakeConfig()

        def research_status(self, local_research_root, *, remote_research_dir=None):
            captured["status_root"] = local_research_root
            captured["status_remote"] = remote_research_dir
            return FakeResult("ok", "out_of_sync", {"sync_status": "out_of_sync", "missing_remote": ["AGENTS.md"]})

        def sync_research_to_remote(self, local_research_root, *, remote_research_dir=None, dry_run=True):
            captured["sync_root"] = local_research_root
            captured["sync_remote"] = remote_research_dir
            captured["dry_run"] = dry_run
            return FakeResult("planned" if dry_run else "synced", "sync", {"dry_run": dry_run})

    monkeypatch.setattr("aether_dft.research_sync.SSHRemoteRunner", FakeRunner)
    registry = ToolRegistry()

    status = registry.run_tool("research_workspace_diff", {"remote_research_dir": "/home/szhang/research"})
    assert status["result"]["status"] == "ok"
    assert status["result"]["details"]["missing_remote"] == ["AGENTS.md"]

    dry = registry.run_tool("research_workspace_sync_to_cluster", {})
    assert dry["result"]["status"] == "planned"
    assert captured["dry_run"] is True

    applied = registry.run_tool("research_workspace_sync_to_cluster", {"apply": True})
    assert applied["result"]["status"] == "synced"
    assert captured["dry_run"] is False


def test_turn_mode_uses_concrete_execution_evidence_not_user_phrases():
    assert infer_turn_mode("看看怎么样了") == "discussion"
    assert infer_turn_mode("这个体系下一步怎么做") == "discussion"
    assert infer_turn_mode("这个 OUTCAR 收敛了吗？") == "discussion"
    assert infer_turn_mode("squeue 里 job_id 12345 状态") == "discussion"
    assert infer_turn_mode("[execution-mode] 只做一次状态检查") == "execution"
    assert infer_turn_mode("[discussion-mode] 先讨论机理，不跑工具") == "discussion"


def test_discussion_mode_exposes_lean_tool_schema_surface():
    registry = ToolRegistry()
    all_tools = registry.openai_tool_schemas()
    discussion_tools = registry.openai_tool_schemas(interaction_mode="discussion")
    discussion_names = {item["function"]["name"] for item in discussion_tools}
    assert len(discussion_tools) <= int(len(all_tools) * 0.6)
    assert "project_continuity_digest" in discussion_names
    assert "literature_search" in discussion_names
    assert "research_workspace_diff" in discussion_names
    for read_only_tool in {
        "cluster_my_jobs",
        "cluster_job_status_brief",
        "cluster_job_tail_log",
        "cluster_job_partial_outcar",
        "cluster_job_progress_estimate",
        "slab_surface_inspect",
        "adsorbate_chemistry_hint",
        "structure_enumerate_sites",
        "manifest_audit",
        "candidate_quality_score",
    }:
        assert read_only_tool in discussion_names
    assert "cluster_remote_submit" not in discussion_names
    assert "structure_add_adsorbate" not in discussion_names


class HugeToolResultAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:qwen3.7-max"})()

    def __init__(self):
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_runs",
                        "type": "function",
                        "function": {"name": "huge_result", "arguments": "{}"},
                    }
                ],
            }
        tool_messages = [item for item in messages if item.get("role") == "tool"]
        assert tool_messages
        assert len(tool_messages[-1]["content"]) < 14000
        return {"content": "工具结果已压缩给模型，但完整结果仍在记录中。", "finish_reason": "stop", "tool_calls": []}


class HugeToolRegistry:
    def openai_tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "huge_result",
                    "description": "return an oversized result",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def list_tools(self):
        return [{"name": "huge_result", "parameters": {"type": "object", "properties": {}}}]

    def run_tool(self, name, arguments):
        return {"name": name, "arguments": arguments, "result": {"status": "ok", "payload": "x" * 50000}}


class SessionReplayAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:qwen3.7-max"})()

    def __init__(self):
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return {"content": "第一轮已完成。", "finish_reason": "stop", "tool_calls": []}
        system_prompt = messages[0]["content"]
        assert "Session Context" in system_prompt
        assert "第一轮提问" in system_prompt
        assert "第一轮已完成" in system_prompt
        return {"content": "已续接前文，可以继续推进。", "finish_reason": "stop", "tool_calls": []}


def test_agent_harness_truncates_model_visible_tool_result(tmp_path: Path):
    sessions = HarnessSessionStore(tmp_path / "sessions")
    harness = AgentHarness(adapter=HugeToolResultAdapter(), registry=HugeToolRegistry(), sessions=sessions)
    record = harness.run_turn("检查一个很大的工具结果", max_steps=3)

    assert record["response"] == "工具结果已压缩给模型，但完整结果仍在记录中。"
    assert record["tool_executions"][0]["name"] == "huge_result"
    assert record["tool_executions"][0]["result"]["microcompacted"] is True
    assert "payload" not in record["tool_executions"][0]["result"]
    compacted = record["tool_executions"][0]["result"]
    persisted = Path(record["tool_executions"][0]["persisted_output_path"])
    assert persisted.exists()
    assert compacted["tool_name"] == "huge_result"
    assert compacted["tool_call_id"] == "call_runs"
    assert compacted["persisted_output_sha256"]
    assert compacted["persisted_output_bytes"] > 50000
    persisted_payload = json.loads(persisted.read_text(encoding="utf-8"))
    assert len(persisted_payload["payload"]) == 50000


class ParallelReadOnlyAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:qwen3.7-max"})()

    def __init__(self):
        self.calls = 0

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "slow_read_a", "arguments": "{}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "slow_read_b", "arguments": "{}"}},
                ],
            }
        return {"content": "两个只读检查已并发完成。", "finish_reason": "stop", "tool_calls": []}


class ParallelReadOnlyRegistry:
    def __init__(self):
        self.barrier = threading.Barrier(2)

    def openai_tool_schemas(self):
        return [
            {"type": "function", "function": {"name": "slow_read_a", "description": "read A", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "slow_read_b", "description": "read B", "parameters": {"type": "object", "properties": {}}}},
        ]

    def list_tools(self):
        return [{"name": "slow_read_a", "read_only": True}, {"name": "slow_read_b", "read_only": True}]

    def is_read_only_tool(self, name):
        return True

    def run_tool(self, name, arguments):
        try:
            self.barrier.wait(timeout=1.0)
            barrier_passed = True
        except threading.BrokenBarrierError:
            barrier_passed = False
        time.sleep(0.05)
        return {"name": name, "arguments": {}, "result": {"status": "ok", "name": name, "barrier_passed": barrier_passed}}


def test_agent_harness_runs_multiple_read_only_tools_in_parallel(tmp_path: Path):
    sessions = HarnessSessionStore(tmp_path / "sessions")
    events: list[dict[str, Any]] = []
    registry = ParallelReadOnlyRegistry()
    harness = AgentHarness(adapter=ParallelReadOnlyAdapter(), registry=registry, sessions=sessions)

    record = harness.run_turn("并发读取两个只读证据", max_steps=3, progress_callback=events.append)

    assert record["response"] == "两个只读检查已并发完成。"
    assert [item["name"] for item in record["tool_executions"]] == ["slow_read_a", "slow_read_b"]
    assert any(event.get("event") == "tool_parallel_start" for event in events)
    assert [item["result"]["barrier_passed"] for item in record["tool_executions"]] == [True, True]


class HeartbeatAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:qwen3.7-max"})()

    def __init__(self):
        self.calls = 0

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [{"id": "call_slow", "type": "function", "function": {"name": "slow_read", "arguments": "{}"}}],
            }
        return {"content": "慢工具执行期间已给出心跳。", "finish_reason": "stop", "tool_calls": []}


class HeartbeatRegistry:
    def openai_tool_schemas(self):
        return [{"type": "function", "function": {"name": "slow_read", "description": "slow read", "parameters": {"type": "object", "properties": {}}}}]

    def list_tools(self):
        return [{"name": "slow_read", "read_only": True}]

    def run_tool(self, name, arguments):
        time.sleep(0.12)
        return {"name": name, "arguments": {}, "result": {"status": "ok"}}


def test_agent_harness_emits_tool_progress_heartbeat(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(harness_core, "TOOL_HEARTBEAT_SECONDS", 0.04)
    sessions = HarnessSessionStore(tmp_path / "sessions")
    events: list[dict[str, Any]] = []
    harness = AgentHarness(adapter=HeartbeatAdapter(), registry=HeartbeatRegistry(), sessions=sessions)

    record = harness.run_turn("读一个较慢的外部证据", max_steps=3, progress_callback=events.append)

    assert record["response"] == "慢工具执行期间已给出心跳。"
    assert any(event.get("event") == "tool_progress" and event.get("name") == "slow_read" for event in events)


def test_token_guard_marks_context_for_finalization(monkeypatch):
    monkeypatch.setenv("AETHER_DFT_CONTEXT_MAX_CHARS", "12000")
    messages = [{"role": "user", "content": "x" * 11500}]

    guard = harness_core._token_guard_status(messages, model_id="fake:qwen3.7-max")

    assert guard["should_finalize"] is True
    assert guard["usage_ratio"] >= 0.88


def test_agent_harness_replays_recent_session_context(tmp_path: Path):
    sessions = HarnessSessionStore(tmp_path / "sessions")
    harness = AgentHarness(adapter=SessionReplayAdapter(), sessions=sessions)

    first = harness.run_turn("第一轮提问", max_steps=2)
    assert first["response"] == "第一轮已完成。"

    second = harness.run_turn("第二轮继续追问", session_id=first["session_id"], max_steps=2)
    assert second["response"] == "已续接前文，可以继续推进。"


def test_agent_harness_auto_compacts_before_model_request(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AETHER_DFT_CONTEXT_MAX_CHARS", "90000")
    monkeypatch.setenv("AETHER_DFT_AUTO_COMPACT_RATIO", "0.95")
    sessions = HarnessSessionStore(tmp_path / "sessions")
    session_id = sessions.store.start_session(project="demo")
    for index in range(86):
        sessions.store.append_turn(
            session_id,
            {
                "project": "demo",
                "prompt": f"历史问题 {index} " + "p" * 600,
                "response": f"历史回答 {index} " + "r" * 600,
            },
        )

    monkeypatch.setenv("AETHER_DFT_CONTEXT_MAX_CHARS", "12000")
    monkeypatch.setenv("AETHER_DFT_AUTO_COMPACT_RATIO", "0.50")
    events: list[dict[str, Any]] = []
    harness = AgentHarness(adapter=SessionReplayAdapter(), sessions=sessions)

    record = harness.run_turn("继续这个长 session", session_id=session_id, max_steps=2, progress_callback=events.append)

    assert record["response"] in {"第一轮已完成。", "已续接前文，可以继续推进。"}
    assert any(event.get("event") == "session_auto_compacted" for event in events)
    state = sessions.store.load_state(session_id)
    assert state["last_compact_trigger"] == "automatic"
    assert state["last_compact_stats"]["context_budget"]["auto_compact_chars"] == 8000


class ApprovalRetryAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:qwen3.7-max"})()

    def __init__(self):
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_write",
                        "type": "function",
                        "function": {"name": "write_note", "arguments": "{\"message\":\"approval flow\"}"},
                    }
                ],
            }
        tool_messages = [item for item in messages if item.get("role") == "tool"]
        assert tool_messages
        assert any("approval flow" in str(item.get("content") or "") for item in tool_messages)
        return {"content": "已完成需要审批的写入。", "finish_reason": "stop", "tool_calls": []}


class ApprovalRetryRegistry:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def openai_tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_note",
                    "description": "write a note",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def list_tools(self):
        return [{"name": "write_note", "read_only": False}]

    def run_tool(self, name, arguments):
        if isinstance(arguments, str):
            payload = json.loads(arguments)
        else:
            payload = dict(arguments or {})
        self.calls.append(payload)
        if not payload.get("_permission_granted"):
            return {
                "name": name,
                "arguments": payload,
                "result": {
                    "status": "permission_required",
                    "message": "need approval",
                    "permission_mode": "ask",
                    "permission_label": "需要用户同意",
                    "reason": "ask mode requires user approval before non-read-only tool execution",
                },
            }
        return {"name": name, "arguments": payload, "result": {"status": "ok", "saved": payload["message"]}}


def test_agent_harness_prompts_for_permission_and_retries_approved_tool(tmp_path: Path):
    sessions = HarnessSessionStore(tmp_path / "sessions")
    harness = AgentHarness(adapter=ApprovalRetryAdapter(), registry=ApprovalRetryRegistry(), sessions=sessions)

    prompts: list[dict[str, Any]] = []

    record = harness.run_turn(
        "写一条需要审批的笔记",
        max_steps=3,
        permission_prompt_callback=lambda details: prompts.append(details) or True,
    )

    assert prompts and prompts[0]["tool_name"] == "write_note"
    assert record["tool_executions"][0]["result"]["status"] == "ok"
    assert record["response"] == "已完成需要审批的写入。"
    assert "_permission_granted" not in harness.registry.calls[0]
    assert harness.registry.calls[1]["_permission_granted"] is True


def test_workflow_map_exposes_capabilities_not_fixed_steps():
    result = ToolRegistry().run_tool("computational_chemistry_workflow_map", {})
    payload = result["result"]
    assert payload["status"] == "ok"
    assert "不是固定流程" in payload["principle"]
    categories = {item["category"]: set(item["tools"]) for item in payload["capability_stages"]}
    assert "mainline" not in payload
    assert "project_context" in categories
    assert "structure_modeling" in categories
    assert "dft_execution" in categories
    assert "realtime_cluster_status" in categories
    assert "research_onboarding_context" in categories["project_context"]
    assert "research_proposal_plan" in categories["project_context"]
    assert "structure_modeling_tool_status" in categories["structure_modeling"]
    assert "structure_resolve" in categories["structure_modeling"]
    assert "research_vasp_template_resolve" in categories["dft_execution"]
    assert "vasp_input_preflight_check" in categories["dft_execution"]
    assert "cluster_remote_submit" in categories["dft_execution"]
    assert "cluster_job_partial_outcar" in categories["realtime_cluster_status"]


def test_structure_modeling_tool_status_is_decision_matrix_not_fixed_pipeline():
    result = ToolRegistry().run_tool("structure_modeling_tool_status", {})
    payload = result["result"]
    assert payload["status"] == "ok"
    assert "固定流水线" in payload["principle"]
    intents = {item["intent"] for item in payload["decision_matrix"]}
    assert "吸附候选" in intents
    assert "缺陷/掺杂" in intents
    adsorption = next(item for item in payload["decision_matrix"] if item["intent"] == "吸附候选")
    assert "adsorption_candidate_plan" in adsorption["tools"]
    assert "候选数量、位点、取向由 plan.rationale 决定" in adsorption["not_a_fixed_program"]
    assert payload["completion"]["adsorption_model_authored_candidates"] == "ready"


def test_structure_modeling_intent_plan_guides_adsorption_without_fixed_pipeline(tmp_path: Path):
    result = ToolRegistry().run_tool(
        "structure_modeling_intent_plan",
        {
            "intent": "为 H2O 在 Pt(111) 上生成少量有科学理由的吸附候选",
            "task_type": "adsorption",
            "available_inputs": {
                "adsorbate": "H2O",
                "material": "Pt(111)",
                "output_dir": str(tmp_path / "candidates"),
            },
            "project": "chem-demo",
        },
    )
    payload = result["result"]
    assert payload["status"] == "ok"
    assert payload["task_type"] == "adsorption"
    assert "固定程序" in payload["principle"]
    assert "slab_path_or_material_source" not in payload["missing_inputs"]
    tool_names = {
        tool
        for group in payload["tool_groups"]
        for tool in group["candidate_tools"]
    }
    assert "adsorbate_chemistry_hint" in tool_names
    assert "adsorption_candidate_plan" in tool_names
    assert "structure_add_adsorbate" in tool_names
    assert "candidate_quality_score" in tool_names
    assert any("fallback_only" in gate for gate in payload["quality_gates"])


def test_agent_can_use_step2_tools_to_create_model_authored_adsorption_manifest(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "projects"
    knowledge_dir = tmp_path / "knowledge_base"
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", knowledge_dir)
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", knowledge_dir)

    slab_path = tmp_path / "POSCAR_Pt111"
    atoms = fcc111("Pt", size=(2, 2, 3), vacuum=8.0)
    AseAtomsAdaptor.get_structure(atoms).to(fmt="poscar", filename=str(slab_path))
    candidate_path = tmp_path / "model_pick_01.POSCAR"
    manifest_dir = tmp_path / "manifest"

    class Step2ModelingAdapter:
        runtime = type("Runtime", (), {"model_id": "fake:step2-model"})()

        def __init__(self):
            self.calls = 0
            self.site_id = "ontop_default"
            self.site_coords = None
            self.plan_id = None

        @staticmethod
        def _tool_payload(messages: list[dict[str, Any]], name: str) -> dict[str, Any]:
            for message in reversed(messages):
                if message.get("role") == "tool" and message.get("name") == name:
                    return json.loads(str(message.get("content") or "{}"))
            return {}

        def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
            self.calls += 1
            if self.calls == 1:
                assert any(tool["function"]["name"] == "structure_modeling_intent_plan" for tool in tools)
                return {
                    "content": "",
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "call_intent",
                            "type": "function",
                            "function": {
                                "name": "structure_modeling_intent_plan",
                                "arguments": json.dumps(
                                    {
                                        "intent": "为 H2O/Pt(111) 生成一个模型自主选择的吸附候选并检查质量",
                                        "available_inputs": {
                                            "slab_path": str(slab_path),
                                            "adsorbate": "H2O",
                                            "material": "Pt(111)",
                                            "output_dir": str(manifest_dir),
                                        },
                                        "project": "step2-demo",
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            if self.calls == 2:
                return {
                    "content": "",
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "call_hint",
                            "type": "function",
                            "function": {"name": "adsorbate_chemistry_hint", "arguments": '{"adsorbate":"H2O"}'},
                        },
                        {
                            "id": "call_prior",
                            "type": "function",
                            "function": {
                                "name": "knowledge_search_for_system",
                                "arguments": '{"material":"Pt(111)","adsorbate":"H2O","max_results":3}',
                            },
                        },
                        {
                            "id": "call_surface",
                            "type": "function",
                            "function": {"name": "slab_surface_inspect", "arguments": json.dumps({"slab_path": str(slab_path)})},
                        },
                        {
                            "id": "call_sites",
                            "type": "function",
                            "function": {
                                "name": "structure_enumerate_sites",
                                "arguments": json.dumps({"slab_path": str(slab_path), "max_sites_per_family": 1}),
                            },
                        },
                    ],
                }
            if self.calls == 3:
                enum_payload = self._tool_payload(messages, "structure_enumerate_sites")
                sites = enum_payload.get("sites") or []
                if sites:
                    self.site_id = str(sites[0]["site_id"])
                    self.site_coords = sites[0]["cart_coords"]
                return {
                    "content": "",
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "call_plan",
                            "type": "function",
                            "function": {
                                "name": "adsorption_candidate_plan",
                                "arguments": json.dumps(
                                    {
                                        "material": "Pt(111)",
                                        "adsorbate": "H2O",
                                        "rationale": "H2O 以 O 原子 lone pair 与 Pt 表面配位，先选一个对称代表位点做 O-down upright 候选，避免无脑全枚举。",
                                        "expected_binding_motif": "O-down upright atop/near-surface Pt coordination",
                                        "anchor_atom": "O",
                                        "target_sites": [
                                            {
                                                "site_id": self.site_id,
                                                "reason": "表面对称代表位点，用于验证 O-down 初猜是否稳定。",
                                            }
                                        ],
                                        "target_orientations": ["upright"],
                                        "symmetry_pruning_applied": True,
                                        "priors_consulted": {"chemistry_hint": "H2O O anchor", "project_prior_hits": 0},
                                        "project": "step2-demo",
                                        "task_id": "step2_model_authored",
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            if self.calls == 4:
                plan_payload = self._tool_payload(messages, "adsorption_candidate_plan")
                self.plan_id = str(plan_payload["plan_id"])
                add_args = {
                    "slab_path": str(slab_path),
                    "adsorbate": "H2O",
                    "output_path": str(candidate_path),
                    "orientation": "upright",
                    "anchor_symbol": "O",
                    "fixed_bottom_layers": 2,
                }
                if self.site_coords is not None:
                    add_args["cart_coords"] = self.site_coords
                return {
                    "content": "",
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "call_add",
                            "type": "function",
                            "function": {"name": "structure_add_adsorbate", "arguments": json.dumps(add_args)},
                        },
                        {
                            "id": "call_quality",
                            "type": "function",
                            "function": {
                                "name": "candidate_quality_score",
                                "arguments": json.dumps(
                                    {
                                        "slab_path": str(slab_path),
                                        "candidate_path": str(candidate_path),
                                        "adsorbate": "H2O",
                                        "anchor_symbol": "O",
                                    }
                                ),
                            },
                        },
                    ],
                }
            if self.calls == 5:
                return {
                    "content": "",
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "call_manifest",
                            "type": "function",
                            "function": {
                                "name": "adsorption_candidate_manifest_compose",
                                "arguments": json.dumps(
                                    {
                                        "task_id": "step2_model_authored",
                                        "material_name": "Pt(111)",
                                        "source_prompt": "模型自主调用 Step 2 工具生成一个 H2O/Pt(111) 候选",
                                        "slab_source": str(slab_path),
                                        "adsorbate_source": "H2O",
                                        "output_dir": str(manifest_dir),
                                        "plan_id": self.plan_id,
                                        "project": "step2-demo",
                                        "candidates": [
                                            {
                                                "candidate_id": "model_pick_01",
                                                "poscar_path": str(candidate_path),
                                                "site_label": self.site_id,
                                                "site_family": self.site_id.split("_", 1)[0],
                                                "orientation_label": "upright",
                                                "anchor_symbol": "O",
                                                "height": 2.0,
                                                "reason": "基于 H2O 的 O anchor 化学先验和 Pt(111) 表面对称代表位点生成，作为少量候选而非全枚举。",
                                            }
                                        ],
                                        "metadata": {"step2_model_driven": True},
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            return {
                "content": "Step 2 已按模型判断调用工具完成：识别意图、收集证据、生成候选、质量检查并写入 manifest。",
                "finish_reason": "stop",
                "tool_calls": [],
            }

    harness = AgentHarness(
        adapter=Step2ModelingAdapter(),
        registry=ToolRegistry(permission_mode="dev"),
        sessions=HarnessSessionStore(tmp_path / "sessions"),
    )

    record = harness.run_turn("请完成 H2O 在 Pt(111) 上的第二步建模", project="step2-demo", max_steps=8)

    tool_names = [item["name"] for item in record["tool_executions"]]
    assert tool_names[:5] == [
        "structure_modeling_intent_plan",
        "adsorbate_chemistry_hint",
        "knowledge_search_for_system",
        "slab_surface_inspect",
        "structure_enumerate_sites",
    ]
    assert "adsorption_candidate_plan" in tool_names
    assert "structure_add_adsorbate" in tool_names
    assert "candidate_quality_score" in tool_names
    assert "adsorption_candidate_manifest_compose" in tool_names
    manifest_result = record["tool_executions"][-1]["result"]
    assert manifest_result["status"] == "composed"
    assert Path(manifest_result["manifest_json"]).exists()
    assert candidate_path.exists()
    assert "Step 2 已按模型判断调用工具完成" in record["response"]


def test_transition_state_tools_are_available():
    registry = ToolRegistry()
    catalog = registry.run_tool("task_type_catalog", {})
    assert catalog["result"]["status"] == "ok"
    types = {item["task_type"] for item in catalog["result"]["task_types"]}
    assert "transition_state_search" in types

    ts_plan = registry.run_tool(
        "transition_state_plan",
        {
            "prompt": "找 Pt(111) 上 H2 解离的过渡态",
            "material": "Pt(111)",
            "persist": False,
        },
    )
    assert ts_plan["result"]["status"] == "ok"
    assert ts_plan["result"]["task"]["plan"]["experiment_type"] == "transition_state_search"
    assert "transition_state_search" in ts_plan["result"]["task"]["dft_command"]
    assert ts_plan["result"]["dry_run"] is True
    assert ts_plan["result"]["deprecated_alias_removed"] == "transition_state_dry_run"


def test_dft_run_tools_return_safe_structured_outputs():
    registry = ToolRegistry()
    report = registry.run_tool("dft_run_report", {"run_id": "missing-run"})
    assert report["result"]["status"] in {"failed", "error"}
    listed = registry.run_tool("dft_run_list", {"limit": 1})
    assert listed["result"]["status"] in {"ok", "failed"}


def test_cluster_execution_intent_plan_teaches_build_preflight_submit_sequence():
    result = ToolRegistry().run_tool(
        "cluster_execution_intent_plan",
        {
            "intent": "把第二步生成的 H2O/Pt(111) POSCAR 按 research 模板生成 VASP 输入，核对后提交集群",
            "project": "MCH-Pt-Br",
            "available_inputs": {
                "structure_path": "candidate.POSCAR",
                "material": "Pt(111)-H2O",
                "task_type": "relax",
                "submit_profile": "c32",
            },
            "allow_submit": False,
        },
    )
    payload = result["result"]
    assert payload["status"] == "ok"
    assert payload["step"] == 3
    assert "教模型如何判断和调用工具" in payload["principle"]
    assert payload["recommended_task_type"] == "relax"
    assert "allow_submit=False" in payload["stop_conditions"][-1]
    assert "model_operating_contract" in payload
    assert "decision_loop" in payload
    assert "adaptive_branches" in payload
    assert any("不要默认从头跑" in item for item in payload["model_operating_contract"])
    assert any(branch["situation"] == "已有 run_root，只想提交" for branch in payload["adaptive_branches"])
    assert payload["template_preview"]["template_id"] == "mch_pt_br_local_relax_alignment"
    tool_names = {
        tool
        for group in payload["tool_groups"]
        for tool in group["candidate_tools"]
    }
    assert "research_onboarding_context" in tool_names
    assert "research_vasp_template_resolve" in tool_names
    assert "dft_run_task" in tool_names
    assert "vasp_input_preflight_check" in tool_names
    assert "cluster_probe" in tool_names
    assert "cluster_remote_submit" in tool_names
    assert any("DFT任务与自由能校正规则" in item["path"] for item in payload["research_rule_paths"])
    assert all("model_decision" in group for group in payload["tool_groups"])
    assert payload["next_decision"]["next_action"] == "resolve_research_constraints"


def test_cluster_execution_intent_uses_project_from_available_inputs():
    result = ToolRegistry().run_tool(
        "cluster_execution_intent_plan",
        {
            "intent": "给 MCH-Pt-Br 已优化中间体做频率计算输入",
            "task_type": "vibrational_frequency",
            "available_inputs": {
                "project": "MCH-Pt-Br",
                "structure_path": "candidate.POSCAR",
                "material": "MCH/Pt(111)",
            },
        },
    )
    payload = result["result"]
    assert payload["project"] == "MCH-Pt-Br"
    assert payload["template_preview"]["template_id"] == "mch_pt_br_stable_intermediate_frequency"
    build_group = next(group for group in payload["tool_groups"] if "dft_run_task" in group["candidate_tools"])
    assert build_group["recommended_arguments"]["project"] == "MCH-Pt-Br"


def test_cluster_execution_intent_does_not_infer_task_type_from_language():
    result = ToolRegistry().run_tool(
        "cluster_execution_intent_plan",
        {
            "intent": "给 MCH-Pt-Br 已优化中间体做频率计算输入",
            "available_inputs": {
                "project": "MCH-Pt-Br",
                "structure_path": "candidate.POSCAR",
                "material": "MCH/Pt(111)",
            },
        },
    )
    payload = result["result"]
    assert payload["recommended_task_type"] == ""
    assert "task_type" in payload["missing_inputs"]
    assert payload["next_decision"]["next_action"] == "model_select_task_type"


def test_cluster_execution_intent_adapts_when_run_root_already_exists():
    result = ToolRegistry().run_tool(
        "cluster_execution_intent_plan",
        {
            "intent": "这个 run 已经 build 好了，帮我核对后提交",
            "project": "MCH-Pt-Br",
            "available_inputs": {
                "run_root": ".aether/runs/task_x/run_y",
                "structure_path": "candidate.POSCAR",
                "material": "MCH/Pt(111)",
                "task_type": "relax",
            },
        },
    )
    assert result["result"]["next_decision"]["next_action"] == "preflight_existing_run"
    assert "dft_run_task build" in result["result"]["next_decision"]["do_not_call"]


def test_cluster_execution_intent_blocks_submit_when_preflight_blocked():
    result = ToolRegistry().run_tool(
        "cluster_execution_intent_plan",
        {
            "intent": "preflight 有 blocker 但我想提交",
            "project": "MCH-Pt-Br",
            "available_inputs": {
                "run_root": ".aether/runs/task_x/run_y",
                "structure_path": "candidate.POSCAR",
                "material": "MCH/Pt(111)",
                "task_type": "relax",
                "preflight_status": "blocked",
            },
        },
    )
    decision = result["result"]["next_decision"]
    assert decision["next_action"] == "fix_preflight_blockers"
    assert "cluster_remote_submit" in decision["do_not_call"]


def test_vasp_input_preflight_checks_inputs_dir_and_research_rules(tmp_path: Path):
    run_root = tmp_path / "run"
    inputs = run_root / "inputs"
    metadata = run_root / "metadata"
    report = run_root / "report"
    inputs.mkdir(parents=True)
    metadata.mkdir()
    report.mkdir()

    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.7, 0, 0]], cell=[8, 8, 12], pbc=True)
    write(inputs / "POSCAR", atoms, format="vasp")
    (inputs / "INCAR").write_text("PREC = Normal\nEDIFF = 1E-5\nIBRION = 2\nNSW = 100\nMAGMOM = 2*0\n", encoding="utf-8")
    (inputs / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (inputs / "job.slurm").write_text("#!/bin/bash\n#SBATCH -J test\nmpirun vasp_std > vasp.out\n", encoding="utf-8")
    (inputs / "POTCAR.mapping.json").write_text('{"Pt": "Pt"}\n', encoding="utf-8")
    (metadata / "experiment_spec.json").write_text('{"task_type": "relax"}\n', encoding="utf-8")
    (report / "pre_submit_checklist.md").write_text("# checklist\n", encoding="utf-8")

    result = ToolRegistry().run_tool(
        "vasp_input_preflight_check",
        {"run_root": str(run_root), "project": "MCH-Pt-Br", "task_type": "relax"},
    )
    payload = result["result"]
    assert payload["status"] == "ready"
    assert payload["files"]["POSCAR"]["exists"] is True
    assert payload["files"]["INCAR"]["exists"] is True
    assert payload["files"]["job.slurm"]["exists"] is True
    assert payload["poscar"]["n_sites"] == 2
    assert payload["incar"]["PREC"] == "Normal"
    assert any("POTCAR.mapping.json" in warning for warning in payload["warnings"])

    summary = ToolRegistry().run_tool("vasp_input_summary", {"run_root": str(run_root)})
    assert summary["result"]["inputs_dir"].endswith("inputs")
    assert summary["result"]["files"]["job.slurm"] is True
    assert summary["result"]["incar"]["EDIFF"] == "1E-5"


def test_vasp_input_preflight_blocks_frequency_without_research_parameters(tmp_path: Path):
    run_root = tmp_path / "freq"
    inputs = run_root / "inputs"
    inputs.mkdir(parents=True)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=False)
    write(inputs / "POSCAR", atoms, format="vasp")
    (inputs / "INCAR").write_text("IBRION = 2\nNSW = 50\n", encoding="utf-8")
    (inputs / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (inputs / "job.slurm").write_text("#!/bin/bash\n", encoding="utf-8")

    result = ToolRegistry().run_tool(
        "vasp_input_preflight_check",
        {"run_root": str(run_root), "project": "MCH-Pt-Br", "task_type": "vibrational_frequency"},
    )

    assert result["result"]["status"] == "blocked"
    assert "频率任务期望 IBRION=5" in result["result"]["blockers"]
    assert result["result"]["research_template"]["template_id"] == "mch_pt_br_stable_intermediate_frequency"


def test_research_vasp_template_resolve_returns_mch_frequency_rules():
    result = ToolRegistry().run_tool(
        "research_vasp_template_resolve",
        {"project": "MCH-Pt-Br", "task_type": "vibrational_frequency", "prompt": "已优化中间体频率自由能校正"},
    )
    template = result["result"]["template"]
    assert template["template_found"] is True
    assert template["template_id"] == "mch_pt_br_stable_intermediate_frequency"
    assert template["incar_overrides"]["IBRION"] == 5
    assert template["incar_overrides"]["POTIM"] == 0.015
    assert any(item["exists"] and "DFT任务与自由能校正规则" in item["path"] for item in template["source_paths"])
    assert "model_instructions" in result["result"]
    assert "not_a_fixed_program" in result["result"]["model_instructions"]
    assert all("sha256" in item for item in template["source_paths"] if item["exists"] and item["label"].startswith("project_common"))
    assert template["source_integrity"]["status"] == "ok"
    assert template["requires_template_review"] is False


def test_vasp_input_preflight_blocks_dimer_research_parameter_mismatch(tmp_path: Path):
    run_root = tmp_path / "dimer"
    inputs = run_root / "inputs"
    inputs.mkdir(parents=True)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=False)
    write(inputs / "POSCAR", atoms, format="vasp")
    (inputs / "INCAR").write_text(
        "PREC = Accurate\nEDIFF = 1E-6\nIOPT = 2\nICHAIN = 1\nMAGMOM = 2*0\n",
        encoding="utf-8",
    )
    (inputs / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (inputs / "job.slurm").write_text("#!/bin/bash\n", encoding="utf-8")

    result = ToolRegistry().run_tool(
        "vasp_input_preflight_check",
        {"run_root": str(run_root), "project": "MCH-Pt-Br", "task_type": "transition_state_search"},
    )
    payload = result["result"]
    assert payload["status"] == "blocked"
    assert payload["research_template"]["template_id"] == "mch_pt_br_vasp_dimer_refinement"
    assert any("PREC=Normal" in blocker for blocker in payload["blockers"])
    assert any("IOPT=1" in blocker for blocker in payload["blockers"])


def test_dft_run_task_submit_is_blocked_without_submit_permission(monkeypatch):
    called = False

    def fake_run_dft_task(*args, **kwargs):
        nonlocal called
        called = True
        return {"status": "ok"}

    monkeypatch.setattr("aether_dft.runtime_harness.tool_registry.run_dft_task", fake_run_dft_task)
    result = ToolRegistry(allow_cluster_submit=False).run_tool(
        "dft_run_task",
        {"prompt": "submit", "execution_mode": "remote_submit"},
    )
    assert result["result"]["status"] == "blocked"
    assert called is False


def test_create_task_plan_applies_research_template_to_spec():
    from aether_dft.task_bridge import create_task_plan

    envelope = create_task_plan(
        "MCH-Pt-Br 已优化中间体做振动频率计算",
        project="MCH-Pt-Br",
        material="MCH/Pt(111)",
        structure_path="candidate.POSCAR",
        task_type="vibrational_frequency",
        persist=False,
    )
    assert envelope.spec is not None
    assert envelope.spec["incar_overrides"]["IBRION"] == 5
    assert envelope.spec["incar_overrides"]["PREC"] == "Normal"
    assert envelope.spec["notes"]["research_template"]["template_id"] == "mch_pt_br_stable_intermediate_frequency"


def test_create_task_plan_records_step2_lineage_without_forcing_workflow():
    from aether_dft.task_bridge import create_task_plan

    envelope = create_task_plan(
        "把模型选中的吸附构型准备为 relax 计算",
        project="demo",
        material="H2O/Pt(111)",
        structure_path="candidate.POSCAR",
        task_type="relax",
        model_spec_path="model_spec.json",
        step2_manifest_path="manifest.json",
        candidate_id="model_pick_01",
        persist=False,
    )
    assert envelope.spec is not None
    lineage = envelope.spec["notes"]["step2_lineage"]
    assert lineage["model_spec_path"] == "model_spec.json"
    assert lineage["step2_manifest_path"] == "manifest.json"
    assert lineage["candidate_id"] == "model_pick_01"
    assert "--model-spec-path" in envelope.dft_command


def test_vasp_input_preflight_accepts_frequency_when_research_template_matches(tmp_path: Path):
    run_root = tmp_path / "freq_ready"
    inputs = run_root / "inputs"
    metadata = run_root / "metadata"
    report = run_root / "report"
    inputs.mkdir(parents=True)
    metadata.mkdir()
    report.mkdir()
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=False)
    write(inputs / "POSCAR", atoms, format="vasp")
    (inputs / "INCAR").write_text(
        "IBRION = 5\nNFREE = 2\nPOTIM = 0.015\nNSW = 1\nISYM = 0\nPREC = Normal\nLREAL = Auto\nEDIFF = 1E-5\nEDIFFG = -0.03\nMAGMOM = 2*0\n",
        encoding="utf-8",
    )
    (inputs / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (inputs / "job.slurm").write_text("#!/bin/bash\n#SBATCH -J freq\nmpirun vasp_std > vasp.out\n", encoding="utf-8")
    (inputs / "POTCAR.mapping.json").write_text('{"H": "H"}\n', encoding="utf-8")
    (metadata / "experiment_spec.json").write_text('{"task_type": "vibrational_frequency"}\n', encoding="utf-8")
    (report / "pre_submit_checklist.md").write_text("# checklist\n", encoding="utf-8")

    result = ToolRegistry().run_tool(
        "vasp_input_preflight_check",
        {"run_root": str(run_root), "project": "MCH-Pt-Br", "task_type": "vibrational_frequency"},
    )
    payload = result["result"]
    assert payload["status"] == "ready"
    assert payload["research_template"]["template_id"] == "mch_pt_br_stable_intermediate_frequency"
    assert not payload["blockers"]


def test_submit_runner_blocks_when_current_inputs_do_not_match_research_gate(tmp_path: Path):
    from dft_app.models import ExperimentSpec, PipelinePhase, RunRecord, TaskType
    from dft_app.runner import SlurmRunner

    run_root = tmp_path / "run"
    inputs = run_root / "inputs"
    inputs.mkdir(parents=True)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=False)
    write(inputs / "POSCAR", atoms, format="vasp")
    (inputs / "INCAR").write_text("IBRION = 2\nNSW = 50\nMAGMOM = 2*0\n", encoding="utf-8")
    (inputs / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (inputs / "job.slurm").write_text("#!/bin/bash\nsbatch should_not_run\n", encoding="utf-8")
    (inputs / "POTCAR.mapping.json").write_text('{"H": "H"}\n', encoding="utf-8")
    spec = ExperimentSpec(
        task_id="task_gate",
        task_type=TaskType.VIBRATIONAL_FREQUENCY,
        material_name="H2",
        source_prompt="freq",
        structure_path=str(inputs / "POSCAR"),
        notes={
            "research_template": {
                "template_found": True,
                "template_id": "test_freq",
                "expected_incar": {"IBRION": 5, "NSW": 1},
                "severity_by_key": {"IBRION": "blocker", "NSW": "blocker"},
            }
        },
    )
    record = RunRecord(
        task_id="task_gate",
        run_id="run_gate",
        run_root=str(run_root),
        checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
    )
    record.complete_phase(PipelinePhase.BUILD, artifacts=[str(inputs / "POSCAR")])
    record.mark_ready()

    result = SlurmRunner().submit(spec, record)

    assert result.status == "blocked"
    assert any("IBRION=5" in item for item in result.details["blockers"])
    assert (run_root / "metadata" / "pre_submit_gate.json").exists()


def test_local_submit_runner_requires_real_potcar_not_only_mapping(tmp_path: Path):
    from dft_app.models import ExperimentSpec, PipelinePhase, RunRecord, TaskType
    from dft_app.runner import SlurmRunner

    run_root = tmp_path / "local_submit"
    inputs = run_root / "inputs"
    inputs.mkdir(parents=True)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=False)
    write(inputs / "POSCAR", atoms, format="vasp")
    (inputs / "INCAR").write_text("IBRION = 2\nNSW = 50\nMAGMOM = 2*0\n", encoding="utf-8")
    (inputs / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (inputs / "job.slurm").write_text("#!/bin/bash\nmpirun vasp_std > vasp.out\n", encoding="utf-8")
    (inputs / "POTCAR.mapping.json").write_text('{"H": "H"}\n', encoding="utf-8")
    spec = ExperimentSpec(
        task_id="task_local_potcar",
        task_type=TaskType.RELAX,
        material_name="H2",
        source_prompt="relax",
        structure_path=str(inputs / "POSCAR"),
    )
    record = RunRecord(
        task_id="task_local_potcar",
        run_id="run_local_potcar",
        run_root=str(run_root),
        checkpoint_path=str(run_root / "outputs" / ".pipeline_checkpoint.json"),
    )
    record.complete_phase(PipelinePhase.BUILD, artifacts=[str(inputs / "POSCAR")])
    record.mark_ready()

    result = SlurmRunner().submit(spec, record)

    assert result.status == "blocked"
    assert any("真实 POTCAR" in item for item in result.details["blockers"])


def test_experiment_spec_from_dict_accepts_legacy_minimal_metadata():
    from dft_app.models import experiment_spec_from_dict

    spec = experiment_spec_from_dict(
        {
            "task_id": "legacy",
            "task_type": "relax",
            "material_name": "Pt",
            "source_prompt": "legacy relax",
            "structure_path": "POSCAR",
        }
    )
    assert spec.task_id == "legacy"
    assert spec.kpoints_strategy.mode == "auto_density"
    assert spec.job_overrides.nodes == 1


def test_structure_analysis_tools_run_on_real_structures(tmp_path: Path):
    initial = tmp_path / "POSCAR"
    final = tmp_path / "CONTCAR"
    atoms_i = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=False)
    atoms_f = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.84]], cell=[8, 8, 8], pbc=False)
    write(initial, atoms_i, format="vasp")
    write(final, atoms_f, format="vasp")

    registry = ToolRegistry()
    bonds = registry.run_tool("structure_bond_analyze", {"structure_path": str(initial)})
    assert bonds["result"]["status"] == "ok"
    assert bonds["result"]["report"]["n_bonds"] >= 1

    displacement = registry.run_tool(
        "structure_displacement_compare",
        {"initial_path": str(initial), "final_path": str(final), "top_n": 1},
    )
    assert displacement["result"]["status"] == "ok"
    assert displacement["result"]["report"]["max_displacement"] > 0


def test_vasp_scan_tools_work_without_run_record(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "INCAR").write_text("ENCUT = 400\nEDIFF = 1E-5\n", encoding="utf-8")
    (root / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=False)
    write(root / "POSCAR", atoms, format="vasp")
    (root / "OUTCAR").write_text(
        " free  energy   TOTEN  =       -6.123456 eV\n"
        " E-fermi :   1.2345\n"
        " reached required accuracy - stopping structural energy minimisation\n"
        " General timing and accounting informations for this job\n",
        encoding="utf-8",
    )
    (root / "OSZICAR").write_text(" 1 F= -.61234560E+01 E0= -.61234560E+01\n", encoding="utf-8")

    registry = ToolRegistry()
    scanned = registry.run_tool("vasp_output_scan", {"run_root": str(root)})
    assert scanned["result"]["status"] == "completed"
    assert scanned["result"]["outcar"]["exists"] is True
    assert scanned["result"]["outcar"]["has_required_accuracy"] is True
    assert scanned["result"]["outcar"]["last_toten"] == -6.123456

    inputs = registry.run_tool("vasp_input_summary", {"run_root": str(root)})
    assert inputs["result"]["status"] == "ok"
    assert inputs["result"]["incar"]["ENCUT"] == "400"
    assert inputs["result"]["poscar"]["n_sites"] == 2


def test_vasp_scan_does_not_claim_completion_without_convergence(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "OUTCAR").write_text(
        " free  energy   TOTEN  =       -6.123456 eV\n"
        " E-fermi :   1.2345\n"
        " some intermediate step without convergence\n",
        encoding="utf-8",
    )
    (root / "OSZICAR").write_text(" 1 F= -.61234560E+01 E0= -.61234560E+01\n", encoding="utf-8")

    registry = ToolRegistry()
    scanned = registry.run_tool("vasp_output_scan", {"run_root": str(root)})
    assert scanned["result"]["status"] == "incomplete"
    assert scanned["result"]["outcar"]["exists"] is True
    assert scanned["result"]["outcar"]["has_required_accuracy"] is False
    assert scanned["result"]["outcar"]["last_toten"] == -6.123456


def test_vasp_scan_flags_synthetic_smoke_output(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "OUTCAR").write_text(
        "AETHER synthetic VASP-like output for smoke-test validation\n"
        "free  energy   TOTEN  =      -6.123456 eV\n"
        "reached required accuracy\n",
        encoding="utf-8",
    )
    (root / "OSZICAR").write_text(" 1 F= -.61234560E+01 E0= -.61234560E+01\n", encoding="utf-8")

    scanned = ToolRegistry().run_tool("vasp_output_scan", {"run_root": str(root)})["result"]

    assert scanned["status"] == "test_output"
    assert scanned["outcar"]["is_synthetic_output"] is True
    assert scanned["synthetic_output"]["detected"] is True
    assert "不能作为真实 VASP 科学结果" in scanned["warnings"][0]


def test_vasp_input_summary_distinguishes_remote_materialized_potcar(tmp_path: Path):
    run_root = tmp_path / "run"
    inputs = run_root / "inputs"
    metadata = run_root / "metadata"
    inputs.mkdir(parents=True)
    metadata.mkdir()
    (inputs / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (inputs / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    write(inputs / "POSCAR", Atoms("Pt", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True), format="vasp")
    (inputs / "POTCAR.mapping.json").write_text('{"Pt": "/home/user/POTCAR/Pt/POTCAR"}', encoding="utf-8")
    (metadata / "run_record.json").write_text(
        json.dumps(
            {
                "notes": {
                    "remote": {
                        "uploaded_files": [
                            "/home/user/aether/run/inputs/INCAR",
                            "/home/user/aether/run/inputs/POTCAR",
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    summary = ToolRegistry().run_tool("vasp_input_summary", {"run_root": str(run_root)})["result"]

    assert summary["status"] == "ok"
    assert summary["files"]["POTCAR"] is False
    assert summary["potcar_status"]["mapping_exists"] is True
    assert summary["potcar_status"]["remote_uploaded"] is True
    assert "不要仅凭本地缺失判断远程会失败" in summary["potcar_status"]["guidance"]


def test_job_watcher_understands_slurm_cancelled_by_user_state():
    from aether_dft.job_watcher import _job_followup_options

    options = _job_followup_options({"last_known_state": "CANCELLED BY 20020"})
    goals = {item["goal"] for item in options}

    assert "diagnose_stopped_job" in goals
    assert "clarify_unknown_state" not in goals


def test_dft_run_step_does_not_pretend_execution():
    registry = ToolRegistry()
    result = registry.run_tool("dft_run_step", {"phase": "submit"})
    assert result["result"]["status"] == "needs_inputs"
    assert "不会伪造" in result["result"]["message"]
    assert "prompt" in result["result"]["required_inputs"]


def test_ts_neb_dimer_check_tools_are_honest(tmp_path: Path):
    registry = ToolRegistry()
    cfg = registry.run_tool("ts_workflow_config", {})
    assert cfg["result"]["status"] == "ok"
    assert "不会假装" in cfg["result"]["boundary"]

    neb = registry.run_tool("neb_input_check", {"n_images": 4})
    assert neb["result"]["status"] == "needs_inputs"
    assert "initial_path" in neb["result"]["missing"]
    assert "不执行 MACE" in neb["result"]["boundary"]

    work_dir = tmp_path / "dimer"
    work_dir.mkdir()
    (work_dir / "POSCAR").write_text("dummy\n", encoding="utf-8")
    dimer = registry.run_tool("dimer_input_check", {"work_dir": str(work_dir)})
    assert dimer["result"]["status"] == "needs_inputs"
    assert "MODECAR" in dimer["result"]["missing"]
    assert "不执行远程提交" in dimer["result"]["boundary"]


def test_adsorption_candidates_tool_runs_through_registry(tmp_path: Path):
    atoms = fcc111("Pt", size=(1, 1, 3), vacuum=8.0)
    slab_path = tmp_path / "POSCAR"
    AseAtomsAdaptor.get_structure(atoms).to(fmt="poscar", filename=str(slab_path))

    result = ToolRegistry().run_tool(
        "adsorption_candidates",
        {
            "slab_path": str(slab_path),
            "adsorbate": "H2O",
            "material": "Pt(111)",
            "output_dir": str(tmp_path / "candidates"),
            "max_sites_per_family": 1,
        },
    )

    assert result["result"]["status"] == "ok"
    assert result["result"]["result"]["candidate_count"] > 0


def test_research_onboarding_and_proposal_tools_read_project_progress_without_identity_leak():
    registry = ToolRegistry()
    context = registry.run_tool("research_onboarding_context", {"project": "MCH-Pt-Br", "max_chars": 8000})
    assert context["result"]["status"] == "ok"
    assert "避坑清单" in context["result"]["context"]
    assert "研究进展" in context["result"]["context"]
    assert "Zhang Song" not in context["result"]["context"]

    proposal = registry.run_tool(
        "research_proposal_plan",
        {"project": "MCH-Pt-Br", "prompt": "讨论 Pt(111) 上 MCH 脱氢下一步需要什么结构和 DFT 证据"},
    )
    assert proposal["result"]["status"] in {"ready", "needs_inputs"}
    assert "required_structures" in proposal["result"]["proposal"]
    assert proposal["result"]["onboarding_files_read"]


def test_structure_operation_tools_cover_first_modeling_steps(tmp_path: Path):
    atoms = fcc111("Pt", size=(1, 1, 3), vacuum=8.0)
    slab_path = tmp_path / "POSCAR"
    AseAtomsAdaptor.get_structure(atoms).to(fmt="poscar", filename=str(slab_path))
    registry = ToolRegistry()

    sanity = registry.run_tool("structure_sanity_check", {"structure_path": str(slab_path)})
    assert sanity["result"]["status"] in {"ok", "warning"}
    assert sanity["result"]["summary"]["atom_count"] == 3

    supercell_path = tmp_path / "POSCAR_super"
    supercell = registry.run_tool(
        "structure_supercell",
        {"input_path": str(slab_path), "output_path": str(supercell_path), "scaling_matrix": [2, 1, 1]},
    )
    assert supercell["result"]["status"] == "ok"
    assert supercell["result"]["summary"]["atom_count"] == 6

    ads_path = tmp_path / "POSCAR_H2O"
    ads = registry.run_tool(
        "structure_add_adsorbate",
        {"slab_path": str(slab_path), "adsorbate": "H2O", "output_path": str(ads_path), "height": 2.0, "anchor_symbol": "O"},
    )
    assert ads["result"]["status"] == "ok"
    assert ads["result"]["summary"]["atom_count"] > 3
    assert ads["result"]["anchor_symbol"] == "O"

    vacancy_path = tmp_path / "POSCAR_vac"
    vacancy = registry.run_tool(
        "structure_add_vacancy",
        {"input_path": str(slab_path), "output_path": str(vacancy_path), "species": "Pt"},
    )
    assert vacancy["result"]["status"] == "ok"
    assert vacancy["result"]["summary"]["atom_count"] == 2

    doped_path = tmp_path / "POSCAR_Au"
    doped = registry.run_tool(
        "structure_add_dopant",
        {"input_path": str(slab_path), "output_path": str(doped_path), "species": "Pt", "dopant": "Au", "surface_only": True},
    )
    assert doped["result"]["status"] == "ok"
    assert "Au" in doped["result"]["summary"]["species"]

    defect_path = tmp_path / "POSCAR_defect"
    defect = registry.run_tool(
        "structure_defect",
        {"input_path": str(slab_path), "output_path": str(defect_path), "mode": "vacancy", "species": "Pt"},
    )
    assert defect["result"]["status"] == "ok"
    assert defect["result"]["summary"]["atom_count"] == 2


def test_openai_tool_schema_has_required_fields_and_optional_structure_params():
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in ToolRegistry().openai_tool_schemas()}
    assert set(schemas["structure_supercell"]["required"]) == {"input_path", "output_path", "scaling_matrix"}
    assert "min_slab_size" in schemas["structure_build_slab"]["properties"]
    assert "min_vacuum_size" in schemas["structure_build_slab"]["properties"]
    assert "fixed_bottom_layers" in schemas["structure_build_slab"]["properties"]
    assert "anchor_symbol" in schemas["structure_add_adsorbate"]["properties"]
    assert "candidate_height" in schemas["adsorption_candidates"]["properties"]
    assert schemas["structure_resolve"]["required"] == []


def test_architecture_live_doc_snapshot_tool_returns_semantic_digest():
    result = ToolRegistry().run_tool("architecture_live_doc_snapshot", {"max_chars": 2000})
    assert result["result"]["status"] == "ok"
    snapshot = result["result"]["snapshot"]
    assert "Step 1" in snapshot["architecture_live_doc_digest_text"]
    assert "Step 2" in snapshot["architecture_live_doc_digest_text"]
    assert snapshot["architecture_live_doc_path"].endswith("智能体架构.md")


def test_architecture_live_doc_update_tool_appends_block(tmp_path: Path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    root_doc = project_root / "智能体架构.md"
    root_doc.write_text("# AETHER-DFT 智能体架构\n\n原始内容\n", encoding="utf-8")
    monkeypatch.setattr(paths, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", tmp_path / "kb")
    import aether_dft.prompt_engine as prompt_engine

    monkeypatch.setattr(prompt_engine, "PROJECT_ROOT", project_root)

    result = ToolRegistry().run_tool(
        "architecture_live_doc_update",
        {"title": "测试块", "content": "Step 1\nStep 2"},
    )
    assert result["result"]["status"] == "ok"
    updated = root_doc.read_text(encoding="utf-8")
    assert "测试块" in updated
    assert "Step 1" in updated
    assert "Step 2" in updated


def test_vasp_output_scan_finds_fetched_inputs_subdir_outputs(tmp_path: Path):
    from aether_dft.runtime_harness.tool_registry import ToolRegistry

    run_root = tmp_path / "run"
    output_inputs = run_root / "outputs" / "inputs"
    output_inputs.mkdir(parents=True)
    (output_inputs / "OUTCAR").write_text(
        "free  energy   TOTEN  =      -6.123456 eV\n reached required accuracy\n",
        encoding="utf-8",
    )
    (output_inputs / "OSZICAR").write_text("  1 F= -.6123456E+01 E0= -.6123456E+01\n", encoding="utf-8")

    result = ToolRegistry().run_tool("vasp_output_scan", {"run_root": str(run_root)})["result"]

    assert result["status"] == "completed"
    assert result["outcar"]["last_toten"] == -6.123456
    assert result["outcar"]["path"].endswith("outputs\\inputs\\OUTCAR") or result["outcar"]["path"].endswith("outputs/inputs/OUTCAR")
    assert result["oszicar_exists"] is True
