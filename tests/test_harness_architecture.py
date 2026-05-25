from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.build import fcc111
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

from aether_dft import paths, project_state
from aether_dft.prompt_engine import load_base_system_prompt
from aether_dft.runtime_harness.core import AgentHarness
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
    assert "transition_state_dry_run" in names
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
    assert "dft_task_plan" in names
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
        "dft_task_plan",
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
    assert len(record["tool_executions"][0]["result"]["payload"]) == 50000
    persisted = Path(record["tool_executions"][0]["persisted_output_path"])
    assert persisted.exists()
    assert persisted.read_text(encoding="utf-8")


def test_agent_harness_replays_recent_session_context(tmp_path: Path):
    sessions = HarnessSessionStore(tmp_path / "sessions")
    harness = AgentHarness(adapter=SessionReplayAdapter(), sessions=sessions)

    first = harness.run_turn("第一轮提问", max_steps=2)
    assert first["response"] == "第一轮已完成。"

    second = harness.run_turn("第二轮继续追问", session_id=first["session_id"], max_steps=2)
    assert second["response"] == "已续接前文，可以继续推进。"


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


def test_workflow_map_exposes_full_computational_chemistry_flow():
    result = ToolRegistry().run_tool("computational_chemistry_workflow_map", {})
    assert result["result"]["status"] == "ok"
    assert result["result"]["mainline"][0]["step"] == 1
    assert result["result"]["mainline"][1]["step"] == 2
    assert "research_onboarding_context" in result["result"]["mainline"][0]["tools"]
    assert "research_proposal_plan" in result["result"]["mainline"][0]["tools"]
    assert "architecture_live_doc_snapshot" in result["result"]["mainline"][0]["tools"]
    assert "architecture_live_doc_update" in result["result"]["mainline"][0]["tools"]
    assert "structure_modeling_tool_status" in result["result"]["mainline"][1]["tools"]
    assert "structure_resolve" in result["result"]["mainline"][1]["tools"]
    assert "structure_sanity_check" in result["result"]["mainline"][1]["tools"]
    assert "structure_defect" in result["result"]["mainline"][1]["tools"]
    assert "adsorption_plan" in result["result"]["mainline"][1]["tools"]
    phases = [item["phase"] for item in result["result"]["workflow"]]
    assert phases == [
        "project_context",
        "structure_io",
        "adsorption_modeling",
        "dft_tasking",
        "cluster_execution",
        "parse",
        "knowledge_backflow",
    ]


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


def test_dft_run_tools_return_safe_structured_outputs():
    registry = ToolRegistry()
    report = registry.run_tool("dft_run_report", {"run_id": "missing-run"})
    assert report["result"]["status"] in {"failed", "error"}
    listed = registry.run_tool("dft_run_list", {"limit": 1})
    assert listed["result"]["status"] in {"ok", "failed"}


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
