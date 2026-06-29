from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from ase import Atoms
from ase.io import write

from aether_dft.context_digests import build_cluster_runtime_digest, build_job_watch_digest
from aether_dft import research_workspace
from aether_dft.runtime_harness.core import AgentHarness
from aether_dft.runtime_harness.tool_registry import ToolRegistry


@dataclass
class _FakeRemoteResult:
    status: str
    message: str
    details: dict[str, Any]


class _FakeRemoteConfig:
    user = "tester"


class _FakeResearchRunner:
    def _load_config(self):
        return _FakeRemoteConfig()

    def research_status(self, local_research_root, *, remote_research_dir=None):
        return _FakeRemoteResult(
            "ok",
            "diff",
            {
                "sync_status": "out_of_sync",
                "local_root": str(local_research_root),
                "remote_research_dir": remote_research_dir,
                "missing_remote": ["研究进展.md"],
                "missing_local": ["Learning/cluster.md"],
                "differing": ["Common/避坑清单.md"],
            },
        )

    def sync_research_to_remote(self, local_research_root, *, remote_research_dir=None, dry_run=True):
        return _FakeRemoteResult(
            "planned" if dry_run else "synced",
            "push",
            {"local_root": str(local_research_root), "remote_research_dir": remote_research_dir, "dry_run": dry_run},
        )


def test_general_agent_tools_are_registered_and_honest_fallbacks():
    registry = ToolRegistry()
    names = {tool["name"] for tool in registry.list_tools()}
    assert {
        "web_search",
        "literature_search",
        "chemistry_compute",
        "image_understand",
        "discussion_state_snapshot",
        "behavior_audit",
    }.issubset(names)

    web = registry.run_tool("web_search", {"query": "Pt(111) water adsorption DFT"})["result"]
    assert web["status"] == "ok"
    assert web["mode"] == "connector_required"
    assert web["results"] == []
    assert "不要把这个空结果当成事实" in web["guidance"]

    boltz = registry.run_tool(
        "chemistry_compute",
        {"operation": "boltzmann_population", "energies_ev": [0.0, 0.1], "temperature_k": 300},
    )["result"]
    assert boltz["status"] == "ok"
    assert boltz["populations"][0] > boltz["populations"][1]


def test_research_workspace_tools_are_project_scoped_and_dry_run(tmp_path: Path, monkeypatch):
    root = tmp_path / "research"
    demo = root / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "研究进展.md").write_text("# 研究进展\n", encoding="utf-8")
    monkeypatch.setattr(research_workspace, "RESEARCH_ROOT", root)
    monkeypatch.setattr("aether_dft.research_sync.RESEARCH_ROOT", root)
    monkeypatch.setattr("aether_dft.research_sync.SSHRemoteRunner", _FakeResearchRunner)
    registry = ToolRegistry()
    names = {tool["name"] for tool in registry.list_tools()}
    assert {
        "research_workspace_diff",
        "research_workspace_sync_to_cluster",
        "research_workspace_sync_from_cluster",
        "research_workspace_pull_logs",
        "research_learning_capture",
    }.issubset(names)

    diff = registry.run_tool("research_workspace_diff", {"project": "demo"})["result"]
    assert diff["status"] == "ok"
    assert diff["details"]["remote_research_dir"].endswith("/research/demo")

    push = registry.run_tool("research_workspace_sync_to_cluster", {"project": "demo"})["result"]
    assert push["status"] == "planned"
    assert push["details"]["dry_run"] is True

    pull = registry.run_tool("research_workspace_sync_from_cluster", {"project": "demo"})["result"]
    assert pull["status"] == "planned"
    assert set(pull["details"]["would_pull"]) == {"Learning/cluster.md", "Common/避坑清单.md"}


def test_research_workspace_rejects_unknown_project(tmp_path: Path, monkeypatch):
    root = tmp_path / "research"
    root.mkdir()
    monkeypatch.setattr(research_workspace, "RESEARCH_ROOT", root)
    monkeypatch.setattr("aether_dft.research_sync.RESEARCH_ROOT", root)
    monkeypatch.setattr("aether_dft.research_sync.SSHRemoteRunner", _FakeResearchRunner)

    result = ToolRegistry().run_tool("research_workspace_sync_from_cluster", {"project": "../escape", "apply": True})["result"]
    assert result["status"] == "error"
    assert "不存在" in result["message"]


def test_research_learning_capture_writes_project_learning(tmp_path: Path, monkeypatch):
    root = tmp_path / "research"
    project = root / "DemoProject"
    project.mkdir(parents=True)
    (project / "研究进展.md").write_text("# 研究进展\n", encoding="utf-8")
    monkeypatch.setattr("aether_dft.research_workspace.RESEARCH_ROOT", root)
    monkeypatch.setattr("aether_dft.research_sync.RESEARCH_ROOT", root)

    registry = ToolRegistry()
    result = registry.run_tool(
        "research_learning_capture",
        {"project": "DemoProject", "title": "Pt water prior", "content": "ontop is a diagnostic baseline", "tags": "adsorption;baseline"},
    )["result"]
    assert result["status"] == "ok"
    path = Path(result["learning_path"])
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "ontop is a diagnostic baseline" in text
    assert "#adsorption #baseline" in text


def test_tool_arguments_leniently_accept_markdown_content_with_literal_newlines(tmp_path: Path, monkeypatch):
    root = tmp_path / "research"
    project = root / "DemoProject"
    project.mkdir(parents=True)
    (project / "研究进展.md").write_text("# 研究进展\n", encoding="utf-8")
    monkeypatch.setattr("aether_dft.research_workspace.RESEARCH_ROOT", root)
    monkeypatch.setattr("aether_dft.research_sync.RESEARCH_ROOT", root)

    raw_args = '{"project":"DemoProject","title":"Dimer note","content":"# Title\nline two","tags":"dimer;freq"}'
    result = ToolRegistry().run_tool("research_learning_capture", raw_args)["result"]

    assert result["status"] == "ok"
    text = Path(result["learning_path"]).read_text(encoding="utf-8")
    assert "# Title\nline two" in text
    assert "#dimer #freq" in text


def test_research_progress_append_preserves_project_heading(tmp_path: Path, monkeypatch):
    root = tmp_path / "research"
    project = root / "DemoProject"
    project.mkdir(parents=True)
    progress = project / "研究进展.md"
    progress.write_text(
        "# DemoProject 研究进展\n\n> existing rules\n\n---\n\n### 2026-01-01\n\n- ✅ old item\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("aether_dft.research_workspace.RESEARCH_ROOT", root)
    monkeypatch.setattr("aether_dft.research_sync.RESEARCH_ROOT", root)

    result = ToolRegistry().run_tool(
        "research_progress_append",
        {"project": "DemoProject", "completed": "new item"},
    )["result"]

    assert result["status"] == "ok"
    text = progress.read_text(encoding="utf-8")
    assert text.count("# DemoProject 研究进展") == 1
    assert "# 研究进展" not in text
    assert text.index("- ✅ new item") < text.index("- ✅ old item")


def test_result_interpret_and_next_experiment_tools(tmp_path: Path):
    (tmp_path / "OUTCAR").write_text(
        "free  energy   TOTEN  =      -123.456 eV\nreached required accuracy\n",
        encoding="utf-8",
    )
    (tmp_path / "OSZICAR").write_text("1 F= -122.0\n2 F= -123.0\n", encoding="utf-8")
    (tmp_path / "CONTCAR").write_text("placeholder", encoding="utf-8")

    registry = ToolRegistry()
    interpreted = registry.run_tool("result_interpret", {"run_root": str(tmp_path)})["result"]
    assert interpreted["status"] == "ok"
    assert interpreted["verdict"] == "finished_converged"
    assert interpreted["energy"]["last_toten_ev"] == pytest.approx(-123.456)

    proposed = registry.run_tool("next_experiment_propose", {"project": "demo", "recent_results": [interpreted]})["result"]
    assert proposed["status"] == "ok"
    assert len(proposed["proposals"]) == 3


def test_result_interpret_does_not_treat_synthetic_smoke_as_science(tmp_path: Path):
    (tmp_path / "OUTCAR").write_text(
        "AETHER synthetic VASP-like output for smoke-test validation\n"
        "free  energy   TOTEN  =      -123.456 eV\n"
        "reached required accuracy\n",
        encoding="utf-8",
    )
    (tmp_path / "OSZICAR").write_text("1 F= -123.0\n", encoding="utf-8")
    (tmp_path / "CONTCAR").write_text("placeholder", encoding="utf-8")

    interpreted = ToolRegistry().run_tool("result_interpret", {"run_root": str(tmp_path)})["result"]

    assert interpreted["status"] == "ok"
    assert interpreted["verdict"] == "test_output_detected"
    assert interpreted["synthetic_output"]["detected"] is True
    assert any("不能作为真实 VASP 科学结果" in warning for warning in interpreted["warnings"])


def test_result_interpret_recognizes_finished_frequency_without_imaginary_modes(tmp_path: Path):
    (tmp_path / "OUTCAR").write_text(
        "\n".join(
            [
                " free  energy   TOTEN  =      -640.56417145 eV",
                " Eigenvectors and eigenvalues of the dynamical matrix",
                " 1 f  =   17.692543 THz   111.166000 2PiTHz  590.166 cm-1",
                " 2 f  =   22.123456 THz   138.000000 2PiTHz  738.000 cm-1",
                " General timing and accounting informations for this job",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "OSZICAR").write_text("DAV: 1 -640.0\n", encoding="utf-8")

    result = ToolRegistry().run_tool("result_interpret", {"run_root": str(tmp_path)})["result"]

    assert result["status"] == "ok"
    assert result["verdict"] == "frequency_finished_no_imaginary_modes"
    assert result["frequency"]["detected"] is True
    assert result["frequency"]["real_mode_count"] == 2
    assert result["frequency"]["imaginary_mode_count"] == 0
    assert not any("reached required accuracy" in warning for warning in result["warnings"])


def test_result_interpret_flags_possible_adsorbate_dissociation(tmp_path: Path):
    initial = Atoms(
        "PtOH",
        positions=[(0, 0, 0), (0, 0, 2.0), (0, 0, 2.97)],
        cell=[8, 8, 12],
        pbc=[True, True, False],
    )
    final = Atoms(
        "PtOH",
        positions=[(0, 0, 0), (0, 0, 2.0), (0, 0, 5.4)],
        cell=[8, 8, 12],
        pbc=[True, True, False],
    )
    write(tmp_path / "POSCAR", initial, format="vasp", direct=True)
    write(tmp_path / "CONTCAR", final, format="vasp", direct=True)
    (tmp_path / "OUTCAR").write_text("free  energy   TOTEN  = -10.0 eV\nreached required accuracy\n", encoding="utf-8")
    (tmp_path / "OSZICAR").write_text("1 F= -10.0\n", encoding="utf-8")

    result = ToolRegistry().run_tool("result_interpret", {"run_root": str(tmp_path)})["result"]
    assert result["status"] == "ok"
    assert result["structure_change"]["status"] == "ok"
    assert result["adsorption_interpretation"] in {
        "possible_adsorbate_dissociation",
        "possible_desorption_or_large_migration",
        "bonding_changed_review_geometry",
    }
    assert result["structure_change"]["broken_bonds"] or result["structure_change"]["adsorbate_drift"]


def test_cluster_runtime_digest_filters_to_current_project(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / ".aether" / "runs"
    for task_id, run_id, job_id in [
        ("DemoProject-task", "run-demo", "111"),
        ("OtherProject-task", "run-other", "222"),
        ("DemoProject2-task", "run-demo2", "333"),
    ]:
        meta = runs / task_id / run_id / "metadata"
        meta.mkdir(parents=True)
        (meta / "run_record.json").write_text(
            (
                "{"
                f"\"task_id\":\"{task_id}\",\"run_id\":\"{run_id}\",\"run_root\":\"{(runs / task_id / run_id).as_posix()}\","
                "\"created_at\":\"2026-01-01T00:00:00\",\"updated_at\":\"2026-01-01T00:00:00\","
                "\"overall_status\":\"running\",\"current_phase\":\"submit\","
                f"\"scheduler_job_id\":\"{job_id}\",\"checkpoint_path\":\"x\",\"tags\":[],\"notes\":{{}},\"phases\":{{}}"
                "}"
            ),
            encoding="utf-8",
        )

    digest = build_cluster_runtime_digest(project="DemoProject")
    assert "111" in digest
    assert "222" not in digest
    assert "333" not in digest


def test_empty_cluster_runtime_digest_does_not_claim_live_queue_state(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    digest = build_cluster_runtime_digest(project="DemoProject")

    assert "local run store" in digest
    assert "not live scheduler evidence" in digest
    assert "does not prove the user's cluster queue is empty" in digest
    assert "cluster_my_jobs" in digest


def test_nonempty_cluster_runtime_digest_marks_local_only_evidence(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    meta = tmp_path / ".aether" / "runs" / "DemoProject-task" / "run-demo" / "metadata"
    meta.mkdir(parents=True)
    (meta / "run_record.json").write_text(
        (
            "{"
            "\"task_id\":\"DemoProject-task\",\"run_id\":\"run-demo\","
            f"\"run_root\":\"{(tmp_path / '.aether' / 'runs' / 'DemoProject-task' / 'run-demo').as_posix()}\","
            "\"created_at\":\"2026-01-01T00:00:00\",\"updated_at\":\"2026-01-01T00:00:00\","
            "\"overall_status\":\"running\",\"current_phase\":\"submit\","
            "\"scheduler_job_id\":\"111\",\"checkpoint_path\":\"x\",\"tags\":[],\"notes\":{},\"phases\":{}"
            "}"
        ),
        encoding="utf-8",
    )

    digest = build_cluster_runtime_digest(project="DemoProject")

    assert "111" in digest
    assert "not live scheduler evidence" in digest
    assert "before claiming current state" in digest


def test_job_watch_digest_filters_to_current_project(tmp_path: Path, monkeypatch):
    import aether_dft.paths as paths
    import aether_dft.job_watcher as job_watcher

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")

    class DemoRecord:
        task_id = "DemoProject-task"
        run_id = "run-demo"
        run_root = str(tmp_path / ".aether" / "runs" / "DemoProject-task" / "run-demo")
        scheduler_job_id = "444"
        notes = {"remote": {"remote_run_root": "/home/user/research/DemoProject/run-demo"}}

    class OtherRecord:
        task_id = "OtherProject-task"
        run_id = "run-other"
        run_root = str(tmp_path / ".aether" / "runs" / "OtherProject-task" / "run-other")
        scheduler_job_id = "555"
        notes = {"remote": {"remote_run_root": "/home/user/research/OtherProject/run-other"}}

    job_watcher.register_run_record(DemoRecord(), cluster_alias="demo")
    job_watcher.register_run_record(OtherRecord(), cluster_alias="other")

    digest = build_job_watch_digest(project="DemoProject")
    assert "444" in digest
    assert "555" not in digest
    assert "followup_goals" in digest
    assert "fixed workflow" in digest


def test_behavior_audit_flags_claim_without_evidence():
    result = ToolRegistry().run_tool(
        "behavior_audit",
        {"goal": "check", "proposed_actions": [], "proposed_reply": "计算已提交，能量为 -1.0 eV"},
    )["result"]
    assert result["status"] == "ok"
    assert any(item["code"] == "claim_without_evidence" for item in result["findings"])


class _NoToolAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:model"})()

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        return {"content": "ok", "finish_reason": "stop", "tool_calls": []}


class _MemorySessionStore:
    def ensure_session(self, session_id=None, project=None, first_prompt=None):
        return session_id or "session-1"

    def build_session_context(self, session_id):
        return ""

    def append_turn(self, session_id, record):
        return Path("memory-transcript.jsonl")


def test_harness_default_steps_do_not_keyword_route_natural_language():
    harness = AgentHarness(adapter=_NoToolAdapter(), registry=ToolRegistry(), sessions=_MemorySessionStore())
    discussion = harness.run_turn("先聊聊 H2O 在 Pt(111) 上可能怎么吸附")
    natural_execution_request = harness.run_turn("把这个 POSCAR 生成 INCAR 并准备提交集群")
    explicit_execution = harness.run_turn("[execution-mode] 把这个 POSCAR 生成 INCAR 并准备提交集群")

    assert discussion["interaction_mode"] == "discussion"
    assert discussion["max_steps_used"] == 8
    assert natural_execution_request["interaction_mode"] == "discussion"
    assert natural_execution_request["max_steps_used"] == 8
    assert explicit_execution["interaction_mode"] == "execution"
    assert explicit_execution["max_steps_used"] == 15
