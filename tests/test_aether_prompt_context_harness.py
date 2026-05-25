from __future__ import annotations

from pathlib import Path

from aether_dft.context import build_context_payload, render_context_markdown
from aether_dft.harness import preflight
from aether_dft import paths, project_state
from aether_dft.prompt_engine import build_prompt_packet, render_compiled_system_prompt


def test_root_architecture_doc_is_volatile_prompt_context():
    packet = build_prompt_packet()
    prompt = packet["prompt"]
    layers = prompt["layers"]
    architecture_layer = next(item for item in layers if item["name"] == "architecture_live_doc")

    assert architecture_layer["included"] is True
    assert architecture_layer["cache_scope"] == "volatile_suffix"
    assert "智能体架构" in prompt["volatile_suffix_text"]
    assert "AETHER-DFT 智能体架构" in prompt["compiled_system_prompt"]
    assert "Step 1" in prompt["architecture_live_doc_digest_text"]
    assert "Step 2" in prompt["architecture_live_doc_digest_text"]
    assert "智能体架构" not in prompt["stable_prefix_text"]
    assert "architecture_live_doc" in prompt["volatile_layer_names"]
    assert "architecture_live_doc" not in prompt["stable_layer_names"]
    assert prompt["architecture_live_doc_path"].endswith("智能体架构.md")
    assert prompt["architecture_live_doc_digest"]


def test_architecture_doc_does_not_override_project_state_truth(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "projects"
    knowledge_dir = tmp_path / "knowledge_base"
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", knowledge_dir)
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", knowledge_dir)

    project_state.init_project("chem-demo", description="demo project", overwrite=True)
    project_state.append_progress(
        "chem-demo",
        completed=["project state truth test"],
        blockers=["none"],
        next_steps=["continue"],
    )

    packet = build_prompt_packet(project="chem-demo")
    assert "project state truth test" in packet["project_context"]
    assert "智能体架构" in packet["prompt"]["volatile_suffix_text"]
    assert "project state truth test" not in packet["prompt"]["stable_prefix_text"]
    assert packet["prompt"]["volatile_layer_names"].count("project_context") == 1
    assert packet["prompt"]["volatile_layer_names"].count("architecture_live_doc") == 1


def test_prompt_packet_compiles_aether_domain_prompt():
    packet = build_prompt_packet()
    compiled = packet["prompt"]["compiled_system_prompt"]
    assert "运行时契约" in render_compiled_system_prompt()
    assert "AETHER-DFT" in compiled
    assert "计算化学" in compiled or "DFT" in compiled
    assert "持续科研合伙人契约" in compiled
    assert "计算化学阶段图" in compiled
    assert "研究执行闭环" in compiled
    assert packet["prompt"]["runtime_contract"]
    assert packet["prompt"]["tool_policy"]
    assert len(compiled) >= len(packet["prompt"]["base_prompt"])
    layer_names = [item["name"] for item in packet["prompt"]["layers"] if item["included"]]
    assert "base_role" in layer_names
    assert "tool_discovery" in layer_names
    assert "architecture_live_doc" in layer_names
    assert packet["prompt"]["compile_projection"]["compile_strategy"] == "aether_section_compiler"


def test_context_payload_embeds_prompt_packet():
    payload = build_context_payload()
    assert payload["prompt"]["compiled_system_prompt"]
    assert "AETHER-DFT" in payload["prompt"]["compiled_system_prompt"]
    rendered = render_context_markdown(payload)
    assert "Prompt Packet" in rendered
    assert "System Prompt File" in rendered
    assert "Compiled System Prompt" in rendered
    assert "Prompt Layers" in rendered


def test_harness_preflight_reports_prompt_and_runtime_surface():
    payload = preflight()
    assert payload["prompt"]["base_prompt_length"] > 0
    assert payload["prompt"]["compiled_prompt_length"] >= payload["prompt"]["base_prompt_length"]
    assert payload["runtime"]["session_dir"]
    assert payload["runtime"]["context_dir"]


def test_project_state_markdown_is_written(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "projects"
    knowledge_dir = tmp_path / "knowledge_base"
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(paths, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", knowledge_dir)
    monkeypatch.setattr(paths, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(project_state, "PROJECTS_DIR", project_dir)
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", knowledge_dir)

    project_state.init_project("chem-demo", description="demo project", overwrite=True)
    state_md = project_state.project_paths("chem-demo").state_md
    assert state_md.exists()
    text = state_md.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "current_focus" in text
    assert "项目状态" in text
