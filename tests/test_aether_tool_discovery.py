from __future__ import annotations

from pathlib import Path
from typing import Any

from aether_dft.runtime_harness.core import AgentHarness
from aether_dft.runtime_harness.session import HarnessSessionStore
from aether_dft.runtime_harness.tool_registry import ToolRegistry, list_capability_categories


def _tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    return {str(item["function"]["name"]) for item in tools or []}


def test_capability_map_and_discover_tools_are_first_class_tools():
    registry = ToolRegistry()
    names = {item["name"] for item in registry.list_tools()}

    assert "aether_capability_map" in names
    assert "aether_discover_tools" in names

    categories = {item["category"] for item in list_capability_categories()}
    assert {"project_memory", "structure_modeling", "dft_tasking", "cluster_runtime", "result_analysis"} <= categories

    discovered = registry.run_tool(
        "aether_discover_tools",
        {"category": "structure_modeling", "include_schemas": True, "max_tools": 30},
    )["result"]

    assert discovered["status"] == "ok"
    assert "structure_add_adsorbate" in discovered["tool_names"]
    assert any(schema["function"]["name"] == "structure_add_adsorbate" for schema in discovered["schemas"])


def test_discussion_mode_uses_lazy_schema_unlock_for_heavy_tools():
    registry = ToolRegistry()
    initial = registry.openai_tool_schemas(interaction_mode="discussion")
    initial_names = _tool_names(initial)

    assert "aether_capability_map" in initial_names
    assert "aether_discover_tools" in initial_names
    assert "project_state_read" in initial_names
    assert "cluster_profile_list" in initial_names
    assert "cluster_config" in initial_names
    assert "structure_add_adsorbate" not in initial_names

    unlocked = registry.openai_tool_schemas(
        interaction_mode="discussion",
        include_tool_names=["structure_add_adsorbate", "vasp_input_preflight_check"],
    )
    unlocked_names = _tool_names(unlocked)

    assert "structure_add_adsorbate" in unlocked_names
    assert "vasp_input_preflight_check" in unlocked_names
    assert len(unlocked_names) < len(_tool_names(registry.openai_tool_schemas()))


class DiscoveryThenAnswerAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:discover"})()

    def __init__(self):
        self.tool_batches: list[set[str]] = []

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        names = _tool_names(tools)
        self.tool_batches.append(names)
        if len(self.tool_batches) == 1:
            assert "aether_discover_tools" in names
            assert "structure_add_adsorbate" not in names
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_discover",
                        "type": "function",
                        "function": {
                            "name": "aether_discover_tools",
                            "arguments": '{"category":"structure_modeling","max_tools":30}',
                        },
                    }
                ],
            }
        assert "structure_add_adsorbate" in names
        return {"content": "已按需解锁结构建模工具，后续可由模型选择具体建模动作。", "finish_reason": "stop", "tool_calls": []}


def test_agent_harness_unlocks_discovered_tool_schemas_between_steps(tmp_path: Path):
    adapter = DiscoveryThenAnswerAdapter()
    harness = AgentHarness(adapter=adapter, registry=ToolRegistry(), sessions=HarnessSessionStore(tmp_path / "sessions"))

    record = harness.run_turn("[discussion-mode] 先聊聊 H2O 在 Pt(111) 上怎么建模", project="demo", max_steps=3)

    assert record["response"].startswith("已按需解锁结构建模工具")
    assert record["tool_executions"][0]["name"] == "aether_discover_tools"
    assert "structure_add_adsorbate" not in adapter.tool_batches[0]
    assert "structure_add_adsorbate" in adapter.tool_batches[1]


class ToolLimitFinalReplyAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:tool-limit"})()

    def __init__(self):
        self.final_tools = None
        self.final_tool_choice = None
        self.calls = 0

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call_discover",
                        "type": "function",
                        "function": {
                            "name": "aether_discover_tools",
                            "arguments": '{"category":"structure_modeling","max_tools":5}',
                        },
                    }
                ],
            }
        self.final_tools = tools
        self.final_tool_choice = tool_choice
        assert any("工具步数已经用尽" in str(message.get("content")) for message in messages if message.get("role") == "system")
        return {"content": "已发现结构建模能力，但本轮工具步数到顶；下一步应读取 slab 并枚举吸附位点。", "finish_reason": "stop", "tool_calls": []}


def test_agent_harness_finalizes_natural_reply_after_tool_limit(tmp_path: Path):
    adapter = ToolLimitFinalReplyAdapter()
    harness = AgentHarness(adapter=adapter, registry=ToolRegistry(), sessions=HarnessSessionStore(tmp_path / "sessions"))

    record = harness.run_turn("[discussion-mode] 只测试工具发现后总结", project="demo", max_steps=1)

    assert record["finish_reason"] == "tool_loop_limit_finalized"
    assert "已发现结构建模能力" in record["response"]
    assert adapter.final_tools == []
    assert adapter.final_tool_choice == "none"


class LengthThenFinalAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:length"})()

    def __init__(self):
        self.calls = 0
        self.retry_tools = None
        self.retry_tool_choice = None

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return {"content": "当前证据：H2O 以 O-down", "finish_reason": "length", "tool_calls": []}
        self.retry_tools = tools
        self.retry_tool_choice = tool_choice
        assert any("上一条回复因为 token 限制被截断" in str(message.get("content")) for message in messages if message.get("role") == "system")
        return {
            "content": "当前证据支持 O-down atop 作为主候选；关键决策是 slab 来源和候选数量；最小下一动作是确认 slab 参数后生成一个主候选。",
            "finish_reason": "stop",
            "tool_calls": [],
        }


def test_agent_harness_retries_concise_reply_after_length_finish(tmp_path: Path):
    adapter = LengthThenFinalAdapter()
    harness = AgentHarness(adapter=adapter, registry=ToolRegistry(), sessions=HarnessSessionStore(tmp_path / "sessions"))

    record = harness.run_turn("[discussion-mode] 简短回答", project="demo", max_steps=1, max_tokens=200)

    assert record["finish_reason"] == "length_finalized"
    assert "最小下一动作" in record["response"]
    assert adapter.retry_tools == []
    assert adapter.retry_tool_choice == "none"
