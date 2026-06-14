from __future__ import annotations

from pathlib import Path

from aether_dft.agent import run_agent_once
from aether_dft.agent_tools import AetherToolRunner
from dft_app.remote.config import RemoteClusterConfig, parse_ssh_config_host
from dft_app.remote.ssh_remote_runner import SSHRemoteRunner


def test_parse_windows_ssh_config_alias(tmp_path):
    config_path = tmp_path / "config"
    config_path.write_text(
        "\n".join(
            [
                "Host szhang",
                "    HostName 59.77.33.28",
                "    User szhang",
                "    Port 22",
                r"    IdentityFile C:\Users\24651\.ssh\id_rsa_szhang",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_ssh_config_host(config_path, "szhang")

    assert parsed is not None
    assert parsed["hostname"] == "59.77.33.28"
    assert parsed["user"] == "szhang"
    assert parsed["identityfile"].endswith("id_rsa_szhang")


def test_openssh_commands_use_project_ssh_config_alias():
    config = RemoteClusterConfig(
        host="59.77.33.28",
        user="szhang",
        remote_base_dir="/home/szhang/aether-dft-runs",
        ssh_config_path=r"F:\AETHER-DFT\.secrets\ssh_config",
        ssh_host_alias="szhang",
        ssh_key_path=r"C:\Users\24651\.ssh\id_rsa_szhang",
        ignore_local_ssh_config=False,
    )

    command = SSHRemoteRunner._build_ssh_command(config, "hostname")

    assert "-F" in command
    assert r"F:\AETHER-DFT\.secrets\ssh_config" in command
    assert command[-2] == "szhang"
    assert command[-1] == "hostname"
    assert "szhang@59.77.33.28" not in command


def test_qwen_tool_surface_blocks_remote_submit_without_explicit_flag():
    runner = AetherToolRunner(allow_cluster_submit=False)

    result = runner.run(
        "adsorption_workflow_remote_submit",
        {"run_root": r"F:\AETHER-DFT\runs\task_x\run_y"},
    )

    assert result.result["status"] == "blocked"
    assert "--allow-cluster-submit" in result.result["message"]


def test_agent_loop_executes_model_requested_tool(monkeypatch, tmp_path):
    import aether_dft.paths as paths

    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")

    class FakeLLM:
        def __init__(self, app_root: Path):
            self.calls = 0

        def call_messages_with_tools(self, messages, *, tools, provider_id, model_id, max_tokens=None, tool_choice="auto"):
            self.calls += 1
            if self.calls == 1:
                return {
                    "content": "",
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "cluster_config", "arguments": "{}"},
                        }
                    ],
                }
            assert any(message.get("role") == "tool" for message in messages)
            return {
                "content": "已调用 cluster_config，当前集群配置可用于后续 SSH 探测。",
                "finish_reason": "stop",
                "tool_calls": [],
            }

    class FakeSSHRemoteRunner:
        def describe_config(self):
            return {"ssh_host_alias": "szhang", "backend": "openssh"}

    monkeypatch.setattr("aether_dft.agent.DomesticCopilotLLM", FakeLLM)
    monkeypatch.setattr("aether_dft.agent_tools.SSHRemoteRunner", FakeSSHRemoteRunner)

    record = run_agent_once("检查集群配置", model_id="bailian:qwen3.7-max")

    assert record["tool_executions"][0]["name"] == "cluster_config"
    assert record["tool_executions"][0]["result"]["config"]["ssh_host_alias"] == "szhang"
    assert "已调用 cluster_config" in record["response"]


def test_agent_loop_attaches_research_progress_and_next_step(monkeypatch, tmp_path):
    import aether_dft.paths as paths
    import aether_dft.project_state as project_state

    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(project_state, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")

    project_state.init_project("chem-demo", description="demo", overwrite=True)

    class FakeLLM:
        def __init__(self, app_root: Path):
            self.calls = 0

        def call_messages_with_tools(self, messages, *, tools, provider_id, model_id, max_tokens=None, tool_choice="auto"):
            self.calls += 1
            return {
                "content": "建议先做 slab，再生成吸附候选。",
                "finish_reason": "stop",
                "tool_calls": [],
            }

    monkeypatch.setattr("aether_dft.agent.DomesticCopilotLLM", FakeLLM)

    from aether_dft.agent import run_agent_once

    record = run_agent_once("继续推进吸附课题", project="chem-demo", model_id="bailian:qwen3.7-max")

    assert record["response"] == "建议先做 slab，再生成吸附候选。"
    assert record["progress"]["next_steps"]
    assert Path(record["project_progress_path"]).exists()
    assert "证据盘点" in record["progress"]["next_steps"][0]


def test_agent_loop_does_not_persist_meta_preload_check(monkeypatch, tmp_path):
    import aether_dft.paths as paths
    import aether_dft.project_state as project_state

    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(project_state, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")

    project_state.init_project("chem-demo", description="demo", overwrite=True)
    before = project_state.project_paths("chem-demo").progress.read_text(encoding="utf-8")

    class FakeLLM:
        def __init__(self, app_root: Path):
            pass

        def call_messages_with_tools(self, messages, *, tools, provider_id, model_id, max_tokens=None, tool_choice="auto"):
            return {
                "content": "已预加载项目设定。",
                "finish_reason": "stop",
                "tool_calls": [],
            }

    monkeypatch.setattr("aether_dft.agent.DomesticCopilotLLM", FakeLLM)

    record = run_agent_once("先不要调用工具，只用两句话说明你启动时已经预加载了哪些设定。", project="chem-demo")

    after = project_state.project_paths("chem-demo").progress.read_text(encoding="utf-8")
    assert record["progress"]["persisted"] is False
    assert record["progress"]["next_steps"] == []
    assert "project_progress_path" not in record
    assert after == before


def test_agent_loop_respects_explicit_read_only_even_with_tool_use(monkeypatch, tmp_path):
    import aether_dft.paths as paths
    import aether_dft.project_state as project_state

    monkeypatch.setattr(paths, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(paths, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setattr(paths, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(project_state, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_state, "KNOWLEDGE_BASE_DIR", tmp_path / "knowledge_base")

    project_state.init_project("chem-demo", description="demo", overwrite=True)
    before = project_state.project_paths("chem-demo").progress.read_text(encoding="utf-8")

    from aether_dft.research_loop import summarize_research_turn

    record = summarize_research_turn(
        {
            "prompt": "只读测试：可以调用只读工具，但不要写 research、不要提交。",
            "response": "已读取能力地图。",
            "finish_reason": "stop",
            "tool_executions": [{"name": "aether_capability_map", "result": {"status": "ok"}}],
        },
        project="chem-demo",
    )

    after = project_state.project_paths("chem-demo").progress.read_text(encoding="utf-8")
    assert record["progress"]["persisted"] is False
    assert record["progress"]["next_steps"] == []
    assert "project_progress_path" not in record
    assert after == before
