from __future__ import annotations

import json
from pathlib import Path

from aether_dft.agent import run_agent_once
from aether_dft.agent_tools import AetherToolRunner
from dft_app.remote.config import (
    RemoteClusterConfig,
    config_for_local_cluster_alias,
    list_local_cluster_profiles,
    parse_ssh_config_host,
    parse_ssh_config_hosts,
    use_local_cluster_profile,
)
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


def test_parse_ssh_config_hosts_lists_project_clusters(tmp_path):
    config_path = tmp_path / "config"
    config_path.write_text(
        "\n".join(
            [
                "Host szhang",
                "    HostName 59.77.33.28",
                "    User szhang",
                "    IdentityFile C:\\Users\\24651\\.ssh\\id_rsa_szhang",
                "Host fghe",
                "    HostName 59.77.33.28",
                "    User fghe",
                "Host rxqin",
                "    HostName 59.77.33.28",
                "    User rxqin",
                "Host *",
                "    ServerAliveInterval 60",
            ]
        ),
        encoding="utf-8",
    )

    hosts = parse_ssh_config_hosts(config_path)

    aliases = {item["alias"] for item in hosts}
    assert aliases == {"szhang", "fghe", "rxqin"}
    assert next(item for item in hosts if item["alias"] == "szhang")["identityfile_configured"] is True


def test_parse_ssh_config_host_matches_openssh_first_value_for_duplicates(tmp_path):
    config_path = tmp_path / "config"
    config_path.write_text(
        "\n".join(
            [
                "Host fghe",
                "    HostName 59.77.33.28",
                "    User fghe",
                "Host fghe",
                "    HostName 10.26.14.64",
                "    User fghe",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_ssh_config_host(config_path, "fghe")

    assert parsed is not None
    assert parsed["hostname"] == "59.77.33.28"


def test_use_local_cluster_profile_selects_active_alias(tmp_path, monkeypatch):
    config_path = tmp_path / "ssh_config"
    profile_path = tmp_path / "cluster.local.json"
    config_path.write_text(
        "\n".join(
            [
                "Host rxqin",
                "    HostName 59.77.33.28",
                "    User rxqin",
                "    Port 22",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(RemoteClusterConfig, "_default_ssh_config_path", classmethod(lambda cls: config_path))
    monkeypatch.setattr(RemoteClusterConfig, "_local_profile_path", classmethod(lambda cls: profile_path))

    payload = use_local_cluster_profile("rxqin")

    assert payload["status"] == "ok"
    assert payload["ssh_host_alias"] == "rxqin"
    assert payload["remote_base_dir"] == "/home/rxqin/aether-dft-runs"
    listed = list_local_cluster_profiles()
    assert listed["active_alias"] == "rxqin"
    assert listed["clusters"][0]["alias"] == "rxqin"


def test_config_for_local_cluster_alias_does_not_mutate_active_alias(tmp_path, monkeypatch):
    config_path = tmp_path / "ssh_config"
    profile_path = tmp_path / "cluster.local.json"
    config_path.write_text(
        "\n".join(
            [
                "Host szhang",
                "    HostName 59.77.33.28",
                "    User szhang",
                "Host rxqin",
                "    HostName 59.77.33.28",
                "    User rxqin",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(RemoteClusterConfig, "_default_ssh_config_path", classmethod(lambda cls: config_path))
    monkeypatch.setattr(RemoteClusterConfig, "_local_profile_path", classmethod(lambda cls: profile_path))
    use_local_cluster_profile("szhang")

    config = config_for_local_cluster_alias("rxqin")

    assert config.ssh_host_alias == "rxqin"
    assert config.user == "rxqin"
    assert json.loads(profile_path.read_text(encoding="utf-8"))["ssh_host_alias"] == "szhang"


def test_registry_cluster_config_accepts_cluster_alias(tmp_path, monkeypatch):
    from aether_dft.runtime_harness.tool_registry import ToolRegistry

    config_path = tmp_path / "ssh_config"
    profile_path = tmp_path / "cluster.local.json"
    config_path.write_text(
        "\n".join(
            [
                "Host rxqin",
                "    HostName 59.77.33.28",
                "    User rxqin",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(RemoteClusterConfig, "_default_ssh_config_path", classmethod(lambda cls: config_path))
    monkeypatch.setattr(RemoteClusterConfig, "_local_profile_path", classmethod(lambda cls: profile_path))
    use_local_cluster_profile("rxqin")

    result = ToolRegistry().run_tool("cluster_config", {"cluster_alias": "rxqin"})["result"]

    assert result["status"] == "ok"
    assert result["config"]["ssh_host_alias"] == "rxqin"
    assert result["config"]["user"] == "rxqin"


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


def test_remote_cluster_config_loads_remote_potcar_roots(tmp_path, monkeypatch):
    config_path = tmp_path / "ssh_config"
    profile_path = tmp_path / "cluster.local.json"
    config_path.write_text(
        "\n".join(
            [
                "Host szhang",
                "    HostName 59.77.33.28",
                "    User szhang",
            ]
        ),
        encoding="utf-8",
    )
    profile_path.write_text(
        json.dumps(
            {
                "ssh_host_alias": "szhang",
                "ssh_config_path": str(config_path),
                "remote_base_dir": "/home/szhang/aether-dft-runs",
                "remote_potcar_roots": ["/share/paw/pbe", "/home/szhang/potcars"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(RemoteClusterConfig, "_default_ssh_config_path", classmethod(lambda cls: config_path))
    monkeypatch.setattr(RemoteClusterConfig, "_local_profile_path", classmethod(lambda cls: profile_path))

    config = RemoteClusterConfig.from_env()

    assert config.remote_potcar_roots == ("/share/paw/pbe", "/home/szhang/potcars")
    assert config.public_dict()["remote_potcar_roots"] == ["/share/paw/pbe", "/home/szhang/potcars"]


def test_remote_potcar_script_uses_mapping_without_single_root_potcar_for_multi_element():
    runner = SSHRemoteRunner()

    script = runner._build_remote_potcar_script(
        ["/share/paw/pbe"], ["Pt", "Br"], "/home/szhang/aether-dft-runs/task/run/inputs/POTCAR"
    )

    assert "POTCAR.Pt" not in script  # symbol expands on the remote side rather than hardcoding one element
    assert '"$root/POTCAR"' not in script
    assert "missing POTCAR for $sym" in script
    assert "/home/szhang/aether-dft-runs/task/run/inputs/POTCAR" in script


def test_remote_potcar_root_rejects_lateral_paths():
    runner = SSHRemoteRunner()
    config = RemoteClusterConfig(host="fake", user="szhang", remote_base_dir="/home/szhang/aether-dft-runs")

    assert runner._safe_remote_potcar_root("/share/paw", config) == "/share/paw"
    for bad in ["/etc/paw", "/home/other/paw", "/share/../etc", "/share/paw;rm -rf"]:
        try:
            runner._safe_remote_potcar_root(bad, config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe root unexpectedly accepted: {bad}")


def test_remote_submit_command_runs_inside_inputs_dir():
    runner = SSHRemoteRunner()

    command = runner._build_remote_submit_command("/home/szhang/aether-dft-runs/task/run")

    assert "cd '/home/szhang/aether-dft-runs/task/run/inputs'" in command
    assert "mkdir -p logs" in command
    assert "sbatch job.slurm" in command
    assert "sbatch inputs/job.slurm" not in command
