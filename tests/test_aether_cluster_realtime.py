"""集群实时轻量查询工具测试。

测试策略：mock 掉 SSHRemoteRunner 内部命令执行，验证：
1. 命令构造正确
2. 输出解析正确
3. 集群不可达时优雅降级（status≠ok，不抛异常）
4. job_id 反查 remote_run_root 不阻挡 remote_run_root 直传
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dft_app.remote import realtime
from dft_app.remote.config import RemoteClusterConfig
from dft_app.remote.ssh_remote_runner import SSHRemoteRunner
from dft_app.models import RunRecord


@dataclass
class _FakeCommandResult:
    returncode: int
    stdout: str
    stderr: str


class _FakeRunner:
    """最小可控的 SSHRemoteRunner 替身。"""

    def __init__(self, responses: dict[str, _FakeCommandResult] | None = None, fail_tools: bool = False):
        self.config = RemoteClusterConfig(
            host="fake.host",
            user="tester",
            port=22,
            remote_base_dir="/home/tester/runs",
            backend="openssh",
        )
        self.calls: list[str] = []
        self._responses = responses or {}
        self._default = _FakeCommandResult(0, "", "")
        self._fail_tools = fail_tools

    def _load_config(self):
        return self.config

    def _select_backend(self, _config):
        return "openssh"

    def _ensure_local_tools(self, _config, _backend):
        return "ssh 缺失" if self._fail_tools else None

    def _run_remote_command(self, _config, command: str, *, timeout: int, backend: str):
        self.calls.append(command)
        for key, response in self._responses.items():
            if key in command:
                return response
        return self._default


def _patch_runner(monkeypatch, fake: _FakeRunner) -> None:
    monkeypatch.setattr(realtime, "_runner", lambda config=None: fake)


def test_job_status_brief_parses_squeue_output(monkeypatch):
    fake = _FakeRunner({"squeue -j 12345": _FakeCommandResult(0, "RUNNING|0:32:11|node07|None", "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_status_brief("12345")
    assert result["status"] == "ok"
    assert result["scheduler_state"] == "RUNNING"
    assert result["active"] is True
    assert result["elapsed"] == "0:32:11"
    assert result["node"] == "node07"
    assert result["source"] == "squeue"


def test_job_status_brief_falls_back_to_sacct(monkeypatch):
    fake = _FakeRunner(
        {
            "squeue -j 99999": _FakeCommandResult(0, "", ""),
            "sacct -j 99999": _FakeCommandResult(0, "99999|COMPLETED|0:42:00|node07|0:0", ""),
        }
    )
    _patch_runner(monkeypatch, fake)
    result = realtime.job_status_brief("99999")
    assert result["status"] == "ok"
    assert result["scheduler_state"] == "COMPLETED"
    assert result["active"] is False
    assert result["source"] == "sacct"


def test_job_status_brief_unknown_when_both_fail(monkeypatch):
    fake = _FakeRunner(
        {
            "squeue -j 11111": _FakeCommandResult(0, "", ""),
            "sacct -j 11111": _FakeCommandResult(0, "", ""),
        }
    )
    _patch_runner(monkeypatch, fake)
    result = realtime.job_status_brief("11111")
    assert result["status"] == "unknown"


def test_job_status_brief_graceful_when_tools_missing(monkeypatch):
    fake = _FakeRunner(fail_tools=True)
    _patch_runner(monkeypatch, fake)
    result = realtime.job_status_brief("12345")
    assert result["status"] == "error"
    assert "缺失" in result["message"]


def test_job_status_brief_rejects_empty_job_id(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    assert realtime.job_status_brief("")["status"] == "error"


def test_job_status_brief_rejects_unsafe_job_id(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_status_brief("123; rm -rf /")
    assert result["status"] == "error"
    assert fake.calls == []


def test_job_cancel_calls_scancel_and_verifies_absent(monkeypatch):
    fake = _FakeRunner(
        {
            "scancel 12345": _FakeCommandResult(0, "", ""),
            "squeue -j 12345": _FakeCommandResult(0, "", ""),
        }
    )
    _patch_runner(monkeypatch, fake)
    result = realtime.job_cancel("12345")
    assert result["status"] == "canceled"
    assert result["verified_absent_from_squeue"] is True
    assert any(call.startswith("scancel 12345") for call in fake.calls)
    assert any(call.startswith("squeue -j 12345") for call in fake.calls)


def test_job_cancel_rejects_unsafe_job_id(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_cancel("12345; scancel --me")
    assert result["status"] == "error"
    assert fake.calls == []


def test_my_jobs_parses_squeue_me(monkeypatch):
    stdout = "\n".join(
        [
            "12345|h2o_relax|RUNNING|0:32:11|node07|None",
            "12346|co_relax|PENDING|0:00:00|(Priority)|(Priority)",
        ]
    )
    fake = _FakeRunner({"squeue --me": _FakeCommandResult(0, stdout, "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.my_jobs()
    assert result["status"] == "ok"
    assert result["count"] == 2
    assert result["jobs"][0]["job_id"] == "12345"
    assert result["jobs"][0]["scheduler_state"] == "RUNNING"
    assert result["jobs"][1]["scheduler_state"] == "PENDING"


def test_job_tail_log_uses_explicit_remote_run_root(monkeypatch):
    body = "__AETHER_LOG_PATH__=vasp.out\nline1\nline2\nline3"
    fake = _FakeRunner({"tail -n 10": _FakeCommandResult(0, body, "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="/home/tester/runs/x/y", lines=10)
    assert result["status"] == "ok"
    assert result["log_path_relative"] == "vasp.out"
    assert result["lines_returned"] == 3
    assert "line2" in result["tail"]


def test_job_tail_log_missing_file_returns_status_missing(monkeypatch):
    fake = _FakeRunner({"tail -n": _FakeCommandResult(0, "", "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="/home/tester/runs/x/y")
    assert result["status"] == "missing"


def test_job_tail_log_without_job_or_root_is_unavailable(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log()
    assert result["status"] == "unavailable"


def test_job_tail_log_rejects_unsafe_log_name(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="/home/tester/runs/x/y", log_name="../../etc/passwd")
    assert result["status"] == "error"
    assert fake.calls == []


def test_job_partial_outcar_parses_energy_and_force(monkeypatch):
    outcar = """
 -----------------------------------------
 Iteration   12(   5)
 -----------------------------------------
 free  energy   TOTEN  =      -123.456789 eV
 TOTEN  =      -123.456789 eV
 FORCES: max atom, RMS    0.012345    0.003210
 reached required accuracy
""".strip()
    fake = _FakeRunner({"OUTCAR": _FakeCommandResult(0, outcar, "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_partial_outcar(remote_run_root="/home/tester/runs/x/y")
    assert result["status"] == "ok"
    assert result["last_toten_ev"] == pytest.approx(-123.456789)
    assert result["last_free_energy_ev"] == pytest.approx(-123.456789)
    assert result["max_force_ev_a"] == pytest.approx(0.012345)
    assert result["rms_force_ev_a"] == pytest.approx(0.003210)
    assert result["last_ionic_step"] == 12
    assert result["last_scf_iter_within_step"] == 5
    assert result["accuracy_reached"] is True


def test_job_partial_outcar_missing(monkeypatch):
    fake = _FakeRunner({"OUTCAR": _FakeCommandResult(0, "__AETHER_NO_OUTCAR__", "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_partial_outcar(remote_run_root="/home/tester/runs/x/y")
    assert result["status"] == "missing"


def test_job_progress_estimate_detects_monotonic_descent(monkeypatch):
    # 单调下降的 F= 序列
    oszicar = "\n".join(
        f"   {i}  F= -10.{i:02d}  E0= -10.{i:02d}"
        for i in range(1, 8)
    )
    fake = _FakeRunner({"OSZICAR": _FakeCommandResult(0, oszicar, "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_progress_estimate(remote_run_root="/home/tester/runs/x/y")
    assert result["status"] == "ok"
    assert result["ionic_steps_seen"] == 7
    assert result["monotonic_decreasing_tail"] is True
    assert result["oscillating"] is False


def test_job_progress_estimate_parses_vasp_scientific_notation_without_leading_digit(monkeypatch):
    oszicar = "\n".join(
        [
            "  10 F= -.72808730E+03 E0= -.72808730E+03  d E =-.303067E-03",
            "  11 F= -.72808752E+03 E0= -.72808752E+03  d E =-.219838E-03",
        ]
    )
    fake = _FakeRunner({"OSZICAR": _FakeCommandResult(0, oszicar, "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_progress_estimate(remote_run_root="/home/tester/runs/x/y")
    assert result["status"] == "ok"
    assert result["ionic_steps_seen"] == 2
    assert result["last_energy_ev"] == -728.08752
    assert result["last_delta_ev"] == pytest.approx(-0.00022)


def test_job_progress_estimate_detects_oscillation(monkeypatch):
    # 震荡序列
    oszicar = "\n".join(
        [
            "   1  F= -10.10",
            "   2  F= -10.05",  # 上升
            "   3  F= -10.20",  # 下降
            "   4  F= -10.10",  # 上升
            "   5  F= -10.25",  # 下降
        ]
    )
    fake = _FakeRunner({"OSZICAR": _FakeCommandResult(0, oszicar, "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_progress_estimate(remote_run_root="/home/tester/runs/x/y")
    assert result["status"] == "ok"
    assert result["oscillating"] is True


def test_job_progress_estimate_partial_when_few_steps(monkeypatch):
    fake = _FakeRunner({"OSZICAR": _FakeCommandResult(0, "   1  F= -10.10", "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_progress_estimate(remote_run_root="/home/tester/runs/x/y")
    assert result["status"] == "partial"


def test_job_tail_log_rejects_unsafe_remote_run_root(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="/home/tester/runs/x/../../etc")
    assert result["status"] == "error"
    assert ".." in result["message"]
    assert fake.calls == []


def test_job_tail_log_rejects_shell_metachars_in_remote_run_root(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="/home/tester/runs/x/y; rm -rf /")
    assert result["status"] == "error"
    assert fake.calls == []


def test_job_tail_log_rejects_out_of_scope_remote_run_root(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="/etc")
    assert result["status"] == "error"
    assert "允许范围" in result["message"]
    assert fake.calls == []


def test_job_tail_log_allows_tilde_under_current_user(monkeypatch):
    body = "__AETHER_LOG_PATH__=vasp.out\nok"
    fake = _FakeRunner({"tail -n 1": _FakeCommandResult(0, body, "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="~/runs/x/y", lines=1)
    assert result["status"] == "ok"
    assert result["remote_run_root"] == "/home/tester/runs/x/y"


def test_job_tail_log_allows_cluster_research_workspace(monkeypatch):
    body = "__AETHER_LOG_PATH__=OUTCAR\nok"
    fake = _FakeRunner({"tail -n 1": _FakeCommandResult(0, body, "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="~/research/MCH-Pt-Br/calc", log_name="OUTCAR", lines=1)
    assert result["status"] == "ok"
    assert result["remote_run_root"] == "/home/tester/research/MCH-Pt-Br/calc"


def test_job_tail_log_allows_unicode_research_workspace(monkeypatch):
    body = "__AETHER_LOG_PATH__=OUTCAR\nok"
    fake = _FakeRunner({"tail -n 1": _FakeCommandResult(0, body, "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="~/research/MCH-Pt-Br/微观动力学/calc", log_name="OUTCAR", lines=1)
    assert result["status"] == "ok"
    assert result["remote_run_root"] == "/home/tester/research/MCH-Pt-Br/微观动力学/calc"


def test_job_tail_log_rejects_tilde_outside_remote_base(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="~/.ssh", log_name="id_ed25519")
    assert result["status"] == "error"
    assert fake.calls == []


def test_job_partial_outcar_rejects_unsafe_remote_run_root(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_partial_outcar(remote_run_root="/home/tester/runs/x/y; touch /tmp/pwn")
    assert result["status"] == "error"
    assert fake.calls == []


def test_job_progress_estimate_rejects_unsafe_remote_run_root(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_progress_estimate(remote_run_root="/home/tester/runs/x/../../etc/passwd")
    assert result["status"] == "error"
    assert fake.calls == []


def test_my_jobs_invalid_limit_returns_error(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.my_jobs(limit="abc")  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "limit" in result["message"]
    assert fake.calls == []


def test_my_jobs_clamps_limit_to_safe_range(monkeypatch):
    fake = _FakeRunner({"squeue --me": _FakeCommandResult(0, "", "")})
    _patch_runner(monkeypatch, fake)
    realtime.my_jobs(limit=99999)
    # head -n 应被夹到 200 上限
    assert any("head -n 200" in call for call in fake.calls)


def test_job_tail_log_invalid_lines_returns_error(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="/home/tester/runs/x/y", lines="not-int")  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "lines" in result["message"]
    assert fake.calls == []


def test_job_tail_log_rejects_log_name_with_slash(monkeypatch):
    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    result = realtime.job_tail_log(remote_run_root="/home/tester/runs/x/y", log_name="logs/foo")
    assert result["status"] == "error"
    assert fake.calls == []


def test_tools_registered_in_registry():
    from aether_dft.runtime_harness.tool_registry import ToolRegistry

    registry = ToolRegistry()
    names = {tool["name"] for tool in registry.list_tools()}
    assert {
        "cluster_job_status_brief",
        "cluster_my_jobs",
        "cluster_job_tail_log",
        "cluster_job_partial_outcar",
        "cluster_job_progress_estimate",
        "cluster_job_cancel",
    }.issubset(names)

    cancel_spec = next(tool for tool in registry.list_tools() if tool["name"] == "cluster_job_cancel")
    assert cancel_spec["read_only"] is False
    assert cancel_spec["explicit_human_required"] is True


def test_registry_cluster_job_cancel_requires_explicit_human_permission_even_in_dev(monkeypatch):
    from aether_dft.runtime_harness.tool_registry import ToolRegistry

    fake = _FakeRunner({"scancel 12345": _FakeCommandResult(0, "", "")})
    _patch_runner(monkeypatch, fake)
    result = ToolRegistry(permission_mode="dev").run_tool("cluster_job_cancel", {"job_id": "12345"})

    assert result["result"]["status"] == "permission_required"
    assert result["result"]["reason"] == "explicit_human_required"
    assert fake.calls == []


def test_registry_cluster_job_cancel_rejects_model_forged_permission(monkeypatch):
    from aether_dft.runtime_harness.tool_registry import ToolRegistry

    fake = _FakeRunner(
        {
            "scancel 12345": _FakeCommandResult(0, "", ""),
            "squeue -j 12345": _FakeCommandResult(0, "", ""),
        }
    )
    _patch_runner(monkeypatch, fake)
    result = ToolRegistry(permission_mode="dev").run_tool(
        "cluster_job_cancel",
        {"job_id": "12345", "_permission_granted": True},
    )

    assert result["result"]["status"] == "permission_required"
    assert fake.calls == []


def test_registry_realtime_handlers_return_errors_instead_of_raising(monkeypatch):
    from aether_dft.runtime_harness.tool_registry import ToolRegistry

    fake = _FakeRunner()
    _patch_runner(monkeypatch, fake)
    registry = ToolRegistry()

    result = registry.run_tool("cluster_my_jobs", {"limit": "abc"})["result"]
    assert result["status"] == "error"
    assert "limit" in result["message"]

    result = registry.run_tool(
        "cluster_job_tail_log",
        {"remote_run_root": "/home/tester/runs/x/y", "lines": "abc"},
    )["result"]
    assert result["status"] == "error"
    assert "lines" in result["message"]


def test_prompt_includes_cluster_realtime_section():
    from aether_dft.prompt_engine import render_compiled_system_prompt

    rendered = render_compiled_system_prompt()
    assert "集群随时可问" in rendered
    assert "cluster_job_status_brief" in rendered
    assert "cluster_job_partial_outcar" in rendered
    assert "没有调用 `cluster_my_jobs`" in rendered
    assert "不要说“集群没有任务”" in rendered


def test_ssh_runner_safe_extract_tar_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"pwn"
        info = tarfile.TarInfo("../evil.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(ValueError):
            SSHRemoteRunner._safe_extract_tar(tar, tmp_path / "out")

    assert not (tmp_path / "evil.txt").exists()


def test_ssh_runner_safe_extract_tar_enforces_allowlist(tmp_path):
    archive = tmp_path / "extra.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("project/allowed.md", "project/extra.md"):
            payload = name.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(ValueError):
            SSHRemoteRunner._safe_extract_tar(
                tar,
                tmp_path / "out",
                allowed_members={"project/allowed.md"},
            )


class _SSHRunnerNoRemote(SSHRemoteRunner):
    def __init__(self):
        super().__init__(
            RemoteClusterConfig(
                host="fake.host",
                user="tester",
                port=22,
                remote_base_dir="/home/tester/runs",
                backend="openssh",
            )
        )
        self.calls: list[str] = []

    def _ensure_local_tools(self, _config, _backend):
        return None

    def _run_remote_command(self, _config, command: str, *, timeout: int, backend: str):
        self.calls.append(command)
        return _FakeCommandResult(0, "", "")


def test_research_status_rejects_out_of_scope_remote_dir(tmp_path):
    runner = _SSHRunnerNoRemote()
    result = runner.research_status(Path(tmp_path), remote_research_dir="/etc")
    assert result.status == "blocked"
    assert "remote_research_dir" in result.message
    assert runner.calls == []


def test_pull_remote_run_outputs_rejects_out_of_scope_remote_root(tmp_path):
    runner = _SSHRunnerNoRemote()
    result = runner.pull_remote_run_outputs("/home/otheruser/secret", tmp_path)
    assert result.status == "blocked"
    assert "remote_run_root" in result.message
    assert runner.calls == []


def test_find_remote_outcars_parses_safe_research_tree():
    class Finder(_SSHRunnerNoRemote):
        def _run_remote_command(self, _config, command: str, *, timeout: int, backend: str):
            self.calls.append(command)
            return _FakeCommandResult(
                0,
                "2026-06-06 09:10|12345|/home/tester/research/proj/run/OUTCAR\n",
                "",
            )

    runner = Finder()
    result = runner.find_remote_outcars(search_root="~/research", limit=3, max_depth=5)
    assert result.status == "ok"
    assert result.details["outcars"][0]["path"] == "/home/tester/research/proj/run/OUTCAR"
    assert result.details["outcars"][0]["run_root"] == "/home/tester/research/proj/run"
    assert "find '/home/tester/research'" in runner.calls[0]
    assert "-maxdepth 5" in runner.calls[0]
    assert "head -n 3" in runner.calls[0]


def test_find_remote_outcars_rejects_out_of_scope_root():
    runner = _SSHRunnerNoRemote()
    result = runner.find_remote_outcars(search_root="/etc", limit=1)
    assert result.status == "blocked"
    assert "search_root" in result.message
    assert runner.calls == []


def test_pull_remote_outcar_context_rejects_non_outcar_name(tmp_path):
    runner = _SSHRunnerNoRemote()
    result = runner.pull_remote_outcar_context("/home/tester/research/proj/run/CONTCAR", tmp_path)
    assert result.status == "blocked"
    assert "OUTCAR" in result.message
    assert runner.calls == []


def test_pull_remote_outcar_context_downloads_neighboring_evidence(tmp_path):
    class Puller(_SSHRunnerNoRemote):
        def _run_remote_command(self, _config, command: str, *, timeout: int, backend: str):
            self.calls.append(command)
            return _FakeCommandResult(0, "OUTCAR\nOSZICAR\nPOSCAR\n", "")

        def _download_from_remote(self, _config, remote_path: str, local_path: Path, *, timeout: int, backend: str):
            local_path.write_text(f"copied {remote_path}", encoding="utf-8")

    runner = Puller()
    result = runner.pull_remote_outcar_context("/home/tester/research/proj/run/OUTCAR", tmp_path)
    assert result.status == "synced"
    assert (tmp_path / "OUTCAR").read_text(encoding="utf-8").endswith("/home/tester/research/proj/run/OUTCAR")
    assert (tmp_path / "OSZICAR").exists()
    assert len(result.details["downloaded"]) == 3


def test_monitor_rejects_unsafe_scheduler_job_id(tmp_path):
    runner = _SSHRunnerNoRemote()
    record = RunRecord(
        task_id="task",
        run_id="run",
        run_root=str(tmp_path / "run"),
        checkpoint_path=str(tmp_path / "run" / "checkpoint.json"),
        scheduler_job_id="123; touch /tmp/pwn",
        notes={"remote": {"remote_run_root": "/home/tester/runs/task/run"}},
    )

    result = runner.monitor(record)

    assert result.status == "blocked"
    assert "scheduler_job_id" in result.message
    assert runner.calls == []
