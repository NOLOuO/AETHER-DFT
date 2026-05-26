"""集群实时轻量查询工具测试。

测试策略：mock 掉 SSHRemoteRunner 内部命令执行，验证：
1. 命令构造正确
2. 输出解析正确
3. 集群不可达时优雅降级（status≠ok，不抛异常）
4. job_id 反查 remote_run_root 不阻挡 remote_run_root 直传
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from dft_app.remote import realtime
from dft_app.remote.config import RemoteClusterConfig


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
    result = realtime.job_tail_log(remote_run_root="/x/y", log_name="../../etc/passwd")
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
    result = realtime.job_partial_outcar(remote_run_root="/x/y")
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
    result = realtime.job_partial_outcar(remote_run_root="/x/y")
    assert result["status"] == "missing"


def test_job_progress_estimate_detects_monotonic_descent(monkeypatch):
    # 单调下降的 F= 序列
    oszicar = "\n".join(
        f"   {i}  F= -10.{i:02d}  E0= -10.{i:02d}"
        for i in range(1, 8)
    )
    fake = _FakeRunner({"OSZICAR": _FakeCommandResult(0, oszicar, "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_progress_estimate(remote_run_root="/x/y")
    assert result["status"] == "ok"
    assert result["ionic_steps_seen"] == 7
    assert result["monotonic_decreasing_tail"] is True
    assert result["oscillating"] is False


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
    result = realtime.job_progress_estimate(remote_run_root="/x/y")
    assert result["status"] == "ok"
    assert result["oscillating"] is True


def test_job_progress_estimate_partial_when_few_steps(monkeypatch):
    fake = _FakeRunner({"OSZICAR": _FakeCommandResult(0, "   1  F= -10.10", "")})
    _patch_runner(monkeypatch, fake)
    result = realtime.job_progress_estimate(remote_run_root="/x/y")
    assert result["status"] == "partial"


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
    }.issubset(names)


def test_prompt_includes_cluster_realtime_section():
    from aether_dft.prompt_engine import render_compiled_system_prompt

    rendered = render_compiled_system_prompt()
    assert "集群随时可问" in rendered
    assert "cluster_job_status_brief" in rendered
    assert "cluster_job_partial_outcar" in rendered
