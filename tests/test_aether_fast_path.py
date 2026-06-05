from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aether_dft.fast_path import FastPathResponse, dispatch_fast_path


class FakeRegistry:
    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run_tool(self, name: str, arguments: dict[str, Any] | str | None = None) -> dict[str, Any]:
        payload = dict(arguments or {}) if isinstance(arguments, dict) else {}
        self.calls.append((name, payload))
        if name == "cluster_my_jobs":
            return {
                "name": name,
                "arguments": payload,
                "result": {
                    "status": "ok",
                    "count": 2,
                    "jobs": [
                        {"job_id": "12345", "name": "relax_a", "scheduler_state": "RUNNING", "elapsed": "1:23", "node": "c001", "reason": ""},
                        {"job_id": "12346", "name": "freq_b", "scheduler_state": "PENDING", "elapsed": "0:00", "node": "", "reason": "(Priority)"},
                    ],
                },
            }
        if name == "cluster_job_status_brief":
            return {
                "name": name,
                "arguments": payload,
                "result": {"status": "ok", "job_id": payload.get("job_id"), "scheduler_state": "RUNNING", "elapsed": "1:23", "node": "c001"},
            }
        if name == "cluster_job_tail_log":
            return {
                "name": name,
                "arguments": payload,
                "result": {"status": "ok", "log_path_relative": "logs/slurm.out", "tail": "hello\nworld"},
            }
        if name == "cluster_job_partial_outcar":
            return {
                "name": name,
                "arguments": payload,
                "result": {"status": "ok", "last_toten_ev": -1.23, "accuracy_reached": False},
            }
        if name == "cluster_job_progress_estimate":
            return {
                "name": name,
                "arguments": payload,
                "result": {"status": "ok", "ionic_steps_seen": 4, "convergence_score": 0.7},
            }
        if name == "dft_run_list":
            return {
                "name": name,
                "arguments": payload,
                "result": {"status": "ok", "runs": [{"run_id": "r1", "task_id": "t1", "overall_status": "ready", "run_root": "R"}]},
            }
        raise AssertionError(f"unexpected tool: {name}")


def test_fast_path_status_overview_uses_cluster_my_jobs():
    registry = FakeRegistry()
    response = dispatch_fast_path("看看怎么样了", registry=registry)

    assert response.handled is True
    assert response.route == "my_jobs"
    assert registry.calls == [("cluster_my_jobs", {"limit": 20})]
    assert "12345" in response.text
    assert "未调用 LLM" in response.text


def test_fast_path_single_job_status_tails_log():
    registry = FakeRegistry()
    response = dispatch_fast_path("job 12345 怎么样", registry=registry)

    assert response.handled is True
    assert response.route == "job_status"
    assert [name for name, _ in registry.calls] == ["cluster_job_status_brief", "cluster_job_tail_log"]
    assert "RUNNING" in response.text
    assert "hello" in response.text


def test_fast_path_job_convergence_uses_outcar_and_progress():
    registry = FakeRegistry()
    response = dispatch_fast_path("12345 收敛了吗", registry=registry)

    assert response.handled is True
    assert response.route == "job_convergence"
    assert [name for name, _ in registry.calls] == [
        "cluster_job_status_brief",
        "cluster_job_partial_outcar",
        "cluster_job_progress_estimate",
    ]
    assert "last_toten_ev=-1.23" in response.text
    assert "convergence_score=0.7" in response.text


def test_fast_path_misses_open_ended_science_prompt():
    response = dispatch_fast_path("讨论一下 H2O 在 Pt(111) 上的吸附机理", registry=FakeRegistry())

    assert response.handled is False


def test_cli_top_level_fast_path_bypasses_parser_and_llm(monkeypatch, capsys):
    import aether_dft.fast_path as fast_path
    from aether_dft import cli

    monkeypatch.setattr(
        fast_path,
        "dispatch_fast_path",
        lambda query: FastPathResponse(True, text=f"FAST:{query}", route="test"),
    )

    assert cli.main(["看看", "怎么样了"]) == 0
    assert "FAST:看看 怎么样了" in capsys.readouterr().out
