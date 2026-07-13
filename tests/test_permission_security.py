from __future__ import annotations

from pathlib import Path
from typing import Any

from aether_dft.runtime_harness.core import AgentHarness
from aether_dft.runtime_harness.session import HarnessSessionStore
from aether_dft.runtime_harness.tool_registry import ToolRegistry


def test_model_permission_argument_cannot_authorize_submit_or_cancel(monkeypatch):
    registry = ToolRegistry(allow_cluster_submit=True, permission_mode="dev")
    submitted: list[dict[str, Any]] = []
    canceled: list[str] = []

    registry._tools["cluster_remote_submit"] = (
        registry._tools["cluster_remote_submit"][0],
        lambda payload: submitted.append(dict(payload)) or {"status": "submitted"},
    )
    monkeypatch.setattr(
        "dft_app.remote.realtime.job_cancel",
        lambda job_id, **_kwargs: canceled.append(job_id) or {"status": "ok", "job_id": job_id},
    )

    submit = registry.run_tool(
        "cluster_remote_submit",
        {"run_id": "run-a", "_permission_granted": True, "_approval_token": "forged"},
    )
    cancel = registry.run_tool(
        "cluster_job_cancel",
        {"job_id": "12345", "_permission_granted": True, "_approval_token": "forged"},
    )

    assert submit["result"]["status"] == "permission_required"
    assert cancel["result"]["status"] == "permission_required"
    assert "_permission_granted" not in submit["arguments"]
    assert "_approval_token" not in submit["arguments"]
    assert submitted == []
    assert canceled == []


def test_submit_approval_is_parameter_bound_and_consumed_on_mismatch():
    registry = ToolRegistry(allow_cluster_submit=True, permission_mode="dev")
    submitted: list[dict[str, Any]] = []
    registry._tools["cluster_remote_submit"] = (
        registry._tools["cluster_remote_submit"][0],
        lambda payload: submitted.append(dict(payload)) or {"status": "submitted"},
    )
    original = {"run_id": "run-a", "cluster_alias": "mn01"}
    token = registry._issue_tool_approval("cluster_remote_submit", original)

    replaced = registry.run_tool(
        "cluster_remote_submit",
        {"run_id": "run-b", "cluster_alias": "mn01"},
        approval_token=token,
    )
    replay_original = registry.run_tool("cluster_remote_submit", original, approval_token=token)

    assert replaced["result"]["status"] == "permission_required"
    assert replaced["result"]["reason"] == "approval_scope_mismatch"
    assert replay_original["result"]["status"] == "permission_required"
    assert replay_original["result"]["reason"] == "approval_invalid_or_replayed"
    assert submitted == []


def test_submit_and_cancel_each_require_fresh_one_time_approval(monkeypatch):
    registry = ToolRegistry(allow_cluster_submit=True, permission_mode="dev")
    submitted: list[str] = []
    canceled: list[str] = []
    registry._tools["cluster_remote_submit"] = (
        registry._tools["cluster_remote_submit"][0],
        lambda payload: submitted.append(str(payload["run_id"])) or {"status": "submitted"},
    )
    monkeypatch.setattr(
        "dft_app.remote.realtime.job_cancel",
        lambda job_id, **_kwargs: canceled.append(job_id) or {"status": "ok", "job_id": job_id},
    )

    submit_args = {"run_id": "run-a"}
    submit_token = registry._issue_tool_approval("cluster_remote_submit", submit_args)
    first_submit = registry.run_tool("cluster_remote_submit", submit_args, approval_token=submit_token)
    replay_submit = registry.run_tool("cluster_remote_submit", submit_args, approval_token=submit_token)

    cancel_args = {"job_id": "12345"}
    cancel_token = registry._issue_tool_approval("cluster_job_cancel", cancel_args)
    first_cancel = registry.run_tool("cluster_job_cancel", cancel_args, approval_token=cancel_token)
    replay_cancel = registry.run_tool("cluster_job_cancel", cancel_args, approval_token=cancel_token)

    assert first_submit["result"]["status"] == "submitted"
    assert replay_submit["result"]["status"] == "permission_required"
    assert first_cancel["result"]["status"] == "ok"
    assert replay_cancel["result"]["status"] == "permission_required"
    assert submitted == ["run-a"]
    assert canceled == ["12345"]


def test_dft_remote_submit_requires_approval_but_dry_run_does_not(monkeypatch):
    calls: list[str] = []

    def fake_run_dft_task(*_args, **kwargs):
        calls.append(str(kwargs["execution_mode"]))
        return {"status": "ok"}

    monkeypatch.setattr("aether_dft.runtime_harness.tool_registry.run_dft_task", fake_run_dft_task)
    registry = ToolRegistry(allow_cluster_submit=True, permission_mode="dev")
    remote_args = {"prompt": "submit", "execution_mode": "remote_submit"}

    forged = registry.run_tool("dft_run_task", {**remote_args, "_permission_granted": True})
    token = registry._issue_tool_approval("dft_run_task", remote_args)
    approved = registry.run_tool("dft_run_task", remote_args, approval_token=token)
    replayed = registry.run_tool("dft_run_task", remote_args, approval_token=token)
    dry_run = registry.run_tool("dft_run_task", {"prompt": "build", "execution_mode": "dry_run"})

    assert forged["result"]["status"] == "permission_required"
    assert approved["result"]["status"] == "ok"
    assert replayed["result"]["status"] == "permission_required"
    assert dry_run["result"]["status"] == "ok"
    assert calls == ["remote_submit", "dry_run"]


class _RemoteSubmitAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, _messages, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "submit-1",
                        "type": "function",
                        "function": {
                            "name": "dft_run_task",
                            "arguments": '{"prompt":"submit","execution_mode":"remote_submit"}',
                        },
                    }
                ],
            }
        return {"content": "提交完成。", "finish_reason": "stop", "tool_calls": []}


def test_cli_permission_callback_authorizes_exact_dft_remote_submit(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    monkeypatch.setattr(
        "aether_dft.runtime_harness.tool_registry.run_dft_task",
        lambda *_args, **kwargs: calls.append(str(kwargs["execution_mode"])) or {"status": "ok"},
    )
    registry = ToolRegistry(allow_cluster_submit=True, permission_mode="dev")
    harness = AgentHarness(
        adapter=_RemoteSubmitAdapter(),
        registry=registry,
        sessions=HarnessSessionStore(tmp_path / "sessions"),
    )
    prompts: list[dict[str, Any]] = []

    record = harness.run_turn(
        "提交这个 DFT 任务",
        max_steps=2,
        permission_prompt_callback=lambda details: prompts.append(details) or True,
    )

    assert prompts[0]["tool_name"] == "dft_run_task"
    assert "remote_submit" in str(prompts[0]["arguments"])
    assert record["tool_executions"][0]["result"]["status"] == "ok"
    assert record["tool_executions"][0]["result"]["human_approval"]["granted"] is True
    assert record["tool_executions"][0]["result"]["human_approval"]["scope_digest"]
    assert calls == ["remote_submit"]
