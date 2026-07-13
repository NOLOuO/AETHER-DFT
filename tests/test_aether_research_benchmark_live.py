from __future__ import annotations

import pytest

from aether_dft.research_benchmark import LONG_HORIZON_CASES
from aether_dft.research_benchmark_live import (
    BENCHMARK_VARIANTS,
    BenchmarkSandboxRegistry,
    _benchmark_prompt,
    _benchmark_system_prompt,
    _force_real_compaction,
    run_live_research_benchmark,
)
from aether_dft.session_store import AetherSessionStore


def test_benchmark_sandbox_never_executes_real_cluster_side_effects():
    case = next(item for item in LONG_HORIZON_CASES if item.case_id == "unauthorized_cluster_side_effect")
    registry = BenchmarkSandboxRegistry(case)
    result = registry.run_tool("cluster_job_cancel", {})["result"]
    assert result["status"] == "permission_required"
    assert registry.actions[0]["authorized"] is False


def test_benchmark_sandbox_emits_typed_live_evidence():
    case = next(item for item in LONG_HORIZON_CASES if item.case_id == "stale_record_vs_live_cluster")
    registry = BenchmarkSandboxRegistry(case)
    result = registry.run_tool("cluster_job_status_brief", {})["result"]
    assert result["source_type"] == "live_cluster"
    assert result["live"] is True


def test_benchmark_sandbox_accepts_json_string_arguments():
    case = next(item for item in LONG_HORIZON_CASES if item.case_id == "human_scientific_boundary")
    registry = BenchmarkSandboxRegistry(case)
    result = registry.run_tool("auto_human_question", '{"question":"Which branch?"}')["result"]
    assert result["status"] == "waiting_for_human"
    assert registry.questions == ["Which branch?"]


def test_live_prompt_hides_evaluator_gold_fields():
    case = next(item for item in LONG_HORIZON_CASES if item.case_id == "stale_record_vs_live_cluster")
    prompt = _benchmark_prompt(case, 0, final_stage=False)
    assert case.case_id not in prompt
    assert "required_memory_facts" not in prompt
    assert "Live evidence required" not in prompt
    assert "Human question required" not in prompt
    assert "local record is not live evidence" not in prompt


def test_no_human_gate_records_attempt_and_realized_effect_separately():
    case = next(item for item in LONG_HORIZON_CASES if item.case_id == "unauthorized_cluster_side_effect")
    registry = BenchmarkSandboxRegistry(case, variant=BENCHMARK_VARIANTS["no_human_gate"])
    result = registry.run_tool("cluster_job_cancel", {"job_id": "benchmark-001"})["result"]
    assert result["status"] == "simulated_effect"
    assert registry.actions[0]["authorized"] is False
    assert registry.actions[0]["realized"] is True


def test_no_evidence_guard_removes_trusted_live_provenance():
    case = next(item for item in LONG_HORIZON_CASES if item.case_id == "stale_record_vs_live_cluster")
    registry = BenchmarkSandboxRegistry(case, variant=BENCHMARK_VARIANTS["no_evidence_guard"])
    result = registry.run_tool("cluster_job_status_brief", {})["result"]
    assert result["source_type"] == "project_record"
    assert result["live"] is False
    assert result["producer"] == "untyped"


def test_compaction_stage_uses_real_session_store(tmp_path):
    sessions = AetherSessionStore(tmp_path / "sessions")
    session_id = sessions.start_session(project="benchmark")
    sessions.append_turn(
        session_id,
        {"prompt": "accepted TS protocol", "response": "frequency validation required", "tool_executions": []},
    )
    _force_real_compaction(sessions, session_id)
    state = sessions.load_state(session_id)
    assert state["last_compact_trigger"] == "benchmark"
    assert state["compacted_turn_count"] > 0
    assert "accepted TS protocol" in state["compact_summary"]


class _LongitudinalAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:research"})()

    def chat(self, messages, *, tools=None, **_kwargs):
        names = {item["function"]["name"] for item in tools or []}
        tool_names = [str(item.get("name") or "") for item in messages if item.get("role") == "tool"]
        if "project_continuity_digest" in names and "project_continuity_digest" not in tool_names:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "digest",
                        "type": "function",
                        "function": {"name": "project_continuity_digest", "arguments": "{}"},
                    }
                ],
            }
        if "research_cycle_checkpoint" in names and "research_cycle_checkpoint" not in tool_names:
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "checkpoint",
                        "type": "function",
                        "function": {
                            "name": "research_cycle_checkpoint",
                            "arguments": (
                                '{"goal":"Identify the most stable adsorption candidate and validate it with DFT evidence.",'
                                '"accepted_facts":["same research goal","latest accepted candidate"],'
                                '"evidence_refs":["project-state"],"next_action":"validate candidate"}'
                            ),
                        },
                    }
                ],
            }
        if "benchmark_finalize" in names:
            if "benchmark_finalize" in tool_names:
                return {"content": "Episode finalized.", "finish_reason": "stop", "tool_calls": []}
            if "project_state_read" in names and "project_state_read" not in tool_names:
                return {
                    "content": "",
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "state",
                            "type": "function",
                            "function": {"name": "project_state_read", "arguments": "{}"},
                        }
                    ],
                }
            return {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "final",
                        "type": "function",
                        "function": {
                            "name": "benchmark_finalize",
                            "arguments": (
                                '{"research_goal":"Identify the most stable adsorption candidate and validate it with DFT evidence.",'
                                '"claim":"Continue validation of the latest accepted candidate.",'
                                '"evidence_refs":["project-state"],"status":"active"}'
                            ),
                        },
                    }
                ],
            }
        return {"content": "Stage state persisted.", "finish_reason": "stop", "tool_calls": []}


def test_live_runner_restarts_harness_and_scores_durable_state(monkeypatch, tmp_path):
    monkeypatch.setattr("aether_dft.research_benchmark_live._ModuleAdapter", lambda _model_id: _LongitudinalAdapter())
    traces = run_live_research_benchmark(
        model_id="fake:research",
        output_dir=tmp_path / "live",
        case_ids=["resume_after_session_break"],
        variant_names=["aether_full"],
        max_steps=5,
        case_timeout_seconds=10,
    )
    trace = traces[0]
    assert trace["longitudinal_turn_count"] == 2
    assert trace["session_reused"] is True
    assert trace["final_goal"].startswith("Identify the most stable")
    assert trace["final_memory"] == ["same research goal", "latest accepted candidate"]
    assert len(trace["record_paths"]) == 2


def test_finalize_tool_requires_exact_machine_resolvable_evidence_references():
    registry = BenchmarkSandboxRegistry(LONG_HORIZON_CASES[0])
    registry.final_stage = True

    finalize = next(
        item for item in registry.openai_tool_schemas() if item["function"]["name"] == "benchmark_finalize"
    )

    description = finalize["function"]["description"].lower()
    assert "exact evidence_id or locator" in description
    assert "do not annotate" in description


def test_cluster_tool_descriptions_teach_applicability_without_fixed_routing():
    registry = BenchmarkSandboxRegistry(LONG_HORIZON_CASES[0])
    descriptions = {
        item["function"]["name"]: item["function"]["description"].lower()
        for item in registry.openai_tool_schemas()
    }

    assert "only" in descriptions["cluster_job_status_brief"]
    assert "do not probe" in descriptions["cluster_job_status_brief"]
    assert "failed job" in descriptions["cluster_job_tail_log"]
    assert "failure evidence" in descriptions["diagnose_failure"]


def test_benchmark_system_prompt_matches_the_sandbox_tool_surface():
    prompt = _benchmark_system_prompt(project="benchmark-demo", session_context="accepted candidate h2o-atop-1")

    assert "only the tool schemas supplied" in prompt.lower()
    assert "do not emit" in prompt.lower()
    assert "accepted candidate h2o-atop-1" in prompt
    assert "structure_resolve" not in prompt


@pytest.mark.parametrize("variant_name", ["stateless_agent", "transcript_only"])
def test_non_structured_baseline_cannot_reread_project_state_after_process_boundary(variant_name):
    registry = BenchmarkSandboxRegistry(
        LONG_HORIZON_CASES[0],
        variant=BENCHMARK_VARIANTS[variant_name],
    )
    assert registry.run_tool("project_continuity_digest", {})["result"]["status"] == "ok"

    registry.final_stage = True
    result = registry.run_tool("project_continuity_digest", {})["result"]

    assert result["status"] == "unavailable"
    assert "process boundary" in result["message"]


def test_full_agent_can_reread_durable_project_state_after_process_boundary():
    registry = BenchmarkSandboxRegistry(
        LONG_HORIZON_CASES[0],
        variant=BENCHMARK_VARIANTS["aether_full"],
    )
    registry.final_stage = True

    result = registry.run_tool("project_continuity_digest", {})["result"]

    assert result["status"] == "ok"
    assert result["research_goal"] == LONG_HORIZON_CASES[0].initial_goal


def test_live_runner_skips_completed_episode_keys_without_calling_model(monkeypatch, tmp_path):
    case = LONG_HORIZON_CASES[0]
    key = f"fake:research|aether_full|1|{case.case_id}"
    monkeypatch.setattr(
        "aether_dft.research_benchmark_live._ModuleAdapter",
        lambda _model_id: (_ for _ in ()).throw(AssertionError("completed episode must be skipped")),
    )
    traces = run_live_research_benchmark(
        model_id="fake:research",
        output_dir=tmp_path / "resume",
        case_ids=[case.case_id],
        variant_names=["aether_full"],
        completed_episode_keys={key},
    )
    assert traces == []


def test_live_runner_rejects_invalid_shard_without_calling_model(tmp_path):
    with pytest.raises(ValueError, match="shard_index"):
        run_live_research_benchmark(
            model_id="fake:research",
            output_dir=tmp_path / "bad-shard",
            shard_count=4,
            shard_index=4,
        )


class _ConnectionFailingAdapter:
    runtime = type("Runtime", (), {"model_id": "fake:connection-failure"})()

    def chat(self, *_args, **_kwargs):
        raise RuntimeError("provider connection error")


def test_live_runner_records_provider_failures_and_continues_to_next_case(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "aether_dft.research_benchmark_live._ModuleAdapter",
        lambda _model_id: _ConnectionFailingAdapter(),
    )
    case_ids = [LONG_HORIZON_CASES[0].case_id, LONG_HORIZON_CASES[1].case_id]

    traces = run_live_research_benchmark(
        model_id="fake:connection-failure",
        output_dir=tmp_path / "provider-failures",
        case_ids=case_ids,
        max_steps=2,
        case_timeout_seconds=10,
    )

    assert [trace["case_id"] for trace in traces] == case_ids
    assert all(trace["provider_error"] is True for trace in traces)
    assert all(trace["finish_reason"] == "model_provider_error" for trace in traces)
