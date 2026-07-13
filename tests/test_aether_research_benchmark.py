from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from aether_dft.research_benchmark import (
    benchmark_case_suite,
    build_parameterized_cases,
    experiment_matrix_summary,
    reference_ablation_traces,
    reference_traces,
    score_benchmark,
    score_research_episode,
    write_benchmark_report,
)
from aether_dft.scientific_state import audit_scientific_state, normalize_scientific_state
from aether_dft.runtime_harness.tool_registry import ToolRegistry


def test_scientific_state_audit_requires_live_evidence_for_live_claims():
    state = normalize_scientific_state(
        "demo",
        {
            "research_goal": "report current job state",
            "evidence": [{"id": "local-1", "source_type": "project_record", "path": "run.json"}],
            "claims": [
                {
                    "id": "claim-1",
                    "claim": "job is running",
                    "evidence_refs": ["local-1"],
                    "requires_live_evidence": True,
                }
            ],
        },
    )
    result = audit_scientific_state(state)
    assert result["verdict"] == "needs_attention"
    assert any(item["code"] == "live_claim_without_live_evidence" for item in result["findings"])


def test_scientific_state_audit_accepts_stable_locator_references():
    result = audit_scientific_state(
        normalize_scientific_state(
            "demo",
            {
                "research_goal": "report current job state",
                "evidence": [
                    {
                        "id": "live-1",
                        "source_type": "live_cluster",
                        "locator": "benchmark://job/live-1",
                        "live": True,
                        "producer": "tool:cluster_job_status_brief",
                    }
                ],
                "claims": [
                    {
                        "id": "claim-1",
                        "claim": "job completed",
                        "evidence_refs": ["benchmark://job/live-1"],
                        "requires_live_evidence": True,
                    }
                ],
            },
        )
    )
    assert result["verdict"] == "valid"


def test_scientific_state_rejects_stale_or_model_claimed_live_evidence():
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    result = audit_scientific_state(
        normalize_scientific_state(
            "demo",
            {
                "research_goal": "report current job state",
                "evidence": [
                    {
                        "id": "live-1",
                        "source_type": "live_cluster",
                        "locator": "scheduler://job/1",
                        "live": True,
                        "producer": "model",
                        "observed_at": old,
                    }
                ],
                "claims": [
                    {
                        "id": "claim-1",
                        "claim": "job is running",
                        "evidence_refs": ["live-1"],
                        "requires_live_evidence": True,
                    }
                ],
            },
        )
    )
    codes = {item["code"] for item in result["findings"]}
    assert "untrusted_live_evidence_producer" in codes
    assert "live_claim_without_live_evidence" in codes


def test_reference_fixture_suite_passes_all_cases(tmp_path):
    result = score_benchmark(reference_traces())
    summary = result["variants"]["aether_full_reference_fixture"]
    assert result["case_count"] == 6
    assert summary["pass_rate"] == 1.0
    assert summary["mean_tool_calls"] > 0
    assert summary["unauthorized_side_effects"] == 0
    report = write_benchmark_report(result, tmp_path / "report.md")
    assert "Reference fixtures" in report.read_text(encoding="utf-8")


def test_parameterized_suite_contains_60_balanced_opaque_cases():
    cases = benchmark_case_suite("parameterized")
    categories = {}
    for case in cases:
        categories[case.category] = categories.get(case.category, 0) + 1
    assert len(cases) == 60
    assert len({case.case_id for case in cases}) == 60
    assert set(categories.values()) == {10}
    assert all(case.case_id.startswith("aether_eval_") for case in cases)
    assert all(case.user_turns and case.environment for case in cases)


def test_formal_experiment_matrix_counts_all_episodes():
    matrix = experiment_matrix_summary(
        suite="parameterized",
        model_ids=["deepseek:deepseek-v4-pro"],
        variants=["aether_full", "stateless_agent", "transcript_only", "fixed_workflow"],
        repeats=3,
        max_steps=8,
        case_timeout_seconds=180,
        shard_count=8,
    )
    assert matrix["case_count"] == 60
    assert matrix["episode_count"] == 720
    assert matrix["max_model_steps"] == 11520
    assert matrix["approx_episodes_per_shard"] == 90


def test_reference_ablation_fixtures_score_below_full_system():
    result = score_benchmark(reference_traces() + reference_ablation_traces())
    full = result["variants"]["aether_full_reference_fixture"]
    assert full["pass_rate"] == 1.0
    assert result["variants"]["stateless_reference_fixture"]["mean_score"] < full["mean_score"]
    assert result["variants"]["no_evidence_guard_reference_fixture"]["mean_score"] < full["mean_score"]
    assert result["variants"]["no_human_gate_reference_fixture"]["mean_score"] < full["mean_score"]
    assert result["variants"]["fixed_workflow_reference_fixture"]["mean_score"] < full["mean_score"]
    stateless_effect = next(
        item
        for item in result["paired_effects"]
        if item["baseline_variant"] == "stateless_reference_fixture"
    )
    assert stateless_effect["mean_score_difference"] > 0


def test_deadline_exceeded_episode_never_passes_even_with_complete_scientific_state():
    trace = reference_traces()[0]
    trace["deadline_exceeded"] = True

    result = score_research_episode(trace)

    assert result["score"] == 0.0
    assert result["passed"] is False
    assert "runtime_deadline" in result["failures"]
    assert result["diagnostics"]["deadline_exceeded"] is True


def test_benchmark_summary_aggregates_provider_token_usage():
    trace = reference_traces()[0]
    trace["input_tokens"] = 1200
    trace["output_tokens"] = 300
    result = score_benchmark([trace])
    summary = result["variants"]["aether_full_reference_fixture"]
    assert summary["input_tokens"] == 1200
    assert summary["output_tokens"] == 300


def test_unsafe_ablation_fails_cluster_side_effect_case():
    trace = next(item for item in reference_traces() if item["case_id"] == "unauthorized_cluster_side_effect")
    trace = dict(trace)
    trace["variant"] = "no_human_gate"
    trace["actions"] = [{"name": "cluster_job_cancel", "side_effect": True, "authorized": False}]
    result = score_research_episode(trace)
    assert result["passed"] is False
    assert "side_effect_safety" in result["failures"]


def test_stateless_ablation_fails_resume_case():
    trace = next(item for item in reference_traces() if item["case_id"] == "resume_after_session_break")
    trace = dict(trace)
    trace["variant"] = "stateless_agent"
    trace["final_goal"] = ""
    trace["final_memory"] = []
    trace["actions"] = []
    result = score_research_episode(trace)
    assert result["passed"] is False
    assert {"goal_continuity", "memory_retention", "required_action_completion"}.issubset(result["failures"])


def test_memory_retention_is_scored_from_observed_goal_and_claim_not_model_self_report():
    trace = next(item for item in reference_traces() if item["case_id"] == "resume_after_session_break")
    trace = dict(trace)
    trace["final_memory"] = []
    trace["final_goal"] = trace["initial_goal"]
    trace["final_state"] = {
        **trace["final_state"],
        "claims": [
            {
                "id": "continuity-claim",
                "claim": "Continue with the latest accepted candidate; its identity remains durable across the restart.",
                "evidence_refs": ["project-state"],
            }
        ],
    }

    result = score_research_episode(trace)

    assert result["metrics"]["memory_retention"] == 1.0


def test_parameterized_continuity_gold_uses_values_not_evaluator_labels():
    case = next(item for item in build_parameterized_cases() if item.category == "continuity")

    assert case.initial_goal in case.required_memory_facts
    assert case.environment["candidate"] in case.required_memory_facts
    assert "same research goal" not in case.required_memory_facts
    assert not any(item.startswith("latest accepted candidate ") for item in case.required_memory_facts)


def test_scientific_state_audit_is_available_to_the_model():
    result = ToolRegistry().run_tool(
        "scientific_state_audit",
        {
            "state": {
                "project": "demo",
                "research_goal": "inspect live job",
                "claims": [{"id": "c1", "claim": "job is running", "requires_live_evidence": True}],
            }
        },
    )["result"]
    assert result["verdict"] == "needs_attention"
    assert any(item["code"] == "claim_without_evidence" for item in result["findings"])


def test_cli_research_benchmark_reference_fixtures(tmp_path, capsys):
    from aether_dft import cli

    assert cli.main(
        [
            "benchmark",
            "research",
            "--reference-fixtures",
            "--output-dir",
            str(tmp_path / "benchmark"),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "aether_full_reference_fixture" in output
    assert (tmp_path / "benchmark" / "report.md").exists()
    assert (tmp_path / "benchmark" / "traces.jsonl").exists()
    manifest = json.loads((tmp_path / "benchmark" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_commit"]
    assert "aether_dft/research_benchmark.py" in manifest["source_sha256"]


def test_cli_live_benchmark_routes_variants_and_repeats(monkeypatch, tmp_path, capsys):
    from aether_dft import cli

    captured = {}

    def fake_live(**kwargs):
        captured.update(kwargs)
        trace = reference_traces()[0]
        trace["variant"] = "aether_full_live:deepseek:deepseek-v4-pro"
        return [trace]

    monkeypatch.setattr("aether_dft.research_benchmark_live.run_live_research_benchmark", fake_live)
    assert cli.main(
        [
            "benchmark",
            "research",
            "--live-model",
            "deepseek:deepseek-v4-pro",
            "--variant",
            "aether_full",
            "--repeats",
            "3",
            "--case",
            "resume_after_session_break",
            "--output-dir",
            str(tmp_path / "live-benchmark"),
        ]
    ) == 0
    assert captured["variant_names"] == ["aether_full"]
    assert captured["repeats"] == 3
    assert captured["case_ids"] == ["resume_after_session_break"]
    assert "aether_full_live" in capsys.readouterr().out


def test_cli_benchmark_plan_only_does_not_call_model(monkeypatch, tmp_path, capsys):
    from aether_dft import cli

    monkeypatch.setattr(
        "aether_dft.research_benchmark_live.run_live_research_benchmark",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("model must not run")),
    )
    assert cli.main(
        [
            "benchmark",
            "research",
            "--live-model",
            "deepseek:deepseek-v4-pro",
            "--suite",
            "parameterized",
            "--variant",
            "aether_full",
            "--variant",
            "stateless_agent",
            "--repeats",
            "3",
            "--plan-only",
            "--output-dir",
            str(tmp_path / "plan"),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert '"case_count": 60' in output
    assert '"episode_count": 360' in output
