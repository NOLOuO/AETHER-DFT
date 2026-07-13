from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aether_dft.research_benchmark import (
    benchmark_case_records_digest,
    build_benchmark_manifest,
    load_jsonl,
    reference_ablation_traces,
    reference_traces,
    experiment_matrix_summary,
    list_long_horizon_cases,
    recorded_case_suite,
    score_benchmark,
    write_benchmark_report,
)
from aether_dft.research_benchmark_live import run_live_research_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Score AETHER-DFT long-horizon research-agent traces.")
    parser.add_argument("--input", help="JSONL file containing recorded benchmark traces.")
    parser.add_argument(
        "--reference-fixtures",
        action="store_true",
        help="Run deterministic engineering fixtures only.",
    )
    parser.add_argument("--live-model", action="append", help="Run a real model against simulated benchmark tools.")
    parser.add_argument("--case", action="append", dest="case_ids", help="Limit live runs to selected case IDs.")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--case-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Live executable variant: aether_full/stateless_agent/transcript_only/no_evidence_guard/no_human_gate/fixed_workflow.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--suite", choices=("pilot", "parameterized"), default="pilot")
    parser.add_argument("--resume", action="store_true", help="Skip live episode keys already present in output traces.jsonl.")
    parser.add_argument("--plan-only", action="store_true", help="Print the experiment matrix without calling a model.")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-dir", default=".aether/benchmarks/latest")
    args = parser.parse_args()
    if not args.input and not args.reference_fixtures and not args.live_model:
        parser.error("provide --input, --reference-fixtures, or --live-model")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = args.variants or ["aether_full"]
    input_traces = load_jsonl(args.input) if args.input else []
    recorded_input_only = bool(args.input and not args.reference_fixtures and not args.live_model)
    suite_records = recorded_case_suite(input_traces) if recorded_input_only else list_long_horizon_cases(suite=args.suite)
    suite_path = output_dir / "case_suite.json"
    suite_path.write_text(
        json.dumps(suite_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.plan_only:
        print(
            json.dumps(
                experiment_matrix_summary(
                    suite=args.suite,
                    model_ids=args.live_model or [],
                    variants=variants,
                    repeats=args.repeats,
                    max_steps=args.max_steps,
                    case_timeout_seconds=args.case_timeout_seconds,
                    shard_count=args.shard_count,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    traces = []
    existing_trace_path = output_dir / "traces.jsonl"
    if args.resume and existing_trace_path.exists():
        traces.extend(load_jsonl(existing_trace_path))
    completed_episode_keys = {
        str(trace.get("episode_key") or "") for trace in traces if str(trace.get("episode_key") or "")
    }
    if args.reference_fixtures:
        traces.extend(reference_traces() + reference_ablation_traces())
    if args.input:
        traces.extend(input_traces)
    for model_id in args.live_model or []:
        traces.extend(
            run_live_research_benchmark(
                model_id=model_id,
                output_dir=Path(args.output_dir) / "live" / model_id.replace(":", "_"),
                case_ids=args.case_ids,
                max_steps=args.max_steps,
                max_tokens=args.max_tokens,
                case_timeout_seconds=args.case_timeout_seconds,
                variant_names=args.variants,
                repeats=args.repeats,
                suite=args.suite,
                completed_episode_keys=completed_episode_keys,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
        )
    result = score_benchmark(traces)
    traces_path = output_dir / "traces.jsonl"
    traces_path.write_text(
        "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces),
        encoding="utf-8",
    )
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_benchmark_report(result, output_dir / "report.md")
    manifest_arguments = dict(vars(args))
    if recorded_input_only:
        manifest_arguments["suite"] = "recorded_input"
    manifest = build_benchmark_manifest(
        arguments=manifest_arguments,
        source_paths=[
            "aether_dft/research_benchmark.py",
            "aether_dft/research_benchmark_live.py",
            "aether_dft/scientific_state.py",
            "aether_dft/runtime_harness/core.py",
        ],
    )
    manifest["suite_sha256"] = benchmark_case_records_digest(suite_records)
    if args.input:
        manifest["recorded_input_sha256"] = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "results": str(json_path),
                "report": str(report_path),
                "traces": str(traces_path),
                "manifest": str(manifest_path),
                "case_suite": str(suite_path),
                "summary": result["variants"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
