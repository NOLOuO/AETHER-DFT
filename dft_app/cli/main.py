from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from dft_app.cluster_profiles import SUBMIT_PROFILES
from dft_app.modeling import (
    AdsorbateStructureFactory,
    AdsorptionCandidateGenerator,
    AdsorptionGenerationRequest,
    CandidateManifest,
    CandidateManifestWriter,
    ConfirmedCandidateHandoff,
    TaskModeler,
    model_spec_from_dict,
)
from dft_app.models import (
    ExperimentPlan,
    ExperimentSpec,
    PhaseStatus,
    PipelinePhase,
    RunRecord,
    RunStatus,
    StructureSource,
    TaskType,
)
from dft_app.planner import AutoPlanner, PlanningResult
from dft_app.storage import RecordStore


SUPPORTED_TASK_TYPES = [task_type.value for task_type in TaskType]
SUPPORTED_STEP_PHASES = [phase.value for phase in PipelinePhase]
SUPPORTED_SUBMIT_PROFILES = sorted(SUBMIT_PROFILES)
ADSORPTION_WORKFLOW_SUBTASKS = ("clean_slab", "isolated_adsorbate", "adsorbed_system")
ADSORPTION_OUTPUT_FILES = ("vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR", "vasp.out", "slurm.out")
PLANNER = AutoPlanner()
MODELER = TaskModeler()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE = RecordStore(PROJECT_ROOT)


def ensure_console_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def get_builder():
    from dft_app.builder import WorkspaceBuilder

    return WorkspaceBuilder(PROJECT_ROOT)


def get_runner():
    from dft_app.runner import SlurmRunner

    return SlurmRunner()


def get_output_parser():
    from dft_app.parser import VaspOutputParser

    return VaspOutputParser()


def get_analyzer():
    from dft_app.analyzer import MarkdownReportAnalyzer

    return MarkdownReportAnalyzer()


def get_exporter():
    from dft_app.exporter import DeliverableExporter

    return DeliverableExporter()


def get_remote_runner():
    from dft_app.remote import SSHRemoteRunner

    return SSHRemoteRunner()


def get_orchestrator():
    from dft_app.orchestrator import ComplexWorkflowOrchestrator

    return ComplexWorkflowOrchestrator(PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dft",
        description="Semi-automatic DFT platform CLI skeleton.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the main workflow.")
    run_parser.add_argument("prompt", nargs="?", help="Natural language task prompt.")
    run_parser.add_argument(
        "--task-type",
        choices=SUPPORTED_TASK_TYPES,
        default=None,
        help="Task type for the first version.",
    )
    run_parser.add_argument("--material", help="Material name.")
    run_parser.add_argument("--structure-path", help="Local structure file path.")
    run_parser.add_argument(
        "--submit-profile",
        choices=SUPPORTED_SUBMIT_PROFILES,
        help="Cluster submit profile, for example c32 or b96.",
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Only print plan.")
    run_parser.add_argument(
        "--submit", action="store_true", help="After build, attempt Slurm submission."
    )
    run_parser.add_argument(
        "--remote", action="store_true", help="Use SSH remote submission instead of local sbatch."
    )
    run_parser.add_argument(
        "--selected-candidate-id",
        help="For adsorption complex tasks, select a generated candidate id and continue into builder/submit in the same mainline command.",
    )
    run_parser.add_argument("--candidate-id", dest="candidate_id", help="Alias for the selected Step 2 candidate id used for lineage.")
    run_parser.add_argument("--model-spec-path", help="Step 2 model_spec.json to preserve model-authored build evidence.")
    run_parser.add_argument("--step2-manifest-path", help="Step 2 candidate/structure manifest path to preserve lineage.")
    run_parser.add_argument(
        "--status", action="store_true", help="Show workflow status placeholder."
    )
    run_parser.add_argument(
        "--reset", action="store_true", help="Reset workflow status placeholder."
    )
    run_parser.add_argument("--run-id", help="Target run id for status/reset.")
    run_parser.add_argument("--run-root", help="Target run root for status/reset.")

    step_parser = subparsers.add_parser("step", help="Run a single workflow phase.")
    step_parser.add_argument("phase", choices=SUPPORTED_STEP_PHASES)
    step_parser.add_argument("--run-id", help="Target run id.")
    step_parser.add_argument("--run-root", help="Target run root.")
    step_parser.add_argument(
        "--remote", action="store_true", help="Use SSH remote mode for submit/monitor."
    )

    list_parser = subparsers.add_parser("list", help="List historical runs.")
    list_parser.add_argument("--limit", type=int, default=20, help="Max runs to show.")

    report_parser = subparsers.add_parser("report", help="Show report placeholder.")
    report_parser.add_argument("run_id")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch remote outputs via SSH.")
    fetch_parser.add_argument("--run-id", required=True, help="Target run id.")
    fetch_parser.add_argument("--run-root", help="Target run root.")

    dft_tools_explain_parser = subparsers.add_parser(
        "dft-tools-explain",
        help="Call dft_tools explain API for an existing run and persist the structured result.",
    )
    dft_tools_explain_parser.add_argument("--run-id", help="Target run id.")
    dft_tools_explain_parser.add_argument("--run-root", help="Target run root.")
    dft_tools_explain_parser.add_argument(
        "--base-url",
        default=None,
        help="dft_tools Web base URL, default uses DFT_TOOLS_BASE_URL or http://127.0.0.1:8000",
    )
    dft_tools_explain_parser.add_argument(
        "--no-kb-ingest",
        action="store_true",
        help="Skip automatic dft_tools knowledge-base ingest and only persist explain artifacts.",
    )

    adsorption_generate_parser = subparsers.add_parser(
        "adsorption-candidates",
        help="Generate adsorption candidates from a slab structure and adsorbate input.",
    )
    adsorption_generate_parser.add_argument("--slab-path", required=True, help="Local slab structure path.")
    adsorption_generate_parser.add_argument("--adsorbate", required=True, help="Adsorbate name, SMILES, or molecule file path.")
    adsorption_generate_parser.add_argument("--material", required=True, help="Material label for this candidate generation run.")
    adsorption_generate_parser.add_argument("--prompt", required=True, help="Original adsorption modeling prompt or plan summary.")
    adsorption_generate_parser.add_argument("--output-dir", help="Optional output directory for candidate manifest/artifacts.")
    adsorption_generate_parser.add_argument("--task-id", help="Optional task id override.")
    adsorption_generate_parser.add_argument("--candidate-height", type=float, default=2.1, help="Initial adsorption height in angstrom.")
    adsorption_generate_parser.add_argument("--max-sites-per-family", type=int, default=2, help="Maximum adsorption sites to keep per site family.")
    adsorption_generate_parser.add_argument("--preferred-site", help="Preferred site family hint, such as ontop/bridge/hollow.")
    adsorption_generate_parser.add_argument("--preferred-orientation", help="Preferred orientation hint, such as upright/flat/tilted.")
    adsorption_generate_parser.add_argument("--vacancy-species", help="Optional vacancy species to remove before generating candidates, e.g. O.")

    adsorption_select_parser = subparsers.add_parser(
        "adsorption-select",
        help="Select one generated adsorption candidate and hand it off to the existing builder/submit workflow.",
    )
    adsorption_select_parser.add_argument("--manifest", required=True, help="Path to candidate_manifest.json.")
    adsorption_select_parser.add_argument("--candidate-id", required=True, help="Candidate id from the manifest.")
    adsorption_select_parser.add_argument("--material", required=True, help="Material label for the selected adsorption task.")
    adsorption_select_parser.add_argument("--prompt", required=True, help="Original adsorption prompt or plan summary.")
    adsorption_select_parser.add_argument("--submit-profile", choices=SUPPORTED_SUBMIT_PROFILES, help="Cluster submit profile, for example c32 or b96.")
    adsorption_select_parser.add_argument("--submit", action="store_true", help="After build, attempt Slurm submission.")
    adsorption_select_parser.add_argument("--remote", action="store_true", help="Use SSH remote submission instead of local sbatch.")

    workflow_parser = subparsers.add_parser(
        "adsorption-workflow",
        help="Execute or summarize a prepared adsorption workflow bundle under a complex run root.",
    )
    workflow_parser.add_argument("--run-root", required=True, help="Complex adsorption run root containing adsorption_workflow_bundle.json.")
    workflow_parser.add_argument("--submit", action="store_true", help="Submit clean_slab / isolated_adsorbate / adsorbed_system subtasks if they are ready.")
    workflow_parser.add_argument("--remote", action="store_true", help="Use SSH remote submission for subtask submit.")
    workflow_parser.add_argument(
        "--status",
        action="store_true",
        help="Summarize workflow bundle readiness, per-subtask state, and next recommended action.",
    )
    workflow_parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor submitted subtasks; remote subtasks auto-use remote monitor.",
    )
    workflow_parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch remote outputs for subtasks that already have remote execution metadata.",
    )
    workflow_parser.add_argument(
        "--parse-analyze",
        action="store_true",
        help="Parse/analyze completed subtask outputs and aggregate adsorption energy summary when possible.",
    )

    web_parser = subparsers.add_parser("web", help="Start the local visual Web UI.")
    web_parser.add_argument("--host", default="127.0.0.1", help="Bind host for the local Web UI.")
    web_parser.add_argument("--port", type=int, default=8787, help="Bind port for the local Web UI.")

    return parser


def create_plan_result(args: argparse.Namespace) -> PlanningResult:
    return PLANNER.plan(
        prompt=args.prompt or "未提供任务描述",
        task_id=f"task_{uuid4().hex[:8]}",
        material_name=args.material,
        structure_path=args.structure_path,
        forced_task_type=TaskType(args.task_type) if args.task_type else None,
        submit_profile=args.submit_profile,
    )


def create_demo_run_record(spec: ExperimentSpec) -> RunRecord:
    run_id = f"run_{uuid4().hex[:8]}"
    run_root = PROJECT_ROOT / ".aether" / "runs" / spec.task_id / run_id
    checkpoint_path = run_root / "outputs" / ".pipeline_checkpoint.json"
    return RunRecord(
        task_id=spec.task_id,
        run_id=run_id,
        run_root=str(run_root),
        checkpoint_path=str(checkpoint_path),
        tags=spec.tags.copy(),
    )


def create_complex_run_record(plan: ExperimentPlan) -> RunRecord:
    run_id = f"run_{uuid4().hex[:8]}"
    run_root = PROJECT_ROOT / ".aether" / "runs" / plan.task_id / run_id
    checkpoint_path = run_root / "outputs" / ".pipeline_checkpoint.json"
    return RunRecord(
        task_id=plan.task_id,
        run_id=run_id,
        run_root=str(run_root),
        checkpoint_path=str(checkpoint_path),
        tags=[],
    )


def print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def normalize_adsorbate_hint(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    formula_matches = re.findall(
        r"\b(H2O|OH|CO2|CO|NH3|NO2|NO|O2|N2|H2|CH4|HCOOH|HCOO|CHO|CH3OH)\b",
        text,
        flags=re.IGNORECASE,
    )
    if formula_matches:
        return formula_matches[-1]
    generic_match = re.search(r"([A-Z][a-z]?\d?(?:[A-Z][a-z]?\d?)*)", text)
    if generic_match:
        return generic_match.group(1)
    text = re.sub(r"^(计算|研究|评估|分析|构建|生成)\s*", "", text)
    text = re.sub(r"\s*(吸附|adsorption|adsorb).*$", "", text, flags=re.IGNORECASE)
    return text.strip() or None


def infer_adsorbate_hint(plan: ExperimentPlan) -> str | None:
    raw_plan = plan.raw_plan or {}
    for key in ("adsorbate", "adsorbate_hint"):
        value = raw_plan.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_adsorbate_hint(value)

    prompt = plan.source_prompt
    direct_matches = [
        r"\b(H2O|OH|CO2|CO|NH3|NO|NO2|O2|N2|H2|CH4)\b",
        r"([A-Z][a-z]?\d?(?:[A-Z][a-z]?\d?)*)\s*(?:吸附|adsorption|adsorb)",
        r"(?:吸附|adsorption|adsorb)\s*([A-Z][a-z]?\d?(?:[A-Z][a-z]?\d?)*)",
    ]
    for pattern in direct_matches:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return normalize_adsorbate_hint(match.group(1))

    try:
        hint = PLANNER.rule_planner._extract_adsorbate_hint(prompt)  # type: ignore[attr-defined]
        if isinstance(hint, str) and hint.strip():
            return normalize_adsorbate_hint(hint)
    except Exception:
        pass
    return None


def infer_workflow_material_label(
    plan: ExperimentPlan,
    *,
    explicit_material: str | None = None,
    adsorbate_source: str | None = None,
) -> str:
    if explicit_material:
        return explicit_material

    raw_plan = plan.raw_plan or {}
    material_name = raw_plan.get("material_name")
    if isinstance(material_name, str) and material_name.strip():
        return material_name.strip()

    rule_hints = raw_plan.get("rule_hints") if isinstance(raw_plan.get("rule_hints"), dict) else {}
    surface_hint = rule_hints.get("surface_hint") if isinstance(rule_hints, dict) else None
    substrate = None
    miller = None
    if isinstance(surface_hint, dict):
        substrate = surface_hint.get("substrate")
        miller = surface_hint.get("miller_index")
    if substrate and miller:
        miller_text = "".join(str(item) for item in miller)
        if adsorbate_source:
            return f"{adsorbate_source}/{substrate}({miller_text})"
        return f"{substrate}({miller_text})"

    summary = plan.summary.strip()
    if summary and "回退到规则解析" not in summary:
        return summary
    if adsorbate_source:
        return f"{adsorbate_source}/adsorption"
    return "adsorption_workflow"


def maybe_generate_adsorption_candidates(
    *,
    args: argparse.Namespace,
    plan: ExperimentPlan,
    run_record: RunRecord,
) -> dict[str, Any] | None:
    if plan.experiment_type != "adsorption_energy":
        return None
    if not args.structure_path:
        return {
            "status": "skipped",
            "reason": "未提供 --structure-path，无法在主线中自动生成 adsorption candidates。",
        }

    adsorbate_hint = infer_adsorbate_hint(plan)
    if not adsorbate_hint:
        return {
            "status": "skipped",
            "reason": "未能从 prompt/plan 中推断 adsorbate，当前先保留 scaffold，不自动生成 candidates。",
        }

    material_label = infer_workflow_material_label(
        plan,
        explicit_material=getattr(args, "material", None),
        adsorbate_source=adsorbate_hint,
    )
    spec = ExperimentSpec(
        task_id=plan.task_id,
        task_type=TaskType.RELAX,
        material_name=material_label,
        source_prompt=plan.source_prompt,
        structure_path=args.structure_path,
    )
    builder = get_builder()
    structure_resolution = builder.structure_resolver.resolve(spec)
    if structure_resolution.structure is None:
        return {
            "status": "blocked",
            "reason": structure_resolution.message,
        }

    defect_recipe = None
    raw_plan = plan.raw_plan or {}
    if isinstance(raw_plan.get("defect_hint"), dict):
        defect_recipe = raw_plan["defect_hint"]

    output_dir = Path(run_record.run_root) / "scaffold" / "adsorption_candidates"
    generator = AdsorptionCandidateGenerator()
    candidates = generator.generate(
        AdsorptionGenerationRequest(
            slab_structure=structure_resolution.structure,
            adsorbate_source=adsorbate_hint,
            task_id=plan.task_id,
            material_name=material_label,
            source_prompt=plan.source_prompt,
            defect_recipe=defect_recipe,
        )
    )
    manifest = CandidateManifest(
        task_id=plan.task_id,
        material_name=material_label,
        source_prompt=plan.source_prompt,
        slab_source=args.structure_path,
        adsorbate_source=adsorbate_hint,
        candidates=candidates,
        metadata={
            "structure_resolution": structure_resolution.metadata,
            "defect_recipe": defect_recipe,
            "source": "mainline_run_scaffold",
        },
    )
    paths = CandidateManifestWriter().write(manifest, output_dir)
    metadata_path = STORE.write_metadata(
        Path(run_record.run_root),
        "adsorption_candidate_generation.json",
        {
            "status": "generated",
            "candidate_count": len(candidates),
            "adsorbate_hint": adsorbate_hint,
            "manifest": paths,
        },
    )
    run_record.notes["adsorption_candidates"] = {
        "candidate_count": len(candidates),
        "manifest_json": paths["manifest_json"],
        "manifest_md": paths["manifest_md"],
        "adsorbate_hint": adsorbate_hint,
    }
    return {
        "status": "generated",
        "candidate_count": len(candidates),
        "adsorbate_hint": adsorbate_hint,
        "manifest": paths,
        "metadata_path": str(metadata_path),
        "top_candidates": [candidate.to_dict() for candidate in candidates[:5]],
    }


def maybe_select_adsorption_candidate(
    *,
    args: argparse.Namespace,
    plan: ExperimentPlan,
    run_record: RunRecord,
    adsorption_candidate_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    selected_candidate_id = getattr(args, "selected_candidate_id", None) or getattr(args, "candidate_id", None)
    if not selected_candidate_id:
        return None
    if not adsorption_candidate_result or adsorption_candidate_result.get("status") != "generated":
        return {
            "status": "blocked",
            "reason": "当前 run 没有可用的 adsorption candidates，无法在主线继续选择候选。",
        }

    manifest_path = Path(adsorption_candidate_result["manifest"]["manifest_json"])
    selection = ConfirmedCandidateHandoff().materialize_selection(
        manifest_path=manifest_path,
        candidate_id=selected_candidate_id,
        output_dir=Path(run_record.run_root) / "scaffold" / "adsorption_candidates" / "selected" / selected_candidate_id,
    )

    spec = ExperimentSpec(
        task_id=plan.task_id,
        task_type=TaskType.RELAX_SCF,
        material_name=args.material or plan.summary or "adsorption_selected_candidate",
        source_prompt=plan.source_prompt,
        structure_path=selection.selected_poscar_path,
        functional=(plan.raw_plan or {}).get("functional") or "PBE",
        submit_profile=args.submit_profile or plan.recommended_submit_profile,
        notes={
            "adsorption_candidate": selection.to_dict(),
            "adsorption_manifest_path": str(manifest_path),
        },
    )
    selection_summary_path = STORE.write_metadata(
        Path(run_record.run_root),
        "adsorption_selection.json",
        selection.to_dict(),
    )
    append_phase_artifact(run_record, PipelinePhase.PLAN, selection_summary_path)

    build_result = get_builder().build_initial_workspace(spec, run_record)
    submit_result = None
    if args.submit and run_record.overall_status == RunStatus.READY:
        if args.remote:
            submit_result = get_remote_runner().submit(spec, run_record)
            summary_name = "remote_submit_summary.json"
        else:
            submit_result = get_runner().submit(spec, run_record)
            summary_name = "submit_summary.json"
        submit_summary_path = STORE.write_metadata(
            Path(run_record.run_root),
            summary_name,
            {
                "status": submit_result.status,
                "message": submit_result.message,
                "details": submit_result.details,
            },
        )
        append_phase_artifact(run_record, PipelinePhase.SUBMIT, submit_summary_path)
    STORE.save_run_record(run_record)
    return {
        "status": "selected",
        "selection": selection.to_dict(),
        "build_result": build_result,
        "submit_result": (
            {
                "status": submit_result.status,
                "message": submit_result.message,
                "details": submit_result.details,
            }
            if submit_result is not None
            else None
        ),
    }


def maybe_materialize_adsorption_workflow_bundle(
    *,
    args: argparse.Namespace,
    plan: ExperimentPlan,
    run_record: RunRecord,
    selected_candidate_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not selected_candidate_result or selected_candidate_result.get("status") != "selected":
        return None
    if not args.structure_path:
        return {
            "status": "blocked",
            "reason": "缺少 --structure-path，无法继续生成 clean slab 子任务。",
        }

    adsorbate_source = infer_adsorbate_hint(plan)
    if not adsorbate_source:
        return {
            "status": "blocked",
            "reason": "未能推断 adsorbate，无法继续生成 isolated adsorbate 子任务。",
        }

    from dft_app.builder.workspace_builder import WorkspaceBuilder

    builder = get_builder()
    functional = (plan.raw_plan or {}).get("functional") or "PBE"
    submit_profile = args.submit_profile or plan.recommended_submit_profile
    run_root = Path(run_record.run_root)
    scaffold_root = run_root / "scaffold" / "subtasks"
    selection = selected_candidate_result["selection"]
    workflow_label = infer_workflow_material_label(
        plan,
        explicit_material=getattr(args, "material", None),
        adsorbate_source=adsorbate_source,
    )

    clean_spec = ExperimentSpec(
        task_id=plan.task_id,
        task_type=TaskType.RELAX_SCF,
        material_name=f"{workflow_label}-clean-slab",
        source_prompt=plan.source_prompt,
        structure_source=StructureSource.LOCAL_FILE,
        structure_path=args.structure_path,
        functional=functional,
        submit_profile=submit_profile,
        notes={"system_role": "slab"},
    )
    clean_resolution = builder.structure_resolver.resolve(clean_spec)
    if clean_resolution.structure is None:
        return {
            "status": "blocked",
            "reason": clean_resolution.message,
        }
    # 为 clean slab 自动冻结底部原子
    from dft_app.modeling.adsorption_candidate_generator import apply_selective_dynamics
    clean_resolution.structure = apply_selective_dynamics(
        clean_resolution.structure,
        slab_atom_count=len(clean_resolution.structure),
    )

    adsorbate_structure = AdsorbateStructureFactory().build_boxed_structure(adsorbate_source)
    adsorbate_spec = ExperimentSpec(
        task_id=plan.task_id,
        task_type=TaskType.RELAX_SCF,
        material_name=f"{adsorbate_source}-isolated",
        source_prompt=plan.source_prompt,
        structure_source=StructureSource.MANUAL_BUILD,
        functional=functional,
        submit_profile=submit_profile,
        notes={"system_role": "molecule"},
    )

    adsorbed_spec = ExperimentSpec(
        task_id=plan.task_id,
        task_type=TaskType.RELAX_SCF,
        material_name=f"{workflow_label}-adsorbed",
        source_prompt=plan.source_prompt,
        structure_source=StructureSource.LOCAL_FILE,
        structure_path=selection["selected_poscar_path"],
        functional=functional,
        submit_profile=submit_profile,
        notes={"system_role": "adsorbate_slab", "adsorption_candidate": selection},
    )
    adsorbed_resolution = builder.structure_resolver.resolve(adsorbed_spec)
    if adsorbed_resolution.structure is None:
        return {
            "status": "blocked",
            "reason": adsorbed_resolution.message,
        }

    task_dirs = {
        "clean_slab": scaffold_root / "01_clean_slab",
        "isolated_adsorbate": scaffold_root / "02_isolated_adsorbate",
        "adsorbed_system": scaffold_root / "03_adsorbed_system",
        "analysis": scaffold_root / "04_analysis",
    }

    clean_bundle = _materialize_subtask_bundle(
        task_dir=task_dirs["clean_slab"],
        spec=clean_spec,
        structure=clean_resolution.structure,
        structure_metadata=clean_resolution.metadata or {},
        checklist_builder=builder,
    )
    adsorbate_bundle = _materialize_subtask_bundle(
        task_dir=task_dirs["isolated_adsorbate"],
        spec=adsorbate_spec,
        structure=adsorbate_structure,
        structure_metadata={"input_format": "generated_adsorbate", "adsorbate_source": adsorbate_source},
        checklist_builder=builder,
    )
    adsorbed_bundle = _materialize_subtask_bundle(
        task_dir=task_dirs["adsorbed_system"],
        spec=adsorbed_spec,
        structure=adsorbed_resolution.structure,
        structure_metadata=adsorbed_resolution.metadata or {},
        checklist_builder=builder,
    )

    analysis_dir = task_dirs["analysis"]
    (analysis_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (analysis_dir / "report").mkdir(parents=True, exist_ok=True)
    analysis_plan = {
        "formula": "E_ads = E_adsorbate_slab - E_slab - E_molecule",
        "inputs": {
            "clean_slab": clean_bundle["bundle_root"],
            "isolated_adsorbate": adsorbate_bundle["bundle_root"],
            "adsorbed_system": adsorbed_bundle["bundle_root"],
        },
        "selected_candidate_id": selection["candidate_id"],
    }
    (analysis_dir / "metadata" / "analysis_plan.json").write_text(
        json.dumps(analysis_plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (analysis_dir / "report" / "README.md").write_text(
        "# 吸附能分析占位\n\n"
        "- 公式：`E_ads = E_adsorbate_slab - E_slab - E_molecule`\n"
        f"- selected_candidate_id: `{selection['candidate_id']}`\n"
        "- 当前阶段已完成三体系输入包准备，待后续计算结果齐全后进入自动汇总分析。\n",
        encoding="utf-8",
    )

    bundle_summary = {
        "status": "prepared",
        "selected_candidate_id": selection["candidate_id"],
        "subtasks": {
            "clean_slab": clean_bundle,
            "isolated_adsorbate": adsorbate_bundle,
            "adsorbed_system": adsorbed_bundle,
            "analysis": {"bundle_root": str(analysis_dir)},
        },
    }
    summary_path = STORE.write_metadata(run_root, "adsorption_workflow_bundle.json", bundle_summary)
    run_record.complete_phase(
        PipelinePhase.BUILD,
        artifacts=[
            str(summary_path),
            str(Path(clean_bundle["summary_path"])),
            str(Path(adsorbate_bundle["summary_path"])),
            str(Path(adsorbed_bundle["summary_path"])),
            str(analysis_dir / "metadata" / "analysis_plan.json"),
            str(analysis_dir / "report" / "README.md"),
        ],
        message="adsorption workflow 子任务 bundle 已准备完成，可继续执行 submit/monitor/fetch/status。",
    )
    run_record.mark_ready()
    run_record.notes.setdefault("complex_workflow", {})
    run_record.notes["complex_workflow"]["execution_bundle"] = {
        **bundle_summary,
        "summary_path": str(summary_path),
    }
    STORE.save_run_record(run_record)
    return {**bundle_summary, "summary_path": str(summary_path)}


def _materialize_subtask_bundle(
    *,
    task_dir: Path,
    spec: ExperimentSpec,
    structure: Any,
    structure_metadata: dict[str, Any],
    checklist_builder: Any,
) -> dict[str, Any]:
    from dft_app.builder.workspace_builder import WorkspaceBuilder

    inputs_dir = task_dir / "inputs"
    outputs_dir = task_dir / "outputs"
    report_dir = task_dir / "report"
    metadata_dir = task_dir / "metadata"
    logs_dir = task_dir / "logs"
    for path in (inputs_dir, outputs_dir, report_dir, metadata_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    generated_inputs = checklist_builder.input_generator.generate(
        spec,
        structure,
        inputs_dir,
        structure_metadata,
    )
    slurm_path = inputs_dir / "job.slurm"
    slurm_path.write_text(WorkspaceBuilder._build_slurm_template(spec), encoding="utf-8", newline="\n")
    checklist_path = report_dir / "pre_submit_checklist.md"
    checklist_path.write_text(
        checklist_builder._build_pre_submit_checklist(spec, None),
        encoding="utf-8",
    )
    experiment_spec_path = metadata_dir / "experiment_spec.json"
    experiment_spec_path.write_text(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    subtask_run_record = RunRecord(
        task_id=f"{spec.task_id}:{task_dir.name}",
        run_id=task_dir.name,
        run_root=str(task_dir),
        checkpoint_path=str(outputs_dir / ".pipeline_checkpoint.json"),
    )
    subtask_run_record.start_phase(PipelinePhase.PLAN, message="已生成子任务 experiment_spec。")
    subtask_run_record.complete_phase(
        PipelinePhase.PLAN,
        artifacts=[str(experiment_spec_path)],
        message="子任务 experiment_spec 已写入 metadata。",
    )
    subtask_run_record.start_phase(PipelinePhase.BUILD, message="已生成子任务输入包。")
    bundle_summary = {
        "bundle_root": str(task_dir),
        "task_type": spec.task_type.value,
        "material_name": spec.material_name,
        "inputs": generated_inputs,
        "job_slurm": str(slurm_path),
        "checklist": str(checklist_path),
    }
    summary_path = metadata_dir / "subtask_bundle_summary.json"
    summary_path.write_text(json.dumps(bundle_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    subtask_run_record.complete_phase(
        PipelinePhase.BUILD,
        artifacts=[
            str(slurm_path),
            str(checklist_path),
            str(summary_path),
            *[
                str(Path(path))
                for path in (
                    generated_inputs["poscar_path"],
                    generated_inputs["incar_path"],
                    generated_inputs["kpoints_path"],
                    generated_inputs["potcar_map_path"],
                )
            ],
        ],
        message="子任务输入包已准备完成。",
    )
    subtask_run_record.mark_ready()
    STORE.save_run_record(subtask_run_record)
    return {**bundle_summary, "summary_path": str(summary_path)}


def resolve_target_run(args: argparse.Namespace) -> Path:
    return STORE.resolve_run_root(run_root=args.run_root, run_id=args.run_id)


def load_run_context(args: argparse.Namespace) -> tuple[Path, ExperimentSpec, RunRecord]:
    run_root = resolve_target_run(args)
    spec = STORE.load_experiment_spec(run_root)
    run_record = STORE.load_run_record(run_root)
    return run_root, spec, run_record


def append_phase_artifact(run_record: RunRecord, phase: PipelinePhase, artifact_path: Path) -> None:
    artifacts = run_record.phases[phase.value].artifacts
    artifact = str(artifact_path)
    if artifact not in artifacts:
        artifacts.append(artifact)


def _persist_execution_summary(
    *,
    run_root: Path,
    run_record: RunRecord,
    filename: str,
    phase: PipelinePhase,
    result: Any,
) -> dict[str, Any]:
    summary_path = STORE.write_metadata(
        run_root,
        filename,
        {
            "status": result.status,
            "message": result.message,
            "details": result.details,
        },
    )
    append_phase_artifact(run_record, phase, summary_path)
    STORE.save_run_record(run_record)
    return {
        "status": result.status,
        "message": result.message,
        "details": result.details,
    }


def _read_optional_metadata(run_root: Path, filenames: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, filename in filenames.items():
        data = STORE.read_metadata_file(run_root, filename)
        if data is not None:
            payload[key] = data
    return payload


def _should_use_remote_workflow_runner(run_record: RunRecord, requested_remote: bool) -> bool:
    if requested_remote:
        return True
    remote_info = run_record.notes.get("remote", {})
    return bool(remote_info.get("remote_run_root"))


def _submit_is_allowed(run_record: RunRecord) -> bool:
    if run_record.scheduler_job_id is not None:
        return False
    if run_record.overall_status == RunStatus.READY:
        return True
    build_phase = run_record.phases[PipelinePhase.BUILD.value]
    submit_phase = run_record.phases[PipelinePhase.SUBMIT.value]
    return (
        run_record.overall_status == RunStatus.WAITING_CONFIRMATION
        and build_phase.status == PhaseStatus.COMPLETED
        and submit_phase.status == PhaseStatus.BLOCKED
    )


def collect_adsorption_workflow_status(run_root: Path) -> dict[str, Any]:
    bundle = STORE.read_metadata_file(run_root, "adsorption_workflow_bundle.json")
    if bundle is None:
        raise FileNotFoundError(f"未找到 adsorption_workflow_bundle.json: {run_root}")

    subtasks: dict[str, Any] = {}
    status_counts: Counter[str] = Counter()
    submit_ready: list[str] = []
    monitor_pending: list[str] = []
    fetch_ready: list[str] = []
    parse_ready: list[str] = []
    analyze_ready: list[str] = []

    for name in ADSORPTION_WORKFLOW_SUBTASKS:
        subtask_root = Path(bundle["subtasks"][name]["bundle_root"])
        run_record = STORE.load_run_record(subtask_root)
        phase_statuses = {
            phase_name: phase_record.status.value
            for phase_name, phase_record in run_record.phases.items()
        }
        summaries = _read_optional_metadata(
            subtask_root,
            {
                "submit": "submit_summary.json",
                "remote_submit": "remote_submit_summary.json",
                "monitor": "monitor_summary.json",
                "remote_monitor": "remote_monitor_summary.json",
                "remote_fetch": "remote_fetch_summary.json",
                "parse": "parse_summary.json",
                "parsed_result": "parsed_result.json",
                "analyze": "analysis_summary.json",
            },
        )
        remote_info = run_record.notes.get("remote", {})
        has_remote = bool(remote_info.get("remote_run_root"))
        outputs_dir = subtask_root / "outputs"
        has_outputs = any((outputs_dir / filename).exists() for filename in ("OUTCAR", "vasprun.xml", "CONTCAR"))
        phase_monitor_status = phase_statuses[PipelinePhase.MONITOR.value]
        phase_parse_status = phase_statuses[PipelinePhase.PARSE.value]
        phase_analyze_status = phase_statuses[PipelinePhase.ANALYZE.value]

        if _submit_is_allowed(run_record):
            submit_ready.append(name)
        if run_record.scheduler_job_id and phase_monitor_status != "completed":
            monitor_pending.append(name)
        if (
            has_remote
            and run_record.scheduler_job_id
            and not has_outputs
            and (
                phase_monitor_status == PhaseStatus.COMPLETED.value
                or summaries.get("remote_fetch") is not None
            )
        ):
            fetch_ready.append(name)
        if has_outputs and phase_parse_status != "completed":
            parse_ready.append(name)
        if summaries.get("parsed_result") is not None and phase_analyze_status != "completed":
            analyze_ready.append(name)

        subtasks[name] = {
            "run_root": str(subtask_root),
            "overall_status": run_record.overall_status.value,
            "current_phase": run_record.current_phase.value if run_record.current_phase else None,
            "scheduler_job_id": run_record.scheduler_job_id,
            "has_remote": has_remote,
            "has_outputs": has_outputs,
            "phase_statuses": phase_statuses,
            "last_error": run_record.last_error,
            "summaries": summaries,
        }
        status_counts[run_record.overall_status.value] += 1

    aggregate_ready = STORE.read_metadata_file(run_root, "adsorption_energy_result.json") is not None
    if aggregate_ready:
        status = "aggregated"
        recommended_next_steps: list[str] = []
    else:
        recommended_next_steps = []
        if submit_ready:
            recommended_next_steps.append("提交尚未启动的子任务（adsorption-workflow --submit）。")
        if monitor_pending:
            recommended_next_steps.append("轮询已提交子任务状态（adsorption-workflow --monitor）。")
        if fetch_ready:
            recommended_next_steps.append("同步远程输出文件（adsorption-workflow --fetch）。")
        if parse_ready or analyze_ready:
            recommended_next_steps.append("对已同步输出执行 parse/analyze（adsorption-workflow --parse-analyze）。")
        if not recommended_next_steps:
            recommended_next_steps.append("当前没有新的自动化动作；请检查各子任务 summaries 或等待集群继续运行。")

        if monitor_pending:
            status = "running"
        elif fetch_ready:
            status = "ready_for_fetch"
        elif submit_ready and len(submit_ready) == len(ADSORPTION_WORKFLOW_SUBTASKS):
            status = "prepared"
        elif parse_ready or analyze_ready:
            status = "ready_for_analysis"
        else:
            status = "in_progress"

    payload = {
        "status": status,
        "run_root": str(run_root),
        "subtask_count": len(ADSORPTION_WORKFLOW_SUBTASKS),
        "status_counts": dict(status_counts),
        "submit_ready": submit_ready,
        "monitor_pending": monitor_pending,
        "fetch_ready": fetch_ready,
        "parse_ready": parse_ready,
        "analyze_ready": analyze_ready,
        "aggregate_ready": aggregate_ready,
        "recommended_next_steps": recommended_next_steps,
        "subtasks": subtasks,
    }
    STORE.write_metadata(run_root, "adsorption_workflow_status.json", payload)
    return payload


def _summarize_adsorption_workflow_if_ready(run_root: Path) -> dict[str, Any]:
    missing_parsed_results: list[str] = []
    missing_total_energy: list[str] = []
    bundle = STORE.read_metadata_file(run_root, "adsorption_workflow_bundle.json")
    if bundle is None:
        raise FileNotFoundError(f"未找到 adsorption_workflow_bundle.json: {run_root}")

    for name in ADSORPTION_WORKFLOW_SUBTASKS:
        subtask_root = Path(bundle["subtasks"][name]["bundle_root"])
        parsed_payload = STORE.read_metadata_file(subtask_root, "parsed_result.json")
        if parsed_payload is None:
            missing_parsed_results.append(name)
            continue
        if parsed_payload.get("total_energy") is None:
            missing_total_energy.append(name)

    if missing_parsed_results or missing_total_energy:
        return {
            "status": "blocked",
            "message": "尚未满足吸附能自动汇总条件。",
            "missing_parsed_results": missing_parsed_results,
            "missing_total_energy": missing_total_energy,
        }

    return summarize_adsorption_workflow(run_root)


def handle_run(args: argparse.Namespace) -> int:
    if args.status:
        run_root = resolve_target_run(args)
        run_record = STORE.load_run_record(run_root)
        print_json(run_record.to_dict())
        return 0

    if args.reset:
        print("重置骨架已就位，后续将接入 checkpoint 清理。")
        return 0

    planning_result = create_plan_result(args)
    modeling_result = MODELER.build(
        spec=planning_result.spec,
        plan=planning_result.plan,
    )
    model_spec = modeling_result.model_spec
    if getattr(args, "model_spec_path", None):
        model_spec = model_spec_from_dict(json.loads(Path(args.model_spec_path).read_text(encoding="utf-8")))
    if getattr(args, "step2_manifest_path", None) or getattr(args, "candidate_id", None):
        model_spec.metadata.setdefault("step2_lineage", {})
        model_spec.metadata["step2_lineage"].update(
            {
                key: value
                for key, value in {
                    "step2_manifest_path": getattr(args, "step2_manifest_path", None),
                    "candidate_id": getattr(args, "candidate_id", None),
                }.items()
                if value
            }
        )
    spec = planning_result.spec
    if args.dry_run and spec is None:
        print("=== Planning Result ===")
        print_json(PLANNER.explain(planning_result))
        print("=== ModelSpec ===")
        print_json(model_spec.to_dict())
        return 0
    if spec is None:
        run_record = create_complex_run_record(planning_result.plan)
        result = get_orchestrator().scaffold(
            planning_result.plan,
            run_record,
            model_spec=model_spec,
            structure_path=args.structure_path,
        )
        selected_candidate_result = maybe_select_adsorption_candidate(
            args=args,
            plan=planning_result.plan,
            run_record=run_record,
            adsorption_candidate_result=result.details.get("adsorption_candidates"),
        )
        workflow_bundle_result = maybe_materialize_adsorption_workflow_bundle(
            args=args,
            plan=planning_result.plan,
            run_record=run_record,
            selected_candidate_result=selected_candidate_result,
        )
        STORE.save_run_record(run_record)
        print("=== Planning Result ===")
        print_json(PLANNER.explain(planning_result))
        print("=== ModelSpec ===")
        print_json(model_spec.to_dict())
        print("=== RunRecord ===")
        print_json(run_record.to_dict())
        print("=== Complex Workflow Result ===")
        print_json(
            {
                "status": result.status,
                "message": result.message,
                "details": result.details,
                "selected_candidate": selected_candidate_result,
                "workflow_bundle": workflow_bundle_result,
            }
        )
        return 0
    run_record = create_demo_run_record(spec)

    if args.dry_run:
        print("=== Planning Result ===")
        print_json(PLANNER.explain(planning_result))
        print("=== ModelSpec ===")
        print_json(model_spec.to_dict())
        print("=== ExperimentSpec ===")
        print_json(spec.to_dict())
        print("=== RunRecord ===")
        print_json(run_record.to_dict())
        return 0

    builder = get_builder()
    build_result = builder.build_initial_workspace(
        spec,
        run_record,
        model_spec=model_spec,
    )
    submit_result = None
    if args.submit and run_record.overall_status == RunStatus.READY:
        if args.remote:
            submit_result = get_remote_runner().submit(spec, run_record)
            summary_name = "remote_submit_summary.json"
        else:
            submit_result = get_runner().submit(spec, run_record)
            summary_name = "submit_summary.json"

        submit_summary_path = STORE.write_metadata(
            Path(run_record.run_root),
            summary_name,
            {
                "status": submit_result.status,
                "message": submit_result.message,
                "details": submit_result.details,
            },
        )
        append_phase_artifact(run_record, PipelinePhase.SUBMIT, submit_summary_path)
        STORE.save_run_record(run_record)

    print("=== Planning Result ===")
    print_json(PLANNER.explain(planning_result))
    print("=== ModelSpec ===")
    print_json(model_spec.to_dict())
    print("=== ExperimentSpec ===")
    print_json(spec.to_dict())
    print("=== RunRecord ===")
    print_json(run_record.to_dict())
    print("=== Build Result ===")
    print_json(build_result)
    if submit_result is not None:
        print("=== Submit Result ===")
        print_json(
            {
                "status": submit_result.status,
                "message": submit_result.message,
                "details": submit_result.details,
            }
        )
    return 0


def handle_step(args: argparse.Namespace) -> int:
    run_root, spec, run_record = load_run_context(args)

    if args.phase == PipelinePhase.SUBMIT.value:
        if args.remote:
            result = get_remote_runner().submit(spec, run_record)
            summary_name = "remote_submit_summary.json"
        else:
            result = get_runner().submit(spec, run_record)
            summary_name = "submit_summary.json"
        summary_path = STORE.write_metadata(
            run_root,
            summary_name,
            {
                "status": result.status,
                "message": result.message,
                "details": result.details,
            },
        )
        append_phase_artifact(run_record, PipelinePhase.SUBMIT, summary_path)
        STORE.save_run_record(run_record)
        print_json(
            {
                "phase": args.phase,
                "status": result.status,
                "message": result.message,
                "details": result.details,
                "run_record": run_record.to_dict(),
            }
        )
        return 0

    if args.phase == PipelinePhase.MONITOR.value:
        if args.remote:
            result = get_remote_runner().monitor(run_record)
            summary_name = "remote_monitor_summary.json"
        else:
            result = get_runner().monitor(run_record)
            summary_name = "monitor_summary.json"
        summary_path = STORE.write_metadata(
            run_root,
            summary_name,
            {
                "status": result.status,
                "message": result.message,
                "details": result.details,
            },
        )
        append_phase_artifact(run_record, PipelinePhase.MONITOR, summary_path)
        STORE.save_run_record(run_record)
        print_json(
            {
                "phase": args.phase,
                "status": result.status,
                "message": result.message,
                "details": result.details,
                "run_record": run_record.to_dict(),
            }
        )
        return 0

    if args.phase == PipelinePhase.PARSE.value:
        result = get_output_parser().parse(spec, run_record)
        payload = {
            "status": result.status,
            "message": result.message,
            "details": result.details,
        }
        if result.parsed_result is not None:
            payload["parsed_result"] = result.parsed_result.to_dict()
            parsed_result_path = STORE.write_metadata(
                run_root,
                "parsed_result.json",
                result.parsed_result.to_dict(),
            )
            append_phase_artifact(run_record, PipelinePhase.PARSE, parsed_result_path)

        summary_path = STORE.write_metadata(run_root, "parse_summary.json", payload)
        append_phase_artifact(run_record, PipelinePhase.PARSE, summary_path)
        STORE.save_run_record(run_record)
        print_json(
            {
                "phase": args.phase,
                "status": result.status,
                "message": result.message,
                "details": result.details,
                "parsed_result": result.parsed_result.to_dict()
                if result.parsed_result is not None
                else None,
                "run_record": run_record.to_dict(),
            }
        )
        return 0

    if args.phase == PipelinePhase.ANALYZE.value:
        parsed_result = STORE.load_parsed_result(run_root)
        result = get_analyzer().analyze(spec, run_record, parsed_result)
        payload = {
            "status": result.status,
            "message": result.message,
            "analysis_summary": result.analysis_summary,
            "report_path": result.report_path,
        }
        summary_path = STORE.write_metadata(run_root, "analysis_summary.json", payload)
        append_phase_artifact(run_record, PipelinePhase.ANALYZE, summary_path)
        if result.report_path:
            append_phase_artifact(run_record, PipelinePhase.ANALYZE, Path(result.report_path))
        STORE.save_run_record(run_record)
        print_json(
            {
                "phase": args.phase,
                "status": result.status,
                "message": result.message,
                "analysis_summary": result.analysis_summary,
                "report_path": result.report_path,
                "run_record": run_record.to_dict(),
            }
        )
        return 0

    if args.phase == PipelinePhase.EXPORT.value:
        result = get_exporter().export(spec, run_record)
        payload = {
            "status": result.status,
            "message": result.message,
            "export_manifest": result.export_manifest,
            "export_root": result.export_root,
            "package_path": result.package_path,
        }
        summary_path = STORE.write_metadata(run_root, "export_summary.json", payload)
        append_phase_artifact(run_record, PipelinePhase.EXPORT, summary_path)
        if result.package_path:
            append_phase_artifact(run_record, PipelinePhase.EXPORT, Path(result.package_path))
        STORE.save_run_record(run_record)
        print_json(
            {
                "phase": args.phase,
                "status": result.status,
                "message": result.message,
                "export_manifest": result.export_manifest,
                "export_root": result.export_root,
                "package_path": result.package_path,
                "run_record": run_record.to_dict(),
            }
        )
        return 0

    print(f"单步执行尚未实现：phase={args.phase}")
    return 0


def handle_adsorption_candidates(args: argparse.Namespace) -> int:
    task_id = args.task_id or f"task_{uuid4().hex[:8]}"
    spec = ExperimentSpec(
        task_id=task_id,
        task_type=TaskType.RELAX,
        material_name=args.material,
        source_prompt=args.prompt,
        structure_path=args.slab_path,
    )
    builder = get_builder()
    structure_resolution = builder.structure_resolver.resolve(spec)
    if structure_resolution.structure is None:
        raise RuntimeError(structure_resolution.message)

    run_record = create_demo_run_record(spec)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(run_record.run_root) / "scaffold" / "adsorption_candidates"
    )
    defect_recipe = None
    if args.vacancy_species:
        defect_recipe = {
            "mode": "vacancy",
            "species": args.vacancy_species,
            "surface_only": True,
        }
    generator = AdsorptionCandidateGenerator()
    candidates = generator.generate(
        AdsorptionGenerationRequest(
            slab_structure=structure_resolution.structure,
            adsorbate_source=args.adsorbate,
            task_id=task_id,
            material_name=args.material,
            source_prompt=args.prompt,
            defect_recipe=defect_recipe,
            preferred_site=args.preferred_site,
            preferred_orientation=args.preferred_orientation,
            candidate_height=args.candidate_height,
            max_sites_per_family=args.max_sites_per_family,
        )
    )
    manifest = CandidateManifest(
        task_id=task_id,
        material_name=args.material,
        source_prompt=args.prompt,
        slab_source=args.slab_path,
        adsorbate_source=args.adsorbate,
        candidates=candidates,
        metadata={
            "structure_resolution": structure_resolution.metadata,
            "defect_recipe": defect_recipe,
        },
    )
    writer = CandidateManifestWriter()
    paths = writer.write(manifest, output_dir)
    summary_path = STORE.write_metadata(
        Path(run_record.run_root),
        "adsorption_candidate_generation.json",
        {
            "task_id": task_id,
            "status": "generated",
            "candidate_count": len(candidates),
            "output_dir": str(output_dir),
            "manifest": paths,
        },
    )
    print_json(
        {
            "status": "generated",
            "task_id": task_id,
            "run_root": run_record.run_root,
            "candidate_count": len(candidates),
            "output_dir": str(output_dir),
            "manifest": paths,
            "summary_path": str(summary_path),
            "top_candidates": [candidate.to_dict() for candidate in candidates[:5]],
        }
    )
    return 0


def handle_adsorption_select(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    handoff = ConfirmedCandidateHandoff()
    manifest_payload = handoff.load_manifest(manifest_path)
    selection = handoff.materialize_selection(
        manifest_path=manifest_path,
        candidate_id=args.candidate_id,
        output_dir=manifest_path.parent / "selected" / args.candidate_id,
    )
    spec = ExperimentSpec(
        task_id=f"task_{uuid4().hex[:8]}",
        task_type=TaskType.RELAX_SCF,
        material_name=args.material,
        source_prompt=args.prompt,
        structure_path=selection.selected_poscar_path,
        submit_profile=args.submit_profile,
        notes={
            "adsorption_candidate": selection.to_dict(),
            "adsorption_manifest": manifest_payload,
        },
    )
    run_record = create_demo_run_record(spec)
    builder = get_builder()
    build_result = builder.build_initial_workspace(spec, run_record)
    selection_summary_path = STORE.write_metadata(
        Path(run_record.run_root),
        "adsorption_selection.json",
        selection.to_dict(),
    )
    append_phase_artifact(run_record, PipelinePhase.PLAN, selection_summary_path)

    submit_result = None
    if args.submit and run_record.overall_status == RunStatus.READY:
        if args.remote:
            submit_result = get_remote_runner().submit(spec, run_record)
            summary_name = "remote_submit_summary.json"
        else:
            submit_result = get_runner().submit(spec, run_record)
            summary_name = "submit_summary.json"
        submit_summary_path = STORE.write_metadata(
            Path(run_record.run_root),
            summary_name,
            {
                "status": submit_result.status,
                "message": submit_result.message,
                "details": submit_result.details,
            },
        )
        append_phase_artifact(run_record, PipelinePhase.SUBMIT, submit_summary_path)

    STORE.save_run_record(run_record)
    print_json(
        {
            "status": "selected",
            "selection": selection.to_dict(),
            "build_result": build_result,
            "submit_result": (
                {
                    "status": submit_result.status,
                    "message": submit_result.message,
                    "details": submit_result.details,
                }
                if submit_result is not None
                else None
            ),
            "run_record": run_record.to_dict(),
        }
    )
    return 0


def handle_list(args: argparse.Namespace) -> int:
    print_json({"runs": STORE.list_runs(limit=args.limit)})
    return 0


def execute_adsorption_workflow(
    *,
    run_root: Path,
    submit: bool = False,
    status: bool = False,
    monitor: bool = False,
    fetch: bool = False,
    remote: bool = False,
    parse_analyze: bool = False,
) -> dict[str, Any]:
    bundle = STORE.read_metadata_file(run_root, "adsorption_workflow_bundle.json")
    if bundle is None:
        raise FileNotFoundError(f"未找到 adsorption_workflow_bundle.json: {run_root}")

    if not any([submit, status, monitor, fetch, parse_analyze]):
        status = True

    subtask_results: dict[str, Any] = {}
    for name in ADSORPTION_WORKFLOW_SUBTASKS:
        subtask_root = Path(bundle["subtasks"][name]["bundle_root"])
        spec = STORE.load_experiment_spec(subtask_root)
        run_record = STORE.load_run_record(subtask_root)
        result_payload: dict[str, Any] = {
            "run_root": str(subtask_root),
            "execution_mode": "remote"
            if _should_use_remote_workflow_runner(run_record, False)
            else "local",
        }

        if submit:
            if _submit_is_allowed(run_record):
                if remote:
                    submit_result = get_remote_runner().submit(spec, run_record)
                    summary_name = "remote_submit_summary.json"
                else:
                    submit_result = get_runner().submit(spec, run_record)
                    summary_name = "submit_summary.json"
                result_payload["submit"] = _persist_execution_summary(
                    run_root=subtask_root,
                    run_record=run_record,
                    filename=summary_name,
                    phase=PipelinePhase.SUBMIT,
                    result=submit_result,
                )
            else:
                result_payload["submit"] = {
                    "status": "skipped",
                    "message": f"子任务当前状态为 {run_record.overall_status.value}，跳过 submit。",
                    "details": {"overall_status": run_record.overall_status.value},
                }

        use_remote_runner = _should_use_remote_workflow_runner(run_record, remote)
        if monitor:
            if use_remote_runner:
                monitor_result = get_remote_runner().monitor(run_record)
                monitor_filename = "remote_monitor_summary.json"
            else:
                monitor_result = get_runner().monitor(run_record)
                monitor_filename = "monitor_summary.json"
            result_payload["monitor"] = _persist_execution_summary(
                run_root=subtask_root,
                run_record=run_record,
                filename=monitor_filename,
                phase=PipelinePhase.MONITOR,
                result=monitor_result,
            )

        if fetch:
            if use_remote_runner:
                fetch_result = get_remote_runner().fetch_outputs(run_record)
                result_payload["fetch"] = _persist_execution_summary(
                    run_root=subtask_root,
                    run_record=run_record,
                    filename="remote_fetch_summary.json",
                    phase=PipelinePhase.MONITOR,
                    result=fetch_result,
                )
            else:
                result_payload["fetch"] = {
                    "status": "skipped",
                    "message": "本子任务不是 remote 模式，无需 fetch。",
                    "details": {"execution_mode": "local"},
                }

        if parse_analyze:
            parsed_result = None
            parsed_payload = STORE.read_metadata_file(subtask_root, "parsed_result.json")
            if parsed_payload is None:
                parse_result = get_output_parser().parse(spec, run_record)
                parsed_result = parse_result.parsed_result
                parse_summary_payload = {
                    "status": parse_result.status,
                    "message": parse_result.message,
                    "details": parse_result.details,
                    "parsed_result": parsed_result.to_dict() if parsed_result is not None else None,
                }
                if parsed_result is not None:
                    parsed_result_path = STORE.write_metadata(
                        subtask_root, "parsed_result.json", parsed_result.to_dict()
                    )
                    append_phase_artifact(run_record, PipelinePhase.PARSE, parsed_result_path)
                parse_summary_path = STORE.write_metadata(
                    subtask_root,
                    "parse_summary.json",
                    parse_summary_payload,
                )
                append_phase_artifact(run_record, PipelinePhase.PARSE, parse_summary_path)
                STORE.save_run_record(run_record)
                result_payload["parse"] = {
                    "status": parse_result.status,
                    "message": parse_result.message,
                    "details": parse_result.details,
                }
            else:
                parsed_result = STORE.load_parsed_result(subtask_root)
                result_payload["parse"] = {
                    "status": "existing",
                    "message": "复用现有 parsed_result.json。",
                    "details": {
                        "parsed_result_path": str(subtask_root / "metadata" / "parsed_result.json")
                    },
                }

            if parsed_result is not None:
                analysis_payload = STORE.read_metadata_file(subtask_root, "analysis_summary.json")
                if analysis_payload is None:
                    analyze_result = get_analyzer().analyze(spec, run_record, parsed_result)
                    analysis_summary_path = STORE.write_metadata(
                        subtask_root,
                        "analysis_summary.json",
                        {
                            "status": analyze_result.status,
                            "message": analyze_result.message,
                            "analysis_summary": analyze_result.analysis_summary,
                            "report_path": analyze_result.report_path,
                        },
                    )
                    append_phase_artifact(run_record, PipelinePhase.ANALYZE, analysis_summary_path)
                    if analyze_result.report_path:
                        append_phase_artifact(run_record, PipelinePhase.ANALYZE, Path(analyze_result.report_path))
                    STORE.save_run_record(run_record)
                    result_payload["analyze"] = {
                        "status": analyze_result.status,
                        "message": analyze_result.message,
                        "report_path": analyze_result.report_path,
                    }
                else:
                    result_payload["analyze"] = {
                        "status": "existing",
                        "message": "复用现有 analysis_summary.json。",
                        "report_path": analysis_payload.get("report_path"),
                    }

        subtask_results[name] = result_payload

    aggregate = _summarize_adsorption_workflow_if_ready(run_root) if parse_analyze else None
    workflow_status = collect_adsorption_workflow_status(run_root)

    return {
        "run_root": str(run_root),
        "actions": {
            "submit": submit,
            "status": status,
            "monitor": monitor,
            "fetch": fetch,
            "parse_analyze": parse_analyze,
        },
        "subtasks": subtask_results,
        "status": workflow_status,
        "workflow_status": workflow_status,
        "aggregate": aggregate,
    }


def summarize_adsorption_workflow(run_root: Path) -> dict[str, Any]:
    bundle = STORE.read_metadata_file(run_root, "adsorption_workflow_bundle.json")
    if bundle is None:
        raise FileNotFoundError(f"未找到 adsorption_workflow_bundle.json: {run_root}")

    energy_sources: dict[str, float] = {}
    parsed_results: dict[str, Any] = {}
    for name in ["clean_slab", "isolated_adsorbate", "adsorbed_system"]:
        subtask_root = Path(bundle["subtasks"][name]["bundle_root"])
        parsed_result = STORE.load_parsed_result(subtask_root)
        if parsed_result.total_energy is None:
            raise RuntimeError(f"{name} 缺少 total_energy，无法汇总吸附能。")
        energy_sources[name] = parsed_result.total_energy
        parsed_results[name] = parsed_result.to_dict()

    adsorption_energy = (
        energy_sources["adsorbed_system"]
        - energy_sources["clean_slab"]
        - energy_sources["isolated_adsorbate"]
    )
    payload = {
        "status": "aggregated",
        "formula": "E_ads = E_adsorbate_slab - E_slab - E_molecule",
        "energies": energy_sources,
        "adsorption_energy": adsorption_energy,
        "parsed_results": parsed_results,
    }
    summary_path = STORE.write_metadata(run_root, "adsorption_energy_result.json", payload)
    report_dir = run_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "adsorption_energy_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# 吸附能汇总报告",
                "",
                f"- 公式：`{payload['formula']}`",
                f"- E_slab = {energy_sources['clean_slab']:.6f} eV",
                f"- E_molecule = {energy_sources['isolated_adsorbate']:.6f} eV",
                f"- E_adsorbate_slab = {energy_sources['adsorbed_system']:.6f} eV",
                f"- **E_ads = {adsorption_energy:.6f} eV**",
                "",
                "## 数据来源",
                *[
                    f"- `{name}`: `{bundle['subtasks'][name]['bundle_root']}`"
                    for name in ["clean_slab", "isolated_adsorbate", "adsorbed_system"]
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parent_record = STORE.load_run_record(run_root)
    parent_record.complete_phase(
        PipelinePhase.ANALYZE,
        artifacts=[str(summary_path), str(report_path)],
        message="吸附能汇总完成。",
    )
    parent_record.mark_completed(str(report_path))
    STORE.save_run_record(parent_record)
    return {**payload, "summary_path": str(summary_path), "report_path": str(report_path)}


def handle_adsorption_workflow(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    result = execute_adsorption_workflow(
        run_root=run_root,
        submit=args.submit,
        status=args.status,
        monitor=args.monitor,
        fetch=args.fetch,
        remote=args.remote,
        parse_analyze=args.parse_analyze,
    )
    print_json(result)
    return 0


def handle_report(args: argparse.Namespace) -> int:
    run_root = STORE.resolve_run_root(run_id=args.run_id)
    payload = {
        "run_root": str(run_root),
        "experiment_plan": STORE.read_metadata_file(run_root, "experiment_plan.json"),
        "experiment_spec": STORE.read_metadata_file(run_root, "experiment_spec.json"),
        "model_spec": STORE.read_metadata_file(run_root, "model_spec.json"),
        "modeling_summary": STORE.read_metadata_file(run_root, "modeling_summary.json"),
        "workflow_scaffold": STORE.read_metadata_file(run_root, "workflow_scaffold.json"),
        "complex_workflow_summary": STORE.read_metadata_file(run_root, "complex_workflow_summary.json"),
        "adsorption_candidate_generation": STORE.read_metadata_file(run_root, "adsorption_candidate_generation.json"),
        "adsorption_selection": STORE.read_metadata_file(run_root, "adsorption_selection.json"),
        "adsorption_workflow_bundle": STORE.read_metadata_file(run_root, "adsorption_workflow_bundle.json"),
        "adsorption_workflow_status": STORE.read_metadata_file(run_root, "adsorption_workflow_status.json"),
        "build_summary": STORE.read_metadata_file(run_root, "build_summary.json"),
        "submit_summary": STORE.read_metadata_file(run_root, "submit_summary.json"),
        "remote_submit_summary": STORE.read_metadata_file(run_root, "remote_submit_summary.json"),
        "monitor_summary": STORE.read_metadata_file(run_root, "monitor_summary.json"),
        "remote_monitor_summary": STORE.read_metadata_file(run_root, "remote_monitor_summary.json"),
        "remote_fetch_summary": STORE.read_metadata_file(run_root, "remote_fetch_summary.json"),
        "parse_summary": STORE.read_metadata_file(run_root, "parse_summary.json"),
        "parsed_result": STORE.read_metadata_file(run_root, "parsed_result.json"),
        "analysis_summary": STORE.read_metadata_file(run_root, "analysis_summary.json"),
        "dft_tools_explain_result": STORE.read_metadata_file(run_root, "dft_tools_explain_result.json"),
        "dft_tools_knowledge_backflow_payload": STORE.read_metadata_file(run_root, "dft_tools_knowledge_backflow_payload.json"),
        "dft_tools_kb_ingest_result": STORE.read_metadata_file(run_root, "dft_tools_kb_ingest_result.json"),
        "export_summary": STORE.read_metadata_file(run_root, "export_summary.json"),
        "run_record": STORE.read_metadata_file(run_root, "run_record.json"),
    }
    print_json(payload)
    return 0


def handle_fetch(args: argparse.Namespace) -> int:
    run_root = STORE.resolve_run_root(run_root=args.run_root, run_id=args.run_id)
    run_record = STORE.load_run_record(run_root)
    result = get_remote_runner().fetch_outputs(run_record)
    summary_path = STORE.write_metadata(
        run_root,
        "remote_fetch_summary.json",
        {
            "status": result.status,
            "message": result.message,
            "details": result.details,
        },
    )
    append_phase_artifact(run_record, PipelinePhase.MONITOR, summary_path)
    STORE.save_run_record(run_record)
    print_json(
        {
            "status": result.status,
            "message": result.message,
            "details": result.details,
            "run_record": run_record.to_dict(),
        }
    )
    return 0


def handle_dft_tools_explain(args: argparse.Namespace) -> int:
    from dft_app.integrations import run_dft_tools_explain_bridge

    run_root = STORE.resolve_run_root(run_root=args.run_root, run_id=args.run_id)
    result = run_dft_tools_explain_bridge(
        STORE,
        run_root,
        base_url=args.base_url,
        ingest_kb=not args.no_kb_ingest,
    )
    print_json(result)
    return 0


def handle_web(args: argparse.Namespace) -> int:
    from dft_app.web import run_server

    run_server(PROJECT_ROOT, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    ensure_console_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return handle_run(args)
        if args.command == "step":
            return handle_step(args)
        if args.command == "list":
            return handle_list(args)
        if args.command == "report":
            return handle_report(args)
        if args.command == "fetch":
            return handle_fetch(args)
        if args.command == "dft-tools-explain":
            return handle_dft_tools_explain(args)
        if args.command == "adsorption-candidates":
            return handle_adsorption_candidates(args)
        if args.command == "adsorption-select":
            return handle_adsorption_select(args)
        if args.command == "adsorption-workflow":
            return handle_adsorption_workflow(args)
        if args.command == "web":
            return handle_web(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}")
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
