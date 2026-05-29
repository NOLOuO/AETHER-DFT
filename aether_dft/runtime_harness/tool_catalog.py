from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ToolTier = Literal["primary", "fallback", "compat", "diagnostic"]


@dataclass(frozen=True)
class ToolCatalogEntry:
    name: str
    tier: ToolTier
    preferred_for: str
    use_instead: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


# N3 first pass: do not delete tools. Make routing preferences explicit so the
# prompt layer and future compact schema profiles can reduce model choice fatigue
# without breaking existing tests or compatibility callers.
TOOL_CATALOG: tuple[ToolCatalogEntry, ...] = (
    ToolCatalogEntry(
        "structure_modeling_intent_plan",
        "primary",
        "Classify Step 2 modeling intent and list missing evidence before writing structures.",
    ),
    ToolCatalogEntry(
        "structure_build_slab",
        "primary",
        "Build slab models from explicit material/source/miller/vacuum/fixed-layer evidence.",
    ),
    ToolCatalogEntry(
        "slab_surface_inspect",
        "primary",
        "Inspect surface coordination, top-layer atoms, and symmetry before choosing sites.",
    ),
    ToolCatalogEntry(
        "structure_enumerate_sites",
        "primary",
        "Enumerate adsorption sites used by model-authored adsorption candidates.",
    ),
    ToolCatalogEntry(
        "adsorbate_chemistry_hint",
        "primary",
        "Resolve anchor atoms, binding motifs, and initial heights for adsorbates.",
    ),
    ToolCatalogEntry(
        "knowledge_search_for_system",
        "primary",
        "Search project and cross-project priors before selecting adsorption candidates.",
    ),
    ToolCatalogEntry(
        "adsorption_candidate_plan",
        "primary",
        "Persist model-authored adsorption reasoning before candidate POSCAR generation.",
    ),
    ToolCatalogEntry(
        "structure_add_adsorbate",
        "primary",
        "Write one candidate POSCAR from explicit site coordinates and anchor/orientation evidence.",
    ),
    ToolCatalogEntry(
        "candidate_quality_score",
        "primary",
        "Score model-authored adsorption candidate geometry before composing manifest.",
    ),
    ToolCatalogEntry(
        "adsorption_candidate_manifest_compose",
        "primary",
        "Collect model-authored candidates into a traceable manifest with soft warnings and audit.",
    ),
    ToolCatalogEntry(
        "adsorption_build_slab",
        "compat",
        "Legacy adsorption slab builder entry point.",
        use_instead="structure_build_slab",
    ),
    ToolCatalogEntry(
        "adsorption_candidates",
        "fallback",
        "Black-box adsorption candidate generation when model-authored primitives are unavailable or explicitly requested.",
        use_instead="adsorption_candidate_plan -> structure_add_adsorbate -> adsorption_candidate_manifest_compose",
    ),
    ToolCatalogEntry(
        "adsorption_full_workflow",
        "fallback",
        "One-shot adsorption workflow fallback; not the default model-directed route.",
        use_instead="Step 2 primary path plus Step 3 primary path",
    ),
    ToolCatalogEntry(
        "defect_site_enumerate",
        "primary",
        "List candidate defect/dopant sites before writing defect structures.",
    ),
    ToolCatalogEntry(
        "structure_defect",
        "primary",
        "Apply vacancy or substitution after choosing a site with explicit evidence.",
    ),
    ToolCatalogEntry(
        "structure_add_vacancy",
        "compat",
        "Low-level vacancy primitive for explicit user requests.",
        use_instead="defect_site_enumerate -> structure_defect",
    ),
    ToolCatalogEntry(
        "structure_add_dopant",
        "compat",
        "Low-level dopant primitive for explicit user requests.",
        use_instead="defect_site_enumerate -> structure_defect",
    ),
    ToolCatalogEntry(
        "neb_input_check",
        "primary",
        "Check IS/FS before TS/NEB interpolation.",
    ),
    ToolCatalogEntry(
        "ts_midpoint_candidates_enumerate",
        "primary",
        "Generate TS/NEB midpoint initial guesses after IS/FS checks.",
    ),
    ToolCatalogEntry(
        "transition_state_plan",
        "compat",
        "Legacy transition-state planning envelope.",
        use_instead="neb_input_check -> ts_midpoint_candidates_enumerate",
    ),
    ToolCatalogEntry(
        "transition_state_dry_run",
        "compat",
        "Legacy TS dry-run envelope.",
        use_instead="neb_input_check -> ts_midpoint_candidates_enumerate",
    ),
    ToolCatalogEntry(
        "cluster_execution_intent_plan",
        "primary",
        "Classify Step 3 execution intent and missing evidence before build/submit.",
    ),
    ToolCatalogEntry(
        "research_onboarding_context",
        "primary",
        "Read project research rules, progress, and common pitfalls.",
    ),
    ToolCatalogEntry(
        "research_vasp_template_resolve",
        "primary",
        "Resolve research-backed VASP template expectations and blockers.",
    ),
    ToolCatalogEntry(
        "dft_run_task",
        "primary",
        "Build or run DFT task using evidence-preserving task bridge.",
    ),
    ToolCatalogEntry(
        "dft_task_plan",
        "compat",
        "Legacy task planning wrapper.",
        use_instead="cluster_execution_intent_plan -> dft_run_task",
    ),
    ToolCatalogEntry(
        "dft_run_step",
        "compat",
        "Legacy single-step DFT runner wrapper.",
        use_instead="dft_run_task",
    ),
    ToolCatalogEntry(
        "vasp_input_preflight_check",
        "primary",
        "Check input package readiness before cluster submission.",
    ),
    ToolCatalogEntry(
        "vasp_input_summary",
        "primary",
        "Summarize generated VASP inputs and template alignment.",
    ),
    ToolCatalogEntry(
        "cluster_config",
        "primary",
        "Inspect configured SSH/cluster profile.",
    ),
    ToolCatalogEntry(
        "cluster_probe",
        "primary",
        "Verify remote connectivity before submit/monitor/fetch.",
    ),
    ToolCatalogEntry(
        "cluster_research_status",
        "primary",
        "Check remote ~/research consistency.",
    ),
    ToolCatalogEntry(
        "cluster_research_sync",
        "primary",
        "Synchronize research rules to cluster when explicitly needed.",
    ),
    ToolCatalogEntry(
        "research_workspace_diff",
        "diagnostic",
        "Inspect local research workspace differences only; not remote ~/research status.",
        use_instead="cluster_research_status for remote consistency",
    ),
    ToolCatalogEntry(
        "cluster_remote_submit",
        "primary",
        "Submit a preflight-ready run_root with adaptive gate rechecks.",
    ),
    ToolCatalogEntry(
        "cluster_remote_monitor",
        "primary",
        "Monitor AETHER run_root/run_id state.",
    ),
    ToolCatalogEntry(
        "cluster_remote_fetch",
        "primary",
        "Fetch AETHER run outputs back from cluster.",
    ),
    ToolCatalogEntry(
        "vasp_output_scan",
        "primary",
        "Scan fetched VASP outputs for convergence and energy evidence.",
    ),
    ToolCatalogEntry(
        "cluster_my_jobs",
        "diagnostic",
        "Inspect scheduler jobs when no AETHER run_id/run_root is available.",
        use_instead="cluster_remote_monitor for AETHER runs",
    ),
    ToolCatalogEntry(
        "cluster_job_tail_log",
        "diagnostic",
        "Tail scheduler log for ad-hoc job debugging.",
        use_instead="cluster_remote_monitor / cluster_remote_fetch for AETHER runs",
    ),
    ToolCatalogEntry(
        "cluster_job_partial_outcar",
        "diagnostic",
        "Inspect partial OUTCAR for ad-hoc job debugging.",
        use_instead="cluster_remote_fetch -> vasp_output_scan for AETHER runs",
    ),
    ToolCatalogEntry(
        "cluster_job_progress_estimate",
        "diagnostic",
        "Estimate progress for ad-hoc scheduler jobs.",
        use_instead="cluster_remote_monitor for AETHER runs",
    ),
    ToolCatalogEntry(
        "result_interpret",
        "primary",
        "Interpret VASP output evidence before write-back.",
    ),
    ToolCatalogEntry(
        "candidate_outcome_record",
        "primary",
        "Write candidate outcome and calculation summary back to KB.",
    ),
    ToolCatalogEntry(
        "research_learning_capture",
        "primary",
        "Capture reusable research learning after evidence is available.",
    ),
    ToolCatalogEntry(
        "knowledge_note_add",
        "primary",
        "Persist reusable project knowledge.",
    ),
    ToolCatalogEntry(
        "next_experiment_propose",
        "diagnostic",
        "Suggest next experiments; does not replace result interpretation/write-back.",
        use_instead="result_interpret -> candidate_outcome_record / research_learning_capture",
    ),
)


PRIMARY_TOOL_NAMES: tuple[str, ...] = tuple(entry.name for entry in TOOL_CATALOG if entry.tier == "primary")
FALLBACK_TOOL_NAMES: tuple[str, ...] = tuple(entry.name for entry in TOOL_CATALOG if entry.tier in {"fallback", "compat", "diagnostic"})


def list_tool_catalog() -> list[dict[str, str | None]]:
    return [entry.to_dict() for entry in TOOL_CATALOG]


def describe_tool_route(name: str) -> dict[str, str | None]:
    for entry in TOOL_CATALOG:
        if entry.name == name:
            return entry.to_dict()
    return {"name": name, "tier": None, "preferred_for": None, "use_instead": None, "note": "unclassified"}
