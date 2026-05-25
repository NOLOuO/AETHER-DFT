from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ModelSourceKind(str, Enum):
    SIMPLE_SPEC = "simple_spec"
    COMPLEX_PLAN = "complex_plan"


class ConfirmationLevel(str, Enum):
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    AUTO = "auto"


class BuildOperation(str, Enum):
    DIRECT_USE = "direct_use"
    BUILD_SLAB = "build_slab"
    BUILD_ISOLATED_BOX = "build_isolated_box"
    BUILD_DEFECT_SUPERCELL = "build_defect_supercell"
    PLACE_ADSORBATE = "place_adsorbate"
    ENUMERATE_ADSORPTION_CANDIDATES = "enumerate_adsorption_candidates"
    BUILD_TS_GUESS = "build_ts_guess"
    DERIVE_FROM_PREVIOUS_RESULT = "derive_from_previous_result"
    SET_SPIN_CONFIGURATION = "set_spin_configuration"
    ANALYSIS_ONLY = "analysis_only"


@dataclass
class BuildSpec:
    source_type: str
    source_ref: str | None = None
    operations: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    template_hints: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class CalcSpec:
    code: str = "vasp"
    task_type: str | None = None
    workflow: list[str] = field(default_factory=list)
    functional: str | None = None
    incar_overrides: dict[str, Any] = field(default_factory=dict)
    kpoints: dict[str, Any] = field(default_factory=dict)
    encut: dict[str, Any] = field(default_factory=dict)
    smearing: dict[str, Any] = field(default_factory=dict)
    spin: dict[str, Any] = field(default_factory=dict)
    convergence: dict[str, Any] = field(default_factory=dict)
    submit_profile: str | None = None
    job: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfirmationEntry:
    field: str
    level: ConfirmationLevel
    reason: str
    current_value: Any = None


@dataclass
class SystemSpec:
    name: str
    role: str
    summary: str
    build: BuildSpec
    calc: CalcSpec
    dependencies: list[str] = field(default_factory=list)
    confirmation_items: list[ConfirmationEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStepSpec:
    name: str
    goal: str
    system: str | None = None
    depends_on: list[str] = field(default_factory=list)
    task_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowSpec:
    workflow_type: str
    steps: list[WorkflowStepSpec] = field(default_factory=list)
    analysis_formula: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    task_id: str
    model_type: str
    source_kind: ModelSourceKind
    source_prompt: str
    readiness: str
    requires_confirmation: bool
    summary: str
    systems: list[SystemSpec] = field(default_factory=list)
    workflow: WorkflowSpec = field(
        default_factory=lambda: WorkflowSpec(workflow_type="single_task")
    )
    confirmation_summary: list[ConfirmationEntry] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_spec_from_dict(data: dict[str, Any]) -> ModelSpec:
    """Rehydrate a ModelSpec while preserving model-authored Step 2 lineage."""

    def build_spec(payload: dict[str, Any] | None) -> BuildSpec:
        payload = payload or {}
        return BuildSpec(
            source_type=str(payload.get("source_type") or "unknown"),
            source_ref=payload.get("source_ref"),
            operations=list(payload.get("operations") or []),
            parameters=dict(payload.get("parameters") or {}),
            template_hints=list(payload.get("template_hints") or []),
            notes=list(payload.get("notes") or []),
        )

    def calc_spec(payload: dict[str, Any] | None) -> CalcSpec:
        payload = payload or {}
        return CalcSpec(
            code=payload.get("code", "vasp"),
            task_type=payload.get("task_type"),
            workflow=list(payload.get("workflow") or []),
            functional=payload.get("functional"),
            incar_overrides=dict(payload.get("incar_overrides") or {}),
            kpoints=dict(payload.get("kpoints") or {}),
            encut=dict(payload.get("encut") or {}),
            smearing=dict(payload.get("smearing") or {}),
            spin=dict(payload.get("spin") or {}),
            convergence=dict(payload.get("convergence") or {}),
            submit_profile=payload.get("submit_profile"),
            job=dict(payload.get("job") or {}),
        )

    def confirmation(payload: dict[str, Any]) -> ConfirmationEntry:
        return ConfirmationEntry(
            field=str(payload.get("field") or ""),
            level=ConfirmationLevel(payload.get("level", ConfirmationLevel.RECOMMENDED.value)),
            reason=str(payload.get("reason") or ""),
            current_value=payload.get("current_value"),
        )

    def system(payload: dict[str, Any]) -> SystemSpec:
        return SystemSpec(
            name=str(payload.get("name") or "system"),
            role=str(payload.get("role") or "unknown"),
            summary=str(payload.get("summary") or ""),
            build=build_spec(payload.get("build")),
            calc=calc_spec(payload.get("calc")),
            dependencies=list(payload.get("dependencies") or []),
            confirmation_items=[confirmation(item) for item in payload.get("confirmation_items", [])],
            metadata=dict(payload.get("metadata") or {}),
        )

    workflow_payload = data.get("workflow") or {}
    workflow = WorkflowSpec(
        workflow_type=str(workflow_payload.get("workflow_type") or "single_task"),
        steps=[
            WorkflowStepSpec(
                name=str(item.get("name") or "step"),
                goal=str(item.get("goal") or ""),
                system=item.get("system"),
                depends_on=list(item.get("depends_on") or []),
                task_type=item.get("task_type"),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in workflow_payload.get("steps", [])
        ],
        analysis_formula=workflow_payload.get("analysis_formula"),
        metadata=dict(workflow_payload.get("metadata") or {}),
    )

    return ModelSpec(
        task_id=str(data.get("task_id") or ""),
        model_type=str(data.get("model_type") or "unknown"),
        source_kind=ModelSourceKind(data.get("source_kind", ModelSourceKind.SIMPLE_SPEC.value)),
        source_prompt=str(data.get("source_prompt") or ""),
        readiness=str(data.get("readiness") or "needs_confirmation"),
        requires_confirmation=bool(data.get("requires_confirmation", True)),
        summary=str(data.get("summary") or ""),
        systems=[system(item) for item in data.get("systems", [])],
        workflow=workflow,
        confirmation_summary=[confirmation(item) for item in data.get("confirmation_summary", [])],
        missing_information=list(data.get("missing_information") or []),
        assumptions=list(data.get("assumptions") or []),
        metadata=dict(data.get("metadata") or {}),
    )
