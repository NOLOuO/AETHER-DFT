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
