from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    SINGLE_POINT = "single_point"
    GEOMETRY_OPTIMIZATION = "geometry_optimization"
    STATIC_REFINEMENT = "static_refinement"
    DOS = "dos"
    PDOS = "pdos"
    BAND_STRUCTURE = "band_structure"
    CHARGE_ANALYSIS = "charge_analysis"
    WORK_FUNCTION = "work_function"
    VIBRATIONAL_FREQUENCY = "vibrational_frequency"
    TRANSITION_STATE_SEARCH = "transition_state_search"
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    SPIN_RELATED = "spin_related"
    DEFECT_DOPING = "defect_doping"
    RELAX = "relax"
    RELAX_SCF = "relax_scf"
    RELAX_SCF_BAND = "relax_scf_band"
    ENCUT_CONVERGENCE = "encut_convergence"
    KPOINTS_CONVERGENCE = "kpoints_convergence"
    EOS = "eos"


class StructureSource(str, Enum):
    LOCAL_FILE = "local_file"
    MATERIALS_PROJECT = "materials_project"
    MANUAL_BUILD = "manual_build"
    DERIVED = "derived"


class ConfirmationItem(str, Enum):
    STRUCTURE = "structure"
    PARAMETERS = "parameters"
    SUBMISSION = "submission"


@dataclass
class StructureConstraint:
    phase: str | None = None
    space_group: str | None = None
    supercell: list[list[int]] | None = None
    surface: dict[str, Any] | None = None
    defect: dict[str, Any] | None = None


@dataclass
class KpointsStrategy:
    mode: str = "auto_density"
    value: int | tuple[int, int, int] | str | None = 40


@dataclass
class EncutStrategy:
    mode: str = "auto"
    value: int | None = None


@dataclass
class SmearingSettings:
    ismear: int = 0
    sigma: float = 0.05


@dataclass
class SpinSettings:
    is_spin_polarized: bool = False
    is_soc: bool = False


@dataclass
class ConvergenceSettings:
    ediff: float = 1e-6
    ediffg: float | None = -0.01
    nsw: int = 100


@dataclass
class JobSettings:
    partition: str | None = None
    nodes: int | None = 1
    ntasks: int | None = None
    ntasks_per_node: int | None = None
    cpus_per_task: int | None = None
    walltime: str | None = None
    memory: str | None = None
    memory_per_cpu: str | None = None
    vasp_variant: str | None = None


@dataclass
class ExperimentSpec:
    task_id: str
    task_type: TaskType
    material_name: str
    source_prompt: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    chemical_formula: str | None = None
    description: str | None = None
    structure_source: StructureSource = StructureSource.LOCAL_FILE
    structure_path: str | None = None
    structure_id: str | None = None
    structure_constraints: StructureConstraint = field(
        default_factory=StructureConstraint
    )
    workflow: list[str] = field(default_factory=list)
    code: str = "vasp"
    functional: str = "PBE"
    task_goal: str | None = None
    incar_overrides: dict[str, Any] = field(default_factory=dict)
    kpoints_strategy: KpointsStrategy = field(default_factory=KpointsStrategy)
    encut_strategy: EncutStrategy = field(default_factory=EncutStrategy)
    smearing: SmearingSettings = field(default_factory=SmearingSettings)
    spin_settings: SpinSettings = field(default_factory=SpinSettings)
    convergence_settings: ConvergenceSettings = field(
        default_factory=ConvergenceSettings
    )
    workflow_parameters: dict[str, Any] = field(default_factory=dict)
    submit_profile: str | None = None
    scheduler: str = "slurm"
    job_overrides: JobSettings = field(default_factory=JobSettings)
    requires_confirmation: bool = True
    confirmation_items: list[ConfirmationItem] = field(
        default_factory=lambda: [
            ConfirmationItem.STRUCTURE,
            ConfirmationItem.PARAMETERS,
            ConfirmationItem.SUBMISSION,
        ]
    )
    allow_reuse_previous_results: bool = True
    restart_from_task_id: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code != "vasp":
            raise ValueError("第一版 ExperimentSpec 目前只支持 code='vasp'")
        if self.scheduler != "slurm":
            raise ValueError("第一版 ExperimentSpec 目前只支持 scheduler='slurm'")
        if not self.material_name.strip():
            raise ValueError("material_name 不能为空")
        if not self.source_prompt.strip():
            raise ValueError("source_prompt 不能为空")

        if not self.workflow:
            self.workflow = self._default_workflow_for_task_type(self.task_type)

        self._validate_structure_source()
        self._validate_workflow()

    @staticmethod
    def _default_workflow_for_task_type(task_type: TaskType) -> list[str]:
        defaults = {
            TaskType.SINGLE_POINT: ["single_point"],
            TaskType.GEOMETRY_OPTIMIZATION: ["relax"],
            TaskType.STATIC_REFINEMENT: ["scf"],
            TaskType.DOS: ["scf", "dos"],
            TaskType.PDOS: ["scf", "pdos"],
            TaskType.BAND_STRUCTURE: ["scf", "band"],
            TaskType.CHARGE_ANALYSIS: ["scf", "charge_analysis"],
            TaskType.WORK_FUNCTION: ["scf", "work_function"],
            TaskType.VIBRATIONAL_FREQUENCY: ["relax", "frequency"],
            TaskType.TRANSITION_STATE_SEARCH: ["transition_state_search"],
            TaskType.MOLECULAR_DYNAMICS: ["molecular_dynamics"],
            TaskType.SPIN_RELATED: ["relax", "scf"],
            TaskType.DEFECT_DOPING: ["relax", "scf"],
            TaskType.RELAX: ["relax"],
            TaskType.RELAX_SCF: ["relax", "scf"],
            TaskType.RELAX_SCF_BAND: ["relax", "scf", "band"],
            TaskType.ENCUT_CONVERGENCE: ["encut_convergence"],
            TaskType.KPOINTS_CONVERGENCE: ["kpoints_convergence"],
            TaskType.EOS: ["eos"],
        }
        return defaults[task_type].copy()

    def _validate_structure_source(self) -> None:
        if (
            self.structure_source == StructureSource.LOCAL_FILE
            and not self.structure_path
        ):
            raise ValueError("structure_source=local_file 时，必须提供 structure_path")

        if (
            self.structure_source == StructureSource.MATERIALS_PROJECT
            and not self.structure_id
        ):
            raise ValueError(
                "structure_source=materials_project 时，必须提供 structure_id"
            )

    def _validate_workflow(self) -> None:
        valid_steps = {
            "single_point",
            "relax",
            "scf",
            "band",
            "dos",
            "pdos",
            "charge_analysis",
            "work_function",
            "frequency",
            "transition_state_search",
            "molecular_dynamics",
            "encut_convergence",
            "kpoints_convergence",
            "eos",
        }
        invalid_steps = [step for step in self.workflow if step not in valid_steps]
        if invalid_steps:
            raise ValueError(f"workflow 中包含不支持的步骤: {invalid_steps}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
