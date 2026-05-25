from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LatticeParameters:
    a: float | None = None
    b: float | None = None
    c: float | None = None
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None


@dataclass
class ParsedResult:
    task_id: str
    run_id: str
    calc_type: str
    parsed_at: str = field(default_factory=_utcnow)
    completed: bool = False
    converged: bool = False
    total_energy: float | None = None
    energy_per_atom: float | None = None
    band_gap: float | None = None
    efermi: float | None = None
    is_metal: bool | None = None
    volume: float | None = None
    lattice_parameters: LatticeParameters = field(default_factory=LatticeParameters)
    ionic_steps: int | None = None
    electronic_steps: int | None = None
    max_force: float | None = None
    warnings: list[str] = field(default_factory=list)
    source_files: dict[str, str] = field(default_factory=dict)
    derived_metrics: dict[str, Any] = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)
    raw_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id 不能为空")
        if not self.run_id.strip():
            raise ValueError("run_id 不能为空")
        if not self.calc_type.strip():
            raise ValueError("calc_type 不能为空")

    def has_electronic_results(self) -> bool:
        return self.band_gap is not None or self.efermi is not None

    def has_structural_results(self) -> bool:
        return any(
            value is not None
            for value in (
                self.lattice_parameters.a,
                self.lattice_parameters.b,
                self.lattice_parameters.c,
                self.volume,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
