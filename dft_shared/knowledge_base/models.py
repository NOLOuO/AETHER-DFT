"""知识库统一数据模型。

所有工具（vasp_result_analyzer、mace_ts_search、xsd_mace_preopt、OpenClaw）
共用同一套数据结构，通过 task_type 字段区分来源。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TaskRecord:
    """一次计算任务的完整记录——无论来源、成功与否都用这个结构。"""

    task_name: str
    task_type: str = "unknown"          # relax | ts_search | ts_dimer | freq | single_point | ...
    source_tool: str = "unknown"        # vasp_result_analyzer | mace_ts_search | xsd_mace_preopt | ...

    # 状态
    completed: bool = False
    converged: bool = False
    status: str = "unknown"             # success | failed | running | unknown
    failure_reason: str | None = None

    # 能量与力
    total_energy: float | None = None
    energy_per_atom: float | None = None
    max_force: float | None = None
    efermi: float | None = None
    band_gap: float | None = None

    # 步数
    ionic_steps: int | None = None
    electronic_steps: int | None = None

    # 结构上下文（由 structure_analyzer 填充）
    structure_context: dict[str, Any] = field(default_factory=dict)
    # 包括: formula, reduced_formula, atom_count, element_counts,
    #        likely_surface, structure_role, vacuum_axis, vacuum_thickness,
    #        substrate_species, adsorbate_species, substrate_family, adsorbate_family,
    #        site_signature, site_family, fixed_atom_count, fixed_atom_fraction

    # 输入参数上下文
    input_context: dict[str, Any] = field(default_factory=dict)
    # 包括: incar={ISPIN, IVDW, EDIFFG, ...}, kpoints={mesh}, potcar_elements

    # 结构对比上下文（POSCAR vs CONTCAR）
    displacement_context: dict[str, Any] = field(default_factory=dict)
    # 包括: max_displacement, mean_displacement, top_movers, bond_anomalies

    # 标签（用于检索）
    tags: list[str] = field(default_factory=list)

    # 建议
    warnings: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)

    # 原始数据（兜底）
    raw: dict[str, Any] = field(default_factory=dict)

    # 元信息
    remote_job_dir: str | None = None
    parsed_at: str | None = None
    version: int = 1
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class MatchResult:
    """知识库匹配结果。"""

    record: TaskRecord
    score: float
    reasons: list[str] = field(default_factory=list)
