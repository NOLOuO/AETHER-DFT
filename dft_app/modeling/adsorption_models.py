from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pymatgen.core import Structure


@dataclass
class CandidateScore:
    total: float
    breakdown: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdsorptionCandidate:
    candidate_id: str
    site_family: str
    site_label: str
    orientation_label: str
    anchor_symbol: str
    height: float
    defect_label: str | None = None
    structure: Structure | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: CandidateScore | None = None
    exported_files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("structure", None)
        if self.score is not None:
            payload["score"] = self.score.to_dict()
        return payload


@dataclass
class CandidateManifest:
    task_id: str
    material_name: str
    source_prompt: str
    slab_source: str
    adsorbate_source: str
    candidates: list[AdsorptionCandidate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "material_name": self.material_name,
            "source_prompt": self.source_prompt,
            "slab_source": self.slab_source,
            "adsorbate_source": self.adsorbate_source,
            "metadata": self.metadata,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass
class CandidateSelection:
    manifest_path: str
    candidate_id: str
    selected_poscar_path: str
    selected_cif_path: str | None = None
    selected_summary_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
