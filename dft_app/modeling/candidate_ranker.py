from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pymatgen.core import Structure

from .adsorption_models import AdsorptionCandidate, CandidateScore


@dataclass
class AdsorptionRankingContext:
    preferred_site: str | None = None
    preferred_orientation: str | None = None
    target_height: float = 2.1


class AdsorptionCandidateRanker:
    """Heuristic ranker for first-version adsorption candidates."""

    def rank(
        self,
        candidates: Iterable[AdsorptionCandidate],
        *,
        context: AdsorptionRankingContext | None = None,
    ) -> list[AdsorptionCandidate]:
        ctx = context or AdsorptionRankingContext()
        ranked: list[AdsorptionCandidate] = []
        for candidate in candidates:
            candidate.score = self._score_candidate(candidate, ctx)
            ranked.append(candidate)
        ranked.sort(
            key=lambda item: (
                -(item.score.total if item.score is not None else 0.0),
                item.candidate_id,
            )
        )
        return ranked

    def _score_candidate(
        self,
        candidate: AdsorptionCandidate,
        context: AdsorptionRankingContext,
    ) -> CandidateScore:
        breakdown: dict[str, float] = {}
        total = 0.0

        clearance = float(candidate.metadata.get("minimum_clearance", 0.0) or 0.0)
        if clearance < 0.8:
            breakdown["clearance"] = -60.0
        elif clearance < 1.0:
            breakdown["clearance"] = -20.0
        elif clearance < 1.2:
            breakdown["clearance"] = 5.0
        elif clearance < 1.6:
            breakdown["clearance"] = 18.0
        else:
            breakdown["clearance"] = 24.0
        total += breakdown["clearance"]

        height_gap = abs(candidate.height - context.target_height)
        breakdown["height_sanity"] = max(0.0, 16.0 - height_gap * 10.0)
        total += breakdown["height_sanity"]

        if context.preferred_site and candidate.site_family == context.preferred_site:
            breakdown["site_preference"] = 12.0
        else:
            breakdown["site_preference"] = 0.0
        total += breakdown["site_preference"]

        if context.preferred_orientation and candidate.orientation_label == context.preferred_orientation:
            breakdown["orientation_preference"] = 8.0
        else:
            breakdown["orientation_preference"] = 0.0
        total += breakdown["orientation_preference"]

        adsorption_z = candidate.metadata.get("adsorbate_centroid_z")
        slab_top_z = candidate.metadata.get("slab_top_z")
        if adsorption_z is not None and slab_top_z is not None:
            z_gap = float(adsorption_z) - float(slab_top_z)
            breakdown["surface_gap"] = max(0.0, 14.0 - abs(z_gap - candidate.height) * 8.0)
            total += breakdown["surface_gap"]

        breakdown["stability_bias"] = self._stability_bias(candidate)
        total += breakdown["stability_bias"]

        reason_parts = [
            f"clearance={clearance:.3f}A",
            f"height={candidate.height:.2f}A",
            f"site={candidate.site_family}",
            f"orientation={candidate.orientation_label}",
        ]
        return CandidateScore(total=round(total, 3), breakdown=breakdown, reason=", ".join(reason_parts))

    @staticmethod
    def _stability_bias(candidate: AdsorptionCandidate) -> float:
        family = candidate.site_family.lower()
        orientation = candidate.orientation_label.lower()
        bias = 0.0
        if family in {"ontop", "top"}:
            bias += 4.0
        elif family in {"bridge"}:
            bias += 3.0
        elif family in {"hollow", "fcc", "hcp"}:
            bias += 2.0
        if orientation == "upright":
            bias += 3.0
        elif orientation == "tilted":
            bias += 1.0
        return bias


def minimum_intergroup_distance(structure: Structure, slab_atom_count: int) -> float:
    if len(structure) <= slab_atom_count or slab_atom_count <= 0:
        return 0.0
    coords = np.array(structure.cart_coords)
    slab_coords = coords[:slab_atom_count]
    ads_coords = coords[slab_atom_count:]
    min_distance = float("inf")
    for ads_coord in ads_coords:
        distances = np.linalg.norm(slab_coords - ads_coord, axis=1)
        min_distance = min(min_distance, float(np.min(distances)))
    return 0.0 if min_distance == float("inf") else float(min_distance)
