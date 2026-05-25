"""键长与配位环境分析模块。

检测异常键长、断键、新成键——这些是人在 VESTA 里一眼就能看到的信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.neighborlist import natural_cutoffs, neighbor_list


@dataclass
class BondInfo:
    """一根键的信息。"""

    atom_i: int
    atom_j: int
    element_i: str
    element_j: str
    distance: float
    covalent_sum: float
    ratio: float            # distance / covalent_sum
    is_anomalous: bool = False
    anomaly_type: str = ""  # "too_short" | "too_long" | "broken" | "formed"


@dataclass
class BondReport:
    """键长分析报告。"""

    n_bonds: int
    bonds: list[BondInfo]
    anomalous_bonds: list[BondInfo]
    coordination: list[dict]        # [{index, element, n_neighbors, neighbors}, ...]
    formed_bonds: list[BondInfo]    # 新成键（初态无、末态有）
    broken_bonds: list[BondInfo]    # 断键（初态有、末态无）
    anomalies: list[str] = field(default_factory=list)


def analyze_bonds(
    atoms: Atoms,
    *,
    scale: float = 1.2,
    short_threshold: float = 0.75,
    long_threshold: float = 1.35,
) -> BondReport:
    """分析单个结构的键长和配位。

    Args:
        atoms: 结构
        scale: 邻居列表截断 = covalent_radii * scale
        short_threshold: distance/cov_sum < 此值视为过短
        long_threshold: distance/cov_sum > 此值视为过长
    """
    symbols = atoms.get_chemical_symbols()
    cutoffs = natural_cutoffs(atoms, mult=scale)
    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoffs)

    bonds: list[BondInfo] = []
    seen = set()

    for idx in range(len(i_list)):
        i, j = int(i_list[idx]), int(j_list[idx])
        if i >= j:
            continue
        pair = (i, j)
        if pair in seen:
            continue
        seen.add(pair)

        dist = float(d_list[idx])
        cov_i = covalent_radii[atomic_numbers[symbols[i]]]
        cov_j = covalent_radii[atomic_numbers[symbols[j]]]
        cov_sum = cov_i + cov_j
        ratio = dist / cov_sum if cov_sum > 0 else 999.0

        is_anom = ratio < short_threshold or ratio > long_threshold
        anom_type = ""
        if ratio < short_threshold:
            anom_type = "too_short"
        elif ratio > long_threshold:
            anom_type = "too_long"

        bonds.append(BondInfo(
            atom_i=i, atom_j=j,
            element_i=symbols[i], element_j=symbols[j],
            distance=dist, covalent_sum=cov_sum, ratio=ratio,
            is_anomalous=is_anom, anomaly_type=anom_type,
        ))

    anomalous = [b for b in bonds if b.is_anomalous]

    # 配位数
    coord_counts: dict[int, list[int]] = {}
    for idx in range(len(i_list)):
        i = int(i_list[idx])
        coord_counts.setdefault(i, []).append(int(j_list[idx]))

    coordination = []
    for atom_idx in range(len(atoms)):
        nbrs = coord_counts.get(atom_idx, [])
        coordination.append({
            "index": atom_idx,
            "element": symbols[atom_idx],
            "n_neighbors": len(nbrs),
            "neighbors": nbrs,
        })

    anomalies: list[str] = []
    if anomalous:
        short = [b for b in anomalous if b.anomaly_type == "too_short"]
        long_ = [b for b in anomalous if b.anomaly_type == "too_long"]
        if short:
            anomalies.append(f"{len(short)} 根键过短（可能原子重叠或初始结构不合理）")
        if long_:
            anomalies.append(f"{len(long_)} 根键偏长（可能即将断裂）")

    return BondReport(
        n_bonds=len(bonds),
        bonds=bonds,
        anomalous_bonds=anomalous,
        coordination=coordination,
        formed_bonds=[],
        broken_bonds=[],
        anomalies=anomalies,
    )


def compare_bonds(
    initial: Atoms,
    final: Atoms,
    *,
    scale: float = 1.2,
) -> tuple[list[BondInfo], list[BondInfo]]:
    """对比初末态的成键差异，返回 (formed_bonds, broken_bonds)。"""
    bonds_i = _get_bond_set(initial, scale)
    bonds_f = _get_bond_set(final, scale)

    symbols = initial.get_chemical_symbols()
    formed = []
    broken = []

    # 新成的键
    for pair, dist in bonds_f.items():
        if pair not in bonds_i:
            i, j = pair
            cov_sum = _cov_sum(symbols[i], symbols[j])
            formed.append(BondInfo(
                atom_i=i, atom_j=j,
                element_i=symbols[i], element_j=symbols[j],
                distance=dist, covalent_sum=cov_sum,
                ratio=dist / cov_sum if cov_sum > 0 else 999.0,
                is_anomalous=True, anomaly_type="formed",
            ))

    # 断了的键
    for pair, dist in bonds_i.items():
        if pair not in bonds_f:
            i, j = pair
            cov_sum = _cov_sum(symbols[i], symbols[j])
            broken.append(BondInfo(
                atom_i=i, atom_j=j,
                element_i=symbols[i], element_j=symbols[j],
                distance=dist, covalent_sum=cov_sum,
                ratio=dist / cov_sum if cov_sum > 0 else 999.0,
                is_anomalous=True, anomaly_type="broken",
            ))

    return formed, broken


def _get_bond_set(atoms: Atoms, scale: float) -> dict[tuple[int, int], float]:
    cutoffs = natural_cutoffs(atoms, mult=scale)
    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoffs)
    bonds = {}
    for idx in range(len(i_list)):
        i, j = int(i_list[idx]), int(j_list[idx])
        if i < j:
            bonds[(i, j)] = float(d_list[idx])
    return bonds


def _cov_sum(elem_a: str, elem_b: str) -> float:
    return covalent_radii[atomic_numbers[elem_a]] + covalent_radii[atomic_numbers[elem_b]]
