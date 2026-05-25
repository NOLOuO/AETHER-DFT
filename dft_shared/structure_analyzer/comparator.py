"""结构对比模块：POSCAR vs CONTCAR 位移分析。

替代人眼在 VESTA 中手动检查"哪些原子跑了"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.io import read


@dataclass
class DisplacementReport:
    """位移分析报告。"""

    max_displacement: float                  # Å
    mean_displacement: float                 # Å
    max_displacement_atom: int               # 0-based index
    max_displacement_element: str
    per_atom: list[dict]                     # [{index, element, displacement, dx, dy, dz}, ...]
    top_movers: list[dict]                   # 位移最大的 N 个原子
    layer_summary: list[dict]               # 按层统计平均位移
    adsorbate_drift: dict | None = None     # 吸附质整体漂移信息
    anomalies: list[str] = field(default_factory=list)


def compare_structures(
    initial: Atoms | str,
    final: Atoms | str,
    *,
    top_n: int = 10,
    z_tol: float = 0.15,
    anomaly_threshold: float = 2.0,
) -> DisplacementReport:
    """对比初末态结构，返回位移分析报告。

    Args:
        initial: 初始结构 (POSCAR) 或文件路径
        final: 最终结构 (CONTCAR) 或文件路径
        top_n: 报告位移最大的几个原子
        z_tol: 层识别容差 (Å)
        anomaly_threshold: 超过此位移(Å)视为异常
    """
    if isinstance(initial, str):
        initial = read(initial)
    if isinstance(final, str):
        final = read(final)

    pos_i = initial.get_positions()
    pos_f = final.get_positions()
    symbols = initial.get_chemical_symbols()
    n = len(initial)

    # MIC 位移
    cell = np.array(initial.get_cell())
    pbc = np.array(initial.get_pbc(), dtype=bool)
    diff = pos_f - pos_i
    if cell.any():
        cell_inv = np.linalg.inv(cell)
        scaled = diff @ cell_inv
        for a in range(3):
            if pbc[a]:
                scaled[:, a] -= np.round(scaled[:, a])
        diff = scaled @ cell

    disp_norms = np.linalg.norm(diff, axis=1)

    # 逐原子信息
    per_atom = []
    for i in range(n):
        per_atom.append({
            "index": i,
            "element": symbols[i],
            "displacement": float(disp_norms[i]),
            "dx": float(diff[i, 0]),
            "dy": float(diff[i, 1]),
            "dz": float(diff[i, 2]),
        })

    # Top movers
    sorted_idx = np.argsort(-disp_norms)
    top_movers = [per_atom[int(idx)] for idx in sorted_idx[:top_n]]

    # 按层统计
    layer_summary = _layer_displacement_summary(pos_i, disp_norms, symbols, z_tol)

    # 异常检测
    anomalies = []
    max_idx = int(np.argmax(disp_norms))
    max_disp = float(disp_norms[max_idx])

    large_movers = [i for i in range(n) if disp_norms[i] > anomaly_threshold]
    if large_movers:
        elements_moved = set(symbols[i] for i in large_movers)
        anomalies.append(
            f"{len(large_movers)} 个原子位移 > {anomaly_threshold:.1f} Å "
            f"(涉及元素: {', '.join(sorted(elements_moved))})"
        )

    # 检测吸附质脱附
    adsorbate_drift = _detect_adsorbate_drift(initial, final, diff, disp_norms, z_tol)
    if adsorbate_drift and adsorbate_drift.get("drifted"):
        anomalies.append(f"吸附质整体漂移 {adsorbate_drift['drift_magnitude']:.2f} Å — 可能脱附或迁移")

    # 检测底层异常位移
    bottom_movers = _detect_fixed_layer_motion(pos_i, disp_norms, z_tol)
    if bottom_movers:
        anomalies.append(f"底层原子异常位移: {len(bottom_movers)} 个原子移动 > 0.1 Å — 检查 FixAtoms 约束")

    return DisplacementReport(
        max_displacement=max_disp,
        mean_displacement=float(np.mean(disp_norms)),
        max_displacement_atom=max_idx,
        max_displacement_element=symbols[max_idx],
        per_atom=per_atom,
        top_movers=top_movers,
        layer_summary=layer_summary,
        adsorbate_drift=adsorbate_drift,
        anomalies=anomalies,
    )


def _layer_displacement_summary(
    positions: np.ndarray,
    disp_norms: np.ndarray,
    symbols: list[str],
    z_tol: float,
) -> list[dict]:
    """按 z 层聚类，统计每层的平均位移。"""
    z = positions[:, 2]
    z_sorted = np.sort(z)
    centers: list[float] = []
    for zi in z_sorted:
        if not centers or abs(zi - centers[-1]) > z_tol:
            centers.append(zi)

    layers = []
    for layer_idx, z0 in enumerate(centers):
        mask = np.abs(z - z0) < z_tol
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        layer_disp = disp_norms[indices]
        layer_elements = set(symbols[i] for i in indices)
        layers.append({
            "layer_index": layer_idx,
            "z_center": float(z0),
            "n_atoms": int(len(indices)),
            "elements": sorted(layer_elements),
            "mean_displacement": float(np.mean(layer_disp)),
            "max_displacement": float(np.max(layer_disp)),
        })

    return layers


def _detect_adsorbate_drift(
    initial: Atoms,
    final: Atoms,
    diff: np.ndarray,
    disp_norms: np.ndarray,
    z_tol: float,
) -> dict | None:
    """检测吸附质（顶层非金属原子）是否整体漂移。"""
    z = initial.get_positions()[:, 2]
    symbols = initial.get_chemical_symbols()

    # 简单启发：非金属 + 在最高两层
    z_sorted = np.sort(z)
    centers: list[float] = []
    for zi in z_sorted:
        if not centers or abs(zi - centers[-1]) > z_tol:
            centers.append(zi)

    if len(centers) < 3:
        return None

    top_z_threshold = centers[-2] - z_tol
    metals = {"Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
              "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
              "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Hf",
              "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Pb", "Bi"}

    adsorbate_idx = [
        i for i in range(len(initial))
        if z[i] > top_z_threshold and symbols[i] not in metals
    ]

    if not adsorbate_idx:
        return None

    ads_diff = diff[adsorbate_idx]
    ads_com_drift = np.mean(ads_diff, axis=0)
    drift_mag = float(np.linalg.norm(ads_com_drift))

    return {
        "n_adsorbate_atoms": len(adsorbate_idx),
        "elements": sorted(set(symbols[i] for i in adsorbate_idx)),
        "drift_vector": [float(x) for x in ads_com_drift],
        "drift_magnitude": drift_mag,
        "mean_individual_displacement": float(np.mean(disp_norms[adsorbate_idx])),
        "drifted": drift_mag > 1.0,
    }


def _detect_fixed_layer_motion(
    positions: np.ndarray,
    disp_norms: np.ndarray,
    z_tol: float,
    threshold: float = 0.1,
) -> list[int]:
    """检测底层原子是否有意外位移。"""
    z = positions[:, 2]
    z_min = z.min()
    bottom_mask = z < (z_min + z_tol * 2)
    bottom_movers = [
        int(i) for i in np.where(bottom_mask)[0]
        if disp_norms[i] > threshold
    ]
    return bottom_movers
