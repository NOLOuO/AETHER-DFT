"""结构序列化模块：Atoms → JSON / XYZ 供 3Dmol.js 渲染。

支持按位移/力/元素着色，支持 NEB 路径动画。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


def atoms_to_xyz_string(atoms: Atoms) -> str:
    """将 Atoms 转为 XYZ 格式字符串（3Dmol.js 原生支持）。"""
    import io
    buf = io.StringIO()
    write(buf, atoms, format="xyz")
    return buf.getvalue()


def atoms_to_json(
    atoms: Atoms,
    *,
    displacements: np.ndarray | None = None,
    forces: np.ndarray | None = None,
    labels: dict[int, str] | None = None,
) -> dict:
    """将 Atoms 转为 JSON 字典，附带可选的着色信息。

    Returns:
        {
            "cell": [[ax,ay,az], ...],
            "pbc": [true, true, false],
            "atoms": [
                {"index": 0, "element": "Pt", "x": ..., "y": ..., "z": ...,
                 "displacement": 0.12, "force": 0.05, "label": ""},
                ...
            ],
            "xyz": "... XYZ format string ..."
        }
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    cell = atoms.get_cell().tolist()
    pbc = atoms.get_pbc().tolist()

    atom_list = []
    for i in range(len(atoms)):
        entry = {
            "index": i,
            "element": symbols[i],
            "x": float(positions[i, 0]),
            "y": float(positions[i, 1]),
            "z": float(positions[i, 2]),
        }
        if displacements is not None:
            entry["displacement"] = float(displacements[i])
        if forces is not None and len(forces) > i:
            entry["force"] = float(np.linalg.norm(forces[i]))
        if labels and i in labels:
            entry["label"] = labels[i]
        atom_list.append(entry)

    return {
        "cell": cell,
        "pbc": pbc,
        "atoms": atom_list,
        "xyz": atoms_to_xyz_string(atoms),
    }


def neb_path_to_json(images: list[Atoms]) -> dict:
    """将 NEB 路径转为 JSON，支持动画播放。"""
    frames = []
    energies = []
    for i, img in enumerate(images):
        frame = atoms_to_json(img)
        frame["image_index"] = i
        try:
            e = float(img.get_potential_energy())
            energies.append(e)
            frame["energy"] = e
        except Exception:
            energies.append(None)
        frames.append(frame)

    return {
        "n_images": len(images),
        "frames": frames,
        "energies": energies,
    }


def read_structure_file(path: str | Path) -> Atoms:
    """智能读取结构文件（POSCAR/CONTCAR/XYZ/CIF 等）。"""
    return read(str(path))


def displacement_colormap(displacements: np.ndarray) -> list[str]:
    """将位移数组映射为 hex 颜色列表（蓝→白→红）。"""
    if len(displacements) == 0:
        return []
    d_min, d_max = float(np.min(displacements)), float(np.max(displacements))
    if d_max - d_min < 1e-10:
        return ["#ffffff"] * len(displacements)

    colors = []
    for d in displacements:
        t = (d - d_min) / (d_max - d_min)  # 0..1
        if t < 0.5:
            # 蓝 → 白
            s = t * 2
            r = int(255 * s)
            g = int(255 * s)
            b = 255
        else:
            # 白 → 红
            s = (t - 0.5) * 2
            r = 255
            g = int(255 * (1 - s))
            b = int(255 * (1 - s))
        colors.append(f"#{r:02x}{g:02x}{b:02x}")

    return colors
