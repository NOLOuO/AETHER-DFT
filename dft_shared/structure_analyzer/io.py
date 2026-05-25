from __future__ import annotations

from pathlib import Path
from typing import Any


def xsd_to_poscar(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Convert Materials Studio .xsd to VASP POSCAR.

    The heavy dependencies are imported lazily because AETHER-DFT's CLI and tests should remain usable
    before the scientific Python stack is installed.
    """
    from dft_app.builder.structure_resolver import StructureResolver
    from pymatgen.io.vasp import Poscar

    input_file = Path(input_path)
    output_file = Path(output_path)
    resolver = StructureResolver()
    structure, metadata = resolver._load_xsd_structure(input_file)  # reuse imported upstream behavior
    selective = structure.site_properties.get("selective_dynamics")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    Poscar(structure, selective_dynamics=selective).write_file(output_file)
    return {"input": str(input_file), "output": str(output_file), "metadata": metadata}


def poscar_to_xsd(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Convert POSCAR/CONTCAR to XSD using ASE's writer when available."""
    from ase.io import write
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    input_file = Path(input_path)
    output_file = Path(output_path)
    structure = Structure.from_file(input_file)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        write(str(output_file), atoms, format="xsd")
    except Exception as exc:
        raise RuntimeError("当前 ASE 安装不支持写出 XSD；请升级 ASE 或改用 xsd_to_poscar 单向入口。") from exc
    return {"input": str(input_file), "output": str(output_file), "metadata": {"atom_count": len(atoms)}}


def convert_structure(input_path: str | Path, output_path: str | Path, *, fmt: str | None = None) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_path)
    target = (fmt or output_file.suffix.lstrip(".")).lower()
    if input_file.suffix.lower() == ".xsd" and target in {"poscar", "vasp", ""}:
        return xsd_to_poscar(input_file, output_file)
    if input_file.name.upper() in {"POSCAR", "CONTCAR"} or input_file.suffix.lower() in {".vasp", ".poscar"}:
        if target == "xsd":
            return poscar_to_xsd(input_file, output_file)
    raise ValueError(f"不支持的结构转换: {input_file} -> {output_file} (fmt={fmt})")
