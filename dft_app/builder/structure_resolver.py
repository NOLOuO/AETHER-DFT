from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from dft_app.models import StructureSource
from dft_app.models.experiment_spec import ExperimentSpec
from dft_app.llm.key_store import resolve_api_key


@dataclass
class ResolvedStructure:
    structure: Structure | None
    source: str
    source_detail: str | None
    status: str
    message: str
    metadata: dict[str, Any] | None = None


class StructureResolver:
    """Resolve structures from local files or Materials Project."""

    def resolve(self, spec: ExperimentSpec) -> ResolvedStructure:
        if spec.structure_source == StructureSource.LOCAL_FILE:
            return self._resolve_local_file(spec)
        if spec.structure_source == StructureSource.MATERIALS_PROJECT:
            return self._resolve_materials_project(spec)
        return ResolvedStructure(
            structure=None,
            source=spec.structure_source.value,
            source_detail=spec.structure_id or spec.structure_path,
            status="needs_confirmation",
            message="当前任务还没有可直接解析的结构来源，需要人工确认或补充结构文件。",
            metadata={},
        )

    def _resolve_local_file(self, spec: ExperimentSpec) -> ResolvedStructure:
        if not spec.structure_path:
            return ResolvedStructure(
                structure=None,
                source=StructureSource.LOCAL_FILE.value,
                source_detail=None,
                status="error",
                message="structure_path 为空，无法解析本地结构文件。",
            )

        structure_path = Path(spec.structure_path)
        if not structure_path.exists():
            return ResolvedStructure(
                structure=None,
                source=StructureSource.LOCAL_FILE.value,
                source_detail=str(structure_path),
                status="error",
                message=f"本地结构文件不存在: {structure_path}",
            )

        try:
            structure, metadata = self._load_local_structure(structure_path)
        except Exception as exc:
            return ResolvedStructure(
                structure=None,
                source=StructureSource.LOCAL_FILE.value,
                source_detail=str(structure_path),
                status="error",
                message=f"读取本地结构文件失败: {exc}",
                metadata={},
            )
        return ResolvedStructure(
            structure=structure,
            source=StructureSource.LOCAL_FILE.value,
            source_detail=str(structure_path),
            status="resolved",
            message="已成功解析本地结构文件。",
            metadata=metadata,
        )

    def _resolve_materials_project(self, spec: ExperimentSpec) -> ResolvedStructure:
        if not spec.structure_id or spec.structure_id == "TO_BE_CONFIRMED":
            return ResolvedStructure(
                structure=None,
                source=StructureSource.MATERIALS_PROJECT.value,
                source_detail=spec.structure_id,
                status="needs_confirmation",
                message="结构来源是 Materials Project，但缺少明确的 mp-id。",
                metadata={},
            )

        api_key = resolve_api_key(
            Path.cwd(),
            aliases=("materials_project", "mp", "MP_API_KEY", "MATERIALS_PROJECT_API_KEY"),
            env_names=("MP_API_KEY", "MATERIALS_PROJECT_API_KEY"),
        )
        if not api_key:
            return ResolvedStructure(
                structure=None,
                source=StructureSource.MATERIALS_PROJECT.value,
                source_detail=spec.structure_id,
                status="missing_api_key",
                message="未找到 MP_API_KEY 或 MATERIALS_PROJECT_API_KEY，无法从 Materials Project 获取结构。",
                metadata={},
            )

        try:
            from mp_api.client import MPRester

            with MPRester(api_key) as rester:
                structure = rester.get_structure_by_material_id(spec.structure_id)
        except Exception as exc:
            return ResolvedStructure(
                structure=None,
                source=StructureSource.MATERIALS_PROJECT.value,
                source_detail=spec.structure_id,
                status="error",
                message=f"从 Materials Project 获取结构失败: {exc}",
                metadata={},
            )

        return ResolvedStructure(
            structure=structure,
            source=StructureSource.MATERIALS_PROJECT.value,
            source_detail=spec.structure_id,
            status="resolved",
            message="已成功从 Materials Project 获取结构。",
            metadata={"input_format": "materials_project"},
        )

    def _load_local_structure(self, structure_path: Path) -> tuple[Structure, dict[str, Any]]:
        suffix = structure_path.suffix.lower()
        metadata = self._detect_local_templates(structure_path)
        metadata["input_path"] = str(structure_path)
        metadata["input_format"] = suffix.lstrip(".") or "unknown"

        if suffix == ".xsd":
            structure, xsd_metadata = self._load_xsd_structure(structure_path)
            metadata.update(xsd_metadata)
            return structure, metadata

        if structure_path.name.upper() in {"POSCAR", "CONTCAR"}:
            poscar = Poscar.from_file(structure_path)
            structure = poscar.structure
            selective = getattr(poscar, "selective_dynamics", None)
            if selective is not None:
                structure.add_site_property("selective_dynamics", [list(flags) for flags in selective])
                metadata["fixed_atom_count"] = sum(
                    1 for flags in selective if not any(bool(flag) for flag in flags)
                )
                metadata["has_selective_dynamics"] = True
            return structure, metadata

        structure = Structure.from_file(structure_path)
        return structure, metadata

    @staticmethod
    def _detect_local_templates(structure_path: Path) -> dict[str, Any]:
        template_dir = structure_path.parent
        metadata: dict[str, Any] = {
            "template_dir": str(template_dir),
            "template_incar_path": None,
            "template_kpoints_path": None,
        }

        incar_path = template_dir / "INCAR"
        if incar_path.exists():
            metadata["template_incar_path"] = str(incar_path)

        kpoints_path = template_dir / "KPOINTS"
        if kpoints_path.exists():
            metadata["template_kpoints_path"] = str(kpoints_path)

        return metadata

    def _load_xsd_structure(self, structure_path: Path) -> tuple[Structure, dict[str, Any]]:
        try:
            from ase.io import read as ase_read
            from pymatgen.io.ase import AseAtomsAdaptor
        except Exception as exc:
            raise RuntimeError(
                f".xsd 导入依赖 ASE/Pymatgen-ASE 适配链，但当前环境不可用: {exc}"
            ) from exc

        atoms = ase_read(structure_path)
        atom_count = len(atoms)
        fixed_flags = self._get_xsd_constraints(structure_path, atom_count)

        sorted_indices = atoms.numbers.argsort()
        sorted_atoms = atoms[sorted_indices]
        selective_dynamics: list[list[bool]] = []
        fixed_count = 0
        for sorted_index in sorted_indices:
            is_fixed = fixed_flags[int(sorted_index)]
            if is_fixed:
                fixed_count += 1
                selective_dynamics.append([False, False, False])
            else:
                selective_dynamics.append([True, True, True])

        structure = AseAtomsAdaptor.get_structure(sorted_atoms)
        structure.add_site_property("selective_dynamics", selective_dynamics)
        structure = Structure.from_sites(structure.sites)

        metadata = {
            "input_format": "xsd",
            "has_selective_dynamics": True,
            "fixed_atom_count": fixed_count,
            "atom_count": atom_count,
            "sorted_by_atomic_number": True,
        }
        return structure, metadata

    @staticmethod
    def _get_xsd_constraints(structure_path: Path, ase_atom_count: int) -> list[bool]:
        tree = ET.parse(structure_path)
        root = tree.getroot()

        symmetry_system = None
        for elem in root.iter():
            if elem.tag.endswith("SymmetrySystem"):
                symmetry_system = elem
                break

        target_node = symmetry_system if symmetry_system is not None else root
        xml_atoms = [elem for elem in target_node.iter() if elem.tag.endswith("Atom3d")]

        if len(xml_atoms) == ase_atom_count:
            return ["XYZ" in elem.get("RestrictedProperties", "") for elem in xml_atoms]

        if ase_atom_count > len(xml_atoms) > 0 and ase_atom_count % len(xml_atoms) == 0:
            base_fixed = ["XYZ" in elem.get("RestrictedProperties", "") for elem in xml_atoms]
            return base_fixed * (ase_atom_count // len(xml_atoms))

        real_atoms = [atom for atom in xml_atoms if "ImageOf" not in atom.attrib]
        if len(real_atoms) == ase_atom_count:
            return ["XYZ" in elem.get("RestrictedProperties", "") for elem in real_atoms]

        return [False] * ase_atom_count
