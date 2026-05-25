from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar

from dft_app.models.experiment_spec import ExperimentSpec


class VaspInputGenerator:
    """Generate VASP inputs while preserving local template conventions when available."""

    def generate(
        self,
        spec: ExperimentSpec,
        structure: Structure,
        inputs_dir: Path,
        structure_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        structure_metadata = structure_metadata or {}
        poscar_path = inputs_dir / "POSCAR"
        incar_path = inputs_dir / "INCAR"
        kpoints_path = inputs_dir / "KPOINTS"
        potcar_map_path = inputs_dir / "POTCAR.mapping.json"
        potcar_path = inputs_dir / "POTCAR"

        selective_dynamics = structure.site_properties.get("selective_dynamics")
        poscar = Poscar(structure, selective_dynamics=selective_dynamics)
        poscar.write_file(poscar_path)

        incar = Incar(self._build_incar_dict(spec, structure, structure_metadata))
        incar.write_file(incar_path)

        kpoints = self._build_kpoints(spec, structure, structure_metadata)
        kpoints.write_file(kpoints_path)

        potcar_map = self._build_potcar_map(structure)
        potcar_map_path.write_text(
            json.dumps(potcar_map, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        potcar_generated = self._try_generate_potcar(potcar_map, potcar_path)

        return {
            "poscar_path": str(poscar_path),
            "incar_path": str(incar_path),
            "kpoints_path": str(kpoints_path),
            "potcar_map_path": str(potcar_map_path),
            "potcar_path": str(potcar_path) if potcar_generated else None,
            "potcar_map": potcar_map,
            "potcar_generated": potcar_generated,
            "template_incar_path": structure_metadata.get("template_incar_path"),
            "template_kpoints_path": structure_metadata.get("template_kpoints_path"),
        }

    def _build_incar_dict(
        self,
        spec: ExperimentSpec,
        structure: Structure,
        structure_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        template_incar_path = structure_metadata.get("template_incar_path")
        if template_incar_path and Path(template_incar_path).exists():
            incar: dict[str, Any] = dict(Incar.from_file(template_incar_path))
        else:
            incar = {
                "PREC": "Accurate",
                "EDIFF": spec.convergence_settings.ediff,
                "EDIFFG": spec.convergence_settings.ediffg,
                "NSW": spec.convergence_settings.nsw,
                "ISMEAR": spec.smearing.ismear,
                "SIGMA": spec.smearing.sigma,
                "ENCUT": spec.encut_strategy.value
                if spec.encut_strategy.mode == "fixed"
                else 520,
            }
            if spec.task_type.value in {"relax", "geometry_optimization"}:
                incar["IBRION"] = 2
                incar["ISIF"] = 3
            elif spec.task_type.value in {"relax_scf", "relax_scf_band", "defect_doping", "spin_related"}:
                incar["IBRION"] = 2
                incar["ISIF"] = 3
                incar["LWAVE"] = True
                incar["LCHARG"] = True
            elif spec.task_type.value in {
                "single_point",
                "static_refinement",
                "band_structure",
                "dos",
                "pdos",
                "charge_analysis",
                "work_function",
            }:
                incar["IBRION"] = -1
                incar["NSW"] = 0
                incar["LCHARG"] = True
                incar["LWAVE"] = True
                if spec.task_type.value in {"dos", "pdos"}:
                    incar["NEDOS"] = 2000
                if spec.task_type.value == "pdos":
                    incar["LORBIT"] = 11
                if spec.task_type.value == "charge_analysis":
                    incar["LAECHG"] = True
                if spec.task_type.value == "work_function":
                    incar["LVHAR"] = True
            elif spec.task_type.value == "vibrational_frequency":
                incar["IBRION"] = 5
                incar["NFREE"] = 2
                incar["NSW"] = 1
                incar["ISIF"] = 2
            elif spec.task_type.value == "transition_state_search":
                incar["IBRION"] = 3
                incar["ICHAIN"] = 2
                incar["POTIM"] = 0
                incar["ISIF"] = 2
                incar["NSW"] = 500
            elif spec.task_type.value == "molecular_dynamics":
                incar["IBRION"] = 0
                incar["POTIM"] = 1.0
                incar["SMASS"] = 0
                incar["TEBEG"] = 300
                incar["TEEND"] = 300
            elif spec.task_type.value in {"encut_convergence", "kpoints_convergence", "eos"}:
                incar["IBRION"] = 2
                incar["ISIF"] = 3

        # --- 表面/吸附体系参数覆盖 ---
        system_role = (
            spec.notes.get("system_role")
            if isinstance(spec.notes, dict)
            else None
        )
        if system_role in ("slab", "adsorbate_slab", "molecule"):
            # 表面 slab / 孤立分子均不应改变晶胞体积
            incar["ISIF"] = 2
        if system_role in ("slab", "adsorbate_slab"):
            # 表面体系需要偶极校正
            incar.setdefault("LDIPOL", True)
            incar.setdefault("IDIPOL", 3)

        incar.setdefault("EDIFF", spec.convergence_settings.ediff)
        if spec.convergence_settings.ediffg is not None:
            incar.setdefault("EDIFFG", spec.convergence_settings.ediffg)
        incar.setdefault("NSW", spec.convergence_settings.nsw)
        incar.setdefault("ISMEAR", spec.smearing.ismear)
        incar.setdefault("SIGMA", spec.smearing.sigma)

        if spec.encut_strategy.mode == "fixed" and spec.encut_strategy.value is not None:
            incar["ENCUT"] = spec.encut_strategy.value

        if spec.task_type.value == "spin_related":
            incar.setdefault("ISPIN", 2)

        if not spec.spin_settings.is_spin_polarized and "MAGMOM" not in spec.incar_overrides:
            incar["MAGMOM"] = self._build_default_magmom(structure)

        incar.update(spec.incar_overrides)
        return incar

    def _build_kpoints(
        self,
        spec: ExperimentSpec,
        structure: Structure,
        structure_metadata: dict[str, Any],
    ) -> Kpoints:
        template_kpoints_path = structure_metadata.get("template_kpoints_path")
        if template_kpoints_path and Path(template_kpoints_path).exists():
            return Kpoints.from_file(template_kpoints_path)

        strategy = spec.kpoints_strategy
        if strategy.mode == "explicit_mesh" and isinstance(strategy.value, (list, tuple)):
            kpts = tuple(int(v) for v in strategy.value)
            return Kpoints.gamma_automatic(kpts=kpts)

        density = 40
        if strategy.mode == "auto_density" and isinstance(strategy.value, int):
            density = strategy.value

        a = structure.lattice.a
        b = structure.lattice.b
        c = structure.lattice.c
        volume = structure.lattice.volume
        factor = density * ((a * b * c) / volume) ** (1 / 3)
        kpts = (
            max(math.ceil(factor / a), 1),
            max(math.ceil(factor / b), 1),
            max(math.ceil(factor / c), 1),
        )
        return Kpoints.gamma_automatic(kpts=kpts)

    @staticmethod
    def _build_default_magmom(structure: Structure) -> str:
        grouped_counts: list[tuple[str, int]] = []
        for site in structure:
            symbol = site.specie.symbol
            if grouped_counts and grouped_counts[-1][0] == symbol:
                prev_symbol, prev_count = grouped_counts[-1]
                grouped_counts[-1] = (prev_symbol, prev_count + 1)
            else:
                grouped_counts.append((symbol, 1))
        return " ".join(f"{count}*0" for _, count in grouped_counts)

    @staticmethod
    def _try_generate_potcar(potcar_map: dict[str, str], potcar_path: Path) -> bool:
        potcar_dir = os.getenv("SEMI_DFT_POTCAR_DIR")
        if not potcar_dir:
            return False

        potcar_root = Path(potcar_dir)
        chunks: list[str] = []
        for mapped_symbol in potcar_map.values():
            candidate = potcar_root / f"POTCAR.{mapped_symbol}"
            if not candidate.exists():
                return False
            chunks.append(candidate.read_text(encoding="utf-8", errors="ignore"))

        potcar_path.write_text("".join(chunks), encoding="utf-8")
        return True

    def _build_potcar_map(self, structure: Structure) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for site in structure:
            symbol = site.specie.symbol
            if symbol not in mapping:
                mapping[symbol] = symbol
        return mapping
