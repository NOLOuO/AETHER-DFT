from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Oszicar, Vasprun

from dft_app.models import (
    ExperimentSpec,
    LatticeParameters,
    ParsedResult,
    PipelinePhase,
    RunRecord,
)


@dataclass
class ParsedExecutionResult:
    status: str
    message: str
    parsed_result: ParsedResult | None
    details: dict[str, Any]


class VaspOutputParser:
    """Parse VASP outputs from a run workspace into ParsedResult."""

    def parse(self, spec: ExperimentSpec, run_record: RunRecord) -> ParsedExecutionResult:
        run_root = Path(run_record.run_root)
        located = self._locate_output_files(run_root)

        if not any(located.values()):
            message = "未找到 vasprun.xml、OUTCAR、OSZICAR 或 CONTCAR，当前还没有可解析的 VASP 输出。"
            run_record.block_phase(PipelinePhase.PARSE, message)
            return ParsedExecutionResult("blocked", message, None, {"source_files": {}})

        parsed_result = ParsedResult(
            task_id=spec.task_id,
            run_id=run_record.run_id,
            calc_type=spec.task_type.value,
            source_files={key: str(value) for key, value in located.items() if value is not None},
        )

        warnings: list[str] = []

        if located["vasprun.xml"] is not None:
            self._parse_vasprun(located["vasprun.xml"], parsed_result, warnings)
        if located["OUTCAR"] is not None:
            self._parse_outcar_text(located["OUTCAR"], parsed_result, warnings)
        if located["OSZICAR"] is not None:
            self._parse_oszicar(located["OSZICAR"], parsed_result, warnings)
        if located["CONTCAR"] is not None:
            self._parse_contcar(located["CONTCAR"], parsed_result, warnings)

        parsed_result.warnings = warnings
        parsed_result.raw_summary = {
            "source_files": parsed_result.source_files,
            "warnings_count": len(warnings),
        }

        if parsed_result.completed or parsed_result.total_energy is not None:
            message = "VASP 输出解析完成。"
            run_record.complete_phase(PipelinePhase.PARSE, message=message)
            run_record.mark_ready()
            status = "parsed"
        else:
            message = "找到输出文件，但未提取到足够结果，建议人工检查原始输出。"
            run_record.block_phase(PipelinePhase.PARSE, message)
            status = "partial"

        return ParsedExecutionResult(
            status=status,
            message=message,
            parsed_result=parsed_result,
            details={
                "source_files": parsed_result.source_files,
                "warnings": warnings,
            },
        )

    def _locate_output_files(self, run_root: Path) -> dict[str, Path | None]:
        filenames = ["vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR"]
        located: dict[str, Path | None] = {name: None for name in filenames}
        search_roots = [run_root / "outputs", run_root]

        for filename in filenames:
            for base in search_roots:
                if not base.exists():
                    continue
                direct = base / filename
                if direct.exists():
                    located[filename] = direct
                    break
                matches = list(base.rglob(filename))
                if matches:
                    located[filename] = matches[0]
                    break

        return located

    def _parse_vasprun(
        self, vasprun_path: Path, parsed_result: ParsedResult, warnings: list[str]
    ) -> None:
        try:
            vasprun = Vasprun(
                str(vasprun_path),
                parse_potcar_file=False,
                exception_on_bad_xml=False,
            )
        except Exception as exc:
            warnings.append(f"vasprun.xml 解析失败: {exc}")
            return

        parsed_result.completed = True
        parsed_result.converged = bool(getattr(vasprun, "converged", False))

        final_energy = getattr(vasprun, "final_energy", None)
        if final_energy is not None:
            parsed_result.total_energy = float(final_energy)

        final_structure = getattr(vasprun, "final_structure", None)
        if final_structure is not None:
            parsed_result.volume = float(final_structure.volume)
            parsed_result.lattice_parameters = LatticeParameters(
                a=float(final_structure.lattice.a),
                b=float(final_structure.lattice.b),
                c=float(final_structure.lattice.c),
                alpha=float(final_structure.lattice.alpha),
                beta=float(final_structure.lattice.beta),
                gamma=float(final_structure.lattice.gamma),
            )
            if parsed_result.total_energy is not None and len(final_structure) > 0:
                parsed_result.energy_per_atom = parsed_result.total_energy / len(final_structure)

        efermi = getattr(vasprun, "efermi", None)
        if efermi is not None:
            parsed_result.efermi = float(efermi)

        try:
            band_gap = float(vasprun.eigenvalue_band_properties[0])
            parsed_result.band_gap = band_gap
            parsed_result.is_metal = band_gap <= 1e-8
        except Exception:
            warnings.append("未能从 vasprun.xml 提取能带信息。")

        ionic_steps = getattr(vasprun, "ionic_steps", []) or []
        if ionic_steps:
            parsed_result.ionic_steps = len(ionic_steps)
            parsed_result.electronic_steps = sum(
                len(step.get("electronic_steps", []) or []) for step in ionic_steps
            )
            last_forces = ionic_steps[-1].get("forces")
            max_force = self._max_force_from_forces(last_forces)
            if max_force is not None:
                parsed_result.max_force = max_force

    def _parse_outcar_text(
        self, outcar_path: Path, parsed_result: ParsedResult, warnings: list[str]
    ) -> None:
        try:
            text = outcar_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            warnings.append(f"OUTCAR 读取失败: {exc}")
            return

        if not parsed_result.completed and "General timing and accounting informations for this job" in text:
            parsed_result.completed = True

        if not parsed_result.converged and "reached required accuracy" in text.lower():
            parsed_result.converged = True

        if parsed_result.total_energy is None:
            energy_matches = re.findall(r"TOTEN\s*=\s*([-\d\.Ee+]+)", text)
            if energy_matches:
                try:
                    parsed_result.total_energy = float(energy_matches[-1])
                except ValueError:
                    warnings.append("OUTCAR 中 TOTEN 解析失败。")

        if parsed_result.efermi is None:
            efermi_matches = re.findall(r"E-fermi\s*:\s*([-\d\.Ee+]+)", text)
            if efermi_matches:
                try:
                    parsed_result.efermi = float(efermi_matches[-1])
                except ValueError:
                    warnings.append("OUTCAR 中 E-fermi 解析失败。")

    def _parse_oszicar(
        self, oszicar_path: Path, parsed_result: ParsedResult, warnings: list[str]
    ) -> None:
        try:
            oszicar = Oszicar(str(oszicar_path))
        except Exception as exc:
            warnings.append(f"OSZICAR 解析失败: {exc}")
            return

        ionic_steps = getattr(oszicar, "ionic_steps", None)
        if parsed_result.ionic_steps is None and ionic_steps:
            parsed_result.ionic_steps = len(ionic_steps)

        electronic_steps = getattr(oszicar, "electronic_steps", None)
        if parsed_result.electronic_steps is None and electronic_steps:
            if isinstance(electronic_steps, list):
                parsed_result.electronic_steps = len(electronic_steps)

        final_energy = getattr(oszicar, "final_energy", None)
        if parsed_result.total_energy is None and final_energy is not None:
            try:
                parsed_result.total_energy = float(final_energy)
            except (TypeError, ValueError):
                warnings.append("OSZICAR 中 final_energy 解析失败。")

    def _parse_contcar(
        self, contcar_path: Path, parsed_result: ParsedResult, warnings: list[str]
    ) -> None:
        try:
            structure = Structure.from_file(contcar_path)
        except Exception as exc:
            warnings.append(f"CONTCAR 解析失败: {exc}")
            return

        if parsed_result.volume is None:
            parsed_result.volume = float(structure.volume)
        if parsed_result.lattice_parameters.a is None:
            parsed_result.lattice_parameters = LatticeParameters(
                a=float(structure.lattice.a),
                b=float(structure.lattice.b),
                c=float(structure.lattice.c),
                alpha=float(structure.lattice.alpha),
                beta=float(structure.lattice.beta),
                gamma=float(structure.lattice.gamma),
            )
        if parsed_result.total_energy is not None and len(structure) > 0 and parsed_result.energy_per_atom is None:
            parsed_result.energy_per_atom = parsed_result.total_energy / len(structure)

    @staticmethod
    def _max_force_from_forces(forces: Any) -> float | None:
        if forces is None:
            return None

        max_force = 0.0
        found = False
        for vector in forces:
            if vector is None:
                continue
            try:
                norm = math.sqrt(sum(float(component) ** 2 for component in vector))
            except (TypeError, ValueError):
                continue
            max_force = max(max_force, norm)
            found = True
        return max_force if found else None
