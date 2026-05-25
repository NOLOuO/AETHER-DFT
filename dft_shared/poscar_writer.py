from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian

from .constants import POSCAR_DEFAULT_DECIMALS, POSCAR_SNAP_TOLERANCE


def write_poscar(
    atoms: Atoms,
    output_path: str | Path,
    *,
    title: str = "Generated",
    decimals: int = POSCAR_DEFAULT_DECIMALS,
    snap_tolerance: float = POSCAR_SNAP_TOLERANCE,
    boundary_one_to_zero: bool = False,
) -> None:
    output = Path(output_path)
    lines: list[str] = []
    lines.append(title)
    lines.append("1.0")

    for vector in atoms.cell:
        lines.append(f"{vector[0]:12.{decimals}f} {vector[1]:12.{decimals}f} {vector[2]:12.{decimals}f}")

    ordered_symbols, counts = _species_summary(atoms)
    lines.append(" ".join(ordered_symbols))
    lines.append(" ".join(str(count) for count in counts))

    lines.append("Selective Dynamics")
    lines.append("Direct")

    selective_dynamics = _extract_selective_dynamics(atoms)
    scaled_positions = atoms.get_scaled_positions(wrap=False)

    for index, frac in enumerate(scaled_positions):
        values = [_normalize_fractional(value, snap_tolerance, boundary_one_to_zero=boundary_one_to_zero) for value in frac]
        flags = " ".join("F" if fixed else "T" for fixed in selective_dynamics[index])
        lines.append(f"{values[0]:12.{decimals}f} {values[1]:12.{decimals}f} {values[2]:12.{decimals}f} {flags}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _species_summary(atoms: Atoms) -> tuple[list[str], list[int]]:
    symbols = atoms.get_chemical_symbols()
    if not symbols:
        return [], []

    ordered_symbols = [symbols[0]]
    counts = [1]
    for symbol in symbols[1:]:
        if symbol == ordered_symbols[-1]:
            counts[-1] += 1
            continue
        ordered_symbols.append(symbol)
        counts.append(1)
    return ordered_symbols, counts


def _extract_selective_dynamics(atoms: Atoms) -> list[tuple[bool, bool, bool]]:
    fixed_mask = [(False, False, False) for _ in range(len(atoms))]
    constraints = atoms.constraints or []
    for constraint in constraints:
        if isinstance(constraint, FixAtoms):
            for index in constraint.get_indices():
                fixed_mask[int(index)] = (True, True, True)
        elif isinstance(constraint, FixCartesian):
            mask = tuple(bool(x) for x in constraint.mask.tolist())
            for index in constraint.get_indices():
                merged = tuple(old or new for old, new in zip(fixed_mask[int(index)], mask))
                fixed_mask[int(index)] = merged
    return fixed_mask


def _normalize_fractional(value: float, snap_tolerance: float, *, boundary_one_to_zero: bool = False) -> float:
    normalized = float(value)
    if normalized < 0.0 or normalized > 1.0:
        normalized = normalized % 1.0

    if abs(normalized) < snap_tolerance:
        return 0.0
    if abs(normalized - 1.0) < snap_tolerance:
        return 0.0 if boundary_one_to_zero else 1.0
    if abs(normalized + 1.0) < snap_tolerance:
        return 0.0
    return normalized
