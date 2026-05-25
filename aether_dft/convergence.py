"""Convergence-test planning primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import ensure_runtime_dir


def compose_convergence_plan(
    *,
    material: str,
    property_target: str = "energy",
    encut_values: list[int] | None = None,
    kpoint_grids: list[list[int]] | None = None,
    force_threshold_ev_a: float = 0.03,
    energy_tolerance_mev_atom: float = 5.0,
    project: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic convergence matrix for model review.

    The plan is a calculation design artifact only; it does not submit or mark
    calculations as completed.
    """
    material_clean = material.strip()
    if not material_clean:
        raise ValueError("material 不能为空。")
    encuts = encut_values or [350, 400, 450, 500]
    grids = kpoint_grids or [[3, 3, 1], [4, 4, 1], [5, 5, 1]]
    matrix = [
        {
            "case_id": f"encut_{encut}_k_{grid[0]}x{grid[1]}x{grid[2]}",
            "encut": int(encut),
            "kpoints": [int(v) for v in grid],
            "property_target": property_target,
        }
        for encut in encuts
        for grid in grids
    ]
    plan = {
        "status": "planned",
        "kind": "convergence_plan",
        "project": project,
        "material": material_clean,
        "property_target": property_target,
        "criteria": {
            "force_threshold_ev_a": force_threshold_ev_a,
            "energy_tolerance_mev_atom": energy_tolerance_mev_atom,
        },
        "matrix": matrix,
        "next_step": "为每个 case 生成真实 VASP 输入并执行；完成后比较能量/力变化，不要在执行前声称收敛。",
    }
    if output_dir:
        target = Path(output_dir)
    else:
        target = ensure_runtime_dir("convergence_plans") / material_clean.replace("/", "_").replace(" ", "_")
    target.mkdir(parents=True, exist_ok=True)
    import json

    plan_path = target / "convergence_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    plan["plan_path"] = str(plan_path)
    return plan
