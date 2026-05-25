"""迁移脚本：将旧版 vasp_result_analyzer/knowledge_base 数据导入共享知识库。

用法::
    python -m dft_shared.knowledge_base.migrate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .models import TaskRecord
from .store import KnowledgeStore

OLD_KB_DIR = PROJECT_ROOT / "vasp_result_analyzer" / "knowledge_base"
OLD_TASK_SNAPSHOTS = OLD_KB_DIR / "task_snapshots"
OLD_SUCCESS_SNAPSHOTS = OLD_KB_DIR / "successful_case_snapshots"


def migrate(*, target_store: KnowledgeStore | None = None) -> dict[str, int]:
    if target_store is None:
        target_store = KnowledgeStore()
    stats = {"task_snapshots": 0, "success_snapshots": 0, "skipped": 0}

    for snap_dir, key, converter in [
        (OLD_TASK_SNAPSHOTS, "task_snapshots", _convert_old_record),
        (OLD_SUCCESS_SNAPSHOTS, "success_snapshots", _convert_old_success),
    ]:
        if not snap_dir.exists():
            continue
        for path in sorted(snap_dir.glob("*.latest.json")):
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                record = converter(old)
                target_store.ingest(record)
                stats[key] += 1
            except Exception as e:
                print(f"  skip {path.name}: {e}")
                stats["skipped"] += 1
    return stats


def _merge_context(structure_ctx, model_ctx, local_env):
    merged = dict(structure_ctx)
    merged["structure_role"] = model_ctx.get("structure_role")
    merged["vacuum_axis"] = model_ctx.get("vacuum_axis")
    merged["vacuum_thickness"] = model_ctx.get("vacuum_thickness_estimate")
    merged["fixed_atom_count"] = model_ctx.get("fixed_atom_count")
    merged["fixed_atom_fraction"] = model_ctx.get("fixed_atom_fraction")
    sig = (local_env.get("surface_site_signature") or {})
    for k in ("substrate_species", "substrate_family", "adsorbate_species",
              "adsorbate_family", "site_family"):
        merged[k] = sig.get(k)
    merged["site_signature"] = sig.get("signature_label")
    return merged


def _convert_old_record(old):
    raw = old.get("raw_summary") or {}
    merged = _merge_context(
        raw.get("structure_context") or {},
        raw.get("model_context") or {},
        raw.get("local_environment_context") or {},
    )
    completed, converged = bool(old.get("completed")), bool(old.get("converged"))
    status = "success" if completed and converged else ("failed" if completed else "unknown")
    return TaskRecord(
        task_name=old.get("task_name", ""), task_type=old.get("task_family", "unknown"),
        source_tool="vasp_result_analyzer", completed=completed, converged=converged,
        status=status, failure_reason=old.get("failure_reason"),
        total_energy=old.get("total_energy"), energy_per_atom=old.get("energy_per_atom"),
        max_force=old.get("max_force"), efermi=old.get("efermi"), band_gap=old.get("band_gap"),
        ionic_steps=old.get("ionic_steps"), electronic_steps=old.get("electronic_steps"),
        structure_context=merged, input_context=raw.get("input_context") or {},
        warnings=old.get("warnings") or [], recommended_actions=old.get("recommended_actions") or [],
        remote_job_dir=old.get("remote_job_dir"), parsed_at=old.get("parsed_at"),
        version=old.get("version", 1), raw=raw,
    )


def _convert_old_success(old):
    merged = _merge_context(
        old.get("structure_context") or {},
        old.get("model_context") or {},
        old.get("local_environment_context") or {},
    )
    return TaskRecord(
        task_name=old.get("task_name", ""), task_type=old.get("task_family", "unknown"),
        source_tool="vasp_result_analyzer", completed=True, converged=True, status="success",
        total_energy=old.get("total_energy"), energy_per_atom=old.get("energy_per_atom"),
        max_force=old.get("max_force"), efermi=old.get("efermi"), band_gap=old.get("band_gap"),
        structure_context=merged, input_context=old.get("input_context") or {},
        tags=list(old.get("case_tags") or []),
        remote_job_dir=old.get("remote_job_dir"), parsed_at=old.get("parsed_at"),
        version=old.get("version", 1),
    )


if __name__ == "__main__":
    print("migrating old KB -> shared KB ...")
    store = KnowledgeStore()
    print(f"  source: {OLD_KB_DIR}")
    print(f"  target: {store.data_dir}")
    stats = migrate(target_store=store)
    print(f"  done: {stats}")
