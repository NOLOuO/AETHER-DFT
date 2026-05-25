from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ase.io import read

from dft_app.models import ExperimentSpec, PhaseStatus, PipelinePhase, RunRecord, RunStatus


REQUIRED_INPUT_FILES = ("POSCAR", "INCAR", "KPOINTS", "job.slurm")
SNAPSHOT_INPUT_FILES = (*REQUIRED_INPUT_FILES, "POTCAR", "POTCAR.mapping.json")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        raw = path.read_bytes()
        payload.update(
            {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "mtime_ns": path.stat().st_mtime_ns,
            }
        )
    return payload


def snapshot_inputs(run_root: Path) -> dict[str, dict[str, Any]]:
    inputs_dir = run_root / "inputs"
    return {name: _file_fingerprint(inputs_dir / name) for name in SNAPSHOT_INPUT_FILES}


def _parse_incar(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].split("!", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip().upper()] = value.strip().split()[0] if value.strip() else ""
    return result


def _expected_value_matches(actual: str | None, expected: Any) -> bool:
    if actual is None:
        return False
    actual_clean = str(actual).strip()
    expected_clean = str(expected).strip()
    if actual_clean.lower() == expected_clean.lower():
        return True
    try:
        return abs(float(actual_clean) - float(expected_clean)) <= 1e-12
    except Exception:
        return False


def _research_template_from_spec(spec: ExperimentSpec) -> dict[str, Any]:
    notes = spec.notes if isinstance(spec.notes, dict) else {}
    template = notes.get("research_template") if isinstance(notes.get("research_template"), dict) else {}
    return template or {}


def _input_snapshot_matches(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    for name in SNAPSHOT_INPUT_FILES:
        left_item = left.get(name) or {}
        right_item = right.get(name) or {}
        if bool(left_item.get("exists")) != bool(right_item.get("exists")):
            return False
        if left_item.get("exists") and left_item.get("sha256") != right_item.get("sha256"):
            return False
    return True


def load_existing_submission_evidence(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "metadata" / "pre_submit_gate.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def verify_submission_evidence(
    spec: ExperimentSpec,
    run_record: RunRecord,
    *,
    mode: str,
    remote_probe: dict[str, Any] | None = None,
    require_potcar: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    """Validate current evidence before submit without imposing a fixed workflow.

    The gate does not require callers to run a prescribed sequence first.  It
    inspects whatever exists now: build status, files, research-template notes,
    current file hashes, and optional remote probe evidence.  Submitters can
    therefore reuse already-built runs or fresh builds, but they cannot bypass
    the same evidence standard.
    """

    run_root = Path(run_record.run_root)
    inputs_dir = run_root / "inputs"
    metadata_dir = run_root / "metadata"
    blockers: list[str] = []
    warnings: list[str] = []

    build_phase = run_record.phases.get(PipelinePhase.BUILD.value)
    if build_phase is None or build_phase.status != PhaseStatus.COMPLETED:
        blockers.append("build 阶段未完成，不能提交。")
    if run_record.overall_status != RunStatus.READY:
        blockers.append(f"run 当前状态为 {run_record.overall_status.value}，不是 ready。")

    snapshot = snapshot_inputs(run_root)
    for name in REQUIRED_INPUT_FILES:
        if not snapshot[name]["exists"]:
            blockers.append(f"缺少 {name}")

    if require_potcar and not snapshot["POTCAR"]["exists"]:
        blockers.append("缺少 POTCAR，且当前提交要求真实 POTCAR。")
    elif not snapshot["POTCAR"]["exists"]:
        if snapshot["POTCAR.mapping.json"]["exists"]:
            warnings.append("未生成真实 POTCAR；存在 POTCAR.mapping.json，需确认集群端赝势库可用。")
        else:
            warnings.append("POTCAR 与 POTCAR.mapping.json 都不存在，提交前需确认赝势来源。")

    incar = _parse_incar(inputs_dir / "INCAR")
    template = _research_template_from_spec(spec)
    if template.get("requires_template_review"):
        blockers.append("research 模板源文件已变化，需要模型重新读取 research 并确认模板后才能提交。")
    expected_incar = dict(template.get("expected_incar") or {})
    severity_by_key = {str(key).upper(): str(value) for key, value in (template.get("severity_by_key") or {}).items()}
    if expected_incar:
        for key, expected in expected_incar.items():
            key_upper = str(key).upper()
            if _expected_value_matches(incar.get(key_upper), expected):
                continue
            message = f"research 模板期望 {key_upper}={expected}，当前为 {incar.get(key_upper, '<missing>')}"
            if severity_by_key.get(key_upper, "warning") == "blocker":
                blockers.append(message)
            else:
                warnings.append(message)
    else:
        warnings.append("当前 ExperimentSpec 没有可机读 research_template.expected_incar；提交前需模型说明模板来源。")

    if "MAGMOM" not in incar:
        warnings.append("INCAR 未显式 MAGMOM；若体系涉及自旋/开壳层/缺陷需补充或说明不需要。")

    poscar_path = inputs_dir / "POSCAR"
    if poscar_path.exists():
        try:
            atoms = read(poscar_path)
            if len(atoms) <= 0:
                blockers.append("POSCAR 原子数为 0。")
        except Exception as exc:
            blockers.append(f"POSCAR 不可读取: {exc}")

    if mode == "remote_submit":
        if not remote_probe:
            blockers.append("远程提交缺少 cluster_probe 证据。")
        elif remote_probe.get("status") != "ok":
            blockers.append(f"cluster_probe 未通过: {remote_probe.get('status')} - {remote_probe.get('message')}")

    previous = load_existing_submission_evidence(run_root)
    previous_reused = bool(
        previous
        and previous.get("status") == "ready"
        and _input_snapshot_matches(previous.get("input_snapshot"), snapshot)
    )

    status = "ready" if not blockers else "blocked"
    report = {
        "status": status,
        "mode": mode,
        "checked_at": _utcnow(),
        "run_root": str(run_root),
        "inputs_dir": str(inputs_dir),
        "input_snapshot": snapshot,
        "previous_ready_evidence_reused": previous_reused,
        "research_template": {
            "template_found": template.get("template_found"),
            "template_id": template.get("template_id"),
            "requires_template_review": template.get("requires_template_review", False),
            "expected_incar": expected_incar,
            "severity_by_key": severity_by_key,
        },
        "remote_probe": remote_probe,
        "blockers": blockers,
        "warnings": warnings,
        "principle": "自适应证据门槛：不要求固定步骤顺序，但提交时必须证明当前输入包与 research/集群证据一致。",
    }
    if write_report:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "pre_submit_gate.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return report
