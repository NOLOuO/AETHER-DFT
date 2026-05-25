from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request

from dft_app.storage import RecordStore


DEFAULT_DFT_TOOLS_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_AUTO_INGEST = True


def build_dft_tools_manual_payload(store: RecordStore, run_root: Path) -> dict[str, Any]:
    experiment_spec = store.read_metadata_file(run_root, "experiment_spec.json") or {}
    parsed_result = store.read_metadata_file(run_root, "parsed_result.json") or {}
    analysis_summary = store.read_metadata_file(run_root, "analysis_summary.json") or {}
    build_summary = store.read_metadata_file(run_root, "build_summary.json") or {}
    structure_resolution = store.read_metadata_file(run_root, "structure_resolution.json") or {}
    run_record = store.read_metadata_file(run_root, "run_record.json") or {}

    task_goal_text = (
        experiment_spec.get("task_goal")
        or experiment_spec.get("source_prompt")
        or experiment_spec.get("description")
        or ""
    )

    structure_context = {
        "material_name": experiment_spec.get("material_name"),
        "task_type": experiment_spec.get("task_type"),
        "structure_source": experiment_spec.get("structure_source"),
        "structure_path": experiment_spec.get("structure_path"),
        "structure_resolution": structure_resolution,
        "build_summary": build_summary,
    }

    input_context = {
        "workflow": experiment_spec.get("workflow") or [],
        "functional": experiment_spec.get("functional"),
        "kpoints_strategy": experiment_spec.get("kpoints_strategy"),
        "encut_strategy": experiment_spec.get("encut_strategy"),
        "smearing": experiment_spec.get("smearing"),
        "spin_settings": experiment_spec.get("spin_settings"),
        "convergence_settings": experiment_spec.get("convergence_settings"),
        "submit_profile": experiment_spec.get("submit_profile"),
        "job_overrides": experiment_spec.get("job_overrides"),
    }

    result_summary = {
        "task_id": experiment_spec.get("task_id") or run_record.get("task_id"),
        "run_id": run_record.get("run_id"),
        "status_assessment": (
            (analysis_summary.get("analysis_summary") or {}).get("convergence_assessment")
            or analysis_summary.get("status")
            or ("completed" if parsed_result.get("completed") else "incomplete")
        ),
        "completed": parsed_result.get("completed"),
        "converged": parsed_result.get("converged"),
        "failure_reason": parsed_result.get("raw_summary", {}).get("failure_reason"),
        "total_energy": parsed_result.get("total_energy"),
        "energy_per_atom": parsed_result.get("energy_per_atom"),
        "band_gap": parsed_result.get("band_gap"),
        "efermi": parsed_result.get("efermi"),
        "max_force": parsed_result.get("max_force"),
        "ionic_steps": parsed_result.get("ionic_steps"),
        "electronic_steps": parsed_result.get("electronic_steps"),
        "warnings": parsed_result.get("warnings") or [],
        "recommended_actions": (analysis_summary.get("analysis_summary") or {}).get("recommended_actions") or [],
    }

    return {
        "task_name": run_record.get("task_id") or experiment_spec.get("task_id") or run_root.name,
        "task_goal_text": task_goal_text,
        "structure_context": structure_context,
        "input_context": input_context,
        "result_summary": result_summary,
    }


def request_dft_tools_explain(base_url: str, payload: dict[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
    target_url = base_url.rstrip("/") + "/api/explain/manual"
    req = request.Request(
        url=target_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"dft_tools explain HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"dft_tools explain 不可达: {exc.reason}") from exc


def build_dft_tools_kb_ingest_payload(
    explain_request_payload: dict[str, Any],
    knowledge_backflow_payload: dict[str, Any],
) -> dict[str, Any]:
    result_summary = explain_request_payload.get("result_summary") or {}
    structure_context = explain_request_payload.get("structure_context") or {}
    input_context = explain_request_payload.get("input_context") or {}
    status = "unknown"
    completed = bool(result_summary.get("completed"))
    converged = bool(result_summary.get("converged"))
    if completed and converged:
        status = "success"
    elif completed:
        status = "failed"

    tags = [
        "auto_dft",
        "dft_tools_bridge",
    ]
    task_type = str(structure_context.get("task_type") or "unknown")
    if task_type and task_type != "unknown":
        tags.append(f"task_type:{task_type}")
    workflow = input_context.get("workflow") or []
    tags.extend(f"workflow:{item}" for item in workflow if item)

    return {
        "task_name": str(
            knowledge_backflow_payload.get("task_name")
            or explain_request_payload.get("task_name")
            or "unknown_task"
        ),
        "task_type": task_type,
        "source_tool": "auto_dft",
        "completed": completed,
        "converged": converged,
        "status": status,
        "failure_reason": result_summary.get("failure_reason"),
        "total_energy": result_summary.get("total_energy"),
        "max_force": result_summary.get("max_force"),
        "structure_context": structure_context,
        "input_context": {
            **input_context,
            "task_goal_text": knowledge_backflow_payload.get("task_goal_text") or explain_request_payload.get("task_goal_text"),
            "dft_tools_explain_status": knowledge_backflow_payload.get("status_judgement"),
        },
        "tags": sorted({tag for tag in tags if tag}),
        "warnings": list(result_summary.get("warnings") or []),
        "recommended_actions": list(knowledge_backflow_payload.get("next_actions") or []),
    }


def request_dft_tools_kb_ingest(base_url: str, payload: dict[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
    target_url = base_url.rstrip("/") + "/api/kb/ingest"
    req = request.Request(
        url=target_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"dft_tools kb ingest HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"dft_tools kb ingest 不可达: {exc.reason}") from exc


def run_dft_tools_explain_bridge(
    store: RecordStore,
    run_root: Path,
    *,
    base_url: str | None = None,
    ingest_kb: bool | None = None,
) -> dict[str, Any]:
    resolved_base_url = (base_url or "").strip() or os.getenv("DFT_TOOLS_BASE_URL", DEFAULT_DFT_TOOLS_BASE_URL)
    resolved_auto_ingest = (
        ingest_kb
        if ingest_kb is not None
        else os.getenv("AUTO_DFT_DFT_TOOLS_AUTO_INGEST", "1" if DEFAULT_AUTO_INGEST else "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    payload = build_dft_tools_manual_payload(store, run_root)
    response = request_dft_tools_explain(resolved_base_url, payload)

    metadata_dir = run_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    explain_path = metadata_dir / "dft_tools_explain_result.json"
    backflow_path = metadata_dir / "dft_tools_knowledge_backflow_payload.json"
    kb_ingest_path = metadata_dir / "dft_tools_kb_ingest_result.json"
    markdown_path = run_root / "report" / "dft_tools_explain_summary.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    explain_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    knowledge_payload = response.get("knowledge_backflow_payload") if isinstance(response, dict) else None
    backflow_path.write_text(json.dumps(knowledge_payload or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_explain_markdown(response), encoding="utf-8")
    kb_ingest_result: dict[str, Any] = {
        "enabled": resolved_auto_ingest,
        "status": "skipped",
        "payload": None,
        "response": None,
        "error": None,
    }
    if resolved_auto_ingest and knowledge_payload:
        ingest_payload = build_dft_tools_kb_ingest_payload(payload, knowledge_payload)
        kb_ingest_result["payload"] = ingest_payload
        try:
            ingest_response = request_dft_tools_kb_ingest(resolved_base_url, ingest_payload)
            kb_ingest_result["status"] = "ingested"
            kb_ingest_result["response"] = ingest_response
        except Exception as exc:
            kb_ingest_result["status"] = "failed"
            kb_ingest_result["error"] = str(exc)
    elif resolved_auto_ingest:
        kb_ingest_result["status"] = "missing_knowledge_payload"
    kb_ingest_path.write_text(json.dumps(kb_ingest_result, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "base_url": resolved_base_url,
        "ingest_kb": resolved_auto_ingest,
        "request_payload": payload,
        "response": response,
        "kb_ingest_result": kb_ingest_result,
        "artifacts": {
            "explain_result_json": str(explain_path),
            "knowledge_backflow_json": str(backflow_path),
            "kb_ingest_result_json": str(kb_ingest_path),
            "markdown_summary": str(markdown_path),
        },
    }


def _render_explain_markdown(payload: dict[str, Any]) -> str:
    causes = payload.get("likely_causes") or []
    actions = payload.get("next_actions") or []
    evidence = payload.get("evidence_used") or []
    lines = [
        "# dft_tools 结果解释摘要",
        "",
        f"- 状态判断: {payload.get('status_judgement', 'N/A')}",
        f"- Provider / Model: {payload.get('provider', 'none')} / {payload.get('model', 'none')}",
        "",
        "## 主要原因",
    ]
    if causes:
        lines.extend(f"- {item}" for item in causes)
    else:
        lines.append("- 暂无")
    lines.extend(["", "## 下一步建议"])
    if actions:
        lines.extend(f"- {item}" for item in actions)
    else:
        lines.append("- 暂无")
    lines.extend(["", "## 证据依据"])
    if evidence:
        for item in evidence:
            if isinstance(item, dict):
                lines.append(f"- {item.get('source', 'evidence')}: {item.get('detail', '')}")
            else:
                lines.append(f"- {item}")
    else:
        lines.append("- 暂无")
    return "\n".join(lines) + "\n"
