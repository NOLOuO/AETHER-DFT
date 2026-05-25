from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dft_app.storage import RecordStore


def load_run_detail_view(store: RecordStore, run_root: Path) -> dict[str, Any]:
    payload = _load_payload(store, run_root)

    workflow_status = None
    if payload.get("adsorption_workflow_bundle") is not None:
        from dft_app.cli.main import collect_adsorption_workflow_status

        workflow_status = collect_adsorption_workflow_status(run_root)

    candidate_state = _build_candidate_state(payload)
    explain_state = _build_explain_state(run_root, payload)
    backflow_state = _build_backflow_state(run_root, payload)
    next_step_cards = _build_next_step_cards(candidate_state, workflow_status, explain_state, backflow_state)
    user_flow_steps = _build_user_flow_steps(candidate_state, workflow_status, explain_state, backflow_state)

    return {
        "payload": payload,
        "workflow_status": workflow_status,
        "candidate_state": candidate_state,
        "explain_state": explain_state,
        "backflow_state": backflow_state,
        "next_step_cards": next_step_cards,
        "user_flow_steps": user_flow_steps,
    }


def _load_payload(store: RecordStore, run_root: Path) -> dict[str, Any]:
    record = store.load_run_record(run_root)
    return {
        "run_record": record.to_dict(),
        "experiment_spec": store.read_metadata_file(run_root, "experiment_spec.json"),
        "experiment_plan": store.read_metadata_file(run_root, "experiment_plan.json"),
        "build_summary": store.read_metadata_file(run_root, "build_summary.json"),
        "parsed_result": store.read_metadata_file(run_root, "parsed_result.json"),
        "analysis_summary": store.read_metadata_file(run_root, "analysis_summary.json"),
        "adsorption_candidate_generation": store.read_metadata_file(run_root, "adsorption_candidate_generation.json"),
        "adsorption_selection": store.read_metadata_file(run_root, "adsorption_selection.json"),
        "adsorption_workflow_bundle": store.read_metadata_file(run_root, "adsorption_workflow_bundle.json"),
        "adsorption_workflow_status": store.read_metadata_file(run_root, "adsorption_workflow_status.json"),
        "dft_tools_explain_result": store.read_metadata_file(run_root, "dft_tools_explain_result.json"),
        "dft_tools_knowledge_backflow_payload": store.read_metadata_file(run_root, "dft_tools_knowledge_backflow_payload.json"),
        "dft_tools_kb_ingest_result": store.read_metadata_file(run_root, "dft_tools_kb_ingest_result.json"),
    }


def _build_candidate_state(payload: dict[str, Any]) -> dict[str, Any]:
    generation = payload.get("adsorption_candidate_generation") or {}
    selection = payload.get("adsorption_selection") or {}
    bundle = payload.get("adsorption_workflow_bundle") or {}
    manifest = _load_candidate_manifest(generation)

    selected_candidate_id = (
        selection.get("selected_candidate_id")
        or selection.get("candidate_id")
        or bundle.get("selected_candidate_id")
    )
    candidate_ids = [
        item.get("candidate_id")
        for item in (manifest.get("candidates") or [])
        if isinstance(item, dict) and item.get("candidate_id")
    ]

    if selected_candidate_id:
        status = "selected"
        next_step = "候选已选定，可继续进入 workflow bundle / submit 阶段。"
    elif generation:
        status = "needs_selection"
        next_step = "请先在本页选择一个 candidate，然后再继续 workflow。"
    else:
        status = "missing"
        next_step = "当前 run 尚未生成 adsorption candidates。"

    return {
        "status": status,
        "candidate_count": generation.get("candidate_count") or manifest.get("candidate_count") or len(candidate_ids),
        "manifest_json": (generation.get("manifest") or {}).get("manifest_json"),
        "manifest_md": (generation.get("manifest") or {}).get("manifest_md"),
        "selected_candidate_id": selected_candidate_id,
        "candidate_ids": candidate_ids[:8],
        "next_step": next_step,
        "selection_payload": selection,
    }


def _load_candidate_manifest(generation: dict[str, Any]) -> dict[str, Any]:
    manifest_json = (generation.get("manifest") or {}).get("manifest_json")
    if not manifest_json:
        return {}
    path = Path(str(manifest_json))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_explain_state(run_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    explain = payload.get("dft_tools_explain_result")
    parsed_result = payload.get("parsed_result")
    analysis_summary = payload.get("analysis_summary")
    summary_path = run_root / "report" / "dft_tools_explain_summary.md"
    if explain:
        return {
            "status": "completed",
            "message": "dft_tools explain 已完成，可查看解释结果。",
            "result_path": str(summary_path) if summary_path.exists() else None,
        }
    if parsed_result or analysis_summary:
        return {
            "status": "ready",
            "message": "当前 run 已具备 explain 前置条件，可执行 dft_tools explain。",
            "result_path": None,
        }
    return {
        "status": "blocked",
        "message": "需先完成 parse / analyze，才能执行 dft_tools explain。",
        "result_path": None,
    }


def _build_backflow_state(run_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    kb_ingest = payload.get("dft_tools_kb_ingest_result")
    explain = payload.get("dft_tools_explain_result")
    if kb_ingest:
        status = kb_ingest.get("status") or "unknown"
        message = (
            "知识库回流已完成。"
            if status in {"ingested", "completed", "ok"}
            else kb_ingest.get("error") or "知识库回流已执行。"
        )
        return {
            "status": status,
            "message": message,
            "result_path": str(run_root / "metadata" / "dft_tools_kb_ingest_result.json"),
        }
    if explain:
        return {
            "status": "ready",
            "message": "已有 explain 结果，可继续执行 knowledge backflow。",
            "result_path": None,
        }
    return {
        "status": "blocked",
        "message": "需先完成 dft_tools explain，才能继续回流知识库。",
        "result_path": None,
    }


def _build_next_step_cards(
    candidate_state: dict[str, Any],
    workflow_status: dict[str, Any] | None,
    explain_state: dict[str, Any],
    backflow_state: dict[str, Any],
) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    if candidate_state["status"] == "needs_selection":
        cards.append({
            "title": "先完成 candidate 选择",
            "description": candidate_state["next_step"],
        })
    elif workflow_status and workflow_status.get("recommended_next_steps"):
        for item in workflow_status["recommended_next_steps"]:
            cards.append({"title": "Workflow 下一步", "description": str(item)})
    if explain_state["status"] == "ready":
        cards.append({"title": "执行结果解释", "description": explain_state["message"]})
    if backflow_state["status"] == "ready":
        cards.append({"title": "执行知识回流", "description": backflow_state["message"]})
    if not cards:
        cards.append({"title": "当前状态稳定", "description": "当前没有新的自动动作，请查看各阶段结果与产物。"})
    return cards


def _build_user_flow_steps(
    candidate_state: dict[str, Any],
    workflow_status: dict[str, Any] | None,
    explain_state: dict[str, Any],
    backflow_state: dict[str, Any],
) -> list[dict[str, str]]:
    workflow_step_status = workflow_status.get("status") if workflow_status else "pending"
    return [
        {"name": "Candidate", "status": candidate_state["status"], "description": candidate_state["next_step"]},
        {"name": "Workflow", "status": workflow_step_status, "description": (workflow_status.get("recommended_next_steps") or ["查看 workflow 状态后决定下一步。"])[0] if workflow_status else "等待 workflow bundle。"},
        {"name": "Explain", "status": explain_state["status"], "description": explain_state["message"]},
        {"name": "Backflow", "status": backflow_state["status"], "description": backflow_state["message"]},
    ]
