"""Project-local autonomous campaign tracking.

The campaign layer is deliberately a state board, not a fixed workflow engine.
It lets the model manage a computational search space: candidate structures,
cheap filters, run/job bindings, results, pruning decisions, and next batches.
The model still chooses which scientific tools to call next.
"""

from __future__ import annotations

from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from .project_state import project_paths


TERMINAL_CANDIDATE_STATES = {"discarded", "failed", "completed"}
RUNNING_CANDIDATE_STATES = {"submitted", "running", "queued"}
READY_CANDIDATE_STATES = {"candidate", "quality_pass", "ready", "promising", "retry"}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_id(value: str | None, *, prefix: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")
    return text or f"{prefix}_{uuid4().hex[:8]}"


def _campaign_root(project: str) -> Path:
    paths = project_paths(project)
    root = paths.root / ".aether" / "campaigns"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _campaign_path(project: str, campaign_id: str) -> Path:
    return _campaign_root(project) / f"{_safe_id(campaign_id, prefix='campaign')}.json"


def _load_path(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _save(project: str, campaign: dict[str, Any]) -> dict[str, Any]:
    campaign["updated_at"] = _now_iso()
    path = _campaign_path(project, str(campaign.get("campaign_id") or "campaign"))
    path.write_text(json.dumps(campaign, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "path": str(path), "campaign": campaign}


def list_campaigns(*, project: str, include_closed: bool = False, limit: int = 20) -> dict[str, Any]:
    project_clean = str(project or "").strip()
    if not project_clean:
        return {"status": "error", "message": "project 不能为空。", "campaigns": []}
    try:
        limit_int = max(1, min(int(limit or 20), 100))
    except (TypeError, ValueError):
        return {"status": "error", "message": "limit 必须是整数。", "campaigns": []}
    rows: list[dict[str, Any]] = []
    for path in sorted(_campaign_root(project_clean).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = _load_path(path)
        if not data:
            continue
        if not include_closed and str(data.get("status") or "").lower() in {"closed", "done", "cancelled", "canceled"}:
            continue
        rows.append(_campaign_summary(data))
        if len(rows) >= limit_int:
            break
    return {"status": "ok", "project": project_clean, "campaigns": rows}


def load_campaign(*, project: str, campaign_id: str | None = None) -> dict[str, Any]:
    project_clean = str(project or "").strip()
    if not project_clean:
        return {"status": "error", "message": "project 不能为空。"}
    if campaign_id:
        data = _load_path(_campaign_path(project_clean, campaign_id))
        if data:
            return {"status": "ok", "project": project_clean, "campaign": data, "summary": _campaign_summary(data)}
        return {"status": "missing", "message": f"未找到 campaign: {campaign_id}", "project": project_clean}
    listed = list_campaigns(project=project_clean, include_closed=False, limit=1)
    campaigns = listed.get("campaigns") or []
    if not campaigns:
        return {"status": "missing", "message": "当前项目没有 active campaign。", "project": project_clean}
    return load_campaign(project=project_clean, campaign_id=campaigns[0]["campaign_id"])


def start_campaign(
    *,
    project: str,
    goal: str,
    campaign_id: str | None = None,
    strategy: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_clean = str(project or "").strip()
    goal_clean = str(goal or "").strip()
    if not project_clean:
        return {"status": "error", "message": "project 不能为空。"}
    if not goal_clean:
        return {"status": "error", "message": "goal 不能为空。"}
    cid = _safe_id(campaign_id, prefix="campaign")
    now = _now_iso()
    campaign = {
        "version": 1,
        "campaign_id": cid,
        "project": project_clean,
        "goal": goal_clean,
        "strategy": str(strategy or "enumerate candidates -> cheap filters -> batch calculations -> prune/refine").strip(),
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "metadata": metadata if isinstance(metadata, dict) else {},
        "candidates": [],
        "events": [
            {
                "at": now,
                "event": "campaign_started",
                "note": "Campaign opened for computational search-space exploration.",
            }
        ],
    }
    saved = _save(project_clean, campaign)
    return {**saved, "summary": _campaign_summary(saved["campaign"])}


def register_candidates(
    *,
    project: str,
    campaign_id: str | None = None,
    candidates: list[dict[str, Any]] | None = None,
    source_manifest_path: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    payload = load_campaign(project=project, campaign_id=campaign_id)
    if payload.get("status") != "ok":
        return payload
    campaign = payload["campaign"]
    existing = {str(item.get("candidate_id") or ""): item for item in campaign.get("candidates", []) if isinstance(item, dict)}
    manifest_candidates = _load_manifest_candidates(source_manifest_path)
    merged_inputs: dict[str, dict[str, Any]] = {
        str(item.get("candidate_id") or ""): item
        for item in manifest_candidates
        if isinstance(item, dict) and str(item.get("candidate_id") or "")
    }
    for raw in candidates or []:
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get("candidate_id") or raw.get("id") or "").strip()
        if cid and cid in merged_inputs:
            merged_inputs[cid] = {**merged_inputs[cid], **raw}
        elif cid:
            merged_inputs[cid] = raw
        else:
            merged_inputs[f"__anon_{len(merged_inputs) + 1}"] = raw
    added: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    for index, raw in enumerate(merged_inputs.values(), start=1):
        if not isinstance(raw, dict):
            continue
        candidate_id = _safe_id(str(raw.get("candidate_id") or raw.get("id") or f"cand_{len(existing) + index:03d}"), prefix="cand")
        entry = {
            "candidate_id": candidate_id,
            "status": str(raw.get("status") or "candidate").strip() or "candidate",
            "structure_path": str(raw.get("structure_path") or raw.get("poscar_path") or "").strip(),
            "material": str(raw.get("material") or "").strip(),
            "adsorbate": str(raw.get("adsorbate") or "").strip(),
            "motif": str(raw.get("motif") or raw.get("site_label") or raw.get("site_family") or "").strip(),
            "orientation": str(raw.get("orientation") or raw.get("orientation_label") or "").strip(),
            "reason": str(raw.get("reason") or raw.get("rationale") or "").strip(),
            "quality_score": _maybe_float(raw.get("quality_score")),
            "priority": _maybe_float(raw.get("priority")),
            "run_id": str(raw.get("run_id") or "").strip(),
            "run_root": str(raw.get("run_root") or "").strip(),
            "job_id": str(raw.get("job_id") or "").strip(),
            "remote_run_root": str(raw.get("remote_run_root") or "").strip(),
            "result": raw.get("result") if isinstance(raw.get("result"), dict) else {},
            "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
            "created_at": raw.get("created_at") or _now_iso(),
            "updated_at": _now_iso(),
        }
        if source_manifest_path:
            entry["source_manifest_path"] = str(source_manifest_path)
        if candidate_id in existing:
            merged = {**existing[candidate_id], **{k: v for k, v in entry.items() if v not in ("", None, {})}}
            merged["updated_at"] = _now_iso()
            existing[candidate_id] = merged
            updated.append(merged)
        else:
            existing[candidate_id] = entry
            added.append(entry)
    campaign["candidates"] = list(existing.values())
    _append_event(
        campaign,
        "candidates_registered",
        note or f"added={len(added)} updated={len(updated)} manifest_imported={len(manifest_candidates)}",
    )
    saved = _save(str(campaign["project"]), campaign)
    return {
        **saved,
        "added": added,
        "updated": updated,
        "manifest_imported": len(manifest_candidates),
        "summary": _campaign_summary(saved["campaign"]),
        "guidance": "继续用 candidate_quality_score / dft_run_task / cluster_remote_submit 等工具推进；campaign 只追踪批量状态。",
    }


def update_candidate(
    *,
    project: str,
    campaign_id: str | None = None,
    candidate_id: str,
    status: str | None = None,
    quality_score: float | None = None,
    run_id: str | None = None,
    run_root: str | None = None,
    job_id: str | None = None,
    remote_run_root: str | None = None,
    result: dict[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    payload = load_campaign(project=project, campaign_id=campaign_id)
    if payload.get("status") != "ok":
        return payload
    campaign = payload["campaign"]
    cid = _safe_id(candidate_id, prefix="cand")
    found = None
    for item in campaign.get("candidates", []):
        if isinstance(item, dict) and str(item.get("candidate_id") or "") == cid:
            found = item
            break
    if found is None:
        return {"status": "missing", "message": f"未找到 candidate: {cid}", "campaign_id": campaign.get("campaign_id")}
    if status:
        found["status"] = str(status).strip()
    if quality_score is not None:
        found["quality_score"] = _maybe_float(quality_score)
    for key, value in {
        "run_id": run_id,
        "run_root": run_root,
        "job_id": job_id,
        "remote_run_root": remote_run_root,
    }.items():
        if value is not None:
            found[key] = str(value).strip()
    if isinstance(result, dict):
        found["result"] = {**(found.get("result") or {}), **result}
    found["updated_at"] = _now_iso()
    _append_event(campaign, "candidate_updated", note or cid, candidate_id=cid)
    saved = _save(str(campaign["project"]), campaign)
    return {"status": "ok", "candidate": found, "summary": _campaign_summary(saved["campaign"]), "path": saved["path"]}


def next_batch(
    *,
    project: str,
    campaign_id: str | None = None,
    max_candidates: int = 4,
    min_quality_score: float | None = None,
) -> dict[str, Any]:
    payload = load_campaign(project=project, campaign_id=campaign_id)
    if payload.get("status") != "ok":
        return payload
    campaign = payload["campaign"]
    try:
        limit = max(1, min(int(max_candidates or 4), 50))
    except (TypeError, ValueError):
        return {"status": "error", "message": "max_candidates 必须是整数。"}
    threshold = _maybe_float(min_quality_score)
    ready = []
    for item in campaign.get("candidates", []):
        if not isinstance(item, dict):
            continue
        state = str(item.get("status") or "").lower()
        if state in TERMINAL_CANDIDATE_STATES or state in RUNNING_CANDIDATE_STATES:
            continue
        if item.get("run_id") or item.get("job_id"):
            continue
        score = _maybe_float(item.get("quality_score"))
        if threshold is not None and score is not None and score < threshold:
            continue
        ready.append(item)
    ready.sort(key=lambda item: (_score_for_rank(item), str(item.get("candidate_id") or "")), reverse=True)
    batch = ready[:limit]
    return {
        "status": "ok",
        "campaign_id": campaign.get("campaign_id"),
        "count": len(batch),
        "candidates": batch,
        "recommended_action": "For each candidate, build/preflight a run with dft_run_task, submit if allowed, then update_candidate with run_id/job_id.",
    }


def prune_plan(
    *,
    project: str,
    campaign_id: str | None = None,
    keep_top: int = 4,
    min_quality_score: float | None = None,
    max_energy_ev: float | None = None,
    apply: bool = False,
    rationale: str | None = None,
) -> dict[str, Any]:
    payload = load_campaign(project=project, campaign_id=campaign_id)
    if payload.get("status") != "ok":
        return payload
    campaign = payload["campaign"]
    try:
        keep = max(1, min(int(keep_top or 4), 100))
    except (TypeError, ValueError):
        return {"status": "error", "message": "keep_top 必须是整数。"}
    qmin = _maybe_float(min_quality_score)
    emax = _maybe_float(max_energy_ev)
    ranked = sorted(
        [item for item in campaign.get("candidates", []) if isinstance(item, dict)],
        key=lambda item: (_result_rank(item), _score_for_rank(item), str(item.get("candidate_id") or "")),
        reverse=True,
    )
    keepers: list[dict[str, Any]] = []
    discards: list[dict[str, Any]] = []
    for item in ranked:
        score = _maybe_float(item.get("quality_score"))
        energy = _candidate_energy(item)
        reason = []
        if qmin is not None and score is not None and score < qmin:
            reason.append(f"quality_score<{qmin}")
        if emax is not None and energy is not None and energy > emax:
            reason.append(f"energy>{emax} eV")
        if len(keepers) >= keep:
            reason.append(f"outside_top_{keep}")
        if reason:
            marked = {**item, "prune_reason": "; ".join(reason)}
            discards.append(marked)
        else:
            keepers.append(item)
    if apply:
        discard_ids = {str(item.get("candidate_id") or "") for item in discards}
        keep_ids = {str(item.get("candidate_id") or "") for item in keepers}
        for item in campaign.get("candidates", []):
            cid = str(item.get("candidate_id") or "")
            if cid in discard_ids and str(item.get("status") or "").lower() not in RUNNING_CANDIDATE_STATES:
                item["status"] = "discarded"
                item["prune_reason"] = next((d.get("prune_reason") for d in discards if d.get("candidate_id") == cid), "")
                item["updated_at"] = _now_iso()
            elif cid in keep_ids and str(item.get("status") or "").lower() in {"candidate", "quality_pass", "ready"}:
                item["status"] = "promising"
                item["updated_at"] = _now_iso()
        _append_event(campaign, "prune_applied", rationale or f"keep={len(keepers)} discard={len(discards)}")
        saved = _save(str(campaign["project"]), campaign)
        summary = _campaign_summary(saved["campaign"])
    else:
        summary = _campaign_summary(campaign)
    return {
        "status": "ok",
        "campaign_id": campaign.get("campaign_id"),
        "apply": bool(apply),
        "keepers": keepers,
        "discards": discards,
        "summary": summary,
        "guidance": "apply=false 只给剪枝建议；apply=true 才更新 candidate 状态。",
    }


def _append_event(campaign: dict[str, Any], event: str, note: str, **extra: Any) -> None:
    rows = campaign.setdefault("events", [])
    rows.append({"at": _now_iso(), "event": event, "note": str(note or ""), **extra})
    campaign["events"] = rows[-200:]


def _load_manifest_candidates(source_manifest_path: str | None) -> list[dict[str, Any]]:
    text = str(source_manifest_path or "").strip()
    if not text:
        return []
    path = Path(text)
    if not path.exists() or not path.is_file():
        return []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(manifest, dict):
        return []
    material = str(manifest.get("material_name") or "").strip()
    adsorbate = str(manifest.get("adsorbate_source") or "").strip()
    rows = manifest.get("candidates") or []
    if not isinstance(rows, list):
        return []
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        exported = row.get("exported_files") if isinstance(row.get("exported_files"), dict) else {}
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        score = row.get("score") if isinstance(row.get("score"), dict) else {}
        score_total = _maybe_float(score.get("total")) if isinstance(score, dict) else None
        reason = ""
        if isinstance(score, dict):
            reason = str(score.get("reason") or "").strip()
        reason = reason or str(metadata.get("model_reason") or "").strip()
        parsed.append(
            {
                "candidate_id": candidate_id,
                "status": "candidate",
                "structure_path": str(exported.get("poscar_path") or metadata.get("source_poscar_path") or "").strip(),
                "material": material,
                "adsorbate": adsorbate,
                "motif": str(row.get("site_label") or row.get("site_family") or "").strip(),
                "orientation": str(row.get("orientation_label") or "").strip(),
                "reason": reason,
                "quality_score": score_total,
                "priority": score_total,
                "metadata": {
                    "manifest_task_id": manifest.get("task_id"),
                    "manifest_path": str(path),
                    "site_family": row.get("site_family"),
                    "site_label": row.get("site_label"),
                    "anchor_symbol": row.get("anchor_symbol"),
                    "height": row.get("height"),
                    "rank": metadata.get("rank"),
                },
            }
        )
    return parsed


def _maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_energy(item: dict[str, Any]) -> float | None:
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    for key in ("adsorption_energy_ev", "energy_ev", "final_energy_ev", "delta_e_ev"):
        val = _maybe_float(result.get(key) if key in result else item.get(key))
        if val is not None:
            return val
    return None


def _score_for_rank(item: dict[str, Any]) -> float:
    for key in ("priority", "quality_score"):
        val = _maybe_float(item.get(key))
        if val is not None:
            return val
    return 0.0


def _result_rank(item: dict[str, Any]) -> float:
    energy = _candidate_energy(item)
    if energy is None:
        return 0.0
    # Lower adsorption energies are often preferred; invert for descending sort.
    return -energy


def _campaign_summary(campaign: dict[str, Any]) -> dict[str, Any]:
    candidates = [item for item in campaign.get("candidates", []) if isinstance(item, dict)]
    counts: dict[str, int] = {}
    for item in candidates:
        key = str(item.get("status") or "unknown").lower()
        counts[key] = counts.get(key, 0) + 1
    missing_quality = [item.get("candidate_id") for item in candidates if item.get("quality_score") in (None, "")]
    ready = [item.get("candidate_id") for item in candidates if str(item.get("status") or "").lower() in READY_CANDIDATE_STATES and not item.get("run_id") and not item.get("job_id")]
    running = [item.get("candidate_id") for item in candidates if str(item.get("status") or "").lower() in RUNNING_CANDIDATE_STATES or item.get("job_id")]
    completed = [item.get("candidate_id") for item in candidates if str(item.get("status") or "").lower() == "completed"]
    if not candidates:
        next_focus = "enumerate_or_register_candidates"
    elif missing_quality:
        next_focus = "cheap_quality_filter"
    elif ready:
        next_focus = "build_preflight_submit_next_batch"
    elif running:
        next_focus = "monitor_fetch_parse_running_jobs"
    elif completed:
        next_focus = "prune_refine_or_close_campaign"
    else:
        next_focus = "reopen_search_space_or_ask_human_if_goal_met"
    return {
        "campaign_id": campaign.get("campaign_id"),
        "project": campaign.get("project"),
        "goal": campaign.get("goal"),
        "status": campaign.get("status"),
        "candidate_count": len(candidates),
        "counts_by_status": counts,
        "missing_quality_count": len(missing_quality),
        "ready_count": len(ready),
        "running_count": len(running),
        "completed_count": len(completed),
        "next_focus": next_focus,
        "updated_at": campaign.get("updated_at"),
    }
