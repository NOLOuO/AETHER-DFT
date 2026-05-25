"""结构化吸附候选推理 plan：让模型在生成 POSCAR 之前必须先把判断说清楚。

plan 是模型与下游 manifest 之间的"思考门槛"——没有合规的 plan 就不允许 compose_manifest。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .paths import ensure_runtime_dir
from .project_state import project_paths


RATIONALE_MIN_CHARS = 30
SITE_REASON_MIN_CHARS = 10
EXCLUSION_REASON_MIN_CHARS = 8


@dataclass(frozen=True)
class AdsorptionCandidatePlan:
    plan_id: str
    project: str | None
    task_id: str | None
    material: str
    adsorbate: str
    rationale: str
    expected_binding_motif: str
    anchor_atom: str
    target_sites: list[dict[str, Any]]
    target_orientations: list[str]
    excluded_sites_with_reason: list[dict[str, Any]] = field(default_factory=list)
    symmetry_pruning_applied: bool = False
    priors_consulted: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    created_at: str = ""
    plan_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def target_site_ids(self) -> list[str]:
        return [str(item.get("site_id", "")).strip() for item in self.target_sites if item.get("site_id")]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _plans_dir(project: str | None) -> Path:
    if project:
        path = project_paths(project).root / "adsorption_plans"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return ensure_runtime_dir("adsorption_plans")


def _validate_target_sites(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("target_sites 必须是非空数组；每项需至少含 site_id 和 reason。")
    seen_ids: set[str] = set()
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"target_sites[{index}] 不是对象。")
        site_id = str(raw.get("site_id", "")).strip()
        if not site_id:
            raise ValueError(f"target_sites[{index}] 缺少非空 site_id。")
        if site_id in seen_ids:
            raise ValueError(f"target_sites[{index}] site_id 重复: {site_id}")
        seen_ids.add(site_id)

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value, start=1):
        site_id = str(raw.get("site_id", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        if len(reason) < SITE_REASON_MIN_CHARS:
            raise ValueError(
                f"target_sites[{index}].reason 太短（{len(reason)} < {SITE_REASON_MIN_CHARS}），"
                "写明为什么选这个位点（化学依据 / 对称依据）。"
            )
        entry = {"site_id": site_id, "reason": reason}
        for key, value in raw.items():
            if key not in {"site_id", "reason"}:
                entry[str(key)] = value
        normalized.append(entry)
    return normalized


def _validate_excluded(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("excluded_sites_with_reason 必须是数组或省略。")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"excluded_sites_with_reason[{index}] 不是对象。")
        site_id = str(raw.get("site_id", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        if not site_id:
            raise ValueError(f"excluded_sites_with_reason[{index}] 缺少 site_id。")
        if len(reason) < EXCLUSION_REASON_MIN_CHARS:
            raise ValueError(
                f"excluded_sites_with_reason[{index}].reason 太短（{len(reason)} < {EXCLUSION_REASON_MIN_CHARS}）。"
            )
        normalized.append({"site_id": site_id, "reason": reason})
    return normalized


def _validate_orientations(value: Any) -> list[str]:
    if value is None or value == []:
        raise ValueError("target_orientations 必须至少给一个 orientation（例如 upright / flat / tilted）。")
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("target_orientations 必须是字符串数组。")
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    if not cleaned:
        raise ValueError("target_orientations 至少需要一个非空字符串。")
    return cleaned


def create_candidate_plan(
    *,
    material: str,
    adsorbate: str,
    rationale: str,
    expected_binding_motif: str,
    anchor_atom: str,
    target_sites: list[dict[str, Any]],
    target_orientations: list[str] | str,
    excluded_sites_with_reason: list[dict[str, Any]] | None = None,
    symmetry_pruning_applied: bool = False,
    priors_consulted: dict[str, Any] | None = None,
    project: str | None = None,
    task_id: str | None = None,
    notes: str = "",
) -> AdsorptionCandidatePlan:
    """创建并持久化结构化推理 plan。"""

    material_clean = (material or "").strip()
    if not material_clean:
        raise ValueError("material 不能为空。")
    adsorbate_clean = (adsorbate or "").strip()
    if not adsorbate_clean:
        raise ValueError("adsorbate 不能为空。")
    rationale_clean = (rationale or "").strip()
    if len(rationale_clean) < RATIONALE_MIN_CHARS:
        raise ValueError(
            f"rationale 太短（{len(rationale_clean)} < {RATIONALE_MIN_CHARS}）；"
            "写清你为什么这样选位点 / 取向 / anchor，至少 30 字。"
        )
    motif_clean = (expected_binding_motif or "").strip()
    if not motif_clean:
        raise ValueError("expected_binding_motif 不能为空，例如 'atop O-down'。")
    anchor_clean = (anchor_atom or "").strip()
    if not anchor_clean:
        raise ValueError("anchor_atom 不能为空，例如 'O' / 'C' / 'N'。")

    sites = _validate_target_sites(target_sites)
    orientations = _validate_orientations(target_orientations)
    excluded = _validate_excluded(excluded_sites_with_reason)

    plan_id = f"plan_{uuid4().hex[:8]}"
    plan = AdsorptionCandidatePlan(
        plan_id=plan_id,
        project=project,
        task_id=task_id,
        material=material_clean,
        adsorbate=adsorbate_clean,
        rationale=rationale_clean,
        expected_binding_motif=motif_clean,
        anchor_atom=anchor_clean,
        target_sites=sites,
        target_orientations=orientations,
        excluded_sites_with_reason=excluded,
        symmetry_pruning_applied=bool(symmetry_pruning_applied),
        priors_consulted=dict(priors_consulted or {}),
        notes=(notes or "").strip(),
        created_at=_now(),
    )
    path = _plans_dir(project) / f"{plan_id}.json"
    payload = plan.to_dict()
    payload["plan_path"] = str(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return AdsorptionCandidatePlan(**payload)


def load_candidate_plan(plan_id: str, *, project: str | None = None) -> AdsorptionCandidatePlan:
    """按 plan_id 在项目目录与 runtime 目录中查找。"""
    candidates: list[Path] = []
    if project:
        candidates.append(_plans_dir(project) / f"{plan_id}.json")
    candidates.append(_plans_dir(None) / f"{plan_id}.json")
    # Fallback：跨所有项目模糊查找
    from .paths import PROJECTS_DIR
    if PROJECTS_DIR.exists():
        candidates.extend(PROJECTS_DIR.glob(f"*/adsorption_plans/{plan_id}.json"))

    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("plan_path", str(path))
            return AdsorptionCandidatePlan(**data)
    raise FileNotFoundError(f"找不到 adsorption plan: {plan_id}")


def list_candidate_plans(project: str | None = None) -> list[dict[str, Any]]:
    directory = _plans_dir(project)
    plans: list[dict[str, Any]] = []
    for path in sorted(directory.glob("plan_*.json"), reverse=True):
        try:
            plans.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return plans
