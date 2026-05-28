"""结构化吸附候选推理 plan：让模型在生成 POSCAR 之前把判断说清楚。

plan 是模型与下游 manifest 之间的可追溯草稿纸。质量问题不再阻断 plan
落盘，而是写入 quality_warnings；只有真正结构性输入错误才 raise。
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
    quality_warnings: list[dict[str, Any]] = field(default_factory=list)

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


def _warn(warnings: list[dict[str, Any]], code: str, message: str, **context: Any) -> None:
    warnings.append({"code": code, "message": message, "context": dict(context)})


def _validate_target_sites(value: Any, warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("target_sites 必须是数组；每项需至少含 site_id 和 reason。")
    if not value:
        _warn(
            warnings,
            "target_sites_empty",
            "target_sites 为空；plan 已落盘，但候选生成和 manifest 对齐会缺少可追溯位点。",
        )
        return []

    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"target_sites[{index}] 不是对象。")
        site_id = str(raw.get("site_id", "")).strip()
        if not site_id:
            raise ValueError(f"target_sites[{index}] 缺少非空 site_id。")
        if site_id in seen_ids:
            raise ValueError(f"target_sites[{index}] site_id 重复: {site_id}")
        seen_ids.add(site_id)

        reason = str(raw.get("reason", "")).strip()
        if len(reason) < SITE_REASON_MIN_CHARS:
            _warn(
                warnings,
                "target_site_reason_too_short",
                f"target_sites[{index}].reason 只有 {len(reason)} 字（建议 ≥ {SITE_REASON_MIN_CHARS}）。",
                index=index,
                site_id=site_id,
                reason_length=len(reason),
                threshold=SITE_REASON_MIN_CHARS,
            )
        entry = {"site_id": site_id, "reason": reason}
        for key, item in raw.items():
            if key not in {"site_id", "reason"}:
                entry[str(key)] = item
        normalized.append(entry)
    return normalized


def _validate_excluded(value: Any, warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            _warn(
                warnings,
                "excluded_site_reason_too_short",
                f"excluded_sites_with_reason[{index}].reason 只有 {len(reason)} 字（建议 ≥ {EXCLUSION_REASON_MIN_CHARS}）。",
                index=index,
                site_id=site_id,
                reason_length=len(reason),
                threshold=EXCLUSION_REASON_MIN_CHARS,
            )
        normalized.append({"site_id": site_id, "reason": reason})
    return normalized


def _validate_orientations(value: Any, warnings: list[dict[str, Any]]) -> list[str]:
    if value is None or value == []:
        _warn(
            warnings,
            "target_orientations_empty",
            "target_orientations 为空；plan 已落盘，但后续生成 POSCAR 时需要模型自行说明取向或补充 orientation。",
        )
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("target_orientations 必须是字符串数组。")
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    if not cleaned:
        _warn(
            warnings,
            "target_orientations_empty",
            "target_orientations 没有非空字符串；plan 已落盘，但后续生成 POSCAR 时需要补充 orientation。",
        )
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
    """创建并持久化结构化推理 plan。

    guided-not-enforced：质量问题写入 quality_warnings，不阻断 plan_id / plan_path
    生成；只有 schema 畸形、缺关键实体或重复 site_id 这类结构性错误才 raise。
    """

    quality_warnings: list[dict[str, Any]] = []

    material_clean = (material or "").strip()
    if not material_clean:
        raise ValueError("material 不能为空。")
    adsorbate_clean = (adsorbate or "").strip()
    if not adsorbate_clean:
        raise ValueError("adsorbate 不能为空。")

    rationale_clean = (rationale or "").strip()
    if len(rationale_clean) < RATIONALE_MIN_CHARS:
        _warn(
            quality_warnings,
            "rationale_too_short",
            f"rationale 只有 {len(rationale_clean)} 字（建议 ≥ {RATIONALE_MIN_CHARS}）。",
            length=len(rationale_clean),
            threshold=RATIONALE_MIN_CHARS,
        )

    motif_clean = (expected_binding_motif or "").strip()
    if not motif_clean:
        _warn(
            quality_warnings,
            "expected_binding_motif_missing",
            "expected_binding_motif 为空；建议写明例如 'atop O-down'。",
        )

    anchor_clean = (anchor_atom or "").strip()
    if not anchor_clean:
        _warn(
            quality_warnings,
            "anchor_atom_missing",
            "anchor_atom 为空；建议写明例如 'O' / 'C' / 'N'。",
        )

    sites = _validate_target_sites(target_sites, quality_warnings)
    orientations = _validate_orientations(target_orientations, quality_warnings)
    excluded = _validate_excluded(excluded_sites_with_reason, quality_warnings)

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
        quality_warnings=quality_warnings,
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
            data.setdefault("quality_warnings", [])
            return AdsorptionCandidatePlan(**data)
    raise FileNotFoundError(f"找不到 adsorption plan: {plan_id}")


def list_candidate_plans(project: str | None = None) -> list[dict[str, Any]]:
    directory = _plans_dir(project)
    plans: list[dict[str, Any]] = []
    for path in sorted(directory.glob("plan_*.json"), reverse=True):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            item.setdefault("quality_warnings", [])
            plans.append(item)
        except Exception:
            continue
    return plans
