from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adsorption_models import AdsorptionCandidate, CandidateManifest, CandidateScore


class CandidateManifestWriter:
    def write(
        self,
        manifest: CandidateManifest,
        output_dir: Path,
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates_dir = output_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

        for index, candidate in enumerate(manifest.candidates, start=1):
            candidate_dir = candidates_dir / candidate.candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            if candidate.structure is not None:
                poscar_path = candidate_dir / "POSCAR"
                cif_path = candidate_dir / "structure.cif"
                poscar_path.write_text(candidate.structure.to(fmt="poscar"), encoding="utf-8")
                cif_path.write_text(candidate.structure.to(fmt="cif"), encoding="utf-8")
                candidate.exported_files.update(
                    {
                        "poscar_path": str(poscar_path),
                        "cif_path": str(cif_path),
                    }
                )
            summary_path = candidate_dir / "candidate_summary.json"
            summary_path.write_text(
                json.dumps(candidate.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            candidate.exported_files["summary_path"] = str(summary_path)
            candidate.metadata.setdefault("rank", index)

        manifest_json = output_dir / "candidate_manifest.json"
        manifest_md = output_dir / "candidate_manifest.md"
        manifest_json.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest_md.write_text(self._render_markdown(manifest), encoding="utf-8")
        return {
            "manifest_json": str(manifest_json),
            "manifest_md": str(manifest_md),
            "candidates_dir": str(candidates_dir),
        }

    @staticmethod
    def _render_markdown(manifest: CandidateManifest) -> str:
        lines = [
            f"# 吸附候选清单：{manifest.material_name}",
            "",
            f"- task_id: `{manifest.task_id}`",
            f"- slab_source: `{manifest.slab_source}`",
            f"- adsorbate_source: `{manifest.adsorbate_source}`",
            f"- candidate_count: `{len(manifest.candidates)}`",
            "",
            "## 候选排序",
            "",
            "| Rank | Candidate ID | Site | Orientation | Score | Reason |",
            "| ---: | --- | --- | --- | ---: | --- |",
        ]
        for rank, candidate in enumerate(manifest.candidates, start=1):
            score = candidate.score.total if candidate.score is not None else 0.0
            reason = candidate.score.reason if candidate.score is not None else ""
            lines.append(
                f"| {rank} | `{candidate.candidate_id}` | `{candidate.site_label}` | `{candidate.orientation_label}` | {score:.2f} | {reason} |"
            )

        lines.extend(["", "## 人工确认项"])
        lines.extend(
            [
                "- 确认最终采用哪个 candidate_id",
                "- 确认 slab 参数、缺陷构型与吸附物初始取向是否符合科研意图",
                "- 确认是否直接进入现有 builder / submit 主线",
            ]
        )
        return "\n".join(lines) + "\n"


CANDIDATE_REASON_MIN_CHARS = 20
PRUNE_RATIONALE_THRESHOLD = 6


def _warn(warnings: list[dict[str, Any]], code: str, message: str, **context: Any) -> None:
    warnings.append({"code": code, "message": message, "context": dict(context)})


def compose_manifest_from_authored_candidates(
    *,
    task_id: str,
    material_name: str,
    source_prompt: str,
    slab_source: str,
    adsorbate_source: str,
    output_dir: str | Path,
    candidates: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    plan_payload: dict[str, Any] | None = None,
    prune_rationale: str | None = None,
) -> dict[str, Any]:
    """把模型一个个生成的 POSCAR 收编成与黑盒生成器同 schema 的 manifest。

    设计原则（guided-not-enforced）：本函数**不拦截模型**——除真正结构性错误（无候选、
    candidate_id 缺失/重复、POSCAR 文件不存在）外，质量类问题（reason 过短、site_label
    与 plan 不对齐、候选数超阈值无 prune_rationale、plan_payload.target_sites 为空、
    缺 plan_payload）都通过返回值的 ``quality_warnings: [...]`` 报回去；模型自己根据
    warnings 决定是否回炉重做或调 ``manifest_audit`` 反思。

    返回字段除原来的 status/task_id/candidate_count/manifest_json/manifest_md/
    candidates_dir/plan_id 外，新增：
      - ``quality_warnings``: ``[{code, message, context}, ...]``，可能为空数组
      - ``has_warnings``: bool
    """
    from pymatgen.core import Structure

    if not candidates:
        raise ValueError("candidates 不能为空；模型至少需要先生成一个 POSCAR 才能整合 manifest。")

    quality_warnings: list[dict[str, Any]] = []

    if plan_payload is None:
        _warn(
            quality_warnings,
            "plan_missing",
            "本次 compose 没有提供 plan_payload；建议先调 adsorption_candidate_plan 写明推理再回来。",
        )
    if len(candidates) > PRUNE_RATIONALE_THRESHOLD and not (prune_rationale or "").strip():
        _warn(
            quality_warnings,
            "prune_rationale_missing",
            f"候选数 {len(candidates)} 超过阈值 {PRUNE_RATIONALE_THRESHOLD} 但未提供 prune_rationale；"
            "可考虑收敛对称等价位点或显式说明为什么这么多。",
            candidate_count=len(candidates),
            threshold=PRUNE_RATIONALE_THRESHOLD,
        )

    plan_site_ids: set[str] | None = None
    if plan_payload is not None:
        plan_targets = plan_payload.get("target_sites") or []
        plan_site_ids = {
            str(item.get("site_id", "")).strip()
            for item in plan_targets
            if isinstance(item, dict) and str(item.get("site_id", "")).strip()
        }
        if not plan_site_ids:
            _warn(
                quality_warnings,
                "plan_target_sites_empty",
                "plan_payload 已提供但 target_sites 为空；候选与 plan 之间的可追溯链已断。",
            )
            plan_site_ids = None  # 后续不再做 site_label 对齐校验

    seen_ids: set[str] = set()
    items: list[AdsorptionCandidate] = []
    for entry in candidates:
        candidate_id = str(entry.get("candidate_id") or "").strip()
        poscar_path = entry.get("poscar_path")
        if not candidate_id:
            raise ValueError("每个 candidate 必须有非空 candidate_id。")
        if candidate_id in seen_ids:
            raise ValueError(f"candidate_id 重复: {candidate_id}")
        if not poscar_path:
            raise ValueError(f"candidate {candidate_id} 缺少 poscar_path。")
        reason = str(entry.get("reason") or "").strip()
        if len(reason) < CANDIDATE_REASON_MIN_CHARS:
            _warn(
                quality_warnings,
                "reason_too_short",
                f"candidate {candidate_id} 的 reason 只有 {len(reason)} 字（建议 ≥ {CANDIDATE_REASON_MIN_CHARS}）；"
                "写明科学依据（来自 chemistry_hint / knowledge prior / 对称判断）会让审计更有信心。",
                candidate_id=candidate_id,
                reason_length=len(reason),
                threshold=CANDIDATE_REASON_MIN_CHARS,
            )
        if plan_site_ids is not None:
            site_ref = str(entry.get("site_label") or entry.get("site_family") or "").strip()
            if site_ref and site_ref not in plan_site_ids:
                _warn(
                    quality_warnings,
                    "site_label_not_in_plan",
                    f"candidate {candidate_id} 的 site_label='{site_ref}' 不在 plan.target_sites 列表里："
                    f"{sorted(plan_site_ids)}；可能 plan 漏写了这个位点，或 candidate 用了别的命名。",
                    candidate_id=candidate_id,
                    site_label=site_ref,
                    plan_target_site_ids=sorted(plan_site_ids),
                )
        source_poscar = Path(poscar_path)
        if not source_poscar.exists():
            raise FileNotFoundError(f"candidate {candidate_id} 的 POSCAR 不存在: {source_poscar}")
        seen_ids.add(candidate_id)

        structure = Structure.from_file(source_poscar)
        score_payload = entry.get("score")
        score = None
        if isinstance(score_payload, dict):
            score = CandidateScore(
                total=float(score_payload.get("total") or 0.0),
                breakdown={str(k): float(v) for k, v in (score_payload.get("breakdown") or {}).items()},
                reason=str(score_payload.get("reason") or entry.get("reason") or ""),
            )
        elif entry.get("reason"):
            score = CandidateScore(total=0.0, breakdown={}, reason=str(entry["reason"]))

        merged_metadata: dict[str, Any] = {
            "authored_by": "model",
            "source_poscar_path": str(source_poscar),
        }
        merged_metadata.update(entry.get("metadata") or {})
        if entry.get("reason"):
            merged_metadata.setdefault("model_reason", str(entry["reason"]))

        items.append(
            AdsorptionCandidate(
                candidate_id=candidate_id,
                site_family=str(entry.get("site_family") or "model_authored"),
                site_label=str(entry.get("site_label") or candidate_id),
                orientation_label=str(entry.get("orientation_label") or "model_chosen"),
                anchor_symbol=str(entry.get("anchor_symbol") or ""),
                height=float(entry.get("height") or 0.0),
                defect_label=entry.get("defect_label"),
                structure=structure,
                metadata=merged_metadata,
                score=score,
            )
        )

    final_metadata: dict[str, Any] = {"authored_by": "model"}
    if plan_payload is not None:
        final_metadata["plan"] = plan_payload
        final_metadata["plan_id"] = plan_payload.get("plan_id")
    if prune_rationale:
        final_metadata["prune_rationale"] = prune_rationale.strip()
    if quality_warnings:
        final_metadata["quality_warnings"] = quality_warnings
    final_metadata.update(metadata or {})

    manifest = CandidateManifest(
        task_id=task_id,
        material_name=material_name,
        source_prompt=source_prompt,
        slab_source=str(slab_source),
        adsorbate_source=str(adsorbate_source),
        candidates=items,
        metadata=final_metadata,
    )
    artifacts = CandidateManifestWriter().write(manifest, Path(output_dir))
    return {
        "status": "composed",
        "task_id": task_id,
        "candidate_count": len(items),
        "manifest_json": artifacts["manifest_json"],
        "manifest_md": artifacts["manifest_md"],
        "candidates_dir": artifacts["candidates_dir"],
        "plan_id": (plan_payload or {}).get("plan_id"),
        "quality_warnings": quality_warnings,
        "has_warnings": bool(quality_warnings),
        "guidance": (
            "本工具不会因为质量问题挡你。warnings 是给模型的反馈："
            "如果你认可这些警告，可以回去调 adsorption_candidate_plan / 调整 reason / 合并对称等价位点再 compose；"
            "也可以调 manifest_audit 让 harness 进一步反思。"
        ) if quality_warnings else "本次 compose 未触发质量警告。",
    }


def audit_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """读已存盘 manifest 给"行为画像"。

    不挡路、不报错；返回评分 + 建议，由模型自己决定要不要回炉。
    """
    path = Path(manifest_path)
    if not path.exists():
        return {
            "status": "missing",
            "manifest_path": str(path),
            "message": "manifest 文件不存在；先调 adsorption_candidate_manifest_compose 生成。",
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates") or []
    meta = manifest.get("metadata") or {}
    findings: list[dict[str, Any]] = []

    plan = meta.get("plan") or {}
    plan_id = meta.get("plan_id")
    plan_target_ids = {
        str(item.get("site_id", "")).strip()
        for item in (plan.get("target_sites") or [])
        if isinstance(item, dict) and str(item.get("site_id", "")).strip()
    }

    def _add(level: str, code: str, message: str, **ctx: Any) -> None:
        findings.append({"level": level, "code": code, "message": message, "context": dict(ctx)})

    # 维度 1：plan 关联
    plan_score = 1.0
    if not plan_id:
        plan_score = 0.0
        _add("warn", "no_plan_link", "manifest 没有 plan_id；本次候选生成无法追溯到结构化推理。")
    elif not plan_target_ids:
        plan_score = 0.3
        _add("warn", "plan_targets_empty", "plan_id 存在但 target_sites 为空。", plan_id=plan_id)

    # 维度 2：reason 质量
    reason_lengths = []
    for cand in candidates:
        reason = ""
        score = cand.get("score") or {}
        if isinstance(score, dict):
            reason = str(score.get("reason") or "").strip()
        if not reason:
            reason = str((cand.get("metadata") or {}).get("model_reason") or "").strip()
        reason_lengths.append(len(reason))
    if reason_lengths:
        avg_len = sum(reason_lengths) / len(reason_lengths)
        short_count = sum(1 for length in reason_lengths if length < CANDIDATE_REASON_MIN_CHARS)
        reason_score = max(0.0, min(1.0, avg_len / float(CANDIDATE_REASON_MIN_CHARS * 2)))
        if short_count:
            _add(
                "warn",
                "reasons_too_short",
                f"{short_count}/{len(reason_lengths)} 个候选的 reason 短于建议长度（≥ {CANDIDATE_REASON_MIN_CHARS}）；"
                "审计建议把化学/对称依据写进 reason 而不是工具调用堆里。",
                short_count=short_count,
                avg_length=round(avg_len, 1),
            )
    else:
        reason_score = 0.0
        _add("warn", "no_reasons", "manifest 中没有可读 reason；下游 review 缺少推理痕迹。")

    # 维度 3：site_label 与 plan 对齐
    if plan_target_ids:
        unaligned = []
        for cand in candidates:
            label = str(cand.get("site_label") or "").strip()
            if label and label not in plan_target_ids:
                unaligned.append({"candidate_id": cand.get("candidate_id"), "site_label": label})
        if unaligned:
            align_score = max(0.0, 1.0 - len(unaligned) / max(1, len(candidates)))
            _add(
                "warn",
                "site_label_misaligned",
                f"{len(unaligned)}/{len(candidates)} 个候选的 site_label 不在 plan.target_sites；"
                "审计建议在 plan 里补齐这些位点的科学理由。",
                unaligned=unaligned,
            )
        else:
            align_score = 1.0
    else:
        align_score = 0.5  # 没有 plan 就不评分对齐

    # 维度 4：prior 引用
    priors_consulted = (plan.get("priors_consulted") or {}) if plan else {}
    if priors_consulted:
        prior_score = 1.0 if any(priors_consulted.values()) else 0.4
        if prior_score < 1.0:
            _add(
                "warn",
                "priors_marked_but_empty",
                "plan.priors_consulted 存在但所有值都为空；过去经验可能没真查过。",
                priors=priors_consulted,
            )
    else:
        prior_score = 0.0
        _add(
            "info",
            "no_priors_consulted",
            "plan 未记录 priors_consulted；建议在生成候选前调 knowledge_search_for_system 与 adsorbate_chemistry_hint 后回填。",
        )

    # 维度 5：候选数量是否合理
    n_cand = len(candidates)
    prune_rationale = meta.get("prune_rationale")
    if n_cand > PRUNE_RATIONALE_THRESHOLD and not prune_rationale:
        size_score = 0.5
        _add(
            "warn",
            "too_many_candidates_no_prune_rationale",
            f"候选数 {n_cand} 超过阈值 {PRUNE_RATIONALE_THRESHOLD} 且无 prune_rationale；"
            "可能没做对称等价合并。",
            n_candidates=n_cand,
            threshold=PRUNE_RATIONALE_THRESHOLD,
        )
    elif n_cand == 0:
        size_score = 0.0
        _add("warn", "no_candidates", "manifest 没有候选。")
    elif n_cand >= 2 and n_cand <= PRUNE_RATIONALE_THRESHOLD:
        size_score = 1.0
    else:
        size_score = 0.8

    weights = {"plan": 0.25, "reason": 0.2, "align": 0.2, "prior": 0.15, "size": 0.2}
    total = (
        weights["plan"] * plan_score
        + weights["reason"] * reason_score
        + weights["align"] * align_score
        + weights["prior"] * prior_score
        + weights["size"] * size_score
    )

    suggestions: list[str] = []
    if plan_score < 0.5:
        suggestions.append("调 adsorption_candidate_plan 写明 rationale / motif / anchor / target_sites 再 compose。")
    if reason_score < 0.6:
        suggestions.append("为每个候选写一句带科学依据的 reason（chemistry_hint 引用 / 对称结论 / prior 引用）。")
    if align_score < 0.8:
        suggestions.append("把 candidate.site_label 与 plan.target_sites[*].site_id 对齐，或在 plan 里加这些位点。")
    if prior_score < 0.5:
        suggestions.append("生成候选前 knowledge_search_for_system + adsorbate_chemistry_hint，把结果写进 plan.priors_consulted。")
    if size_score < 1.0 and n_cand > PRUNE_RATIONALE_THRESHOLD:
        suggestions.append("用 slab_surface_inspect 的 symmetry_groups 合并对称等价位点；保留的位点用 prune_rationale 说明。")
    if not suggestions:
        suggestions.append("当前 manifest 质量画像良好；下一步可走 adsorption-select 选 top 候选进入 VASP 工作区。")

    return {
        "status": "ok",
        "manifest_path": str(path),
        "candidate_count": n_cand,
        "total_score": round(total, 3),
        "score_breakdown": {
            "plan_link": round(plan_score, 2),
            "reason_quality": round(reason_score, 2),
            "site_alignment": round(align_score, 2),
            "prior_consultation": round(prior_score, 2),
            "candidate_size": round(size_score, 2),
        },
        "weights": weights,
        "findings": findings,
        "suggestions": suggestions,
        "guidance": (
            "本评分是 harness 行为画像，不是科学正确性判定。"
            "总分 ≥ 0.75 通常意味着可以放心 review；< 0.5 建议先按 suggestions 回炉再 compose。"
        ),
    }
