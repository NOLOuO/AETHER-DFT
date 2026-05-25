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

    ``candidates`` 中每一项至少包含：
      - ``candidate_id`` (str)：必须唯一
      - ``poscar_path`` (str)：模型先前用 ``structure_add_adsorbate`` 生成的 POSCAR
      - ``reason`` (str)：模型解释为何选这个候选；至少 20 字
      - ``site_label`` (str)：必须与 plan_payload.target_sites[*].site_id 对齐（当传 plan_payload 时）
    可选：
      - ``site_family`` / ``orientation_label`` / ``anchor_symbol``
      - ``height`` (float)
      - ``defect_label`` (str | None)
      - ``score`` (dict[str, Any])
      - ``metadata`` (dict)

    ``plan_payload``：可选，由 aether_dft.adsorption_authoring.create_candidate_plan 产生的
    plan dict，会被嵌入 manifest.metadata，并且会被用来校验候选 site_id 对齐。

    ``prune_rationale``：当候选数 > 6 时必填，解释为什么裁掉了别的对称等价位点。
    """
    from pymatgen.core import Structure

    if not candidates:
        raise ValueError("candidates 不能为空；模型至少需要先生成一个 POSCAR 才能整合 manifest。")
    if len(candidates) > PRUNE_RATIONALE_THRESHOLD and not (prune_rationale or "").strip():
        raise ValueError(
            f"候选数 {len(candidates)} 超过阈值 {PRUNE_RATIONALE_THRESHOLD}，"
            "必须提供 prune_rationale 说明为什么没有进一步收敛。"
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
            raise ValueError("plan_payload.target_sites 为空，无法校验候选。")

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
            raise ValueError(
                f"candidate {candidate_id} 的 reason 太短（{len(reason)} < {CANDIDATE_REASON_MIN_CHARS}）；"
                "写明科学依据（来自 chemistry_hint / knowledge prior / 对称判断），不能用 'selected by model'。"
            )
        if plan_site_ids is not None:
            site_ref = str(entry.get("site_label") or entry.get("site_family") or "").strip()
            if site_ref and site_ref not in plan_site_ids:
                raise ValueError(
                    f"candidate {candidate_id} 的 site_label='{site_ref}' 不在 plan.target_sites 列表里："
                    f"{sorted(plan_site_ids)}；要么改 plan，要么改 candidate。"
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
    }
