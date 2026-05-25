"""知识库相似度匹配引擎。

任何工具都可以用这个模块查找相似的历史案例。
"""

from __future__ import annotations

from .models import MatchResult, TaskRecord
from .store import KnowledgeStore, get_default_store


def find_similar(
    target: TaskRecord | dict,
    *,
    store: KnowledgeStore | None = None,
    pool_status: str | None = None,
    limit: int = 5,
    min_score: float = 1.0,
) -> list[MatchResult]:
    """在知识库中查找与 target 最相似的记录。

    Args:
        target: 当前任务（TaskRecord 或 dict）
        store: 知识库实例，默认用全局单例
        pool_status: 只在特定状态的记录中搜索（如 "success"）
        limit: 返回最多几条
        min_score: 最低分数阈值
    """
    if store is None:
        store = get_default_store()

    if isinstance(target, dict):
        target = TaskRecord.from_dict(target)

    candidates = store.load_all_snapshots(status=pool_status) if pool_status else store.load_all_snapshots()

    results: list[MatchResult] = []
    for cand_dict in candidates:
        if cand_dict.get("task_name") == target.task_name:
            continue
        score, reasons = _compute_similarity(target, cand_dict)
        if score >= min_score:
            results.append(MatchResult(
                record=TaskRecord.from_dict(cand_dict),
                score=score,
                reasons=reasons,
            ))

    results.sort(key=lambda x: (-x.score, x.record.task_name))
    return results[:limit]


def _compute_similarity(target: TaskRecord, candidate: dict) -> tuple[float, list[str]]:
    """计算两个任务之间的相似度分数。"""
    score = 0.0
    reasons: list[str] = []

    t_ctx = target.structure_context
    c_ctx = candidate.get("structure_context") or {}

    # 1. 元素重叠
    t_elements = set((t_ctx.get("element_counts") or {}).keys())
    c_elements = set((c_ctx.get("element_counts") or {}).keys())
    overlap = t_elements & c_elements
    if overlap:
        score += min(len(overlap), 3)
        reasons.append(f"元素交集: {', '.join(sorted(overlap))}")

    # 2. 约化式一致
    if t_ctx.get("reduced_formula") and t_ctx["reduced_formula"] == c_ctx.get("reduced_formula"):
        score += 3
        reasons.append(f"约化式一致: {t_ctx['reduced_formula']}")

    # 3. 结构角色
    t_role = t_ctx.get("structure_role")
    c_role = c_ctx.get("structure_role")
    if t_role and t_role == c_role:
        score += 5
        reasons.append(f"结构角色一致: {t_role}")
    elif t_role and c_role:
        if "molecule_in_box" in {t_role, c_role}:
            score -= 4
        else:
            score -= 2

    # 4. 表面/非表面
    t_surface = t_ctx.get("likely_surface")
    c_surface = c_ctx.get("likely_surface")
    if t_surface is not None and c_surface is not None:
        if t_surface == c_surface:
            score += 2
            reasons.append("表面判定一致")
        else:
            score -= 3

    # 5. 基底元素
    t_substrate = set(t_ctx.get("substrate_species") or [])
    c_substrate = set(c_ctx.get("substrate_species") or [])
    if t_substrate and c_substrate:
        if t_substrate == c_substrate:
            score += 5
            reasons.append(f"基底元素一致: {', '.join(sorted(t_substrate))}")
        elif t_substrate & c_substrate:
            score += 2
            reasons.append(f"基底元素部分重合")

    # 6. 基底家族
    t_sub_fam = t_ctx.get("substrate_family")
    c_sub_fam = c_ctx.get("substrate_family")
    if t_sub_fam and t_sub_fam == c_sub_fam:
        score += 4
        reasons.append(f"基底家族一致: {t_sub_fam}")
    elif t_sub_fam and c_sub_fam:
        score -= 2

    # 7. 吸附物元素
    t_ads = set(t_ctx.get("adsorbate_species") or [])
    c_ads = set(c_ctx.get("adsorbate_species") or [])
    if t_ads and c_ads:
        if t_ads == c_ads:
            score += 5
            reasons.append(f"吸附物元素一致: {', '.join(sorted(t_ads))}")
        elif t_ads & c_ads:
            score += 2

    # 8. 吸附物家族
    t_ads_fam = t_ctx.get("adsorbate_family")
    c_ads_fam = c_ctx.get("adsorbate_family")
    if t_ads_fam and t_ads_fam == c_ads_fam:
        score += 4
        reasons.append(f"吸附物家族一致: {t_ads_fam}")
    elif t_ads_fam and c_ads_fam:
        score -= 2

    # 9. 位点家族
    t_site = t_ctx.get("site_family")
    c_site = c_ctx.get("site_family")
    if t_site and t_site == c_site:
        score += 4
        reasons.append(f"位点家族一致: {t_site}")

    # 10. 任务类型
    if target.task_type and target.task_type == candidate.get("task_type"):
        score += 2
        reasons.append(f"任务类型一致: {target.task_type}")

    # 11. INCAR 参数
    t_incar = (target.input_context.get("incar") or {})
    c_incar = ((candidate.get("input_context") or {}).get("incar") or {})
    if t_incar.get("ISPIN") and t_incar["ISPIN"] == c_incar.get("ISPIN"):
        score += 1

    return score, reasons
