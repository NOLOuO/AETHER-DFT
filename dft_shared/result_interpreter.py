"""结果解释服务：把证据 bundle 转成结构化解释结果。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .analysis_dossier import (
    EvidenceBundle,
    EvidenceGap,
    InterpretationRequest,
    InterpretationResult,
    KnowledgeBackflowPayload,
)
from .llm_client import call_openai_compatible_result, maybe_strip_markdown_fence


def interpret_result(request: InterpretationRequest) -> InterpretationResult:
    """生成结构化解释结果。

    正常路径走 LLM；缺 key / 接口失败 / JSON 不合法时自动降级为启发式结果。
    """
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        messages = _build_messages(request.evidence)
        llm_result = call_openai_compatible_result(
            request.provider,
            request.model,
            "",
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            timeout=60,
        )
        parsed = _parse_llm_payload(llm_result["content"])
        result = InterpretationResult(
            task_name=request.task_name,
            status_judgement=_coerce_text(parsed.get("status_judgement")) or _fallback_status(request.evidence),
            likely_causes=_coerce_string_list(parsed.get("likely_causes")),
            next_actions=_coerce_string_list(parsed.get("next_actions")),
            evidence_used=_coerce_evidence_list(parsed.get("evidence_used")),
            missing_evidence=_merge_missing_evidence(
                request.evidence.missing_evidence,
                parsed.get("missing_evidence"),
            ),
            raw_llm_text=maybe_strip_markdown_fence(llm_result["content"]),
            provider=str(llm_result.get("provider") or "none"),
            model=str(llm_result.get("model") or "none"),
            generated_at=generated_at,
            confidence=_coerce_text(parsed.get("confidence")) or _infer_confidence(request.evidence),
            degraded=False,
            notes=[],
        )
        if not result.likely_causes:
            result.likely_causes = _fallback_likely_causes(request.evidence)
        if not result.next_actions:
            result.next_actions = _fallback_next_actions(request.evidence)
        if not result.evidence_used:
            result.evidence_used = _build_evidence_used(request.evidence)
        result.knowledge_backflow_payload = build_knowledge_backflow_payload(request.evidence, result)
        return result
    except Exception as exc:
        if not request.allow_degraded:
            raise
        result = _build_degraded_result(request.evidence, request.task_name, generated_at, str(exc))
        result.knowledge_backflow_payload = build_knowledge_backflow_payload(request.evidence, result)
        return result


def build_knowledge_backflow_payload(
    evidence: EvidenceBundle,
    result: InterpretationResult,
) -> KnowledgeBackflowPayload:
    return KnowledgeBackflowPayload(
        task_name=result.task_name,
        task_goal_text=evidence.task_goal_text,
        status_judgement=result.status_judgement,
        likely_causes=list(result.likely_causes),
        next_actions=list(result.next_actions),
        evidence_digest={
            "status_summary": evidence.status_summary,
            "result_summary": evidence.result_summary,
            "structure_context": evidence.structure_context,
            "input_context": evidence.input_context,
            "structure_analysis": evidence.structure_analysis,
            "similar_cases": evidence.similar_cases[:3],
            "evidence_used": result.evidence_used,
        },
        missing_evidence=[gap.to_dict() for gap in result.missing_evidence],
        provider=result.provider,
        model=result.model,
        generated_at=result.generated_at,
        inferred_fields={
            "confidence": result.confidence,
            "degraded": result.degraded,
            "notes": list(result.notes),
        },
    )


def render_interpretation_markdown(
    evidence: EvidenceBundle,
    result: InterpretationResult,
) -> str:
    lines = [
        f"# 结果解释: {result.task_name}",
        "",
        f"- 生成时间: `{result.generated_at or 'N/A'}`",
        f"- Provider / Model: `{result.provider}` / `{result.model}`",
        f"- 状态判断: `{result.status_judgement}`",
        f"- 置信度: `{result.confidence}`",
        f"- 是否降级: `{result.degraded}`",
        "",
        "## 任务目标",
        evidence.task_goal_text.strip() if (evidence.task_goal_text or "").strip() else "- 未提供任务自然语言目标",
        "",
        "## 最可能原因",
    ]
    if result.likely_causes:
        lines.extend(f"- {item}" for item in result.likely_causes)
    else:
        lines.append("- 暂无。")

    lines.extend(["", "## 下一步建议"])
    if result.next_actions:
        lines.extend(f"- {item}" for item in result.next_actions)
    else:
        lines.append("- 暂无。")

    lines.extend(["", "## 证据依据"])
    if result.evidence_used:
        for item in result.evidence_used:
            source = item.get("source") or item.get("kind") or "evidence"
            detail = item.get("detail") or item.get("value") or item
            lines.append(f"- `{source}`: {detail}")
    else:
        lines.append("- 暂无。")

    lines.extend(["", "## 缺失证据"])
    if result.missing_evidence:
        for gap in result.missing_evidence:
            extra = f"；建议：{gap.suggestion}" if gap.suggestion else ""
            lines.append(f"- `{gap.field}`: {gap.reason}{extra}")
    else:
        lines.append("- 无。")

    if result.notes:
        lines.extend(["", "## 备注"])
        lines.extend(f"- {item}" for item in result.notes)

    lines.extend(["", "## 原始模型输出", result.raw_llm_text.strip() or "- 无。", ""])
    return "\n".join(lines)


def _build_messages(evidence: EvidenceBundle) -> list[dict[str, str]]:
    system_prompt = (
        "你是 DFT 结果解释服务。必须严格基于用户给定证据输出 JSON，不要输出 Markdown。\n"
        "返回字段只能包含：status_judgement, likely_causes, next_actions, evidence_used, missing_evidence, confidence。\n"
        "其中 likely_causes / next_actions 为字符串数组；evidence_used 为 {source, detail} 数组；"
        "missing_evidence 为 {field, reason, suggestion} 数组。\n"
        "如果任务自然语言目标缺失，必须把 task_goal_text 写入 missing_evidence，"
        "并在结论中体现“解释证据不完整”。不要编造未给出的输入参数或实验目标。"
    )
    user_prompt = json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_llm_payload(text: str) -> dict[str, Any]:
    stripped = maybe_strip_markdown_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _build_degraded_result(
    evidence: EvidenceBundle,
    task_name: str,
    generated_at: str,
    error_message: str,
) -> InterpretationResult:
    result = InterpretationResult(
        task_name=task_name,
        status_judgement=_fallback_status(evidence),
        likely_causes=_fallback_likely_causes(evidence),
        next_actions=_fallback_next_actions(evidence),
        evidence_used=_build_evidence_used(evidence),
        missing_evidence=list(evidence.missing_evidence),
        raw_llm_text=f"结果解释降级：{error_message}",
        provider="none",
        model="none",
        generated_at=generated_at,
        confidence=_infer_confidence(evidence),
        degraded=True,
        notes=["当前结果由本地启发式规则生成，未获得结构化 LLM 响应。"],
    )
    return result


def _fallback_status(evidence: EvidenceBundle) -> str:
    status = _coerce_text(evidence.status_summary.get("status_assessment"))
    status_map = {
        "success": "状态判断：本次计算已完成并收敛，可进入结果复核或后续分析阶段。",
        "completed": "状态判断：本次计算已完成，但仍需结合任务目标进一步复核。",
        "completed_but_not_converged": "状态判断：本次计算已跑完，但尚未稳定收敛，需要继续优化或复核参数。",
        "not_converged": "状态判断：本次计算尚未稳定收敛，需要继续优化并排查原因。",
        "needs_review": "状态判断：当前结果可读，但仍需人工复核关键结构和参数后再决定下一步。",
        "likely_incomplete_or_failed": "状态判断：当前结果更像未完成或失败，需要优先排查异常并决定是否重跑。",
        "failed": "状态判断：本次计算未成功完成，需要先定位失败原因。",
        "unknown": "状态判断：当前证据不足，暂时无法明确判定结果状态。",
    }
    if status:
        mapped = status_map.get(status, f"状态判断：当前状态为 `{status}`，请结合证据进一步复核。")
        if evidence.task_goal_text:
            return mapped
        return f"{mapped}（注意：缺少任务自然语言目标，当前解释证据不完整。）"
    return "状态判断：当前证据不足，暂时无法明确判定结果状态。"


def _fallback_likely_causes(evidence: EvidenceBundle) -> list[str]:
    causes: list[str] = []
    if not (evidence.task_goal_text or "").strip():
        causes.append("缺少任务自然语言目标，当前只能基于结构、输入参数和解析摘要做不完整解释。")
    failure_reason = _coerce_text(evidence.status_summary.get("failure_reason"))
    if failure_reason:
        causes.append(f"解析结果标记的主要失败类型为 `{failure_reason}`。")
    warnings = _coerce_string_list(evidence.status_summary.get("warnings"))
    causes.extend(warnings[:3])
    anomalies = _coerce_string_list((evidence.structure_analysis.get("displacement_report") or {}).get("anomalies"))
    causes.extend(anomalies[:2])
    if not causes:
        causes.append("现有证据不足以稳定定位原因，建议补充任务目标与更多输入/输出上下文。")
    return _dedupe_keep_order(causes)


def _fallback_next_actions(evidence: EvidenceBundle) -> list[str]:
    actions = _coerce_string_list(evidence.status_summary.get("recommended_actions"))
    if not (evidence.task_goal_text or "").strip():
        actions.insert(0, "补充本次计算的自然语言任务目标，避免解释器误把通用失败模式当成真实原因。")
    if not evidence.input_context:
        actions.append("补充 INCAR/KPOINTS 等输入参数摘要，便于判断参数是否与任务目标匹配。")
    if not evidence.structure_analysis:
        actions.append("补充 POSCAR/CONTCAR 结构对比结果，确认是否存在异常位移、断键或吸附质漂移。")
    if not actions:
        actions.append("当前无自动建议，请结合 result_summary 和 task_overview 人工复核。")
    return _dedupe_keep_order(actions)[:5]


def _build_evidence_used(evidence: EvidenceBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status = evidence.status_summary
    if status.get("status_assessment"):
        rows.append({"source": "status_summary", "detail": f"status_assessment={status['status_assessment']}"})
    if status.get("failure_reason"):
        rows.append({"source": "status_summary", "detail": f"failure_reason={status['failure_reason']}"})
    if status.get("max_force") is not None:
        rows.append({"source": "result_summary", "detail": f"max_force={status['max_force']}"})
    if evidence.structure_context.get("reduced_formula"):
        rows.append({"source": "structure_context", "detail": f"reduced_formula={evidence.structure_context['reduced_formula']}"})
    incar = evidence.input_context.get("incar") or {}
    if incar:
        rows.append({"source": "input_context", "detail": f"INCAR keys={sorted(incar.keys())[:8]}"})
    displacement_report = evidence.structure_analysis.get("displacement_report") or {}
    if displacement_report.get("max_displacement") is not None:
        rows.append(
            {
                "source": "structure_analysis",
                "detail": f"max_displacement={displacement_report['max_displacement']}",
            }
        )
    if evidence.similar_cases:
        rows.append(
            {
                "source": "similar_cases",
                "detail": f"matched_cases={len(evidence.similar_cases[:3])}",
            }
        )
    return rows


def _merge_missing_evidence(
    base_gaps: list[EvidenceGap],
    llm_gaps: Any,
) -> list[EvidenceGap]:
    merged: list[EvidenceGap] = list(base_gaps)
    seen = {(gap.field, gap.reason) for gap in merged}
    for raw in llm_gaps or []:
        if isinstance(raw, str):
            gap = EvidenceGap(field=raw, reason="模型认为该字段缺失")
        elif isinstance(raw, dict):
            gap = EvidenceGap(
                field=_coerce_text(raw.get("field")) or "unknown",
                reason=_coerce_text(raw.get("reason")) or "模型认为该字段缺失",
                severity=_coerce_text(raw.get("severity")) or "warning",
                suggestion=_coerce_text(raw.get("suggestion")),
            )
        else:
            continue
        key = (gap.field, gap.reason)
        if key in seen:
            continue
        merged.append(gap)
        seen.add(key)
    return merged


def _infer_confidence(evidence: EvidenceBundle) -> str:
    if evidence.missing_evidence:
        return "low"
    if evidence.similar_cases or evidence.structure_analysis:
        return "medium"
    return "low"


def _coerce_text(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return text


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _coerce_evidence_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            detail = _coerce_text(item.get("detail")) or _coerce_text(item.get("value"))
            source = _coerce_text(item.get("source")) or _coerce_text(item.get("kind")) or "evidence"
            if detail:
                rows.append({"source": source, "detail": detail})
            continue
        text = _coerce_text(item)
        if text:
            rows.append({"source": "evidence", "detail": text})
    return rows


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows
