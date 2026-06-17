from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Iterable

from .paths import PROJECT_ROOT


Selector = Callable[[str, list[dict[str, Any]], int], list[dict[str, Any]]]


@dataclass(frozen=True)
class SessionSearchHit:
    summary: Any
    score: int = 0
    reason: str = ""
    selection_method: str = "lexical"


def _collapse(value: Any, *, limit: int = 360) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _tokens(query: str) -> list[str]:
    return [token for token in re.split(r"[\s/，,；;()（）]+", str(query or "").lower()) if token]


def _transcript_excerpt(rows: Iterable[dict[str, Any]], *, max_chars: int = 900) -> str:
    chunks: list[str] = []
    materialized = list(rows)
    if len(materialized) > 6:
        materialized = materialized[:3] + materialized[-3:]
    for row in materialized:
        record = row.get("record") if isinstance(row, dict) else {}
        if not isinstance(record, dict):
            continue
        prompt = _collapse(record.get("prompt"), limit=160)
        response = _collapse(record.get("response"), limit=160)
        if prompt:
            chunks.append(f"user: {prompt}")
        if response:
            chunks.append(f"assistant: {response}")
        if sum(len(item) for item in chunks) >= max_chars:
            break
    text = " | ".join(chunks)
    return text[:max_chars].rstrip()


def _session_text(item: Any, transcript: str = "") -> str:
    return "\n".join(
        [
            str(getattr(item, "session_id", "") or ""),
            str(getattr(item, "project", "") or ""),
            str(getattr(item, "title", "") or ""),
            str(getattr(item, "first_prompt", "") or ""),
            str(getattr(item, "last_response", "") or ""),
            str(getattr(item, "pending_prompt", "") or ""),
            transcript,
        ]
    ).lower()


def _lexical_score(item: Any, query: str, transcript: str = "") -> int:
    toks = _tokens(query)
    if not toks:
        return 0
    haystack = _session_text(item, transcript)
    score = 0
    query_lower = str(query or "").lower().strip()
    if query_lower and query_lower in haystack:
        score += max(3, len(toks))
    score += sum(1 for token in toks if token in haystack)
    return score


def _catalog_for_sessions(
    sessions: list[Any],
    *,
    transcript_loader: Callable[[str], list[dict[str, Any]]] | None = None,
    max_sessions: int = 80,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    catalog: list[dict[str, Any]] = []
    transcript_by_rank: dict[int, str] = {}
    for rank, item in enumerate(sessions[:max_sessions], start=1):
        session_id = str(getattr(item, "session_id", "") or "")
        transcript = ""
        if transcript_loader and session_id:
            try:
                transcript = _transcript_excerpt(transcript_loader(session_id))
            except Exception:
                transcript = ""
        transcript_by_rank[rank] = transcript
        catalog.append(
            {
                "rank": rank,
                "session_id": session_id,
                "project": getattr(item, "project", None),
                "title": _collapse(getattr(item, "title", ""), limit=120),
                "updated_at": str(getattr(item, "updated_at", "") or ""),
                "turn_count": int(getattr(item, "turn_count", 0) or 0),
                "first_prompt": _collapse(getattr(item, "first_prompt", ""), limit=260),
                "last_response": _collapse(getattr(item, "last_response", ""), limit=260),
                "pending": _collapse(getattr(item, "pending_prompt", ""), limit=220),
                "transcript_excerpt": transcript,
            }
        )
    return catalog, transcript_by_rank


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _default_session_selector(query: str, catalog: list[dict[str, Any]], max_results: int) -> list[dict[str, Any]]:
    from aether_dft.model_catalog import resolve_effective_model_id, split_model_id
    from dft_app.llm import DomesticCopilotLLM

    provider, model = split_model_id(resolve_effective_model_id())
    messages = [
        {
            "role": "system",
            "content": (
                "你是 AETHER-DFT 的会话恢复选择器。用户正在用自然语言寻找应该续接的历史科研对话。"
                "只根据给定 metadata / transcript excerpt 选择最相关的 session；不要发明不存在的 session。"
                "优先匹配科研对象、结构/反应主题、集群任务、未完成 pending、项目名和用户意图。"
                "要包容：语义相关但字面不同也可以选；不确定时返回少量候选而不是硬猜一个。"
                "唯一输出 JSON：{\"selected_ranks\":[整数...],\"reason\":\"简短中文理由\"}。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"query": query, "max_results": max_results, "sessions": catalog},
                ensure_ascii=False,
            ),
        },
    ]
    result = DomesticCopilotLLM(PROJECT_ROOT).call_messages_inline(
        messages,
        provider_id=provider,
        model_id=model,
        max_tokens=2048,
        tools=[],
        tool_choice="none",
    )
    data = _extract_json_object(str(result.get("content") or ""))
    if not data and result.get("reasoning_content"):
        data = _extract_json_object(str(result.get("reasoning_content") or ""))
    ranks = data.get("selected_ranks") if isinstance(data, dict) else []
    selected: list[dict[str, Any]] = []
    for value in ranks or []:
        try:
            rank = int(value)
        except Exception:
            continue
        if 1 <= rank <= len(catalog) and all(item.get("rank") != rank for item in selected):
            selected.append({"rank": rank, "reason": str(data.get("reason") or "")})
        if len(selected) >= max_results:
            break
    return selected


def rank_session_summaries(
    query: str,
    sessions: list[Any],
    *,
    transcript_loader: Callable[[str], list[dict[str, Any]]] | None = None,
    max_results: int = 8,
    selector: Selector | None = None,
    semantic: bool = True,
) -> dict[str, Any]:
    """Rank resumable sessions for a natural-language query.

    This is deliberately not a natural-language if/else router.  It first
    builds a compact metadata catalog, then lets a selector model choose
    relevant sessions.  If the selector is unavailable, it falls back to
    lexical scoring so `/resume` remains usable offline.
    """

    query = str(query or "").strip()
    max_results = max(1, min(int(max_results or 8), 20))
    if not query or not sessions:
        return {"status": "empty", "matches": [], "selection_method": "empty", "selection_error": ""}

    catalog, transcript_by_rank = _catalog_for_sessions(sessions, transcript_loader=transcript_loader)
    by_rank = {index: item for index, item in enumerate(sessions[: len(catalog)], start=1)}
    selection_error = ""
    if semantic and catalog:
        try:
            selected_refs = (selector or _default_session_selector)(query, catalog, max_results)
            hits: list[SessionSearchHit] = []
            seen: set[int] = set()
            for ref in selected_refs:
                try:
                    rank = int(ref.get("rank") if isinstance(ref, dict) else ref)
                except Exception:
                    continue
                if rank in seen or rank not in by_rank:
                    continue
                summary = by_rank[rank]
                hits.append(
                    SessionSearchHit(
                        summary=summary,
                        score=_lexical_score(summary, query, transcript_by_rank.get(rank, "")),
                        reason=str(ref.get("reason") or "") if isinstance(ref, dict) else "",
                        selection_method="semantic",
                    )
                )
                seen.add(rank)
                if len(hits) >= max_results:
                    break
            if hits:
                return {
                    "status": "ok",
                    "matches": hits,
                    "selection_method": "semantic",
                    "selection_error": "",
                    "catalog_size": len(catalog),
                }
        except Exception as exc:
            selection_error = str(exc)

    scored: list[SessionSearchHit] = []
    for rank, summary in by_rank.items():
        score = _lexical_score(summary, query, transcript_by_rank.get(rank, ""))
        if score > 0:
            scored.append(SessionSearchHit(summary=summary, score=score, selection_method="lexical"))
    scored.sort(key=lambda item: item.score, reverse=True)
    return {
        "status": "ok" if scored else "empty",
        "matches": scored[:max_results],
        "selection_method": "lexical_fallback" if selection_error else "lexical",
        "selection_error": selection_error,
        "catalog_size": len(catalog),
    }
