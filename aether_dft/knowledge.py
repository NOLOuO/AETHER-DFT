from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .paths import KNOWLEDGE_BASE_DIR, PROJECT_ROOT
from .project_state import project_paths


@dataclass(frozen=True)
class KnowledgeNote:
    note_id: str
    project: str
    title: str
    content: str
    tags: list[str]
    created_at: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value.strip()).strip("-")
    return text[:60] or "note"


def _notes_dir(project: str) -> Path:
    path = project_paths(project).knowledge / "notes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_note(project: str, title: str, content: str, *, tags: list[str] | None = None) -> KnowledgeNote:
    if not project.strip():
        raise ValueError("project 不能为空")
    if not title.strip():
        raise ValueError("title 不能为空")
    if not content.strip():
        raise ValueError("content 不能为空")
    note_id = f"note_{uuid4().hex[:8]}"
    note = KnowledgeNote(
        note_id=note_id,
        project=project,
        title=title.strip(),
        content=content.strip(),
        tags=[tag.strip() for tag in (tags or []) if tag.strip()],
        created_at=_now(),
    )
    path = _notes_dir(project) / f"{note_id}-{_slug(title)}.md"
    path.write_text(render_note_markdown(note), encoding="utf-8")
    return KnowledgeNote(**{**note.to_dict(), "path": str(path)})


def render_note_markdown(note: KnowledgeNote) -> str:
    return (
        f"# {note.title}\n\n"
        f"- Note ID: `{note.note_id}`\n"
        f"- Project: `{note.project}`\n"
        f"- Created: {note.created_at}\n"
        f"- Tags: {', '.join(note.tags) if note.tags else 'none'}\n\n"
        "## Content\n\n"
        f"{note.content}\n"
    )


def _parse_note(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = path.stem
    if text.startswith("# "):
        title = text.splitlines()[0][2:].strip()
    return {"path": str(path), "title": title, "content": text}


def list_notes(project: str) -> list[dict[str, Any]]:
    return [_parse_note(path) for path in sorted(_notes_dir(project).glob("*.md"), reverse=True)]


def search_notes(project: str, query: str) -> list[dict[str, Any]]:
    terms = [term.lower() for term in re.split(r"\s+", query.strip()) if term.strip()]
    if not terms:
        return list_notes(project)
    matches: list[dict[str, Any]] = []
    for note in list_notes(project):
        haystack = (note["title"] + "\n" + note["content"]).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            note = dict(note)
            note["score"] = score
            matches.append(note)
    return sorted(matches, key=lambda item: item["score"], reverse=True)


_DEFAULT_RESEARCH_GLOBS = (
    "research/Common/*.md",
    "research/*/研究进展.md",
    "research/*/Learning/*.md",
    "research/*/common/*.md",
    "research/*/文献参考/*.md",
)


def _expand_tokens(*values: str | None) -> list[str]:
    tokens: list[str] = []
    for value in values:
        if not value:
            continue
        cleaned = re.split(r"[\s/，,；;()（）]+", str(value).strip().lower())
        tokens.extend(token for token in cleaned if token)
    return tokens


def _score_text_against_tokens(text: str, tokens: list[str]) -> int:
    lowered = text.lower()
    score = 0
    for token in tokens:
        if token and token in lowered:
            score += 1
    return score


def _iter_cross_project_notes() -> list[Path]:
    if not KNOWLEDGE_BASE_DIR.exists():
        return []
    return sorted(KNOWLEDGE_BASE_DIR.glob("*/notes/*.md"))


def _iter_research_workspace_files() -> list[Path]:
    files: list[Path] = []
    for pattern in _DEFAULT_RESEARCH_GLOBS:
        files.extend(PROJECT_ROOT.glob(pattern))
    return sorted(files)


def search_for_system(
    *,
    material: str | None = None,
    adsorbate: str | None = None,
    extra_terms: list[str] | None = None,
    project_priority: str | None = None,
    max_results: int = 12,
    semantic: bool = True,
    selector: Callable[[str, list[dict[str, Any]], int], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """跨项目 / research workspace 搜索与给定体系相关的先验笔记。

    匹配粒度：material + adsorbate + 额外关键词分词后做大小写无关的 token 命中计数。
    返回按 score 倒序的若干条候选，每条带 path / score / preview。
    """
    tokens = _expand_tokens(material, adsorbate, *(extra_terms or []))
    if not tokens:
        raise ValueError("search_for_system 需要至少一个 material / adsorbate / extra_terms。")

    matches: list[dict[str, Any]] = []
    semantic_pool: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    def _push(path: Path, *, source: str, project_slug: str | None) -> None:
        if not path.exists() or path.is_dir():
            return
        key = str(path.resolve())
        if key in seen_paths:
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        lexical_score = _score_text_against_tokens(text, tokens)
        score = lexical_score
        if lexical_score > 0 and project_priority and project_slug == project_priority:
            score += 1
        title = path.stem
        first_line = next((line for line in text.splitlines() if line.strip()), "").strip()
        if first_line.startswith("# "):
            title = first_line[2:].strip() or title
        preview_lines: list[str] = []
        lowered = text.lower()
        for token in tokens:
            position = lowered.find(token)
            if position >= 0:
                start = max(0, position - 60)
                end = min(len(text), position + 120)
                preview_lines.append(text[start:end].replace("\n", " "))
            if sum(len(line) for line in preview_lines) > 600:
                break
        seen_paths.add(key)
        item = {
            "path": str(path),
            "title": title,
            "score": score,
            "source": source,
            "project": project_slug,
            "preview": " ... ".join(preview_lines) or text[:240].replace("\n", " "),
        }
        if lexical_score > 0:
            matches.append(item)
        elif semantic:
            semantic_pool.append(item)

    for note_path in _iter_cross_project_notes():
        project_slug = note_path.parent.parent.name
        _push(note_path, source="knowledge_base", project_slug=project_slug)

    for workspace_path in _iter_research_workspace_files():
        try:
            relative = workspace_path.relative_to(PROJECT_ROOT)
        except ValueError:
            relative = workspace_path
        project_slug = None
        parts = relative.parts
        if len(parts) >= 2 and parts[0] == "research" and parts[1] != "Common":
            project_slug = parts[1]
        _push(workspace_path, source="research_workspace", project_slug=project_slug)

    matches.sort(key=lambda item: item["score"], reverse=True)
    lexical_limited = matches[:max_results]
    selection_method = "lexical"
    selection_error = ""
    limited = lexical_limited
    semantic_candidates = matches + semantic_pool[: max(0, 40 - len(matches))]
    if semantic and semantic_candidates:
        query_text = _semantic_query_text(material=material, adsorbate=adsorbate, extra_terms=extra_terms or [])
        try:
            selected = semantic_select_memories(
                query_text,
                semantic_candidates,
                max_results=min(max_results, 5),
                selector=selector,
            )
            if selected:
                limited = selected
                selection_method = "semantic"
        except Exception as exc:
            selection_error = str(exc)
            limited = lexical_limited
            selection_method = "lexical_fallback"
    return {
        "status": "ok",
        "query": {
            "material": material,
            "adsorbate": adsorbate,
            "extra_terms": list(extra_terms or []),
            "tokens": tokens,
        },
        "selection_method": selection_method,
        "selection_error": selection_error,
        "total_matches": len(matches),
        "semantic_candidates_considered": len(semantic_candidates) if semantic else 0,
        "returned": len(limited),
        "matches": limited,
        "guidance": (
            "把命中条目里有领域结论 / 参数经验 / 避坑的内容当作 prior，写进 adsorption_candidate_plan.rationale；"
            "优先使用 warnings/gotchas/known issues 这类避坑信息；不要因为正在用某个工具就选择它的 API 文档，"
            "但要选择关于该工具/方法的失败模式和注意事项。"
            "如果没有命中，再考虑用 adsorbate_chemistry_hint 的通用先验，并在 plan 里标注 'no project prior found'。"
        ),
    }



def _memory_description(text: str, *, limit: int = 420) -> str:
    """Extract a cheap, header-first description for semantic memory selection."""

    lines = text.splitlines()
    description_lines: list[str] = []
    in_description = False
    for raw in lines[:80]:
        line = raw.strip()
        if not line:
            if in_description and description_lines:
                break
            continue
        lower = line.lower().lstrip("# ")
        if lower.startswith(("description", "summary", "摘要", "说明", "结论", "gotcha", "warning", "避坑")):
            in_description = True
            description_lines.append(line.lstrip("#-:： "))
            continue
        if in_description:
            if line.startswith("#") and description_lines:
                break
            description_lines.append(line.lstrip("- "))
    if not description_lines:
        for raw in lines[:40]:
            line = raw.strip()
            if line and not line.startswith("- Note ID:") and not line.startswith("- Project:"):
                description_lines.append(line.lstrip("# "))
            if sum(len(item) for item in description_lines) >= limit:
                break
    text = " ".join(description_lines).strip()
    return text[:limit].rstrip()


def _semantic_query_text(*, material: str | None, adsorbate: str | None, extra_terms: list[str]) -> str:
    parts = []
    if material:
        parts.append(f"material/system={material}")
    if adsorbate:
        parts.append(f"adsorbate={adsorbate}")
    if extra_terms:
        parts.append("extra=" + ", ".join(extra_terms))
    return "; ".join(parts)


def _memory_catalog(candidates: list[dict[str, Any]], *, max_candidates: int = 40) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for idx, item in enumerate(candidates[:max_candidates], start=1):
        text = str(item.get("content") or item.get("preview") or "")
        if not text:
            path = Path(str(item.get("path") or ""))
            if path.exists() and path.is_file():
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    text = ""
        catalog.append(
            {
                "rank": idx,
                "path": str(item.get("path") or ""),
                "title": str(item.get("title") or ""),
                "source": str(item.get("source") or ""),
                "project": item.get("project"),
                "lexical_score": item.get("score"),
                "description": _memory_description(text or str(item.get("preview") or "")),
            }
        )
    return catalog


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


def _default_memory_selector(query: str, catalog: list[dict[str, Any]], max_results: int) -> list[dict[str, Any]]:
    from aether_dft.model_catalog import resolve_effective_model_id, split_model_id
    from dft_app.llm import DomesticCopilotLLM

    model_id = resolve_effective_model_id()
    provider, model = split_model_id(model_id)
    messages = [
        {
            "role": "system",
            "content": (
                "你是计算化学科研记忆选择器。只根据 memory metadata 选择最相关的条目。"
                "优先 warnings/gotchas/known issues/避坑/失败模式/参数适用边界；"
                "不要选择正在使用工具的普通 API 文档，除非它记录了坑或已知问题。"
                "不要解释，不要输出推理过程。第一行且唯一输出必须是 JSON: "
                "{\"selected_ranks\":[整数...], \"reason\":\"简短中文理由\"}。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"query": query, "max_results": max_results, "memories": catalog},
                ensure_ascii=False,
            ),
        },
    ]
    result = DomesticCopilotLLM(PROJECT_ROOT).call_messages_inline(
        messages,
        provider_id=provider,
        model_id=model,
        max_tokens=4096,
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
            selected.append({"rank": rank, "semantic_reason": str(data.get("reason") or "")})
        if len(selected) >= max_results:
            break
    return selected


def semantic_select_memories(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    max_results: int = 5,
    selector: Callable[[str, list[dict[str, Any]], int], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Select relevant memories by showing only path/title/description metadata to a model.

    The full note bodies are not sent to the selector.  If no selector is
    provided, AETHER's configured model is used; callers can inject a selector
    in tests.  Returned entries keep the original note payload so downstream
    tools can still show previews/paths without another lookup.
    """

    if not candidates:
        return []
    max_results = max(1, min(int(max_results or 5), 8))
    catalog = _memory_catalog(candidates)
    select = selector or _default_memory_selector
    selected_refs = select(query, catalog, max_results)
    by_rank = {idx: item for idx, item in enumerate(candidates[: len(catalog)], start=1)}
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for ref in selected_refs:
        try:
            rank = int(ref.get("rank") if isinstance(ref, dict) else ref)
        except Exception:
            continue
        if rank in seen or rank not in by_rank:
            continue
        item = dict(by_rank[rank])
        item["semantic_rank"] = len(selected) + 1
        if isinstance(ref, dict) and ref.get("semantic_reason"):
            item["semantic_reason"] = str(ref.get("semantic_reason"))
        selected.append(item)
        seen.add(rank)
        if len(selected) >= max_results:
            break
    return selected


def show_note(path_or_id: str, *, project: str | None = None) -> dict[str, Any]:
    raw = Path(path_or_id)
    candidates: list[Path] = []
    if raw.exists():
        candidates.append(raw)
    if project:
        candidates.extend(_notes_dir(project).glob(f"{path_or_id}*.md"))
        candidates.extend(_notes_dir(project).glob(f"*{path_or_id}*.md"))
    if not candidates:
        root = KNOWLEDGE_BASE_DIR
        candidates.extend(root.glob(f"*/notes/{path_or_id}*.md"))
        candidates.extend(root.glob(f"*/notes/*{path_or_id}*.md"))
    if not candidates:
        raise FileNotFoundError(f"未找到知识条目: {path_or_id}")
    return _parse_note(candidates[0])
