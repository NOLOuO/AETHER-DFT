from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any
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
) -> dict[str, Any]:
    """跨项目 / research workspace 搜索与给定体系相关的先验笔记。

    匹配粒度：material + adsorbate + 额外关键词分词后做大小写无关的 token 命中计数。
    返回按 score 倒序的若干条候选，每条带 path / score / preview。
    """
    tokens = _expand_tokens(material, adsorbate, *(extra_terms or []))
    if not tokens:
        raise ValueError("search_for_system 需要至少一个 material / adsorbate / extra_terms。")

    matches: list[dict[str, Any]] = []
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
        score = _score_text_against_tokens(text, tokens)
        if score <= 0:
            return
        if project_priority and project_slug == project_priority:
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
        matches.append(
            {
                "path": str(path),
                "title": title,
                "score": score,
                "source": source,
                "project": project_slug,
                "preview": " ... ".join(preview_lines) or text[:240].replace("\n", " "),
            }
        )

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
    limited = matches[:max_results]
    return {
        "status": "ok",
        "query": {
            "material": material,
            "adsorbate": adsorbate,
            "extra_terms": list(extra_terms or []),
            "tokens": tokens,
        },
        "total_matches": len(matches),
        "returned": len(limited),
        "matches": limited,
        "guidance": (
            "把命中条目里有领域结论 / 参数经验 / 避坑的内容当作 prior，写进 adsorption_candidate_plan.rationale；"
            "如果没有命中，再考虑用 adsorbate_chemistry_hint 的通用先验，并在 plan 里标注 'no project prior found'。"
        ),
    }


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
