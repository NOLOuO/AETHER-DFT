from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any

from .paths import KNOWLEDGE_BASE_DIR, PROJECTS_DIR, ensure_project_dirs
from .research_workspace import list_research_projects, read_research_onboarding_context, resolve_research_project


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text)
    text = text.strip("-")
    return text or "project"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class ProjectPaths:
    slug: str
    root: Path
    metadata: Path
    progress: Path
    state: Path
    state_md: Path
    runs: Path
    knowledge: Path


def project_paths(slug: str) -> ProjectPaths:
    clean = slugify(slug)
    root = PROJECTS_DIR / clean
    return ProjectPaths(
        slug=clean,
        root=root,
        metadata=root / "project.json",
        progress=root / "research_progress.md",
        state=root / "state" / "current_state.json",
        state_md=root / "_state.md",
        runs=root / "runs",
        knowledge=KNOWLEDGE_BASE_DIR / clean,
    )


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return '""'
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _render_state_markdown(metadata: dict[str, Any], state: dict[str, Any], progress_text: str) -> str:
    front_matter = {
        "project": metadata.get("slug") or metadata.get("name") or "",
        "name": metadata.get("name") or "",
        "description": metadata.get("description") or "",
        "status": metadata.get("status") or "",
        "current_focus": state.get("current_focus") or "",
        "blockers": state.get("blockers") or [],
        "next_steps": state.get("next_steps") or [],
        "updated_at": state.get("updated_at") or metadata.get("updated_at") or now_iso(),
    }

    lines = ["---"]
    for key, value in front_matter.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {_yaml_scalar(item)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.extend([
        "---",
        "",
        "# 项目状态",
        "",
        "## 当前聚焦",
        "",
        str(front_matter["current_focus"]).strip() or "（未设置）",
        "",
        "## 阻塞项",
        "",
    ])
    blockers = [str(item).strip() for item in (state.get("blockers") or []) if str(item).strip()]
    if blockers:
        lines.extend([f"- {item}" for item in blockers])
    else:
        lines.append("- （暂无）")
    lines.extend([
        "",
        "## 下一步",
        "",
    ])
    next_steps = [str(item).strip() for item in (state.get("next_steps") or []) if str(item).strip()]
    if next_steps:
        lines.extend([f"- {item}" for item in next_steps])
    else:
        lines.append("- （暂无）")
    if progress_text.strip():
        lines.extend([
            "",
            "## 研究进展",
            "",
            progress_text.strip(),
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _sync_state_markdown(paths: ProjectPaths) -> Path:
    metadata: dict[str, Any] = {}
    state: dict[str, Any] = {}
    if paths.metadata.exists():
        try:
            metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    if paths.state.exists():
        try:
            state = json.loads(paths.state.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    progress_text = paths.progress.read_text(encoding="utf-8") if paths.progress.exists() else ""
    paths.state_md.write_text(_render_state_markdown(metadata, state, progress_text), encoding="utf-8")
    return paths.state_md


def init_project(name: str, *, description: str = "", overwrite: bool = False) -> dict[str, Any]:
    ensure_project_dirs()
    paths = project_paths(name)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state").mkdir(parents=True, exist_ok=True)
    paths.runs.mkdir(parents=True, exist_ok=True)
    paths.knowledge.mkdir(parents=True, exist_ok=True)
    created_at = now_iso()
    metadata = {
        "slug": paths.slug,
        "name": name,
        "description": description,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "active",
        "privacy": "do-not-store-personal-identifying-information",
    }
    if overwrite or not paths.metadata.exists():
        paths.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    if not paths.progress.exists():
        paths.progress.write_text(
            "# 研究进展\n\n"
            "### " + datetime.now().strftime("%Y-%m-%d") + "\n\n"
            "- ✅ 项目容器已创建。\n"
            "- ⚠️ 尚未录入具体计算卡点。\n"
            "- ⬜ 下一步：通过 `aether-dft chat --project " + paths.slug + "` 持续推进。\n",
            encoding="utf-8",
        )
    if not paths.state.exists():
        write_project_state(paths.slug, {"current_focus": "", "blockers": [], "next_steps": []})
    else:
        _sync_state_markdown(paths)
    return {"project": metadata, "paths": {k: str(v) for k, v in paths.__dict__.items() if k != "slug"}}


def list_projects() -> list[dict[str, Any]]:
    ensure_project_dirs()
    items_by_key: dict[str, dict[str, Any]] = {}
    for slug in list_research_projects():
        research_paths = resolve_research_project(slug)
        title = slug
        updated_at = ""
        progress_path = str(research_paths.progress) if research_paths else ""
        if research_paths and research_paths.progress.exists():
            try:
                updated_at = datetime.fromtimestamp(research_paths.progress.stat().st_mtime).astimezone().isoformat(timespec="seconds")
            except Exception:
                updated_at = ""
        items_by_key[slugify(slug)] = {
            "slug": slug,
            "name": slug,
            "title": title,
            "description": f"research/{slug}",
            "status": "active",
            "source": "research",
            "research_project": True,
            "research_root": str(research_paths.root) if research_paths else "",
            "research_progress": progress_path,
            "updated_at": updated_at,
        }
    for metadata_path in sorted(PROJECTS_DIR.glob("*/project.json")):
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        slug = str(data.get("slug") or metadata_path.parent.name)
        key = slugify(slug)
        if key in items_by_key:
            merged = dict(data)
            merged.update({k: v for k, v in items_by_key[key].items() if v not in ("", None, [], {})})
            merged["source"] = "research+aether"
            items_by_key[key] = merged
        else:
            data.setdefault("slug", slug)
            data.setdefault("source", "aether")
            data.setdefault("research_project", False)
            items_by_key[key] = data
    return sorted(items_by_key.values(), key=lambda item: (0 if item.get("research_project") else 1, str(item.get("slug") or "").lower()))


def load_project(slug: str) -> dict[str, Any]:
    research_paths = resolve_research_project(slug)
    if research_paths is not None:
        metadata: dict[str, Any] = {
            "slug": research_paths.slug,
            "name": research_paths.slug,
            "title": research_paths.slug,
            "description": f"research/{research_paths.slug}",
            "status": "active",
            "source": "research",
            "research_project": True,
            "research": research_paths.to_dict(),
        }
        aether_paths = project_paths(research_paths.slug)
        if aether_paths.metadata.exists():
            try:
                saved = json.loads(aether_paths.metadata.read_text(encoding="utf-8"))
                metadata.update({k: v for k, v in saved.items() if v not in ("", None, [], {})})
                metadata["source"] = "research+aether"
                metadata["research_project"] = True
                metadata["research"] = research_paths.to_dict()
            except Exception:
                pass
        return metadata
    paths = project_paths(slug)
    if not paths.metadata.exists():
        raise FileNotFoundError(f"项目不存在: {slug}")
    return json.loads(paths.metadata.read_text(encoding="utf-8"))


def _truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n\n...[truncated to {max_chars} chars; read project files directly for full context]"


def read_project_context(slug: str, *, max_chars: int | None = None) -> str:
    research_paths = resolve_research_project(slug)
    paths = project_paths(slug)
    parts = []
    if research_paths is not None:
        onboarding = read_research_onboarding_context(research_paths.slug, max_chars=max_chars or 14000)
        if str(onboarding.get("context") or "").strip():
            parts.append("## Research workspace context\n" + str(onboarding.get("context") or ""))
    if paths.state_md.exists():
        parts.append("## Project state markdown\n" + paths.state_md.read_text(encoding="utf-8"))
    if paths.metadata.exists():
        parts.append("## Project metadata\n" + paths.metadata.read_text(encoding="utf-8"))
    if paths.state.exists():
        parts.append("## Current state\n" + paths.state.read_text(encoding="utf-8"))
    if paths.progress.exists():
        parts.append("## Research progress\n" + paths.progress.read_text(encoding="utf-8"))
    return _truncate_text("\n\n".join(parts), max_chars)


def read_project_context_digest(slug: str, *, max_chars: int = 7000) -> str:
    """Prompt-safe project context: current state first, newest progress next."""

    research_paths = resolve_research_project(slug)
    paths = project_paths(slug)
    parts = []
    if research_paths is not None:
        onboarding = read_research_onboarding_context(research_paths.slug, max_chars=max_chars)
        if str(onboarding.get("context") or "").strip():
            parts.append("## Research workspace digest\n" + str(onboarding.get("context") or ""))
    if paths.metadata.exists():
        parts.append("## Project metadata\n" + paths.metadata.read_text(encoding="utf-8"))
    if paths.state.exists():
        parts.append("## Current state\n" + paths.state.read_text(encoding="utf-8"))
    if paths.progress.exists():
        parts.append("## Latest research progress\n" + paths.progress.read_text(encoding="utf-8"))
    if paths.state_md.exists():
        parts.append("## State markdown path\n" + str(paths.state_md))
    return _truncate_text("\n\n".join(parts), max_chars)


def write_project_state(slug: str, state: dict[str, Any]) -> Path:
    paths = project_paths(slug)
    paths.state.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = now_iso()
    paths.state.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _sync_state_markdown(paths)
    return paths.state


def append_progress(slug: str, *, completed: list[str] | None = None, blockers: list[str] | None = None, next_steps: list[str] | None = None) -> Path:
    paths = project_paths(slug)
    paths.root.mkdir(parents=True, exist_ok=True)
    lines = [f"### {datetime.now().strftime('%Y-%m-%d')}", ""]
    for item in completed or []:
        lines.append(f"- ✅ {item}")
    for item in blockers or []:
        lines.append(f"- ⚠️ {item}")
    for item in next_steps or []:
        lines.append(f"- ⬜ {item}")
    if len(lines) == 2:
        lines.append("- ✅ 已记录一次项目交互。")
    lines.append("")
    old = paths.progress.read_text(encoding="utf-8") if paths.progress.exists() else "# 研究进展\n\n"
    header = "# 研究进展\n\n"
    body = old[len(header):] if old.startswith(header) else old
    paths.progress.write_text(header + "\n".join(lines) + "\n" + body, encoding="utf-8")
    _sync_state_markdown(paths)
    return paths.progress
