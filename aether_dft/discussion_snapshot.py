"""讨论态快照：把当前对话推进到哪一步主动整理成 markdown，可选写回项目状态。

设计：模型在长对话中产生了共识、待澄清点、下一步计划时，调用本工具沉淀；
工具本身不抽取意义（那是模型的事），只负责"持久化 + 命名 + 索引"。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .paths import ensure_runtime_dir
from .project_state import project_paths


_SLUG_RE = re.compile(r"[^A-Za-z0-9一-鿿\-_]+")


@dataclass(frozen=True)
class DiscussionSnapshot:
    snapshot_id: str
    project: str | None
    title: str
    summary: str
    consensus: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    snapshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug(value: str) -> str:
    cleaned = _SLUG_RE.sub("_", str(value).strip()).strip("_")
    return cleaned[:60] or "snapshot"


def _snapshots_dir(project: str | None) -> Path:
    if project:
        directory = project_paths(project).root / "discussion_snapshots"
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    return ensure_runtime_dir("discussion_snapshots")


def _render_markdown(snapshot: DiscussionSnapshot) -> str:
    lines = [f"# {snapshot.title or '讨论快照'}", ""]
    lines.append(f"- snapshot_id: `{snapshot.snapshot_id}`")
    lines.append(f"- project: `{snapshot.project or '(no-project)'}`")
    lines.append(f"- captured_at: {snapshot.created_at}")
    if snapshot.tags:
        lines.append("- tags: " + ", ".join(f"`{t}`" for t in snapshot.tags))
    lines.append("")
    if snapshot.summary.strip():
        lines.extend(["## Summary", "", snapshot.summary.strip(), ""])
    if snapshot.consensus:
        lines.extend(["## Consensus", ""])
        lines.extend(f"- {item}" for item in snapshot.consensus)
        lines.append("")
    if snapshot.open_questions:
        lines.extend(["## Open questions", ""])
        lines.extend(f"- ❓ {item}" for item in snapshot.open_questions)
        lines.append("")
    if snapshot.next_steps:
        lines.extend(["## Next steps", ""])
        lines.extend(f"- ⬜ {item}" for item in snapshot.next_steps)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def capture_discussion_snapshot(
    *,
    title: str,
    summary: str,
    consensus: list[str] | None = None,
    open_questions: list[str] | None = None,
    next_steps: list[str] | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    write_to_project_state: bool = False,
) -> dict[str, Any]:
    """落一份讨论快照到 markdown + JSON。

    ``write_to_project_state=True`` 时还会 append 到项目 ``research_progress.md``，
    让"讨论 → 进展"形成自然衔接。
    """
    title_clean = str(title or "").strip()
    summary_clean = str(summary or "").strip()
    if not title_clean and not summary_clean:
        return {"status": "error", "message": "title 和 summary 至少要给一个非空字段。"}

    consensus_list = [str(item).strip() for item in (consensus or []) if str(item).strip()]
    open_q_list = [str(item).strip() for item in (open_questions or []) if str(item).strip()]
    next_list = [str(item).strip() for item in (next_steps or []) if str(item).strip()]
    tags_list = [str(item).strip() for item in (tags or []) if str(item).strip()]

    snapshot_id = f"snap_{uuid4().hex[:8]}"
    project_clean = str(project or "").strip() or None
    snapshot = DiscussionSnapshot(
        snapshot_id=snapshot_id,
        project=project_clean,
        title=title_clean or "讨论快照",
        summary=summary_clean,
        consensus=consensus_list,
        open_questions=open_q_list,
        next_steps=next_list,
        tags=tags_list,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )

    directory = _snapshots_dir(project_clean)
    base_name = f"{snapshot_id}-{_slug(title_clean or 'snapshot')}"
    md_path = directory / f"{base_name}.md"
    json_path = directory / f"{base_name}.json"
    md_path.write_text(_render_markdown(snapshot), encoding="utf-8")
    payload = snapshot.to_dict()
    payload["snapshot_path"] = str(md_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    progress_path: str | None = None
    if write_to_project_state and project_clean:
        try:
            from .project_state import append_progress

            entry = []
            if title_clean:
                entry.append(f"💬 讨论快照 `{snapshot_id}`：{title_clean}")
            for item in next_list:
                entry.append(item)
            written = append_progress(project_clean, completed=entry[:1], next_steps=next_list)
            progress_path = str(written)
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "status": "warning",
                "snapshot_id": snapshot_id,
                "snapshot_path": str(md_path),
                "snapshot_json": str(json_path),
                "message": f"写回项目状态失败但快照已保存: {exc}",
            }

    return {
        "status": "ok",
        "snapshot_id": snapshot_id,
        "snapshot_path": str(md_path),
        "snapshot_json": str(json_path),
        "project_progress_path": progress_path,
        "snapshot": payload,
        "guidance": (
            "快照适合作为长对话的 anchor 点。如果共识或下一步需要让团队/后续会话看到，"
            "可以再用 research_progress_append 或 knowledge_note_add 把要点带出去。"
        ),
    }


def list_discussion_snapshots(project: str | None = None) -> list[dict[str, Any]]:
    directory = _snapshots_dir(project)
    snapshots: list[dict[str, Any]] = []
    for path in sorted(directory.glob("snap_*.json"), reverse=True):
        try:
            snapshots.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return snapshots
