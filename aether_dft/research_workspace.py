from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT

RESEARCH_ROOT = PROJECT_ROOT / "research"
COMMON_DIR = RESEARCH_ROOT / "Common"


def _truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n\n...[truncated to {max_chars} chars]"


def _read_text(path: Path, *, max_chars: int | None = None, redact_personal: bool = True) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if redact_personal:
        redacted_lines: list[str] = []
        for line in text.splitlines():
            if re.search(r"\*\*身份\*\*|维护者|NOL|张松|Zhang Song|厦门大学|课题组|硕士研究生", line):
                redacted_lines.append("[redacted personal/workspace identity]")
            else:
                redacted_lines.append(line)
        text = "\n".join(redacted_lines)
    return _truncate(text, max_chars)


@dataclass(frozen=True)
class ResearchProjectPaths:
    slug: str
    root: Path
    progress: Path
    common_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "slug": self.slug,
            "root": str(self.root),
            "progress": str(self.progress),
            "common_dir": str(self.common_dir),
        }


def list_research_projects() -> list[str]:
    if not RESEARCH_ROOT.exists():
        return []
    ignored = {"common", ".omx", ".omc", "__pycache__"}
    return [
        item.name
        for item in sorted(RESEARCH_ROOT.iterdir())
        if item.is_dir() and item.name.lower() not in ignored and not item.name.startswith(".")
    ]


def resolve_research_project(project: str | None) -> ResearchProjectPaths | None:
    if not project or not RESEARCH_ROOT.exists():
        return None
    raw = str(project).strip()
    candidates = list_research_projects()
    by_lower = {item.lower(): item for item in candidates}
    resolved = by_lower.get(raw.lower())
    if resolved is None:
        normalized = re.sub(r"[^a-z0-9]+", "", raw.lower())
        for item in candidates:
            if re.sub(r"[^a-z0-9]+", "", item.lower()) == normalized:
                resolved = item
                break
    if resolved is None:
        return None
    root = RESEARCH_ROOT / resolved
    return ResearchProjectPaths(
        slug=resolved,
        root=root,
        progress=root / "研究进展.md",
        common_dir=root / "common",
    )


def read_research_onboarding_context(
    project: str | None = None,
    *,
    max_chars: int = 14000,
    redact_personal: bool = True,
) -> dict[str, Any]:
    """Read the human-maintained research workspace handoff context.

    This is the AETHER equivalent of the user's original AGENTS.md workflow:
    map -> pitfalls -> project progress.  It is intentionally read-only and
    redacts personal identity lines by default because the agent needs the
    research process, not the user's private identity.
    """

    parts: list[str] = []
    files: list[str] = []
    root_agents = RESEARCH_ROOT / "AGENTS.md"
    if root_agents.exists():
        parts.append("## research/AGENTS.md\n" + _read_text(root_agents, redact_personal=redact_personal))
        files.append(str(root_agents))
    pitfalls = COMMON_DIR / "避坑清单.md"
    if pitfalls.exists():
        parts.append("## research/Common/避坑清单.md\n" + _read_text(pitfalls, redact_personal=redact_personal))
        files.append(str(pitfalls))
    project_paths = resolve_research_project(project)
    if project_paths and project_paths.progress.exists():
        parts.append(
            f"## research/{project_paths.slug}/研究进展.md\n"
            + _read_text(project_paths.progress, redact_personal=redact_personal)
        )
        files.append(str(project_paths.progress))
    if project_paths and project_paths.common_dir.exists():
        for common_file in sorted(project_paths.common_dir.glob("*.md")):
            parts.append(
                f"## research/{project_paths.slug}/common/{common_file.name}\n"
                + _read_text(common_file, max_chars=3500, redact_personal=redact_personal)
            )
            files.append(str(common_file))
    context = _truncate("\n\n".join(part for part in parts if part.strip()), max_chars)
    return {
        "status": "ok" if context.strip() else "empty",
        "research_root": str(RESEARCH_ROOT),
        "project": project_paths.slug if project_paths else project,
        "project_found": project_paths is not None,
        "available_projects": list_research_projects(),
        "files_read": files,
        "context": context,
        "redacted_personal_identity": redact_personal,
    }


def build_research_proposal(prompt: str, *, project: str | None = None) -> dict[str, Any]:
    text = " ".join(str(prompt or "").split())
    onboarding = read_research_onboarding_context(project, max_chars=9000)
    missing: list[str] = []
    lowered = text.lower()
    if not text:
        missing.append("research_question")
    if not project:
        missing.append("project")
    if not any(token in lowered for token in ["结构", "structure", "slab", "cluster", "poscar", "cif", "xsd"]):
        missing.append("initial_structure_or_model")
    if not any(token in lowered for token in ["证据", "计算", "dft", "能量", "吸附", "ts", "频率", "路径", "mechanism"]):
        missing.append("target_evidence")

    likely_stage = "discussion"
    if any(token in lowered for token in ["poscar", "cif", "xsd", "slab", "吸附", "adsorption"]):
        likely_stage = "structure_modeling"
    if any(token in lowered for token in ["ts", "neb", "dimer", "频率"]):
        likely_stage = "method_planning"

    next_actions = [
        "确认科学问题、目标结构和需要的计算证据。",
        "读取或生成初始结构，并先做 sanity check。",
        "若是吸附问题，走 adsorption_plan -> adsorption_build_slab -> adsorption_candidates。",
    ]
    if likely_stage == "method_planning":
        next_actions.insert(1, "先核对 research/Common/避坑清单.md 与项目 common 规则，避免违反既定方法学。")
    if missing:
        next_actions.insert(0, "补齐缺失输入：" + "、".join(missing))

    return {
        "status": "needs_inputs" if missing else "ready",
        "project": project,
        "prompt": text,
        "likely_stage": likely_stage,
        "missing_inputs": missing,
        "proposal": {
            "scientific_question": text or "未提供",
            "hypothesis": "待与用户讨论确认；不要硬编码材料、吸附质或反应路径。",
            "required_structures": ["初始 bulk/slab/cluster 结构", "必要时的缺陷/掺杂/吸附构型"],
            "required_evidence": ["结构合理性检查", "后续 DFT/频率/路径计算证据（按项目规则选择）"],
            "next_actions": next_actions,
        },
        "onboarding_files_read": onboarding["files_read"],
        "research_context_excerpt": onboarding["context"][:2200],
    }


def append_research_progress(
    project: str,
    *,
    completed: list[str] | None = None,
    blockers: list[str] | None = None,
    next_steps: list[str] | None = None,
) -> dict[str, Any]:
    paths = resolve_research_project(project)
    if paths is None:
        return {
            "status": "error",
            "message": f"research 项目不存在: {project}",
            "available_projects": list_research_projects(),
        }
    old = paths.progress.read_text(encoding="utf-8") if paths.progress.exists() else "# 研究进展\n\n"
    header = "# 研究进展\n\n"
    body = old[len(header) :] if old.startswith(header) else old
    lines = [f"### {datetime.now().strftime('%Y-%m-%d')}", ""]
    for item in completed or []:
        lines.append(f"- ✅ {item}")
    for item in blockers or []:
        lines.append(f"- ⚠️ {item}")
    for item in next_steps or []:
        lines.append(f"- ⬜ {item}")
    if len(lines) == 2:
        lines.append("- ✅ 已记录一次 AETHER 对话推进。")
    paths.progress.write_text(header + "\n".join(lines).rstrip() + "\n\n" + body, encoding="utf-8")
    return {"status": "ok", "project": paths.slug, "progress_path": str(paths.progress)}
