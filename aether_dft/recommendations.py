from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .knowledge import search_notes
from .project_state import list_projects, read_project_context
from .research_workspace import read_research_onboarding_context
from .session_store import AetherSessionStore
from .task_bridge import list_task_records


@dataclass(frozen=True)
class Recommendation:
    title: str
    reason: str
    command: str | None = None
    priority: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "reason": self.reason,
            "command": self.command,
            "priority": self.priority,
        }


def recommend_next_tasks(project: str | None = None, *, focus: str | None = None) -> list[dict[str, Any]]:
    recommendations: list[Recommendation] = []
    if project:
        context = read_project_context(project)
        research_context = read_research_onboarding_context(project, max_chars=10000)["context"]
        context = (context + "\n\n" + research_context).strip()
        tasks = list_task_records(project)
        notes = search_notes(project, focus or "吸附 adsorption DFT")
        outcome_notes = search_notes(project, "candidate_outcome adsorption outcome")
        sessions = AetherSessionStore().list_sessions(project=project, limit=3)
    else:
        context = ""
        tasks = []
        notes = []
        outcome_notes = []
        sessions = AetherSessionStore().list_sessions(limit=3)

    lowered = (context + "\n" + "\n".join(str(task) for task in tasks) + "\n" + "\n".join(item.first_prompt for item in sessions)).lower()
    if focus and any(token in focus.lower() for token in ["吸附", "adsorption", "adsorb"]):
        lowered += " adsorption"

    if "adsorption" in lowered or "吸附" in lowered or not tasks:
        recommendations.extend(
            [
                Recommendation(
                    title="准备 slab 结构并生成吸附候选",
                    reason="吸附任务的第一可执行瓶颈通常是表面模型 + adsorbate + 位点/取向枚举。",
                    command=(
                        "aether-dft adsorption candidates --slab-path <POSCAR> "
                        "--adsorbate <ADSORBATE> --material <MATERIAL> --project <PROJECT>"
                    ),
                    priority="high",
                ),
                Recommendation(
                    title="选择候选并记录下一步计算方案",
                    reason="AETHER 当前主线先收束到结构候选与方案沉淀，不默认直接提交或解析计算。",
                    command="在对话中让 AETHER 总结候选选择依据，并按项目规则追加 research progress。",
                    priority="high",
                ),
                Recommendation(
                    title="进入后续 DFT 前先做结构 sanity check",
                    reason="真实计算前应先检查最短距离、真空层、固定层和吸附物初始高度。",
                    command="让 AETHER 调用 structure_sanity_check / structure_bond_analyze 检查候选 POSCAR。",
                    priority="medium",
                ),
            ]
        )
    if notes:
        recommendations.append(
            Recommendation(
                title="复用项目知识库中的相关结论",
                reason=f"检索到 {len(notes)} 条相关知识条目，应先检查是否已有参数/位点经验。",
                command=f"aether-dft kb search {project} {focus or 'adsorption'}" if project else None,
                priority="medium",
            )
        )
    if outcome_notes:
        recommendations.append(
            Recommendation(
                title="先复用已完成候选的 outcome 经验",
                reason=f"知识库中有 {len(outcome_notes)} 条候选计算复盘；生成新候选前应优先检索 material/adsorbate prior。",
                command=(
                    f"aether-dft tools run knowledge_search_for_system "
                    f"--arguments '{{\"project_priority\":\"{project}\",\"extra_terms\":[\"candidate_outcome\"]}}'"
                    if project
                    else None
                ),
                priority="high",
            )
        )
    if sessions:
        latest = sessions[0]
        recommendations.append(
            Recommendation(
                title="续接最近一次科研对话",
                reason=f"最近 session `{latest.session_id}` 有 {latest.turn_count} 轮记录，可直接从上下文继续。",
                command=f"aether-dft session resume {latest.session_id}",
                priority="medium",
            )
        )
    if project and not context.strip():
        recommendations.append(
            Recommendation(
                title="初始化项目状态",
                reason="持续推荐依赖 project state / research_progress / knowledge_base。",
                command=f"aether-dft project init {project}",
                priority="high",
            )
        )
    if not project and list_projects():
        recommendations.append(
            Recommendation(
                title="选择一个项目上下文",
                reason="持续科研任务推荐需要知道当前项目。",
                command="aether-dft project list",
                priority="medium",
            )
        )
    return [item.to_dict() for item in recommendations]
