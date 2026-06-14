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
        notes = search_notes(project, focus or "")
        outcome_notes = search_notes(project, "candidate_outcome outcome")
        sessions = AetherSessionStore().list_sessions(project=project, limit=3)
    else:
        context = ""
        tasks = []
        notes = []
        outcome_notes = []
        sessions = AetherSessionStore().list_sessions(limit=3)

    if project:
        recommendations.extend(
            [
                Recommendation(
                    title="先做项目证据盘点",
                    reason="下一步应由 research 进展、最近 session、run 记录和集群状态共同决定，而不是由关键词分支决定。",
                    command=f'aether-dft "{project} 这个课题现在有哪些证据、缺口和最小下一步？"',
                    priority="high",
                ),
                Recommendation(
                    title="让模型选择必要工具",
                    reason="如果需要建模、输入生成或集群执行，模型应先查看能力地图并按需解锁工具 schema。",
                    command="在 REPL 中自然语言说明目标；不要手动套固定流程。",
                    priority="high",
                ),
            ]
        )
    elif not tasks:
        recommendations.append(
            Recommendation(
                title="先选择 research 课题",
                reason="没有 project 时，系统无法可靠关联 research 进展、会话和 run records。",
                command="aether-dft project list",
                priority="high",
            )
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
