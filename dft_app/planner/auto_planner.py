from __future__ import annotations

from dft_app.models import ExperimentSpec
from dft_app.planner.llm_planner import LLMPlanner, PlanningResult
from dft_app.planner.rule_based_planner import RuleBasedPlanner


class AutoPlanner:
    """Prefer LLM planning and fall back to rule-based planning when needed."""

    def __init__(
        self,
        llm_planner: LLMPlanner | None = None,
        rule_planner: RuleBasedPlanner | None = None,
    ):
        self.llm_planner = llm_planner or LLMPlanner()
        self.rule_planner = rule_planner or RuleBasedPlanner()

    def plan(self, **kwargs) -> PlanningResult:
        return self.llm_planner.plan(**kwargs)

    def explain(self, result: PlanningResult) -> dict:
        return self.llm_planner.explain(result)

    @staticmethod
    def require_spec(result: PlanningResult) -> ExperimentSpec:
        if result.spec is None:
            raise RuntimeError(
                "当前任务已被识别为复杂组合任务，尚未实现自动执行。请先使用 --dry-run 查看 LLM 规划结果，并补充必要信息。"
            )
        return result.spec
