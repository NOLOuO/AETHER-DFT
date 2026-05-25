from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class PlanComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class ExecutionReadiness(str, Enum):
    READY = "ready"
    NEEDS_CONFIRMATION = "needs_confirmation"
    NEEDS_IMPLEMENTATION = "needs_implementation"


@dataclass
class PlanSubtask:
    name: str
    goal: str
    system_role: str
    task_type: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentPlan:
    task_id: str
    source_prompt: str
    experiment_type: str
    summary: str
    complexity: PlanComplexity
    readiness: ExecutionReadiness
    requires_confirmation: bool
    missing_information: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    subtasks: list[PlanSubtask] = field(default_factory=list)
    recommended_submit_profile: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    raw_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
