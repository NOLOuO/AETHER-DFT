from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from dft_app.models import ExperimentPlan


@dataclass
class WorkflowTaskScaffold:
    name: str
    system_role: str
    goal: str
    task_type: str | None
    relative_dir: str
    status: str = "pending_definition"
    blockers: list[str] = field(default_factory=list)
    suggested_inputs: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowScaffold:
    task_id: str
    workflow_type: str
    summary: str
    readiness: str
    requires_confirmation: bool
    confirmation_items: list[str] = field(default_factory=list)
    shared_assumptions: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    tasks: list[WorkflowTaskScaffold] = field(default_factory=list)
    analysis_steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ComplexWorkflowBuilder:
    workflow_type: str = "unknown"

    def build(self, plan: ExperimentPlan) -> WorkflowScaffold:
        raise NotImplementedError
