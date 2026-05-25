"""结果解释服务的共享 contract。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceGap:
    field: str
    reason: str
    severity: str = "warning"
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBundle:
    task_name: str
    task_goal_text: str | None = None
    task_dir: str | None = None
    status_summary: dict[str, Any] = field(default_factory=dict)
    structure_context: dict[str, Any] = field(default_factory=dict)
    input_context: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)
    structure_analysis: dict[str, Any] = field(default_factory=dict)
    similar_cases: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    missing_evidence: list[EvidenceGap] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_evidence"] = [gap.to_dict() for gap in self.missing_evidence]
        return payload


@dataclass
class InterpretationRequest:
    task_name: str
    evidence: EvidenceBundle
    provider: str = "auto"
    model: str | None = None
    max_tokens: int = 1800
    temperature: float = 0.2
    allow_degraded: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = self.evidence.to_dict()
        return payload


@dataclass
class KnowledgeBackflowPayload:
    task_name: str
    task_goal_text: str | None
    status_judgement: str
    likely_causes: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    evidence_digest: dict[str, Any] = field(default_factory=dict)
    missing_evidence: list[dict[str, Any]] = field(default_factory=list)
    provider: str = "none"
    model: str = "none"
    generated_at: str | None = None
    inferred_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InterpretationResult:
    task_name: str
    status_judgement: str
    likely_causes: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    evidence_used: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[EvidenceGap] = field(default_factory=list)
    raw_llm_text: str = ""
    provider: str = "none"
    model: str = "none"
    generated_at: str | None = None
    confidence: str = "medium"
    degraded: bool = False
    knowledge_backflow_payload: KnowledgeBackflowPayload | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_evidence"] = [gap.field for gap in self.missing_evidence]
        payload["missing_evidence_details"] = [gap.to_dict() for gap in self.missing_evidence]
        payload["knowledge_backflow_payload"] = (
            self.knowledge_backflow_payload.to_dict() if self.knowledge_backflow_payload else None
        )
        return payload
