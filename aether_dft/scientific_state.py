from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any


EVIDENCE_SOURCE_TYPES = {
    "calculation_output",
    "live_cluster",
    "literature",
    "project_record",
    "human_answer",
    "model_inference",
}
LIVE_EVIDENCE_MAX_AGE = timedelta(minutes=15)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clean_strings(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(value).strip() for value in values if str(value).strip()]


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source_type: str
    locator: str
    summary: str
    observed_at: str = field(default_factory=_now_iso)
    content_digest: str = ""
    live: bool = False
    producer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificClaim:
    claim_id: str
    statement: str
    evidence_refs: list[str] = field(default_factory=list)
    confidence: str = "unspecified"
    requires_live_evidence: bool = False
    status: str = "provisional"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScientificProjectState:
    project: str
    research_goal: str
    schema_version: str = "1.0"
    success_criteria: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    claims: list[ScientificClaim] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    jobs: list[dict[str, Any]] = field(default_factory=list)
    current_focus: str = ""
    next_actions: list[str] = field(default_factory=list)
    status: str = "active"
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = [item.to_dict() for item in self.evidence]
        payload["claims"] = [item.to_dict() for item in self.claims]
        return payload


def normalize_scientific_state(project: str, payload: dict[str, Any] | None = None) -> ScientificProjectState:
    raw = dict(payload or {})
    evidence: list[EvidenceRecord] = []
    for index, item in enumerate(raw.get("evidence") or []):
        if not isinstance(item, dict):
            continue
        evidence.append(
            EvidenceRecord(
                evidence_id=str(item.get("evidence_id") or item.get("id") or f"evidence-{index + 1}"),
                source_type=str(item.get("source_type") or "project_record"),
                locator=str(item.get("locator") or item.get("path") or item.get("source") or ""),
                summary=str(item.get("summary") or item.get("detail") or ""),
                observed_at=str(item.get("observed_at") or raw.get("updated_at") or _now_iso()),
                content_digest=str(item.get("content_digest") or ""),
                live=bool(item.get("live")),
                producer=str(item.get("producer") or ""),
            )
        )
    claims: list[ScientificClaim] = []
    for index, item in enumerate(raw.get("claims") or []):
        if not isinstance(item, dict):
            continue
        claims.append(
            ScientificClaim(
                claim_id=str(item.get("claim_id") or item.get("id") or f"claim-{index + 1}"),
                statement=str(item.get("statement") or item.get("claim") or item.get("text") or ""),
                evidence_refs=_clean_strings(item.get("evidence_refs") or item.get("refs")),
                confidence=str(item.get("confidence") or "unspecified"),
                requires_live_evidence=bool(item.get("requires_live_evidence")),
                status=str(item.get("status") or "provisional"),
            )
        )
    return ScientificProjectState(
        project=str(raw.get("project") or project).strip(),
        research_goal=str(raw.get("research_goal") or raw.get("current_focus") or "").strip(),
        schema_version=str(raw.get("schema_version") or "1.0"),
        success_criteria=_clean_strings(raw.get("success_criteria")),
        hypotheses=_clean_strings(raw.get("hypotheses")),
        evidence=evidence,
        claims=claims,
        decisions=[dict(item) for item in (raw.get("decisions") or []) if isinstance(item, dict)],
        open_questions=_clean_strings(raw.get("open_questions")),
        blockers=_clean_strings(raw.get("blockers")),
        jobs=[dict(item) for item in (raw.get("jobs") or []) if isinstance(item, dict)],
        current_focus=str(raw.get("current_focus") or "").strip(),
        next_actions=_clean_strings(raw.get("next_actions") or raw.get("next_steps")),
        status=str(raw.get("status") or "active"),
        updated_at=str(raw.get("updated_at") or _now_iso()),
    )


def audit_scientific_state(
    state: ScientificProjectState | dict[str, Any],
    *,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    normalized = (
        state
        if isinstance(state, ScientificProjectState)
        else normalize_scientific_state(str(state.get("project") or ""), state)
    )
    evidence_by_ref: dict[str, EvidenceRecord] = {}
    for item in normalized.evidence:
        evidence_by_ref[item.evidence_id] = item
        if item.locator:
            evidence_by_ref[item.locator] = item
    findings: list[dict[str, str]] = []
    now = reference_time or datetime.now().astimezone()
    if now.tzinfo is None:
        now = now.astimezone()

    def trusted_fresh_live(item: EvidenceRecord) -> bool:
        if not (item.live and item.source_type == "live_cluster" and item.producer.startswith("tool:")):
            return False
        try:
            observed = datetime.fromisoformat(item.observed_at)
            if observed.tzinfo is None:
                observed = observed.astimezone()
        except ValueError:
            return False
        return now - observed <= LIVE_EVIDENCE_MAX_AGE

    for claim in normalized.claims:
        missing = [ref for ref in claim.evidence_refs if ref not in evidence_by_ref]
        if not claim.evidence_refs:
            findings.append({"code": "claim_without_evidence", "claim_id": claim.claim_id})
        if missing:
            findings.append({"code": "missing_evidence_ref", "claim_id": claim.claim_id, "detail": ", ".join(missing)})
        if claim.requires_live_evidence:
            resolved = [evidence_by_ref[ref] for ref in claim.evidence_refs if ref in evidence_by_ref]
            if not any(trusted_fresh_live(item) for item in resolved):
                findings.append({"code": "live_claim_without_live_evidence", "claim_id": claim.claim_id})
    for item in normalized.evidence:
        if item.source_type == "live_cluster" and not item.producer.startswith("tool:"):
            findings.append({"code": "untrusted_live_evidence_producer", "claim_id": "", "detail": item.evidence_id})
        elif item.source_type == "live_cluster" and not trusted_fresh_live(item):
            findings.append({"code": "stale_live_evidence", "claim_id": "", "detail": item.evidence_id})
    if normalized.status in {"complete", "completed", "converged"} and not normalized.evidence:
        findings.append({"code": "completion_without_evidence", "claim_id": ""})
    invalid_sources = [
        item.evidence_id
        for item in normalized.evidence
        if item.source_type not in EVIDENCE_SOURCE_TYPES
    ]
    for evidence_id in invalid_sources:
        findings.append({"code": "unknown_evidence_source", "claim_id": "", "detail": evidence_id})
    return {
        "status": "ok",
        "project": normalized.project,
        "finding_count": len(findings),
        "verdict": "valid" if not findings else "needs_attention",
        "findings": findings,
        "state": normalized.to_dict(),
    }
