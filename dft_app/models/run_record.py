from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus(str, Enum):
    PENDING = "pending"
    WAITING_CONFIRMATION = "waiting_confirmation"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelinePhase(str, Enum):
    PLAN = "plan"
    BUILD = "build"
    SUBMIT = "submit"
    MONITOR = "monitor"
    PARSE = "parse"
    ANALYZE = "analyze"
    EXPORT = "export"


class PhaseStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class PhaseRecord:
    phase: PipelinePhase
    status: PhaseStatus = PhaseStatus.NOT_STARTED
    started_at: str | None = None
    finished_at: str | None = None
    artifacts: list[str] = field(default_factory=list)
    message: str | None = None
    error: str | None = None


@dataclass
class RunRecord:
    task_id: str
    run_id: str
    run_root: str
    checkpoint_path: str
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    overall_status: RunStatus = RunStatus.PENDING
    current_phase: PipelinePhase | None = None
    report_path: str | None = None
    scheduler_job_id: str | None = None
    last_error: str | None = None
    restart_from_run_id: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    phases: dict[str, PhaseRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id 不能为空")
        if not self.run_id.strip():
            raise ValueError("run_id 不能为空")
        if not self.run_root.strip():
            raise ValueError("run_root 不能为空")
        if not self.checkpoint_path.strip():
            raise ValueError("checkpoint_path 不能为空")

        if not self.phases:
            self.phases = {
                phase.value: PhaseRecord(phase=phase) for phase in PipelinePhase
            }

    def touch(self) -> None:
        self.updated_at = _utcnow()

    def start_phase(self, phase: PipelinePhase, message: str | None = None) -> None:
        phase_record = self.phases[phase.value]
        phase_record.status = PhaseStatus.RUNNING
        phase_record.started_at = phase_record.started_at or _utcnow()
        phase_record.finished_at = None
        phase_record.message = message
        phase_record.error = None
        self.current_phase = phase
        self.overall_status = RunStatus.RUNNING
        self.last_error = None
        self.touch()

    def complete_phase(
        self,
        phase: PipelinePhase,
        artifacts: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        phase_record = self.phases[phase.value]
        phase_record.status = PhaseStatus.COMPLETED
        phase_record.started_at = phase_record.started_at or _utcnow()
        phase_record.finished_at = _utcnow()
        if artifacts:
            phase_record.artifacts.extend(artifacts)
        if message is not None:
            phase_record.message = message
        phase_record.error = None
        self.current_phase = phase
        self.touch()

    def fail_phase(self, phase: PipelinePhase, error: str) -> None:
        phase_record = self.phases[phase.value]
        phase_record.status = PhaseStatus.FAILED
        phase_record.started_at = phase_record.started_at or _utcnow()
        phase_record.finished_at = _utcnow()
        phase_record.error = error
        self.current_phase = phase
        self.overall_status = RunStatus.FAILED
        self.last_error = error
        self.touch()

    def block_phase(self, phase: PipelinePhase, message: str) -> None:
        phase_record = self.phases[phase.value]
        phase_record.status = PhaseStatus.BLOCKED
        phase_record.message = message
        self.current_phase = phase
        self.overall_status = RunStatus.WAITING_CONFIRMATION
        self.touch()

    def mark_ready(self) -> None:
        self.overall_status = RunStatus.READY
        self.touch()

    def mark_paused(self, message: str | None = None) -> None:
        self.overall_status = RunStatus.PAUSED
        if self.current_phase is not None and message is not None:
            self.phases[self.current_phase.value].message = message
        self.touch()

    def mark_completed(self, report_path: str | None = None) -> None:
        self.overall_status = RunStatus.COMPLETED
        if report_path is not None:
            self.report_path = report_path
        self.touch()

    def mark_cancelled(self, message: str | None = None) -> None:
        self.overall_status = RunStatus.CANCELLED
        if self.current_phase is not None and message is not None:
            self.phases[self.current_phase.value].message = message
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
