from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dft_app.models import (
    ExperimentSpec,
    LatticeParameters,
    PhaseRecord,
    PhaseStatus,
    PipelinePhase,
    ParsedResult,
    RunRecord,
    RunStatus,
    experiment_spec_from_dict,
)


class RecordStore:
    """Read and write task definitions and run records from the project workspace."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.runs_root = project_root / ".aether" / "runs"

    def resolve_run_root(
        self, run_root: str | None = None, run_id: str | None = None
    ) -> Path:
        if run_root:
            path = Path(run_root)
            if not path.exists():
                raise FileNotFoundError(f"run_root 不存在: {path}")
            return path

        if run_id:
            matches = list(self.runs_root.glob(f"*/{run_id}"))
            if not matches:
                raise FileNotFoundError(f"未找到 run_id={run_id} 对应的任务目录")
            if len(matches) > 1:
                raise ValueError(f"run_id={run_id} 匹配到多个目录，请改用 --run-root")
            return matches[0]

        raise ValueError("必须提供 run_root 或 run_id")

    def save_run_record(self, run_record: RunRecord) -> None:
        run_root = Path(run_record.run_root)
        checkpoint_path = Path(run_record.checkpoint_path)
        metadata_dir = run_root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(metadata_dir / "run_record.json", run_record.to_dict())
        self._write_json(checkpoint_path, run_record.to_dict())

    def write_metadata(self, run_root: Path, filename: str, payload: dict[str, Any]) -> Path:
        metadata_dir = run_root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        output_path = metadata_dir / filename
        self._write_json(output_path, payload)
        return output_path

    def load_run_record(self, run_root: Path) -> RunRecord:
        data = self._read_json(run_root / "metadata" / "run_record.json")
        phases: dict[str, PhaseRecord] = {}
        for key, value in data.get("phases", {}).items():
            phases[key] = PhaseRecord(
                phase=PipelinePhase(value["phase"]),
                status=PhaseStatus(value["status"]),
                started_at=value.get("started_at"),
                finished_at=value.get("finished_at"),
                artifacts=value.get("artifacts", []),
                message=value.get("message"),
                error=value.get("error"),
            )

        return RunRecord(
            task_id=data["task_id"],
            run_id=data["run_id"],
            run_root=data["run_root"],
            checkpoint_path=data["checkpoint_path"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            overall_status=RunStatus(data["overall_status"]),
            current_phase=PipelinePhase(data["current_phase"])
            if data.get("current_phase")
            else None,
            report_path=data.get("report_path"),
            scheduler_job_id=data.get("scheduler_job_id"),
            last_error=data.get("last_error"),
            restart_from_run_id=data.get("restart_from_run_id"),
            tags=data.get("tags", []),
            notes=data.get("notes", {}),
            phases=phases,
        )

    def load_experiment_spec(self, run_root: Path) -> ExperimentSpec:
        data = self._read_json(run_root / "metadata" / "experiment_spec.json")
        return experiment_spec_from_dict(data)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for run_record_path in self.runs_root.glob("*/*/metadata/run_record.json"):
            data = self._read_json(run_record_path)
            results.append(
                {
                    "task_id": data["task_id"],
                    "run_id": data["run_id"],
                    "run_root": data["run_root"],
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "overall_status": data["overall_status"],
                    "current_phase": data.get("current_phase"),
                    "scheduler_job_id": data.get("scheduler_job_id"),
                }
            )
        results.sort(key=lambda item: item["created_at"], reverse=True)
        return results[:limit]

    def load_parsed_result(self, run_root: Path) -> ParsedResult:
        data = self._read_json(run_root / "metadata" / "parsed_result.json")
        return ParsedResult(
            task_id=data["task_id"],
            run_id=data["run_id"],
            calc_type=data["calc_type"],
            parsed_at=data["parsed_at"],
            completed=data.get("completed", False),
            converged=data.get("converged", False),
            total_energy=data.get("total_energy"),
            energy_per_atom=data.get("energy_per_atom"),
            band_gap=data.get("band_gap"),
            efermi=data.get("efermi"),
            is_metal=data.get("is_metal"),
            volume=data.get("volume"),
            lattice_parameters=LatticeParameters(**data["lattice_parameters"]),
            ionic_steps=data.get("ionic_steps"),
            electronic_steps=data.get("electronic_steps"),
            max_force=data.get("max_force"),
            warnings=data.get("warnings", []),
            source_files=data.get("source_files", {}),
            derived_metrics=data.get("derived_metrics", {}),
            plots=data.get("plots", []),
            raw_summary=data.get("raw_summary", {}),
        )

    def read_metadata_file(self, run_root: Path, filename: str) -> dict[str, Any] | None:
        path = run_root / "metadata" / filename
        if not path.exists():
            return None
        return self._read_json(path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
