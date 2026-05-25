from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dft_app.models import ExperimentSpec, PipelinePhase, RunRecord


@dataclass
class ExportExecutionResult:
    status: str
    message: str
    export_manifest: dict[str, Any] | None
    export_root: str | None
    package_path: str | None


class DeliverableExporter:
    """Create a lightweight deliverable bundle and zip package for one run."""

    METADATA_FILES = [
        "experiment_spec.json",
        "planner_summary.json",
        "build_summary.json",
        "structure_resolution.json",
        "submit_summary.json",
        "monitor_summary.json",
        "parse_summary.json",
        "parsed_result.json",
        "analysis_summary.json",
        "run_record.json",
    ]

    INPUT_FILES = [
        "POSCAR",
        "INCAR",
        "KPOINTS",
        "job.slurm",
        "POTCAR.mapping.json",
        "POSCAR.notes.txt",
        "INCAR.preview.json",
        "KPOINTS.preview.json",
    ]

    REPORT_FILES = ["analysis_report.md"]

    def export(self, spec: ExperimentSpec, run_record: RunRecord) -> ExportExecutionResult:
        run_root = Path(run_record.run_root)
        metadata_dir = run_root / "metadata"
        report_dir = run_root / "report"
        inputs_dir = run_root / "inputs"
        outputs_dir = run_root / "outputs"

        parsed_result_path = metadata_dir / "parsed_result.json"
        analysis_summary_path = metadata_dir / "analysis_summary.json"
        report_path = report_dir / "analysis_report.md"

        if not parsed_result_path.exists():
            message = "当前 run 缺少 parsed_result.json，至少需要先完成 parse。"
            run_record.block_phase(PipelinePhase.EXPORT, message)
            return ExportExecutionResult("blocked", message, None, None, None)

        if not analysis_summary_path.exists() or not report_path.exists():
            message = "当前 run 缺少分析结果，建议先执行 analyze 再 export。"
            run_record.block_phase(PipelinePhase.EXPORT, message)
            return ExportExecutionResult("blocked", message, None, None, None)

        export_root = run_root / "export"
        bundle_root = export_root / "bundle"
        bundle_metadata_dir = bundle_root / "metadata"
        bundle_inputs_dir = bundle_root / "inputs"
        bundle_report_dir = bundle_root / "report"

        for path in (bundle_metadata_dir, bundle_inputs_dir, bundle_report_dir):
            path.mkdir(parents=True, exist_ok=True)

        copied_files: list[str] = []
        copied_files.extend(
            self._copy_selected_files(metadata_dir, bundle_metadata_dir, self.METADATA_FILES)
        )
        copied_files.extend(
            self._copy_selected_files(inputs_dir, bundle_inputs_dir, self.INPUT_FILES)
        )
        copied_files.extend(
            self._copy_selected_files(report_dir, bundle_report_dir, self.REPORT_FILES)
        )

        outputs_index = self._build_outputs_index(outputs_dir, run_root)
        outputs_index_path = bundle_metadata_dir / "outputs_index.json"
        outputs_index_path.write_text(
            json.dumps(outputs_index, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        copied_files.append(str(outputs_index_path))

        manifest_path = export_root / "export_manifest.json"
        bundle_manifest_path = bundle_metadata_dir / "export_manifest.json"
        package_path = export_root / f"{spec.task_id}_{run_record.run_id}_deliverable.zip"

        export_manifest = {
            "task_id": spec.task_id,
            "run_id": run_record.run_id,
            "export_root": str(export_root),
            "bundle_root": str(bundle_root),
            "report_path": str(report_path),
            "packaged_at": datetime.now(timezone.utc).isoformat(),
            "included_files": copied_files + [str(bundle_manifest_path)],
            "outputs_index_file": str(outputs_index_path),
            "outputs_file_count": len(outputs_index),
            "manifest_path": str(manifest_path),
            "bundle_manifest_path": str(bundle_manifest_path),
            "package_path": str(package_path),
        }

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(export_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        bundle_manifest_path.write_text(
            json.dumps(export_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._zip_directory(bundle_root, package_path)

        run_record.complete_phase(
            PipelinePhase.EXPORT,
            artifacts=[str(manifest_path), str(package_path)],
            message="交付包已生成。",
        )
        self._deduplicate_phase_artifacts(run_record, PipelinePhase.EXPORT)
        run_record.mark_completed(report_path=str(report_path))

        return ExportExecutionResult(
            status="exported",
            message="交付包已生成。",
            export_manifest=export_manifest,
            export_root=str(export_root),
            package_path=str(package_path),
        )

    @staticmethod
    def _copy_selected_files(
        source_dir: Path, target_dir: Path, filenames: list[str]
    ) -> list[str]:
        copied: list[str] = []
        if not source_dir.exists():
            return copied

        for filename in filenames:
            source_path = source_dir / filename
            if not source_path.exists():
                continue
            target_path = target_dir / filename
            shutil.copy2(source_path, target_path)
            copied.append(str(target_path))
        return copied

    @staticmethod
    def _build_outputs_index(outputs_dir: Path, run_root: Path) -> list[dict[str, Any]]:
        if not outputs_dir.exists():
            return []

        index: list[dict[str, Any]] = []
        for path in sorted(outputs_dir.rglob("*")):
            if not path.is_file():
                continue
            index.append(
                {
                    "name": path.name,
                    "relative_path": str(path.relative_to(run_root)),
                    "size_bytes": path.stat().st_size,
                }
            )
        return index

    @staticmethod
    def _zip_directory(source_dir: Path, package_path: Path) -> None:
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(source_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(source_dir)))

    @staticmethod
    def _deduplicate_phase_artifacts(run_record: RunRecord, phase: PipelinePhase) -> None:
        artifacts = run_record.phases[phase.value].artifacts
        seen: set[str] = set()
        deduped: list[str] = []
        for artifact in artifacts:
            if artifact in seen:
                continue
            seen.add(artifact)
            deduped.append(artifact)
        run_record.phases[phase.value].artifacts = deduped
