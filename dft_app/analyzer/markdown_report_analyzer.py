from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dft_app.models import ExperimentSpec, ParsedResult, PipelinePhase, RunRecord, RunStatus


@dataclass
class AnalysisExecutionResult:
    status: str
    message: str
    analysis_summary: dict[str, Any] | None
    report_path: str | None


class MarkdownReportAnalyzer:
    """Turn ParsedResult into a concise analysis summary and Markdown report."""

    def analyze(
        self,
        spec: ExperimentSpec,
        run_record: RunRecord,
        parsed_result: ParsedResult | None,
    ) -> AnalysisExecutionResult:
        if parsed_result is None:
            message = "当前 run 没有 parsed_result，无法执行 analyze。"
            run_record.block_phase(PipelinePhase.ANALYZE, message)
            return AnalysisExecutionResult("blocked", message, None, None)

        summary = self._build_analysis_summary(spec, parsed_result)
        report_path = Path(run_record.run_root) / "report" / "analysis_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            self._render_markdown_report(spec, run_record, parsed_result, summary),
            encoding="utf-8",
        )

        run_record.complete_phase(
            PipelinePhase.ANALYZE,
            artifacts=[str(report_path)],
            message="分析完成，已生成 Markdown 报告。",
        )
        run_record.report_path = str(report_path)
        run_record.overall_status = RunStatus.READY
        run_record.touch()

        return AnalysisExecutionResult(
            status="analyzed",
            message="分析完成，已生成 Markdown 报告。",
            analysis_summary=summary,
            report_path=str(report_path),
        )

    def _build_analysis_summary(
        self, spec: ExperimentSpec, parsed_result: ParsedResult
    ) -> dict[str, Any]:
        electronic_character = self._infer_electronic_character(parsed_result)
        convergence_assessment = self._infer_convergence_assessment(parsed_result)
        structural_quality = self._infer_structural_quality(parsed_result)
        recommended_actions = self._recommend_next_actions(spec, parsed_result)

        key_metrics = {
            "total_energy": parsed_result.total_energy,
            "energy_per_atom": parsed_result.energy_per_atom,
            "band_gap": parsed_result.band_gap,
            "efermi": parsed_result.efermi,
            "is_metal": parsed_result.is_metal,
            "volume": parsed_result.volume,
            "ionic_steps": parsed_result.ionic_steps,
            "electronic_steps": parsed_result.electronic_steps,
            "max_force": parsed_result.max_force,
        }

        return {
            "task_type": spec.task_type.value,
            "material_name": spec.material_name,
            "electronic_character": electronic_character,
            "convergence_assessment": convergence_assessment,
            "structural_quality": structural_quality,
            "key_metrics": key_metrics,
            "recommended_actions": recommended_actions,
            "warnings": parsed_result.warnings,
        }

    def _render_markdown_report(
        self,
        spec: ExperimentSpec,
        run_record: RunRecord,
        parsed_result: ParsedResult,
        summary: dict[str, Any],
    ) -> str:
        candidate_context = spec.notes.get("adsorption_candidate") if isinstance(spec.notes, dict) else None
        lines = [
            f"# DFT 分析报告: {spec.material_name}",
            "",
            "## 任务概览",
            f"- task_id: `{spec.task_id}`",
            f"- run_id: `{run_record.run_id}`",
            f"- 任务类型: `{spec.task_type.value}`",
            f"- workflow: `{', '.join(spec.workflow)}`",
            f"- 泛函: `{spec.functional}`",
            f"- 结构来源: `{spec.structure_source.value}`",
            "",
        ]

        if candidate_context:
            candidate_meta = candidate_context.get("metadata", {}).get("candidate", {})
            lines.extend(
                [
                    "## 方案/候选上下文",
                    f"- candidate_id: `{candidate_context.get('candidate_id')}`",
                    f"- site_label: `{candidate_meta.get('site_label', 'N/A')}`",
                    f"- orientation: `{candidate_meta.get('orientation_label', 'N/A')}`",
                    f"- defect: `{candidate_meta.get('defect_label', 'N/A')}`",
                    f"- 推荐分数: `{candidate_meta.get('score', {}).get('total', 'N/A')}`",
                    "",
                ]
            )

        lines.extend(
            [
            "## 执行结论",
            f"- 完成状态: `{parsed_result.completed}`",
            f"- 收敛状态: `{parsed_result.converged}`",
            f"- 电子结构判断: `{summary['electronic_character']}`",
            f"- 收敛评估: `{summary['convergence_assessment']}`",
            f"- 结构质量判断: `{summary['structural_quality']}`",
            "",
            "## 关键结果",
            f"- 总能量: {self._format_float(parsed_result.total_energy, 'eV')}",
            f"- 每原子能量: {self._format_float(parsed_result.energy_per_atom, 'eV/atom')}",
            f"- 带隙: {self._format_float(parsed_result.band_gap, 'eV')}",
            f"- 费米能级: {self._format_float(parsed_result.efermi, 'eV')}",
            f"- 体积: {self._format_float(parsed_result.volume, 'A^3')}",
            f"- 最大力: {self._format_float(parsed_result.max_force, 'eV/A')}",
            f"- 离子步数: `{parsed_result.ionic_steps}`",
            f"- 电子步数: `{parsed_result.electronic_steps}`",
            "",
            "## 晶格参数",
            f"- a: {self._format_float(parsed_result.lattice_parameters.a, 'A')}",
            f"- b: {self._format_float(parsed_result.lattice_parameters.b, 'A')}",
            f"- c: {self._format_float(parsed_result.lattice_parameters.c, 'A')}",
            f"- alpha: {self._format_float(parsed_result.lattice_parameters.alpha, 'deg')}",
            f"- beta: {self._format_float(parsed_result.lattice_parameters.beta, 'deg')}",
            f"- gamma: {self._format_float(parsed_result.lattice_parameters.gamma, 'deg')}",
            "",
            "## 建议",
            ]
        )

        for item in summary["recommended_actions"]:
            lines.append(f"- {item}")

        lines.extend(["", "## 输出来源"])
        for name, path in parsed_result.source_files.items():
            lines.append(f"- `{name}`: `{path}`")

        lines.extend(["", "## 警告"])
        if parsed_result.warnings:
            for warning in parsed_result.warnings:
                lines.append(f"- {warning}")
        else:
            lines.append("- 无")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _infer_electronic_character(parsed_result: ParsedResult) -> str:
        if parsed_result.is_metal is True:
            return "metal"
        if parsed_result.band_gap is None:
            return "unknown"
        if parsed_result.band_gap < 0.05:
            return "metal_or_zero_gap"
        if parsed_result.band_gap < 3.0:
            return "semiconductor"
        return "insulator"

    @staticmethod
    def _infer_convergence_assessment(parsed_result: ParsedResult) -> str:
        if parsed_result.converged:
            return "good"
        if parsed_result.completed:
            return "completed_but_not_converged"
        return "incomplete"

    @staticmethod
    def _infer_structural_quality(parsed_result: ParsedResult) -> str:
        if parsed_result.max_force is None:
            return "unknown"
        if parsed_result.max_force <= 0.03:
            return "well_relaxed"
        if parsed_result.max_force <= 0.1:
            return "acceptable"
        return "needs_further_relaxation"

    def _recommend_next_actions(
        self, spec: ExperimentSpec, parsed_result: ParsedResult
    ) -> list[str]:
        actions: list[str] = []
        task_type = spec.task_type.value

        if not parsed_result.converged:
            actions.append("优先检查 ENCUT、KPOINTS、EDIFF 和最大电子步数设置。")

        if parsed_result.max_force is not None and parsed_result.max_force > 0.03:
            actions.append("结构仍存在较大残余力，建议继续弛豫或收紧离子优化参数。")

        if task_type in {"relax", "geometry_optimization"} and parsed_result.converged:
            actions.append("下一步可继续执行 SCF、DOS 或 band 计算。")

        if task_type == "single_point":
            actions.append("可将该单点能作为能量对比基准，并结合优化结构继续做电子结构分析。")

        if task_type == "static_refinement":
            actions.append("若静态结果已稳定，可继续执行 DOS、PDOS、功函数或电荷分析。")

        if task_type in {"relax_scf_band", "band_structure"}:
            if parsed_result.band_gap is None:
                actions.append("当前结果未提取到带隙，建议检查非自洽能带输出是否完整。")
            else:
                actions.append("可进一步绘制能带图并标注高对称点路径。")

        if task_type == "dos":
            actions.append("建议结合更密 KPOINTS 或更细 NEDOS 设置复核 DOS 峰位置。")

        if task_type == "pdos":
            actions.append("建议结合轨道投影结果分析成键贡献，并与总 DOS 互相校验。")

        if task_type == "charge_analysis":
            actions.append("建议导出 CHGCAR / AECCAR* 并继续执行 Bader 或差分电荷分析。")

        if task_type == "work_function":
            actions.append("建议检查平面平均静电势是否形成稳定真空平台，并确认偶极修正设置。")

        if task_type == "vibrational_frequency":
            actions.append("建议重点检查是否存在虚频，并区分真实过渡态模式与数值伪虚频。")

        if task_type == "transition_state_search":
            actions.append("建议确认仅存在一个主虚频，并验证过渡态是否正确连接目标初态和终态。")

        if task_type == "molecular_dynamics":
            actions.append("建议进一步分析温度波动、总能漂移、MSD 或 RDF 等轨迹统计量。")

        if task_type == "spin_related":
            actions.append("建议比较不同磁构型或 SOC 开关下的总能、磁矩与电子结构差异。")

        if task_type == "defect_doping":
            actions.append("建议与本征超胞结果对比，并进一步计算缺陷形成能或掺杂稳定性。")

        if task_type in {"encut_convergence", "kpoints_convergence"}:
            actions.append("建议结合多组任务结果绘制收敛曲线后再确定最终参数。")

        if task_type == "eos":
            actions.append("建议基于多体积点结果继续拟合状态方程。")

        # --- 表面/吸附体系专属建议 ---
        system_role = spec.notes.get("system_role") if isinstance(spec.notes, dict) else None
        if system_role == "slab":
            actions.append("建议检查表面弛豫是否合理：顶层原子位移不应超过 0.3 A。")
            actions.append("确认底层原子是否已正确固定（selective dynamics）。")
        elif system_role == "adsorbate_slab":
            actions.append("建议检查吸附构型是否稳定：adsorbate 未脱离表面、未嵌入表面。")
            actions.append("建议检查真空层厚度是否足够（≥ 12 A）。")
            if parsed_result.max_force is not None and parsed_result.max_force > 0.05:
                actions.append("吸附体系残余力偏大，建议增加 NSW 或收紧 EDIFFG。")
        elif system_role == "molecule":
            actions.append("建议确认孤立分子 box 尺寸足够大（≥ 15 A），避免周期性镜像相互作用。")

        if not actions:
            actions.append("结果已可用于后续人工复核与归档。")

        return actions

    @staticmethod
    def _format_float(value: float | None, unit: str) -> str:
        if value is None:
            return "N/A"
        return f"{value:.6f} {unit}"
